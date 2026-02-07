"""Compute fixation binary-vector cross-correlations within and across sessions."""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.data.gaze_data import FixationBinaryVectorsData
from dal_monte_2022_analysis.io.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


@dataclass
class FixCrossCorrelationSettings:
    """Configuration for fixation cross-correlation analysis."""

    cfg_path: str
    input_modality: str = "fixation_binary_vectors"
    fixation_label: str = "face"
    output_subdir: str = "fix_cross_correlation"
    within_filename: str = "within_session_face_fix_cross_correlation.csv"
    cross_filename: str = "cross_session_face_fix_cross_correlation.csv"
    max_lag: Optional[int] = None
    cross_pairs_max: Optional[int] = None
    cross_pairs_seed: int = 13
    cross_exclude_same_session: bool = True
    cross_exclude_same_date: bool = False
    test_single: bool = False


def _load_pickle(path):
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_fixation_vector(
    obj,
    fixation_label: str,
) -> Optional[np.ndarray]:
    """Extract a fixation vector from supported object layouts."""
    if isinstance(obj, FixationBinaryVectorsData):
        vectors = obj.vectors
    elif isinstance(obj, dict) and "vectors" in obj:
        vectors = obj["vectors"]
    elif isinstance(obj, dict):
        vectors = obj
    else:
        return None

    if not vectors or fixation_label not in vectors:
        return None

    vec = np.asarray(vectors[fixation_label])
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    return vec.astype(bool, copy=False)


def _extract_monkey_name(obj) -> Optional[str]:
    """Extract monkey name metadata if present."""
    if isinstance(obj, FixationBinaryVectorsData):
        return obj.context.monkey_name
    if isinstance(obj, dict):
        context = obj.get("context")
        if context is not None:
            if hasattr(context, "monkey_name"):
                return getattr(context, "monkey_name")
            if isinstance(context, dict) and "monkey_name" in context:
                return context.get("monkey_name")
        if "monkey_name" in obj:
            return obj.get("monkey_name")
    return None


def _load_fixation_vector(
    path,
    fixation_label: str,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Load a fixation vector and monkey name from a pickle path."""
    obj = _load_pickle(path)
    return _extract_fixation_vector(obj, fixation_label), _extract_monkey_name(obj)


def _index_agent_paths(cfg: dict, modality: str) -> tuple[dict, dict]:
    """Index m1/m2 fixation vector paths by (date, session)."""
    index_df = index_processed_dataset(cfg, modality)
    rows = index_df.to_dict(orient="records")

    m1_paths: dict[tuple[str, str], object] = {}
    m2_paths: dict[tuple[str, str], object] = {}
    for row in rows:
        agent = row.get("agent")
        if agent == "m1":
            m1_paths[(row["date"], row["session"])] = row["path"]
        elif agent == "m2":
            m2_paths[(row["date"], row["session"])] = row["path"]

    return m1_paths, m2_paths


def _build_cross_pairs(
    settings: FixCrossCorrelationSettings,
    m1_keys: list[tuple[str, str]],
    m2_keys: list[tuple[str, str]],
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Generate cross-session pairs with optional exclusions and subsampling."""
    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for key1 in m1_keys:
        for key2 in m2_keys:
            if settings.cross_exclude_same_session and key1 == key2:
                continue
            if settings.cross_exclude_same_date and key1[0] == key2[0]:
                continue
            pairs.append((key1, key2))

    if settings.cross_pairs_max is not None and len(pairs) > settings.cross_pairs_max:
        rng = random.Random(settings.cross_pairs_seed)
        pairs = rng.sample(pairs, settings.cross_pairs_max)

    return pairs


def _fft_cross_correlation(
    x_bool: np.ndarray,
    y_bool: np.ndarray,
    *,
    max_lag: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full linear cross-correlation via FFT.

    Returns lags spanning [-(len(y)-1), len(x)-1] and correlation counts.
    """
    x = np.asarray(x_bool, dtype=np.float64)
    y = np.asarray(y_bool, dtype=np.float64)
    n = int(x.size)
    m = int(y.size)
    if n == 0 or m == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    full_len = n + m - 1
    nfft = 1 << (full_len - 1).bit_length()
    corr_circular = np.fft.irfft(
        np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(y, nfft)),
        nfft,
    )
    if m == 1:
        corr_full = corr_circular[:n]
    else:
        corr_full = np.concatenate([corr_circular[-(m - 1):], corr_circular[:n]])
    lags = np.arange(-(m - 1), n, dtype=np.int64)

    if max_lag is not None:
        max_lag = max(0, int(max_lag))
        keep = np.abs(lags) <= max_lag
        lags = lags[keep]
        corr_full = corr_full[keep]

    corr_full = np.rint(corr_full).astype(np.int64)
    return lags, corr_full


def _serialize_array(arr: np.ndarray) -> str:
    """Serialize a 1D array as compact JSON text."""
    return json.dumps(arr.tolist(), separators=(",", ":"))


def _summarize_corr(
    lags: np.ndarray,
    corr: np.ndarray,
    *,
    x_count: int,
    y_count: int,
) -> dict:
    """Compute summary stats for one cross-correlation trace."""
    if corr.size == 0:
        return {
            "n_lags": 0,
            "zero_lag_correlation": None,
            "zero_lag_correlation_norm": None,
            "peak_lag": None,
            "peak_correlation": None,
            "peak_lag_abs": None,
            "peak_correlation_abs": None,
            "peak_correlation_norm": None,
        }

    zero_lag_corr = None
    zero_matches = np.where(lags == 0)[0]
    if zero_matches.size > 0:
        zero_lag_corr = int(corr[int(zero_matches[0])])

    peak_idx = int(np.argmax(corr))
    peak_abs_idx = int(np.argmax(np.abs(corr)))

    norm_factor = float(np.sqrt(float(x_count) * float(y_count)))
    zero_norm = None
    peak_norm = None
    if norm_factor > 0.0:
        if zero_lag_corr is not None:
            zero_norm = float(zero_lag_corr) / norm_factor
        peak_norm = float(corr[peak_idx]) / norm_factor

    return {
        "n_lags": int(corr.size),
        "zero_lag_correlation": zero_lag_corr,
        "zero_lag_correlation_norm": zero_norm,
        "peak_lag": int(lags[peak_idx]),
        "peak_correlation": int(corr[peak_idx]),
        "peak_lag_abs": int(lags[peak_abs_idx]),
        "peak_correlation_abs": int(np.abs(corr[peak_abs_idx])),
        "peak_correlation_norm": peak_norm,
    }


def _build_result_row(
    *,
    lags: np.ndarray,
    corr: np.ndarray,
    x_bool: np.ndarray,
    y_bool: np.ndarray,
    monkey_name_m1: Optional[str],
    monkey_name_m2: Optional[str],
) -> dict:
    """Create one output row with metadata and correlation summaries."""
    x_count = int(np.count_nonzero(x_bool))
    y_count = int(np.count_nonzero(y_bool))
    row = {
        "fixation_label": None,
        "n_samples_m1": int(x_bool.size),
        "n_samples_m2": int(y_bool.size),
        "m1_fixation_count": x_count,
        "m2_fixation_count": y_count,
        "monkey_name_m1": monkey_name_m1,
        "monkey_name_m2": monkey_name_m2,
        "lags_json": _serialize_array(lags),
        "cross_correlation_json": _serialize_array(corr),
    }
    row.update(_summarize_corr(lags, corr, x_count=x_count, y_count=y_count))
    return row


def _build_within_session_rows(
    settings: FixCrossCorrelationSettings,
    m1_paths: dict,
    m2_paths: dict,
) -> list[dict]:
    """Build within-session cross-correlation rows."""
    rows: list[dict] = []
    shared_keys = sorted(set(m1_paths).intersection(m2_paths))

    if settings.test_single and shared_keys:
        shared_keys = [shared_keys[0]]

    for date, session in tqdm(shared_keys, desc="Within-session xcorr", unit="session"):
        key = (date, session)
        m1_vec, m1_name = _load_fixation_vector(m1_paths[key], settings.fixation_label)
        m2_vec, m2_name = _load_fixation_vector(m2_paths[key], settings.fixation_label)
        if m1_vec is None or m2_vec is None:
            continue
        if m1_vec.size == 0 or m2_vec.size == 0:
            continue

        lags, corr = _fft_cross_correlation(m1_vec, m2_vec, max_lag=settings.max_lag)
        row = _build_result_row(
            lags=lags,
            corr=corr,
            x_bool=m1_vec,
            y_bool=m2_vec,
            monkey_name_m1=m1_name,
            monkey_name_m2=m2_name,
        )
        row.update({
            "fixation_label": settings.fixation_label,
            "date": date,
            "session": session,
        })
        rows.append(row)

    return rows


def _build_cross_session_rows(
    settings: FixCrossCorrelationSettings,
    m1_paths: dict,
    m2_paths: dict,
) -> list[dict]:
    """Build cross-session cross-correlation rows."""
    rows: list[dict] = []
    m1_keys = sorted(m1_paths)
    m2_keys = sorted(m2_paths)

    pairs = _build_cross_pairs(settings, m1_keys, m2_keys)
    if settings.test_single and pairs:
        pairs = pairs[: min(10, len(pairs))]

    m1_cache: dict[tuple[str, str], np.ndarray] = {}
    m2_cache: dict[tuple[str, str], np.ndarray] = {}
    m1_name_cache: dict[tuple[str, str], Optional[str]] = {}
    m2_name_cache: dict[tuple[str, str], Optional[str]] = {}

    for (date1, session1), (date2, session2) in tqdm(
        pairs,
        desc="Cross-session xcorr",
        unit="pair",
    ):
        key1 = (date1, session1)
        key2 = (date2, session2)

        if key1 not in m1_cache:
            m1_vec, m1_name = _load_fixation_vector(m1_paths[key1], settings.fixation_label)
            if m1_vec is None:
                continue
            m1_cache[key1] = m1_vec
            m1_name_cache[key1] = m1_name
        if key2 not in m2_cache:
            m2_vec, m2_name = _load_fixation_vector(m2_paths[key2], settings.fixation_label)
            if m2_vec is None:
                continue
            m2_cache[key2] = m2_vec
            m2_name_cache[key2] = m2_name

        m1_vec = m1_cache[key1]
        m2_vec = m2_cache[key2]
        if m1_vec.size == 0 or m2_vec.size == 0:
            continue

        lags, corr = _fft_cross_correlation(m1_vec, m2_vec, max_lag=settings.max_lag)
        row = _build_result_row(
            lags=lags,
            corr=corr,
            x_bool=m1_vec,
            y_bool=m2_vec,
            monkey_name_m1=m1_name_cache.get(key1),
            monkey_name_m2=m2_name_cache.get(key2),
        )
        row.update({
            "fixation_label": settings.fixation_label,
            "date_m1": date1,
            "session_m1": session1,
            "date_m2": date2,
            "session_m2": session2,
        })
        rows.append(row)

    return rows


def run_fix_cross_correlation_analysis(
    settings: FixCrossCorrelationSettings,
    *,
    compute_cross: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run fixation cross-correlation analysis and persist outputs."""
    cfg = load_dataset_config(settings.cfg_path)
    m1_paths, m2_paths = _index_agent_paths(cfg, settings.input_modality)

    if not m1_paths or not m2_paths:
        raise RuntimeError(
            "Missing fixation binary vectors for m1 or m2. "
            f"Found m1={len(m1_paths)} m2={len(m2_paths)}."
        )

    within_rows = _build_within_session_rows(settings, m1_paths, m2_paths)
    within_df = pd.DataFrame.from_records(within_rows)

    cross_df = None
    if compute_cross:
        cross_rows = _build_cross_session_rows(settings, m1_paths, m2_paths)
        cross_df = pd.DataFrame.from_records(cross_rows)

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    within_path = out_dir / settings.within_filename
    within_df.to_csv(within_path, index=False)

    if cross_df is not None:
        cross_path = out_dir / settings.cross_filename
        cross_df.to_csv(cross_path, index=False)

    return within_df, cross_df
