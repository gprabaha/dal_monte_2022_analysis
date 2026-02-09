"""Compute fixation binary-vector cross-correlations within and across sessions."""

from __future__ import annotations

from multiprocessing import Pool
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
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


@dataclass
class FixCrossCorrelationSettings:
    """Configuration for fixation cross-correlation analysis."""

    cfg_path: str
    input_modality: str = "fixation_binary_vectors"
    fixation_label: str = "face"
    output_subdir: str = "fix_cross_correlation"
    within_filename: str = "within_session_face_fix_cross_correlation.pkl"
    cross_filename: str = "cross_session_face_fix_cross_correlation.pkl"
    lags_filename: Optional[str] = None
    max_lag: Optional[int] = 60000
    cross_pairs_max: Optional[int] = None
    cross_pairs_seed: int = 13
    cross_exclude_same_session: bool = True
    cross_exclude_same_date: bool = False
    parallelize_across_crosscorr_pairs: bool = False
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


def _print_test_single_debug(
    *,
    mode: str,
    settings: FixCrossCorrelationSettings,
    m1_id: tuple[str, str],
    m2_id: tuple[str, str],
    m1_len: int,
    m2_len: int,
    m1_fix_count: int,
    m2_fix_count: int,
    lags: np.ndarray,
    corr: np.ndarray,
) -> None:
    """Print selected task info and quick sanity checks for test-single mode."""
    max_allowed = 1.0 + 1e-9
    finite_ok = bool(np.isfinite(corr).all())
    in_range_ok = bool(((corr >= -1e-9) & (corr <= max_allowed)).all()) if corr.size else True
    has_zero_lag = bool(np.any(lags == 0))

    if corr.size:
        min_corr = float(np.min(corr))
        max_corr = float(np.max(corr))
        mean_corr = float(np.mean(corr))
    else:
        min_corr = float("nan")
        max_corr = float("nan")
        mean_corr = float("nan")

    print("\n[test-single] --------------------------------------------------")
    print(f"[test-single] mode: {mode}")
    print(
        "[test-single] selected pair: "
        f"m1(date={m1_id[0]}, session={m1_id[1]}) "
        f"vs m2(date={m2_id[0]}, session={m2_id[1]})"
    )
    print(
        "[test-single] vector sizes/counts: "
        f"m1_len={m1_len}, m2_len={m2_len}, "
        f"m1_fix={m1_fix_count}, m2_fix={m2_fix_count}"
    )
    if lags.size:
        print(
            "[test-single] lag window: "
            f"{int(lags[0])}..{int(lags[-1])} (n_lags={lags.size}, max_lag={settings.max_lag})"
        )
    else:
        print("[test-single] lag window: empty")
    print(
        "[test-single] corr summary: "
        f"min={min_corr:.6f}, max={max_corr:.6f}, mean={mean_corr:.6f}"
    )
    print(
        "[test-single] sanity: "
        f"finite={finite_ok}, within_[0,1]={in_range_ok}, has_zero_lag={has_zero_lag}"
    )
    if corr.size:
        preview = np.array2string(corr[:10], precision=4, separator=", ")
        print(f"[test-single] corr preview (first 10): {preview}")
    print("[test-single] --------------------------------------------------\n")


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


def _summarize_corr(
    lags: np.ndarray,
    corr: np.ndarray,
) -> dict:
    """Compute summary stats for one cross-correlation trace."""
    if corr.size == 0:
        return {
            "n_lags": 0,
            "zero_lag_correlation": None,
            "peak_lag": None,
            "peak_correlation": None,
        }

    zero_lag_corr = None
    zero_matches = np.where(lags == 0)[0]
    if zero_matches.size > 0:
        zero_lag_corr = float(corr[int(zero_matches[0])])

    peak_idx = int(np.argmax(corr))
    return {
        "n_lags": int(corr.size),
        "zero_lag_correlation": zero_lag_corr,
        "peak_lag": int(lags[peak_idx]),
        "peak_correlation": float(corr[peak_idx]),
    }


def _normalize_corr(corr: np.ndarray, x_count: int, y_count: int) -> np.ndarray:
    """Normalize cross-correlation by sqrt(m1_count * m2_count)."""
    norm_factor = float(np.sqrt(float(x_count) * float(y_count)))
    if norm_factor <= 0.0:
        return np.zeros(corr.size, dtype=np.float32)
    return (np.asarray(corr, dtype=np.float64) / norm_factor).astype(np.float32)


def _build_pair_result(
    fixation_label: str,
    max_lag: Optional[int],
    key1: tuple[str, str],
    key2: tuple[str, str],
    m1_path,
    m2_path,
) -> Optional[dict]:
    """Build one pair result, including normalized cross-correlation and metadata."""
    m1_vec, m1_name = _load_fixation_vector(m1_path, fixation_label)
    if m1_vec is None or m1_vec.size == 0:
        return None
    m2_vec, m2_name = _load_fixation_vector(m2_path, fixation_label)
    if m2_vec is None or m2_vec.size == 0:
        return None

    lags, corr = _fft_cross_correlation(m1_vec, m2_vec, max_lag=max_lag)
    m1_fix_count = int(np.count_nonzero(m1_vec))
    m2_fix_count = int(np.count_nonzero(m2_vec))
    corr_normalized = _normalize_corr(corr, m1_fix_count, m2_fix_count)

    result = {
        "key1": key1,
        "key2": key2,
        "lags": lags,
        "cross_correlation": corr_normalized,
        "n_samples_m1": int(m1_vec.size),
        "n_samples_m2": int(m2_vec.size),
        "m1_fixation_count": m1_fix_count,
        "m2_fixation_count": m2_fix_count,
        "monkey_name_m1": m1_name,
        "monkey_name_m2": m2_name,
    }
    result.update(_summarize_corr(lags, corr_normalized))
    return result


def _build_pair_result_worker(
    args: tuple[str, Optional[int], tuple[str, str], tuple[str, str], object, object],
) -> Optional[dict]:
    """Pool worker wrapper for computing one pair result."""
    fixation_label, max_lag, key1, key2, m1_path, m2_path = args
    return _build_pair_result(
        fixation_label=fixation_label,
        max_lag=max_lag,
        key1=key1,
        key2=key2,
        m1_path=m1_path,
        m2_path=m2_path,
    )


def _assert_lags_match(expected_lags: np.ndarray, lags: np.ndarray) -> None:
    """Validate that all results share the same lag axis."""
    if expected_lags.shape != lags.shape or not np.array_equal(expected_lags, lags):
        raise RuntimeError(
            "Encountered mismatched lag vectors across pairs. "
            "Use a fixed max_lag (e.g., 60000) so lags are consistent."
        )


def _build_within_session_rows(
    settings: FixCrossCorrelationSettings,
    m1_paths: dict,
    m2_paths: dict,
) -> tuple[list[dict], dict[tuple[str, str], dict], Optional[np.ndarray]]:
    """Build within-session rows and return per-session metadata + lag axis."""
    rows: list[dict] = []
    metadata_by_key: dict[tuple[str, str], dict] = {}
    lag_axis: Optional[np.ndarray] = None

    shared_keys = sorted(set(m1_paths).intersection(m2_paths))
    if settings.test_single and shared_keys:
        shared_keys = [random.choice(shared_keys)]

    tasks = [
        (
            settings.fixation_label,
            settings.max_lag,
            key,
            key,
            m1_paths[key],
            m2_paths[key],
        )
        for key in shared_keys
    ]

    use_parallel = (
        settings.parallelize_across_crosscorr_pairs
        and not settings.test_single
        and len(tasks) > 1
    )

    if use_parallel:
        n_proc = get_n_processes(max_procs=32)
        with Pool(processes=n_proc) as pool:
            iterator = pool.imap(_build_pair_result_worker, tasks)
            for result in tqdm(
                iterator,
                total=len(tasks),
                desc=f"Within-session xcorr ({n_proc} workers)",
                unit="session",
            ):
                if result is None:
                    continue
                lags = result["lags"]
                if lag_axis is None:
                    lag_axis = lags
                else:
                    _assert_lags_match(lag_axis, lags)

                key = result["key1"]
                row = {
                    "fixation_label": settings.fixation_label,
                    "date": key[0],
                    "session": key[1],
                    "n_samples_m1": result["n_samples_m1"],
                    "n_samples_m2": result["n_samples_m2"],
                    "m1_fixation_count": result["m1_fixation_count"],
                    "m2_fixation_count": result["m2_fixation_count"],
                    "monkey_name_m1": result["monkey_name_m1"],
                    "monkey_name_m2": result["monkey_name_m2"],
                    "cross_correlation": result["cross_correlation"],
                    "n_lags": result["n_lags"],
                    "zero_lag_correlation": result["zero_lag_correlation"],
                    "peak_lag": result["peak_lag"],
                    "peak_correlation": result["peak_correlation"],
                }
                rows.append(row)
                metadata_by_key[key] = {
                    "monkey_name_m1": result["monkey_name_m1"],
                    "monkey_name_m2": result["monkey_name_m2"],
                }
        return rows, metadata_by_key, lag_axis

    for task in tqdm(tasks, desc="Within-session xcorr", unit="session"):
        result = _build_pair_result_worker(task)
        if result is None:
            continue
        lags = result["lags"]
        if lag_axis is None:
            lag_axis = lags
        else:
            _assert_lags_match(lag_axis, lags)

        key = result["key1"]
        if settings.test_single:
            _print_test_single_debug(
                mode="within-session",
                settings=settings,
                m1_id=key,
                m2_id=key,
                m1_len=result["n_samples_m1"],
                m2_len=result["n_samples_m2"],
                m1_fix_count=result["m1_fixation_count"],
                m2_fix_count=result["m2_fixation_count"],
                lags=lags,
                corr=result["cross_correlation"],
            )

        row = {
            "fixation_label": settings.fixation_label,
            "date": key[0],
            "session": key[1],
            "n_samples_m1": result["n_samples_m1"],
            "n_samples_m2": result["n_samples_m2"],
            "m1_fixation_count": result["m1_fixation_count"],
            "m2_fixation_count": result["m2_fixation_count"],
            "monkey_name_m1": result["monkey_name_m1"],
            "monkey_name_m2": result["monkey_name_m2"],
            "cross_correlation": result["cross_correlation"],
            "n_lags": result["n_lags"],
            "zero_lag_correlation": result["zero_lag_correlation"],
            "peak_lag": result["peak_lag"],
            "peak_correlation": result["peak_correlation"],
        }
        rows.append(row)
        metadata_by_key[key] = {
            "monkey_name_m1": result["monkey_name_m1"],
            "monkey_name_m2": result["monkey_name_m2"],
        }

    return rows, metadata_by_key, lag_axis


def _build_cross_session_control_rows(
    settings: FixCrossCorrelationSettings,
    m1_paths: dict,
    m2_paths: dict,
    within_keys: list[tuple[str, str]],
    metadata_by_key: dict[tuple[str, str], dict],
    lag_axis: Optional[np.ndarray],
) -> tuple[list[dict], Optional[np.ndarray]]:
    """Build cross-session control rows aggregated per within-session key.

    For each within-session key K, aggregates all cross-session pairs where
    K appears as m1 side OR m2 side, and stores mean/std over those pairs.
    """
    m1_keys = sorted(m1_paths)
    m2_keys = sorted(m2_paths)
    within_key_set = set(within_keys)
    pairs = _build_cross_pairs(settings, m1_keys, m2_keys)

    if settings.test_single and pairs:
        pairs = [random.choice(pairs)]

    tasks = [
        (
            settings.fixation_label,
            settings.max_lag,
            key1,
            key2,
            m1_paths[key1],
            m2_paths[key2],
        )
        for key1, key2 in pairs
    ]

    accum: dict[tuple[str, str], dict] = {}

    def _update_accum(key: tuple[str, str], corr: np.ndarray) -> None:
        if key not in accum:
            accum[key] = {
                "sum": np.zeros(corr.size, dtype=np.float64),
                "sum_sq": np.zeros(corr.size, dtype=np.float64),
                "n_pairs": 0,
            }
        accum[key]["sum"] += corr
        accum[key]["sum_sq"] += corr * corr
        accum[key]["n_pairs"] += 1

    use_parallel = (
        settings.parallelize_across_crosscorr_pairs
        and not settings.test_single
        and len(tasks) > 1
    )

    if use_parallel:
        n_proc = get_n_processes(max_procs=32)
        with Pool(processes=n_proc) as pool:
            iterator = pool.imap(_build_pair_result_worker, tasks)
            for result in tqdm(
                iterator,
                total=len(tasks),
                desc=f"Cross-session xcorr ({n_proc} workers)",
                unit="pair",
            ):
                if result is None:
                    continue
                lags = result["lags"]
                if lag_axis is None:
                    lag_axis = lags
                else:
                    _assert_lags_match(lag_axis, lags)

                key1 = result["key1"]
                key2 = result["key2"]
                corr = result["cross_correlation"]
                if key1 in within_key_set:
                    _update_accum(key1, corr)
                if key2 in within_key_set and key2 != key1:
                    _update_accum(key2, corr)
    else:
        for task in tqdm(tasks, desc="Cross-session xcorr", unit="pair"):
            result = _build_pair_result_worker(task)
            if result is None:
                continue
            lags = result["lags"]
            if lag_axis is None:
                lag_axis = lags
            else:
                _assert_lags_match(lag_axis, lags)

            key1 = result["key1"]
            key2 = result["key2"]
            corr = result["cross_correlation"]

            if settings.test_single:
                _print_test_single_debug(
                    mode="cross-session",
                    settings=settings,
                    m1_id=key1,
                    m2_id=key2,
                    m1_len=result["n_samples_m1"],
                    m2_len=result["n_samples_m2"],
                    m1_fix_count=result["m1_fixation_count"],
                    m2_fix_count=result["m2_fixation_count"],
                    lags=lags,
                    corr=corr,
                )

            if key1 in within_key_set:
                _update_accum(key1, corr)
            if key2 in within_key_set and key2 != key1:
                _update_accum(key2, corr)

    rows: list[dict] = []
    if lag_axis is None:
        return rows, lag_axis

    for key in within_keys:
        meta = metadata_by_key.get(key, {})
        stats = accum.get(key)
        if stats is None or stats["n_pairs"] == 0:
            mean_corr = np.full(lag_axis.size, np.nan, dtype=np.float32)
            std_corr = np.full(lag_axis.size, np.nan, dtype=np.float32)
            n_pairs = 0
        else:
            n_pairs = int(stats["n_pairs"])
            mean = stats["sum"] / float(n_pairs)
            if n_pairs > 1:
                var = (stats["sum_sq"] - (stats["sum"] * stats["sum"]) / float(n_pairs)) / float(
                    n_pairs - 1
                )
                var = np.maximum(var, 0.0)
                std = np.sqrt(var)
            else:
                std = np.zeros_like(mean)
            mean_corr = mean.astype(np.float32)
            std_corr = std.astype(np.float32)

        rows.append({
            "fixation_label": settings.fixation_label,
            "date": key[0],
            "session": key[1],
            "monkey_name_m1": meta.get("monkey_name_m1"),
            "monkey_name_m2": meta.get("monkey_name_m2"),
            "n_pairs": n_pairs,
            "cross_correlation_mean": mean_corr,
            "cross_correlation_std": std_corr,
        })

    return rows, lag_axis


def _resolve_lags_filename(settings: FixCrossCorrelationSettings) -> str:
    """Return lag-axis output filename."""
    if settings.lags_filename:
        return settings.lags_filename
    return f"{settings.fixation_label}_crosscorrelation_lags.pkl"


def run_fix_cross_correlation_analysis(
    settings: FixCrossCorrelationSettings,
    *,
    compute_cross: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run fixation cross-correlation analysis and persist outputs.

    Output files:
    - within_filename: per-session within-session cross-correlation vectors.
    - cross_filename: per-session aggregated cross-session controls (mean/std/n_pairs).
    - <fixation_label>_crosscorrelation_lags.pkl (or settings.lags_filename): shared lag axis.
    """
    cfg = load_dataset_config(settings.cfg_path)
    m1_paths, m2_paths = _index_agent_paths(cfg, settings.input_modality)

    if not m1_paths or not m2_paths:
        raise RuntimeError(
            "Missing fixation binary vectors for m1 or m2. "
            f"Found m1={len(m1_paths)} m2={len(m2_paths)}."
        )

    within_rows, metadata_by_key, lag_axis = _build_within_session_rows(settings, m1_paths, m2_paths)
    within_df = pd.DataFrame.from_records(within_rows)

    cross_df = None
    if compute_cross:
        within_keys = sorted(metadata_by_key)
        cross_rows, lag_axis = _build_cross_session_control_rows(
            settings=settings,
            m1_paths=m1_paths,
            m2_paths=m2_paths,
            within_keys=within_keys,
            metadata_by_key=metadata_by_key,
            lag_axis=lag_axis,
        )
        cross_df = pd.DataFrame.from_records(cross_rows)

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    within_path = out_dir / settings.within_filename
    within_df.to_pickle(within_path)

    if cross_df is not None:
        cross_path = out_dir / settings.cross_filename
        cross_df.to_pickle(cross_path)

    if lag_axis is not None:
        lags_path = out_dir / _resolve_lags_filename(settings)
        with open(lags_path, "wb") as f:
            pickle.dump(lag_axis, f)

    return within_df, cross_df
