"""Compute face fixation probability stats within and across sessions."""

from __future__ import annotations

import pdb
from dataclasses import dataclass
from decimal import Decimal, localcontext
import pickle
import random
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.data.gaze_data import FixationBinaryVectorsData
from dal_monte_2022_analysis.io.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


@dataclass
class FaceFixationProbabilitySettings:
    """Configuration for face fixation probability analysis."""
    cfg_path: str
    input_modality: str = "fixation_binary_vectors"
    face_label: str = "face"
    output_subdir: str = "face_fixation_probability"
    within_filename: str = "within_session_face_fixation_probability.csv"
    cross_filename: str = "cross_session_face_fixation_probability.csv"
    violin_filename: str = "face_fixation_probability_violin.csv"
    decimal_precision: int = 50
    cross_pairs_max: Optional[int] = None
    cross_pairs_seed: int = 13
    cross_exclude_same_session: bool = True
    cross_exclude_same_date: bool = False
    test_single: bool = False


def _load_pickle(path):
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_face_vector(obj, face_label: str) -> Optional[np.ndarray]:
    """Extract a face fixation vector from supported inputs."""
    if isinstance(obj, FixationBinaryVectorsData):
        vectors = obj.vectors
    elif isinstance(obj, dict) and "vectors" in obj:
        vectors = obj["vectors"]
    elif isinstance(obj, dict):
        vectors = obj
    else:
        return None

    if not vectors or face_label not in vectors:
        return None

    vec = np.asarray(vectors[face_label])
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    return vec


def _to_bool(vec: np.ndarray) -> np.ndarray:
    """Coerce a vector to a 1D boolean array."""
    return np.asarray(vec).astype(bool, copy=False)


def _decimal_ratio(numer: int, denom: int, precision: int) -> Optional[Decimal]:
    """Return numer/denom as a Decimal with specified precision."""
    if denom <= 0:
        return None
    with localcontext() as ctx:
        ctx.prec = precision
        return Decimal(numer) / Decimal(denom)


def _decimal_str(value: Optional[Decimal]) -> Optional[str]:
    """Render a Decimal as a fixed-point string (no exponent)."""
    if value is None:
        return None
    return format(value, "f")


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


def _load_face_vector(path, face_label: str) -> Optional[np.ndarray]:
    """Load a face fixation vector from a pickle path."""
    obj = _load_pickle(path)
    return _extract_face_vector(obj, face_label)


def _build_within_session_rows(
    settings: FaceFixationProbabilitySettings,
    m1_paths: dict,
    m2_paths: dict,
) -> list[dict]:
    """Build within-session probability rows."""
    rows: list[dict] = []
    shared_keys = sorted(set(m1_paths).intersection(m2_paths))

    if settings.test_single and shared_keys:
        shared_keys = [shared_keys[0]]

    for key in tqdm(shared_keys, desc="Within-session face fixation", unit="session"):
        date, session = key
        m1_vec = _load_face_vector(m1_paths[key], settings.face_label)
        m2_vec = _load_face_vector(m2_paths[key], settings.face_label)
        if m1_vec is None or m2_vec is None:
            continue

        m1_bool = _to_bool(m1_vec)
        m2_bool = _to_bool(m2_vec)
        if m1_bool.size == 0 or m2_bool.size == 0:
            continue
        if m1_bool.size != m2_bool.size:
            continue

        n_samples = int(m1_bool.size)
        m1_count = int(np.count_nonzero(m1_bool))
        m2_count = int(np.count_nonzero(m2_bool))
        joint_count = int(np.count_nonzero(m1_bool & m2_bool))

        p_m1 = _decimal_ratio(m1_count, n_samples, settings.decimal_precision)
        p_m2 = _decimal_ratio(m2_count, n_samples, settings.decimal_precision)
        p_joint = _decimal_ratio(joint_count, n_samples, settings.decimal_precision)
        p_product = None
        if p_m1 is not None and p_m2 is not None:
            with localcontext() as ctx:
                ctx.prec = settings.decimal_precision
                p_product = p_m1 * p_m2

        rows.append({
            "date": date,
            "session": session,
            "n_samples": n_samples,
            "m1_face_count": m1_count,
            "m2_face_count": m2_count,
            "joint_face_count": joint_count,
            "p_m1_decimal": _decimal_str(p_m1),
            "p_m2_decimal": _decimal_str(p_m2),
            "p_joint_decimal": _decimal_str(p_joint),
            "p_product_decimal": _decimal_str(p_product),
        })

    return rows


def _build_cross_pairs(
    settings: FaceFixationProbabilitySettings,
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


def _build_cross_session_rows(
    settings: FaceFixationProbabilitySettings,
    m1_paths: dict,
    m2_paths: dict,
) -> list[dict]:
    """Build cross-session probability rows."""
    rows: list[dict] = []
    m1_keys = sorted(m1_paths)
    m2_keys = sorted(m2_paths)

    pairs = _build_cross_pairs(settings, m1_keys, m2_keys)
    if settings.test_single and pairs:
        pairs = pairs[: min(10, len(pairs))]

    m1_cache: dict[tuple[str, str], np.ndarray] = {}
    m2_cache: dict[tuple[str, str], np.ndarray] = {}

    for (date1, session1), (date2, session2) in tqdm(
        pairs,
        desc="Cross-session face fixation",
        unit="pair",
    ):
        key1 = (date1, session1)
        key2 = (date2, session2)

        if key1 not in m1_cache:
            m1_vec = _load_face_vector(m1_paths[key1], settings.face_label)
            if m1_vec is None:
                continue
            m1_cache[key1] = _to_bool(m1_vec)
        if key2 not in m2_cache:
            m2_vec = _load_face_vector(m2_paths[key2], settings.face_label)
            if m2_vec is None:
                continue
            m2_cache[key2] = _to_bool(m2_vec)

        m1_bool = m1_cache[key1]
        m2_bool = m2_cache[key2]
        if m1_bool.size == 0 or m2_bool.size == 0:
            continue

        n_samples_m1 = int(m1_bool.size)
        n_samples_m2 = int(m2_bool.size)
        n_joint = int(min(n_samples_m1, n_samples_m2))
        if n_joint == 0:
            continue

        m1_slice = m1_bool[:n_joint]
        m2_slice = m2_bool[:n_joint]

        m1_count = int(np.count_nonzero(m1_slice))
        m2_count = int(np.count_nonzero(m2_slice))
        joint_count = int(np.count_nonzero(m1_slice & m2_slice))

        p_m1 = _decimal_ratio(m1_count, n_joint, settings.decimal_precision)
        p_m2 = _decimal_ratio(m2_count, n_joint, settings.decimal_precision)
        p_joint = _decimal_ratio(joint_count, n_joint, settings.decimal_precision)
        p_product = None
        if p_m1 is not None and p_m2 is not None:
            with localcontext() as ctx:
                ctx.prec = settings.decimal_precision
                p_product = p_m1 * p_m2

        rows.append({
            "date_m1": date1,
            "session_m1": session1,
            "date_m2": date2,
            "session_m2": session2,
            "n_samples_m1": n_samples_m1,
            "n_samples_m2": n_samples_m2,
            "n_samples_joint": n_joint,
            "m1_face_count_joint": m1_count,
            "m2_face_count_joint": m2_count,
            "joint_face_count": joint_count,
            "p_m1_decimal": _decimal_str(p_m1),
            "p_m2_decimal": _decimal_str(p_m2),
            "p_joint_decimal": _decimal_str(p_joint),
            "p_product_decimal": _decimal_str(p_product),
        })

    return rows


def _build_violin_rows(
    within_df: pd.DataFrame,
    cross_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build a long-form dataframe for violin plotting."""
    records: list[dict] = []

    if within_df is not None and not within_df.empty:
        for _, row in within_df.iterrows():
            records.append({
                "mode": "within_session",
                "comparison": "product",
                "value_decimal": row["p_product_decimal"],
                "date_m1": row["date"],
                "session_m1": row["session"],
                "date_m2": row["date"],
                "session_m2": row["session"],
            })
            records.append({
                "mode": "within_session",
                "comparison": "joint",
                "value_decimal": row["p_joint_decimal"],
                "date_m1": row["date"],
                "session_m1": row["session"],
                "date_m2": row["date"],
                "session_m2": row["session"],
            })

    if cross_df is not None and not cross_df.empty:
        for _, row in cross_df.iterrows():
            records.append({
                "mode": "cross_session",
                "comparison": "product",
                "value_decimal": row["p_product_decimal"],
                "date_m1": row["date_m1"],
                "session_m1": row["session_m1"],
                "date_m2": row["date_m2"],
                "session_m2": row["session_m2"],
            })
            records.append({
                "mode": "cross_session",
                "comparison": "joint",
                "value_decimal": row["p_joint_decimal"],
                "date_m1": row["date_m1"],
                "session_m1": row["session_m1"],
                "date_m2": row["date_m2"],
                "session_m2": row["session_m2"],
            })

    return pd.DataFrame.from_records(records)


def run_face_fixation_probability_analysis(
    settings: FaceFixationProbabilitySettings,
    *,
    compute_cross: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Run face fixation probability analysis and persist outputs."""
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

    violin_df = _build_violin_rows(within_df, cross_df)

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    within_path = out_dir / settings.within_filename
    within_df.to_csv(within_path, index=False)

    if cross_df is not None:
        cross_path = out_dir / settings.cross_filename
        cross_df.to_csv(cross_path, index=False)

    violin_path = out_dir / settings.violin_filename
    violin_df.to_csv(violin_path, index=False)

    return within_df, cross_df, violin_df
