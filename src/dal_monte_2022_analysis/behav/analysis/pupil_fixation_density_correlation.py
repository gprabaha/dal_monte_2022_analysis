"""Compute per-session pupil vs fixation-density correlations."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.data.behavioral_data import (
    FixationDensityVectorsData,
    JointFixationDensityData,
    PupilSizeData,
)
from dal_monte_2022_analysis.behav.preprocessing.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.utils.io import load_pickle
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


@dataclass
class PupilFixationDensityCorrelationSettings:
    """Configuration for pupil-fixation density correlation analysis."""

    cfg_path: str
    pupil_modality: str = "smoothed_pupil_size"
    fixation_density_modality: str = "fixation_density_vectors"
    joint_fixation_density_modality: str = "joint_face_fixation_density"
    face_label: str = "face"
    correlation_method: str = "pearson"
    output_subdir: str = "pupil_fixation_density_correlation"
    output_filename: str = (
        "within_session_pupil_vs_face_fixation_density_correlation.csv"
    )
    use_parallel: bool = False
    parallel_max_procs: int = 32
    test_single: bool = False


_load_pickle = load_pickle


def _extract_monkey_name(obj) -> Optional[str]:
    """Extract monkey name metadata when available."""
    if isinstance(obj, (PupilSizeData, FixationDensityVectorsData, JointFixationDensityData)):
        return obj.context.monkey_name
    if isinstance(obj, dict):
        context = obj.get("context")
        if context is not None:
            if hasattr(context, "monkey_name"):
                return getattr(context, "monkey_name")
            if isinstance(context, dict):
                return context.get("monkey_name")
        return obj.get("monkey_name")
    return None


def _extract_pupil_vector(obj) -> Optional[np.ndarray]:
    """Extract a 1D pupil vector from supported object layouts."""
    if isinstance(obj, PupilSizeData):
        values = obj.d
    elif isinstance(obj, dict) and "d" in obj:
        values = obj["d"]
    else:
        return None
    vec = np.asarray(values, dtype=float).reshape(-1)
    return vec


def _extract_face_density_vector(
    obj,
    face_label: str,
) -> Optional[np.ndarray]:
    """Extract a face-density vector from supported object layouts."""
    if isinstance(obj, FixationDensityVectorsData):
        vectors = obj.vectors
    elif isinstance(obj, dict) and "vectors" in obj:
        vectors = obj["vectors"]
    elif isinstance(obj, dict):
        vectors = obj
    else:
        return None

    if not vectors or face_label not in vectors:
        return None
    vec = np.asarray(vectors[face_label], dtype=float).reshape(-1)
    return vec


def _extract_joint_density_vector(obj) -> Optional[np.ndarray]:
    """Extract a joint face-density vector from supported object layouts."""
    if isinstance(obj, JointFixationDensityData):
        values = obj.density
    elif isinstance(obj, dict) and "density" in obj:
        values = obj["density"]
    else:
        return None
    vec = np.asarray(values, dtype=float).reshape(-1)
    return vec


def _index_agent_paths(cfg: dict, modality: str) -> tuple[dict, dict]:
    """Index m1/m2 pickle paths by (date, session)."""
    index_df = index_processed_dataset(cfg, modality)
    rows = index_df.to_dict(orient="records")

    m1_paths: dict[tuple[str, str], object] = {}
    m2_paths: dict[tuple[str, str], object] = {}
    for row in rows:
        agent = row.get("agent")
        key = (row["date"], row["session"])
        if agent == "m1":
            m1_paths[key] = row["path"]
        elif agent == "m2":
            m2_paths[key] = row["path"]
    return m1_paths, m2_paths


def _index_shared_paths(cfg: dict, modality: str) -> dict:
    """Index shared pickle paths by (date, session)."""
    index_df = index_processed_dataset(cfg, modality)
    rows = index_df.to_dict(orient="records")

    shared_paths: dict[tuple[str, str], object] = {}
    for row in rows:
        if row.get("agent") is None:
            shared_paths[(row["date"], row["session"])] = row["path"]
    return shared_paths


def _pearson_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation from two same-length finite vectors."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= 0.0:
        return float("nan")
    r = float(np.sum(x_centered * y_centered) / denom)
    return float(np.clip(r, -1.0, 1.0))


def _correlation_with_alignment(
    pupil_vec: np.ndarray,
    density_vec: np.ndarray,
    *,
    method: str,
) -> tuple[float, int, int]:
    """Return correlation with min-length alignment and finite-value filtering."""
    p = np.asarray(pupil_vec, dtype=float).reshape(-1)
    d = np.asarray(density_vec, dtype=float).reshape(-1)

    aligned_len = int(min(p.size, d.size))
    if aligned_len == 0:
        return float("nan"), 0, 0

    p = p[:aligned_len]
    d = d[:aligned_len]
    valid = np.isfinite(p) & np.isfinite(d)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 2:
        return float("nan"), aligned_len, n_valid

    p_valid = p[valid]
    d_valid = d[valid]
    method_token = str(method).strip().lower()
    if method_token == "spearman":
        p_input = pd.Series(p_valid).rank(method="average").to_numpy(dtype=float)
        d_input = pd.Series(d_valid).rank(method="average").to_numpy(dtype=float)
    elif method_token == "pearson":
        p_input = p_valid
        d_input = d_valid
    else:
        raise ValueError(
            f"Unsupported correlation method '{method}'. Expected 'pearson' or 'spearman'."
        )

    r = _pearson_coefficient(p_input, d_input)
    return r, aligned_len, n_valid


def _build_within_session_rows(
    settings: PupilFixationDensityCorrelationSettings,
    m1_pupil_paths: dict,
    m2_pupil_paths: dict,
    m1_density_paths: dict,
    m2_density_paths: dict,
    joint_density_paths: dict,
    *,
    date: Optional[str] = None,
    session: Optional[str] = None,
) -> list[dict]:
    """Build long-format correlation rows for each date/session."""
    session_keys = sorted(
        set(m1_pupil_paths)
        .intersection(m2_pupil_paths)
        .intersection(m1_density_paths)
        .intersection(m2_density_paths)
        .intersection(joint_density_paths)
    )

    if date is not None:
        session_keys = [key for key in session_keys if key[0] == date]
    if session is not None:
        session_keys = [key for key in session_keys if key[1] == session]
    if settings.test_single and session_keys:
        session_keys = [session_keys[0]]

    tasks = [
        (
            date_value,
            session_value,
            m1_pupil_paths[(date_value, session_value)],
            m2_pupil_paths[(date_value, session_value)],
            m1_density_paths[(date_value, session_value)],
            m2_density_paths[(date_value, session_value)],
            joint_density_paths[(date_value, session_value)],
            settings.face_label,
            settings.correlation_method,
        )
        for date_value, session_value in session_keys
    ]

    rows: list[dict] = []
    if not settings.use_parallel:
        for task in tqdm(tasks, desc="Pupil-density correlations", unit="session"):
            rows.extend(_build_one_session_rows(task))
        return rows

    n_proc = get_n_processes(max_procs=settings.parallel_max_procs)
    with Pool(processes=n_proc) as pool:
        for session_rows in tqdm(
            pool.imap_unordered(_build_one_session_rows, tasks),
            total=len(tasks),
            desc=f"Pupil-density correlations ({n_proc} workers)",
            unit="session",
        ):
            rows.extend(session_rows)
    return rows


def _build_one_session_rows(args) -> list[dict]:
    """Build all pupil-density correlation rows for one date/session."""
    (
        date_value,
        session_value,
        m1_pupil_path,
        m2_pupil_path,
        m1_density_path,
        m2_density_path,
        joint_density_path,
        face_label,
        correlation_method,
    ) = args

    m1_pupil_obj = _load_pickle(m1_pupil_path)
    m2_pupil_obj = _load_pickle(m2_pupil_path)
    m1_density_obj = _load_pickle(m1_density_path)
    m2_density_obj = _load_pickle(m2_density_path)
    joint_density_obj = _load_pickle(joint_density_path)

    m1_pupil = _extract_pupil_vector(m1_pupil_obj)
    m2_pupil = _extract_pupil_vector(m2_pupil_obj)
    m1_density = _extract_face_density_vector(m1_density_obj, face_label)
    m2_density = _extract_face_density_vector(m2_density_obj, face_label)
    joint_density = _extract_joint_density_vector(joint_density_obj)

    if (
        m1_pupil is None
        or m2_pupil is None
        or m1_density is None
        or m2_density is None
        or joint_density is None
    ):
        return []

    pupil_vectors = {
        "m1": (m1_pupil, _extract_monkey_name(m1_pupil_obj)),
        "m2": (m2_pupil, _extract_monkey_name(m2_pupil_obj)),
    }
    density_vectors = {
        "m1": (m1_density, _extract_monkey_name(m1_density_obj)),
        "m2": (m2_density, _extract_monkey_name(m2_density_obj)),
        "joint": (joint_density, _extract_monkey_name(joint_density_obj)),
    }

    rows: list[dict] = []
    for pupil_agent, (pupil_vec, pupil_monkey_name) in pupil_vectors.items():
        for density_source, (density_vec, density_monkey_name) in density_vectors.items():
            corr_r, aligned_len, n_valid = _correlation_with_alignment(
                pupil_vec,
                density_vec,
                method=correlation_method,
            )
            rows.append({
                "date": date_value,
                "session": session_value,
                "pupil_agent": pupil_agent,
                "pupil_monkey_name": pupil_monkey_name,
                "density_source": density_source,
                "density_monkey_name": density_monkey_name,
                "correlation_method": str(correlation_method).strip().lower(),
                "correlation_r": corr_r,
                # Backward-compat columns used by existing plotting code.
                "pearson_r": corr_r if str(correlation_method).strip().lower() == "pearson" else np.nan,
                "spearman_r": corr_r if str(correlation_method).strip().lower() == "spearman" else np.nan,
                "aligned_n_samples": aligned_len,
                "n_valid_samples": n_valid,
                "pupil_length": int(pupil_vec.size),
                "density_length": int(density_vec.size),
            })

    return rows


def run_pupil_fixation_density_correlation_analysis(
    settings: PupilFixationDensityCorrelationSettings,
    *,
    date: Optional[str] = None,
    session: Optional[str] = None,
) -> pd.DataFrame:
    """Compute and save per-session pupil vs fixation-density correlations."""
    settings.correlation_method = str(settings.correlation_method).strip().lower()
    if settings.correlation_method not in {"pearson", "spearman"}:
        raise ValueError(
            "Unsupported correlation_method. Expected 'pearson' or 'spearman'."
        )

    cfg = load_config(settings.cfg_path)
    m1_pupil_paths, m2_pupil_paths = _index_agent_paths(cfg, settings.pupil_modality)
    m1_density_paths, m2_density_paths = _index_agent_paths(
        cfg,
        settings.fixation_density_modality,
    )
    joint_density_paths = _index_shared_paths(cfg, settings.joint_fixation_density_modality)

    rows = _build_within_session_rows(
        settings,
        m1_pupil_paths,
        m2_pupil_paths,
        m1_density_paths,
        m2_density_paths,
        joint_density_paths,
        date=date,
        session=session,
    )
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(
            ["date", "session", "pupil_agent", "density_source"]
        ).reset_index(drop=True)

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / settings.output_filename
    out_df.to_csv(out_path, index=False)
    print(f"Saved pupil-density correlations to: {out_path} (rows={len(out_df)})")
    return out_df
