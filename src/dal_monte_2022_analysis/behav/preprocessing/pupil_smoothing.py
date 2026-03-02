"""Fixation-guided pupil smoothing and interpolation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.data.behavioral_data import PupilSizeData, RecordingContext
from dal_monte_2022_analysis.behav.preprocessing.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.runtime.io.processed_data import (
    build_processed_pickle_path,
    load_pickle_path,
    save_processed_pickle,
)
from dal_monte_2022_analysis.utils.parallel import get_n_processes


@dataclass
class PupilSmoothingSettings:
    """Configuration for fixation-guided pupil smoothing."""

    cfg_path: str
    input_pupil_modality: str = "pupil_size"
    fixations_modality: str = "fixations"
    output_modality: str = "smoothed_pupil_size"
    base_sigma_samples: float = 6.0
    min_sigma_samples: float = 2.0
    max_sigma_samples: float = 40.0
    adaptive_noise_gain: float = 2.0
    flatten_within_fixation: bool = True
    use_parallel: bool = True
    test_single: bool = False
    agents: Optional[Sequence[str]] = None


def _coerce_intervals(
    fix_df: pd.DataFrame,
    *,
    n_samples: int,
) -> list[tuple[int, int]]:
    """Extract valid clipped fixation start/stop intervals."""
    intervals: list[tuple[int, int]] = []
    if not isinstance(fix_df, pd.DataFrame) or fix_df.empty:
        return intervals
    if "start" not in fix_df.columns or "stop" not in fix_df.columns:
        return intervals

    for _, row in fix_df.iterrows():
        try:
            start = int(row["start"])
            stop = int(row["stop"])
        except (TypeError, ValueError):
            continue
        if stop < 0 or start >= n_samples:
            continue
        start = max(0, start)
        stop = min(n_samples - 1, stop)
        if start > stop:
            continue
        intervals.append((start, stop))

    return intervals


def _build_fixation_mask(
    intervals: list[tuple[int, int]],
    *,
    n_samples: int,
) -> np.ndarray:
    """Create a boolean fixation mask from start/stop intervals."""
    mask = np.zeros(n_samples, dtype=bool)
    for start, stop in intervals:
        mask[start : stop + 1] = True
    return mask


def _interp_nan_1d(values: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 1D array via shape-preserving cubic interpolation (PCHIP)."""
    arr = np.asarray(values, dtype=float).reshape(-1).copy()
    idx = np.arange(arr.size, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=float)
    return _pchip_interpolate_1d(
        query_idx=idx,
        known_idx=idx[valid],
        known_values=arr[valid],
    )


def _pchip_interpolate_1d(
    *,
    query_idx: np.ndarray,
    known_idx: np.ndarray,
    known_values: np.ndarray,
) -> np.ndarray:
    """Interpolate a 1D signal with PCHIP, edge-filling outside known range."""
    q = np.asarray(query_idx, dtype=float).reshape(-1)
    x = np.asarray(known_idx, dtype=float).reshape(-1)
    y = np.asarray(known_values, dtype=float).reshape(-1)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return np.zeros_like(q, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]

    if x.size == 1:
        return np.full_like(q, float(y[0]), dtype=float)

    interpolator = PchipInterpolator(x, y, extrapolate=False)
    out = interpolator(q).astype(float)

    left_mask = q < x[0]
    right_mask = q > x[-1]
    out[left_mask] = y[0]
    out[right_mask] = y[-1]

    nan_mask = ~np.isfinite(out)
    if np.any(nan_mask):
        out[nan_mask & (q <= x[0])] = y[0]
        out[nan_mask & (q >= x[-1])] = y[-1]
    return out


def _interpolate_fixation_gaps(fixation_anchor: np.ndarray) -> np.ndarray:
    """Interpolate non-fixation gaps from fixation anchors using PCHIP."""
    anchor = np.asarray(fixation_anchor, dtype=float).reshape(-1)
    idx = np.arange(anchor.size, dtype=float)
    valid = np.isfinite(anchor)
    if np.count_nonzero(valid) == 0:
        return np.zeros_like(anchor, dtype=float)
    return _pchip_interpolate_1d(
        query_idx=idx,
        known_idx=idx[valid],
        known_values=anchor[valid],
    )


def _gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 1D signal with a Gaussian kernel (reflect padding)."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    sigma = float(max(sigma, 1e-6))
    if arr.size <= 2 or sigma <= 0.25:
        return arr.copy()

    radius = int(max(1, np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)

    padded = np.pad(arr, (radius, radius), mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(float, copy=False)


def _estimate_fixation_noise(
    raw_pupil: np.ndarray,
    intervals: list[tuple[int, int]],
) -> tuple[float, np.ndarray]:
    """Estimate pupil noise from within-fixation residual variability."""
    residuals: list[np.ndarray] = []
    fixation_values: list[np.ndarray] = []

    for start, stop in intervals:
        seg = raw_pupil[start : stop + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        fixation_values.append(seg)
        if seg.size < 2:
            continue
        seg_median = np.median(seg)
        residuals.append(seg - seg_median)

    if fixation_values:
        fix_vals = np.concatenate(fixation_values)
    else:
        fix_vals = raw_pupil[np.isfinite(raw_pupil)]

    if residuals:
        resid = np.concatenate(residuals)
        noise_sigma = 1.4826 * np.median(np.abs(resid))
    elif fix_vals.size > 1:
        noise_sigma = float(np.nanstd(fix_vals))
    else:
        noise_sigma = 0.0

    return float(max(noise_sigma, 0.0)), np.asarray(fix_vals, dtype=float)


def _resolve_sigma(
    *,
    noise_sigma: float,
    fixation_values: np.ndarray,
    base_sigma_samples: float,
    min_sigma_samples: float,
    max_sigma_samples: float,
    adaptive_noise_gain: float,
) -> float:
    """Resolve smoothing sigma from fixation-derived noise variability."""
    arr = np.asarray(fixation_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size > 3:
        q1, q3 = np.percentile(arr, [25.0, 75.0])
        scale = float(max(q3 - q1, 1e-6))
    elif arr.size > 1:
        scale = float(max(np.nanstd(arr), 1e-6))
    else:
        scale = 1.0

    noise_ratio = float(max(noise_sigma / scale, 0.0))
    sigma = float(base_sigma_samples) * (1.0 + float(adaptive_noise_gain) * noise_ratio)
    sigma = float(np.clip(sigma, float(min_sigma_samples), float(max_sigma_samples)))
    return sigma


def smooth_pupil_for_row_agent(
    settings: PupilSmoothingSettings,
    row: dict,
    agent: str,
) -> Optional[PupilSizeData]:
    """Smooth pupil for one date/session/agent using fixation-constrained interpolation."""
    cfg = load_config(settings.cfg_path)
    pupil_path = build_processed_pickle_path(cfg, row, settings.input_pupil_modality, agent)
    fix_path = build_processed_pickle_path(cfg, row, settings.fixations_modality, agent)

    if not pupil_path.exists() or not fix_path.exists():
        return None

    pupil_obj = load_pickle_path(pupil_path)
    fix_df = load_pickle_path(fix_path)
    raw_pupil = np.asarray(pupil_obj.d, dtype=float).reshape(-1)
    if raw_pupil.size == 0:
        return None

    intervals = _coerce_intervals(fix_df, n_samples=raw_pupil.size)
    if not intervals:
        return None

    fixation_mask = _build_fixation_mask(intervals, n_samples=raw_pupil.size)
    if not np.any(fixation_mask):
        return None

    noise_sigma, fixation_values = _estimate_fixation_noise(raw_pupil, intervals)
    sigma = _resolve_sigma(
        noise_sigma=noise_sigma,
        fixation_values=fixation_values,
        base_sigma_samples=settings.base_sigma_samples,
        min_sigma_samples=settings.min_sigma_samples,
        max_sigma_samples=settings.max_sigma_samples,
        adaptive_noise_gain=settings.adaptive_noise_gain,
    )

    nan_filled = _interp_nan_1d(raw_pupil)
    smoothed = _gaussian_smooth_1d(nan_filled, sigma=sigma)

    fixation_anchor = np.full(raw_pupil.shape, np.nan, dtype=float)
    for start, stop in intervals:
        seg = smoothed[start : stop + 1]
        if seg.size == 0:
            continue
        if settings.flatten_within_fixation:
            anchor_val = float(np.median(seg))
            fixation_anchor[start : stop + 1] = anchor_val
        else:
            fixation_anchor[start : stop + 1] = seg

    valid = np.isfinite(fixation_anchor)
    if np.count_nonzero(valid) == 0:
        return None

    interp = _interpolate_fixation_gaps(fixation_anchor)

    monkey_name = getattr(pupil_obj.context, "monkey_name", None)
    context = RecordingContext(
        date=row["date"],
        session=row["session"],
        agent=agent,
        monkey_name=monkey_name,
    )
    return PupilSizeData(context=context, d=interp)


def process_and_save_smoothed_pupil_for_row_agent(
    settings: PupilSmoothingSettings,
    row: dict,
    agent: str,
) -> int:
    """Build and persist smoothed pupil data for one row/agent."""
    data = smooth_pupil_for_row_agent(settings, row, agent)
    if data is None:
        return 0
    cfg = load_config(settings.cfg_path)
    save_processed_pickle(data, cfg, row, settings.output_modality, agent)
    return 1


def _worker(args) -> int:
    """Worker wrapper returning 1 when output was written."""
    settings, row, agent = args
    return process_and_save_smoothed_pupil_for_row_agent(settings, row, agent)


def build_pupil_smoothing_tasks(
    settings: PupilSmoothingSettings,
    *,
    test_single: bool = False,
) -> list[tuple[PupilSmoothingSettings, dict, str]]:
    """Build smoothing tasks from processed pupil files."""
    cfg = load_config(settings.cfg_path)
    index_df = index_processed_dataset(cfg, settings.input_pupil_modality)
    rows = index_df.to_dict(orient="records")

    agent_filter = None
    if settings.agents is not None:
        agent_filter = {str(agent) for agent in settings.agents}

    tasks: list[tuple[PupilSmoothingSettings, dict, str]] = []
    for row in rows:
        agent = row.get("agent")
        if agent is None:
            continue
        if agent_filter is not None and str(agent) not in agent_filter:
            continue
        task_row = {"date": row["date"], "session": row["session"]}
        tasks.append((settings, task_row, str(agent)))

    if test_single and tasks:
        return [tasks[0]]
    return tasks


def run_pupil_smoothing(
    settings: PupilSmoothingSettings,
    *,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> None:
    """Run fixation-guided pupil smoothing across all tasks."""
    parallel = settings.use_parallel if use_parallel is None else bool(use_parallel)
    single = settings.test_single if test_single is None else bool(test_single)

    tasks = build_pupil_smoothing_tasks(settings, test_single=single)
    if not tasks:
        print("No pupil smoothing tasks found.")
        return

    if not parallel or len(tasks) == 1:
        for task in tqdm(tasks, desc="Smoothing pupil (serial)", unit="session"):
            _worker(task)
        return

    n_proc = get_n_processes(max_procs=8)
    with Pool(processes=n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(_worker, tasks),
            total=len(tasks),
            desc=f"Smoothing pupil ({n_proc} workers)",
            unit="session",
        ):
            pass
