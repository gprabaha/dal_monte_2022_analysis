"""Fixation-guided pupil smoothing and interpolation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.behav.pupil_smoothing import (
    build_fixation_mask as _build_fixation_mask,
    coerce_intervals as _coerce_intervals,
    estimate_fixation_noise as _estimate_fixation_noise,
    gaussian_smooth_1d as _gaussian_smooth_1d,
    interp_nan_1d as _interp_nan_1d,
    interpolate_fixation_gaps as _interpolate_fixation_gaps,
    resolve_sigma as _resolve_sigma,
)
from dal_monte_2022_analysis.data.records.behavioral import PupilSizeData, RecordingContext
from dal_monte_2022_analysis.behav.preprocessing.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.runtime.io.processed_data import (
    build_processed_pickle_path,
    load_pickle_path,
    save_processed_pickle,
)
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks


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

    run_tasks(
        _worker,
        tasks,
        desc="Smoothing pupil",
        unit="session",
        use_parallel=parallel,
        max_procs=8,
    )
