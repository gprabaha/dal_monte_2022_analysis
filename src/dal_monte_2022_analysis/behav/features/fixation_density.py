"""Compute fixation density vectors from fixation binary vectors."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.data.records.behavioral import (
    FixationBinaryVectorsData,
    FixationDensityVectorsData,
    RecordingContext,
)
from dal_monte_2022_analysis.behav.preprocessing.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.runtime.io.processed_data import (
    build_processed_pickle_path,
    load_pickle_path,
    save_processed_pickle,
)
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks
from dal_monte_2022_analysis.core.behav.roi_groups import (
    DEFAULT_FIXATION_ROI_GROUPS,
    coerce_location_labels,
    locations_match,
    resolve_agent_roi_groups,
)


DEFAULT_ROI_GROUPS: Dict[str, Sequence[str]] = {
    key: value for key, value in DEFAULT_FIXATION_ROI_GROUPS.items() if key != "out_of_roi"
}


@dataclass
class FixationDensitySettings:
    """Configuration for building fixation density vectors."""
    cfg_path: str
    fixations_modality: str = "fixations"
    fixation_vectors_modality: str = "fixation_binary_vectors"
    output_modality: str = "fixation_density_vectors"
    roi_groups: Dict[str, Sequence[str]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_ROI_GROUPS.items()}
    )
    agent_roi_groups: Optional[Dict[str, Dict[str, Sequence[str]]]] = None
    binwidth_method: str = "mean"
    sigma_method: str = "binwidth"
    kernel_width_factor: float = 6.0
    min_kernel_width: int = 3
    sigma_floor: float = 1.0
    truncate_sigmas: float = 3.0
    inter_fixation_fallback: str = "fixation_duration"
    normalize: bool = True
    use_parallel: bool = False
    test_single: bool = False
    agents: Optional[Sequence[str]] = None


def _resolve_roi_groups(
    settings: FixationDensitySettings,
    agent: str,
) -> Dict[str, list[str]]:
    """Resolve ROI groups for an agent (per-agent overrides take precedence)."""
    return resolve_agent_roi_groups(
        agent=agent,
        roi_groups=settings.roi_groups,
        agent_roi_groups=settings.agent_roi_groups,
        include_defaults=False,
    )


def _extract_fixation_events(
    fix_df: pd.DataFrame,
    keywords: Sequence[str],
) -> list[tuple[int, int]]:
    """Extract (start, stop) events for fixations matching the ROI keywords."""
    events: list[tuple[int, int]] = []
    if fix_df is None or fix_df.empty:
        return events
    if "location" not in fix_df.columns:
        return events

    for _, row in fix_df.iterrows():
        try:
            start = int(row.get("start"))
            stop = int(row.get("stop"))
        except (TypeError, ValueError):
            continue
        if stop < start:
            continue
        locations = coerce_location_labels(row.get("location"), lowercase=True)
        if locations_match(locations, keywords):
            events.append((start, stop))

    events.sort(key=lambda pair: pair[0])
    return events


def _compute_fixation_stats(
    events: Sequence[tuple[int, int]],
    *,
    inter_fixation_fallback: str,
) -> tuple[float, float]:
    """Compute mean fixation duration and inter-fixation gap length."""
    durations = [stop - start + 1 for start, stop in events if stop >= start]
    avg_fix = float(np.mean(durations)) if durations else 0.0

    gaps = []
    for (_, prev_stop), (next_start, _) in zip(events, events[1:]):
        gap = max(0, next_start - prev_stop - 1)
        gaps.append(gap)

    if gaps:
        avg_gap = float(np.mean(gaps))
    elif inter_fixation_fallback == "fixation_duration":
        avg_gap = avg_fix
    else:
        avg_gap = 0.0

    return avg_fix, avg_gap


def _compute_kernel_from_binwidth(
    binwidth_value: float,
    settings: FixationDensitySettings,
) -> tuple[int, float, float]:
    """Resolve kernel binwidth/sigma/truncate from a binwidth value."""
    binwidth = max(settings.min_kernel_width, int(round(float(binwidth_value))))
    if settings.sigma_method == "binwidth_over_factor":
        width_factor = settings.kernel_width_factor if settings.kernel_width_factor > 0 else 1.0
        sigma = binwidth / width_factor
    elif settings.sigma_method == "binwidth":
        sigma = float(binwidth)
    else:
        raise ValueError(
            "Unsupported sigma_method. Expected 'binwidth' or 'binwidth_over_factor'."
        )
    sigma = max(settings.sigma_floor, sigma)
    truncate = max(0.0, float(settings.truncate_sigmas))
    return binwidth, sigma, truncate


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize an array between 0 and 1."""
    if values.size == 0:
        return values
    min_val = float(values.min())
    max_val = float(values.max())
    if np.isclose(max_val, min_val):
        return np.zeros_like(values, dtype=float)
    return (values - min_val) / (max_val - min_val)


def _extract_monkey_name(fix_df: pd.DataFrame) -> Optional[str]:
    """Extract a monkey name from a fixation DataFrame if available."""
    if fix_df is None or fix_df.empty or "monkey_name" not in fix_df.columns:
        return None
    valid = fix_df["monkey_name"].dropna()
    if valid.empty:
        return None
    return str(valid.iloc[0])


def _compute_global_face_kernel_parameters(
    settings: FixationDensitySettings,
) -> tuple[int, float, float, dict]:
    """Compute one shared kernel from m1/m2 face fixation stats across sessions."""
    cfg = load_config(settings.cfg_path)
    fix_index_df = index_processed_dataset(cfg, settings.fixations_modality)
    fix_rows = fix_index_df.to_dict(orient="records")

    required_agents = ("m1", "m2")
    per_agent_stats: dict[str, dict[str, float]] = {}
    for agent in required_agents:
        roi_groups = _resolve_roi_groups(settings, agent)
        face_keywords = roi_groups.get("face")
        if not face_keywords:
            raise RuntimeError(
                f"Face ROI keywords are missing for agent '{agent}' in fixation density settings."
            )

        session_fix_means: list[float] = []
        session_gap_means: list[float] = []
        for row in fix_rows:
            if str(row.get("agent")) != agent:
                continue
            fix_df = load_pickle_path(row["path"])
            if not isinstance(fix_df, pd.DataFrame) or fix_df.empty:
                continue
            face_events = _extract_fixation_events(fix_df, face_keywords)
            if not face_events:
                continue
            avg_fix, avg_gap = _compute_fixation_stats(
                face_events,
                inter_fixation_fallback=settings.inter_fixation_fallback,
            )
            session_fix_means.append(float(avg_fix))
            session_gap_means.append(float(avg_gap))

        if not session_fix_means or not session_gap_means:
            raise RuntimeError(
                f"No usable face fixation events found for agent '{agent}' "
                "while estimating shared fixation-density kernel."
            )

        per_agent_stats[agent] = {
            "avg_fixation_duration": float(np.mean(session_fix_means)),
            "avg_inter_fixation_duration": float(np.mean(session_gap_means)),
            "n_sessions_with_face_events": float(len(session_fix_means)),
        }

    binwidth_components = [
        per_agent_stats["m1"]["avg_fixation_duration"],
        per_agent_stats["m1"]["avg_inter_fixation_duration"],
        per_agent_stats["m2"]["avg_fixation_duration"],
        per_agent_stats["m2"]["avg_inter_fixation_duration"],
    ]
    binwidth_raw = float(np.mean(binwidth_components))
    binwidth, sigma, truncate = _compute_kernel_from_binwidth(binwidth_raw, settings)

    stats = {
        "m1_avg_fixation_duration": per_agent_stats["m1"]["avg_fixation_duration"],
        "m1_avg_inter_fixation_duration": per_agent_stats["m1"][
            "avg_inter_fixation_duration"
        ],
        "m2_avg_fixation_duration": per_agent_stats["m2"]["avg_fixation_duration"],
        "m2_avg_inter_fixation_duration": per_agent_stats["m2"][
            "avg_inter_fixation_duration"
        ],
        "m1_n_sessions_with_face_events": per_agent_stats["m1"][
            "n_sessions_with_face_events"
        ],
        "m2_n_sessions_with_face_events": per_agent_stats["m2"][
            "n_sessions_with_face_events"
        ],
        "global_face_binwidth_raw": binwidth_raw,
        "kernel_binwidth": float(binwidth),
        "kernel_sigma": float(sigma),
        "kernel_truncate": float(truncate),
    }
    return binwidth, sigma, truncate, stats


def build_fixation_density_for_row(
    settings: FixationDensitySettings,
    row: dict,
    agent: str,
    *,
    kernel_sigma: Optional[float] = None,
    kernel_truncate: Optional[float] = None,
) -> Optional[FixationDensityVectorsData]:
    """Build fixation density vectors for one row using a shared global kernel."""
    cfg = load_config(settings.cfg_path)

    fix_path = build_processed_pickle_path(cfg, row, settings.fixations_modality, agent)
    vector_path = build_processed_pickle_path(
        cfg,
        row,
        settings.fixation_vectors_modality,
        agent,
    )

    if not fix_path.exists() or not vector_path.exists():
        return None

    fix_df = load_pickle_path(fix_path)
    vectors_obj = load_pickle_path(vector_path)

    if not isinstance(fix_df, pd.DataFrame):
        return None

    if isinstance(vectors_obj, FixationBinaryVectorsData):
        vectors = vectors_obj.vectors
        context = vectors_obj.context
    elif isinstance(vectors_obj, dict):
        vectors = vectors_obj
        context = None
    else:
        return None

    if kernel_sigma is None or kernel_truncate is None:
        _kernel_binwidth, shared_sigma, shared_truncate, _kernel_stats = (
            _compute_global_face_kernel_parameters(settings)
        )
    else:
        shared_sigma = float(kernel_sigma)
        shared_truncate = float(kernel_truncate)

    roi_groups = _resolve_roi_groups(settings, agent)
    density_vectors: Dict[str, np.ndarray] = {}
    for group in roi_groups:
        if group not in vectors:
            continue
        binary_vec = np.asarray(vectors[group]).astype(float)
        density = gaussian_filter1d(
            binary_vec,
            sigma=shared_sigma,
            mode="constant",
            cval=0.0,
            truncate=shared_truncate,
        )
        if settings.normalize:
            density = _minmax_normalize(density)
        density_vectors[group] = density.astype(np.float32)

    if context is None:
        context = RecordingContext(
            date=row["date"],
            session=row["session"],
            agent=agent,
            monkey_name=_extract_monkey_name(fix_df),
        )

    return FixationDensityVectorsData(context=context, vectors=density_vectors)


def process_fixation_density_for_row(
    settings: FixationDensitySettings,
    row: dict,
    agent: str,
    *,
    kernel_sigma: Optional[float] = None,
    kernel_truncate: Optional[float] = None,
) -> Optional[FixationDensityVectorsData]:
    """Build and persist fixation density vectors for one row/agent."""
    data = build_fixation_density_for_row(
        settings,
        row,
        agent,
        kernel_sigma=kernel_sigma,
        kernel_truncate=kernel_truncate,
    )
    if data is None:
        return None

    cfg = load_config(settings.cfg_path)
    save_processed_pickle(data, cfg, row, settings.output_modality, agent)
    return data


def _build_and_save_worker(args) -> int:
    """Worker wrapper that returns 1 if outputs were written."""
    settings, row, agent, kernel_sigma, kernel_truncate = args
    data = process_fixation_density_for_row(
        settings,
        row,
        agent,
        kernel_sigma=kernel_sigma,
        kernel_truncate=kernel_truncate,
    )
    return 1 if data is not None else 0


def build_tasks(
    settings: FixationDensitySettings,
    *,
    test_single: bool = False,
) -> list[tuple[FixationDensitySettings, dict, str]]:
    """Build (settings, row, agent) tasks from fixation binary vector files."""
    cfg = load_config(settings.cfg_path)
    index_df = index_processed_dataset(cfg, settings.fixation_vectors_modality)
    rows = index_df.to_dict(orient="records")

    agents_filter = None
    if settings.agents is not None:
        agents_filter = {str(agent) for agent in settings.agents}

    tasks: list[tuple[FixationDensitySettings, dict, str]] = []
    for row in rows:
        agent = row.get("agent")
        if agent is None:
            continue
        if agents_filter is not None and agent not in agents_filter:
            continue
        fix_path = build_processed_pickle_path(cfg, row, settings.fixations_modality, agent)
        if not fix_path.exists():
            continue
        tasks.append((settings, row, agent))

    if test_single and tasks:
        return [random.choice(tasks)]
    return tasks


def run_fixation_density_build(
    settings: FixationDensitySettings,
    *,
    use_parallel: bool = False,
    test_single: bool = False,
) -> None:
    """Run fixation density creation across all tasks with one shared face kernel."""
    tasks = build_tasks(settings, test_single=test_single)
    if not tasks:
        print("No fixation density tasks found.")
        return

    kernel_binwidth, kernel_sigma, kernel_truncate, kernel_stats = (
        _compute_global_face_kernel_parameters(settings)
    )
    print(
        "Shared face kernel: "
        f"m1_fix={kernel_stats['m1_avg_fixation_duration']:.3f}, "
        f"m1_gap={kernel_stats['m1_avg_inter_fixation_duration']:.3f}, "
        f"m2_fix={kernel_stats['m2_avg_fixation_duration']:.3f}, "
        f"m2_gap={kernel_stats['m2_avg_inter_fixation_duration']:.3f}, "
        f"binwidth={kernel_binwidth}, sigma={kernel_sigma:.3f}, "
        f"truncate={kernel_truncate:.3f}"
    )

    if test_single:
        settings, row, agent = tasks[0]
        print(f"Test single: date={row['date']} session={row['session']} agent={agent}")
        data = process_fixation_density_for_row(
            settings,
            row,
            agent,
            kernel_sigma=kernel_sigma,
            kernel_truncate=kernel_truncate,
        )
        if data is None or not data.vectors:
            print("No density vectors produced.")
            return
        for label, vec in data.vectors.items():
            arr = np.asarray(vec)
            if arr.size == 0:
                print(f"{label}: empty")
                continue
            print(
                f"{label}: len={arr.size} min={arr.min():.4f} "
                f"max={arr.max():.4f} mean={arr.mean():.4f}"
            )
        return

    worker_args = [
        (*task, kernel_sigma, kernel_truncate)
        for task in tasks
    ]
    run_tasks(
        _build_and_save_worker,
        worker_args,
        desc="Building fixation densities",
        unit="task",
        use_parallel=use_parallel,
        max_procs=32,
    )
