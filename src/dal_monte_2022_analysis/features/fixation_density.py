"""Compute fixation density vectors from fixation binary vectors."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.data.gaze_data import (
    FixationBinaryVectorsData,
    FixationDensityVectorsData,
    RecordingContext,
)
from dal_monte_2022_analysis.preprocessing.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_processed_data_path


DEFAULT_ROI_GROUPS: Dict[str, Sequence[str]] = {
    "face": ("face", "mouth", "eyes_nf"),
    "object": ("right_nonsocial_object", "left_nonsocial_object"),
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


def _load_pickle(path):
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path):
    """Write an object to a pickle file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _normalize_roi_groups(groups: Dict[str, Sequence[str]]) -> Dict[str, list[str]]:
    """Normalize ROI group keywords to lowercase lists."""
    normalized: Dict[str, list[str]] = {}
    for group_name, labels in (groups or {}).items():
        if labels is None:
            continue
        if isinstance(labels, (str, bytes)):
            label_list = [labels]
        else:
            label_list = list(labels)
        normalized[str(group_name)] = [str(label).lower() for label in label_list]
    return normalized


def _resolve_roi_groups(
    settings: FixationDensitySettings,
    agent: str,
) -> Dict[str, list[str]]:
    """Resolve ROI groups for an agent (per-agent overrides take precedence)."""
    if settings.agent_roi_groups and agent in settings.agent_roi_groups:
        return _normalize_roi_groups(settings.agent_roi_groups[agent])
    return _normalize_roi_groups(settings.roi_groups)


def _coerce_location(loc) -> list[str]:
    """Normalize a location entry to a list of strings."""
    if loc is None:
        return []
    if isinstance(loc, (list, tuple, set, np.ndarray)):
        return [str(val) for val in loc if val is not None]
    try:
        if pd.isna(loc):
            return []
    except Exception:
        pass
    return [str(loc)]


def _locations_match(locations: Iterable[str], keywords: Sequence[str]) -> bool:
    """Check whether any location label matches any keyword (substring match)."""
    for loc in locations:
        loc_lower = str(loc).lower()
        for keyword in keywords:
            if keyword in loc_lower:
                return True
    return False


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
        locations = [loc.lower() for loc in _coerce_location(row.get("location"))]
        if _locations_match(locations, keywords):
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


def _compute_kernel_for_events(
    events: Sequence[tuple[int, int]],
    settings: FixationDensitySettings,
) -> tuple[float, float, dict]:
    """Compute Gaussian filter parameters based on fixation event stats."""
    avg_fix, avg_gap = _compute_fixation_stats(
        events,
        inter_fixation_fallback=settings.inter_fixation_fallback,
    )

    if settings.binwidth_method == "sum":
        binwidth = int(round(avg_fix + avg_gap))
    elif settings.binwidth_method == "mean":
        binwidth = int(round(np.mean([avg_fix, avg_gap])))
    else:
        raise ValueError(
            "Unsupported binwidth_method. Expected 'mean' or 'sum'."
        )
    binwidth = max(settings.min_kernel_width, binwidth)
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
    stats = {
        "avg_fixation_duration": avg_fix,
        "avg_inter_fixation_duration": avg_gap,
        "kernel_binwidth": binwidth,
        "kernel_sigma": sigma,
        "kernel_truncate": truncate,
    }
    return sigma, truncate, stats


def build_fixation_density_for_row(
    settings: FixationDensitySettings,
    row: dict,
    agent: str,
) -> Optional[FixationDensityVectorsData]:
    """Build fixation density vectors for a single date/session/agent."""
    cfg = load_dataset_config(settings.cfg_path)

    fix_path = build_processed_data_path(cfg, row, settings.fixations_modality, agent)
    vector_path = build_processed_data_path(cfg, row, settings.fixation_vectors_modality, agent)

    if not fix_path.exists() or not vector_path.exists():
        return None

    fix_df = _load_pickle(fix_path)
    vectors_obj = _load_pickle(vector_path)

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

    roi_groups = _resolve_roi_groups(settings, agent)
    density_vectors: Dict[str, np.ndarray] = {}
    for group, keywords in roi_groups.items():
        if group not in vectors:
            continue
        binary_vec = np.asarray(vectors[group]).astype(float)
        events = _extract_fixation_events(fix_df, keywords)
        sigma, truncate, _stats = _compute_kernel_for_events(events, settings)
        density = gaussian_filter1d(
            binary_vec,
            sigma=sigma,
            mode="constant",
            cval=0.0,
            truncate=truncate,
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
) -> Optional[FixationDensityVectorsData]:
    """Build and persist fixation density vectors for one row/agent."""
    data = build_fixation_density_for_row(settings, row, agent)
    if data is None:
        return None

    cfg = load_dataset_config(settings.cfg_path)
    out_path = build_processed_data_path(cfg, row, settings.output_modality, agent)
    _save_pickle(data, out_path)
    return data


def _build_and_save_worker(args) -> int:
    """Worker wrapper that returns 1 if outputs were written."""
    settings, row, agent = args
    data = process_fixation_density_for_row(settings, row, agent)
    return 1 if data is not None else 0


def build_tasks(
    settings: FixationDensitySettings,
    *,
    test_single: bool = False,
) -> list[tuple[FixationDensitySettings, dict, str]]:
    """Build (settings, row, agent) tasks from fixation binary vector files."""
    cfg = load_dataset_config(settings.cfg_path)
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
        fix_path = build_processed_data_path(cfg, row, settings.fixations_modality, agent)
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
    """Run fixation density creation across all tasks."""
    tasks = build_tasks(settings, test_single=test_single)
    if not tasks:
        print("No fixation density tasks found.")
        return

    if test_single:
        settings, row, agent = tasks[0]
        print(f"Test single: date={row['date']} session={row['session']} agent={agent}")
        data = process_fixation_density_for_row(settings, row, agent)
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

    if not use_parallel:
        for task in tqdm(tasks, desc="Building fixation densities (serial)", unit="task"):
            _build_and_save_worker(task)
        return

    n_proc = get_n_processes(max_procs=32)
    with Pool(processes=n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(_build_and_save_worker, tasks),
            total=len(tasks),
            desc=f"Building fixation densities ({n_proc} workers)",
            unit="task",
        ):
            pass
