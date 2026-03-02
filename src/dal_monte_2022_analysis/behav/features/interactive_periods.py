"""Define interactive periods from joint face fixation density."""

from __future__ import annotations

import random
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.data.behavioral_data import JointFixationDensityData
from dal_monte_2022_analysis.behav.preprocessing.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.utils.io import load_pickle, save_pickle
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_processed_data_path


@dataclass
class InteractivePeriodsSettings:
    """Configuration for building interactive periods."""
    cfg_path: str
    input_modality: str = "joint_face_fixation_density"
    output_modality: str = "interactive_periods"
    threshold_factor: float = 0.34
    include_low: bool = True
    high_label: str = "interactive"
    low_label: str = "non_interactive"
    use_parallel: bool = False
    test_single: bool = False


_load_pickle = load_pickle
_save_pickle = save_pickle


def _as_density(obj) -> Optional[np.ndarray]:
    """Extract a 1D density array from supported inputs."""
    if isinstance(obj, JointFixationDensityData):
        return np.asarray(obj.density).astype(float)
    if isinstance(obj, np.ndarray):
        return np.asarray(obj).astype(float)
    if isinstance(obj, dict) and "density" in obj:
        return np.asarray(obj["density"]).astype(float)
    return None


def _find_contiguous_periods(mask: Iterable[bool]) -> list[tuple[int, int, bool]]:
    """Return start/stop indices for contiguous periods of a boolean mask."""
    periods: list[tuple[int, int, bool]] = []
    mask_list = list(mask)
    if not mask_list:
        return periods
    start = 0
    current = mask_list[0]
    for idx in range(1, len(mask_list)):
        if mask_list[idx] != current:
            periods.append((start, idx - 1, current))
            start = idx
            current = mask_list[idx]
    periods.append((start, len(mask_list) - 1, current))
    return periods


def build_interactive_periods_for_row(
    settings: InteractivePeriodsSettings,
    row: dict,
    *,
    density_path,
) -> Optional[pd.DataFrame]:
    """Build interactive periods for one date/session."""
    obj = _load_pickle(density_path)
    density = _as_density(obj)
    if density is None or density.size == 0:
        return None

    mean_density = float(np.mean(density))
    threshold = settings.threshold_factor * mean_density
    mask = density > threshold
    periods = _find_contiguous_periods(mask)

    rows = []
    for start, stop, is_high in periods:
        if not is_high and not settings.include_low:
            continue
        label = settings.high_label if is_high else settings.low_label
        rows.append({
            "start": start,
            "stop": stop,
            "state": label,
            "mean_density": mean_density,
            "threshold": threshold,
            "date": row["date"],
            "session": row["session"],
        })

    return pd.DataFrame(rows)


def process_interactive_periods_for_row(
    settings: InteractivePeriodsSettings,
    row: dict,
    *,
    density_path,
) -> Optional[pd.DataFrame]:
    """Build and persist interactive periods for one date/session."""
    df = build_interactive_periods_for_row(settings, row, density_path=density_path)
    if df is None:
        return None

    cfg = load_config(settings.cfg_path)
    out_path = build_processed_data_path(cfg, row, settings.output_modality, None)
    _save_pickle(df, out_path)
    return df


def _build_and_save_worker(args) -> int:
    """Worker wrapper that returns 1 if outputs were written."""
    settings, row, density_path = args
    df = process_interactive_periods_for_row(settings, row, density_path=density_path)
    return 1 if df is not None else 0


def build_tasks(
    settings: InteractivePeriodsSettings,
    *,
    test_single: bool = False,
) -> list[tuple[InteractivePeriodsSettings, dict, object]]:
    """Build tasks from joint face fixation density files."""
    cfg = load_config(settings.cfg_path)
    index_df = index_processed_dataset(cfg, settings.input_modality)
    rows = index_df.to_dict(orient="records")

    tasks: list[tuple[InteractivePeriodsSettings, dict, object]] = []
    for row in rows:
        if row.get("agent") is not None:
            continue
        tasks.append((settings, {"date": row["date"], "session": row["session"]}, row["path"]))

    if test_single and tasks:
        return [random.choice(tasks)]
    return tasks


def run_interactive_periods_build(
    settings: InteractivePeriodsSettings,
    *,
    use_parallel: bool = False,
    test_single: bool = False,
) -> None:
    """Run interactive period creation across all tasks."""
    tasks = build_tasks(settings, test_single=test_single)
    if not tasks:
        print("No interactive period tasks found.")
        return

    if test_single:
        settings, row, density_path = tasks[0]
        print(f"Test single: date={row['date']} session={row['session']}")
        df = process_interactive_periods_for_row(
            settings,
            row,
            density_path=density_path,
        )
        if df is None or df.empty:
            print("No interactive periods produced.")
            return
        print(f"Interactive periods df:")
        print(f"{df}")
        counts = df["state"].value_counts().to_dict()
        print(f"Segments: {counts}")
        return

    if not use_parallel:
        for task in tqdm(tasks, desc="Building interactive periods (serial)", unit="task"):
            _build_and_save_worker(task)
        return

    n_proc = get_n_processes()
    with Pool(processes=n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(_build_and_save_worker, tasks),
            total=len(tasks),
            desc=f"Building interactive periods ({n_proc} workers)",
            unit="task",
        ):
            pass
