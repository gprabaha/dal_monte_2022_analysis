"""Build binary fixation vectors aligned to the neural timeline."""

from __future__ import annotations

import pdb
import pickle
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.data.gaze_data import FixationBinaryVectorsData, RecordingContext
from dal_monte_2022_analysis.io.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_processed_data_path


DEFAULT_ROI_GROUPS: Dict[str, Sequence[str]] = {
    "face": ("face", "mouth", "eyes_nf"),
    "object": ("right_nonsocial_object", "left_nonsocial_object"),
}


@dataclass
class FixationBinaryVectorSettings:
    """Configuration for building fixation binary vectors."""
    cfg_path: str
    fixations_modality: str = "fixations"
    timeline_modality: str = "neural_timeline"
    output_modality: str = "fixation_binary_vectors"
    roi_groups: Dict[str, Sequence[str]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_ROI_GROUPS.items()}
    )
    agent_roi_groups: Optional[Dict[str, Dict[str, Sequence[str]]]] = None
    use_parallel: bool = False
    test_single: bool = False
    agents: Optional[Sequence[str]] = None


def _load_pickle(path: Path):
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path):
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
    settings: FixationBinaryVectorSettings,
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


def _apply_interval(vector: np.ndarray, start: int, stop: int) -> None:
    """Set a vector interval (inclusive) to 1s."""
    vector[start:stop + 1] = 1


def _extract_monkey_name(fix_df: pd.DataFrame) -> Optional[str]:
    """Extract a monkey name from a fixation DataFrame if available."""
    if fix_df is None or fix_df.empty or "monkey_name" not in fix_df.columns:
        return None
    valid = fix_df["monkey_name"].dropna()
    if valid.empty:
        return None
    return str(valid.iloc[0])


def build_fixation_binary_vectors_for_row(
    settings: FixationBinaryVectorSettings,
    row: dict,
    agent: str,
) -> Optional[FixationBinaryVectorsData]:
    """Build fixation binary vectors for a single date/session/agent."""
    cfg = load_dataset_config(settings.cfg_path)

    fix_path = build_processed_data_path(cfg, row, settings.fixations_modality, agent)
    timeline_path = build_processed_data_path(cfg, row, settings.timeline_modality, None)

    if not fix_path.exists() or not timeline_path.exists():
        return None

    fix_df = _load_pickle(fix_path)
    timeline = _load_pickle(timeline_path)

    timeline_len = len(timeline.t)
    if timeline_len == 0:
        return None

    roi_groups = _resolve_roi_groups(settings, agent)
    vectors = {
        group: np.zeros(timeline_len, dtype=np.uint8)
        for group in roi_groups
    }

    if isinstance(fix_df, pd.DataFrame) and not fix_df.empty:
        for _, fix_row in fix_df.iterrows():
            try:
                start = int(fix_row.get("start"))
                stop = int(fix_row.get("stop"))
            except (TypeError, ValueError):
                continue

            if stop < 0 or start >= timeline_len:
                continue

            start = max(0, start)
            stop = min(timeline_len - 1, stop)
            if start > stop:
                continue

            locations = [loc.lower() for loc in _coerce_location(fix_row.get("location"))]
            for group, keywords in roi_groups.items():
                if _locations_match(locations, keywords):
                    _apply_interval(vectors[group], start, stop)

    monkey_name = _extract_monkey_name(fix_df) if isinstance(fix_df, pd.DataFrame) else None
    context = RecordingContext(
        date=row["date"],
        session=row["session"],
        agent=agent,
        monkey_name=monkey_name,
    )
    return FixationBinaryVectorsData(context=context, vectors=vectors)


def process_fixation_binary_vectors_for_row(
    settings: FixationBinaryVectorSettings,
    row: dict,
    agent: str,
) -> Optional[FixationBinaryVectorsData]:
    """Build and persist fixation binary vectors for one row/agent."""
    data = build_fixation_binary_vectors_for_row(settings, row, agent)
    if data is None:
        return None

    cfg = load_dataset_config(settings.cfg_path)
    out_path = build_processed_data_path(cfg, row, settings.output_modality, agent)
    _save_pickle(data, out_path)
    return data


def _build_and_save_worker(args) -> int:
    """Worker wrapper that returns 1 if outputs were written."""
    settings, row, agent = args
    data = process_fixation_binary_vectors_for_row(settings, row, agent)
    return 1 if data is not None else 0


def build_tasks(
    settings: FixationBinaryVectorSettings,
    *,
    test_single: bool = False,
) -> list[tuple[FixationBinaryVectorSettings, dict, str]]:
    """Build (settings, row, agent) tasks from fixation files."""
    cfg = load_dataset_config(settings.cfg_path)
    index_df = index_processed_dataset(cfg, settings.fixations_modality)
    rows = index_df.to_dict(orient="records")

    agents_filter = None
    if settings.agents is not None:
        agents_filter = {str(agent) for agent in settings.agents}

    tasks: list[tuple[FixationBinaryVectorSettings, dict, str]] = []
    for row in rows:
        agent = row.get("agent")
        if agent is None:
            continue
        if agents_filter is not None and agent not in agents_filter:
            continue
        tasks.append((settings, row, agent))

    if test_single and tasks:
        return [tasks[0]]
    return tasks


def run_fixation_binary_vector_build(
    settings: FixationBinaryVectorSettings,
    *,
    use_parallel: bool = False,
    test_single: bool = False,
) -> None:
    """Run fixation binary vector creation across all tasks."""
    tasks = build_tasks(settings, test_single=test_single)
    if not tasks:
        print("No fixation tasks found for binary vector creation.")
        return

    if not use_parallel:
        for task in tqdm(tasks, desc="Building fixation vectors (serial)", unit="task"):
            _build_and_save_worker(task)
        return

    n_proc = get_n_processes(max_procs=8)
    with Pool(processes=n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(_build_and_save_worker, tasks),
            total=len(tasks),
            desc=f"Building fixation vectors ({n_proc} workers)",
            unit="task",
        ):
            pass
