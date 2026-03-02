"""Compute joint face fixation density from per-agent densities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.data.records.behavioral import (
    FixationDensityVectorsData,
    JointFixationDensityData,
    RecordingContext,
)
from dal_monte_2022_analysis.behav.preprocessing.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_processed_pickle,
)
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks


@dataclass
class JointFixationDensitySettings:
    """Configuration for building joint face fixation density vectors."""
    cfg_path: str
    input_modality: str = "fixation_density_vectors"
    output_modality: str = "joint_face_fixation_density"
    face_label: str = "face"
    normalize: bool = True
    use_parallel: bool = False
    test_single: bool = False


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize an array between 0 and 1."""
    if values.size == 0:
        return values
    min_val = float(values.min())
    max_val = float(values.max())
    if np.isclose(max_val, min_val):
        return np.zeros_like(values, dtype=float)
    return (values - min_val) / (max_val - min_val)


def _extract_face_density(obj, face_label: str) -> Optional[np.ndarray]:
    """Return the face density vector from a fixation density object."""
    if isinstance(obj, FixationDensityVectorsData):
        vectors = obj.vectors
    elif isinstance(obj, dict):
        vectors = obj
    else:
        return None
    if not vectors or face_label not in vectors:
        return None
    return np.asarray(vectors[face_label]).astype(float)


def build_joint_face_density_for_row(
    settings: JointFixationDensitySettings,
    row: dict,
    *,
    m1_path,
    m2_path,
) -> Optional[JointFixationDensityData]:
    """Build joint face fixation density for a single date/session."""
    m1_obj = load_pickle_path(m1_path)
    m2_obj = load_pickle_path(m2_path)

    m1_face = _extract_face_density(m1_obj, settings.face_label)
    m2_face = _extract_face_density(m2_obj, settings.face_label)
    if m1_face is None or m2_face is None:
        return None
    if m1_face.shape[0] != m2_face.shape[0]:
        return None

    joint = np.sqrt(m1_face * m2_face)
    if settings.normalize:
        joint = _minmax_normalize(joint)

    context = RecordingContext(
        date=row["date"],
        session=row["session"],
        agent=None,
        monkey_name=None,
    )
    return JointFixationDensityData(context=context, density=joint.astype(np.float32))


def process_joint_face_density_for_row(
    settings: JointFixationDensitySettings,
    row: dict,
    *,
    m1_path,
    m2_path,
) -> Optional[JointFixationDensityData]:
    """Build and persist joint face density for one date/session."""
    data = build_joint_face_density_for_row(settings, row, m1_path=m1_path, m2_path=m2_path)
    if data is None:
        return None

    cfg = load_config(settings.cfg_path)
    save_processed_pickle(data, cfg, row, settings.output_modality, None)
    return data


def _build_and_save_worker(args) -> int:
    """Worker wrapper that returns 1 if outputs were written."""
    settings, row, m1_path, m2_path = args
    data = process_joint_face_density_for_row(
        settings,
        row,
        m1_path=m1_path,
        m2_path=m2_path,
    )
    return 1 if data is not None else 0


def build_tasks(
    settings: JointFixationDensitySettings,
    *,
    test_single: bool = False,
) -> list[tuple[JointFixationDensitySettings, dict, object, object]]:
    """Build tasks that include m1/m2 fixation density paths for each session."""
    cfg = load_config(settings.cfg_path)
    index_df = index_processed_dataset(cfg, settings.input_modality)
    rows = index_df.to_dict(orient="records")

    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        agent = row.get("agent")
        if agent not in {"m1", "m2"}:
            continue
        key = (row["date"], row["session"])
        grouped.setdefault(key, {})[agent] = row["path"]

    tasks: list[tuple[JointFixationDensitySettings, dict, object, object]] = []
    for (date, session), agent_paths in grouped.items():
        if "m1" not in agent_paths or "m2" not in agent_paths:
            continue
        row = {"date": date, "session": session}
        tasks.append((settings, row, agent_paths["m1"], agent_paths["m2"]))

    if test_single and tasks:
        return [random.choice(tasks)]
    return tasks


def run_joint_face_density_build(
    settings: JointFixationDensitySettings,
    *,
    use_parallel: bool = False,
    test_single: bool = False,
) -> None:
    """Run joint face fixation density creation across all tasks."""
    tasks = build_tasks(settings, test_single=test_single)
    if not tasks:
        print("No joint face density tasks found.")
        return

    if test_single:
        settings, row, m1_path, m2_path = tasks[0]
        print(f"Test single: date={row['date']} session={row['session']}")
        data = process_joint_face_density_for_row(
            settings,
            row,
            m1_path=m1_path,
            m2_path=m2_path,
        )
        if data is None or data.density.size == 0:
            print("No joint density vector produced.")
            return
        print(
            "joint_face: "
            f"len={data.density.size} min={data.density.min():.4f} "
            f"max={data.density.max():.4f} mean={data.density.mean():.4f}"
        )
        return

    run_tasks(
        _build_and_save_worker,
        tasks,
        desc="Building joint face densities",
        unit="task",
        use_parallel=use_parallel,
        max_procs=32,
    )
