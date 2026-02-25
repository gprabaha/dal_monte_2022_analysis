"""Helpers for loading processed data into Python objects or DataFrames."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.data.gaze_data import (
    FixationBinaryVectorsData,
    FixationDensityVectorsData,
    JointFixationDensityData,
    NeuralTimelineData,
    PositionData,
    PupilSizeData,
    ROIRectsData,
)
from dal_monte_2022_analysis.utils.paths import (
    list_processed_modalities,
    scan_processed_data_paths,
)


# Keep this list in sync with the modalities that are written to disk.
ALLOWED_MODALITIES = {
    "fixation_binary_vectors",
    "fixation_density_vectors",
    "joint_face_fixation_density",
    "interactive_periods",
    "gaze_position",
    "neural_timeline",
    "pupil_size",
    "roi_vertices",
    "fixations",
    "saccades",
}


@dataclass(frozen=True)
class ProcessedItem:
    """Container for one processed data object and its metadata."""
    date: str
    session: str
    agent: Optional[str]
    modality: str
    path: Path
    data: object


def _validate_modality(modality: str, allowed: set[str]) -> str:
    """Validate modality names (accepts cleaned variants of allowed entries)."""
    modality = modality.strip()
    if modality in allowed:
        return modality
    if modality.endswith("_cleaned"):
        base = modality[:-8]
        if base in allowed:
            return modality
    raise ValueError(
        f"Unsupported modality '{modality}'. Allowed entries: {sorted(allowed)} "
        "plus optional *_cleaned variants of those entries."
    )


def _load_pickle(path: Path):
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _metadata_from_context(obj, fallback: dict) -> dict:
    """Extract metadata from a data object's context (fallback to row info)."""
    context = getattr(obj, "context", None)
    if context is None:
        return fallback
    meta = {
        "date": context.date,
        "session": context.session,
        "agent": context.agent,
        "monkey_name": context.monkey_name,
    }
    return {**fallback, **{k: v for k, v in meta.items() if v is not None}}


def _position_to_df(data: PositionData, row_meta: dict) -> pd.DataFrame:
    """Convert PositionData to a tidy DataFrame with sample indices."""
    meta = _metadata_from_context(data, row_meta)
    df = pd.DataFrame({
        "sample": np.arange(len(data.x)),
        "x": data.x,
        "y": data.y,
    })
    for key, value in meta.items():
        df[key] = value
    return df


def _pupil_to_df(data: PupilSizeData, row_meta: dict) -> pd.DataFrame:
    """Convert PupilSizeData to a tidy DataFrame with sample indices."""
    meta = _metadata_from_context(data, row_meta)
    df = pd.DataFrame({
        "sample": np.arange(len(data.d)),
        "d": data.d,
    })
    for key, value in meta.items():
        df[key] = value
    return df


def _timeline_to_df(data: NeuralTimelineData, row_meta: dict) -> pd.DataFrame:
    """Convert NeuralTimelineData to a tidy DataFrame with sample indices."""
    meta = _metadata_from_context(data, row_meta)
    df = pd.DataFrame({
        "sample": np.arange(len(data.t)),
        "t": data.t,
    })
    for key, value in meta.items():
        df[key] = value
    return df


def _roi_to_df(data: ROIRectsData, row_meta: dict) -> pd.DataFrame:
    """Convert ROIRectsData to a DataFrame with one row per ROI."""
    meta = _metadata_from_context(data, row_meta)
    rows = []
    for roi_name, rect in data.rois.items():
        rect = np.asarray(rect).astype(float).reshape(-1)
        if rect.size != 4:
            continue
        x1, y1, x2, y2 = rect
        rows.append({
            "roi_name": roi_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
    df = pd.DataFrame(rows)
    for key, value in meta.items():
        df[key] = value
    return df


def _fixation_vectors_to_df(
    data: FixationBinaryVectorsData,
    row_meta: dict,
) -> pd.DataFrame:
    """Convert FixationBinaryVectorsData to a tidy DataFrame."""
    meta = _metadata_from_context(data, row_meta)
    vectors = {
        name: np.asarray(vec).astype(int)
        for name, vec in (data.vectors or {}).items()
    }
    if not vectors:
        return pd.DataFrame()

    lengths = {arr.shape[0] for arr in vectors.values()}
    if len(lengths) != 1:
        raise ValueError(
            "Fixation binary vectors must share a common length; "
            f"found lengths: {sorted(lengths)}"
        )

    length = lengths.pop()
    df = pd.DataFrame({
        "sample": np.arange(length),
        **vectors,
    })
    for key, value in meta.items():
        df[key] = value
    return df


def _fixation_density_to_df(
    data: FixationDensityVectorsData,
    row_meta: dict,
) -> pd.DataFrame:
    """Convert FixationDensityVectorsData to a tidy DataFrame."""
    meta = _metadata_from_context(data, row_meta)
    vectors = {
        name: np.asarray(vec).astype(float)
        for name, vec in (data.vectors or {}).items()
    }
    if not vectors:
        return pd.DataFrame()

    lengths = {arr.shape[0] for arr in vectors.values()}
    if len(lengths) != 1:
        raise ValueError(
            "Fixation density vectors must share a common length; "
            f"found lengths: {sorted(lengths)}"
        )

    length = lengths.pop()
    df = pd.DataFrame({
        "sample": np.arange(length),
        **vectors,
    })
    for key, value in meta.items():
        df[key] = value
    return df


def _joint_fixation_density_to_df(
    data: JointFixationDensityData,
    row_meta: dict,
) -> pd.DataFrame:
    """Convert JointFixationDensityData to a tidy DataFrame."""
    meta = _metadata_from_context(data, row_meta)
    density = np.asarray(data.density).astype(float)
    df = pd.DataFrame({
        "sample": np.arange(density.shape[0]),
        "joint_face": density,
    })
    for key, value in meta.items():
        df[key] = value
    return df


def _frame_to_df(df: pd.DataFrame, row_meta: dict) -> pd.DataFrame:
    """Ensure metadata columns exist on a DataFrame output."""
    if df.empty:
        return df
    df = df.copy()
    for key, value in row_meta.items():
        if key not in df.columns:
            df[key] = value
    return df


def _object_to_df(obj, row_meta: dict) -> pd.DataFrame:
    """Convert a loaded object to a DataFrame for concatenation."""
    if isinstance(obj, pd.DataFrame):
        return _frame_to_df(obj, row_meta)
    if isinstance(obj, PositionData):
        return _position_to_df(obj, row_meta)
    if isinstance(obj, PupilSizeData):
        return _pupil_to_df(obj, row_meta)
    if isinstance(obj, NeuralTimelineData):
        return _timeline_to_df(obj, row_meta)
    if isinstance(obj, ROIRectsData):
        return _roi_to_df(obj, row_meta)
    if isinstance(obj, FixationBinaryVectorsData):
        return _fixation_vectors_to_df(obj, row_meta)
    if isinstance(obj, FixationDensityVectorsData):
        return _fixation_density_to_df(obj, row_meta)
    if isinstance(obj, JointFixationDensityData):
        return _joint_fixation_density_to_df(obj, row_meta)

    # Fallback: store raw object in a single column.
    return pd.DataFrame([{**row_meta, "data": obj}])


def _resolve_modalities(
    cfg: dict,
    modality: Optional[Sequence[str] | str],
) -> list[str]:
    """Resolve modality inputs into a sorted list of validated names."""
    allowed = set(ALLOWED_MODALITIES) | set(cfg.get("modalities", {}).keys())
    allowed |= set(list_processed_modalities(cfg))

    if modality is None:
        return sorted(allowed)
    if isinstance(modality, str):
        return [_validate_modality(modality, allowed)]
    return [_validate_modality(item, allowed) for item in modality]


def index_processed_data(
    modality: Optional[Sequence[str] | str] = None,
    *,
    cfg_path: str = "configs/dataset.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> pd.DataFrame:
    """Index processed files on disk without loading their contents.

    Args:
        modality: Modality name or list of names. If None, scan all modalities
            discovered on disk plus those in config/ALLOWED_MODALITIES.
        cfg_path: Path to dataset config YAML.
        dates: Optional list of date strings to include (MMDDYYYY).
        sessions: Optional list of session identifiers to include.
        agents: Optional list of agent IDs to include (e.g., "m1", "m2", or None
            for shared outputs). The string "shared" is treated as None.

    Returns:
        DataFrame with columns: date, session, agent, modality, path.
    """
    cfg = load_dataset_config(cfg_path)
    modalities = _resolve_modalities(cfg, modality)

    rows: list[dict] = []
    for mod in modalities:
        for row in scan_processed_data_paths(
            cfg,
            mod,
            dates=dates,
            sessions=sessions,
            agents=agents,
        ):
            row["modality"] = mod
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date", "session", "agent", "modality", "path"])

    df = pd.DataFrame(rows)
    return df[["date", "session", "agent", "modality", "path"]]


def load_processed_objects(
    modality: str,
    *,
    cfg_path: str = "configs/dataset.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> list[ProcessedItem]:
    """Load processed data objects for a single modality.

    Args:
        modality: Modality folder name (e.g., "gaze_position", "fixations").
        cfg_path: Path to dataset config YAML.
        dates: Optional list of date strings to include (MMDDYYYY).
        sessions: Optional list of session identifiers to include.
        agents: Optional list of agent IDs to include (e.g., "m1", "m2", or None
            for shared outputs). The string "shared" is treated as None.

    Returns:
        List of ProcessedItem objects with metadata and loaded data.
    """
    cfg = load_dataset_config(cfg_path)
    allowed = set(ALLOWED_MODALITIES) | set(cfg.get("modalities", {}).keys())
    allowed |= set(list_processed_modalities(cfg))
    modality = _validate_modality(modality, allowed)

    rows = scan_processed_data_paths(
        cfg,
        modality,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )

    items: list[ProcessedItem] = []
    for row in rows:
        data_obj = _load_pickle(row["path"])
        items.append(
            ProcessedItem(
                date=row["date"],
                session=row["session"],
                agent=row["agent"],
                modality=modality,
                path=row["path"],
                data=data_obj,
            )
        )
    return items


def load_processed_dataframe(
    modality: str,
    *,
    cfg_path: str = "configs/dataset.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> pd.DataFrame:
    """Load processed data for a modality and return a concatenated DataFrame.

    Args:
        modality: Modality folder name (e.g., "gaze_position", "fixations").
        cfg_path: Path to dataset config YAML.
        dates: Optional list of date strings to include (MMDDYYYY).
        sessions: Optional list of session identifiers to include.
        agents: Optional list of agent IDs to include (e.g., "m1", "m2", or None
            for shared outputs). The string "shared" is treated as None.

    Returns:
        Concatenated DataFrame containing all available rows for the modality.
    """
    items = load_processed_objects(
        modality,
        cfg_path=cfg_path,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )
    if not items:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for item in items:
        row_meta = {
            "date": item.date,
            "session": item.session,
            "agent": item.agent,
        }
        frames.append(_object_to_df(item.data, row_meta))

    return pd.concat(frames, ignore_index=True)


def group_items(
    items: Iterable[ProcessedItem],
    *,
    key: str = "date_session_agent",
) -> dict:
    """Group ProcessedItem objects by a simple metadata key.

    Args:
        items: Iterable of ProcessedItem entries.
        key: One of "date", "session", "agent", "date_session",
            or "date_session_agent".

    Returns:
        Dict mapping the chosen key to lists of ProcessedItem.
    """
    grouped: dict = {}
    for item in items:
        if key == "date":
            group_key = item.date
        elif key == "session":
            group_key = item.session
        elif key == "agent":
            group_key = item.agent
        elif key == "date_session":
            group_key = (item.date, item.session)
        elif key == "date_session_agent":
            group_key = (item.date, item.session, item.agent)
        else:
            raise ValueError(
                "Unsupported key. Expected one of: date, session, agent, "
                "date_session, date_session_agent."
            )
        grouped.setdefault(group_key, []).append(item)
    return grouped


def load_processed_modality(
    modality: str,
    *,
    cfg_path: str = "configs/dataset.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> pd.DataFrame:
    """Backward-compatible alias for load_processed_dataframe."""
    return load_processed_dataframe(
        modality,
        cfg_path=cfg_path,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )
