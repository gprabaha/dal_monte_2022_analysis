"""Loaders for behavioral data objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import (
    load_config,
    resolve_dataset_cfg_path,
)
from dal_monte_2022_analysis.data.records.behavioral import (
    NeuralTimelineData,
    PositionData,
    PupilSizeData,
    ROIRectsData,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    scan_processed_paths,
)
from dal_monte_2022_analysis.utils.paths import (
    list_processed_modalities,
)


BEHAVIORAL_DATA_MODALITIES = {
    "gaze_position",
    "neural_timeline",
    "pupil_size",
    "roi_vertices",
    "smoothed_pupil_size",
}


@dataclass(frozen=True)
class BehavioralDataItem:
    """Container for one behavioral data object and its metadata."""

    date: str
    session: str
    agent: Optional[str]
    modality: str
    path: Path
    data: object


def _validate_behavioral_modality(modality: str, allowed: set[str]) -> str:
    token = modality.strip()
    if token in allowed:
        return token
    if token.endswith("_cleaned") and token[:-8] in BEHAVIORAL_DATA_MODALITIES:
        return token
    raise ValueError(
        f"Unsupported behavioral modality '{modality}'. Allowed entries: {sorted(allowed)} "
        "plus *_cleaned variants for behavioral data modalities."
    )


def _resolve_behavioral_modalities(cfg: dict, modality: Optional[Sequence[str] | str]) -> list[str]:
    cfg_modalities = {str(name) for name in cfg.get("modalities", {}).keys()}
    discovered = set(list_processed_modalities(cfg))
    discovered_behavioral = {
        name
        for name in discovered
        if name in BEHAVIORAL_DATA_MODALITIES
        or (name.endswith("_cleaned") and name[:-8] in BEHAVIORAL_DATA_MODALITIES)
    }
    allowed = set(BEHAVIORAL_DATA_MODALITIES) | cfg_modalities | discovered_behavioral

    if modality is None:
        return sorted(discovered_behavioral | (cfg_modalities & BEHAVIORAL_DATA_MODALITIES))
    if isinstance(modality, str):
        return [_validate_behavioral_modality(modality, allowed)]
    return [_validate_behavioral_modality(item, allowed) for item in modality]


def index_behavioral_data(
    modality: Optional[Sequence[str] | str] = None,
    *,
    cfg_path: str = "configs/project.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> pd.DataFrame:
    """Index behavioral data files on disk without loading file contents."""
    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    cfg = load_config(dataset_cfg_path)
    modalities = _resolve_behavioral_modalities(cfg, modality)

    rows: list[dict] = []
    for mod in modalities:
        for row in scan_processed_paths(
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


def _behavioral_metadata_from_context(obj, fallback: dict) -> dict:
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
    meta = _behavioral_metadata_from_context(data, row_meta)
    df = pd.DataFrame({"sample": np.arange(len(data.x)), "x": data.x, "y": data.y})
    for key, value in meta.items():
        df[key] = value
    return df


def _pupil_to_df(data: PupilSizeData, row_meta: dict) -> pd.DataFrame:
    meta = _behavioral_metadata_from_context(data, row_meta)
    df = pd.DataFrame({"sample": np.arange(len(data.d)), "d": data.d})
    for key, value in meta.items():
        df[key] = value
    return df


def _timeline_to_df(data: NeuralTimelineData, row_meta: dict) -> pd.DataFrame:
    meta = _behavioral_metadata_from_context(data, row_meta)
    df = pd.DataFrame({"sample": np.arange(len(data.t)), "t": data.t})
    for key, value in meta.items():
        df[key] = value
    return df


def _roi_to_df(data: ROIRectsData, row_meta: dict) -> pd.DataFrame:
    meta = _behavioral_metadata_from_context(data, row_meta)
    rows = []
    for roi_name, rect in data.rois.items():
        arr = np.asarray(rect, dtype=float).reshape(-1)
        if arr.size != 4:
            continue
        rows.append(
            {
                "roi_name": roi_name,
                "x1": arr[0],
                "y1": arr[1],
                "x2": arr[2],
                "y2": arr[3],
            }
        )
    df = pd.DataFrame(rows)
    for key, value in meta.items():
        df[key] = value
    return df


def _behavioral_frame_to_df(df: pd.DataFrame, row_meta: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for key, value in row_meta.items():
        if key not in out.columns:
            out[key] = value
    return out


def _behavioral_object_to_df(obj, row_meta: dict) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return _behavioral_frame_to_df(obj, row_meta)
    if isinstance(obj, PositionData):
        return _position_to_df(obj, row_meta)
    if isinstance(obj, PupilSizeData):
        return _pupil_to_df(obj, row_meta)
    if isinstance(obj, NeuralTimelineData):
        return _timeline_to_df(obj, row_meta)
    if isinstance(obj, ROIRectsData):
        return _roi_to_df(obj, row_meta)
    return pd.DataFrame([{**row_meta, "data": obj}])


def load_behavioral_data_objects(
    modality: str,
    *,
    cfg_path: str = "configs/project.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> list[BehavioralDataItem]:
    """Load behavioral data objects for one data modality."""
    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    cfg = load_config(dataset_cfg_path)
    mod = _resolve_behavioral_modalities(cfg, modality)[0]
    rows = scan_processed_paths(cfg, mod, dates=dates, sessions=sessions, agents=agents)
    return [
        BehavioralDataItem(
            date=row["date"],
            session=row["session"],
            agent=row["agent"],
            modality=mod,
            path=row["path"],
            data=load_pickle_path(row["path"]),
        )
        for row in rows
    ]


def load_behavioral_data_dataframe(
    modality: str,
    *,
    cfg_path: str = "configs/project.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> pd.DataFrame:
    """Load one behavioral data modality as a concatenated DataFrame."""
    items = load_behavioral_data_objects(
        modality,
        cfg_path=cfg_path,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )
    if not items:
        return pd.DataFrame()
    frames = []
    for item in items:
        row_meta = {"date": item.date, "session": item.session, "agent": item.agent}
        frames.append(_behavioral_object_to_df(item.data, row_meta))
    return pd.concat(frames, ignore_index=True)


def group_behavioral_items(
    items: Iterable[BehavioralDataItem],
    *,
    key: str = "date_session_agent",
) -> dict:
    """Group BehavioralDataItem objects by a metadata key."""
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


__all__ = [
    "BehavioralDataItem",
    "group_behavioral_items",
    "index_behavioral_data",
    "load_behavioral_data_dataframe",
    "load_behavioral_data_objects",
]
