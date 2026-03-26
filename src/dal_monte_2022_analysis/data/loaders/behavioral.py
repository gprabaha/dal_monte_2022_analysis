"""Loaders for behavioral data objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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
from dal_monte_2022_analysis.data.transforms.annotate import (
    load_pair_context_table_from_cfg,
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


def _sort_behavioral_index(df: pd.DataFrame, *, modality: str) -> pd.DataFrame:
    if df.empty:
        return df

    sort_dates = pd.to_datetime(df["date"], format="%m%d%Y", errors="coerce")
    if sort_dates.isna().any():
        bad_vals = df.loc[sort_dates.isna(), "date"].head(5).tolist()
        raise RuntimeError(
            f"Could not parse MMDDYYYY dates while indexing '{modality}'; "
            f"examples: {bad_vals}"
        )

    out = df.copy()
    out["_sort_date"] = sort_dates
    if "agent" in out.columns:
        out["_agent_sort"] = out["agent"].fillna("")
        out = out.sort_values(["_sort_date", "session", "_agent_sort"]).drop(columns=["_sort_date", "_agent_sort"])
    else:
        out = out.sort_values(["_sort_date", "session"]).drop(columns=["_sort_date"])
    return out.reset_index(drop=True)


def index_behavioral_source_data_from_cfg(
    cfg: dict,
    modality: str,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Index raw behavioral source files for one configured modality."""
    modality_cfg = cfg.get("modalities", {}).get(modality)
    if modality_cfg is None:
        raise KeyError(f"Behavioral modality '{modality}' is not defined in dataset config.")

    root = Path(cfg["raw_data_root"]) / str(modality_cfg["folder"])
    pattern = re.compile(str(modality_cfg["file_pattern"]))

    rows: list[dict] = []
    for mat_file in root.glob("*.mat"):
        match = pattern.match(mat_file.name)
        if not match:
            continue

        date_str = str(match["date"]).strip()
        if len(date_str) == 7 and date_str.isdigit():
            date_str = date_str.zfill(8)

        rows.append(
            {
                "date": date_str,
                "session": str(match["session"]).strip(),
                "path": mat_file,
            }
        )

    if not rows:
        raise RuntimeError(f"No files found for modality '{modality}'")

    index_df = pd.DataFrame(rows)
    pair_df = load_pair_context_table_from_cfg(cfg)[["date", "monkey_name_m1", "monkey_name_m2"]]
    index_df = index_df.merge(pair_df, on="date", how="inner")

    if dates is not None:
        include_dates = {str(date) for date in dates}
        index_df = index_df[index_df["date"].astype(str).isin(include_dates)]
    if sessions is not None:
        include_sessions = {str(session) for session in sessions}
        index_df = index_df[index_df["session"].astype(str).isin(include_sessions)]

    if index_df.empty:
        raise RuntimeError(
            f"No files left for modality '{modality}' after applying pair-context or row filters."
        )

    return _sort_behavioral_index(index_df, modality=modality)


def index_behavioral_source_data(
    modality: str,
    *,
    cfg_path: str = "configs/project.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Index raw behavioral source files for one configured modality."""
    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    cfg = load_config(dataset_cfg_path)
    return index_behavioral_source_data_from_cfg(
        cfg,
        modality,
        dates=dates,
        sessions=sessions,
    )


def index_behavioral_processed_data_from_cfg(
    cfg: dict,
    modality: str,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
    raise_on_missing: bool = True,
) -> pd.DataFrame:
    """Index processed behavioral artifacts for one modality."""
    rows = scan_processed_paths(
        cfg,
        modality,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )
    if not rows:
        if raise_on_missing:
            raise RuntimeError(f"No processed files found for modality '{modality}'")
        return pd.DataFrame(columns=["date", "session", "agent", "path"])

    df = pd.DataFrame(rows)
    df = _sort_behavioral_index(df, modality=modality)
    return df[["date", "session", "agent", "path"]]


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

    frames: list[pd.DataFrame] = []
    for mod in modalities:
        mod_df = index_behavioral_processed_data_from_cfg(
            cfg,
            mod,
            dates=dates,
            sessions=sessions,
            agents=agents,
            raise_on_missing=False,
        )
        if mod_df.empty:
            continue
        frames.append(mod_df.assign(modality=mod))

    if not frames:
        return pd.DataFrame(columns=["date", "session", "agent", "modality", "path"])
    df = pd.concat(frames, ignore_index=True)
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
    index_df = index_behavioral_processed_data_from_cfg(
        cfg,
        mod,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )
    rows = index_df.to_dict(orient="records")
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
    "index_behavioral_processed_data_from_cfg",
    "index_behavioral_source_data",
    "index_behavioral_source_data_from_cfg",
    "load_behavioral_data_dataframe",
    "load_behavioral_data_objects",
]
