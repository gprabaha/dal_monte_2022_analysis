"""Loaders for derived feature products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config, resolve_dataset_cfg_path
from dal_monte_2022_analysis.data.records.behavioral import (
    FixationBinaryVectorsData,
    FixationDensityVectorsData,
    JointFixationDensityData,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    scan_processed_paths,
)
from dal_monte_2022_analysis.utils.paths import (
    list_processed_modalities,
)


FEATURE_MODALITIES = {
    "fixations",
    "saccades",
    "fixation_binary_vectors",
    "fixation_density_vectors",
    "joint_face_fixation_density",
    "interactive_periods",
}


@dataclass(frozen=True)
class FeatureItem:
    """Container for one feature object and its metadata."""

    date: str
    session: str
    agent: Optional[str]
    modality: str
    path: Path
    data: object


def _validate_feature_modality(modality: str, allowed: set[str]) -> str:
    token = modality.strip()
    if token in allowed:
        return token
    if token.endswith("_cleaned") and token[:-8] in FEATURE_MODALITIES:
        return token
    raise ValueError(
        f"Unsupported feature modality '{modality}'. Allowed entries: {sorted(allowed)} "
        "plus *_cleaned variants for feature modalities."
    )


def _resolve_feature_modalities(cfg: dict, modality: Optional[Sequence[str] | str]) -> list[str]:
    discovered = set(list_processed_modalities(cfg))
    discovered_features = {
        name
        for name in discovered
        if name in FEATURE_MODALITIES or (name.endswith("_cleaned") and name[:-8] in FEATURE_MODALITIES)
    }
    allowed = set(FEATURE_MODALITIES) | discovered_features

    if modality is None:
        return sorted(discovered_features)
    if isinstance(modality, str):
        return [_validate_feature_modality(modality, allowed)]
    return [_validate_feature_modality(item, allowed) for item in modality]


def index_feature_data(
    modality: Optional[Sequence[str] | str] = None,
    *,
    cfg_path: str = "configs/project.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> pd.DataFrame:
    """Index feature product files without loading contents."""
    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    cfg = load_config(dataset_cfg_path)
    modalities = _resolve_feature_modalities(cfg, modality)

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


def _feature_metadata_from_context(obj, fallback: dict) -> dict:
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


def _fixation_vectors_to_df(data: FixationBinaryVectorsData, row_meta: dict) -> pd.DataFrame:
    meta = _feature_metadata_from_context(data, row_meta)
    vectors = {name: np.asarray(vec).astype(int) for name, vec in (data.vectors or {}).items()}
    if not vectors:
        return pd.DataFrame()
    length_set = {arr.shape[0] for arr in vectors.values()}
    if len(length_set) != 1:
        raise ValueError(f"Fixation vectors must share a common length; found {sorted(length_set)}")
    length = next(iter(length_set))
    df = pd.DataFrame({"sample": np.arange(length), **vectors})
    for key, value in meta.items():
        df[key] = value
    return df


def _fixation_density_to_df(data: FixationDensityVectorsData, row_meta: dict) -> pd.DataFrame:
    meta = _feature_metadata_from_context(data, row_meta)
    vectors = {name: np.asarray(vec).astype(float) for name, vec in (data.vectors or {}).items()}
    if not vectors:
        return pd.DataFrame()
    length_set = {arr.shape[0] for arr in vectors.values()}
    if len(length_set) != 1:
        raise ValueError(f"Fixation density vectors must share a common length; found {sorted(length_set)}")
    length = next(iter(length_set))
    df = pd.DataFrame({"sample": np.arange(length), **vectors})
    for key, value in meta.items():
        df[key] = value
    return df


def _joint_fixation_density_to_df(data: JointFixationDensityData, row_meta: dict) -> pd.DataFrame:
    meta = _feature_metadata_from_context(data, row_meta)
    density = np.asarray(data.density, dtype=float)
    df = pd.DataFrame({"sample": np.arange(density.shape[0]), "joint_face": density})
    for key, value in meta.items():
        df[key] = value
    return df


def _frame_to_df(df: pd.DataFrame, row_meta: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for key, value in row_meta.items():
        if key not in out.columns:
            out[key] = value
    return out


def _feature_object_to_df(obj, row_meta: dict) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return _frame_to_df(obj, row_meta)
    if isinstance(obj, FixationBinaryVectorsData):
        return _fixation_vectors_to_df(obj, row_meta)
    if isinstance(obj, FixationDensityVectorsData):
        return _fixation_density_to_df(obj, row_meta)
    if isinstance(obj, JointFixationDensityData):
        return _joint_fixation_density_to_df(obj, row_meta)
    return pd.DataFrame([{**row_meta, "data": obj}])


def load_feature_objects(
    modality: str,
    *,
    cfg_path: str = "configs/project.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> list[FeatureItem]:
    """Load feature objects for one feature modality."""
    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    cfg = load_config(dataset_cfg_path)
    mod = _resolve_feature_modalities(cfg, modality)[0]
    rows = scan_processed_paths(cfg, mod, dates=dates, sessions=sessions, agents=agents)
    return [
        FeatureItem(
            date=row["date"],
            session=row["session"],
            agent=row["agent"],
            modality=mod,
            path=row["path"],
            data=load_pickle_path(row["path"]),
        )
        for row in rows
    ]


def load_feature_dataframe(
    modality: str,
    *,
    cfg_path: str = "configs/project.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> pd.DataFrame:
    """Load one feature modality as a concatenated DataFrame."""
    items = load_feature_objects(
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
        frames.append(_feature_object_to_df(item.data, row_meta))
    return pd.concat(frames, ignore_index=True)


def group_feature_items(items: Iterable[FeatureItem], *, key: str = "date_session_agent") -> dict:
    """Group FeatureItem objects by a metadata key."""
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


def load_feature_modality(
    modality: str,
    *,
    cfg_path: str = "configs/project.yaml",
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> pd.DataFrame:
    """Alias for load_feature_dataframe."""
    return load_feature_dataframe(
        modality,
        cfg_path=cfg_path,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )
