"""Loaders for unit-level ephys tables and spike records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import (
    load_config,
    resolve_dataset_cfg_path,
    resolve_ephys_cfg_path,
)
from dal_monte_2022_analysis.data.records.ephys import EphysUnitContext, UnitSpikeData


DEFAULT_EPHYS_DATA_FILENAME = "ephys_unit_data.pkl"
DEFAULT_EPHYS_REQUIRED_COLUMNS = (
    "session_name",
    "region",
    "unit_uuid",
    "spike_channel",
    "spike_times",
)
DEFAULT_EPHYS_COLUMN_ALIASES = {
    "session_name": ("session_name", "session", "recording_day", "date"),
    "region": ("region", "brain_region"),
    "unit_uuid": ("unit_uuid", "unit_id", "unit"),
    "spike_channel": ("spike_channel", "channel", "chan"),
    "spike_times": ("spike_times", "spike_ts", "spike_timestamps"),
}
DEFAULT_RECORDED_MONKEY_CANDIDATES = ("recorded_monkey", "monkey_name", "monkey", "animal")
DEFAULT_RECORDED_AGENT_CANDIDATES = ("recorded_agent", "agent", "recording_agent")
DEFAULT_AREA_CANDIDATES = ("area", "brain_area", "region")
_SESSION_NAME_PATTERN = re.compile(r"^(?P<date>\d{8})(?:[-_](?P<suffix>.+))?$")


def _as_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _extract_ephys_date_from_session_name(session_name: object) -> tuple[str, Optional[str], str]:
    token = _as_optional_str(session_name)
    if token is None:
        raise ValueError("session_name is empty.")
    if token.isdigit() and len(token) == 7:
        token = token.zfill(8)
    matched = _SESSION_NAME_PATTERN.fullmatch(token)
    if matched is None:
        raise ValueError(f"session_name must look like MMDDYYYY or MMDDYYYY_<session>; got '{token}'.")
    return matched.group("date"), _as_optional_str(matched.group("suffix")), token


def _coerce_spike_times(value: object) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=float)
    try:
        if pd.isna(value):
            return np.empty(0, dtype=float)
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float).reshape(-1)
    elif isinstance(value, (list, tuple, pd.Series)):
        arr = np.asarray(value, dtype=float).reshape(-1)
    else:
        arr = np.asarray([value], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return arr.astype(float, copy=False)
    return np.sort(arr.astype(float, copy=False), kind="mergesort")


def _resolve_ephys_data_path(dataset_cfg: dict, ephys_cfg: dict) -> Path:
    raw_path = ephys_cfg.get("ephys_data_path")
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else dataset_cfg["processed_data_root"] / path
    filename = str(ephys_cfg.get("ephys_data_filename", DEFAULT_EPHYS_DATA_FILENAME)).strip()
    return dataset_cfg["processed_data_root"] / filename


def _normalize_ephys_aliases(ephys_cfg: dict) -> dict[str, tuple[str, ...]]:
    aliases_cfg = ephys_cfg.get("column_aliases", {})
    aliases: dict[str, tuple[str, ...]] = {}
    for canonical, defaults in DEFAULT_EPHYS_COLUMN_ALIASES.items():
        raw = aliases_cfg.get(canonical, defaults)
        candidates = [str(raw)] if isinstance(raw, (str, bytes)) else [str(x) for x in raw]
        if canonical not in candidates:
            candidates = [canonical, *candidates]
        aliases[canonical] = tuple(candidates)
    return aliases


def _apply_column_aliases(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    columns = set(df.columns)
    for canonical, candidates in aliases.items():
        if canonical in columns:
            continue
        for candidate in candidates:
            if candidate in columns:
                rename_map[candidate] = canonical
                columns.add(canonical)
                break
    return df if not rename_map else df.rename(columns=rename_map)


def _first_present_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def load_ephys_unit_dataframe(
    *,
    cfg_path: str = "configs/project.yaml",
    ephys_cfg_path: str = "configs/ephys_data.yaml",
    dates: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load and normalize unit-level ephys data."""
    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    resolved_ephys_cfg_path = resolve_ephys_cfg_path(
        ephys_cfg_path,
        project_cfg_path=cfg_path,
    )
    dataset_cfg = load_config(dataset_cfg_path)
    ephys_cfg = load_config(resolved_ephys_cfg_path)
    ephys_path = _resolve_ephys_data_path(dataset_cfg, ephys_cfg)
    if not ephys_path.exists():
        raise FileNotFoundError(f"Ephys data file not found: {ephys_path}")

    table = pd.read_pickle(ephys_path)
    df = table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    df = _apply_column_aliases(df, _normalize_ephys_aliases(ephys_cfg))

    required_cols = tuple(str(col) for col in ephys_cfg.get("required_columns", DEFAULT_EPHYS_REQUIRED_COLUMNS))
    missing = [name for name in required_cols if name not in df.columns]
    if missing:
        raise RuntimeError(f"Ephys table is missing required columns {missing}; available: {list(df.columns)}")

    parsed = [_extract_ephys_date_from_session_name(value) for value in df["session_name"]]
    out = df.copy()
    out["date"] = [date for date, _, _ in parsed]
    out["legacy_session_token"] = [suffix for _, suffix, _ in parsed]
    out["session_name"] = [source_token for _, _, source_token in parsed]
    out["session"] = None
    out["spike_channel"] = out["spike_channel"].map(_as_optional_str)
    out["spike_times"] = out["spike_times"].map(_coerce_spike_times)
    out["n_spikes"] = out["spike_times"].map(lambda arr: int(arr.size))

    if dates is not None:
        include_dates = {str(date) for date in dates}
        out = out[out["date"].astype(str).isin(include_dates)]
    if regions is not None:
        include_regions = {str(region).strip().lower() for region in regions}
        region_series = out["region"].map(lambda value: str(value).strip().lower())
        out = out[region_series.isin(include_regions)]

    return out.reset_index(drop=True)


def load_ephys_units(
    *,
    cfg_path: str = "configs/project.yaml",
    ephys_cfg_path: str = "configs/ephys_data.yaml",
    dates: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
) -> list[UnitSpikeData]:
    """Load unit-level ephys data as dataclass objects."""
    resolved_ephys_cfg_path = resolve_ephys_cfg_path(
        ephys_cfg_path,
        project_cfg_path=cfg_path,
    )
    ephys_cfg = load_config(resolved_ephys_cfg_path)
    df = load_ephys_unit_dataframe(
        cfg_path=cfg_path,
        ephys_cfg_path=str(resolved_ephys_cfg_path),
        dates=dates,
        regions=regions,
    )
    if df.empty:
        return []

    monkey_col = _first_present_column(
        df,
        tuple(str(col) for col in ephys_cfg.get("recorded_monkey_candidates", DEFAULT_RECORDED_MONKEY_CANDIDATES)),
    )
    agent_col = _first_present_column(
        df,
        tuple(str(col) for col in ephys_cfg.get("recorded_agent_candidates", DEFAULT_RECORDED_AGENT_CANDIDATES)),
    )
    area_col = _first_present_column(
        df,
        tuple(str(col) for col in ephys_cfg.get("area_candidates", DEFAULT_AREA_CANDIDATES)),
    )

    core_cols = {
        "session_name",
        "date",
        "session",
        "legacy_session_token",
        "region",
        "unit_uuid",
        "spike_channel",
        "spike_times",
        "n_spikes",
    }
    if monkey_col is not None:
        core_cols.add(monkey_col)
    if agent_col is not None:
        core_cols.add(agent_col)
    if area_col is not None:
        core_cols.add(area_col)

    units: list[UnitSpikeData] = []
    for row in df.to_dict(orient="records"):
        metadata = {key: value for key, value in row.items() if key not in core_cols}
        context = EphysUnitContext(
            date=str(row["date"]),
            session_name=str(row["session_name"]),
            unit_uuid=str(row["unit_uuid"]),
            legacy_session_token=_as_optional_str(row.get("legacy_session_token")),
            region=_as_optional_str(row.get("region")),
            spike_channel=_as_optional_str(row.get("spike_channel")),
            recorded_agent=(_as_optional_str(row.get(agent_col)) if agent_col else "m1") or "m1",
            recorded_monkey=_as_optional_str(row.get(monkey_col)) if monkey_col else None,
            area=(_as_optional_str(row.get(area_col)) if area_col else _as_optional_str(row.get("region"))),
            metadata=metadata,
        )
        units.append(UnitSpikeData(context=context, spike_ts=_coerce_spike_times(row.get("spike_times"))))
    return units


__all__ = [
    "load_ephys_unit_dataframe",
    "load_ephys_units",
]
