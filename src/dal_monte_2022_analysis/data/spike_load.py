"""Helpers for loading and normalizing per-unit neural spike data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_spike_data_config,
)
from dal_monte_2022_analysis.data.spike_data import (
    NeuralUnitContext,
    SpikeTrainData,
)


DEFAULT_SPIKE_DATA_FILENAME = "spike_data.pkl"
DEFAULT_REQUIRED_COLUMNS = (
    "session_name",
    "region",
    "unit_uuid",
    "channel",
    "spike_ts",
)
DEFAULT_COLUMN_ALIASES = {
    "session_name": ("session_name", "session", "recording_day", "date"),
    "region": ("region", "brain_region"),
    "unit_uuid": ("unit_uuid", "unit_id", "unit"),
    "channel": ("channel", "chan"),
    "spike_ts": ("spike_ts", "spike_times", "spike_timestamps"),
}
DEFAULT_RECORDED_MONKEY_CANDIDATES = (
    "recorded_monkey",
    "monkey_name",
    "monkey",
    "animal",
)
DEFAULT_RECORDED_AGENT_CANDIDATES = (
    "recorded_agent",
    "agent",
    "recording_agent",
)
DEFAULT_AREA_CANDIDATES = ("area", "brain_area", "region")
_SESSION_NAME_PATTERN = re.compile(r"^(?P<date>\d{8})(?:[-_](?P<session>.+))?$")


def _resolve_spike_data_path(dataset_cfg: dict, spike_cfg: dict) -> Path:
    """Resolve the spike data pickle path from config."""
    if "spike_data_path" in spike_cfg:
        spike_path = Path(spike_cfg["spike_data_path"])
        if not spike_path.is_absolute():
            return dataset_cfg["processed_data_root"] / spike_path
        return spike_path

    filename = str(
        spike_cfg.get("spike_data_filename", DEFAULT_SPIKE_DATA_FILENAME)
    ).strip()
    return dataset_cfg["processed_data_root"] / filename


def _normalize_aliases(spike_cfg: dict) -> dict[str, tuple[str, ...]]:
    """Resolve column aliases and ensure each canonical name is included."""
    aliases_cfg = spike_cfg.get("column_aliases", {})
    aliases: dict[str, tuple[str, ...]] = {}

    for canonical, default_candidates in DEFAULT_COLUMN_ALIASES.items():
        raw_candidates = aliases_cfg.get(canonical, default_candidates)
        if isinstance(raw_candidates, (str, bytes)):
            candidates = [str(raw_candidates)]
        else:
            candidates = [str(item) for item in raw_candidates]

        if canonical not in candidates:
            candidates = [canonical, *candidates]
        aliases[canonical] = tuple(candidates)

    return aliases


def _apply_column_aliases(
    df: pd.DataFrame,
    aliases: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Rename known aliases to canonical column names."""
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

    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def _as_optional_str(value: object) -> Optional[str]:
    """Coerce values to a stripped string, returning None for null-like values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return None
    return text


def _parse_session_name(session_name: object) -> tuple[str, Optional[str]]:
    """Parse session_name tokens into a date and optional session chunk."""
    token = _as_optional_str(session_name)
    if token is None:
        raise ValueError("session_name is empty.")

    if token.isdigit() and len(token) == 7:
        token = token.zfill(8)

    matched = _SESSION_NAME_PATTERN.fullmatch(token)
    if matched is None:
        raise ValueError(
            "session_name must look like MMDDYYYY or MMDDYYYY_<session>; "
            f"got '{token}'."
        )

    date = matched.group("date")
    session = _as_optional_str(matched.group("session"))
    return date, session


def _coerce_spike_times(value: object) -> np.ndarray:
    """Coerce spike timestamps to a sorted 1D float array."""
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


def _first_present_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    """Return the first candidate column that exists in a DataFrame."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def annotate_spike_days_with_pair_context(
    spike_df: pd.DataFrame,
    *,
    cfg_path: str = "configs/dataset.yaml",
) -> pd.DataFrame:
    """Annotate spike rows with m1/m2 pair metadata at the recording-day level.

    The spike table is unit/day level and often contains only one recorded agent
    (typically m1). This helper injects day-level social context (m1/m2 names)
    from `ephys_days_and_monkeys.pkl`.
    """
    if spike_df.empty:
        return spike_df.copy()
    if "day" not in spike_df.columns:
        raise ValueError("spike_df must include a 'day' column.")

    dataset_cfg = load_dataset_config(cfg_path)
    ephys_path = Path(dataset_cfg["raw_data_root"]) / "ephys_days_and_monkeys.pkl"
    if not ephys_path.exists():
        raise FileNotFoundError(f"Missing ephys metadata file: {ephys_path}")

    ephys_df = pd.read_pickle(ephys_path)
    required_cols = {"session_name", "m1", "m2"}
    missing_cols = required_cols.difference(ephys_df.columns)
    if missing_cols:
        raise RuntimeError(
            "Ephys metadata missing required columns "
            f"{sorted(missing_cols)}; found: {list(ephys_df.columns)}"
        )

    day_series = ephys_df["session_name"].astype(str).str.strip()
    day_series = day_series.apply(lambda val: val.zfill(8) if len(val) == 7 else val)

    pair_df = pd.DataFrame(
        {
            "day": day_series,
            "m1_name": ephys_df["m1"].astype(str).str.strip(),
            "m2_name": ephys_df["m2"].astype(str).str.strip(),
        }
    )
    pair_df["pair_label"] = (
        pair_df[["m1_name", "m2_name"]]
        .apply(
            lambda row: " + ".join(
                sorted(
                    [
                        row["m1_name"] if row["m1_name"] else "unknown",
                        row["m2_name"] if row["m2_name"] else "unknown",
                    ],
                    key=lambda item: item.casefold(),
                )
            ),
            axis=1,
        )
    )
    pair_df = pair_df.drop_duplicates(subset=["day"], keep="first")

    merged = spike_df.merge(pair_df, on="day", how="left")
    return merged


def load_spike_dataframe(
    *,
    cfg_path: str = "configs/dataset.yaml",
    spike_cfg_path: str = "configs/spike_data.yaml",
    days: Optional[Sequence[str]] = None,
    dates: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load and normalize a per-unit spike table.

    Args:
        cfg_path: Dataset config path.
        spike_cfg_path: Spike-loader config path.
        days: Optional recording-day filter (MMDDYYYY).
        dates: Backward-compatible alias for `days`.
        regions: Optional region filter.

    Returns:
        DataFrame with canonical columns and normalized fields:
        session_name, day, run, date, session, region, unit_uuid, channel, spike_ts, n_spikes.
        (`date`/`session` are compatibility aliases of `day`/`run`.)
    """
    dataset_cfg = load_dataset_config(cfg_path)
    spike_cfg = load_spike_data_config(spike_cfg_path)

    spike_path = _resolve_spike_data_path(dataset_cfg, spike_cfg)
    if not spike_path.exists():
        raise FileNotFoundError(f"Spike data file not found: {spike_path}")

    table = pd.read_pickle(spike_path)
    if isinstance(table, pd.DataFrame):
        df = table.copy()
    else:
        df = pd.DataFrame(table)

    aliases = _normalize_aliases(spike_cfg)
    df = _apply_column_aliases(df, aliases)

    required_cols = tuple(
        str(col) for col in spike_cfg.get("required_columns", DEFAULT_REQUIRED_COLUMNS)
    )
    missing = [name for name in required_cols if name not in df.columns]
    if missing:
        raise RuntimeError(
            "Spike table is missing required columns "
            f"{missing}; available columns: {list(df.columns)}"
        )

    parsed: list[tuple[str, Optional[str]]] = []
    bad_examples: list[str] = []
    for value in df["session_name"]:
        try:
            parsed.append(_parse_session_name(value))
        except ValueError:
            if len(bad_examples) < 5:
                bad_examples.append(str(value))
            parsed.append(("00000000", None))

    if bad_examples:
        raise RuntimeError(
            "Could not parse session_name values in spike data; "
            f"examples: {bad_examples}"
        )

    out = df.copy()
    out["day"] = [date for date, _ in parsed]
    out["run"] = [session for _, session in parsed]
    # Backward compatibility for existing behavior-oriented code.
    out["date"] = out["day"]
    out["session"] = out["run"]
    out["spike_ts"] = out["spike_ts"].map(_coerce_spike_times)
    out["n_spikes"] = out["spike_ts"].map(lambda arr: int(arr.size))

    include_days_raw = days if days is not None else dates
    if include_days_raw is not None:
        include_days = {str(day) for day in include_days_raw}
        out = out[out["day"].astype(str).isin(include_days)]

    if regions is not None:
        include_regions = {str(region).strip().lower() for region in regions}
        region_series = out["region"].map(lambda value: str(value).strip().lower())
        out = out[region_series.isin(include_regions)]

    return out.reset_index(drop=True)


def load_spike_units(
    *,
    cfg_path: str = "configs/dataset.yaml",
    spike_cfg_path: str = "configs/spike_data.yaml",
    days: Optional[Sequence[str]] = None,
    dates: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
) -> list[SpikeTrainData]:
    """Load spike units as dataclass objects.

    Args:
        cfg_path: Dataset config path.
        spike_cfg_path: Spike-loader config path.
        days: Optional recording-day filter (MMDDYYYY).
        dates: Backward-compatible alias for `days`.
        regions: Optional region filter.
    """
    spike_cfg = load_spike_data_config(spike_cfg_path)
    df = load_spike_dataframe(
        cfg_path=cfg_path,
        spike_cfg_path=spike_cfg_path,
        days=days,
        dates=dates,
        regions=regions,
    )
    if df.empty:
        return []

    monkey_candidates = tuple(
        str(col)
        for col in spike_cfg.get(
            "recorded_monkey_candidates",
            DEFAULT_RECORDED_MONKEY_CANDIDATES,
        )
    )
    recorded_agent_candidates = tuple(
        str(col)
        for col in spike_cfg.get(
            "recorded_agent_candidates",
            DEFAULT_RECORDED_AGENT_CANDIDATES,
        )
    )
    area_candidates = tuple(
        str(col)
        for col in spike_cfg.get(
            "area_candidates",
            DEFAULT_AREA_CANDIDATES,
        )
    )

    monkey_col = _first_present_column(df, monkey_candidates)
    recorded_agent_col = _first_present_column(df, recorded_agent_candidates)
    area_col = _first_present_column(df, area_candidates)

    core_cols = {
        "session_name",
        "day",
        "run",
        "date",
        "session",
        "region",
        "unit_uuid",
        "channel",
        "spike_ts",
        "n_spikes",
    }
    if monkey_col is not None:
        core_cols.add(monkey_col)
    if recorded_agent_col is not None:
        core_cols.add(recorded_agent_col)
    if area_col is not None:
        core_cols.add(area_col)

    units: list[SpikeTrainData] = []
    for row in df.to_dict(orient="records"):
        metadata = {key: value for key, value in row.items() if key not in core_cols}
        context = NeuralUnitContext(
            date=str(row["day"]),
            session_name=str(row["session_name"]),
            unit_uuid=str(row["unit_uuid"]),
            session=_as_optional_str(row.get("run")),
            recorded_agent=(
                _as_optional_str(row.get(recorded_agent_col))
                if recorded_agent_col is not None
                else "m1"
            ) or "m1",
            recorded_monkey=(
                _as_optional_str(row.get(monkey_col))
                if monkey_col is not None
                else None
            ),
            region=_as_optional_str(row.get("region")),
            area=(
                _as_optional_str(row.get(area_col))
                if area_col is not None
                else _as_optional_str(row.get("region"))
            ),
            channel=_as_optional_str(row.get("channel")),
            metadata=metadata,
        )
        units.append(
            SpikeTrainData(
                context=context,
                spike_ts=_coerce_spike_times(row.get("spike_ts")),
            )
        )

    return units
