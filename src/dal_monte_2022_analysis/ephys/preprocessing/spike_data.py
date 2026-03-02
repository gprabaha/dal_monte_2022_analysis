"""Preprocessing helpers for unit-level ephys spike data files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

import pandas as pd

from dal_monte_2022_analysis.config.load import (
    load_config,
    resolve_dataset_cfg_path,
    resolve_ephys_cfg_path,
)


DEFAULT_EPHYS_DATA_FILENAME = "ephys_unit_data.pkl"


@dataclass(frozen=True)
class EphysDateColumnUpdateSummary:
    """Summary of a date-column update operation on an ephys pickle."""

    source_path: Path
    output_path: Path
    n_rows: int
    date_column_created: bool
    n_overwritten: int
    backup_path: Optional[Path]


def _resolve_ephys_table_path(
    *,
    cfg_path: str,
    ephys_cfg_path: str,
    input_path: Optional[str] = None,
) -> Path:
    if input_path:
        return Path(input_path).expanduser().resolve()

    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    resolved_ephys_cfg_path = resolve_ephys_cfg_path(
        ephys_cfg_path,
        project_cfg_path=cfg_path,
    )
    dataset_cfg = load_config(dataset_cfg_path)
    ephys_cfg = load_config(resolved_ephys_cfg_path)
    ephys_data_path = ephys_cfg.get("ephys_data_path")
    if ephys_data_path:
        return Path(ephys_data_path).expanduser().resolve()

    filename = str(ephys_cfg.get("ephys_data_filename", DEFAULT_EPHYS_DATA_FILENAME)).strip()
    return (dataset_cfg["processed_data_root"] / filename).resolve()


def _load_as_dataframe(path: Path) -> pd.DataFrame:
    table = pd.read_pickle(path)
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.DataFrame(table)


def _to_string_series(series: pd.Series) -> pd.Series:
    # Keep source values as-is semantically, but normalize to stripped strings for comparisons/writes.
    return series.fillna("").map(lambda v: str(v).strip())


def _build_backup_path(path: Path, suffix: str) -> Path:
    candidate = path.with_name(f"{path.name}{suffix}")
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = path.with_name(f"{path.name}{suffix}.{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def add_date_column_from_session_name(
    df: pd.DataFrame,
    *,
    session_col: str = "session_name",
    date_col: str = "date",
    overwrite_existing: bool = False,
) -> tuple[pd.DataFrame, bool, int]:
    """Return a copy of df with `date_col` set to values from `session_col`."""
    if session_col not in df.columns:
        raise ValueError(f"Missing required column '{session_col}'.")

    out = df.copy()
    session_values = _to_string_series(out[session_col])
    date_created = date_col not in out.columns
    overwritten = 0

    if date_created:
        out[date_col] = session_values
        return out, True, int(session_values.shape[0])

    existing_date = _to_string_series(out[date_col])
    mismatch_mask = existing_date != session_values
    mismatch_count = int(mismatch_mask.sum())
    if mismatch_count == 0:
        return out, False, 0
    if not overwrite_existing:
        raise ValueError(
            f"Column '{date_col}' already exists and {mismatch_count} rows differ from '{session_col}'. "
            "Pass overwrite_existing=True to replace it."
        )

    out[date_col] = session_values
    return out, False, mismatch_count


def add_date_column_to_ephys_pickle(
    *,
    cfg_path: str = "configs/project.yaml",
    ephys_cfg_path: str = "configs/ephys_data.yaml",
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    session_col: str = "session_name",
    date_col: str = "date",
    overwrite_existing: bool = False,
    create_backup: bool = True,
    backup_suffix: str = ".bak",
    dry_run: bool = False,
) -> EphysDateColumnUpdateSummary:
    """Add/update a `date` column in the ephys pickle from `session_name` values."""
    source_path = _resolve_ephys_table_path(
        cfg_path=cfg_path,
        ephys_cfg_path=ephys_cfg_path,
        input_path=input_path,
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Ephys table not found: {source_path}")

    target_path = (
        Path(output_path).expanduser().resolve() if output_path else source_path
    )
    df = _load_as_dataframe(source_path)
    updated, created, overwritten = add_date_column_from_session_name(
        df,
        session_col=session_col,
        date_col=date_col,
        overwrite_existing=overwrite_existing,
    )

    backup_path = None
    in_place = target_path == source_path
    if not dry_run:
        if in_place and create_backup:
            backup_path = _build_backup_path(source_path, backup_suffix)
            shutil.copy2(source_path, backup_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        updated.to_pickle(target_path)

    return EphysDateColumnUpdateSummary(
        source_path=source_path,
        output_path=target_path,
        n_rows=int(updated.shape[0]),
        date_column_created=created,
        n_overwritten=int(overwritten),
        backup_path=backup_path,
    )
