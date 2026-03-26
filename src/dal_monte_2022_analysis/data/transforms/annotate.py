"""Shared annotation helpers for behavioral and ephys tables."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from dal_monte_2022_analysis.config.load import load_config, resolve_dataset_cfg_path

DEFAULT_PAIR_CONTEXT_PATH = "ephys_days_and_monkeys.pkl"
DEFAULT_PAIR_CONTEXT_DATE_COLUMN = "session_name"
DEFAULT_PAIR_CONTEXT_AGENT_NAME_COLUMNS = {
    "m1": "m1",
    "m2": "m2",
}


def _normalize_pair_context_dates(series: pd.Series) -> pd.Series:
    dates = series.astype(str).str.strip()
    dates = dates.apply(lambda val: val.zfill(8) if len(val) == 7 and val.isdigit() else val)

    bad_vals = dates[~dates.str.fullmatch(r"\d{8}")].head(5).tolist()
    if bad_vals:
        raise RuntimeError(
            "Pair-context date column must contain 7 or 8 digit MMDDYYYY values; "
            f"examples: {bad_vals}"
        )
    return dates


def _resolve_pair_context_cfg(dataset_cfg: dict) -> tuple[Path, str, dict[str, str]]:
    pair_cfg = dataset_cfg.get("pair_context", {})
    raw_root = Path(dataset_cfg["raw_data_root"])

    raw_path = pair_cfg.get("path", DEFAULT_PAIR_CONTEXT_PATH)
    pair_path = Path(raw_path)
    if not pair_path.is_absolute():
        pair_path = raw_root / pair_path

    date_col = str(pair_cfg.get("date_column", DEFAULT_PAIR_CONTEXT_DATE_COLUMN))

    raw_agent_cols = pair_cfg.get("agent_name_columns", DEFAULT_PAIR_CONTEXT_AGENT_NAME_COLUMNS)
    if not isinstance(raw_agent_cols, dict):
        raise TypeError("dataset pair_context.agent_name_columns must be a mapping.")

    missing_agents = {"m1", "m2"}.difference(raw_agent_cols)
    if missing_agents:
        raise KeyError(
            "dataset pair_context.agent_name_columns must define both m1 and m2; "
            f"missing: {sorted(missing_agents)}"
        )

    agent_cols = {
        "m1": str(raw_agent_cols["m1"]),
        "m2": str(raw_agent_cols["m2"]),
    }
    return pair_path, date_col, agent_cols


def load_pair_context_table_from_cfg(dataset_cfg: dict) -> pd.DataFrame:
    """Load one row per date with m1/m2 monkey names and pair label."""
    pair_path, date_col, agent_cols = _resolve_pair_context_cfg(dataset_cfg)
    if not pair_path.exists():
        raise FileNotFoundError(f"Missing pair-context metadata file: {pair_path}")

    pair_source = pd.read_pickle(pair_path)
    if not isinstance(pair_source, pd.DataFrame):
        pair_source = pd.DataFrame(pair_source)

    required_cols = {date_col, agent_cols["m1"], agent_cols["m2"]}
    missing_cols = required_cols.difference(pair_source.columns)
    if missing_cols:
        raise RuntimeError(
            "Pair-context metadata missing required columns "
            f"{sorted(missing_cols)}; found: {list(pair_source.columns)}"
        )

    pair_df = pd.DataFrame(
        {
            "date": _normalize_pair_context_dates(pair_source[date_col]),
            "monkey_name_m1": pair_source[agent_cols["m1"]].astype(str).str.strip(),
            "monkey_name_m2": pair_source[agent_cols["m2"]].astype(str).str.strip(),
        }
    )
    pair_df["pair_label"] = (
        pair_df[["monkey_name_m1", "monkey_name_m2"]]
        .apply(
            lambda row: " + ".join(
                sorted(
                    [
                        row["monkey_name_m1"] if row["monkey_name_m1"] else "unknown",
                        row["monkey_name_m2"] if row["monkey_name_m2"] else "unknown",
                    ],
                    key=lambda item: item.casefold(),
                )
            ),
            axis=1,
        )
    )
    return pair_df.drop_duplicates(subset=["date"], keep="first")


def load_pair_context_table(
    *,
    cfg_path: str = "configs/project.yaml",
) -> pd.DataFrame:
    """Load one row per date with m1/m2 monkey names and pair label."""
    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    dataset_cfg = load_config(dataset_cfg_path)
    pair_df = load_pair_context_table_from_cfg(dataset_cfg)
    return pair_df.rename(
        columns={
            "monkey_name_m1": "m1_name",
            "monkey_name_m2": "m2_name",
        }
    )


def annotate_with_pair_context(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    output_date_col: Optional[str] = None,
    cfg_path: str = "configs/project.yaml",
) -> pd.DataFrame:
    """Attach m1/m2/pair metadata to any table with a date column."""
    if df.empty:
        return df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Input table must include '{date_col}'.")

    out_date_col = output_date_col or date_col
    src = df.copy()
    if out_date_col != date_col:
        src[out_date_col] = src[date_col]
    src[out_date_col] = src[out_date_col].astype(str).str.strip()

    pair_df = load_pair_context_table(cfg_path=cfg_path).rename(columns={"date": out_date_col})
    return src.merge(pair_df, on=out_date_col, how="left")


def annotate_ephys_dates_with_pair_context(
    ephys_df: pd.DataFrame,
    *,
    cfg_path: str = "configs/project.yaml",
) -> pd.DataFrame:
    """Attach monkey pair context to ephys tables."""
    date_col = "date" if "date" in ephys_df.columns else "day"
    merged = annotate_with_pair_context(ephys_df, date_col=date_col, cfg_path=cfg_path)
    if "day" in merged.columns and "date" in merged.columns:
        merged["day"] = merged["date"]
    return merged
