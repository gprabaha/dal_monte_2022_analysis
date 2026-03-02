"""Shared annotation helpers for behavioral and ephys tables."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from dal_monte_2022_analysis.config.load import load_config, resolve_dataset_cfg_path


def load_pair_context_table(
    *,
    cfg_path: str = "configs/project.yaml",
) -> pd.DataFrame:
    """Load one row per date with m1/m2 monkey names and pair label."""
    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    dataset_cfg = load_config(dataset_cfg_path)
    pair_path = Path(dataset_cfg["raw_data_root"]) / "ephys_days_and_monkeys.pkl"
    if not pair_path.exists():
        raise FileNotFoundError(f"Missing ephys metadata file: {pair_path}")

    pair_source = pd.read_pickle(pair_path)
    required_cols = {"session_name", "m1", "m2"}
    missing_cols = required_cols.difference(pair_source.columns)
    if missing_cols:
        raise RuntimeError(
            "Ephys metadata missing required columns "
            f"{sorted(missing_cols)}; found: {list(pair_source.columns)}"
        )

    dates = pair_source["session_name"].astype(str).str.strip()
    dates = dates.apply(lambda val: val.zfill(8) if len(val) == 7 else val)
    pair_df = pd.DataFrame(
        {
            "date": dates,
            "m1_name": pair_source["m1"].astype(str).str.strip(),
            "m2_name": pair_source["m2"].astype(str).str.strip(),
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
    return pair_df.drop_duplicates(subset=["date"], keep="first")


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
