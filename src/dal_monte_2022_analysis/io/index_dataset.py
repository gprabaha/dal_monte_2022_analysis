"""Index raw .mat files by session metadata for downstream processing."""

import pdb
import re
from pathlib import Path
import pandas as pd


def _load_ephys_days(cfg: dict) -> pd.DataFrame:
    """Load ephys days and monkey names from raw_data_root."""
    ephys_path = Path(cfg["raw_data_root"]) / "ephys_days_and_monkeys.pkl"
    if not ephys_path.exists():
        raise RuntimeError(f"Missing ephys days file: {ephys_path}")

    ephys_df = pd.read_pickle(ephys_path)

    required = ["session_name", "m1", "m2"]
    missing = [col for col in required if col not in ephys_df.columns]
    if missing:
        raise RuntimeError(
            "Ephys days file missing required columns "
            f"(missing: {missing}, found: {list(ephys_df.columns)})"
        )

    session_str = ephys_df["session_name"].astype(str).str.strip()
    session_str = session_str.apply(lambda val: val.zfill(8) if len(val) == 7 else val)

    bad_vals = session_str[~session_str.str.fullmatch(r"\d{8}")].head(5).tolist()
    if bad_vals:
        raise RuntimeError(
            "Session names must be 7 or 8 digits; "
            f"examples: {bad_vals}"
        )

    ephys_df = ephys_df.assign(
        date=session_str,
        monkey_name_m1=ephys_df["m1"],
        monkey_name_m2=ephys_df["m2"],
    )
    
    return ephys_df[["date", "monkey_name_m1", "monkey_name_m2"]]


def index_dataset(cfg: dict, modality: str) -> pd.DataFrame:
    """Return a DataFrame of available files for a given modality."""
    modality_cfg = cfg["modalities"][modality]

    root = cfg["raw_data_root"] / modality_cfg["folder"]
    pattern = re.compile(modality_cfg["file_pattern"])

    rows = []

    for mat_file in root.glob("*.mat"):
        match = pattern.match(mat_file.name)
        if not match:
            continue

        # Extract date/session identifiers from the filename.
        date_str = str(match["date"]).strip()
        if len(date_str) == 7:
            date_str = date_str.zfill(8)

        rows.append({
            "date": date_str,
            "session": str(match["session"]).strip(),
            "path": mat_file,
        })

    if not rows:
        raise RuntimeError(f"No files found for modality '{modality}'")

    index_df = pd.DataFrame(rows)

    ephys_df = _load_ephys_days(cfg)
    
    index_df = index_df.merge(ephys_df, on="date", how="inner")

    if index_df.empty:
        raise RuntimeError(
            f"No files left for modality '{modality}' after filtering to ephys days."
        )

    sort_dates = pd.to_datetime(
        index_df["date"],
        format="%m%d%Y",
        errors="coerce",
    )
    if sort_dates.isna().any():
        bad_vals = index_df.loc[sort_dates.isna(), "date"].head(5).tolist()
        raise RuntimeError(
            "Could not parse MMDDYYYY dates for sorting; "
            f"examples: {bad_vals}"
        )

    index_df = index_df.assign(_sort_date=sort_dates)
    index_df = index_df.sort_values(["_sort_date", "session"]).drop(columns=["_sort_date"])
    return index_df


def index_processed_dataset(cfg: dict, modality: str) -> pd.DataFrame:
    """Return a DataFrame of available processed files for a given modality."""
    root = Path(cfg["processed_data_root"])
    pattern = root / "date=*" / "session=*" / modality / "*.pkl"

    rows = []
    for pkl_path in root.glob(str(pattern.relative_to(root))):
        parts = pkl_path.parts
        try:
            date_part = next(part for part in parts if part.startswith("date="))
            session_part = next(part for part in parts if part.startswith("session="))
        except StopIteration:
            continue

        date = date_part.split("=", 1)[1]
        session = session_part.split("=", 1)[1]

        agent = None
        stem = pkl_path.stem
        if stem.startswith("agent="):
            agent = stem.split("=", 1)[1]
        elif stem == "shared":
            agent = None

        rows.append({
            "date": date,
            "session": session,
            "agent": agent,
            "path": pkl_path,
        })

    if not rows:
        raise RuntimeError(f"No processed files found for modality '{modality}'")

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "session", "agent"])
    return df
