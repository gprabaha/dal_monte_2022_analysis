"""Indexing helpers for analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def build_analysis_output_dir(cfg: dict, subdir: str) -> Path:
    """Build the canonical analysis-output directory for one analysis subfolder."""
    root_value = cfg["analysis_output_root"] if "analysis_output_root" in cfg else cfg["processed_data_root"]
    root = Path(root_value)
    return root / str(subdir).strip().rstrip("/")


def scan_analysis_paths(
    cfg: dict,
    subdir: str,
    *,
    filename: str,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Scan analysis output tree for date/session-partitioned files."""
    root = build_analysis_output_dir(cfg, subdir)
    pattern = root / "date=*" / "session=*" / Path(str(filename)).name

    date_filter = None if dates is None else {str(val) for val in dates}
    session_filter = None if sessions is None else {str(val) for val in sessions}

    rows: list[dict] = []
    for path in root.glob(str(pattern.relative_to(root))):
        parts = path.parts
        date_part = next((part for part in parts if part.startswith("date=")), None)
        session_part = next((part for part in parts if part.startswith("session=")), None)
        if date_part is None or session_part is None:
            continue

        date = date_part.split("=", 1)[1]
        session = session_part.split("=", 1)[1]
        if date_filter is not None and date not in date_filter:
            continue
        if session_filter is not None and session not in session_filter:
            continue
        rows.append({"date": date, "session": session, "path": path})

    rows.sort(key=lambda row: (row["date"], row["session"]))
    return rows


def scan_analysis_date_paths(
    cfg: dict,
    subdir: str,
    *,
    filename: str,
    dates: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Scan analysis output tree for date-partitioned files."""
    root = build_analysis_output_dir(cfg, subdir)
    pattern = root / "date=*" / Path(str(filename)).name
    date_filter = None if dates is None else {str(val) for val in dates}

    rows: list[dict] = []
    for path in root.glob(str(pattern.relative_to(root))):
        parts = path.parts
        date_part = next((part for part in parts if part.startswith("date=")), None)
        if date_part is None:
            continue

        date = date_part.split("=", 1)[1]
        if date_filter is not None and date not in date_filter:
            continue
        rows.append({"date": date, "path": path})

    rows.sort(key=lambda row: row["date"])
    return rows
