"""Contracts for gaze-event artifacts."""

from __future__ import annotations

import pandas as pd

GAZE_EVENT_REQUIRED_COLUMNS = (
    "date",
    "session",
    "agent",
    "monkey_name",
    "start",
    "stop",
)


def validate_gaze_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a gaze-event dataframe.

    The output dataframe always contains required columns and integer start/stop
    samples, with each interval satisfying start <= stop.
    """
    missing = [col for col in GAZE_EVENT_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required gaze-event columns: {missing}")

    out = df.copy()
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["stop"] = pd.to_numeric(out["stop"], errors="coerce")
    if out["start"].isna().any() or out["stop"].isna().any():
        raise ValueError("Gaze-event start/stop columns contain non-numeric values.")
    out["start"] = out["start"].astype(int)
    out["stop"] = out["stop"].astype(int)

    invalid = out["start"] > out["stop"]
    if bool(invalid.any()):
        raise ValueError("Found gaze-event intervals with start > stop.")

    return out

