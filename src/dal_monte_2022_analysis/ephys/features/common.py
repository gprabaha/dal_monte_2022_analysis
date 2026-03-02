"""Shared helpers for ephys feature builders."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def as_optional_str(value: object) -> Optional[str]:
    """Normalize optional values to stripped strings or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def ensure_pkl_filename(filename: str) -> str:
    """Return filename with `.pkl` suffix."""
    name = str(filename).strip()
    if not name:
        raise ValueError("Output filename cannot be empty.")
    return name if name.endswith(".pkl") else f"{name}.pkl"


def build_symmetric_bin_edges(
    *,
    bin_size_ms: float,
    window_pre_s: float,
    window_post_s: float,
) -> np.ndarray:
    """Build histogram bin edges centered around event anchors."""
    bin_size_s = float(bin_size_ms) / 1000.0
    if bin_size_s <= 0:
        raise ValueError("bin_size_ms must be > 0.")
    if window_pre_s <= 0 or window_post_s <= 0:
        raise ValueError("window_pre_s and window_post_s must be > 0.")
    pre = float(window_pre_s)
    post = float(window_post_s)
    return np.arange(-pre, post + bin_size_s * 0.5, bin_size_s, dtype=float)


def units_to_payloads(units) -> list[dict]:
    """Convert unit dataclasses into worker-safe payload dictionaries."""
    payloads: list[dict] = []
    for unit in units:
        ctx = unit.context
        payloads.append(
            {
                "unit_uuid": str(ctx.unit_uuid),
                "unit_date": str(ctx.date),
                "region": as_optional_str(ctx.region),
                "spike_channel": as_optional_str(ctx.spike_channel),
                "session_name": str(ctx.session_name),
                "recorded_agent": as_optional_str(ctx.recorded_agent),
                "recorded_monkey": as_optional_str(ctx.recorded_monkey),
                "area": as_optional_str(ctx.area),
                "spike_ts": np.asarray(unit.spike_ts, dtype=float),
            }
        )
    return payloads

