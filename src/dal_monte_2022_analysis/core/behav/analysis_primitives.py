"""Shared pure helpers for behavioral analysis modules."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.data.records.behavioral import FixationBinaryVectorsData


def extract_fixation_vector(
    obj,
    fixation_label: str,
) -> Optional[np.ndarray]:
    """Extract a fixation vector from supported object layouts."""
    if isinstance(obj, FixationBinaryVectorsData):
        vectors = obj.vectors
    elif isinstance(obj, dict) and "vectors" in obj:
        vectors = obj["vectors"]
    elif isinstance(obj, dict):
        vectors = obj
    else:
        return None

    if not vectors or fixation_label not in vectors:
        return None

    vec = np.asarray(vectors[fixation_label])
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    return vec


def extract_monkey_name(obj) -> Optional[str]:
    """Extract monkey-name metadata from supported object layouts."""
    if isinstance(obj, FixationBinaryVectorsData):
        return obj.context.monkey_name
    if hasattr(obj, "context"):
        context = getattr(obj, "context")
        if hasattr(context, "monkey_name"):
            return getattr(context, "monkey_name")
    if isinstance(obj, dict):
        context = obj.get("context")
        if context is not None:
            if hasattr(context, "monkey_name"):
                return getattr(context, "monkey_name")
            if isinstance(context, dict) and "monkey_name" in context:
                return context.get("monkey_name")
        if "monkey_name" in obj:
            return obj.get("monkey_name")
    return None


def extract_pupil_vector(obj) -> Optional[np.ndarray]:
    """Extract a 1D pupil vector from supported object layouts."""
    if hasattr(obj, "d"):
        values = getattr(obj, "d")
    elif isinstance(obj, dict) and "d" in obj:
        values = obj["d"]
    else:
        return None

    vec = np.asarray(values, dtype=float).reshape(-1)
    return vec if vec.size else None


def to_bool(vec: np.ndarray) -> np.ndarray:
    """Coerce a vector to a 1D boolean array."""
    return np.asarray(vec).astype(bool, copy=False)


def filter_interactive_periods(
    df: Optional[pd.DataFrame],
    state_label: Optional[str],
) -> pd.DataFrame:
    """Filter interactive-period rows to requested state and valid columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    periods = df
    required_cols = {"start", "stop"}
    if not required_cols.issubset(periods.columns):
        return pd.DataFrame()

    if state_label is not None and "state" in periods.columns:
        periods = periods[periods["state"] == state_label]
    if periods.empty:
        return pd.DataFrame()

    return periods.sort_values(["start", "stop"]).reset_index(drop=True)


def clip_period(
    start,
    stop,
    max_len: int,
) -> Optional[tuple[int, int]]:
    """Clip a start/stop pair to [0, max_len - 1]."""
    if max_len <= 0:
        return None
    start_num = pd.to_numeric(start, errors="coerce")
    stop_num = pd.to_numeric(stop, errors="coerce")
    if pd.isna(start_num) or pd.isna(stop_num):
        return None
    start_idx = int(start_num)
    stop_idx = int(stop_num)
    if stop_idx < 0 or start_idx >= max_len:
        return None
    start_idx = max(0, start_idx)
    stop_idx = min(max_len - 1, stop_idx)
    if start_idx > stop_idx:
        return None
    return start_idx, stop_idx


def build_interactive_mask(
    periods_df: Optional[pd.DataFrame],
    *,
    n_samples: int,
    state_label: Optional[str],
) -> np.ndarray:
    """Build a boolean interactive mask with shape (n_samples,)."""
    mask = np.zeros(int(max(0, n_samples)), dtype=bool)
    periods = filter_interactive_periods(periods_df, state_label)
    if periods.empty:
        return mask

    for _, row in periods.iterrows():
        clipped = clip_period(row.get("start"), row.get("stop"), int(n_samples))
        if clipped is None:
            continue
        start, stop = clipped
        mask[start : stop + 1] = True
    return mask
