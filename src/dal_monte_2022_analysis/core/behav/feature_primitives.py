"""Shared pure primitives used by behavioral feature builders."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1], returning zeros for flat arrays."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    min_val = float(arr.min())
    max_val = float(arr.max())
    if np.isclose(max_val, min_val):
        return np.zeros_like(arr, dtype=float)
    return (arr - min_val) / (max_val - min_val)


def extract_monkey_name(frame, *, monkey_col: str = "monkey_name") -> Optional[str]:
    """Extract a monkey name from a table-like object when present."""
    if frame is None:
        return None
    if getattr(frame, "empty", True):
        return None
    columns = getattr(frame, "columns", [])
    if monkey_col not in columns:
        return None
    valid = frame[monkey_col].dropna()
    if valid.empty:
        return None
    return str(valid.iloc[0])


def extract_density_vector(obj, *, key: Optional[str] = None) -> Optional[np.ndarray]:
    """Extract a 1D density vector from supported object shapes."""
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj, dtype=float).reshape(-1)
        return arr if arr.size else None

    if hasattr(obj, "density"):
        arr = np.asarray(getattr(obj, "density"), dtype=float).reshape(-1)
        return arr if arr.size else None

    if hasattr(obj, "vectors"):
        vectors = getattr(obj, "vectors")
        if key is None or not vectors or key not in vectors:
            return None
        arr = np.asarray(vectors[key], dtype=float).reshape(-1)
        return arr if arr.size else None

    if isinstance(obj, dict):
        if key is None and "density" in obj:
            arr = np.asarray(obj["density"], dtype=float).reshape(-1)
            return arr if arr.size else None
        if key is not None and key in obj:
            arr = np.asarray(obj[key], dtype=float).reshape(-1)
            return arr if arr.size else None

    return None


def find_contiguous_periods(mask: Iterable[bool]) -> list[tuple[int, int, bool]]:
    """Return start/stop indices for contiguous periods of a boolean mask."""
    periods: list[tuple[int, int, bool]] = []
    mask_list = list(mask)
    if not mask_list:
        return periods
    start = 0
    current = bool(mask_list[0])
    for idx in range(1, len(mask_list)):
        state = bool(mask_list[idx])
        if state != current:
            periods.append((start, idx - 1, current))
            start = idx
            current = state
    periods.append((start, len(mask_list) - 1, current))
    return periods


def fixation_durations(events: Sequence[tuple[int, int]]) -> np.ndarray:
    """Compute fixation durations (inclusive sample count) for valid events."""
    values = [stop - start + 1 for start, stop in events if stop >= start]
    return np.asarray(values, dtype=float)
