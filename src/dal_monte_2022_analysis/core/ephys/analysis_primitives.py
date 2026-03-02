"""Shared pure helpers for ephys analysis modules."""

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
    token = str(value).strip()
    return token or None


def ensure_filename(name: str, suffix: str) -> str:
    """Return a non-empty filename with a required suffix."""
    text = str(name).strip()
    if not text:
        raise ValueError("Output filename cannot be empty.")
    return text if text.endswith(suffix) else f"{text}{suffix}"


def as_bool(value: object, interactive_label: Optional[str] = None) -> bool:
    """Coerce mixed boolean-like values to bool."""
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return float(value) != 0.0
    token = str(value).strip().lower()
    accepted = {"1", "true", "t", "yes", "y", "interactive"}
    if interactive_label is not None:
        accepted.add(str(interactive_label).strip().lower())
    return token in accepted


def extract_trials_df_and_meta(obj) -> tuple[pd.DataFrame, dict]:
    """Extract a trials dataframe and metadata from supported layouts."""
    if isinstance(obj, dict) and "trials" in obj:
        df = obj["trials"]
        meta = obj.get("meta", {}) or {}
        return (df if isinstance(df, pd.DataFrame) else pd.DataFrame(), meta)
    if isinstance(obj, pd.DataFrame):
        return obj, {}
    return pd.DataFrame(), {}


def resolve_bin_centers_from_meta(meta: dict) -> Optional[np.ndarray]:
    """Resolve PSTH bin centers from metadata if available."""
    centers = meta.get("bin_centers_s_rel")
    if centers is not None:
        arr = np.asarray(centers, dtype=float).reshape(-1)
        if arr.size > 0:
            return arr
    edges = meta.get("bin_edges_s_rel")
    if edges is not None:
        arr = np.asarray(edges, dtype=float).reshape(-1)
        if arr.size > 1:
            return 0.5 * (arr[:-1] + arr[1:])
    return None
