"""Shared helpers for ephys plotting modules."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.behav.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.io.plot_output import normalize_extension


def safe_optional_str(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def ensure_ext(ext: str, *, fallback: str) -> str:
    return normalize_extension(ext, fallback=fallback)


def iter_trial_rows(
    cfg: dict,
    *,
    modality: str,
    filename: str,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    root = Path(cfg["processed_data_root"])
    resolved_filename = str(filename) if str(filename).endswith(".pkl") else f"{filename}.pkl"
    pattern = root / "date=*" / "session=*" / str(modality) / resolved_filename

    date_filter = None if dates is None else {str(v) for v in dates}
    session_filter = None if sessions is None else {str(v) for v in sessions}

    rows: list[dict] = []
    for path in root.glob(str(pattern.relative_to(root))):
        parts = path.parts
        try:
            date_part = next(part for part in parts if part.startswith("date="))
            session_part = next(part for part in parts if part.startswith("session="))
        except StopIteration:
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


def stable_seed(base_seed: int, *parts: str) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**32 - 1)


def sample_rows(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= int(max_rows):
        return df
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(df), size=int(max_rows), replace=False))
    return df.iloc[picked]


def row_counts(row, n_bins: int) -> Optional[np.ndarray]:
    counts = np.asarray(getattr(row, "psth_counts"), dtype=float).reshape(-1)
    if counts.size != n_bins:
        return None
    return counts


def counts_to_spike_times(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    bin_size_s: float,
    *,
    jitter_within_bin: bool,
    rng: np.random.Generator,
    max_spikes_per_bin: Optional[int] = None,
) -> np.ndarray:
    spike_bins = np.clip(np.rint(np.asarray(counts, dtype=float)).astype(int), 0, None)
    if max_spikes_per_bin is not None and int(max_spikes_per_bin) > 0:
        spike_bins = np.minimum(spike_bins, int(max_spikes_per_bin))
    if spike_bins.size != bin_centers.size:
        return np.array([], dtype=float)
    bin_indices = np.repeat(np.arange(spike_bins.size), spike_bins)
    if bin_indices.size == 0:
        return np.array([], dtype=float)
    spike_ts = np.asarray(bin_centers[bin_indices], dtype=float)
    if jitter_within_bin and float(bin_size_s) > 0:
        jitter = (rng.random(spike_ts.size) - 0.5) * float(bin_size_s)
        spike_ts = spike_ts + jitter
    return np.sort(spike_ts)


def fallback_bin_centers(
    *,
    bin_size_ms_fallback: float,
    window_pre_s: float,
    window_post_s: float,
) -> np.ndarray:
    bin_size_s = float(bin_size_ms_fallback) / 1000.0
    pre = float(window_pre_s)
    post = float(window_post_s)
    edges = np.arange(-pre, post + bin_size_s * 0.5, bin_size_s, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def resolve_figsize_and_dpi(
    *,
    plotting_cfg_path: str,
    output_dpi: Optional[int],
    default_figsize: list[float],
) -> tuple[list[float], Optional[int]]:
    if plotting_cfg_path and Path(plotting_cfg_path).exists():
        plot_cfg = load_config(plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        figsize, cfg_dpi = resolve_figsize(plot_cfg)
    else:
        figsize, cfg_dpi = (None, None)

    if figsize is None:
        figsize = list(default_figsize)
    dpi = output_dpi if output_dpi is not None else cfg_dpi
    return [float(figsize[0]), float(figsize[1])], dpi


def safe_unit_filename(unit_uuid: str) -> str:
    safe = str(unit_uuid).strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace(" ", "_")
    return safe if safe else "unknown"


def safe_region_folder(region: Optional[str]) -> str:
    if region is None:
        return "region=unknown"
    safe = str(region).strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace(" ", "_")
    safe = safe if safe else "unknown"
    return f"region={safe}"


def darken_color(hex_color: str, factor: float):
    fac = min(max(float(factor), 0.0), 1.0)
    r, g, b = to_rgb(hex_color)
    return (r * fac, g * fac, b * fac)
