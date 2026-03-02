"""Shared helpers for cross-correlation plotting modules."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from dal_monte_2022_analysis.utils.io import load_pickle
from dal_monte_2022_analysis.utils.paths import build_fix_cross_correlation_output_filename


def load_lags_for_scope(
    out_dir: Path,
    *,
    fixation_label: str,
    scope: str,
) -> np.ndarray:
    """Load lag axis for one scope."""
    lags_path = out_dir / build_fix_cross_correlation_output_filename(
        fixation_label,
        "lags",
        time_scope=scope,
    )
    if not lags_path.exists():
        raise FileNotFoundError(f"Missing lag file for scope='{scope}': {lags_path}")
    lags = np.asarray(load_pickle(lags_path), dtype=np.int64).reshape(-1)
    if lags.size == 0:
        raise RuntimeError(f"Lag file is empty for scope='{scope}': {lags_path}")
    return lags


def load_df_for_scope(
    out_dir: Path,
    *,
    fixation_label: str,
    scope: str,
    kind: str,
) -> pd.DataFrame:
    """Load cross-correlation dataframe for one scope and output kind."""
    data_path = out_dir / build_fix_cross_correlation_output_filename(
        fixation_label,
        kind,
        time_scope=scope,
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {kind} file for scope='{scope}': {data_path}")
    return pd.read_pickle(data_path)


def as_1d_float(arr) -> np.ndarray:
    """Coerce array-like to 1D float ndarray."""
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def nanmean_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-column mean and SEM with NaN handling."""
    if mat.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    mean = np.nanmean(mat, axis=0)
    finite_counts = np.sum(np.isfinite(mat), axis=0)
    std = np.nanstd(mat, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = std / np.sqrt(finite_counts)
    sem[finite_counts < 2] = np.nan
    return mean, sem


def downsample_indices(n_points: int, max_points: int) -> np.ndarray:
    """Return monotonic indices for plotting downsampling."""
    n = int(max(0, n_points))
    if n == 0:
        return np.asarray([], dtype=np.int64)
    cap = int(max(1, max_points))
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    step = int(np.ceil(n / float(cap)))
    return np.arange(0, n, step, dtype=np.int64)


def downsample_significance_mask(sig_full: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Collapse a full-resolution significance mask onto downsampled indices."""
    if idx.size == 0:
        return np.asarray([], dtype=bool)
    out = np.zeros(idx.size, dtype=bool)
    n = int(sig_full.size)
    for i, start in enumerate(idx):
        stop = int(idx[i + 1]) if i + 1 < idx.size else n
        out[i] = bool(np.any(sig_full[int(start) : stop]))
    return out


def limit_true_markers(mask: np.ndarray, max_true: int) -> np.ndarray:
    """Cap number of True markers by uniform subsampling over True positions."""
    out = np.asarray(mask, dtype=bool).copy()
    cap = int(max_true)
    if cap <= 0:
        out[:] = False
        return out
    true_idx = np.flatnonzero(out)
    if true_idx.size <= cap:
        return out
    keep = np.linspace(0, true_idx.size - 1, num=cap, dtype=int)
    keep_idx = true_idx[keep]
    out[:] = False
    out[keep_idx] = True
    return out


def scope_y_bounds(observed: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    """Return y-bounds from mean +/- SEM envelopes for one scope."""
    obs_mean, obs_sem = nanmean_sem(observed)
    ctl_mean, ctl_sem = nanmean_sem(control)
    y_lo = float(np.nanmin(np.r_[obs_mean - obs_sem, ctl_mean - ctl_sem]))
    y_hi = float(np.nanmax(np.r_[obs_mean + obs_sem, ctl_mean + ctl_sem]))
    if not np.isfinite(y_lo) or not np.isfinite(y_hi):
        return -1.0, 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-6
    return y_lo, y_hi


def _paired_ttest_per_lag_chunk(
    observed: np.ndarray,
    control: np.ndarray,
    *,
    start: int,
    stop: int,
) -> tuple[int, np.ndarray]:
    """Compute paired t-test p-values for one [start:stop) lag chunk."""
    x = observed[:, start:stop]
    y = control[:, start:stop]
    pvals = np.asarray(
        ttest_rel(x, y, axis=0, nan_policy="omit").pvalue,
        dtype=np.float64,
    ).reshape(-1)
    valid_counts = np.sum(np.isfinite(x) & np.isfinite(y), axis=0)
    pvals[valid_counts < 2] = np.nan
    return start, pvals


def paired_ttest_per_lag(
    observed: np.ndarray,
    control: np.ndarray,
    *,
    parallel: bool,
    workers: int | None,
    min_lags_for_parallel: int,
    chunk_size: int,
) -> np.ndarray:
    """Compute per-lag paired t-test p-values (optionally in parallel chunks)."""
    if observed.shape != control.shape:
        raise ValueError("Observed and control matrices must have same shape.")
    n_lags = observed.shape[1]
    if n_lags <= 0:
        return np.array([], dtype=np.float64)

    if (
        not parallel
        or n_lags < int(max(1, min_lags_for_parallel))
        or int(max(1, chunk_size)) >= n_lags
    ):
        pvals = np.asarray(
            ttest_rel(observed, control, axis=0, nan_policy="omit").pvalue,
            dtype=np.float64,
        ).reshape(-1)
        valid_counts = np.sum(np.isfinite(observed) & np.isfinite(control), axis=0)
        pvals[valid_counts < 2] = np.nan
        return pvals

    chunk = int(max(1, chunk_size))
    starts = list(range(0, n_lags, chunk))
    auto_workers = os.cpu_count() or 1
    n_workers = int(max(1, workers if workers is not None else auto_workers))
    n_workers = min(n_workers, len(starts))
    pvals = np.full(n_lags, np.nan, dtype=np.float64)
    if n_workers <= 1:
        for start in starts:
            stop = min(start + chunk, n_lags)
            _, chunk_p = _paired_ttest_per_lag_chunk(
                observed,
                control,
                start=start,
                stop=stop,
            )
            pvals[start:stop] = chunk_p
        return pvals

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for start in starts:
            stop = min(start + chunk, n_lags)
            futures.append(
                executor.submit(
                    _paired_ttest_per_lag_chunk,
                    observed,
                    control,
                    start=start,
                    stop=stop,
                )
            )
        for future in futures:
            start, chunk_p = future.result()
            stop = start + int(chunk_p.size)
            pvals[start:stop] = chunk_p
    return pvals
