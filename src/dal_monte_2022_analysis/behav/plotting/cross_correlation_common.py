"""Shared helpers for cross-correlation plotting modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.core.behav.analysis_filenames import (
    build_fix_cross_correlation_output_filename,
)
from dal_monte_2022_analysis.core.stats.hypothesis import paired_ttest_per_lag
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path


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
    lags = np.asarray(load_pickle_path(lags_path), dtype=np.int64).reshape(-1)
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


def stack_column_arrays(df: pd.DataFrame, col: str) -> np.ndarray:
    """Stack fixed-length arrays from one dataframe column into 2D matrix."""
    mats = [as_1d_float(v) for v in df[col].to_list()]
    if not mats:
        return np.empty((0, 0), dtype=np.float64)
    n_lags = mats[0].size
    for idx, row in enumerate(mats):
        if row.size != n_lags:
            raise RuntimeError(
                f"Array-length mismatch at row={idx} col={col}: {row.size} != {n_lags}",
            )
    return np.vstack(mats)


def paired_session_matrices(
    within_df: pd.DataFrame,
    control_df: pd.DataFrame,
    *,
    control_col: str,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Align within/control by (date, session) and return paired matrices."""
    within_cols = ["date", "session", "cross_correlation"]
    control_cols = ["date", "session", control_col]
    missing_within = set(within_cols).difference(within_df.columns)
    missing_control = set(control_cols).difference(control_df.columns)
    if missing_within:
        raise RuntimeError(f"Within-session table missing columns: {sorted(missing_within)}")
    if missing_control:
        raise RuntimeError(f"Control table missing columns: {sorted(missing_control)}")

    merged = within_df[within_cols].merge(
        control_df[control_cols],
        how="inner",
        on=["date", "session"],
    )
    if merged.empty:
        raise RuntimeError("No overlapping (date, session) rows between observed and control tables.")

    observed = stack_column_arrays(merged, "cross_correlation")
    control = stack_column_arrays(merged, control_col)
    if observed.shape != control.shape:
        raise RuntimeError(
            "Observed/control paired matrices have different shapes: "
            f"{observed.shape} vs {control.shape}"
        )
    return observed, control, int(len(within_df)), int(len(control_df)), int(len(merged))


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


def significance_mask_per_lag(
    observed: np.ndarray,
    control: np.ndarray,
    *,
    alpha: float,
    parallel: bool,
    workers: int | None,
    min_lags_for_parallel: int,
    chunk_size: int,
) -> np.ndarray:
    """Return per-lag significance mask from paired t-test p-values."""
    pvals = paired_ttest_per_lag(
        observed,
        control,
        parallel=parallel,
        workers=workers,
        min_lags_for_parallel=min_lags_for_parallel,
        chunk_size=chunk_size,
    )
    return np.isfinite(pvals) & (pvals < float(alpha))
