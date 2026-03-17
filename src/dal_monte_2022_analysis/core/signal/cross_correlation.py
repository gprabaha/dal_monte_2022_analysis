"""Signal cross-correlation helpers used across behavior and ephys analyses."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import correlate, correlation_lags


def fft_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_lag: Optional[int] = None,
    round_to_int: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full linear cross-correlation using SciPy's FFT-backed correlate."""
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(x_vec.size)
    m = int(y_vec.size)
    if n == 0 or m == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    corr_full = np.asarray(
        correlate(x_vec, y_vec, mode="full", method="fft"),
        dtype=np.float64,
    ).reshape(-1)
    lags = np.asarray(
        correlation_lags(n, m, mode="full"),
        dtype=np.int64,
    ).reshape(-1)

    if max_lag is not None:
        keep = np.abs(lags) <= int(max(0, int(max_lag)))
        lags = lags[keep]
        corr_full = corr_full[keep]

    if round_to_int:
        corr_full = np.rint(corr_full).astype(np.int64)
    return lags, corr_full


def summarize_cross_correlation(lags: np.ndarray, corr: np.ndarray) -> dict:
    """Compute lag/count summary metadata for one cross-correlation trace."""
    vec = np.asarray(corr).reshape(-1)
    if vec.size == 0:
        return {
            "n_lags": 0,
            "zero_lag_correlation": None,
            "peak_lag": None,
            "peak_correlation": None,
        }

    lag_vec = np.asarray(lags, dtype=np.int64).reshape(-1)
    zero_lag = None
    zero_idx = np.where(lag_vec == 0)[0]
    if zero_idx.size > 0:
        zero_lag = float(vec[int(zero_idx[0])])

    peak_idx = int(np.argmax(vec))
    return {
        "n_lags": int(vec.size),
        "zero_lag_correlation": zero_lag,
        "peak_lag": int(lag_vec[peak_idx]),
        "peak_correlation": float(vec[peak_idx]),
    }


def normalize_cross_correlation_energy(
    corr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Normalize by the product of input signal L2 norms."""
    vec = np.asarray(corr, dtype=np.float64).reshape(-1)
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(x_vec) * np.linalg.norm(y_vec))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros_like(vec)
    return vec / denom


def normalize_cross_correlation_sqrt_bin_count(
    corr: np.ndarray,
    x_bin_count: int,
    y_bin_count: int,
) -> np.ndarray:
    """Normalize by sqrt(count_x * count_y)."""
    norm_factor = float(np.sqrt(float(x_bin_count) * float(y_bin_count)))
    if norm_factor <= 0.0:
        return np.zeros(np.asarray(corr).size, dtype=np.float32)
    return (np.asarray(corr, dtype=np.float64) / norm_factor).astype(np.float32)


def assert_lag_axis_match(expected_lags: np.ndarray, lags: np.ndarray, *, message: str) -> None:
    """Raise when two lag vectors are not identical."""
    if expected_lags.shape != lags.shape or not np.array_equal(expected_lags, lags):
        raise ValueError(message)
