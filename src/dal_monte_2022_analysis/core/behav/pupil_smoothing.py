"""Pure fixation-guided pupil smoothing kernels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def coerce_intervals(fix_df: pd.DataFrame, *, n_samples: int) -> list[tuple[int, int]]:
    """Extract valid clipped fixation start/stop intervals."""
    intervals: list[tuple[int, int]] = []
    if not isinstance(fix_df, pd.DataFrame) or fix_df.empty:
        return intervals
    if "start" not in fix_df.columns or "stop" not in fix_df.columns:
        return intervals

    for _, row in fix_df.iterrows():
        try:
            start = int(row["start"])
            stop = int(row["stop"])
        except (TypeError, ValueError):
            continue
        if stop < 0 or start >= n_samples:
            continue
        start = max(0, start)
        stop = min(n_samples - 1, stop)
        if start > stop:
            continue
        intervals.append((start, stop))
    return intervals


def build_fixation_mask(intervals: list[tuple[int, int]], *, n_samples: int) -> np.ndarray:
    """Create a boolean fixation mask from intervals."""
    mask = np.zeros(n_samples, dtype=bool)
    for start, stop in intervals:
        mask[start : stop + 1] = True
    return mask


def pchip_interpolate_1d(*, query_idx: np.ndarray, known_idx: np.ndarray, known_values: np.ndarray) -> np.ndarray:
    """Interpolate a 1D signal with PCHIP and edge fill."""
    q = np.asarray(query_idx, dtype=float).reshape(-1)
    x = np.asarray(known_idx, dtype=float).reshape(-1)
    y = np.asarray(known_values, dtype=float).reshape(-1)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return np.zeros_like(q, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]

    if x.size == 1:
        return np.full_like(q, float(y[0]), dtype=float)

    interpolator = PchipInterpolator(x, y, extrapolate=False)
    out = interpolator(q).astype(float)

    left_mask = q < x[0]
    right_mask = q > x[-1]
    out[left_mask] = y[0]
    out[right_mask] = y[-1]

    nan_mask = ~np.isfinite(out)
    if np.any(nan_mask):
        out[nan_mask & (q <= x[0])] = y[0]
        out[nan_mask & (q >= x[-1])] = y[-1]
    return out


def interp_nan_1d(values: np.ndarray) -> np.ndarray:
    """Fill NaNs in 1D array via PCHIP interpolation."""
    arr = np.asarray(values, dtype=float).reshape(-1).copy()
    idx = np.arange(arr.size, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=float)
    return pchip_interpolate_1d(
        query_idx=idx,
        known_idx=idx[valid],
        known_values=arr[valid],
    )


def interpolate_fixation_gaps(fixation_anchor: np.ndarray) -> np.ndarray:
    """Interpolate non-fixation gaps from fixation anchors."""
    anchor = np.asarray(fixation_anchor, dtype=float).reshape(-1)
    idx = np.arange(anchor.size, dtype=float)
    valid = np.isfinite(anchor)
    if np.count_nonzero(valid) == 0:
        return np.zeros_like(anchor, dtype=float)
    return pchip_interpolate_1d(
        query_idx=idx,
        known_idx=idx[valid],
        known_values=anchor[valid],
    )


def gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 1D signal with Gaussian kernel using reflect padding."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    sigma = float(max(sigma, 1e-6))
    if arr.size <= 2 or sigma <= 0.25:
        return arr.copy()

    radius = int(max(1, np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)

    padded = np.pad(arr, (radius, radius), mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(float, copy=False)


def estimate_fixation_noise(raw_pupil: np.ndarray, intervals: list[tuple[int, int]]) -> tuple[float, np.ndarray]:
    """Estimate pupil noise from within-fixation residual variability."""
    residuals: list[np.ndarray] = []
    fixation_values: list[np.ndarray] = []

    for start, stop in intervals:
        seg = raw_pupil[start : stop + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        fixation_values.append(seg)
        if seg.size < 2:
            continue
        seg_median = np.median(seg)
        residuals.append(seg - seg_median)

    if fixation_values:
        fix_vals = np.concatenate(fixation_values)
    else:
        fix_vals = raw_pupil[np.isfinite(raw_pupil)]

    if residuals:
        resid = np.concatenate(residuals)
        noise_sigma = 1.4826 * np.median(np.abs(resid))
    elif fix_vals.size > 1:
        noise_sigma = float(np.nanstd(fix_vals))
    else:
        noise_sigma = 0.0

    return float(max(noise_sigma, 0.0)), np.asarray(fix_vals, dtype=float)


def resolve_sigma(
    *,
    noise_sigma: float,
    fixation_values: np.ndarray,
    base_sigma_samples: float,
    min_sigma_samples: float,
    max_sigma_samples: float,
    adaptive_noise_gain: float,
) -> float:
    """Resolve smoothing sigma from fixation-derived noise variability."""
    arr = np.asarray(fixation_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size > 3:
        q1, q3 = np.percentile(arr, [25.0, 75.0])
        scale = float(max(q3 - q1, 1e-6))
    elif arr.size > 1:
        scale = float(max(np.nanstd(arr), 1e-6))
    else:
        scale = 1.0

    noise_ratio = float(max(noise_sigma / scale, 0.0))
    sigma = float(base_sigma_samples) * (1.0 + float(adaptive_noise_gain) * noise_ratio)
    sigma = float(np.clip(sigma, float(min_sigma_samples), float(max_sigma_samples)))
    return sigma


__all__ = [
    "build_fixation_mask",
    "coerce_intervals",
    "estimate_fixation_noise",
    "gaussian_smooth_1d",
    "interp_nan_1d",
    "interpolate_fixation_gaps",
    "pchip_interpolate_1d",
    "resolve_sigma",
]
