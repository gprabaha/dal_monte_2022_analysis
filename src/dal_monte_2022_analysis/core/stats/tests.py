"""Shared hypothesis-test helpers for common statistical comparisons."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def _finite_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _finite_paired(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def safe_welch_ttest(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Run a Welch two-sample t-test on finite values only."""
    x = _finite_1d(left)
    y = _finite_1d(right)
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan
    stat, p_value = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return float(stat), float(p_value)


def safe_paired_ttest(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> tuple[float, float, int]:
    """Run a paired t-test on finite paired values only."""
    x, y = _finite_paired(left, right)
    n_pairs = int(x.size)
    if n_pairs < 2:
        return np.nan, np.nan, n_pairs

    diff = x - y
    if np.allclose(diff, diff[0]):
        if float(diff[0]) > 0.0:
            return float("inf"), 0.0, n_pairs
        if float(diff[0]) < 0.0:
            return float("-inf"), 0.0, n_pairs
        return 0.0, 1.0, n_pairs

    stat, p_value = stats.ttest_rel(x, y, nan_policy="omit")
    return float(stat), float(p_value), n_pairs


def safe_one_sample_ttest(
    values: Sequence[float] | np.ndarray,
    *,
    popmean: float = 0.0,
) -> tuple[float, float, int]:
    """Run a one-sample t-test on finite values only."""
    arr = _finite_1d(values)
    n = int(arr.size)
    if n < 2:
        return np.nan, np.nan, n

    diff = arr - float(popmean)
    if np.allclose(diff, diff[0]):
        if float(diff[0]) > 0.0:
            return float("inf"), 0.0, n
        if float(diff[0]) < 0.0:
            return float("-inf"), 0.0, n
        return 0.0, 1.0, n

    stat, p_value = stats.ttest_1samp(arr, popmean=float(popmean), nan_policy="omit")
    return float(stat), float(p_value), n


def one_sided_pvalue_from_ttest(
    statistic: float | np.ndarray,
    p_two_sided: float | np.ndarray,
    *,
    alternative: str = "greater",
) -> float | np.ndarray:
    """Convert two-sided t-test p-values into one-sided p-values."""
    token = str(alternative).strip().lower()
    if token not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'.")

    stat_arr, p_arr = np.broadcast_arrays(
        np.asarray(statistic, dtype=float),
        np.asarray(p_two_sided, dtype=float),
    )
    out = np.full(stat_arr.shape, np.nan, dtype=float)
    valid = ~np.isnan(stat_arr) & np.isfinite(p_arr)
    if np.any(valid):
        stat_vals = stat_arr[valid]
        p_vals = p_arr[valid]
        if token == "greater":
            out[valid] = np.where(stat_vals > 0.0, p_vals / 2.0, 1.0 - (p_vals / 2.0))
        else:
            out[valid] = np.where(stat_vals < 0.0, p_vals / 2.0, 1.0 - (p_vals / 2.0))

    if out.ndim == 0:
        return float(out.reshape(()))
    return out


def safe_one_sample_ttest_greater(
    values: Sequence[float] | np.ndarray,
    *,
    popmean: float = 0.0,
) -> tuple[float, float, int]:
    """Run a one-sample t-test and report a one-sided greater-than p-value."""
    stat, p_two, n = safe_one_sample_ttest(values, popmean=popmean)
    if np.isnan(stat) or not np.isfinite(p_two):
        return stat, np.nan, n
    p_one = one_sided_pvalue_from_ttest(stat, p_two, alternative="greater")
    return float(stat), float(p_one), n


def safe_mannwhitneyu(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Run a Mann-Whitney U test on finite values only."""
    x = _finite_1d(left)
    y = _finite_1d(right)
    if x.size == 0 or y.size == 0:
        return np.nan, np.nan
    stat, p_value = stats.mannwhitneyu(x, y, alternative=alternative)
    return float(stat), float(p_value)


def welch_ttest(
    left: np.ndarray,
    right: np.ndarray,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a vectorized Welch t-test along one axis."""
    res = stats.ttest_ind(left, right, axis=axis, equal_var=False, nan_policy="omit")
    return (
        np.asarray(res.statistic, dtype=float),
        np.asarray(res.pvalue, dtype=float),
    )


def paired_ttest(
    left: np.ndarray,
    right: np.ndarray,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a vectorized paired t-test along one axis."""
    res = stats.ttest_rel(left, right, axis=axis, nan_policy="omit")
    return (
        np.asarray(res.statistic, dtype=float),
        np.asarray(res.pvalue, dtype=float),
    )


def one_sample_ttest(
    values: np.ndarray,
    *,
    popmean: float = 0.0,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a vectorized one-sample t-test along one axis."""
    res = stats.ttest_1samp(values, popmean=float(popmean), axis=axis, nan_policy="omit")
    return (
        np.asarray(res.statistic, dtype=float),
        np.asarray(res.pvalue, dtype=float),
    )


def one_sample_ttest_greater(
    values: np.ndarray,
    *,
    popmean: float = 0.0,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a one-sample t-test and return one-sided greater-than p-values."""
    stat, p_two = one_sample_ttest(values, popmean=popmean, axis=axis)
    p_one = one_sided_pvalue_from_ttest(stat, p_two, alternative="greater")
    return np.asarray(stat, dtype=float), np.asarray(p_one, dtype=float)


def mannwhitneyu_pvalues_per_column(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Run a Mann-Whitney U test independently for each column."""
    mat_left = np.asarray(left, dtype=float)
    mat_right = np.asarray(right, dtype=float)
    if mat_left.ndim != 2 or mat_right.ndim != 2 or mat_left.shape[1] != mat_right.shape[1]:
        raise ValueError("left and right must be 2D arrays with matching column counts.")
    n_cols = int(mat_left.shape[1])
    p_vals = np.full(n_cols, np.nan, dtype=float)
    for idx in range(n_cols):
        _, p_value = safe_mannwhitneyu(mat_left[:, idx], mat_right[:, idx], alternative="two-sided")
        p_vals[idx] = p_value
    return p_vals
