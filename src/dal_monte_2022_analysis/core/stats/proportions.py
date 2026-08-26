"""Proportion estimation and comparison primitives."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

_Z_95 = 1.959963984540054


def wilson_score_interval(
    n_successes: int,
    n_total: int,
    *,
    z: float = _Z_95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval because several region-by-condition cells
    here sit near 0 or 1 with modest n, where Wald limits leave [0, 1] and
    understate uncertainty.
    """
    n = int(n_total)
    if n <= 0:
        return (np.nan, np.nan)
    k = int(n_successes)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def two_proportion_pvalue(
    n_successes_a: int,
    n_total_a: int,
    n_successes_b: int,
    n_total_b: int,
) -> float:
    """Two-sided chi-square p-value for equality of two independent proportions."""
    table = np.array(
        [
            [int(n_successes_a), int(n_total_a) - int(n_successes_a)],
            [int(n_successes_b), int(n_total_b) - int(n_successes_b)],
        ],
        dtype=float,
    )
    if np.any(table < 0) or np.any(table.sum(axis=1) == 0) or np.any(table.sum(axis=0) == 0):
        return float("nan")
    return float(stats.chi2_contingency(table, correction=False)[1])


def multinomial_uniform_pvalue(counts: Sequence[int]) -> tuple[float, float]:
    """Chi-square goodness-of-fit of ``counts`` against a uniform distribution.

    Returns ``(statistic, p_value)``.
    """
    observed = np.asarray(counts, dtype=float).reshape(-1)
    if observed.size < 2 or observed.sum() <= 0:
        return (float("nan"), float("nan"))
    statistic, p_value = stats.chisquare(observed)[:2]
    return (float(statistic), float(p_value))


def significance_stars(p_value: float, *, alpha: float = 0.05) -> str:
    """Conventional star annotation for a p-value; ``n.s.`` when above ``alpha``."""
    if p_value is None or not np.isfinite(p_value):
        return "n.s."
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < float(alpha):
        return "*"
    return "n.s."
