"""Permutation-test helpers shared across analyses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def permutation_mean_difference_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Permutation test for a difference in means."""
    arr_left = np.asarray(left, dtype=float).reshape(-1)
    arr_right = np.asarray(right, dtype=float).reshape(-1)
    arr_left = arr_left[np.isfinite(arr_left)]
    arr_right = arr_right[np.isfinite(arr_right)]
    if arr_left.size == 0 or arr_right.size == 0:
        return np.nan, np.nan

    observed = float(np.mean(arr_left) - np.mean(arr_right))
    if int(n_permutations) <= 0:
        return observed, np.nan

    merged = np.concatenate([arr_left, arr_right], axis=0)
    n_left = int(arr_left.size)
    obs_abs = abs(observed)
    hits = 0
    for _ in range(int(n_permutations)):
        perm = rng.permutation(merged)
        diff = float(np.mean(perm[:n_left]) - np.mean(perm[n_left:]))
        if np.isfinite(diff) and abs(diff) >= obs_abs - 1e-12:
            hits += 1
    p_value = (float(hits) + 1.0) / (float(n_permutations) + 1.0)
    return observed, float(p_value)


def permutation_label_statistic_test(
    data: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, float]:
    """Permutation test for a statistic defined over fixed data and shuffled labels."""
    observed = float(statistic_fn(data, labels))
    if not np.isfinite(observed):
        return observed, np.nan
    if int(n_permutations) <= 0:
        return observed, np.nan

    hits = 0
    labels_arr = np.asarray(labels).copy()
    for _ in range(int(n_permutations)):
        permuted = rng.permutation(labels_arr)
        stat = float(statistic_fn(data, permuted))
        if np.isfinite(stat) and stat >= observed - 1e-12:
            hits += 1
    p_value = (float(hits) + 1.0) / (float(n_permutations) + 1.0)
    return observed, float(p_value)
