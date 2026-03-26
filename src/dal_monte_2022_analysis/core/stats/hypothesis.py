"""Shared hypothesis-test utilities."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

from dal_monte_2022_analysis.core.stats.tests import (
    safe_mannwhitneyu,
    safe_welch_ttest,
)


def two_sample_pvalues(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compute t-test, ranksum, and KS p-values for two samples."""
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return {"ttest": np.nan, "ranksum": np.nan, "ks": np.nan}

    _, ttest_p = safe_welch_ttest(x, y)
    _, ranksum_p = safe_mannwhitneyu(x, y)
    ranksum_res = stats.ranksums(x, y)
    ks_res = stats.ks_2samp(x, y)
    return {
        "ttest": float(ttest_p),
        "ranksum": float(ranksum_p if np.isfinite(ranksum_p) else ranksum_res.pvalue),
        "ks": float(ks_res.pvalue),
    }


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
        stats.ttest_rel(x, y, axis=0, nan_policy="omit").pvalue,
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
            stats.ttest_rel(observed, control, axis=0, nan_policy="omit").pvalue,
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
