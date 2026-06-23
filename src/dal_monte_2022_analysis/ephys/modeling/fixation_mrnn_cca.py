"""CCA helpers for fixation mRNN regional PC dynamics."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _flatten_condition_time(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError("Regional PC tensors must have shape (condition, time, component).")
    flat = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if not np.isfinite(flat).all():
        raise ValueError("CCA input contains non-finite values.")
    return flat


def _inv_sqrt_covariance(values: np.ndarray, *, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    cov = np.cov(values, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals, eigvecs = np.linalg.eigh(cov + float(ridge) * np.eye(cov.shape[0]))
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return cov, inv_sqrt


def _variance_explained_by_scores(values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    total_variance = float(np.sum(np.var(centered, axis=0, ddof=1)))
    out = np.zeros(scores.shape[1], dtype=float)
    if total_variance <= 0:
        return out
    for idx in range(scores.shape[1]):
        score = scores[:, idx] - np.mean(scores[:, idx])
        score_var = float(np.var(score, ddof=1))
        if score_var <= 0:
            continue
        covariance = np.mean((centered - centered.mean(axis=0, keepdims=True)) * score[:, None], axis=0)
        out[idx] = float(np.sum((covariance**2) / score_var) / total_variance)
    return out


def compute_pairwise_regional_pc_cca(
    pcs_by_region: Mapping[str, np.ndarray],
    *,
    region_order: Sequence[str] | None = None,
    max_components: int | None = None,
    ridge: float = 1e-6,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, np.ndarray]]]:
    """Run CCA for all region pairs from condition x time x PC tensors."""
    regions = tuple(region_order or pcs_by_region.keys())
    rows = []
    payloads: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for region_a, region_b in combinations(regions, 2):
        x = _flatten_condition_time(pcs_by_region[region_a])
        y = _flatten_condition_time(pcs_by_region[region_b])
        n_components = min(x.shape[1], y.shape[1])
        if max_components is not None:
            n_components = min(n_components, int(max_components))
        if n_components <= 0:
            continue

        x_centered = x - x.mean(axis=0, keepdims=True)
        y_centered = y - y.mean(axis=0, keepdims=True)
        _, x_inv_sqrt = _inv_sqrt_covariance(x_centered, ridge=ridge)
        _, y_inv_sqrt = _inv_sqrt_covariance(y_centered, ridge=ridge)
        cross_cov = (x_centered.T @ y_centered) / max(x_centered.shape[0] - 1, 1)
        whitened_cross_cov = x_inv_sqrt @ cross_cov @ y_inv_sqrt
        left, singular_values, right_t = np.linalg.svd(whitened_cross_cov, full_matrices=False)

        x_weights = x_inv_sqrt @ left[:, :n_components]
        y_weights = y_inv_sqrt @ right_t.T[:, :n_components]
        x_scores = x_centered @ x_weights
        y_scores = y_centered @ y_weights
        x_var = _variance_explained_by_scores(x_centered, x_scores)
        y_var = _variance_explained_by_scores(y_centered, y_scores)
        canonical_correlations = singular_values[:n_components]
        for idx in range(n_components):
            rows.append(
                {
                    "region_a": region_a,
                    "region_b": region_b,
                    "cca_dimension": idx + 1,
                    "canonical_correlation": float(canonical_correlations[idx]),
                    "region_a_variance_explained": float(x_var[idx]),
                    "region_b_variance_explained": float(y_var[idx]),
                    "mean_variance_explained": float(np.mean([x_var[idx], y_var[idx]])),
                    "region_a_cumulative_variance_explained": float(np.sum(x_var[: idx + 1])),
                    "region_b_cumulative_variance_explained": float(np.sum(y_var[: idx + 1])),
                    "mean_cumulative_variance_explained": float(
                        np.mean([np.sum(x_var[: idx + 1]), np.sum(y_var[: idx + 1])])
                    ),
                }
            )
        payloads[(region_a, region_b)] = {
            "canonical_correlations": canonical_correlations,
            "region_a_weights": x_weights,
            "region_b_weights": y_weights,
            "region_a_scores": x_scores,
            "region_b_scores": y_scores,
            "region_a_variance_explained": x_var,
            "region_b_variance_explained": y_var,
        }
    return pd.DataFrame(rows), payloads


def plot_pairwise_cca_variance(
    cca_df: pd.DataFrame,
    *,
    value_column: str = "mean_cumulative_variance_explained",
    max_dimensions: int = 42,
    figsize: tuple[float, float] = (9.0, 4.8),
    dpi: int = 140,
):
    """Plot CCA canonical correlation and variance explained curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharex=True)
    for (region_a, region_b), sub in cca_df.groupby(["region_a", "region_b"], sort=False):
        sub = sub.sort_values("cca_dimension").head(int(max_dimensions))
        label = f"{region_a}-{region_b}"
        axes[0].plot(sub["cca_dimension"], sub["canonical_correlation"], marker="o", markersize=2.5, label=label)
        axes[1].plot(sub["cca_dimension"], sub[value_column], marker="o", markersize=2.5, label=label)
    axes[0].set_ylabel("Canonical correlation")
    axes[1].set_ylabel(value_column.replace("_", " "))
    for ax in axes:
        ax.set_xlabel("CC dimension")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig, axes


__all__ = [
    "compute_pairwise_regional_pc_cca",
    "plot_pairwise_cca_variance",
]
