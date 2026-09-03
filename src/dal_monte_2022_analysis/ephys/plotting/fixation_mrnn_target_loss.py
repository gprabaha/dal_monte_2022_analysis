"""Figures for the target-and-loss notebook.

The selection here is explicitly multi-objective -- fit against the noise ceiling,
recovery of fast temporal structure, and agreement between independently seeded fits --
so the panels are built to show the trade rather than to rank on one number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    NEUTRAL_EDGE,
    NEUTRAL_FILL,
    nice_axis,
    region_label,
)

BASELINE_COLOR = "#8f8f8f"
IMPROVED_COLOR = "#2a6f4e"
WORSE_COLOR = "#c1121f"


def _panel(ax, letter: str) -> None:
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def _log_y(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())


def plot_loss_trajectories(
    histories: Mapping[str, Sequence[pd.DataFrame]],
    *,
    components: Sequence[str] = ("loss", "reconstruction_loss", "temporal_derivative_loss", "temporal_curvature_loss"),
    figsize: tuple[float, float] = (7.6, 5.4),
    max_variants: int = 9,
):
    """Every variant's optimisation history, one panel each.

    **These curves are not comparable between panels.** Each variant optimises a
    different objective -- reweighting the conditions or the components changes what the
    number means -- so a lower curve here is not a better model. What they are for is
    convergence: whether each fit settled, and which term of the objective was still
    moving when training stopped.
    """
    labels = list(histories)[:max_variants]
    n_columns = min(3, max(len(labels), 1))
    n_rows = int(np.ceil(len(labels) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize, squeeze=False, sharex=True)

    for index, label in enumerate(labels):
        ax = axes[index // n_columns][index % n_columns]
        runs = histories[label]
        for component in components:
            colour = INK if component == "loss" else None
            for seed_index, history in enumerate(runs):
                if component not in history.columns:
                    continue
                values = history[component].to_numpy(dtype=float)
                if not np.any(values > 0):
                    continue
                ax.plot(
                    np.arange(1, len(values) + 1),
                    np.clip(values, 1e-12, None),
                    linewidth=1.1 if component == "loss" else 0.7,
                    color=colour,
                    alpha=0.9 if component == "loss" else 0.6,
                    label=component.replace("_loss", "").replace("_", " ") if seed_index == 0 else None,
                )
        ax.set_title(label, fontsize=6.8, pad=4)
        nice_axis(ax)
        _log_y(ax)
        if index % n_columns == 0:
            ax.set_ylabel("loss")
        if index // n_columns == n_rows - 1:
            ax.set_xlabel("iteration")
    for index in range(len(labels), n_rows * n_columns):
        axes[index // n_columns][index % n_columns].axis("off")
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=6.2, ncol=len(legend_labels), loc="upper center",
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_ceiling_relative_fit(
    fit: pd.DataFrame,
    *,
    baseline: str | None = None,
    figsize: tuple[float, float] = (7.6, 2.9),
):
    """Fit as a fraction of what the data determines, by variant and condition.

    A value of 1.0 means the model has reproduced everything the split-half reliability
    says is reproducible. Above it means the model is reproducing sampling noise, which
    is a diagnosis rather than an achievement, so the axis is drawn symmetrically about 1.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.5, 1.0]})
    labels = list(dict.fromkeys(fit["label"]))
    conditions = [c for c in CONDITION_ORDER if c in set(fit["condition"])]

    ax = axes[0]
    width = 0.8 / max(len(conditions), 1)
    for index, condition in enumerate(conditions):
        values = [
            float(fit[(fit["label"] == label) & (fit["condition"] == condition)]["r2_vs_ceiling"].mean())
            for label in labels
        ]
        ax.bar(np.arange(len(labels)) + index * width - 0.4 + width / 2, values, width=width * 0.9,
               color=CONDITION_COLORS[condition], edgecolor=INK, linewidth=0.6,
               label=CONDITION_SHORT_LABELS[condition], zorder=2)
    ax.axhline(1.0, color=INK, linewidth=0.9, linestyle="--", zorder=3)
    ax.text(-0.45, 1.002, "everything the data determines", fontsize=5.8, color=INK, va="bottom")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=5.6)
    ax.set_ylabel("$R^2$ / noise ceiling")
    ax.set_ylim(0.9, 1.06)
    ax.legend(fontsize=6, loc="lower left", ncol=3, columnspacing=0.7, handletextpad=0.4)
    nice_axis(ax)
    _panel(ax, "A")

    ax = axes[1]
    interactive = fit[fit["condition"] == "face_interactive"]
    values = [float(interactive[interactive["label"] == label]["r2_vs_ceiling"].mean()) for label in labels]
    reference = values[labels.index(baseline)] if baseline in labels else values[0]
    colours = [IMPROVED_COLOR if v > reference + 1e-4 else (WORSE_COLOR if v < reference - 1e-4 else BASELINE_COLOR)
               for v in values]
    order = np.argsort(values)
    ax.barh(np.arange(len(labels)), [values[i] for i in order],
            color=[colours[i] for i in order], edgecolor=INK, linewidth=0.6, height=0.72)
    ax.axvline(reference, color=BASELINE_COLOR, linewidth=1.1, linestyle="--")
    ax.text(reference, len(labels) - 0.4, " baseline", fontsize=5.8, color=MUTED_INK, va="top")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([labels[i] for i in order], fontsize=5.6)
    ax.set_xlabel("interactive face:  $R^2$ / ceiling")
    ax.set_xlim(min(values) * 0.995, max(max(values), reference) * 1.005)
    nice_axis(ax)
    _panel(ax, "B")

    fig.tight_layout()
    return fig


def plot_spectral_recovery(
    recovery: pd.DataFrame,
    *,
    watch_band: str = "10-20 Hz",
    figsize: tuple[float, float] = (7.6, 2.9),
):
    """How much of the observed power each variant reproduces, by band and condition."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1.0]})
    labels = list(dict.fromkeys(recovery["label"]))
    bands = list(dict.fromkeys(recovery["band"]))

    ax = axes[0]
    interactive = recovery[recovery["condition"] == "face_interactive"]
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
    for label, colour in zip(labels, colours):
        values = [float(interactive[(interactive["label"] == label) & (interactive["band"] == b)]["power_ratio"].mean())
                  for b in bands]
        ax.plot(np.arange(len(bands)), values, marker="o", markersize=4, linewidth=1.2, color=colour, label=label)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands, fontsize=6.5)
    ax.set_ylabel("power reproduced / observed")
    ax.set_ylim(0, 1.15)
    ax.set_title("interactive face", fontsize=7.5, pad=5)
    ax.legend(fontsize=5.2, loc="lower left", ncol=2, columnspacing=0.6, handletextpad=0.4)
    nice_axis(ax)
    _panel(ax, "A")

    ax = axes[1]
    watch = recovery[recovery["band"] == watch_band]
    conditions = [c for c in CONDITION_ORDER if c in set(watch["condition"])]
    width = 0.8 / max(len(conditions), 1)
    for index, condition in enumerate(conditions):
        values = [float(watch[(watch["label"] == label) & (watch["condition"] == condition)]["power_ratio"].mean())
                  for label in labels]
        ax.bar(np.arange(len(labels)) + index * width - 0.4 + width / 2, values, width=width * 0.9,
               color=CONDITION_COLORS[condition], edgecolor=INK, linewidth=0.6,
               label=CONDITION_SHORT_LABELS[condition], zorder=2)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=5.6)
    ax.set_ylabel(f"power reproduced, {watch_band}")
    ax.set_ylim(0, 1.15)
    nice_axis(ax)
    _panel(ax, "B")

    fig.tight_layout()
    return fig


def plot_seed_agreement(
    agreement: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.6, 2.9),
):
    """Agreement between independently seeded fits, on three summaries.

    Outputs is the weakest test: two networks can produce the same trajectories by
    different internal means. Current magnitude and drive geometry are what a
    circuit-level claim would rest on, and they are the ones that have historically
    disagreed.
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(dict.fromkeys(agreement["label"]))
    features = list(dict.fromkeys(agreement["feature"]))
    palette = {"output trajectories": NEUTRAL_FILL,
               "inter-regional current magnitude": "#5fa8d3",
               "latent drive geometry": "#1b4965"}
    width = 0.8 / max(len(features), 1)
    for index, feature in enumerate(features):
        values = [float(agreement[(agreement["label"] == label) & (agreement["feature"] == feature)]["mean_agreement"].mean())
                  for label in labels]
        ax.bar(np.arange(len(labels)) + index * width - 0.4 + width / 2, values, width=width * 0.9,
               color=palette.get(feature, NEUTRAL_FILL), edgecolor=INK, linewidth=0.6, label=feature, zorder=2)
    ax.axhline(1.0, color=MUTED_INK, linewidth=0.8, linestyle=":")
    ax.text(-0.45, 1.005, "identical across seeds", fontsize=6, color=MUTED_INK, va="bottom")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("mean pairwise agreement")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=6, loc="lower right", ncol=1)
    nice_axis(ax)
    fig.tight_layout()
    return fig


def plot_variant_tradeoff(
    scores: pd.DataFrame,
    *,
    baseline: str | None = None,
    recovery_column: str | None = None,
    figsize: tuple[float, float] = (7.6, 2.9),
):
    """The trade the selection has to make, drawn rather than collapsed into one number."""
    if recovery_column is None:
        candidates = [c for c in scores.columns if c.startswith("power_recovered_")]
        recovery_column = candidates[0] if candidates else "interactive_r2_vs_ceiling"
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, x_column, x_label in (
        (axes[0], recovery_column, "fast-structure recovery\n(interactive face)"),
        (axes[1], "interactive_r2_vs_ceiling", "interactive face $R^2$ / ceiling"),
    ):
        for _, row in scores.iterrows():
            is_baseline = str(row["label"]) == str(baseline)
            ax.scatter(row[x_column], row["seed_agreement_geometry"], s=54 if is_baseline else 38,
                       color=BASELINE_COLOR if is_baseline else IMPROVED_COLOR,
                       marker="s" if is_baseline else "o",
                       edgecolor=INK, linewidth=0.5, zorder=3)
            ax.annotate(str(row["label"]), (row[x_column], row["seed_agreement_geometry"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=5.0, color=MUTED_INK)
        ax.set_xlabel(x_label)
        ax.set_ylabel("seed agreement\n(latent drive geometry)")
        nice_axis(ax)
    _panel(axes[0], "A")
    _panel(axes[1], "B")
    fig.tight_layout()
    return fig


__all__ = [
    "plot_ceiling_relative_fit",
    "plot_loss_trajectories",
    "plot_seed_agreement",
    "plot_spectral_recovery",
    "plot_variant_tradeoff",
]
