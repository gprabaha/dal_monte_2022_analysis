"""Figures shared by the constraint sweeps.

The question these sweeps ask is not "which model fits best" -- with parameters of the
same order as the dataset, many do -- but "which constraints can the data tolerate". The
panels are built for that: fit is always drawn against the measured noise ceiling and
next to the parameter count, and the gallery exists so a visual check can be made across
a whole sweep rather than one model at a time.
"""

from __future__ import annotations

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

CONVERGED_COLOR = "#2a6f4e"
FAILED_COLOR = "#c1121f"


def _panel(ax, letter: str) -> None:
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def _log_y(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())


def plot_sweep_loss_trajectories(
    histories: Mapping[str, Sequence[pd.DataFrame]],
    *,
    convergence: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (7.6, 4.6),
    n_columns: int = 3,
):
    """Every variant's optimisation history, all seeds overlaid, one panel each.

    Read this before any fit comparison. A constraint that merely makes the model harder
    to optimise looks the same in the final loss as a constraint the data cannot
    tolerate, and only the trajectory separates them.
    """
    labels = list(histories)
    n_columns = min(n_columns, max(len(labels), 1))
    n_rows = int(np.ceil(len(labels) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    status = convergence.set_index("label") if convergence is not None else None

    for index, label in enumerate(labels):
        ax = axes[index // n_columns][index % n_columns]
        for history in histories[label]:
            values = np.clip(history["loss"].to_numpy(dtype=float), 1e-12, None)
            ax.plot(np.arange(1, len(values) + 1), values, linewidth=0.7, color=INK, alpha=0.75)
        title = label
        colour = INK
        if status is not None and label in status.index:
            row = status.loc[label]
            passed = int(row["n_converged"]) == int(row["n_seeds"])
            colour = CONVERGED_COLOR if passed else FAILED_COLOR
            title = f"{label}  ({int(row['n_converged'])}/{int(row['n_seeds'])} converged)"
        ax.set_title(title, fontsize=6.6, pad=4, color=colour)
        nice_axis(ax)
        _log_y(ax)
        if index % n_columns == 0:
            ax.set_ylabel("loss")
        if index // n_columns == n_rows - 1:
            ax.set_xlabel("iteration")
    for index in range(len(labels), n_rows * n_columns):
        axes[index // n_columns][index % n_columns].axis("off")
    fig.tight_layout()
    return fig


def plot_fit_versus_constraint(
    fit: pd.DataFrame,
    parameters: pd.DataFrame,
    *,
    order: Sequence[str] | None = None,
    x_label: str = "variant",
    figsize: tuple[float, float] = (7.6, 2.9),
):
    """Fit against the ceiling, and the parameter count that bought it.

    ``parameters`` needs ``label``, ``total`` and ``target_numbers``. The right-hand panel
    is the one that decides whether a good fit is informative: a model with as many free
    parameters as the data has numbers can fit almost anything.
    """
    labels = list(order) if order is not None else list(dict.fromkeys(fit["label"]))
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    conditions = [c for c in CONDITION_ORDER if c in set(fit["condition"])]
    for condition in conditions:
        block = fit[fit["condition"] == condition]
        values = [float(block[block["label"] == label]["r2_vs_ceiling"].mean()) for label in labels]
        spread = [float(block[block["label"] == label]["r2_vs_ceiling"].std()) for label in labels]
        ax.errorbar(np.arange(len(labels)), values, yerr=spread, marker="o", markersize=4,
                    linewidth=1.2, capsize=2, color=CONDITION_COLORS[condition],
                    label=CONDITION_SHORT_LABELS[condition])
    ax.axhline(1.0, color=INK, linewidth=0.9, linestyle="--")
    ax.text(-0.4, 1.004, "the noise ceiling", fontsize=5.8, color=INK, va="bottom")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
    ax.set_xlabel(x_label)
    ax.set_ylabel("$R^2$ / noise ceiling")
    ax.legend(fontsize=6, loc="lower right", ncol=1)
    nice_axis(ax)
    _panel(ax, "A")

    ax = axes[1]
    counts = parameters.set_index("label")
    totals = [float(counts.loc[label, "total"]) for label in labels if label in counts.index]
    present = [label for label in labels if label in counts.index]
    fit_means = [float(fit[fit["label"] == label]["r2_vs_ceiling"].mean()) for label in present]
    target_numbers = float(counts["target_numbers"].iloc[0]) if len(counts) else np.nan
    ax.scatter(totals, fit_means, s=42, color=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE, linewidth=0.8, zorder=3)
    for label, x, y in zip(present, totals, fit_means):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=5.6, color=MUTED_INK)
    if np.isfinite(target_numbers):
        ax.axvline(target_numbers, color=FAILED_COLOR, linewidth=1.1, linestyle="--")
        ax.text(target_numbers, ax.get_ylim()[0], " as many parameters\n as data points",
                fontsize=5.8, color=FAILED_COLOR, va="bottom", ha="left")
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("mean $R^2$ / ceiling")
    nice_axis(ax)
    _panel(ax, "B")

    fig.tight_layout()
    return fig


def plot_fit_gallery(
    traces: pd.DataFrame,
    *,
    order: Sequence[str] | None = None,
    title: str | None = None,
    panel_height: float = 0.95,
    figure_width: float = 7.8,
):
    """One row per model, one column per condition — a whole sweep on one page.

    Visual verification is the check that decides whether a fit is real, and it does not
    scale: nobody looks at every trace of every model. Every row here shows the *same*
    components of the *same* region so the rows are comparable, at a size chosen for
    scanning rather than for reading detail. Whichever model the sweep selects then gets
    the full-detail trace plots.
    """
    labels = list(order) if order is not None else list(dict.fromkeys(traces["label"]))
    conditions = [c for c in CONDITION_ORDER if c in set(traces["condition"])]
    fig, axes = plt.subplots(
        len(labels), len(conditions),
        figsize=(figure_width, panel_height * len(labels) + 0.9),
        squeeze=False, sharex=True,
    )
    indices = sorted(traces["index"].unique())

    for row, label in enumerate(labels):
        for column, condition in enumerate(conditions):
            ax = axes[row][column]
            block = traces[(traces["label"] == label) & (traces["condition"] == condition)]
            for depth, index in enumerate(indices):
                panel = block[block["index"] == index].sort_values("time_s")
                alpha = 1.0 - 0.3 * depth
                ax.plot(panel["time_s"], panel["observed"], color=CONDITION_COLORS[condition],
                        linewidth=0.9, alpha=alpha, zorder=3)
                ax.plot(panel["time_s"], panel["predicted"], color=INK, linewidth=0.7,
                        linestyle="--", alpha=alpha, zorder=4)
            ax.axvline(0.0, color=MUTED_INK, linewidth=0.5, linestyle=":", zorder=1)
            ax.set_yticks([])
            if row == 0:
                ax.set_title(CONDITION_SHORT_LABELS[condition], fontsize=7, pad=4)
            if column == 0:
                ax.set_ylabel(label, fontsize=6.2, rotation=0, ha="right", va="center", labelpad=6)
            if row == len(labels) - 1:
                ax.set_xlabel("time from fixation (s)", fontsize=6.5)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(length=2, labelsize=6)

    handles = [
        plt.Line2D([], [], color=INK, linewidth=1.0, label="observed"),
        plt.Line2D([], [], color=INK, linewidth=0.8, linestyle="--", label="mRNN"),
    ]
    fig.legend(handles=handles, fontsize=6.5, ncol=2, loc="upper right",
               bbox_to_anchor=(0.99, 1.0), frameon=False)
    if title:
        fig.suptitle(title, fontsize=8.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_seed_agreement_by_variant(
    agreement: pd.DataFrame,
    *,
    order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (7.6, 2.8),
):
    """Agreement between independently seeded fits of each variant.

    A constraint that makes the solution more identifiable should raise this. It is the
    axis on which a restricted model can beat an unrestricted one, and therefore the
    reason for running these sweeps at all.
    """
    labels = list(order) if order is not None else list(dict.fromkeys(agreement["label"]))
    features = list(dict.fromkeys(agreement["feature"]))
    palette = {"output trajectories": NEUTRAL_FILL,
               "inter-regional current magnitude": "#5fa8d3",
               "latent drive geometry": "#1b4965"}
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8 / max(len(features), 1)
    for index, feature in enumerate(features):
        block = agreement[agreement["feature"] == feature].set_index("label")
        values = [float(block.loc[label, "mean_agreement"]) if label in block.index else np.nan
                  for label in labels]
        ax.bar(np.arange(len(labels)) + index * width - 0.4 + width / 2, values, width=width * 0.9,
               color=palette.get(feature, NEUTRAL_FILL), edgecolor=INK, linewidth=0.6,
               label=feature, zorder=2)
    ax.axhline(1.0, color=MUTED_INK, linewidth=0.8, linestyle=":")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.2)
    ax.set_ylabel("mean pairwise agreement")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=6, loc="lower right")
    nice_axis(ax)
    fig.tight_layout()
    return fig


def plot_sparsity_interaction(
    fit: pd.DataFrame,
    sparsity: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.6, 2.9),
):
    """Whether the tolerable within-region sparsity depends on the inter-regional rank.

    ``fit`` and ``sparsity`` both need ``rank`` and ``l1`` columns. Sequential selection --
    fix the rank, then sweep the penalty -- would be valid only if the two did not interact.
    They plausibly do: with a wide inter-regional channel the model can route around its
    within-region blocks and discard them for free, while a narrow one has to keep them.
    The right-hand panel is what shows whether the penalty is producing sparsity or an
    ablation.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ranks = sorted(fit["rank"].dropna().unique())
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(ranks), 1)))

    ax = axes[0]
    for rank, colour in zip(ranks, colours):
        block = fit[fit["rank"] == rank].groupby("l1")["r2_vs_ceiling"].mean().sort_index()
        ax.plot(block.index.to_numpy() + 1e-6, block.to_numpy(), marker="o", markersize=4,
                linewidth=1.2, color=colour, label=f"rank {rank:g}" if np.isfinite(rank) else "dense")
    ax.axhline(1.0, color=INK, linewidth=0.9, linestyle="--")
    ax.text(1.2e-6, 1.002, "the noise ceiling", fontsize=5.8, color=INK, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("within-region $L_1$ scale  (leftmost point is 0)")
    ax.set_ylabel("$R^2$ / noise ceiling")
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    ax.set_title("does a narrower channel need its internal blocks?", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _panel(ax, "A")

    ax = axes[1]
    for rank, colour in zip(ranks, colours):
        block = sparsity[sparsity["rank"] == rank].groupby("l1")["within_to_cross_norm"].mean().sort_index()
        ax.plot(block.index.to_numpy() + 1e-6, block.to_numpy(), marker="o", markersize=4,
                linewidth=1.2, color=colour)
    ax.axhline(1.0, color=MUTED_INK, linewidth=0.8, linestyle=":")
    ax.text(1.2e-6, 1.02, "as strong as the unpenalised blocks", fontsize=5.8, color=MUTED_INK, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("within-region $L_1$ scale")
    ax.set_ylabel("within-block norm /\ncross-block norm")
    ax.set_title("sparsity, or ablation?", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))
    _panel(ax, "B")

    fig.tight_layout()
    return fig


def plot_matched_budget_prongs(
    budget: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.8, 2.9),
):
    """Fit and seed agreement against parameter budget, one curve per constraint family.

    ``budget`` needs ``prong``, ``budget`` (free parameters per constrained block),
    ``fit_vs_ceiling`` and ``agreement``; ``dense_fit`` and ``dense_agreement`` are read
    from the first row if present and drawn as reference lines.

    The x axis is the parameter budget rather than the constraint's own units, which is
    what makes the families comparable: a rank-5 block and a 25%-dense block cost the same
    and differ only in how the budget is spent. Two panels rather than one because fit and
    agreement play different roles -- fit is a constraint the model has to satisfy, and
    agreement is the quantity being maximised among the models that do.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    prongs = list(dict.fromkeys(budget["prong"]))
    colours = plt.cm.viridis(np.linspace(0.1, 0.8, max(len(prongs), 1)))

    for ax, column, label, title in (
        (axes[0], "fit_vs_ceiling", "$R^2$ / noise ceiling", "does the constraint cost fit?"),
        (axes[1], "agreement", "inter-seed agreement", "does it make the solution unique?"),
    ):
        if column not in budget.columns:
            ax.set_axis_off()
            continue
        for prong, colour in zip(prongs, colours):
            block = budget[budget["prong"] == prong].sort_values("budget")
            ax.plot(block["budget"].to_numpy(), block[column].to_numpy(), marker="o",
                    markersize=4, linewidth=1.2, color=colour, label=str(prong))
        reference = f"dense_{'fit' if column == 'fit_vs_ceiling' else 'agreement'}"
        if reference in budget.columns and np.isfinite(budget[reference].iloc[0]):
            ax.axhline(float(budget[reference].iloc[0]), color=MUTED_INK,
                       linewidth=0.9, linestyle="--")
            ax.text(budget["budget"].min(), float(budget[reference].iloc[0]),
                    " dense, unconstrained", fontsize=5.8, color=MUTED_INK, va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("free parameters per constrained block")
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=7, color=MUTED_INK, pad=5)
        nice_axis(ax)
    axes[0].legend(fontsize=6, loc="lower right")
    _panel(axes[0], "A")
    _panel(axes[1], "B")
    fig.tight_layout()
    return fig


__all__ = [
    "plot_fit_gallery",
    "plot_matched_budget_prongs",
    "plot_sparsity_interaction",
    "plot_fit_versus_constraint",
    "plot_seed_agreement_by_variant",
    "plot_sweep_loss_trajectories",
]
