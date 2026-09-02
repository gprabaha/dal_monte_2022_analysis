"""Figures for the fixation mRNN cross-run synthesis.

Every function takes the tidy tables built by
``dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_synthesis`` and returns a
matplotlib figure. Nothing here computes a statistic; the numbers are decided in the
analysis module so the figure and the prose in the notebook cannot disagree.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    NEUTRAL_EDGE,
    NEUTRAL_FILL,
    REGION_COLORS,
    nice_axis,
    region_label,
)

#: Regions in the order the modelling code uses them.
MODEL_REGION_ORDER: tuple[str, ...] = ("ofc", "bla", "dmpfc", "accg")

ARCHITECTURE_LABELS: dict[str, str] = {
    "full": "Full",
    "cross_region_with_self_diagonal": "Cross-region\n+ self-diagonal",
    "within_region": "Within-region\nonly",
}
ARCHITECTURE_COLORS: dict[str, str] = {
    "full": "#1b4965",
    "cross_region_with_self_diagonal": "#5fa8d3",
    "within_region": "#c1121f",
}

CONFOUNDED_GREY = "#b0b0b0"


def _annotate_panel(ax, letter: str) -> None:
    ax.text(
        -0.16,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )


# --------------------------------------------------------------------------------------
# Figure 1 — the model and the run inventory
# --------------------------------------------------------------------------------------


def plot_model_schematic(ax, *, bottleneck_rank: int = 3) -> None:
    """Draw the four-region mRNN: condition input, region blocks, low-rank coupling, readouts."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    positions = {"ofc": (3.1, 4.5), "bla": (6.0, 4.5), "dmpfc": (3.1, 1.9), "accg": (6.0, 1.9)}
    box_w, box_h = 1.5, 1.1

    for region, (x, y) in positions.items():
        ax.add_patch(
            Rectangle(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                facecolor=REGION_COLORS[region],
                edgecolor=INK,
                linewidth=0.9,
                alpha=0.85,
                zorder=3,
            )
        )
        ax.text(x, y + 0.12, region_label(region), ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)
        ax.text(x, y - 0.24, "50 units", ha="center", va="center", fontsize=6.5, color="white", zorder=4)

    # Inter-regional coupling, drawn once per unordered pair as a double-headed arrow.
    seen: set[frozenset[str]] = set()
    for a in positions:
        for b in positions:
            if a == b or frozenset((a, b)) in seen:
                continue
            seen.add(frozenset((a, b)))
            ax.add_patch(
                FancyArrowPatch(
                    positions[a],
                    positions[b],
                    arrowstyle="<|-|>",
                    mutation_scale=6,
                    color=MUTED_INK,
                    linewidth=0.9,
                    shrinkA=26,
                    shrinkB=26,
                    zorder=2,
                )
            )
    ax.text(
        4.55,
        0.55,
        f"every region-to-region block is $L R$ with rank {bottleneck_rank}",
        ha="center",
        va="center",
        fontsize=6.8,
        color=INK,
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=MUTED_INK, linewidth=0.6),
    )

    # Condition input on the left.
    for index, condition in enumerate(CONDITION_ORDER):
        y = 4.6 - 1.15 * index
        ax.add_patch(
            Rectangle(
                (0.15, y - 0.2),
                0.95,
                0.4,
                facecolor=CONDITION_COLORS[condition],
                edgecolor=INK,
                linewidth=0.7,
                zorder=3,
            )
        )
        ax.text(0.625, y, CONDITION_SHORT_LABELS[condition], ha="center", va="center", fontsize=6, color="white", zorder=4)
    ax.text(0.625, 5.35, "condition\none-hot", ha="center", va="center", fontsize=7, color=INK)
    ax.annotate("", xy=(2.3, 3.2), xytext=(1.2, 3.2), arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.0))

    # Readouts on the right.
    ax.annotate("", xy=(8.4, 3.2), xytext=(6.85, 3.2), arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.0))
    ax.add_patch(Rectangle((8.45, 2.35), 1.35, 1.7, facecolor=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE, linewidth=0.9, zorder=3))
    ax.text(9.13, 3.45, "42 PCs", ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold", zorder=4)
    ax.text(9.13, 2.95, "per region", ha="center", va="center", fontsize=6.5, color=INK, zorder=4)
    ax.text(9.13, 2.05, "3 × 100 bins\n−500…+500 ms", ha="center", va="top", fontsize=6.2, color=MUTED_INK)
    ax.text(5.0, 6.05, "Elman mRNN — one block per region, linear readout to that region's PCs",
            ha="center", va="center", fontsize=7.5, color=INK)


def plot_run_inventory(
    family_summary: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 5.4),
):
    """Figure 1: what the model is, and the 60 experiments that were run on it."""
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.32, wspace=0.62)

    ax_schematic = fig.add_subplot(grid[0, :])
    plot_model_schematic(ax_schematic)
    _annotate_panel(ax_schematic, "A")

    ax_counts = fig.add_subplot(grid[1, 0])
    block = family_summary.sort_values("n_runs")
    ax_counts.barh(block["family"], block["n_runs"], color=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE, linewidth=0.8, height=0.7)
    for y, value in enumerate(block["n_runs"]):
        ax_counts.text(value + max(block["n_runs"]) * 0.02, y, str(int(value)), va="center", fontsize=6.5, color=INK)
    ax_counts.set_xlabel("fitted networks")
    ax_counts.set_xlim(0, max(block["n_runs"]) * 1.18)
    nice_axis(ax_counts)
    ax_counts.yaxis.set_major_locator(plt.matplotlib.ticker.FixedLocator(range(len(block))))
    ax_counts.set_yticklabels(block["family"], fontsize=6.5)
    _annotate_panel(ax_counts, "B")

    ax_epochs = fig.add_subplot(grid[1, 1])
    families = list(block["family"])
    rng = np.random.default_rng(0)
    for y, family in enumerate(families):
        values = inventory.loc[inventory["family"] == family, "epochs"].dropna().astype(float)
        if not len(values):
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax_epochs.scatter(values, np.full(len(values), y) + jitter, s=7, color=NEUTRAL_EDGE, alpha=0.55, linewidths=0)
    ax_epochs.set_ylim(ax_counts.get_ylim())
    ax_epochs.set_xscale("log")
    ax_epochs.set_xlabel("training iterations")
    ax_epochs.set_title("families differ in training length,\nso they are not directly comparable", fontsize=7, color=MUTED_INK, pad=6)
    nice_axis(ax_epochs)
    ax_epochs.yaxis.set_major_locator(plt.matplotlib.ticker.FixedLocator(range(len(families))))
    ax_epochs.set_yticklabels([""] * len(families))
    _annotate_panel(ax_epochs, "C")

    return fig


# --------------------------------------------------------------------------------------
# Figure 2 — capacity versus training length
# --------------------------------------------------------------------------------------


def plot_capacity_vs_learning_rate(
    sweep: pd.DataFrame,
    fifty_unit_runs: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.2, 2.8),
):
    """Figure 2: the hidden-unit sweep is confounded by learning rate.

    ``sweep`` is the hidden-unit sweep with ``hidden_units``, ``lr``, ``final_iteration``
    and ``mean_r2``. ``fifty_unit_runs`` is every 50-unit-per-region run in the tree with
    ``lr``, ``final_iteration`` and ``mean_r2``.

    Panel A plots the sweep as it was run. The learning rate was lowered as the models
    got wider, so width and step size vary together and the curve cannot be read as a
    capacity curve. Panel B holds width fixed at 50 units and shows what the learning
    rate alone does.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    diverged_marker = dict(marker="X", s=52, color="#c1121f", linewidths=0.6, edgecolor=INK, zorder=5)

    ax = axes[0]
    block = sweep.sort_values("hidden_units")
    finite = block[np.isfinite(block["mean_r2"]) & (block["mean_r2"] > 0)]
    lost = block[~(np.isfinite(block["mean_r2"]) & (block["mean_r2"] > 0))]
    ax.plot(finite["hidden_units"], finite["mean_r2"], color=NEUTRAL_EDGE, linewidth=1.0, zorder=2)
    scatter = ax.scatter(
        finite["hidden_units"], finite["mean_r2"], c=np.log10(finite["lr"]),
        cmap="plasma", s=46, edgecolor=INK, linewidth=0.5, zorder=3, vmin=-4.05, vmax=-2.95,
    )
    if len(lost):
        ax.scatter(lost["hidden_units"], np.full(len(lost), 0.02), **diverged_marker)
        ax.text(float(lost["hidden_units"].iloc[0]) + 1.5, 0.03, "diverged", fontsize=6, color="#c1121f", va="bottom")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.045, ticks=[-4, -3.5, -3])
    colorbar.ax.set_yticklabels(["1e−4", "3e−4", "1e−3"], fontsize=6)
    colorbar.set_label("learning rate", fontsize=6.5)
    for _, row in finite.iterrows():
        ax.annotate(
            f"{int(row['final_iteration']) // 1000}k",
            (row["hidden_units"], row["mean_r2"]),
            textcoords="offset points", xytext=(0, -11), ha="center", fontsize=5.4, color=MUTED_INK,
        )
    ax.set_xlabel("hidden units per region")
    ax.set_ylabel("mean PC $R^2$")
    ax.set_ylim(0, 1.02)
    ax.set_title("the sweep as it was run: the widest models\nalso got the smallest learning rate", fontsize=6.6, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _annotate_panel(ax, "A")

    ax = axes[1]
    block = fifty_unit_runs.copy()
    finite = block[np.isfinite(block["mean_r2"]) & (block["mean_r2"] > 0)]
    lost = block[~(np.isfinite(block["mean_r2"]) & (block["mean_r2"] > 0))]
    scatter = ax.scatter(
        finite["lr"], finite["mean_r2"], c=np.log10(finite["final_iteration"]),
        cmap="cividis", s=34, edgecolor=INK, linewidth=0.4, zorder=3,
    )
    if len(lost):
        ax.scatter(lost["lr"], np.full(len(lost), 0.52), **diverged_marker)
        ax.text(float(lost["lr"].min()) * 0.85, 0.545, "diverged", fontsize=6, color="#c1121f", ha="right", va="bottom")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.045)
    colorbar.set_label("log$_{10}$ iterations", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("mean PC $R^2$")
    ax.set_ylim(0.5, 1.03)
    ax.axvline(1e-3, color=MUTED_INK, linewidth=0.8, linestyle=":", zorder=1)
    ax.text(1.08e-3, 1.0, "1e−3", fontsize=6, color=MUTED_INK, ha="left", va="top")
    ax.set_title("50 units per region, every run in the tree", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _annotate_panel(ax, "B")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Figure 3 — architecture, and which condition needs the coupling
# --------------------------------------------------------------------------------------


def plot_architecture_comparison(
    cells: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.2, 2.8),
):
    """Figure 3: inter-regional coupling is required, and interactive face needs it most.

    ``cells`` is the region × condition table with one column per architecture.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.25, 1.0]})
    architectures = [a for a in ARCHITECTURE_LABELS if a in cells.columns]

    ax = axes[0]
    x = np.arange(len(architectures))
    for _, row in cells.iterrows():
        ax.plot(
            x,
            [row[a] for a in architectures],
            color=CONDITION_COLORS[row["condition"]],
            marker="o",
            markersize=3,
            linewidth=0.8,
            alpha=0.75,
            zorder=2,
        )
    ax.plot(x, [cells[a].median() for a in architectures], color=INK, marker="s", markersize=5, linewidth=1.8, zorder=4, label="median")
    ax.set_xticks(x)
    ax.set_xticklabels([ARCHITECTURE_LABELS[a] for a in architectures], fontsize=6.5)
    ax.set_ylabel("PC $R^2$")
    ax.set_ylim(0.70, 1.005)
    handles = [Line2D([], [], color=CONDITION_COLORS[c], marker="o", markersize=3, linewidth=0.9, label=CONDITION_SHORT_LABELS[c]) for c in CONDITION_ORDER]
    ax.legend(handles=handles + [Line2D([], [], color=INK, marker="s", markersize=4, linewidth=1.6, label="median")], fontsize=6, loc="lower left")
    ax.set_title("one line per region × condition cell", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _annotate_panel(ax, "A")

    ax = axes[1]
    cost = cells.assign(cost=cells["full"] - cells["within_region"])
    order = list(CONDITION_ORDER)
    medians = [cost.loc[cost["condition"] == c, "cost"].median() for c in order]
    ax.bar(
        np.arange(len(order)),
        medians,
        color=[CONDITION_COLORS[c] for c in order],
        edgecolor=INK,
        linewidth=0.8,
        width=0.62,
        zorder=2,
    )
    rng = np.random.default_rng(1)
    for index, condition in enumerate(order):
        values = cost.loc[cost["condition"] == condition, "cost"].to_numpy()
        ax.scatter(index + rng.uniform(-0.12, 0.12, size=len(values)), values, s=12, color=INK, alpha=0.75, zorder=4, linewidths=0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([CONDITION_SHORT_LABELS[c] for c in order], fontsize=6.5)
    ax.set_ylabel("$R^2$ lost when inter-regional\ncoupling is removed")
    ax.set_title("median cost, dots = the four regions", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _annotate_panel(ax, "B")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Figure 4 — the inter-regional bottleneck
# --------------------------------------------------------------------------------------


def plot_bottleneck_sweep(
    sweep: pd.DataFrame,
    *,
    matched_epochs: int = 100000,
    figsize: tuple[float, float] = (7.2, 2.7),
):
    """Figure 4: rank 3 is the knee, and the higher ranks are not readable.

    ``sweep`` needs ``bottleneck_dim``, ``epochs``, ``region``, ``space`` and ``r2``.
    Panel A is the epoch-matched block on its own scale — the actual result. Panel B is
    the full sweep, with the shorter-trained ranks in grey, which is why the sweep
    cannot currently be read end to end.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    pc = sweep[sweep["space"] == "pc"]
    matched = pc[pc["epochs"] == matched_epochs]

    ax = axes[0]
    for region in MODEL_REGION_ORDER:
        block = matched[matched["region"] == region].sort_values("bottleneck_dim")
        ax.plot(block["bottleneck_dim"], block["r2"], marker="o", markersize=4.5,
                linewidth=1.4, color=REGION_COLORS[region], label=region_label(region), zorder=3)
    ax.axvline(3, color=INK, linewidth=0.9, linestyle=":", zorder=1)
    ax.annotate("knee", xy=(3, ax.get_ylim()[1]), xytext=(0, -2), textcoords="offset points",
                fontsize=6.5, color=INK, va="top", ha="center")
    ax.set_xticks(sorted(matched["bottleneck_dim"].unique()))
    ax.set_xlabel("inter-regional bottleneck rank")
    ax.set_ylabel("PC $R^2$")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    ax.set_title(f"epoch-matched block ({matched_epochs // 1000}k iterations)", fontsize=7.5, pad=5)
    nice_axis(ax)
    _annotate_panel(ax, "A")

    ax = axes[1]
    for region in MODEL_REGION_ORDER:
        block = pc[pc["region"] == region].sort_values("bottleneck_dim")
        short = block[block["epochs"] != matched_epochs]
        ax.plot(block["bottleneck_dim"], block["r2"], marker="o", markersize=3.2,
                linewidth=1.0, color=REGION_COLORS[region], alpha=0.85, zorder=3)
        ax.scatter(short["bottleneck_dim"], short["r2"], s=44, marker="o",
                   facecolor="none", edgecolor=CONFOUNDED_GREY, linewidth=1.4, zorder=4)
    if len(pc[pc["epochs"] != matched_epochs]):
        span_lo = float(pc.loc[pc["epochs"] != matched_epochs, "bottleneck_dim"].min())
        ax.axvspan(span_lo * 0.72, float(pc["bottleneck_dim"].max()) * 1.4, color=CONFOUNDED_GREY, alpha=0.2, zorder=0)
        ax.text(span_lo * 1.75, 0.762, "trained 50k, not 100k — not comparable", fontsize=6.2,
                color=MUTED_INK, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 30])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "30"])
    ax.set_xlabel("inter-regional bottleneck rank")
    ax.set_ylabel("PC $R^2$")
    ax.set_ylim(0.74, 1.005)
    ax.set_title("full sweep", fontsize=7.5, pad=5)
    nice_axis(ax)
    _annotate_panel(ax, "B")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Figure 5 — fits replicate, circuits do not
# --------------------------------------------------------------------------------------


def plot_reproducibility(
    health: pd.DataFrame,
    consistency: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.2, 2.7),
):
    """Figure 5: a quarter of seeds fail, and the survivors do not agree on the circuit."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.0, 1.25]})

    ax = axes[0]
    block = health.sort_values("min_loss").reset_index(drop=True)
    good = ~block["likely_bad_run"].astype(bool)
    ax.scatter(np.arange(len(block))[good], block.loc[good, "min_loss"], s=11, color=NEUTRAL_EDGE, linewidths=0, label=f"healthy ({int(good.sum())})")
    ax.scatter(np.arange(len(block))[~good], block.loc[~good, "min_loss"], s=13, color="#c1121f", marker="x", linewidths=0.9, label=f"flagged ({int((~good).sum())})")
    ax.set_yscale("log")
    ax.set_xlabel("initialization (sorted)")
    ax.set_ylabel("best training loss")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("100 seeds, identical settings", fontsize=7.5, pad=5)
    nice_axis(ax)
    # nice_axis installs a linear MaxNLocator; restore decade ticks for the log axis.
    ax.yaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.LogFormatterSciNotation())
    _annotate_panel(ax, "A")

    ax = axes[1]
    families = [
        "inter-regional current magnitude",
        "relative source contribution",
        "latent geometry (drive RDM / RSA)",
    ]
    short = ["current\nmagnitude", "relative source\ncontribution", "latent geometry\n(RSA)"]
    ensembles = list(dict.fromkeys(consistency["ensemble"]))
    width = 0.8 / max(len(ensembles), 1)
    palette = ["#9ec5ab", "#2a6f4e"]
    for index, ensemble in enumerate(ensembles):
        block = consistency[consistency["ensemble"] == ensemble].set_index("metric_family")
        values = [float(block.loc[f, "consistency_score"]) if f in block.index else np.nan for f in families]
        ax.bar(
            np.arange(len(families)) + index * width - 0.4 + width / 2,
            values,
            width=width * 0.9,
            color=palette[index % len(palette)],
            edgecolor=INK,
            linewidth=0.8,
            label=ensemble,
            zorder=2,
        )
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax.text(-0.55, 1.01, "identical dynamics", fontsize=6, color=MUTED_INK, va="bottom", ha="left")
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(short, fontsize=6.3)
    ax.set_ylabel("cross-seed consistency")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.6, 2.4)
    ax.legend(fontsize=5.8, loc="upper center", ncol=1, bbox_to_anchor=(0.62, 1.0))
    ax.text(0.62, 0.70, "the two ensembles also differ in\ntraining length — this comparison\nis confounded", transform=ax.transAxes,
            fontsize=5.8, color="#c1121f", ha="center", va="top")
    nice_axis(ax)
    _annotate_panel(ax, "B")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Figure 6 — what survives the ensemble: condition modulation of drive
# --------------------------------------------------------------------------------------


def _stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot_condition_invariant(
    contribution: pd.DataFrame,
    contrasts: pd.DataFrame,
    pathway_contrasts: pd.DataFrame,
    *,
    seed_column: str = "init_idx",
    figsize: tuple[float, float] = (7.2, 2.9),
):
    """Figure 6: drive is allocated near-uniformly, but interactive face reweights it."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.5, 1.0]})

    per_seed = (
        contribution.groupby([seed_column, "active_condition", "source_region"])["relative_projection"]
        .mean()
        .reset_index()
    )

    ax = axes[0]
    rng = np.random.default_rng(2)
    for region_index, source in enumerate(MODEL_REGION_ORDER):
        for condition_index, condition in enumerate(CONDITION_ORDER):
            values = per_seed.loc[
                (per_seed["source_region"] == source) & (per_seed["active_condition"] == condition),
                "relative_projection",
            ].to_numpy()
            x = region_index + (condition_index - 1) * 0.26
            ax.scatter(
                x + rng.uniform(-0.055, 0.055, size=len(values)),
                values,
                s=4,
                color=CONDITION_COLORS[condition],
                alpha=0.35,
                linewidths=0,
                zorder=2,
            )
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            ax.errorbar(x, mean, yerr=sem, fmt="_", markersize=13, color=INK, linewidth=1.2, capsize=2.5, zorder=5)
            ax.scatter([x], [mean], s=26, color=CONDITION_COLORS[condition], edgecolor=INK, linewidth=0.7, zorder=6)
    ax.axhline(0.25, color=MUTED_INK, linewidth=0.8, linestyle="--", zorder=1)
    ax.text(-0.47, 0.252, "uniform (1/4)", fontsize=6, color=MUTED_INK, va="bottom", ha="left")
    ax.set_xticks(range(len(MODEL_REGION_ORDER)))
    ax.set_xticklabels([region_label(r) for r in MODEL_REGION_ORDER])
    ax.set_xlabel("source region")
    ax.set_ylabel("share of target's drive")
    ax.set_xlim(-0.5, 3.55)

    key = contrasts[
        (contrasts["condition_a"] == "face_interactive") & (contrasts["condition_b"] == "face_non_interactive")
    ].set_index("source_region")
    top = ax.get_ylim()[1]
    for region_index, source in enumerate(MODEL_REGION_ORDER):
        if source not in key.index:
            continue
        ax.text(region_index, top * 0.995, _stars(float(key.loc[source, "p_holm"])), ha="center", va="top", fontsize=7, color=INK)
    ax.set_title("dots = one fitted network; stars = interactive vs non-interactive face", fontsize=6.8, color=MUTED_INK, pad=5)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=4, color=CONDITION_COLORS[c], label=CONDITION_SHORT_LABELS[c]) for c in CONDITION_ORDER]
    ax.legend(handles=handles, fontsize=6, loc="lower left", ncol=3, columnspacing=0.8, handletextpad=0.3)
    nice_axis(ax)
    _annotate_panel(ax, "A")

    ax = axes[1]
    block = pathway_contrasts.copy()
    y = np.arange(len(block))
    ax.barh(y, block["difference"], color=CONDITION_COLORS["face_interactive"], edgecolor=INK, linewidth=0.8, height=0.62, zorder=2)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=3)
    for index, row in block.reset_index(drop=True).iterrows():
        offset = -0.004 if row["difference"] < 0 else 0.004
        ax.text(row["difference"] + offset, index, _stars(float(row["p_holm"])), va="center", ha="right" if row["difference"] < 0 else "left", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("bla", "BLA").replace("ofc", "OFC").replace("dmpfc", "dmPFC").replace("accg", "ACCg") for p in block["pathway"]], fontsize=6.5)
    ax.set_xlabel("Δ drive share\n(interactive − non-interactive face)")
    ax.set_title("every BLA pathway shifts the same way", fontsize=7, color=MUTED_INK, pad=5)
    ax.margins(x=0.22)
    nice_axis(ax)
    _annotate_panel(ax, "B")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Figure 7 — what does not survive: the identity of the channels
# --------------------------------------------------------------------------------------


def plot_channel_identity(
    pathway_alignment: pd.DataFrame,
    identity_test: pd.DataFrame,
    null_draws: np.ndarray,
    *,
    figsize: tuple[float, float] = (7.2, 2.7),
):
    """Figure 7: the rank-3 channels themselves are not recovered across seeds."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    null_mean = float(np.mean(null_draws))
    null_hi = float(np.percentile(null_draws, 95))
    ax.axhspan(float(np.percentile(null_draws, 5)), null_hi, color=CONFOUNDED_GREY, alpha=0.35, zorder=0)
    ax.axhline(null_mean, color=MUTED_INK, linewidth=0.9, linestyle="--", zorder=1)
    ax.text(-0.45, null_mean, "chance", fontsize=6.2, color=MUTED_INK, va="bottom", ha="left")

    kinds = list(dict.fromkeys(pathway_alignment["kind"]))
    palette = {"write": "#1b4965", "read": "#e07a5f"}
    width = 0.8 / len(kinds)
    pathways = list(dict.fromkeys(pathway_alignment["pathway"]))
    for kind_index, kind in enumerate(kinds):
        block = pathway_alignment[pathway_alignment["kind"] == kind].set_index("pathway").loc[pathways]
        x = np.arange(len(pathways)) + kind_index * width - 0.4 + width / 2
        ax.bar(x, block["mean_alignment"], yerr=block["sd_alignment"], width=width * 0.88,
               color=palette.get(kind, NEUTRAL_FILL), edgecolor=INK, linewidth=0.6,
               error_kw=dict(elinewidth=0.7, capsize=1.5), label=f"{kind} subspace", zorder=2)
    ax.set_xticks(np.arange(len(pathways)))
    ax.set_xticklabels([p.replace("bla", "BLA").replace("ofc", "OFC").replace("dmpfc", "dmPFC").replace("accg", "ACCg") for p in pathways], rotation=60, ha="right", fontsize=5.6)
    ax.set_ylabel("cross-seed subspace alignment")
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.set_title("grey band = 5th–95th percentile of random 3-D subspaces", fontsize=6.6, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _annotate_panel(ax, "A")

    ax = axes[1]
    x = np.arange(len(identity_test))
    ax.bar(x - 0.19, identity_test["mean_matched"], width=0.36, color="#1b4965", edgecolor=INK, linewidth=0.8, label="same pathway,\nother seed", zorder=2)
    ax.bar(x + 0.19, identity_test["mean_mismatched"], width=0.36, color=NEUTRAL_FILL, edgecolor=INK, linewidth=0.8, label="different pathway,\nother seed", zorder=2)
    ax.axhline(null_mean, color=MUTED_INK, linewidth=0.9, linestyle="--", zorder=1)
    for index, row in identity_test.reset_index(drop=True).iterrows():
        top = max(float(row["mean_matched"]), float(row["mean_mismatched"])) + 0.035
        ax.text(index, top, f"p = {float(row['p_one_sided']):.2f}", ha="center", va="bottom", fontsize=6.3, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(identity_test["kind"], fontsize=7)
    ax.set_ylabel("subspace alignment")
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=5.8, loc="upper left", ncol=2, columnspacing=0.6, handletextpad=0.4)
    ax.set_title("a channel is no more like itself than\nlike a different pathway", fontsize=6.8, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _annotate_panel(ax, "B")

    fig.tight_layout()
    return fig


__all__ = [
    "ARCHITECTURE_COLORS",
    "ARCHITECTURE_LABELS",
    "MODEL_REGION_ORDER",
    "plot_architecture_comparison",
    "plot_bottleneck_sweep",
    "plot_capacity_vs_learning_rate",
    "plot_channel_identity",
    "plot_condition_invariant",
    "plot_model_schematic",
    "plot_reproducibility",
    "plot_run_inventory",
]
