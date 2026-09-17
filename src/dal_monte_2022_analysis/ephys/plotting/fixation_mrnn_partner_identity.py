"""Figures for the partner-identity acid test: the 4x4 matrix and its self-vs-cross reading.

Conventions follow the chapter's (``fixation_mrnn_chapter.py``): bars with a capless SEM
error bar and every fit as a dot for grouped comparisons, no horizontal grid lines, only
significant contrasts marked as a plain bar with stars above, self drawn hollow and cross
filled in one shared colour per fixation type (mirroring dense/constrained elsewhere).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.plotting.fixation_mrnn_audit import _panel
from dal_monte_2022_analysis.ephys.plotting.fixation_mrnn_chapter import (
    CONDITION_OFFSETS,
    contrasts_to_marks,
    draw_bars,
    mark_contrasts,
)
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    INK,
    REGION_COLORS,
    REGION_LABELS,
    nice_axis,
)

ARM_COLORS = {"self": "#8a8a8a", "cross": "#2a6f8f"}
DISPLAY_REGION_ORDER = ("bla", "accg", "dmpfc", "ofc")


def plot_partner_identity_matrix(
    matrix_fit: pd.DataFrame,
    *,
    value: str = "r2_vs_ceiling",
    regions: Sequence[str] = DISPLAY_REGION_ORDER,
    figsize: tuple[float, float] = (3.6, 3.2),
):
    """The 4x4 matrix: each scored region's fit, by partner, mean over conditions and fits.

    Rows are the scored region, columns the partner; the diagonal (self-pairs) is outlined.
    """
    per_cell = matrix_fit.groupby(["scored_region", "partner_region"])[value].mean().unstack("partner_region")
    per_cell = per_cell.reindex(index=regions, columns=regions)
    fig, ax = plt.subplots(figsize=figsize)
    vals = per_cell.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap="cividis", aspect="auto")
    for i in range(len(regions)):
        for j in range(len(regions)):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if v < np.nanmedian(vals) else INK)
            if i == j:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor="#c0392b", lw=1.8, zorder=5))
    ax.set_xticks(range(len(regions))); ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
    ax.set_yticks(range(len(regions))); ax.set_yticklabels([REGION_LABELS.get(r, r) for r in regions])
    ax.set_xlabel("partner region")
    ax.set_ylabel("scored region")
    ax.set_title("$R^2$ / ceiling, mean over fixation types and fits\n(red outline: self-pair)", fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=2)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    fig.tight_layout()
    return fig


def plot_self_vs_cross_by_region(
    self_vs_cross_long: pd.DataFrame,
    tests: pd.DataFrame | None,
    *,
    value: str = "value",
    regions: Sequence[str] = DISPLAY_REGION_ORDER,
    figsize: tuple[float, float] = (4.4, 2.6),
):
    """One region per group, self (white) and cross (filled) bars, every seed a dot.

    ``tests`` is a paired-contrasts table keyed by ``scored_region``, ``a``, ``b``
    (:func:`fixation_mrnn_chapter.paired_contrasts` on the melted self/cross table); only
    significant region-level contrasts are marked.
    """
    fig, ax = plt.subplots(figsize=figsize)
    entries = []
    for j, region in enumerate(regions):
        for arm, dx in (("self", -0.16), ("cross", 0.16)):
            block = self_vs_cross_long[(self_vs_cross_long["scored_region"] == region) & (self_vs_cross_long["arm"] == arm)]
            entries.append({"x": j + dx, "values": block[value].to_numpy(float), "color": REGION_COLORS.get(region, INK),
                            "filled": arm == "cross"})
    draw_bars(ax, entries, width=0.28)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
    ax.set_ylabel("$R^2$ / ceiling")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, lw=0.8),
               plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor="none")]
    ax.legend(handles, ["self (paired with an independent half of itself)", "cross (mean over the three other areas)"],
             loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, handlelength=1.0, frameon=False)
    nice_axis(ax)
    if tests is not None and not tests.empty:
        index = {r: k for k, r in enumerate(regions)}
        offset = {"self": -0.16, "cross": 0.16}
        marks = contrasts_to_marks(tests, lambda row, side: index[row["scored_region"]] + offset[row[side]])
        top = float(self_vs_cross_long[value].max())
        mark_contrasts(ax, marks, top=top)
    fig.tight_layout()
    return fig


def plot_self_vs_cross_by_condition(
    matrix_fit: pd.DataFrame,
    tests: pd.DataFrame | None,
    *,
    value: str = "r2_vs_ceiling",
    figsize: tuple[float, float] = (4.2, 2.6),
):
    """One fixation type per group, self (white) and cross (filled) bars, regions and seeds pooled into one dot per fit.

    ``tests`` is a paired-contrasts table keyed by ``condition``, ``a``, ``b`` (self vs
    cross, paired by seed within each fixation type); only significant contrasts marked.
    """
    conditions = [c for c in CONDITION_ORDER if c in set(matrix_fit["condition"])]
    per_fit = matrix_fit.groupby(["arm", "seed", "condition"])[value].mean().reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    entries = []
    for j, condition in enumerate(conditions):
        for arm, dx in (("self", -0.16), ("cross", 0.16)):
            block = per_fit[(per_fit["arm"] == arm) & (per_fit["condition"] == condition)]
            entries.append({"x": j + dx, "values": block[value].to_numpy(float), "color": CONDITION_COLORS[condition],
                            "filled": arm == "cross"})
    draw_bars(ax, entries, width=0.28)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_SHORT_LABELS[c] for c in conditions])
    ax.set_ylabel("$R^2$ / ceiling\n(regions pooled)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, lw=0.8),
               plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor="none")]
    ax.legend(handles, ["self", "cross"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5,
             handlelength=1.0, frameon=False)
    nice_axis(ax)
    if tests is not None and not tests.empty:
        index = {c: k for k, c in enumerate(conditions)}
        offset = {"self": -0.16, "cross": 0.16}
        marks = contrasts_to_marks(tests, lambda row, side: index[row["condition"]] + offset[row[side]])
        mark_contrasts(ax, marks, top=float(per_fit[value].max()))
    fig.tight_layout()
    return fig


def plot_bottleneck_self_vs_cross(
    bottleneck_fit: pd.DataFrame,
    tests: pd.DataFrame | None = None,
    *,
    value: str = "r2_vs_ceiling",
    dims: Sequence[int] | None = None,
    figsize: tuple[float, float] = (4.2, 2.6),
):
    """Self vs. cross, one group per cross-region bottleneck rank, regions and fixation types pooled.

    ``bottleneck_fit`` is one or more cross-region bottleneck ranks' :func:`matrix_fit_table`
    outputs concatenated with a ``cross_bottleneck_dim`` column. ``tests`` is a paired-contrasts
    table keyed by ``cross_bottleneck_dim`` (self vs. cross, paired by seed within each rank).
    """
    dims = sorted(bottleneck_fit["cross_bottleneck_dim"].unique()) if dims is None else list(dims)
    per_seed = bottleneck_fit.groupby(["cross_bottleneck_dim", "arm", "seed"])[value].mean().reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    entries = []
    for j, dim in enumerate(dims):
        for arm, dx in (("self", -0.16), ("cross", 0.16)):
            block = per_seed[(per_seed["cross_bottleneck_dim"] == dim) & (per_seed["arm"] == arm)]
            entries.append({"x": j + dx, "values": block[value].to_numpy(float), "color": ARM_COLORS[arm],
                            "filled": arm == "cross"})
    draw_bars(ax, entries, width=0.28)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"rank {d}" for d in dims])
    ax.set_xlabel("cross-region bottleneck rank (within-region rank fixed at 1)")
    ax.set_ylabel("$R^2$ / ceiling")
    ax.set_ylim(bottom=0.0)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, lw=0.8),
               plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor="none")]
    ax.legend(handles, ["self", "cross"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5,
             handlelength=1.0, frameon=False)
    nice_axis(ax)
    if tests is not None and not tests.empty:
        index = {d: k for k, d in enumerate(dims)}
        offset = {"self": -0.16, "cross": 0.16}
        marks = contrasts_to_marks(tests, lambda row, side: index[row["cross_bottleneck_dim"]] + offset[row[side]])
        mark_contrasts(ax, marks, top=float(per_seed[value].max()))
    fig.tight_layout()
    return fig


def plot_bottleneck_by_region(
    bottleneck_fit: pd.DataFrame,
    *,
    value: str = "r2_vs_ceiling",
    regions: Sequence[str] = DISPLAY_REGION_ORDER,
    dims: Sequence[int] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """One panel per cross-region bottleneck rank: self (hollow) vs. cross (filled) bars by region.

    Shared y-axis across panels, so whether a region's self-vs-cross gap opens up (or flips
    sign) as the channel tightens is a direct visual comparison, not a table lookup.
    """
    dims = sorted(bottleneck_fit["cross_bottleneck_dim"].unique()) if dims is None else list(dims)
    per_seed = bottleneck_fit.groupby(["cross_bottleneck_dim", "scored_region", "arm", "seed"])[value].mean().reset_index()
    figsize = figsize or (2.6 * len(dims) + 0.6, 2.6)
    fig, axes = plt.subplots(1, len(dims), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    top = float(per_seed[value].max())
    for ax, dim in zip(axes, dims):
        entries = []
        for j, region in enumerate(regions):
            for arm, dx in (("self", -0.16), ("cross", 0.16)):
                block = per_seed[(per_seed["cross_bottleneck_dim"] == dim) & (per_seed["scored_region"] == region)
                                 & (per_seed["arm"] == arm)]
                entries.append({"x": j + dx, "values": block[value].to_numpy(float), "color": REGION_COLORS.get(region, INK),
                                "filled": arm == "cross"})
        draw_bars(ax, entries, width=0.28)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
        ax.set_title(f"cross-region rank {dim}", fontsize=7.5)
        ax.set_ylim(0.0, top * 1.05)
        nice_axis(ax)
    axes[0].set_ylabel("$R^2$ / ceiling")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, lw=0.8),
               plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor="none")]
    fig.legend(handles, ["self", "cross"], loc="upper center", ncol=2, fontsize=6.5, handlelength=1.0, frameon=False,
              bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    return fig


def plot_bottleneck_by_condition(
    bottleneck_fit: pd.DataFrame,
    *,
    value: str = "r2_vs_ceiling",
    dims: Sequence[int] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """One panel per cross-region bottleneck rank: self (hollow) vs. cross (filled) bars by fixation type."""
    dims = sorted(bottleneck_fit["cross_bottleneck_dim"].unique()) if dims is None else list(dims)
    conditions = [c for c in CONDITION_ORDER if c in set(bottleneck_fit["condition"])]
    per_seed = bottleneck_fit.groupby(["cross_bottleneck_dim", "condition", "arm", "seed"])[value].mean().reset_index()
    figsize = figsize or (2.4 * len(dims) + 0.6, 2.6)
    fig, axes = plt.subplots(1, len(dims), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    top = float(per_seed[value].max())
    for ax, dim in zip(axes, dims):
        entries = []
        for j, condition in enumerate(conditions):
            for arm, dx in (("self", -0.16), ("cross", 0.16)):
                block = per_seed[(per_seed["cross_bottleneck_dim"] == dim) & (per_seed["condition"] == condition)
                                 & (per_seed["arm"] == arm)]
                entries.append({"x": j + dx, "values": block[value].to_numpy(float), "color": CONDITION_COLORS[condition],
                                "filled": arm == "cross"})
        draw_bars(ax, entries, width=0.28)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([CONDITION_SHORT_LABELS[c] for c in conditions])
        ax.set_title(f"cross-region rank {dim}", fontsize=7.5)
        ax.set_ylim(0.0, top * 1.05)
        nice_axis(ax)
    axes[0].set_ylabel("$R^2$ / ceiling")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, lw=0.8),
               plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor="none")]
    fig.legend(handles, ["self", "cross"], loc="upper center", ncol=2, fontsize=6.5, handlelength=1.0, frameon=False,
              bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    return fig


def plot_bottleneck_cost_vs_asymmetry(
    cost: pd.DataFrame,
    *,
    dims: Sequence[int] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Cost (``1 - r2_vs_ceiling``) against region-pair size ratio, one panel per cross-region bottleneck rank.

    Self-pairs (ratio 1.0, hollow) sit as the matched anchor beside the six cross-pairs
    (filled, one point per seed); a rising trend would say a tight channel costs
    size-mismatched pairs more than matched ones.
    """
    dims = sorted(cost["cross_bottleneck_dim"].unique()) if dims is None else list(dims)
    figsize = figsize or (2.6 * len(dims) + 0.6, 2.6)
    fig, axes = plt.subplots(1, len(dims), figsize=figsize, sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    top = float(cost["cost"].max())
    for ax, dim in zip(axes, dims):
        block = cost[cost["cross_bottleneck_dim"] == dim]
        for arm, color, filled in (("self", ARM_COLORS["self"], False), ("cross", ARM_COLORS["cross"], True)):
            sub = block[block["arm"] == arm]
            ax.scatter(sub["size_ratio"], sub["cost"], s=22, facecolor=color if filled else "white",
                      edgecolor=color, lw=1.1, zorder=3, label=arm)
        r = (np.corrcoef(block["size_ratio"], block["cost"])[0, 1] if block["size_ratio"].nunique() > 1 else float("nan"))
        ax.set_title(f"cross-region rank {dim}  (r = {r:.2f})", fontsize=7.5)
        ax.set_xlabel("region-pair size ratio")
        ax.set_ylim(0.0, top * 1.05)
        nice_axis(ax)
    axes[0].set_ylabel("cost  ($1 - R^2$ / ceiling)")
    axes[0].legend(loc="upper left", fontsize=6.5, frameon=False, handletextpad=0.4)
    fig.tight_layout()
    return fig


__all__ = [
    "ARM_COLORS", "DISPLAY_REGION_ORDER", "plot_bottleneck_by_condition", "plot_bottleneck_by_region",
    "plot_bottleneck_cost_vs_asymmetry", "plot_bottleneck_self_vs_cross", "plot_partner_identity_matrix",
    "plot_self_vs_cross_by_condition", "plot_self_vs_cross_by_region",
]
