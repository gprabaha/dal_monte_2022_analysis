"""Figures for the constrained ensemble (task 03): fit against dense, fixation-type contrasts, dynamics, lesions.

Two conventions run through every figure here. **Every seed is a mark**: with ten fits the
question is always whether an effect is present in all of them, and a mean with an error
bar cannot say. **Two arms, one encoding**: the dense ensemble is drawn hollow and the
constrained ensemble filled, in the condition's colour, so a panel with both arms reads
without a second legend.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from dal_monte_2022_analysis.ephys.plotting.fixation_mrnn_audit import GRID_KW, _panel
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    NEUTRAL_EDGE,
    REGION_COLORS,
    REGION_LABELS,
    nice_axis,
)

ARM_ORDER = ("dense", "constrained")
ARM_COLORS = {"dense": "#8a8a8a", "constrained": "#2a6f8f"}
PAIR_ORDER = ("FI–FN", "FI–OBJ", "FN–OBJ")
PAIR_COLORS = {"FI–FN": "#7b3f9e", "FI–OBJ": "#b64198", "FN–OBJ": "#97ca3d"}


def _arm_marker_kw(arm: str, color: str) -> dict:
    """Hollow for dense, filled for constrained, in the given colour."""
    if arm == "dense":
        return dict(facecolor="white", edgecolor=color, linewidth=0.9)
    return dict(facecolor=color, edgecolor="white", linewidth=0.4)


def _arms_present(frame: pd.DataFrame) -> list[str]:
    if "arm" not in frame.columns:
        return ["constrained"]
    return [a for a in ARM_ORDER if a in set(frame["arm"])] + [a for a in frame["arm"].unique() if a not in ARM_ORDER]


def _conditions_present(frame: pd.DataFrame) -> list[str]:
    return [c for c in CONDITION_ORDER if c in set(frame["condition"])]


# ======================================================================================
# 1. Fit against the dense network
# ======================================================================================


def plot_fit_cells_by_arm(fit: pd.DataFrame, *, bar: float = 0.98, figsize: tuple[float, float] = (7.4, 4.2)):
    """Every region x condition cell in each arm, and the cost (dense minus constrained), as three heatmaps."""
    arms = _arms_present(fit)
    regions = [r for r in REGION_COLORS if r in set(fit["region"])]
    conditions = _conditions_present(fit)
    table = fit.groupby(["arm", "region", "condition"])["r2_vs_ceiling"].mean()
    grids = {arm: np.array([[table.get((arm, r, c), np.nan) for c in conditions] for r in regions]) for arm in arms}
    panels = list(arms)
    if "dense" in grids and "constrained" in grids:
        grids["cost"] = grids["dense"] - grids["constrained"]
        panels.append("cost")
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    axes = np.atleast_1d(axes)
    lo = float(np.nanmin([grids[a] for a in arms])); hi = float(np.nanmax([grids[a] for a in arms]))
    for ax, name, letter in zip(axes, panels, "abc"):
        grid = grids[name]
        if name == "cost":
            image = ax.imshow(grid, cmap="Reds", vmin=0, vmax=float(np.nanmax(grid)), aspect="auto")
        else:
            image = ax.imshow(grid, cmap="cividis", vmin=lo, vmax=hi, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isfinite(v):
                    mid = (0.5 * float(np.nanmax(grid))) if name == "cost" else lo + 0.5 * (hi - lo)
                    ax.text(j, i, f"{v:.4f}" if name == "cost" else f"{v:.3f}", ha="center", va="center", fontsize=6.4,
                            color="white" if v > mid and name == "cost" else ("white" if v < mid and name != "cost" else INK),
                            fontweight="bold" if name != "cost" and v >= bar else "normal")
        ax.set_xticks(range(len(conditions))); ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions], fontsize=6.6)
        ax.set_yticks(range(len(regions))); ax.set_yticklabels([REGION_LABELS.get(r, r) for r in regions], fontsize=6.8)
        ax.set_title({"cost": "cost = dense − constrained"}.get(name, f"{name}  ($R^2$ / ceiling, bold ≥ {bar})"), fontsize=7.4)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        _panel(ax, letter)
    fig.tight_layout()
    return fig


def plot_condition_cost(fit: pd.DataFrame, cost: pd.DataFrame, *, bar: float = 0.98, figsize: tuple[float, float] = (7.6, 2.9)):
    """The test of "is one fixation type hurt more": fit per condition in each arm, the cost, and the unexplained ratio.

    (a) ceiling-relative R² per fixation type (regions pooled per seed), dense hollow and
    constrained filled, every seed a mark. (b) the cost of the constraint per fixation type
    with its bootstrap CI over seeds. (c) the ratio of unexplained variance, constrained
    over dense: scale-free, so the fair comparison between a fixation type with half the
    variance and the other two.
    """
    conditions = _conditions_present(fit)
    arms = _arms_present(fit)
    per = fit.groupby(["arm", "seed", "condition"])["r2_vs_ceiling"].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
    rng = np.random.default_rng(0)

    ax = axes[0]
    ax.grid(axis="y", **GRID_KW)
    offsets = {arm: (-0.18 if arm == "dense" else 0.18) for arm in arms}
    for k, condition in enumerate(conditions):
        color = CONDITION_COLORS.get(condition, INK)
        for arm in arms:
            sub = per[(per["arm"] == arm) & (per["condition"] == condition)]
            x = k + offsets[arm] + rng.uniform(-0.06, 0.06, size=len(sub))
            ax.scatter(x, sub["r2_vs_ceiling"], s=16, zorder=3, **_arm_marker_kw(arm, color))
            ax.hlines(sub["r2_vs_ceiling"].mean(), k + offsets[arm] - 0.13, k + offsets[arm] + 0.13, color=color, linewidth=1.6, zorder=4)
    ax.axhline(bar, color=INK, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(range(len(conditions))); ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions], fontsize=7)
    ax.set_ylabel("$R^2$ / ceiling  (mean over regions)")
    ax.scatter([], [], s=16, label="dense", **_arm_marker_kw("dense", INK))
    ax.scatter([], [], s=16, label="constrained", **_arm_marker_kw("constrained", INK))
    ax.legend(frameon=False, fontsize=6.4, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    nice_axis(ax); _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="y", **GRID_KW)
    for k, condition in enumerate(conditions):
        row = cost[cost["condition"] == condition].iloc[0]
        color = CONDITION_COLORS.get(condition, INK)
        ax.errorbar(k, row["cost"], yerr=[[row["cost"] - row["cost_ci_low"]], [row["cost_ci_high"] - row["cost"]]],
                    fmt="o", color=color, markersize=5, capsize=3, linewidth=1.2, zorder=3)
    ax.axhline(0.0, color=INK, linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(conditions))); ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions], fontsize=7)
    ax.set_ylabel("cost of the constraint\n($\\Delta R^2$ / ceiling, 95% CI)")
    nice_axis(ax); _panel(ax, "b")

    ax = axes[2]
    ax.grid(axis="y", **GRID_KW)
    for k, condition in enumerate(conditions):
        row = cost[cost["condition"] == condition].iloc[0]
        color = CONDITION_COLORS.get(condition, INK)
        ax.errorbar(k, row["unexplained_ratio"],
                    yerr=[[row["unexplained_ratio"] - row["unexplained_ratio_ci_low"]], [row["unexplained_ratio_ci_high"] - row["unexplained_ratio"]]],
                    fmt="o", color=color, markersize=5, capsize=3, linewidth=1.2, zorder=3)
    ax.axhline(1.0, color=INK, linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(conditions))); ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions], fontsize=7)
    ax.set_ylabel("unexplained variance,\nconstrained / dense (95% CI)")
    nice_axis(ax); _panel(ax, "c")
    fig.tight_layout()
    return fig


def plot_loss_curves(histories: Mapping[str, Sequence[pd.DataFrame]], *, figsize: tuple[float, float] = (7.4, 2.7)):
    """Worst-cell and mean-cell unexplained fraction over training, every seed of every arm."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for ax, column, letter, label in zip(axes, ("worst_cell_unexplained", "mean_cell_unexplained"), "ab",
                                         ("worst region × condition cell", "mean over cells")):
        ax.grid(True, **GRID_KW)
        for arm, runs in histories.items():
            color = ARM_COLORS.get(arm, INK)
            for k, history in enumerate(runs):
                if column not in history.columns:
                    continue
                ax.plot(history["iteration"], history[column], color=color, linewidth=0.6, alpha=0.6,
                        label=arm if k == 0 else None, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("iteration"); ax.set_title(label, fontsize=7.5)
        nice_axis(ax); _panel(ax, letter)
    axes[0].set_ylabel("unexplained fraction of the cell's variance")
    axes[0].legend(frameon=False, fontsize=6.4)
    fig.tight_layout()
    return fig


def plot_reconstruction_overlay(traces: pd.DataFrame, *, arm: str = "constrained", title: str | None = None,
                                figsize: tuple[float, float] = (7.4, 4.6)):
    """One region's leading PCs: the target in black, every seed's reconstruction as a thin line."""
    conditions = _conditions_present(traces)
    indices = sorted(traces["index"].unique())
    color = ARM_COLORS.get(arm, INK)
    fig, axes = plt.subplots(len(indices), len(conditions), figsize=figsize, sharex=True, squeeze=False)
    for i, index in enumerate(indices):
        for j, condition in enumerate(conditions):
            ax = axes[i][j]
            block = traces[(traces["index"] == index) & (traces["condition"] == condition)]
            for _, seed_block in block.groupby("seed"):
                ax.plot(seed_block["time_s"], seed_block["predicted"], color=color, linewidth=0.55, alpha=0.5, zorder=2)
            first = block[block["seed"] == block["seed"].iloc[0]]
            ax.plot(first["time_s"], first["observed"], color=INK, linewidth=1.3, zorder=3)
            ax.axvline(0.0, color=MUTED_INK, linewidth=0.6, linestyle=":", zorder=1)
            if i == 0:
                ax.set_title(CONDITION_SHORT_LABELS.get(condition, condition), fontsize=8, color=CONDITION_COLORS.get(condition, INK))
            if j == 0:
                ax.set_ylabel(f"PC {index + 1}", fontsize=7)
            if i == len(indices) - 1:
                ax.set_xlabel("time from fixation (s)")
            nice_axis(ax)
    if title:
        fig.suptitle(title, fontsize=8, y=1.0)
    fig.tight_layout()
    return fig


# ======================================================================================
# 2. Fixation-type similarity
# ======================================================================================


def plot_similarity_by_property(
    similarity: pd.DataFrame,
    *,
    properties: Sequence[str],
    scope: str = "network",
    labels: Mapping[str, str] | None = None,
    figsize_per_panel: tuple[float, float] = (2.45, 2.3),
):
    """One panel per property: the three condition pairs on the x-axis, every seed a mark, both arms.

    The property is read as "which pair is closest". For ``kind == 'similarity'`` higher
    is closer; for ``kind == 'distance'`` lower is closer, and the panel says so.
    """
    labels = dict(labels or {})
    block = similarity[(similarity["scope"] == scope) & (similarity["property"].isin(properties))]
    arms = _arms_present(block)
    n = len(properties)
    n_cols = min(3, n); n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows), squeeze=False)
    rng = np.random.default_rng(3)
    offsets = {arm: (-0.17 if arm == "dense" else 0.17) for arm in arms} if len(arms) > 1 else {arms[0]: 0.0}
    for k, prop in enumerate(properties):
        ax = axes[k // n_cols][k % n_cols]
        sub = block[block["property"] == prop]
        ax.grid(axis="y", **GRID_KW)
        kind = str(sub["kind"].iloc[0]) if not sub.empty else "similarity"
        for p, pair in enumerate(PAIR_ORDER):
            color = PAIR_COLORS[pair]
            for arm in arms:
                chunk = sub[(sub["pair"] == pair) & ((sub["arm"] == arm) if "arm" in sub.columns else True)]
                x = p + offsets[arm] + rng.uniform(-0.05, 0.05, size=len(chunk))
                ax.scatter(x, chunk["value"], s=13, zorder=3, **_arm_marker_kw(arm, color))
                if not chunk.empty:
                    ax.hlines(chunk["value"].mean(), p + offsets[arm] - 0.12, p + offsets[arm] + 0.12, color=color, linewidth=1.5, zorder=4)
        if kind == "similarity":
            ax.axhline(0.0, color=INK, linewidth=0.7, zorder=2)
        ax.set_xticks(range(len(PAIR_ORDER))); ax.set_xticklabels(PAIR_ORDER, fontsize=7)
        ax.set_title(labels.get(prop, prop) + ("  (lower = closer)" if kind == "distance" else ""), fontsize=7.2)
        nice_axis(ax)
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")
    if len(arms) > 1:
        ax = axes[0][0]
        ax.scatter([], [], s=13, label="dense", **_arm_marker_kw("dense", INK))
        ax.scatter([], [], s=13, label="constrained", **_arm_marker_kw("constrained", INK))
        ax.legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()
    return fig


def plot_similarity_scoreboard(summary: pd.DataFrame, *, properties: Sequence[str], labels: Mapping[str, str] | None = None,
                               figsize: tuple[float, float] = (7.4, 3.4)):
    """For every property: how often the two faces are the closest pair, dense against constrained, target as the reference.

    Left: the fraction of fits in which FI–FN is the most similar (or least distant) pair;
    chance for three pairs is one third. Right: the mean value of each pair, so the size
    of the separation is visible next to its consistency.
    """
    labels = dict(labels or {})
    block = summary[summary["property"].isin(properties)].copy()
    block["order"] = block["property"].map({p: i for i, p in enumerate(properties)})
    block = block.sort_values("order")
    arms = [a for a in ARM_ORDER if a in set(block.get("arm", pd.Series(dtype=str)))] or ["constrained"]
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.0, 1.35]})

    ax = axes[0]
    ax.grid(axis="x", **GRID_KW)
    y_positions = {p: i for i, p in enumerate(properties)}
    for arm in arms:
        chunk = block[block["arm"] == arm] if "arm" in block.columns else block
        offset = -0.18 if arm == "dense" else 0.18
        ax.barh([y_positions[p] + offset for p in chunk["property"]], chunk["fi_fn_closest"], 0.34,
                color=ARM_COLORS.get(arm, INK), edgecolor=NEUTRAL_EDGE, linewidth=0.5, label=arm, zorder=3)
    ax.axvline(1 / 3, color=INK, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_yticks(range(len(properties))); ax.set_yticklabels([labels.get(p, p) for p in properties], fontsize=6.8)
    ax.invert_yaxis(); ax.set_xlim(0, 1.02)
    ax.set_xlabel("fraction of fits with FI–FN the closest pair")
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="x", **GRID_KW)
    for arm in arms:
        chunk = block[block["arm"] == arm] if "arm" in block.columns else block
        offset = -0.18 if arm == "dense" else 0.18
        for pair in PAIR_ORDER:
            if pair not in chunk.columns:
                continue
            ax.scatter(chunk[pair], [y_positions[p] + offset for p in chunk["property"]], s=18, zorder=3,
                       **_arm_marker_kw(arm, PAIR_COLORS[pair]))
    for pair in PAIR_ORDER:
        ax.scatter([], [], s=18, color=PAIR_COLORS[pair], label=pair)
    ax.set_yticks(range(len(properties))); ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("mean over fits  (similarity ↑ closer;  distance ↓ closer)")
    ax.legend(frameon=False, fontsize=6.4, loc="best")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _panel(ax, "b")
    fig.tight_layout()
    return fig


# ======================================================================================
# 3. Per-condition properties, two arms
# ======================================================================================


def plot_property_by_condition(
    long: pd.DataFrame,
    prop: str,
    *,
    ylabel: str = "",
    reference: float | None = None,
    ax=None,
    figsize: tuple[float, float] = (3.2, 2.6),
):
    """One property per condition, every seed a mark, dense hollow and constrained filled."""
    sub = long[long["property"] == prop] if "property" in long.columns else long
    conditions = _conditions_present(sub)
    arms = _arms_present(sub)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.grid(axis="y", **GRID_KW)
    rng = np.random.default_rng(4)
    offsets = {arm: (-0.18 if arm == "dense" else 0.18) for arm in arms} if len(arms) > 1 else {arms[0]: 0.0}
    for k, condition in enumerate(conditions):
        color = CONDITION_COLORS.get(condition, INK)
        for arm in arms:
            chunk = sub[(sub["condition"] == condition) & ((sub["arm"] == arm) if "arm" in sub.columns else True)]
            x = k + offsets[arm] + rng.uniform(-0.05, 0.05, size=len(chunk))
            ax.scatter(x, chunk["value"], s=14, zorder=3, **_arm_marker_kw(arm, color))
            if not chunk.empty:
                ax.hlines(chunk["value"].mean(), k + offsets[arm] - 0.13, k + offsets[arm] + 0.13, color=color, linewidth=1.5, zorder=4)
    if reference is not None:
        ax.axhline(reference, color=MUTED_INK, linewidth=0.7, linestyle=":", zorder=1)
    ax.set_xticks(range(len(conditions))); ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions], fontsize=7)
    ax.set_ylabel(ylabel or prop, fontsize=7)
    nice_axis(ax)
    if own:
        if len(arms) > 1:
            ax.scatter([], [], s=14, label="dense", **_arm_marker_kw("dense", INK))
            ax.scatter([], [], s=14, label="constrained", **_arm_marker_kw("constrained", INK))
            ax.legend(frameon=False, fontsize=6.2, loc="best")
        fig.tight_layout()
    return fig


def plot_property_grid(long: pd.DataFrame, properties: Sequence[str], *, labels: Mapping[str, str] | None = None,
                       references: Mapping[str, float] | None = None, n_cols: int = 3, panel: tuple[float, float] = (2.45, 2.2)):
    """Several per-condition properties as one grid of :func:`plot_property_by_condition` panels."""
    labels = dict(labels or {}); references = dict(references or {})
    n = len(properties); n_cols = min(n_cols, n); n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel[0] * n_cols, panel[1] * n_rows), squeeze=False)
    for k, prop in enumerate(properties):
        ax = axes[k // n_cols][k % n_cols]
        plot_property_by_condition(long, prop, ylabel=labels.get(prop, prop), reference=references.get(prop), ax=ax)
        _panel(ax, "abcdefghijkl"[k])
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")
    arms = _arms_present(long)
    if len(arms) > 1:
        ax = axes[0][0]
        ax.scatter([], [], s=14, label="dense", **_arm_marker_kw("dense", INK))
        ax.scatter([], [], s=14, label="constrained", **_arm_marker_kw("constrained", INK))
        ax.legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()
    return fig


def plot_region_property_by_condition(frame: pd.DataFrame, prop: str, *, ylabel: str = "", figsize: tuple[float, float] = (7.4, 2.4)):
    """One panel per region, property per condition, both arms, every seed a mark."""
    regions = [r for r in REGION_COLORS if r in set(frame["region"])]
    fig, axes = plt.subplots(1, len(regions), figsize=figsize, sharey=True, squeeze=False)
    for ax, region, letter in zip(axes[0], regions, "abcd"):
        block = frame[frame["region"] == region].rename(columns={prop: "value"})
        plot_property_by_condition(block, "value", ylabel=ylabel if region == regions[0] else "", ax=ax)
        ax.set_title(REGION_LABELS.get(region, region), fontsize=8, color=REGION_COLORS.get(region, INK))
        _panel(ax, letter)
    arms = _arms_present(frame)
    if len(arms) > 1:
        ax = axes[0][0]
        ax.scatter([], [], s=14, label="dense", **_arm_marker_kw("dense", INK))
        ax.scatter([], [], s=14, label="constrained", **_arm_marker_kw("constrained", INK))
        ax.legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()
    return fig


def plot_time_course_two_arms(frame: pd.DataFrame, value: str, *, ylabel: str = "", trim_bins: int = 2,
                              figsize: tuple[float, float] = (7.4, 2.5)):
    """A per-condition time course, one panel per condition, both arms, seeds thin and mean heavy."""
    conditions = _conditions_present(frame)
    arms = _arms_present(frame)
    start = sorted(frame["time_s"].unique())[trim_bins]
    block = frame[frame["time_s"] >= start]
    keys = ["arm", "seed", "condition", "time_s"] if "arm" in block.columns else ["seed", "condition", "time_s"]
    per_seed = block.groupby(keys)[value].mean().reset_index()
    fig, axes = plt.subplots(1, len(conditions), figsize=figsize, sharey=True, squeeze=False)
    for ax, condition in zip(axes[0], conditions):
        ax.grid(axis="y", **GRID_KW)
        for arm in arms:
            color = ARM_COLORS.get(arm, INK)
            sub = per_seed[(per_seed["condition"] == condition) & ((per_seed["arm"] == arm) if "arm" in per_seed.columns else True)]
            for _, trace in sub.groupby("seed"):
                ax.plot(trace["time_s"], trace[value], color=color, linewidth=0.5, alpha=0.35, zorder=2)
            mean = sub.groupby("time_s")[value].mean()
            ax.plot(mean.index, mean.values, color=color, linewidth=1.8, label=arm, zorder=4)
        ax.axvline(0.0, color=MUTED_INK, linewidth=0.6, linestyle=":", zorder=1)
        ax.set_title(CONDITION_SHORT_LABELS.get(condition, condition), fontsize=8, color=CONDITION_COLORS.get(condition, INK))
        ax.set_xlabel("time from fixation (s)")
        nice_axis(ax)
    axes[0][0].set_ylabel(ylabel or value)
    axes[0][0].legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()
    return fig


def plot_flow_matrix_two_arms(flow_share: pd.DataFrame, *, value: str = "share", figsize: tuple[float, float] = (7.4, 3.2)):
    """Mean pathway share matrix per arm (source rows, target columns), and its seed sd, as four heatmaps."""
    arms = _arms_present(flow_share)
    regions = [r for r in REGION_COLORS if r in set(flow_share["source"])]
    fig, axes = plt.subplots(2, len(arms), figsize=figsize, squeeze=False)
    for j, arm in enumerate(arms):
        block = flow_share[flow_share["arm"] == arm] if "arm" in flow_share.columns else flow_share
        per_seed = block.groupby(["seed", "source", "target"])[value].mean().unstack(["source", "target"])
        mean = per_seed.mean(axis=0); sd = per_seed.std(axis=0)
        for i, (stat, name, cmap) in enumerate(((mean, "mean over seeds", "cividis"), (sd, "sd over seeds", "Greys"))):
            ax = axes[i][j]
            grid = np.array([[stat.get((s, t), np.nan) for t in regions] for s in regions])
            image = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=float(np.nanmax(grid)))
            for a in range(len(regions)):
                for b in range(len(regions)):
                    if np.isfinite(grid[a, b]):
                        ax.text(b, a, f"{grid[a, b]:.2f}", ha="center", va="center", fontsize=6.4,
                                color="white" if grid[a, b] > 0.6 * float(np.nanmax(grid)) else INK)
            ax.set_xticks(range(len(regions))); ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions], fontsize=6.6)
            ax.set_yticks(range(len(regions))); ax.set_yticklabels([REGION_LABELS.get(r, r) for r in regions], fontsize=6.6)
            ax.set_title(f"{arm}: {name}", fontsize=7.4)
            if j == 0:
                ax.set_ylabel("from (source)")
            if i == 1:
                ax.set_xlabel("into (target)")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ======================================================================================
# 4. Dynamics
# ======================================================================================


def plot_flow_fields(fields: Mapping[str, object], *, title: str | None = None, figsize: tuple[float, float] = (7.6, 2.7),
                     arrow_stride: int = 1):
    """One panel per condition: the flow of that condition's map in the plane of its top two state axes.

    Background: speed of the flow ``‖F(h) − h‖`` (log colour). Arrows: the direction of
    the flow projected onto the plane (unit length; the background carries magnitude). Line: the trajectory the model actually runs, light to dark in time,
    circle at the start and square at the end. Stars: fixed points, filled if stable.
    """
    conditions = [c for c in CONDITION_ORDER if c in fields["conditions"]]
    xs, ys = fields["xs"], fields["ys"]
    gx, gy = np.meshgrid(xs, ys)
    speeds = np.concatenate([fields["conditions"][c]["speed"].ravel() for c in conditions])
    norm = LogNorm(vmin=max(float(speeds.min()), 1e-3), vmax=float(speeds.max()))
    fig, axes = plt.subplots(1, len(conditions), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for ax, condition in zip(axes[0], conditions):
        entry = fields["conditions"][condition]
        ax.pcolormesh(gx, gy, entry["speed"], cmap="Greys", norm=norm, shading="nearest", alpha=0.55, zorder=1)
        s = arrow_stride
        # Direction only: the background carries the magnitude, so arrows are unit length
        # (0.8 of the grid spacing) and the small, slow region near the trajectory stays legible.
        magnitude = np.hypot(entry["u"], entry["v"])
        spacing = 0.8 * float(min(xs[1] - xs[0], ys[1] - ys[0]))
        with np.errstate(invalid="ignore", divide="ignore"):
            un = np.where(magnitude > 0, entry["u"] / magnitude * spacing, 0.0)
            vn = np.where(magnitude > 0, entry["v"] / magnitude * spacing, 0.0)
        ax.quiver(gx[::s, ::s], gy[::s, ::s], un[::s, ::s], vn[::s, ::s], color="#4b4b4b",
                  angles="xy", scale_units="xy", scale=1.0, width=0.004, headwidth=3.5, alpha=0.8, zorder=2)
        trajectory = entry["trajectory"]
        color = CONDITION_COLORS.get(condition, INK)
        shades = plt.get_cmap("viridis")(np.linspace(0.15, 0.95, len(trajectory) - 1))
        for t in range(len(trajectory) - 1):
            ax.plot(trajectory[t: t + 2, 0], trajectory[t: t + 2, 1], color=shades[t], linewidth=1.6, zorder=4)
        ax.scatter(*trajectory[0], s=26, color=color, edgecolor="white", linewidth=0.6, zorder=5, marker="o")
        ax.scatter(*trajectory[-1], s=30, color=color, edgecolor="white", linewidth=0.6, zorder=5, marker="s")
        for point, stable in zip(entry["fixed_points"], entry["fixed_point_stable"]):
            ax.scatter(point[0], point[1], s=70, marker="*", facecolor=color if stable else "white", edgecolor=color,
                       linewidth=1.0, zorder=6)
        ax.set_title(CONDITION_SHORT_LABELS.get(condition, condition), fontsize=8, color=color)
        ax.set_xlabel(f"axis 1 ({fields['explained'][0]:.0%} of state variance)", fontsize=6.8)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
        ax.tick_params(labelsize=6.5)
    axes[0][0].set_ylabel(f"axis 2 ({fields['explained'][1]:.0%})", fontsize=6.8)
    if title:
        fig.suptitle(title, fontsize=8, y=1.0)
    fig.tight_layout()
    return fig


def plot_fixed_point_summary(summary: pd.DataFrame, *, figsize: tuple[float, float] = (7.4, 4.6)):
    """Per condition and arm, every seed a mark: how many fixed points, how close the trajectory ends to one, how stable it is."""
    panels = [
        ("n_fixed_points", "distinct fixed points found"),
        ("nearest_distance_to_trajectory_end", "distance, trajectory end → nearest\nfixed point (state extents)"),
        ("trajectory_end_speed", "speed at the trajectory's end\n(state units per bin)"),
        ("nearest_top_modulus", "largest |eigenvalue| at that fixed point"),
        ("nearest_n_expanding", "expanding modes at that fixed point"),
        ("nearest_readout_gap_at_end", "readout at fixed point vs target's last bin\n(target extents)"),
    ]
    long = summary.melt(id_vars=[c for c in ("arm", "seed", "condition") if c in summary.columns],
                        value_vars=[p for p, _ in panels], var_name="property", value_name="value")
    fig = plot_property_grid(long, [p for p, _ in panels], labels=dict(panels),
                             references={"nearest_top_modulus": 1.0}, n_cols=3, panel=(2.45, 2.25))
    fig.set_size_inches(*figsize)
    fig.tight_layout()
    return fig


def plot_jacobian_time(jac: pd.DataFrame, *, scope: str = "network", figsize: tuple[float, float] = (7.4, 2.6)):
    """Local linearisation along the trajectory: largest |eigenvalue| and expanding modes (network) or slowest timescale (region)."""
    block = jac[jac["scope"] == scope]
    arms = _arms_present(block)
    conditions = _conditions_present(block)
    second = "n_expanding" if scope == "network" else "slowest_timescale_s"
    second_label = "expanding modes" if scope == "network" else "slowest timescale (s)"
    fig, axes = plt.subplots(1, 2 if len(arms) == 1 else 2 * len(arms), figsize=figsize, sharex=True, squeeze=False)
    axes = axes[0]
    k = 0
    for arm in arms:
        sub = block[block["arm"] == arm] if "arm" in block.columns else block
        for column, label in ((f"top_modulus", "largest |λ| of the local Jacobian"), (second, second_label)):
            ax = axes[k]; k += 1
            ax.grid(axis="y", **GRID_KW)
            for condition in conditions:
                color = CONDITION_COLORS.get(condition, INK)
                chunk = sub[sub["condition"] == condition]
                for _, trace in chunk.groupby("seed"):
                    ax.plot(trace["time_s"], trace[column], color=color, linewidth=0.45, alpha=0.3, zorder=2)
                mean = chunk.groupby("time_s")[column].mean()
                ax.plot(mean.index, mean.values, color=color, linewidth=1.6, label=CONDITION_SHORT_LABELS.get(condition, condition), zorder=4)
            if column == "top_modulus":
                ax.axhline(1.0, color=INK, linewidth=0.7, linestyle="--", zorder=1)
            ax.axvline(0.0, color=MUTED_INK, linewidth=0.6, linestyle=":", zorder=1)
            ax.set_title(f"{arm}\n{label}", fontsize=7.0)
            ax.set_xlabel("time from fixation (s)")
            nice_axis(ax)
    axes[0].legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()
    return fig


def plot_region_timescales(jac: pd.DataFrame, *, figsize: tuple[float, float] = (7.4, 2.4)):
    """Each region's own local dynamics (its block Jacobian with input held fixed): largest |eigenvalue|, per condition, both arms."""
    block = jac[jac["scope"] != "network"].groupby([c for c in ("arm", "seed", "scope", "condition") if c in jac.columns])["top_modulus"].mean().reset_index()
    block = block.rename(columns={"scope": "region"})
    return plot_region_property_by_condition(block, "top_modulus", ylabel="largest |eigenvalue| of the region's\nown local Jacobian (mean over time)", figsize=figsize)


# ======================================================================================
# 5. Lesions
# ======================================================================================


def plot_pair_lesions(pairs: pd.DataFrame, *, value: str = "damage_share", figsize: tuple[float, float] = (7.6, 3.0)):
    """Bidirectional pair lesions: damage share per pair and condition, every seed a mark, one panel per arm.

    ``damage_share`` is the pair's share of the summed damage over all six pairs within a
    (seed, condition), so the panels compare the *ranking* of pairs rather than the scale
    of damage, which differs between arms and between fixation types.
    """
    arms = _arms_present(pairs)
    conditions = _conditions_present(pairs)
    order = (pairs.groupby("pair")[value].mean().sort_values(ascending=False).index.tolist())
    per = pairs.groupby([c for c in ("arm", "seed", "pair", "condition") if c in pairs.columns])[value].sum().reset_index()
    fig, axes = plt.subplots(1, len(arms), figsize=figsize, sharey=True, squeeze=False)
    rng = np.random.default_rng(5)
    for ax, arm, letter in zip(axes[0], arms, "ab"):
        sub = per[per["arm"] == arm] if "arm" in per.columns else per
        ax.grid(axis="y", **GRID_KW)
        for k, pair in enumerate(order):
            for j, condition in enumerate(conditions):
                color = CONDITION_COLORS.get(condition, INK)
                chunk = sub[(sub["pair"] == pair) & (sub["condition"] == condition)]
                x = k + (j - 1) * 0.24 + rng.uniform(-0.05, 0.05, size=len(chunk))
                ax.scatter(x, chunk[value], s=12, zorder=3, **_arm_marker_kw(arm, color))
                if not chunk.empty:
                    ax.hlines(chunk[value].mean(), k + (j - 1) * 0.24 - 0.1, k + (j - 1) * 0.24 + 0.1, color=color, linewidth=1.4, zorder=4)
        ax.axhline(1 / len(order), color=INK, linewidth=0.7, linestyle="--", zorder=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=6.8, rotation=25, ha="right")
        ax.set_title(arm, fontsize=8)
        nice_axis(ax); _panel(ax, letter)
    axes[0][0].set_ylabel("share of pair-lesion damage\n(within seed × condition)")
    for condition in conditions:
        axes[0][0].scatter([], [], s=12, color=CONDITION_COLORS.get(condition, INK), label=CONDITION_SHORT_LABELS.get(condition, condition))
    axes[0][0].legend(frameon=False, fontsize=6.2, loc="upper right")
    fig.tight_layout()
    return fig


def plot_pair_lesion_by_target(pairs: pd.DataFrame, *, arm: str = "constrained", figsize: tuple[float, float] = (7.4, 2.6)):
    """Which region absorbs the damage when a pair is cut: pair x target-region heatmap, one panel per condition."""
    block = pairs[pairs["arm"] == arm] if "arm" in pairs.columns else pairs
    conditions = _conditions_present(block)
    regions = [r for r in REGION_COLORS if r in set(block["region"])]
    order = block.groupby("pair")["damage_sse"].mean().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(1, len(conditions), figsize=figsize, sharey=True, squeeze=False)
    table = block.groupby(["condition", "pair", "region"])["damage_sse"].mean()
    for ax, condition in zip(axes[0], conditions):
        grid = np.array([[table.get((condition, p, r), np.nan) for r in regions] for p in order])
        grid = grid / np.nansum(grid)  # share of this condition's pair damage
        image = ax.imshow(grid, cmap="cividis", aspect="auto", vmin=0, vmax=float(np.nanmax(grid)))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                            color="white" if grid[i, j] < 0.5 * float(np.nanmax(grid)) else INK)
        ax.set_xticks(range(len(regions))); ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions], fontsize=6.6)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.6)
        ax.set_title(CONDITION_SHORT_LABELS.get(condition, condition), fontsize=8, color=CONDITION_COLORS.get(condition, INK))
        ax.set_xlabel("region whose fit is damaged")
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.02, pad=0.02, label="share of damage")
    fig.suptitle(f"{arm}: where a pair lesion's damage lands", fontsize=8, y=1.0)
    return fig


def plot_lesion_ranking_by_condition(ranking: pd.DataFrame, *, figsize: tuple[float, float] = (7.4, 2.6)):
    """Seed agreement on the lesion ranking, per fixation type and lesion kind, against the permutation null."""
    kinds = list(dict.fromkeys(ranking["lesion_kind"]))
    arms = _arms_present(ranking)
    conditions = _conditions_present(ranking)
    fig, axes = plt.subplots(1, len(kinds), figsize=figsize, sharey=True, squeeze=False)
    for ax, kind, letter in zip(axes[0], kinds, "abc"):
        block = ranking[ranking["lesion_kind"] == kind]
        ax.grid(axis="y", **GRID_KW)
        for k, condition in enumerate(conditions):
            color = CONDITION_COLORS.get(condition, INK)
            for arm in arms:
                row = block[(block["condition"] == condition) & ((block["arm"] == arm) if "arm" in block.columns else True)]
                if row.empty:
                    continue
                row = row.iloc[0]
                x = k + (-0.18 if arm == "dense" else 0.18)
                ax.vlines(x, row["tau_min"], row["tau_mean"], color=color, linewidth=1.0, zorder=2)
                ax.scatter(x, row["tau_mean"], s=28, zorder=3, **_arm_marker_kw(arm, color))
                ax.hlines(row["null_p95"], x - 0.14, x + 0.14, color=INK, linewidth=0.9, linestyle="--", zorder=2)
        ax.axhline(0.0, color=INK, linewidth=0.7, zorder=1)
        ax.set_xticks(range(len(conditions))); ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions], fontsize=7)
        ax.set_title(f"{kind} lesions", fontsize=8)
        nice_axis(ax); _panel(ax, letter)
    axes[0][0].set_ylabel("Kendall τ between seeds' rankings\n(dot = mean, tail to min; dashes = null 95%)")
    axes[0][0].scatter([], [], s=28, label="dense", **_arm_marker_kw("dense", INK))
    axes[0][0].scatter([], [], s=28, label="constrained", **_arm_marker_kw("constrained", INK))
    axes[0][0].legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()
    return fig


def plot_lesion_dynamics(dynamics: pd.DataFrame, *, metric: str, ylabel: str, kinds: Sequence[str] = ("bidirectional", "isolation"),
                         figsize: tuple[float, float] = (7.6, 3.0)):
    """What a lesion does to the lesioned network's own dynamics, per condition, every seed a mark, one panel per arm."""
    block = dynamics[dynamics["lesion_kind"].isin(kinds)]
    arms = _arms_present(block)
    conditions = _conditions_present(block)
    order = list(dict.fromkeys(block.sort_values(["lesion_kind", "lesion"])["lesion"]))
    fig, axes = plt.subplots(1, len(arms), figsize=figsize, sharey=True, squeeze=False)
    rng = np.random.default_rng(6)
    for ax, arm, letter in zip(axes[0], arms, "ab"):
        sub = block[block["arm"] == arm] if "arm" in block.columns else block
        ax.grid(axis="y", **GRID_KW)
        for k, lesion in enumerate(order):
            for j, condition in enumerate(conditions):
                color = CONDITION_COLORS.get(condition, INK)
                chunk = sub[(sub["lesion"] == lesion) & (sub["condition"] == condition)]
                x = k + (j - 1) * 0.24 + rng.uniform(-0.05, 0.05, size=len(chunk))
                ax.scatter(x, chunk[metric], s=11, zorder=3, **_arm_marker_kw(arm, color))
                if not chunk.empty:
                    ax.hlines(chunk[metric].mean(), k + (j - 1) * 0.24 - 0.1, k + (j - 1) * 0.24 + 0.1, color=color, linewidth=1.3, zorder=4)
        ax.axhline(0.0, color=INK, linewidth=0.7, zorder=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=6.4, rotation=35, ha="right")
        ax.set_title(arm, fontsize=8)
        nice_axis(ax); _panel(ax, letter)
    axes[0][0].set_ylabel(ylabel)
    for condition in conditions:
        axes[0][0].scatter([], [], s=11, color=CONDITION_COLORS.get(condition, INK), label=CONDITION_SHORT_LABELS.get(condition, condition))
    axes[0][0].legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()
    return fig


# ======================================================================================
# 6. Consistency scoreboard
# ======================================================================================


def plot_consistency_scoreboard(board: pd.DataFrame, *, properties: Sequence[str], labels: Mapping[str, str] | None = None,
                                figsize: tuple[float, float] = (7.6, 4.2)):
    """Every per-condition quantity on one page: is interactive face the extreme, in what fraction of fits, and how cleanly?

    (a) fraction of fits with interactive face lowest (left of centre, drawn negative) or
    highest (right); a bar reaching ±1 is a property of every fit. (b) separation: mean
    between-condition range over mean seed sd; above 1 the conditions are further apart
    than the fits are from one another.
    """
    labels = dict(labels or {})
    block = board[board["property"].isin(properties)].copy()
    arms = _arms_present(block)
    y = {p: i for i, p in enumerate(properties)}
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1.0]}, sharey=True)

    ax = axes[0]
    ax.grid(axis="x", **GRID_KW)
    for arm in arms:
        chunk = block[block["arm"] == arm] if "arm" in block.columns else block
        offset = -0.18 if arm == "dense" else 0.18
        ys = [y[p] + offset for p in chunk["property"]]
        ax.barh(ys, -chunk["fi_lowest"], 0.34, color=ARM_COLORS.get(arm, INK), edgecolor=NEUTRAL_EDGE, linewidth=0.5, zorder=3, label=arm)
        ax.barh(ys, chunk["fi_highest"], 0.34, color=ARM_COLORS.get(arm, INK), edgecolor=NEUTRAL_EDGE, linewidth=0.5, zorder=3)
    ax.axvline(0, color=INK, linewidth=0.8, zorder=2)
    for v in (-1 / 3, 1 / 3):
        ax.axvline(v, color=INK, linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(["1", "0.5", "0", "0.5", "1"])
    ax.set_xlabel("← fraction of fits with Int face LOWEST        fraction with Int face HIGHEST →", fontsize=6.8)
    ax.set_yticks(range(len(properties))); ax.set_yticklabels([labels.get(p, p) for p in properties], fontsize=6.8)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="x", **GRID_KW)
    for arm in arms:
        chunk = block[block["arm"] == arm] if "arm" in block.columns else block
        offset = -0.18 if arm == "dense" else 0.18
        ax.scatter(chunk["separation"], [y[p] + offset for p in chunk["property"]], s=20, zorder=3, **_arm_marker_kw(arm, ARM_COLORS.get(arm, INK)))
    ax.axvline(1.0, color=INK, linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("separation: between-condition range / seed sd", fontsize=6.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _panel(ax, "b")
    fig.tight_layout()
    return fig


__all__ = [
    "ARM_COLORS", "PAIR_COLORS", "PAIR_ORDER",
    "plot_condition_cost", "plot_consistency_scoreboard", "plot_fit_cells_by_arm", "plot_fixed_point_summary",
    "plot_flow_fields", "plot_flow_matrix_two_arms", "plot_jacobian_time", "plot_lesion_dynamics",
    "plot_lesion_ranking_by_condition", "plot_loss_curves", "plot_pair_lesion_by_target", "plot_pair_lesions",
    "plot_property_by_condition", "plot_property_grid", "plot_reconstruction_overlay", "plot_region_property_by_condition",
    "plot_region_timescales", "plot_similarity_by_property", "plot_similarity_scoreboard", "plot_time_course_two_arms",
]
