"""Figures for the mRNN measurement audit and the invariants that survive it.

Every correction panel is drawn as a *pair*: what the chapter currently reports beside
what the same quantity is when computed as named. A single corrected number would ask the
reader to take the size of the correction on trust; the pair puts it on the page.

The invariant panels are drawn per seed rather than as a mean with error bars. With five
seeds the mean hides the only thing that matters -- whether the effect is present in every
fit or in three of them -- so each seed is a visible mark.
"""

from __future__ import annotations

from typing import Mapping, Sequence

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
)

#: Reported-vs-corrected is the recurring contrast in this module, so it gets one fixed
#: pair of colours used in every panel: warm for the number as it stands, cool for the
#: number the name implies.
REPORTED_COLOR = "#b8621f"
CORRECTED_COLOR = "#2a6f8f"
NULL_COLOR = "#b0b0b0"
GRID_KW = dict(color="#dcdcdc", linewidth=0.6, zorder=0)


def _panel(ax, letter: str) -> None:
    ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def _condition_positions(conditions: Sequence[str]) -> dict[str, int]:
    return {condition: index for index, condition in enumerate(conditions)}


# --------------------------------------------------------------------------------------
# E1 -- the readout rank cap
# --------------------------------------------------------------------------------------


def plot_readout_rank_cap(
    scored: pd.DataFrame,
    cap_by_region: pd.DataFrame,
    *,
    n_components: int,
    figsize: tuple[float, float] = (7.4, 2.9),
):
    """The capacity curve against the ceiling it could not have exceeded.

    ``scored`` comes from :func:`fixation_mrnn_audit.capacity_against_cap`. The left panel
    lays the measured sweep over the structural cap; the right is the measured fit as a
    fraction of that cap, which is the part of the curve that is about the network.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    table = scored.sort_values("width")

    ax = axes[0]
    ax.grid(axis="y", **GRID_KW)
    ax.plot(table["width"], table["ceiling_relative_cap"], "-o", color=REPORTED_COLOR,
            markersize=4.5, linewidth=1.6, label=f"structural cap ({n_components} PCs, rank $h$)",
            zorder=3)
    ax.plot(table["width"], table["measured"], "-o", color=CORRECTED_COLOR,
            markersize=4.5, linewidth=1.6, label="measured fit", zorder=3)
    ax.axhline(1.0, color=MUTED_INK, linewidth=0.7, linestyle=":", zorder=1)
    ax.set_xlabel("hidden units per region")
    ax.set_ylabel(r"$R^2$ / noise ceiling")
    ax.set_xticks(list(table["width"]))
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="y", **GRID_KW)
    ax.plot(table["width"], table["fraction_of_cap"], "-o", color=CORRECTED_COLOR,
            markersize=4.5, linewidth=1.6, zorder=3)
    for _, row in table.iterrows():
        ax.annotate(f"{row['fraction_of_cap']:.3f}", (row["width"], row["fraction_of_cap"]),
                    textcoords="offset points", xytext=(0, -11), ha="center",
                    fontsize=6.2, color=MUTED_INK)
    ax.axhline(1.0, color=MUTED_INK, linewidth=0.7, linestyle=":", zorder=1)
    ax.set_xlabel("hidden units per region")
    ax.set_ylabel("fit as fraction of its cap")
    ax.set_xticks(list(table["width"]))
    ax.set_xlim(min(table["width"]) - 4, max(table["width"]) + 4)
    ax.set_ylim(0.94, 1.02)
    nice_axis(ax)
    _panel(ax, "b")

    fig.tight_layout()
    return fig


def plot_cap_by_region(
    cap_by_region: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (3.9, 2.7),
):
    """Where the cap bites, region by region -- the panel that shows it is not uniform."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", **GRID_KW)
    regions = sorted(cap_by_region["region"].unique())
    palette = plt.get_cmap("cividis")(np.linspace(0.15, 0.85, len(regions)))
    for color, region in zip(palette, regions):
        block = cap_by_region[cap_by_region["region"] == region].sort_values("width")
        ax.plot(block["width"], block["ceiling_relative_cap"], "-o", color=color,
                markersize=3.6, linewidth=1.3, label=region.upper(), zorder=3)
    ax.axhline(1.0, color=MUTED_INK, linewidth=0.7, linestyle=":", zorder=1)
    ax.set_xlabel("hidden units per region")
    ax.set_ylabel("structural cap on $R^2$ / ceiling")
    ax.set_xticks(sorted(cap_by_region["width"].unique()))
    ax.legend(frameon=False, fontsize=6.4, ncol=2)
    nice_axis(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# E2 -- self-drive contamination
# --------------------------------------------------------------------------------------


def plot_current_agreement_decomposition(
    decomposition: pd.DataFrame,
    *,
    order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (7.4, 2.9),
):
    """Cross-seed current agreement before and after excluding the diagonal blocks.

    ``decomposition`` comes from
    :func:`fixation_mrnn_audit.current_agreement_decomposition`. The left panel is the
    reversal; the right shows the mechanism -- the share of the measured vector that a
    bottleneck does not constrain.
    """
    table = decomposition.copy()
    if order is not None:
        table["_order"] = table["label"].map({name: i for i, name in enumerate(order)})
        table = table.sort_values("_order")
    positions = np.arange(len(table))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.grid(axis="y", **GRID_KW)
    ax.plot(positions, table["agreement_all_blocks"], "-o", color=REPORTED_COLOR,
            markersize=4.5, linewidth=1.6, label="as reported (all 16 blocks)", zorder=3)
    ax.plot(positions, table["agreement_inter_only"], "-o", color=CORRECTED_COLOR,
            markersize=4.5, linewidth=1.6, label="inter-regional blocks only", zorder=3)
    ax.axhline(0.0, color=MUTED_INK, linewidth=0.7, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(table["label"], rotation=30, ha="right")
    ax.set_ylabel("cross-seed agreement")
    ax.legend(frameon=False, fontsize=6.4, loc="upper left")
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="y", **GRID_KW)
    ax.bar(positions, table["self_energy_share"], 0.62, color=NEUTRAL_FILL,
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(table["label"], rotation=30, ha="right")
    ax.set_ylabel("within-region share of\ncurrent energy")
    ax.set_ylim(0, 1)
    nice_axis(ax)
    _panel(ax, "b")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# E3 -- the Jacobian bias term
# --------------------------------------------------------------------------------------


def plot_jacobian_bias_correction(
    stability: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 2.9),
):
    """Every seed's leading modulus, before and after restoring the bias.

    Paired marks rather than group means: the correction moves each point by a different
    amount, and one condition loses its expanding modes entirely.
    """
    conditions = [c for c in CONDITION_ORDER if c in set(stability["condition"])]
    positions = _condition_positions(conditions)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.grid(axis="y", **GRID_KW)
    for _, row in stability.iterrows():
        x = positions[row["condition"]]
        ax.plot([x - 0.16, x + 0.16],
                [row["top_modulus_as_computed"], row["top_modulus_corrected"]],
                color=MUTED_INK, linewidth=0.6, zorder=2)
    ax.scatter([positions[c] - 0.16 for c in stability["condition"]],
               stability["top_modulus_as_computed"], s=22, color=REPORTED_COLOR,
               edgecolor="white", linewidth=0.5, zorder=4, label="as computed")
    ax.scatter([positions[c] + 0.16 for c in stability["condition"]],
               stability["top_modulus_corrected"], s=22, color=CORRECTED_COLOR,
               edgecolor="white", linewidth=0.5, zorder=4, label="bias restored")
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions])
    ax.set_ylabel(r"leading $|\lambda|$ at slow point")
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="y", **GRID_KW)
    width = 0.34
    means = stability.groupby("condition")[
        ["n_expanding_as_computed", "n_expanding_corrected"]
    ].mean().reindex(conditions)
    index = np.arange(len(conditions))
    ax.bar(index - width / 2, means["n_expanding_as_computed"], width, color=REPORTED_COLOR,
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="as computed", zorder=3)
    ax.bar(index + width / 2, means["n_expanding_corrected"], width, color=CORRECTED_COLOR,
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="bias restored", zorder=3)
    ax.set_xticks(index)
    ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions])
    ax.set_ylabel(r"modes with $|\lambda|>1$")
    ax.legend(frameon=False, fontsize=6.4)
    nice_axis(ax)
    _panel(ax, "b")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# E4 -- damage in comparable units
# --------------------------------------------------------------------------------------


def plot_damage_unit_confound(
    damage: pd.DataFrame,
    energy: pd.DataFrame,
    *,
    reference: str = "face_non_interactive",
    figsize: tuple[float, float] = (7.4, 2.9),
):
    """Lesion damage per condition in R^2 units and in absolute error, with the reason.

    The left panel is the denominator: interactive face carries roughly half the target
    variance. The right panel is the consequence -- the same perturbation, scored two ways,
    with the effect present in one and absent in the other.
    """
    conditions = [c for c in CONDITION_ORDER if c in set(damage["condition"])]
    index = np.arange(len(conditions))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.grid(axis="y", **GRID_KW)
    pooled = energy.groupby("condition")["ss_tot"].sum().reindex(conditions)
    ax.bar(index, pooled.values, 0.6,
           color=[CONDITION_COLORS.get(c, NEUTRAL_FILL) for c in conditions],
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, zorder=3)
    ax.set_xticks(index)
    ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions])
    ax.set_ylabel("target variance $\\sum ss_{tot}$")
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="y", **GRID_KW)
    means = damage.groupby("condition")[["damage_r2", "damage_sse"]].mean().reindex(conditions)
    ratios = means / means.loc[reference]
    width = 0.34
    ax.bar(index - width / 2, ratios["damage_r2"], width, color=REPORTED_COLOR,
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="as reported ($R^2$ units)", zorder=3)
    ax.bar(index + width / 2, ratios["damage_sse"], width, color=CORRECTED_COLOR,
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="absolute squared error", zorder=3)
    for offset, column in ((-width / 2, "damage_r2"), (width / 2, "damage_sse")):
        for i, value in enumerate(ratios[column]):
            ax.annotate(f"{value:.2f}×", (index[i] + offset, value), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=6.0, color=MUTED_INK)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(index)
    ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions])
    ax.set_ylabel(f"lesion damage,\nrelative to {CONDITION_SHORT_LABELS.get(reference, reference)}")
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    nice_axis(ax)
    _panel(ax, "b")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# The gate -- fitted agreement against every floor
# --------------------------------------------------------------------------------------


def plot_gate_against_floors(
    levels: pd.DataFrame,
    *,
    measure_order: Sequence[str],
    figsize: tuple[float, float] = (7.4, 3.0),
):
    """Fitted seed agreement beside the architectural null and the surrogate-target arms.

    ``levels`` is indexed by measure with one column per arm. Read the margin, not the
    level: a measure sitting at 0.92 whose untrained floor is 0.95 is reporting the
    architecture and not the data.
    """
    table = levels.reindex([m for m in measure_order if m in levels.index])
    index = np.arange(len(table))
    arms = [
        ("architectural null", NULL_COLOR),
        ("phase randomized", "#9ec6d8"),
        ("time shuffled", "#c9b48b"),
        ("fitted", CORRECTED_COLOR),
    ]
    available = [(name, color) for name, color in arms if name in table.columns]
    width = 0.8 / len(available)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.grid(axis="y", **GRID_KW)
    for position, (name, color) in enumerate(available):
        offset = (position - (len(available) - 1) / 2) * width
        ax.bar(index + offset, table[name], width * 0.9, color=color,
               edgecolor=NEUTRAL_EDGE, linewidth=0.5, label=name, zorder=3)
    ax.axhline(0.0, color=MUTED_INK, linewidth=0.7, zorder=1)
    ax.set_xticks(index)
    ax.set_xticklabels([m.replace(" ", "\n") for m in table.index], fontsize=6.2)
    ax.set_ylabel("cross-seed agreement")
    ax.set_ylim(top=float(np.nanmax(table[[n for n, _ in available]].values)) * 1.34)
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), handlelength=1.1, columnspacing=1.0)
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="y", **GRID_KW)
    if "margin_in_null_sd" in table.columns:
        colors = ["#a8352a" if v < 0 else "#2a6f4e" for v in table["margin_in_null_sd"]]
        ax.bar(index, table["margin_in_null_sd"], 0.6, color=colors,
               edgecolor=NEUTRAL_EDGE, linewidth=0.6, zorder=3)
        # Labels sit inside the bar, so the most negative one cannot fall off the axis.
        for i, value in enumerate(table["margin_in_null_sd"]):
            ax.annotate(f"{value:+.1f}", (index[i], value), textcoords="offset points",
                        xytext=(0, 4 if value < 0 else -8), ha="center",
                        va="bottom" if value < 0 else "top", fontsize=6.4, color="white"
                        if abs(value) > 0.6 else MUTED_INK)
        span = float(np.nanmax(np.abs(table["margin_in_null_sd"]))) * 1.12
        ax.set_ylim(-span, span * 0.45)
    ax.axhline(0.0, color=INK, linewidth=0.9, zorder=2)
    ax.set_xticks(index)
    ax.set_xticklabels([m.replace(" ", "\n") for m in table.index], fontsize=6.2)
    ax.set_ylabel("margin over untrained floor\n(in units of the floor's own SD)")
    nice_axis(ax)
    _panel(ax, "b")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# I1 -- the face/object channel split
# --------------------------------------------------------------------------------------


PAIR_SHORT: dict[tuple[str, str], str] = {
    ("face_interactive", "face_non_interactive"): "int face\nvs non-int",
    ("face_interactive", "object"): "int face\nvs object",
    ("face_non_interactive", "object"): "non-int\nvs object",
}


def plot_condition_channel_split(
    fitted: pd.DataFrame,
    null: pd.DataFrame,
    data_side: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 3.0),
):
    """Whether two conditions travel a pathway through the same channel.

    Left: every pathway x seed cosine, fitted against the untrained floor. Middle: the
    fraction on the positive side of zero, which is the statistic the claim rests on.
    Right: the same comparison made on the *targets*, which is what makes the result
    non-trivial -- the two face conditions are near-orthogonal in the data and aligned in
    the model.
    """
    pairs = [p for p in PAIR_SHORT if ((fitted["condition_a"] == p[0]) & (fitted["condition_b"] == p[1])).any()]
    index = np.arange(len(pairs))
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    rng = np.random.default_rng(0)

    ax = axes[0]
    ax.grid(axis="y", **GRID_KW)
    for position, pair in enumerate(pairs):
        for offset, frame, color, label in (
            (-0.18, null, NULL_COLOR, "untrained"),
            (0.18, fitted, CORRECTED_COLOR, "fitted"),
        ):
            values = frame[(frame["condition_a"] == pair[0]) & (frame["condition_b"] == pair[1])]["cosine"]
            jitter = rng.uniform(-0.07, 0.07, size=len(values))
            ax.scatter(np.full(len(values), position + offset) + jitter, values, s=7,
                       color=color, alpha=0.65, linewidth=0, zorder=3,
                       label=label if position == 0 else None)
            ax.plot([position + offset - 0.11, position + offset + 0.11],
                    [values.mean()] * 2, color=INK, linewidth=1.2, zorder=4)
    ax.axhline(0.0, color=INK, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(index)
    ax.set_xticklabels([PAIR_SHORT[p] for p in pairs], fontsize=6.2)
    ax.set_ylabel("cosine between the currents\none pathway carries")
    ax.legend(frameon=False, fontsize=6.4, loc="lower left")
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="y", **GRID_KW)
    width = 0.34
    for offset, frame, color, label in (
        (-width / 2, null, NULL_COLOR, "untrained"),
        (width / 2, fitted, CORRECTED_COLOR, "fitted"),
    ):
        fractions = [
            float((frame[(frame["condition_a"] == p[0]) & (frame["condition_b"] == p[1])]["cosine"] > 0).mean())
            for p in pairs
        ]
        ax.bar(index + offset, fractions, width * 0.92, color=color, edgecolor=NEUTRAL_EDGE,
               linewidth=0.6, label=label, zorder=3)
        for i, value in enumerate(fractions):
            ax.annotate(f"{value:.0%}", (index[i] + offset, value), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=6.0, color=MUTED_INK)
    ax.axhline(0.5, color=INK, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(index)
    ax.set_xticklabels([PAIR_SHORT[p] for p in pairs], fontsize=6.2)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("cells with positive cosine")
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    nice_axis(ax)
    _panel(ax, "b")

    ax = axes[2]
    ax.grid(axis="y", **GRID_KW)
    for position, pair in enumerate(pairs):
        values = data_side[
            (data_side["condition_a"] == pair[0]) & (data_side["condition_b"] == pair[1])
        ]["correlation"]
        ax.scatter(np.full(len(values), position), values, s=18, color="#6b6b6b",
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0.0, color=INK, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(index)
    ax.set_xticklabels([PAIR_SHORT[p] for p in pairs], fontsize=6.2)
    ax.set_ylabel("correlation between the\ntarget trajectories (one mark = region)")
    nice_axis(ax)
    _panel(ax, "c")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# I2 / I3 -- the two interactive-face signatures, per seed
# --------------------------------------------------------------------------------------


def plot_seedwise_condition_effect(
    frame: pd.DataFrame,
    value: str,
    *,
    reference_line: float | None = None,
    ylabel: str = "",
    title: str = "",
    figsize: tuple[float, float] = (3.9, 2.9),
):
    """One line per seed across the three conditions.

    Five seeds is too few for a mean to mean anything, and exactly enough for every fit to
    be a visible line. What the panel is for is the question "is this present in every
    seed or in three of them", which a bar chart cannot answer.
    """
    conditions = [c for c in CONDITION_ORDER if c in set(frame["condition"])]
    positions = _condition_positions(conditions)
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", **GRID_KW)

    for seed, block in frame.groupby("seed"):
        block = block.set_index("condition").reindex(conditions)
        ax.plot([positions[c] for c in conditions], block[value].values, "-o",
                color=MUTED_INK, alpha=0.55, markersize=3.4, linewidth=0.9, zorder=3)
    for condition in conditions:
        values = frame[frame["condition"] == condition][value]
        ax.plot([positions[condition] - 0.19, positions[condition] + 0.19],
                [values.mean()] * 2, color=CONDITION_COLORS.get(condition, INK),
                linewidth=2.4, solid_capstyle="butt", zorder=5)

    if reference_line is not None:
        ax.axhline(reference_line, color=INK, linewidth=0.9, linestyle="--", zorder=2)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([CONDITION_SHORT_LABELS.get(c, c) for c in conditions])
    ax.set_xlim(-0.45, len(conditions) - 0.55)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=8)
    nice_axis(ax)
    fig.tight_layout()
    return fig


def plot_invariant_scoreboard(
    rows: Sequence[Mapping[str, object]],
    *,
    figsize: tuple[float, float] = (7.4, 2.4),
):
    """How many seeds each candidate invariant holds in, against what chance would give.

    The summary panel for the invariant section. A claim that holds in five of five seeds
    and one that holds in three are different claims, and the chapter has both.
    """
    frame = pd.DataFrame(list(rows))
    index = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="x", **GRID_KW)

    colors = ["#2a6f4e" if r >= 1.0 else ("#b8621f" if r >= 0.75 else "#a8352a")
              for r in frame["fraction"]]
    ax.barh(index, frame["fraction"], 0.55, color=colors, edgecolor=NEUTRAL_EDGE,
            linewidth=0.6, zorder=3)
    for i, row in frame.iterrows():
        ax.annotate(str(row["detail"]), (row["fraction"], i), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=6.0, color=MUTED_INK)
    ax.axvline(0.5, color=INK, linewidth=0.9, linestyle="--", zorder=2)
    # Deliberately not nice_axis: these y ticks are categorical, and a MaxNLocator
    # would drop most of the claim labels.
    ax.set_yticks(index)
    ax.set_yticklabels(frame["claim"], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.85)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("fraction of seeds (or cells) the effect holds in")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5, pad=1.5)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_cap_by_region",
    "plot_condition_channel_split",
    "plot_current_agreement_decomposition",
    "plot_damage_unit_confound",
    "plot_gate_against_floors",
    "plot_invariant_scoreboard",
    "plot_jacobian_bias_correction",
    "plot_readout_rank_cap",
    "plot_seedwise_condition_effect",
]


# ======================================================================================
# The solution manifold: the constraint ladder, the agreement battery, the screen
# ======================================================================================

ARM_ACCENT = "#2f4858"


def plot_constraint_ladder(
    ladder: pd.DataFrame,
    *,
    tolerance: float = 0.01,
    figsize: tuple[float, float] = (7.4, 3.6),
):
    """Every constraint that was imposed, what it cost in fit, and what it cost in size.

    The argument in one panel. Each row removes something structural and is refitted from
    scratch; if the bar still reaches the ceiling, the data tolerated the removal. A ladder
    in which nearly every rung clears the line is not a series of failed experiments -- it
    is the finding, and it is what "the solution set is large" means concretely.
    """
    table = ladder.sort_values("fit_vs_ceiling")
    index = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True,
                             gridspec_kw={"width_ratios": [2.0, 1.0]})

    ax = axes[0]
    ax.grid(axis="x", **GRID_KW)
    adequate = table["fit_vs_ceiling"] >= 1.0 - tolerance
    ax.barh(index, table["fit_vs_ceiling"], 0.62,
            color=np.where(adequate, "#2c6b4f", "#a8352a"),
            edgecolor=NEUTRAL_EDGE, linewidth=0.6, zorder=3)
    ax.axvline(1.0, color=INK, linewidth=1.0, linestyle="--", zorder=4)
    ax.set_yticks(index)
    ax.set_yticklabels(table["label"], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0.90, max(1.03, float(table["fit_vs_ceiling"].max()) + 0.01))
    ax.set_xlabel("fit / noise ceiling  (1.0 = the data's own reliability)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="y", length=0)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="x", **GRID_KW)
    ax.barh(index, table["parameters"] / 1000.0, 0.62, color=NEUTRAL_FILL,
            edgecolor=NEUTRAL_EDGE, linewidth=0.6, zorder=3)
    ax.set_xlabel("free parameters (thousands)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="y", length=0)
    _panel(ax, "b")

    fig.tight_layout()
    return fig


def plot_agreement_battery(
    battery: pd.DataFrame,
    order: Sequence[tuple[str, str, str]],
    *,
    arm: str | None = None,
    resolution: float | None = None,
    figsize: tuple[float, float] = (7.4, 3.4),
):
    """Every agreement measure beside its untrained floor, tiered by what it ignores.

    Similarities and distances are drawn on separate panels because their axes point in
    opposite directions -- for a similarity, fitted above the floor is agreement; for a
    distance it is fitted *below* the floor. Mixing them on one axis is how a battery like
    this gets misread.
    """
    block = battery if arm is None else battery[battery["arm"] == arm]
    summary = block.groupby(["measure", "ensemble"])["value"].agg(["mean", "std"]).unstack("ensemble")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, kind, letter in ((axes[0], "similarity", "a"), (axes[1], "distance", "b")):
        names = [n for n, _, k in order if k == kind and n in summary.index]
        index = np.arange(len(names))
        ax.grid(axis="y", **GRID_KW)
        width = 0.36
        for offset, column, color, label in (
            (-width / 2, "untrained", NULL_COLOR, "untrained floor"),
            (width / 2, "fitted", CORRECTED_COLOR, "fitted seeds"),
        ):
            if ("mean", column) not in summary.columns:
                continue
            values = summary.loc[names, ("mean", column)]
            errors = summary.loc[names, ("std", column)].fillna(0.0)
            ax.bar(index + offset, values, width * 0.92, yerr=errors, capsize=2,
                   color=color, edgecolor=NEUTRAL_EDGE, linewidth=0.6, label=label, zorder=3)
        if kind == "distance" and resolution is not None:
            ax.axhline(resolution, color="#a8352a", linewidth=1.0, linestyle=":", zorder=4)
            # Left-anchored inside the axis: at the right edge it falls outside the frame.
            ax.annotate(f"measurement floor ({resolution:.2f})", (-0.42, resolution),
                        textcoords="offset points", xytext=(0, 4), ha="left", va="bottom",
                        fontsize=6.0, color="#a8352a", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.85))
        ax.axhline(0.0, color=MUTED_INK, linewidth=0.7, zorder=1)
        ax.set_xticks(index)
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=6.0)
        ax.set_ylabel("agreement" if kind == "similarity" else "distance (0 = identical)")
        ax.set_title("higher = more alike" if kind == "similarity" else "lower = more alike",
                     fontsize=7.5, color=MUTED_INK)
        ax.legend(frameon=False, fontsize=6.2, loc="upper left" if kind == "similarity" else "upper right")
        nice_axis(ax)
        _panel(ax, letter)

    fig.tight_layout()
    return fig


def plot_battery_across_arms(
    battery: pd.DataFrame,
    measures: Sequence[str],
    *,
    arm_order: Sequence[str],
    figsize: tuple[float, float] = (7.4, 3.4),
):
    """Fitted minus untrained, per measure, for every architecture tested.

    The panel that says the negative result is not about one model. A constraint that made
    the circuit identifiable would show as a column lifting off the zero line, and none does.
    """
    summary = (battery.groupby(["arm", "measure", "ensemble"])["value"].mean()
               .unstack("ensemble"))
    summary["margin"] = summary["fitted"] - summary["untrained"]
    arms = [a for a in arm_order if a in summary.index.get_level_values("arm")]
    index = np.arange(len(arms))

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", **GRID_KW)
    palette = plt.get_cmap("cividis")(np.linspace(0.12, 0.82, len(measures)))
    width = 0.8 / max(len(measures), 1)
    for position, (measure, color) in enumerate(zip(measures, palette)):
        offset = (position - (len(measures) - 1) / 2) * width
        values = [float(summary.loc[(a, measure), "margin"])
                  if (a, measure) in summary.index else np.nan for a in arms]
        ax.bar(index + offset, values, width * 0.9, color=color, edgecolor=NEUTRAL_EDGE,
               linewidth=0.5, label=measure, zorder=3)
    ax.axhline(0.0, color=INK, linewidth=1.0, zorder=4)
    ax.set_xticks(index)
    ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=6.4)
    ax.set_ylabel("fitted − untrained\n(similarity measures; > 0 would be agreement)")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="lower left")
    nice_axis(ax)
    fig.tight_layout()
    return fig


def plot_invariant_screen(
    consistency: pd.DataFrame,
    properties: Sequence[tuple[str, str, str]],
    *,
    figsize: tuple[float, float] = (7.4, 3.0),
):
    """Each candidate property, and the fraction of fits its condition ordering holds in.

    Bars are coloured by whether the property is scale-free. A scale-carrying property that
    holds everywhere is expected -- it tracks the size of the condition's target -- so the
    colour is what stops the panel from being read as six findings.
    """
    kinds = {name: kind for name, kind, _ in properties}
    controls = {name: control for name, _, control in properties}
    table = consistency.set_index("property").reindex([n for n, _, _ in properties]).dropna(how="all")
    index = np.arange(len(table))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    ax.grid(axis="x", **GRID_KW)
    colors = ["#2c6b4f" if kinds.get(n) == "scale-free" else "#b0b0b0" for n in table.index]
    ax.barh(index, table["fi_lowest"], 0.6, color=colors, edgecolor=NEUTRAL_EDGE,
            linewidth=0.6, zorder=3)
    for i, (name, row) in enumerate(table.iterrows()):
        ax.annotate(f"{int(round(row['fi_lowest'] * row['n_fits']))}/{int(row['n_fits'])}",
                    (row["fi_lowest"], i), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=6.2, color=MUTED_INK)
    ax.axvline(1 / 3, color=INK, linewidth=0.9, linestyle="--", zorder=4)
    ax.set_yticks(index)
    ax.set_yticklabels([f"{n}\n({controls.get(n, '')})" for n in table.index], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("fits where interactive face is the extreme condition")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="y", length=0)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="x", **GRID_KW)
    ax.barh(index, table["fi_over_fn_median"], 0.6, color=colors, edgecolor=NEUTRAL_EDGE,
            linewidth=0.6, zorder=3)
    ax.hlines(index, table["fi_over_fn_min"], table["fi_over_fn_max"], color=INK,
              linewidth=1.1, zorder=5)
    ax.axvline(1.0, color=INK, linewidth=0.9, linestyle="--", zorder=4)
    ax.set_xlabel("interactive face / non-interactive face\n(median, and range over all fits)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="y", length=0)
    _panel(ax, "b")

    fig.tight_layout()
    return fig


def plot_model_vs_target_ratio(
    screen: pd.DataFrame,
    control: pd.DataFrame,
    pairs: Sequence[tuple[str, str, str]],
    *,
    figsize: tuple[float, float] = (7.4, 2.9),
):
    """The model's condition ratio beside the target's, for every property that has both.

    The panel that separates a finding from an echo. Where the two bars match, the model is
    reproducing something already in the data and the property is a check on the fit. Where
    the target has no bar at all, the quantity exists only inside the model, and that is
    the only place a mechanistic claim can live.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", **GRID_KW)
    index = np.arange(len(pairs))
    width = 0.36

    model_values, target_values = [], []
    for model_name, target_name, _ in pairs:
        wide = screen.pivot_table(index=["arm", "seed"], columns="condition", values=model_name)
        model_values.append(float((wide["face_interactive"] / wide["face_non_interactive"]).median()))
        if target_name and target_name in control.columns:
            t = control.pivot_table(index="region", columns="condition", values=target_name)
            target_values.append(float((t["face_interactive"] / t["face_non_interactive"]).mean()))
        else:
            target_values.append(np.nan)

    ax.bar(index - width / 2, model_values, width * 0.92, color=CORRECTED_COLOR,
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="in the fitted model", zorder=3)
    ax.bar(index + width / 2, target_values, width * 0.92, color="#b0b0b0",
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="in the target data", zorder=3)
    for i, value in enumerate(target_values):
        if np.isnan(value):
            ax.annotate("no target\ncounterpart", (index[i] + width / 2, 0.04),
                        ha="center", va="bottom", fontsize=5.8, color="#a8352a", rotation=90)
    ax.axhline(1.0, color=INK, linewidth=0.9, linestyle="--", zorder=4)
    ax.set_xticks(index)
    ax.set_xticklabels([label for _, _, label in pairs], fontsize=6.4)
    ax.set_ylabel("interactive face / non-interactive face")
    ax.legend(frameon=False, fontsize=6.4)
    nice_axis(ax)
    fig.tight_layout()
    return fig


def plot_state_truncation(
    truncation: pd.DataFrame,
    *,
    n_units: int = 40,
    figsize: tuple[float, float] = (3.9, 2.9),
):
    """Fit retained when a region's state is cut to its top ``k`` directions.

    The panel that answers "is the network wider than it needs to be". A curve that
    saturates early would say the extra width is slack; one that climbs to the last
    dimension says every direction is carrying output.
    """
    summary = truncation.groupby("k")["r2"].agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", **GRID_KW)
    ax.fill_between(summary.index, summary["min"], summary["max"], color=CORRECTED_COLOR,
                    alpha=0.18, linewidth=0, zorder=2)
    ax.plot(summary.index, summary["mean"], "-o", color=CORRECTED_COLOR, markersize=3.6,
            linewidth=1.6, zorder=3)
    ax.axhline(float(summary["mean"].iloc[-1]), color=MUTED_INK, linewidth=0.8,
               linestyle=":", zorder=1)
    ax.set_xlabel(f"state directions kept (of {n_units})")
    ax.set_ylabel(r"$R^2$ on the 42-PC target")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    nice_axis(ax)
    fig.tight_layout()
    return fig


def plot_invariant_by_arm(
    screen: pd.DataFrame,
    prop: str,
    *,
    arm_order: Sequence[str],
    ylabel: str = "",
    figsize: tuple[float, float] = (7.4, 3.0),
):
    """One property, every architecture, every seed, all three conditions.

    The detail panel for a surviving invariant. Absolute values move several-fold between
    architectures, which is exactly why the claim has to be the *ordering* and the ratio
    rather than the level.
    """
    arms = [a for a in arm_order if a in set(screen["arm"])]
    conditions = [c for c in CONDITION_ORDER if c in set(screen["condition"])]
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", **GRID_KW)
    rng = np.random.default_rng(0)

    for position, arm in enumerate(arms):
        block = screen[screen["arm"] == arm]
        for offset, condition in zip(np.linspace(-0.26, 0.26, len(conditions)), conditions):
            values = block[block["condition"] == condition][prop].values
            jitter = rng.uniform(-0.045, 0.045, size=len(values))
            ax.scatter(np.full(len(values), position + offset) + jitter, values, s=15,
                       color=CONDITION_COLORS.get(condition, INK), edgecolor="white",
                       linewidth=0.4, zorder=4,
                       label=CONDITION_SHORT_LABELS.get(condition, condition) if position == 0 else None)
            ax.plot([position + offset - 0.075, position + offset + 0.075],
                    [values.mean()] * 2, color=INK, linewidth=1.2, zorder=5)
        for seed, seed_block in block.groupby("seed"):
            ordered = seed_block.set_index("condition").reindex(conditions)[prop].values
            ax.plot(position + np.linspace(-0.26, 0.26, len(conditions)), ordered,
                    color=MUTED_INK, linewidth=0.5, alpha=0.5, zorder=3)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=6.4)
    ax.set_ylabel(ylabel or prop)
    ax.legend(frameon=False, fontsize=6.4, ncol=3, loc="upper left")
    nice_axis(ax)
    fig.tight_layout()
    return fig


__all__ += [
    "plot_agreement_battery",
    "plot_battery_across_arms",
    "plot_constraint_ladder",
    "plot_invariant_by_arm",
    "plot_invariant_screen",
    "plot_model_vs_target_ratio",
    "plot_state_truncation",
]


# ======================================================================================
# Lesions on the trained network
# ======================================================================================

LESION_KIND_ORDER = ("within-region", "directed", "bidirectional", "isolation")


def plot_lesion_battery(
    lesions: pd.DataFrame,
    *,
    arm: str | None = None,
    figsize: tuple[float, float] = (7.4, 3.2),
):
    """Damage per lesion, against the matched random control of the same size.

    The control is the panel's whole point. Under a sparsity mask a pathway that kept more
    connections does more damage for that reason alone, so a lesion is only informative to
    the extent it sits away from the random curve at its own size.
    """
    block = lesions if arm is None else lesions[lesions["arm"] == arm]
    per_lesion = (block.groupby(["lesion_kind", "lesion", "seed"])
                  .agg(damage=("damage_sse", "sum"),
                       removed=("live_weights_removed", "first")).reset_index())
    control = per_lesion[per_lesion["lesion_kind"] == "random control"]
    real = per_lesion[per_lesion["lesion_kind"] != "random control"]

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    ax.grid(True, **GRID_KW)
    kinds = [k for k in LESION_KIND_ORDER if k in set(real["lesion_kind"])]
    palette = plt.get_cmap("cividis")(np.linspace(0.12, 0.82, len(kinds)))
    if not control.empty:
        curve = control.groupby("removed")["damage"].agg(["mean", "min", "max"])
        ax.fill_between(curve.index, curve["min"], curve["max"], color=NULL_COLOR,
                        alpha=0.35, linewidth=0, zorder=2, label="random control")
        ax.plot(curve.index, curve["mean"], color="#7a7a7a", linewidth=1.2, zorder=3)
    for color, kind in zip(palette, kinds):
        chunk = real[real["lesion_kind"] == kind]
        ax.scatter(chunk["removed"], chunk["damage"], s=16, color=color, edgecolor="white",
                   linewidth=0.4, zorder=4, label=kind)
    ax.set_xscale("log")
    ax.set_xlabel("live weights removed")
    ax.set_ylabel("damage (absolute squared error)")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left")
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="x", **GRID_KW)
    ranked = (real.groupby(["lesion_kind", "lesion"])["damage"].mean()
              .sort_values(ascending=False).head(14))
    index = np.arange(len(ranked))
    colors = [palette[kinds.index(k)] for k, _ in ranked.index]
    ax.barh(index, ranked.values, 0.62, color=colors, edgecolor=NEUTRAL_EDGE,
            linewidth=0.6, zorder=3)
    ax.set_yticks(index)
    ax.set_yticklabels([name for _, name in ranked.index], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("damage (absolute squared error)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="y", length=0)
    _panel(ax, "b")

    fig.tight_layout()
    return fig


def plot_lesion_ranking_agreement(
    ranking: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 2.9),
):
    """Cross-seed agreement on the *ordering* of lesions, against a permutation null.

    The panel a positive result would appear in. Weight-level agreement is settled and
    negative; a ranking that clears its null would let the chapter say which connections
    matter without claiming to have recovered a circuit.
    """
    table = ranking.copy()
    label = (table["arm"] + " · " + table["lesion_kind"]) if "arm" in table.columns else table["lesion_kind"]
    index = np.arange(len(table))
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", **GRID_KW)

    width = 0.36
    ax.bar(index - width / 2, table["null_p95"], width * 0.92, color=NULL_COLOR,
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="permutation null (95th pct)", zorder=3)
    colors = ["#2c6b4f" if c else "#a8352a" for c in table["clears_null"]]
    ax.bar(index + width / 2, table["tau_mean"], width * 0.92, color=colors,
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="observed (mean over seed pairs)", zorder=3)
    ax.vlines(index + width / 2, table["tau_min"], table["tau_mean"], color=INK,
              linewidth=1.0, zorder=5)
    ax.axhline(0.0, color=INK, linewidth=0.9, zorder=4)
    ax.set_xticks(index)
    ax.set_xticklabels(label, rotation=22, ha="right", fontsize=6.4)
    ax.set_ylabel("Kendall $\\tau$ between seeds'\nlesion-damage rankings")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    for i, row in table.iterrows():
        ax.annotate(f"{int(row['n_lesions'])} lesions\n{int(row['n_seeds'])} seeds",
                    (i, ax.get_ylim()[0]), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=5.6, color=MUTED_INK)
    nice_axis(ax)
    fig.tight_layout()
    return fig


def plot_lesion_condition_contrast(
    contrast: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 2.9),
):
    """Per-condition lesion damage in absolute units, and how many fits agree on it.

    Scored in absolute squared error rather than per-condition R^2, which is the correction
    that removed the previous version of this result: interactive face carries about half
    the target variance, so a variance-normalised score inflated its damage twofold.
    """
    columns = [c for c in contrast.columns if c.endswith("__mean_sse")]
    conditions = [c.replace("__mean_sse", "") for c in columns]
    ordered = [c for c in CONDITION_ORDER if c in conditions]
    labels = contrast["arm"] if "arm" in contrast.columns else pd.Series(["all fits"] * len(contrast))
    index = np.arange(len(contrast))

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1.0]})

    ax = axes[0]
    ax.grid(axis="y", **GRID_KW)
    width = 0.8 / max(len(ordered), 1)
    for position, condition in enumerate(ordered):
        offset = (position - (len(ordered) - 1) / 2) * width
        ax.bar(index + offset, contrast[f"{condition}__mean_sse"], width * 0.9,
               color=CONDITION_COLORS.get(condition, NEUTRAL_FILL), edgecolor=NEUTRAL_EDGE,
               linewidth=0.6, label=CONDITION_SHORT_LABELS.get(condition, condition), zorder=3)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=6.4)
    ax.set_ylabel("lesion damage\n(absolute squared error)")
    ax.legend(frameon=False, fontsize=6.2, ncol=3)
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="y", **GRID_KW)
    colors = ["#2c6b4f" if v >= 0.8 else ("#b8621f" if v >= 0.5 else "#a8352a")
              for v in contrast["fraction_of_fits_fi_highest"]]
    ax.bar(index, contrast["fraction_of_fits_fi_highest"], 0.6, color=colors,
           edgecolor=NEUTRAL_EDGE, linewidth=0.6, zorder=3)
    ax.axhline(1 / 3, color=INK, linewidth=0.9, linestyle="--", zorder=4)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=6.4)
    ax.set_ylabel("fits where interactive face\nis the most damaged")
    ax.set_ylim(0, 1.05)
    nice_axis(ax)
    _panel(ax, "b")

    fig.tight_layout()
    return fig


__all__ += [
    "plot_lesion_battery",
    "plot_lesion_condition_contrast",
    "plot_lesion_ranking_agreement",
]


def plot_fit_versus_identifiability(
    summary: pd.DataFrame,
    *,
    adequate_bar: float = 0.99,
    figsize: tuple[float, float] = (5.4, 4.0),
):
    """Every architecture as one point: how well it fits, against how reproducible it is.

    The single consolidated panel for the chapter's central claim. A project that had
    narrowed its solution set would show a diagonal -- tighter constraints trading fit for
    reproducibility. What these runs show instead is a horizontal band: everything fits, and
    the spread is along an axis nothing reaches the top of.

    ``summary`` needs ``label``, ``family``, ``fit`` (against the ceiling), ``agreement``
    (inter-regional, E2-corrected) and ``parameters``.
    """
    families = list(dict.fromkeys(summary["family"]))
    palette = dict(zip(families, plt.get_cmap("cividis")(np.linspace(0.1, 0.85, len(families)))))
    sizes = summary["parameters"] / summary["parameters"].max()

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, **GRID_KW)
    ax.axhspan(adequate_bar, ax.get_ylim()[1], color="#2c6b4f", alpha=0.06, zorder=1)
    ax.axhline(adequate_bar, color="#2c6b4f", linewidth=1.0, linestyle="--", zorder=2)
    ax.annotate(f"reproduces the data ({adequate_bar:g} of ceiling)",
                (ax.get_xlim()[0], adequate_bar), textcoords="offset points",
                xytext=(4, 4), fontsize=6.2, color="#2c6b4f", va="bottom")

    for family in families:
        block = summary[summary["family"] == family]
        ax.scatter(block["agreement"], block["fit"],
                   s=40 + 150 * sizes.loc[block.index], color=palette[family],
                   edgecolor="white", linewidth=0.7, zorder=4, label=family)
    for _, row in summary.iterrows():
        ax.annotate(row["label"], (row["agreement"], row["fit"]), textcoords="offset points",
                    xytext=(0, -11), ha="center", fontsize=5.6, color=MUTED_INK, zorder=5)

    ax.set_xlabel("cross-seed agreement on inter-regional currents\n(E2-corrected; 1.0 would be identical circuits)")
    ax.set_ylabel("worst region × condition cell\n($R^2$ / noise ceiling)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", title="constraint family",
              title_fontsize=6.4)
    nice_axis(ax)
    fig.tight_layout()
    return fig


__all__ += ["plot_fit_versus_identifiability"]


# ======================================================================================
# The final series
# ======================================================================================

from dal_monte_2022_analysis.ephys.plotting.thesis_common import CONDITION_LABELS, REGION_COLORS, REGION_LABELS


def plot_ladder_curves(
    fit: pd.DataFrame,
    *,
    value: str = "r2_vs_ceiling",
    ylabel: str = "$R^2$ / noise ceiling",
    adequate_bar: float | None = 0.99,
    figsize: tuple[float, float] = (7.4, 2.4),
):
    """Each region's fit against the number of other regions it was fitted with.

    Eight points per region -- alone, three pairs, three triples, full -- every one at the
    same width and readout, so the rank cap is constant across the row and the slope is
    dynamical. One panel per region; individual fits as small marks, the mean per rung as a
    line.
    """
    regions = [r for r in REGION_COLORS if r in set(fit["region"])]
    fig, axes = plt.subplots(1, len(regions), figsize=figsize, sharey=True)
    rng = np.random.default_rng(0)
    for ax, region in zip(np.atleast_1d(axes), regions):
        block = fit[fit["region"] == region]
        per_fit = block.groupby(["label", "seed", "n_partners"])[value].mean().reset_index()
        ax.grid(axis="y", **GRID_KW)
        jitter = rng.uniform(-0.12, 0.12, size=len(per_fit))
        ax.scatter(per_fit["n_partners"] + jitter, per_fit[value], s=9, color=REGION_COLORS[region],
                   alpha=0.55, linewidth=0, zorder=3)
        mean = per_fit.groupby("n_partners")[value].mean()
        ax.plot(mean.index, mean.values, "-o", color=REGION_COLORS[region], markersize=4,
                linewidth=1.5, zorder=4)
        if adequate_bar is not None:
            ax.axhline(adequate_bar, color=INK, linewidth=0.8, linestyle="--", zorder=2)
        ax.set_title(REGION_LABELS.get(region, region), fontsize=8)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("other regions present")
        nice_axis(ax)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_partner_matrix(
    fit: pd.DataFrame,
    *,
    value: str = "r2_vs_ceiling",
    figsize: tuple[float, float] = (3.9, 3.3),
):
    """Gain from one partner: pair fit minus single fit, region (rows) by partner (columns).

    The panel that answers "which region supplies what to whom" at the first rung, where
    the attribution is unambiguous.
    """
    regions = [r for r in REGION_COLORS if r in set(fit["region"])]
    single = fit[fit["n_partners"] == 0].groupby("region")[value].mean()
    matrix = np.full((len(regions), len(regions)), np.nan)
    pairs = fit[fit["n_partners"] == 1]
    for i, region in enumerate(regions):
        for j, partner in enumerate(regions):
            if region == partner:
                continue
            block = pairs[(pairs["region"] == region) & (pairs["partners"].map(lambda p: partner in p))]
            if not block.empty:
                matrix[i, j] = block[value].mean() - single[region]
    fig, ax = plt.subplots(figsize=figsize)
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    for i in range(len(regions)):
        for j in range(len(regions)):
            if i != j and np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=6.4,
                        color=INK if abs(matrix[i, j]) < 0.6 * limit else "white")
    ax.set_xticks(range(len(regions))); ax.set_yticks(range(len(regions)))
    ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions], fontsize=7)
    ax.set_yticklabels([REGION_LABELS.get(r, r) for r in regions], fontsize=7)
    ax.set_xlabel("partner added"); ax.set_ylabel("region scored")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="gain in $R^2$ / ceiling")
    fig.tight_layout()
    return fig


def plot_band_recovery_by_rung(
    recovery: pd.DataFrame,
    *,
    bands: Sequence[str] = ("0-5 Hz", "5-10 Hz", "10-20 Hz", "20-50 Hz"),
    figsize: tuple[float, float] = (7.4, 2.4),
):
    """Residual fraction of target power per band, against number of partners.

    ``recovery`` carries ``n_partners``. One panel per band, one line per region. The slow
    band should be flat -- a region reproduces its own envelope -- and the fast bands are
    where the ladder should drop.
    """
    regions = [r for r in REGION_COLORS if r in set(recovery["region"])]
    fig, axes = plt.subplots(1, len(bands), figsize=figsize, sharey=False)
    for ax, band in zip(axes, bands):
        block = recovery[recovery["band"] == band]
        ax.grid(axis="y", **GRID_KW)
        for region in regions:
            mean = block[block["region"] == region].groupby("n_partners")["residual_fraction"].mean()
            ax.plot(mean.index, mean.values, "-o", color=REGION_COLORS[region], markersize=3.6,
                    linewidth=1.3, label=REGION_LABELS.get(region, region), zorder=3)
        ax.set_title(band, fontsize=8)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("other regions present")
        ax.set_ylim(bottom=0)
        nice_axis(ax)
    axes[0].set_ylabel("target power\nnot reproduced")
    axes[-1].legend(frameon=False, fontsize=6.2)
    fig.tight_layout()
    return fig


def plot_rank_grid(
    adequacy: pd.DataFrame,
    *,
    ranks: Sequence[int],
    value: str = "worst_condition",
    bar: float = 0.99,
    figsize: tuple[float, float] = (4.2, 3.6),
):
    """Worst-cell fit over the within x cross rank grid, with the adequacy frontier drawn."""
    table = adequacy.set_index("label")
    grid = np.full((len(ranks), len(ranks)), np.nan)
    for i, r_w in enumerate(ranks):
        for j, r_c in enumerate(ranks):
            label = f"w{r_w}_c{r_c}"
            if label in table.index:
                grid[i, j] = float(table.loc[label, value])
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(grid, cmap="cividis", vmin=min(0.90, np.nanmin(grid)), vmax=max(1.0, np.nanmax(grid)),
                      origin="lower")
    for i in range(len(ranks)):
        for j in range(len(ranks)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=6.2,
                        color="white" if grid[i, j] < 0.97 else INK,
                        fontweight="bold" if grid[i, j] >= bar else "normal")
    ax.set_xticks(range(len(ranks))); ax.set_yticks(range(len(ranks)))
    ax.set_xticklabels(ranks); ax.set_yticklabels(ranks)
    ax.set_xlabel("cross-region rank $r_c$"); ax.set_ylabel("within-region rank $r_w$")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=f"worst cell, $R^2$ / ceiling (bold ≥ {bar})")
    fig.tight_layout()
    return fig


def plot_rank_marginals(
    adequacy: pd.DataFrame,
    *,
    ranks: Sequence[int],
    dense_value: float | None,
    value: str = "worst_condition",
    bar: float = 0.99,
    figsize: tuple[float, float] = (7.4, 2.8),
):
    """The two pure costs side by side: squeeze one side while the other stays dense.

    Left: cross-region rank with within-region dense. Right: within-region rank with
    cross-region dense. The dashed line is the fully dense model. Drawn on the same axis so
    the eye can compare them -- and against the total-drive-rank panel, which is the check
    on whether the comparison at equal rank means anything.
    """
    table = adequacy.set_index("label")
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, side, letter in ((axes[0], "cross", "a"), (axes[1], "within", "b")):
        ax.grid(axis="y", **GRID_KW)
        labels = [f"wdense_c{r}" for r in ranks] if side == "cross" else [f"w{r}_cdense" for r in ranks]
        values = [float(table.loc[l, value]) if l in table.index else np.nan for l in labels]
        errors = [float(table.loc[l, "seed_spread"]) if l in table.index and "seed_spread" in table else 0.0
                  for l in labels]
        ax.errorbar(ranks, values, yerr=errors, fmt="-o", color=CORRECTED_COLOR if side == "cross" else REPORTED_COLOR,
                    markersize=4.5, linewidth=1.6, capsize=2, zorder=3)
        if dense_value is not None:
            ax.axhline(dense_value, color=MUTED_INK, linewidth=0.8, linestyle=":", zorder=1)
        ax.axhline(bar, color=INK, linewidth=0.8, linestyle="--", zorder=2)
        ax.set_xscale("log"); ax.set_xticks(list(ranks)); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel(f"{side}-region rank (other side dense)")
        ax.set_ylabel("worst cell, $R^2$ / ceiling")
        nice_axis(ax); _panel(ax, letter)

    ax = axes[2]
    ax.grid(axis="y", **GRID_KW)
    grid = adequacy[adequacy["label"].str.match(r"^w\d+_c\d+$")]
    if not grid.empty and "drive_rank" in grid.columns:
        ax.scatter(grid["drive_rank"], grid[value], s=18, color=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE,
                   linewidth=0.5, zorder=3, label="grid cells")
    for side, color, pattern in (("cross", CORRECTED_COLOR, r"^wdense_c\d+$"), ("within", REPORTED_COLOR, r"^w\d+_cdense$")):
        m = adequacy[adequacy["label"].str.match(pattern)]
        if not m.empty and "drive_rank" in m.columns:
            ax.scatter(m["drive_rank"], m[value], s=26, color=color, edgecolor="white", linewidth=0.5,
                       zorder=4, label=f"{side} marginal")
    ax.axhline(bar, color=INK, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xlabel("total drive rank  $r_w + 3\\,r_c$")
    ax.legend(frameon=False, fontsize=6.0, loc="lower right")
    nice_axis(ax); _panel(ax, "c")
    fig.tight_layout()
    return fig


__all__ += [
    "plot_band_recovery_by_rung",
    "plot_ladder_curves",
    "plot_partner_matrix",
    "plot_rank_grid",
    "plot_rank_marginals",
]


def plot_band_recovery_pooled(
    recovery: pd.DataFrame,
    *,
    bands: Sequence[str] = ("0-5 Hz", "5-10 Hz", "10-20 Hz", "20-50 Hz"),
    figsize: tuple[float, float] = (7.4, 3.0),
):
    """All bands on one axis, pooled over regions, so the frequency dependence is the plot.

    Left: fraction of target power not reproduced, log scale, one line per band, the band
    across regions as a ribbon. Right: the same normalised to the single-region rung, so
    every band starts at 1 and the depth of the drop is the network's contribution to that
    band. The slow band should barely move; the fast bands are where partners act.
    """
    palette = plt.get_cmap("cividis")(np.linspace(0.05, 0.9, len(bands)))
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, normalise, letter in ((axes[0], False, "a"), (axes[1], True, "b")):
        ax.grid(axis="y", **GRID_KW)
        for color, band in zip(palette, bands):
            block = recovery[recovery["band"] == band]
            per_region = block.groupby(["region", "n_partners"])["residual_fraction"].mean().unstack("n_partners")
            if normalise:
                per_region = per_region.div(per_region[0], axis=0)
            rungs = per_region.columns.values
            ax.fill_between(rungs, per_region.min(axis=0), per_region.max(axis=0), color=color,
                            alpha=0.18, linewidth=0, zorder=2)
            ax.plot(rungs, per_region.mean(axis=0), "-o", color=color, markersize=4, linewidth=1.6,
                    label=band, zorder=3)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("other regions present")
        if normalise:
            ax.set_ylabel("residual, relative to the region alone")
            ax.axhline(1.0, color=MUTED_INK, linewidth=0.7, linestyle=":", zorder=1)
            ax.set_ylim(0, 1.05)
        else:
            ax.set_yscale("log")
            ax.set_ylabel("target power not reproduced")
        ax.legend(frameon=False, fontsize=6.4, loc="lower left" if normalise else "upper right")
        nice_axis(ax)
        if not normalise:
            # nice_axis installs a linear MaxNLocator, which leaves a log axis unlabelled;
            # put explicit decades and half-decades back.
            import matplotlib.ticker as mticker
            ax.yaxis.set_major_locator(mticker.FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3, 1.0]))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        _panel(ax, letter)
    fig.tight_layout()
    return fig


def plot_ladder_cell_heatmap(
    fit: pd.DataFrame,
    *,
    value: str = "r2_vs_ceiling",
    bar: float | None = 0.99,
    figsize: tuple[float, float] = (4.6, 4.4),
):
    """Every region x condition cell at every rung, as one grid.

    Twelve rows, four columns. The reader sees three things at once: whether cells clear
    the bar (bold), whether conditions sit together within a region (the minimax check),
    and how much each cell gains from partners (left to right).
    """
    regions = [r for r in REGION_COLORS if r in set(fit["region"])]
    conditions = [c for c in CONDITION_ORDER if c in set(fit["condition"])]
    rungs = sorted(fit["n_partners"].unique())
    table = fit.groupby(["region", "condition", "n_partners"])[value].mean()
    grid = np.array([[table.get((r, c, k), np.nan) for k in rungs] for r in regions for c in conditions])
    fig, ax = plt.subplots(figsize=figsize)
    lo = float(np.nanmin(grid)); hi = float(np.nanmax(grid))
    image = ax.imshow(grid, cmap="cividis", vmin=lo, vmax=hi, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.2,
                        color="white" if v < lo + 0.5 * (hi - lo) else INK,
                        fontweight="bold" if bar is not None and v >= bar else "normal")
    ax.set_xticks(range(len(rungs))); ax.set_xticklabels([str(k) for k in rungs])
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([f"{REGION_LABELS.get(r, r)} · {CONDITION_SHORT_LABELS.get(c, c)}" for r in regions for c in conditions],
                       fontsize=6.4)
    for boundary in range(len(conditions), grid.shape[0], len(conditions)):
        ax.axhline(boundary - 0.5, color="white", linewidth=1.6)
    ax.set_xlabel("other regions present")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 label="$R^2$ / ceiling" + (f"  (bold ≥ {bar})" if bar is not None else ""))
    fig.tight_layout()
    return fig


__all__ += ["plot_band_recovery_pooled", "plot_ladder_cell_heatmap"]


def plot_ceiling_comparison(
    cell: pd.DataFrame,
    old_by_region: Mapping[str, float],
    *,
    figsize: tuple[float, float] = (7.4, 2.8),
):
    """The chapter's ceiling beside the matched one, per region and per region x condition.

    Left: the unweighted per-component mean the chapter divided by, the variance-weighted
    version, and the whole-trajectory split-half reliability the R^2 is actually bounded by.
    Right: the matched ceiling per condition -- interactive face has five times the trials
    and the highest reliability in every region, which a pooled ceiling hid.
    """
    regions = [r for r in REGION_COLORS if r in set(cell["region"])]
    pooled = cell[cell["condition"] == "all"].set_index("region")
    index = np.arange(len(regions))
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.grid(axis="y", **GRID_KW)
    width = 0.27
    series = [
        ("chapter: unweighted mean", [old_by_region.get(r, np.nan) for r in regions], NULL_COLOR),
        ("variance-weighted mean", pooled.loc[regions, "variance_weighted_component_mean"].values, REPORTED_COLOR),
        ("matched: whole trajectory", pooled.loc[regions, "reliability"].values, CORRECTED_COLOR),
    ]
    # Bars start at the axis floor, not at zero: the repo's savefig patch strips clipping
    # masks so Illustrator can edit the art, and an unclipped bar body reaching down to
    # y = 0 on an axis zoomed to [0.95, 1] makes bbox_inches="tight" grow the figure to
    # include it -- the first render came out 150 inches tall.
    floor = 0.95
    for k, (label, values, color) in enumerate(series):
        values = np.asarray(values, dtype=float)
        ax.bar(index + (k - 1) * width, values - floor, width * 0.92, bottom=floor, color=color,
               edgecolor=NEUTRAL_EDGE, linewidth=0.5, label=label, zorder=3)
    ax.set_ylim(floor, 1.0)
    ax.set_xticks(index); ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
    ax.set_ylabel("ceiling on $R^2$")
    ax.legend(frameon=False, fontsize=6.0, loc="lower left")
    nice_axis(ax); _panel(ax, "a")

    ax = axes[1]
    ax.grid(axis="y", **GRID_KW)
    conditions = [c for c in CONDITION_ORDER if c in set(cell["condition"])]
    width = 0.8 / len(conditions)
    floor = 0.97
    for k, condition in enumerate(conditions):
        block = cell[cell["condition"] == condition].set_index("region")
        values = block.loc[regions, "reliability"].values.astype(float)
        ax.bar(index + (k - (len(conditions) - 1) / 2) * width, values - floor, width * 0.9, bottom=floor,
               color=CONDITION_COLORS.get(condition, NEUTRAL_FILL), edgecolor=NEUTRAL_EDGE, linewidth=0.5,
               label=CONDITION_SHORT_LABELS.get(condition, condition), zorder=3)
    ax.set_ylim(floor, 1.0)
    ax.set_xticks(index); ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
    ax.set_ylabel("matched ceiling, per condition")
    ax.legend(frameon=False, fontsize=6.0, loc="lower left", ncol=3)
    nice_axis(ax); _panel(ax, "b")
    fig.tight_layout()
    return fig


__all__ += ["plot_ceiling_comparison"]


# ======================================================================================
# Flow over time
# ======================================================================================

CELL_COLORS = ("#2a6f8f", "#b8621f", "#6a6a6a", "#2c6b4f", "#8a4f9e")


def plot_flow_matrix(
    flow: pd.DataFrame,
    *,
    cells: Sequence[str],
    condition: str | None = None,
    normalise: str = "none",
    figsize: tuple[float, float] = (7.6, 6.4),
):
    """``|c_{s->t}(t)|`` for every source (columns) into every target (rows), one line per cell.

    The diagonal is each region's own recurrence. Seed-averaged, one condition. A narrowing
    channel shows up as the off-diagonal panels flattening while the diagonal is left alone --
    or, if the network reroutes, as the diagonal growing to compensate.
    """
    regions = [r for r in REGION_COLORS if r in set(flow["target"])]
    flow = flow[flow["time_s"] >= sorted(flow["time_s"].unique())[2]]  # drop the initial-state transient
    if normalise == "share":
        # Each pathway's energy as a share of the target's total drive energy at that time
        # step. Raw norms are not comparable across cells -- a rank-1 model compensates with
        # larger weights everywhere -- but shares are.
        keys = [k for k in ("label", "seed") if k in flow.columns] + ["target", "condition", "time_s"]
        total = flow.assign(sq=flow["current_norm"] ** 2).groupby(keys)["sq"].transform("sum")
        flow = flow.assign(current_norm=np.where(total > 0, flow["current_norm"] ** 2 / total, np.nan))
    # Either one condition and several cells (how the channel changes the flow), or one cell
    # and the three conditions (how the flow differs between fixation types).
    if condition is not None:
        block = flow[(flow["condition"] == condition) & (flow["label"].isin(cells))]
        lines = [(cell, color, block[block["label"] == cell]) for cell, color in zip(cells, CELL_COLORS)]
        title = f"{CONDITION_LABELS.get(condition, condition)} · inter-regional current over time"
    else:
        block = flow[flow["label"] == cells[0]]
        lines = [(CONDITION_SHORT_LABELS.get(c, c), CONDITION_COLORS.get(c, INK), block[block["condition"] == c])
                 for c in CONDITION_ORDER if c in set(block["condition"])]
        title = f"{cells[0]} · inter-regional current over time, by fixation type"
    fig, axes = plt.subplots(len(regions), len(regions), figsize=figsize, sharex=True,
                             sharey=(normalise == "share"))
    for i, target in enumerate(regions):
        for j, source in enumerate(regions):
            ax = axes[i, j]
            ax.grid(axis="y", **GRID_KW)
            for name, color, sub in lines:
                trace = (sub[(sub["source"] == source) & (sub["target"] == target)]
                         .groupby("time_s")["current_norm"].mean())
                ax.plot(trace.index, trace.values, color=color, linewidth=1.2,
                        label=name if (i, j) == (0, 1) else None, zorder=3)
            ax.axvline(0.0, color=MUTED_INK, linewidth=0.6, linestyle=":", zorder=1)
            if i == j:
                ax.set_facecolor("#f3f3f3")
            if i == 0:
                ax.set_title(f"from {REGION_LABELS.get(source, source)}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"into {REGION_LABELS.get(target, target)}\n"
                              + ("share of drive" if normalise == "share" else "|current|"), fontsize=7)
            if i == len(regions) - 1:
                ax.set_xlabel("time from fixation (s)", fontsize=7)
            ax.tick_params(labelsize=6)
            nice_axis(ax, y_ticks=3)
    axes[0, 1].legend(frameon=False, fontsize=6.2, loc="upper right")
    fig.suptitle(title + (" (share of target's drive energy)" if normalise == "share" else ""),
                 fontsize=8.5, y=0.995)
    fig.tight_layout()
    return fig


def plot_flow_time_by_condition(
    decomposed: pd.DataFrame,
    *,
    cells: Sequence[str],
    value: str = "cross_fraction",
    ylabel: str = "cross-region share of drive energy",
    figsize: tuple[float, float] = (7.6, 5.4),
):
    """Rows: fixation types. Columns: target regions. Lines: cells. Seed-averaged over time.

    For ``cross_fraction`` this is *when* a region is driven by the network rather than by
    itself, and how a narrowing channel changes that -- per fixation type, so the question
    of whether interactive face draws on the network at a different time or to a different
    degree is answered directly.
    """
    regions = [r for r in REGION_COLORS if r in set(decomposed["target"])]
    conditions = [c for c in CONDITION_ORDER if c in set(decomposed["condition"])]
    # The first two bins carry the trained initial state's transient, not dynamics.
    start = sorted(decomposed["time_s"].unique())[2]
    block = decomposed[decomposed["label"].isin(cells) & (decomposed["time_s"] >= start)]
    mean = block.groupby(["label", "target", "condition", "time_s"])[value].mean().reset_index()
    fig, axes = plt.subplots(len(conditions), len(regions), figsize=figsize, sharex=True, sharey=True)
    for i, condition in enumerate(conditions):
        for j, region in enumerate(regions):
            ax = axes[i, j]
            ax.grid(axis="y", **GRID_KW)
            for color, cell in zip(CELL_COLORS, cells):
                trace = mean[(mean["label"] == cell) & (mean["target"] == region) & (mean["condition"] == condition)]
                ax.plot(trace["time_s"], trace[value], color=color, linewidth=1.2,
                        label=cell if (i, j) == (0, 0) else None, zorder=3)
            ax.axvline(0.0, color=MUTED_INK, linewidth=0.6, linestyle=":", zorder=1)
            if i == 0:
                ax.set_title(REGION_LABELS.get(region, region), fontsize=8)
            if j == 0:
                ax.set_ylabel(CONDITION_SHORT_LABELS.get(condition, condition), fontsize=7.5)
            if i == len(conditions) - 1:
                ax.set_xlabel("time from fixation (s)", fontsize=7)
            ax.tick_params(labelsize=6)
            nice_axis(ax, y_ticks=3)
    axes[0, 0].legend(frameon=False, fontsize=6.2, loc="best")
    fig.supylabel(ylabel, fontsize=8)
    fig.tight_layout()
    return fig


def plot_property_vs_rank(
    props: pd.DataFrame,
    prop: str,
    *,
    rank_column: str,
    ylabel: str = "",
    reference: float | None = None,
    figsize: tuple[float, float] = (7.4, 2.5),
):
    """One property against bottleneck rank: panels = regions, lines = fixation types."""
    regions = [r for r in REGION_COLORS if r in set(props["region"])]
    conditions = [c for c in CONDITION_ORDER if c in set(props["condition"])]
    fig, axes = plt.subplots(1, len(regions), figsize=figsize, sharey=True)
    for ax, region in zip(np.atleast_1d(axes), regions):
        ax.grid(axis="y", **GRID_KW)
        for condition in conditions:
            block = props[(props["region"] == region) & (props["condition"] == condition)]
            agg = block.groupby(rank_column)[prop].agg(["mean", "std"])
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"].fillna(0), fmt="-o",
                        color=CONDITION_COLORS.get(condition, INK), markersize=3.4, linewidth=1.2,
                        capsize=1.5, label=CONDITION_SHORT_LABELS.get(condition, condition), zorder=3)
        if reference is not None:
            ax.axhline(reference, color=MUTED_INK, linewidth=0.7, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ticks = sorted(props[rank_column].unique())
        ax.set_xticks(ticks)
        # The largest rank is the dense side (the full width); name it as such.
        ax.set_xticklabels(["dense" if t == max(ticks) and len(ticks) > 1 else f"{t:g}" for t in ticks])
        ax.set_title(REGION_LABELS.get(region, region), fontsize=8)
        ax.set_xlabel({"rank_cross": "cross-region rank", "rank_within": "within-region rank"}.get(rank_column, rank_column))
        nice_axis(ax)
    np.atleast_1d(axes)[0].set_ylabel(ylabel or prop)
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=6.0)
    fig.tight_layout()
    return fig


def plot_condition_fit_vs_rank(
    fit: pd.DataFrame,
    *,
    dense_by_condition: Mapping[str, float] | None = None,
    bar: float = 0.98,
    figsize: tuple[float, float] = (7.4, 2.8),
):
    """Fit per fixation type along each marginal, worst cell over regions and seeds averaged.

    ``fit`` carries ``rank_within`` and ``rank_cross``. Left: cross rank with within dense.
    Right: within rank with cross dense. If one fixation type's line drops away from the
    others as the channel narrows, that type is the one the channel is carrying.
    """
    conditions = [c for c in CONDITION_ORDER if c in set(fit["condition"])]
    hidden = int(max(fit["rank_within"].max(), fit["rank_cross"].max()))
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, (side, other), letter in zip(axes, (("rank_cross", "rank_within"), ("rank_within", "rank_cross")), "ab"):
        block = fit[fit[other] == hidden]
        block = block[block[side] < hidden]
        ax.grid(axis="y", **GRID_KW)
        for condition in conditions:
            per = (block[block["condition"] == condition]
                   .groupby([side, "seed"])["r2_vs_ceiling"].min().groupby(level=0).agg(["mean", "std"]))
            ax.errorbar(per.index, per["mean"], yerr=per["std"].fillna(0), fmt="-o",
                        color=CONDITION_COLORS.get(condition, INK), markersize=4, linewidth=1.4, capsize=2,
                        label=CONDITION_SHORT_LABELS.get(condition, condition), zorder=3)
            if dense_by_condition and condition in dense_by_condition:
                ax.axhline(dense_by_condition[condition], color=CONDITION_COLORS.get(condition, INK),
                           linewidth=0.7, linestyle=":", zorder=1)
        ax.axhline(bar, color=INK, linewidth=0.8, linestyle="--", zorder=2)
        ax.set_xscale("log"); ticks = sorted(block[side].unique()); ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel(f"{side.replace('rank_', '')}-region rank (other side dense)")
        nice_axis(ax); _panel(ax, letter)
    axes[0].set_ylabel("worst region, $R^2$ / ceiling")
    axes[0].legend(frameon=False, fontsize=6.2, loc="lower right")
    fig.tight_layout()
    return fig


__all__ += [
    "plot_condition_fit_vs_rank",
    "plot_flow_matrix",
    "plot_flow_time_by_condition",
    "plot_property_vs_rank",
]
