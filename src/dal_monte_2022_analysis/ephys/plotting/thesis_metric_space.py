"""Trace-shape metric figures: schematic, 2D space, and the CV control.

The chapter describes each unit's response with two orthogonal quantities rather
than one composite score:

``response_duration_ms``
    Dominant-peak width: the full width at half maximum of the excess response.
``peak_isolation``
    Dominant-peak prominence, ``1 - P2/P1``, where P1 is the dominant peak's
    topographic prominence and P2 the largest prominence at least 250 ms away.
    High values mean a single clear peak; low values mean the trace carries
    several peaks of comparable prominence.

They are near-independent (Spearman rho about -0.26 among selective units), and
both are essentially insensitive to trial count -- rho(log N, .) = +0.12 and
-0.001 -- unlike the prominence-based composite they replace, which sits at
-0.76 and is therefore unusable for any comparison across fixation categories.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

from dal_monte_2022_analysis.ephys.analysis.fixation_peakiness import (
    DominantPeakDecomposition,
)
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    NEUTRAL_EDGE,
    NEUTRAL_FILL,
    REGION_ORDER,
    add_analysis_window_bars,
    nice_axis,
    ordinal,
    region_label,
)

WIDTH_LABEL = "Dominant-peak width (ms)"
HALFWIDTH_LABEL = WIDTH_LABEL  # backwards-compatible alias
DURATION_LABEL = HALFWIDTH_LABEL  # backwards-compatible alias
PROMINENCE_LABEL = "Dominant-peak prominence  $1 - P_2/P_1$"
ISOLATION_LABEL = PROMINENCE_LABEL  # backwards-compatible alias
#: Plain-language poles of each axis, used for the corner labels. Width runs
#: narrow-to-wide. Prominence runs from a trace carrying several peaks of
#: comparable prominence to one carrying a single clear peak -- hence
#: multi-peak / single-peak, which say what the trace looks like rather than
#: re-using "dominant", the word already in the metric's own name.
WIDTH_POLES: tuple[str, str] = ("Narrow", "Wide")
PROMINENCE_POLES: tuple[str, str] = ("multi-peak", "single-peak")

#: Corner roles in the duration x isolation space, as (label, duration side,
#: isolation side) where +1 means high.
CORNER_ROLES: tuple[tuple[str, int, int], ...] = (
    ("Narrow, single-peak", -1, +1),
    ("Narrow, multi-peak", -1, -1),
    ("Wide, single-peak", +1, +1),
    ("Wide, multi-peak", +1, -1),
)
CORNER_COLORS: dict[str, str] = {
    "Narrow, single-peak": "#c03a2b",
    "Narrow, multi-peak": "#e6a817",
    "Wide, single-peak": "#2878b5",
    "Wide, multi-peak": "#3f9c45",
}


def plot_isolation_schematic(
    decomposition: DominantPeakDecomposition,
    *,
    unit_label: str,
    duration_ms: Optional[float] = None,
    condition: str = "face_interactive",
    figure_width_in: float = 4.4,
    figure_height_in: float = 2.6,
    display_window_s: tuple[float, float] = (-1.0, 1.0),
) -> plt.Figure:
    """Standalone construction of both metrics on one example trace.

    Kept out of the example-unit row deliberately: sharing a panel forces that
    row wider and mixes two jobs -- showing what a response looks like, and
    showing how it is measured.
    """
    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in))
    centers_s = decomposition.centers_ms / 1000.0
    values = decomposition.values_hz
    mask = (centers_s >= display_window_s[0]) & (centers_s <= display_window_s[1])
    color = CONDITION_COLORS.get(condition, INK)
    ax.plot(centers_s[mask], values[mask], color=color, linewidth=1.6, zorder=5)
    ax.axvline(0.0, color=INK, linestyle="--", linewidth=0.7, zorder=2)

    if decomposition.primary_index is None:
        ax.set_xlabel("Time from fixation onset (s)", fontsize=7.5)
        return fig

    scale = float(decomposition.normalization_scale)
    primary_t = float(centers_s[decomposition.primary_index])
    primary_top = float(values[decomposition.primary_index])
    primary_base = float(decomposition.primary_reference_norm * scale)

    exclusion_s = float(decomposition.competition_exclusion_window_ms) / 1000.0
    ax.axvspan(
        primary_t - exclusion_s, primary_t + exclusion_s,
        color="#9a9a9a", alpha=0.14, linewidth=0, zorder=1,
    )

    x_low, x_high = ax.get_xlim()

    def _span(t_s, top, base, symbol, arrow_color):
        side = -1 if t_s > x_low + 0.72 * (x_high - x_low) else 1
        ax.plot([t_s - 0.13, t_s + 0.13], [base, base], color=arrow_color,
                linewidth=0.7, linestyle=(0, (3, 2)), zorder=6)
        ax.add_patch(FancyArrowPatch((t_s, base), (t_s, top), arrowstyle="<->",
                                     mutation_scale=6, linewidth=1.0, color=arrow_color,
                                     shrinkA=0, shrinkB=0, zorder=7))
        ax.annotate(symbol, xy=(t_s, base + 0.74 * (top - base)),
                    xytext=(7 if side > 0 else -7, 0), textcoords="offset points",
                    ha="left" if side > 0 else "right", va="center",
                    fontsize=8.5, color=arrow_color, zorder=9)

    _span(primary_t, primary_top, primary_base, "$P_1$", INK)
    if decomposition.secondary_index is not None:
        secondary_t = float(centers_s[decomposition.secondary_index])
        _span(
            secondary_t,
            float(values[decomposition.secondary_index]),
            float(decomposition.secondary_reference_norm * scale),
            "$P_2$",
            MUTED_INK,
        )

    # Duration: half-max line across the excess response.
    lines = [f"$P_1$ = {decomposition.primary_prominence:.2f}",
             f"$P_2$ = {decomposition.secondary_prominence:.2f}",
             f"prominence = {1.0 - decomposition.competition_ratio:.2f}"]
    if duration_ms is not None and np.isfinite(duration_ms):
        in_window = (decomposition.centers_ms >= -500.0) & (decomposition.centers_ms <= 500.0)
        windowed = values[in_window]
        baseline = float(np.quantile(windowed, 0.10))
        half = baseline + 0.5 * (float(windowed.max()) - baseline)
        above = decomposition.centers_ms[in_window][windowed >= half] / 1000.0
        if above.size:
            # Draw the half-max reference only across the measured span. A full
            # axhline reads as a threshold applied to the whole trace, which is
            # not what FWHM is.
            pad = 0.10
            ax.plot(
                [float(above.min()) - pad, float(above.max()) + pad], [half, half],
                color="#4f81b8", linewidth=0.9, linestyle=(0, (5, 2)), zorder=4,
            )
            ax.annotate(
                "", xy=(float(above.min()), half), xytext=(float(above.max()), half),
                arrowprops={"arrowstyle": "<->", "color": "#4f81b8", "linewidth": 1.1,
                            "shrinkA": 0, "shrinkB": 0},
                zorder=8,
            )
            ax.annotate(
                f"{duration_ms:.0f} ms",
                xy=(float(np.mean([above.min(), above.max()])), half),
                xytext=(0, -8), textcoords="offset points",
                ha="center", va="top", fontsize=6.6, color="#2f4b6e", zorder=9,
            )
        lines.append(f"width = {duration_ms:.0f} ms")

    ax.text(
        0.015, 0.975, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
        fontsize=6.2, color=INK, linespacing=1.35, zorder=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "#cccccc", "linewidth": 0.5, "alpha": 0.93},
    )
    ax.set_title(
        f"{unit_label} · {CONDITION_SHORT_LABELS.get(condition, condition)} trace",
        fontsize=8.2,
    )
    ax.set_ylabel("Firing rate (Hz)", fontsize=7.5)
    nice_axis(ax, y_ticks=4)
    data_low, data_high = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if data_low <= t <= data_high]
    add_analysis_window_bars(ax, time_scale=1e-3)
    ax.set_yticks(ticks)
    ax.spines["left"].set_bounds(min(ticks), max(ticks))
    ax.set_xlabel("Time from fixation onset (s)", fontsize=7.5)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(length=2.5, pad=1.5)
    fig.tight_layout()
    return fig


def select_corner_units(
    trace_shape: pd.DataFrame,
    *,
    region: Optional[str] = None,
    duration_column: str = "response_duration_ms",
    isolation_column: str = "peak_isolation",
    min_mean_fr_hz: float = 3.0,
    max_mean_fr_hz: float = 30.0,
) -> pd.DataFrame:
    """One unit per corner of the duration x isolation space.

    Corners are defined on within-pool percentile ranks rather than absolute
    cut-offs, so the picks are the most extreme *available* units rather than
    hostages to an arbitrary threshold.
    """
    pool = trace_shape.loc[trace_shape["is_selective"]].copy()
    if region is not None:
        pool = pool.loc[pool["region"].astype(str) == str(region)]
    rate = pd.to_numeric(pool["mean_fr_hz"], errors="coerce")
    pool = pool.loc[rate.between(min_mean_fr_hz, max_mean_fr_hz)]
    pool = pool.dropna(subset=[duration_column, isolation_column])
    if pool.empty:
        return pool

    pool["_d"] = pool[duration_column].rank(pct=True)
    pool["_i"] = pool[isolation_column].rank(pct=True)
    rows = []
    for label, duration_side, isolation_side in CORNER_ROLES:
        score = (
            (pool["_d"] if duration_side > 0 else 1.0 - pool["_d"])
            + (pool["_i"] if isolation_side > 0 else 1.0 - pool["_i"])
        )
        pick = pool.loc[score.idxmax()].to_dict()
        pick["corner"] = label
        rows.append(pick)
    return pd.DataFrame(rows)


def plot_metric_space_panel(
    trace_shape: pd.DataFrame,
    *,
    corners: Optional[pd.DataFrame] = None,
    condition_traces: Optional[pd.DataFrame] = None,
    regions: Sequence[str] = REGION_ORDER,
    duration_column: str = "response_duration_ms",
    isolation_column: str = "peak_isolation",
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.6,
    inset_size: float = 0.235,
    display_window_ms: tuple[float, float] = (-500.0, 500.0),
) -> tuple[plt.Figure, pd.DataFrame]:
    """Halfwidth x isolation scatter per region, with corner exemplars.

    Presented as a space rather than two histograms because the population is
    unimodal on both axes: there is no narrow/wide or single/multi-peak
    dichotomy to threshold,
    and a scatter says so honestly while still letting extreme units be named.

    When ``condition_traces`` is supplied each corner unit's own firing-rate
    trace -- for its preferred fixation category only -- is inset next to its
    point, so the reader can see what a coordinate in this space looks like
    without leaving the panel.
    """
    selective = trace_shape.loc[trace_shape["is_selective"]]
    fig, axes = plt.subplots(
        1, len(regions), figsize=(figure_width_in, figure_height_in), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)

    lookup = {}
    if condition_traces is not None:
        lookup = {
            (row.unit_key, row.condition): (row.bin_centers_s_rel, row.trace_hz)
            for row in condition_traces.itertuples()
        }

    # Inset corners are placed away from the data centroid so they do not sit on
    # top of the cloud they are annotating.
    inset_anchor = {
        "Narrow, single-peak": (0.015, 0.735),
        "Narrow, multi-peak": (0.015, 0.02),
        "Wide, single-peak": (0.75, 0.735),
        "Wide, multi-peak": (0.75, 0.02),
    }

    summary = []
    for ax, region in zip(axes, regions):
        region_units = selective.loc[selective["region"].astype(str) == str(region)]
        ax.scatter(
            region_units[duration_column], region_units[isolation_column],
            s=6, color=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE, linewidth=0.2, alpha=0.9,
        )
        ax.axvline(float(selective[duration_column].median()), color=MUTED_INK,
                   linewidth=0.7, linestyle=":", zorder=1)
        ax.axhline(float(selective[isolation_column].median()), color=MUTED_INK,
                   linewidth=0.7, linestyle=":", zorder=1)
        if corners is not None:
            marked = corners.loc[corners["region"].astype(str) == str(region)]
            for _, unit in marked.iterrows():
                corner = str(unit["corner"])
                color = CORNER_COLORS.get(corner, INK)
                ax.scatter(
                    [unit[duration_column]], [unit[isolation_column]],
                    s=40, color=color, edgecolor="white", linewidth=0.9, zorder=7,
                )
                entry = lookup.get((unit["unit_key"], unit.get("condition")))
                if entry is None:
                    continue
                anchor = inset_anchor.get(corner)
                if anchor is None:
                    continue
                inset = ax.inset_axes([anchor[0], anchor[1], inset_size, inset_size])
                centers_ms = np.asarray(entry[0], dtype=float) * 1000.0
                trace = np.asarray(entry[1], dtype=float)
                mask = (
                    (centers_ms >= display_window_ms[0]) & (centers_ms <= display_window_ms[1])
                )
                inset.plot(centers_ms[mask], trace[mask], color=color, linewidth=0.9)
                inset.axvline(0.0, color=MUTED_INK, linestyle="--", linewidth=0.5)
                inset.set_xticks([])
                inset.set_yticks([])
                inset.patch.set_alpha(0.85)
                for spine in inset.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(0.7)
                # Connect the inset to the point it describes.
                ax.annotate(
                    "",
                    xy=(unit[duration_column], unit[isolation_column]),
                    xytext=(anchor[0] + inset_size / 2, anchor[1] + inset_size / 2),
                    textcoords=ax.transAxes,
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.6,
                                "alpha": 0.7, "shrinkA": 0, "shrinkB": 3},
                    zorder=4,
                )
        ax.set_title(f"{region_label(region)}  (n={len(region_units)})", fontsize=8.2)
        nice_axis(ax, y_ticks=4)
        summary.append(
            {
                "region": region_label(region),
                "n_units": int(len(region_units)),
                "width_median_ms": float(region_units[duration_column].median()),
                "isolation_median": float(region_units[isolation_column].median()),
                "spearman_rho": float(
                    region_units[[duration_column, isolation_column]]
                    .corr(method="spearman").iloc[0, 1]
                ),
            }
        )
    axes[0].set_ylabel(PROMINENCE_LABEL, fontsize=7.2)
    if corners is not None:
        fig.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markersize=6,
                       markerfacecolor=CORNER_COLORS[label], markeredgecolor="white",
                       label=label)
                for label, _, _ in CORNER_ROLES
            ],
            ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.075), fontsize=6.6,
        )
    # supxlabel without an explicit y: matplotlib reserves the strip once during
    # tight_layout, and the legend then sits below it in figure coordinates.
    fig.supxlabel(WIDTH_LABEL, fontsize=7.8)
    fig.tight_layout()
    return fig, pd.DataFrame(summary)


def plot_cv_trial_matched_control(
    unit_cv: pd.DataFrame,
    inflation: pd.DataFrame,
    *,
    regions: Sequence[str] = REGION_ORDER,
    conditions: Sequence[str] = CONDITION_ORDER,
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.5,
) -> plt.Figure:
    """Why the raw CV difference between fixation categories is not usable.

    Left: median CV of the mean trace before and after equalising trial counts
    within unit. Right: the mechanism -- interactive-face trials subsampled to
    smaller N, holding condition and neural signal fixed.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(figure_width_in, figure_height_in),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    ax = axes[0]
    x = np.arange(len(regions))
    width = 0.13
    for block, (variant, alpha_value, hatch) in enumerate(
        [("cv_full", 1.0, ""), ("cv_matched", 0.55, "//")]
    ):
        for index, condition in enumerate(conditions):
            values = [
                float(
                    unit_cv.loc[
                        unit_cv["region"].astype(str) == str(region), f"{condition}_{variant}"
                    ].median()
                )
                for region in regions
            ]
            offset = (block * len(conditions) + index - 2.5) * width
            ax.bar(
                x + offset, values, width=width * 0.9,
                color=CONDITION_COLORS[condition], alpha=alpha_value,
                edgecolor="white", linewidth=0.5, hatch=hatch,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([region_label(region) for region in regions])
    ax.set_ylabel("CV of mean trace\n(across time bins)", fontsize=7.2)
    ax.set_title("Raw (solid) vs trial-matched (hatched)", fontsize=8, pad=16)
    nice_axis(ax, y_ticks=4)
    ax.legend(
        handles=[
            Patch(facecolor=CONDITION_COLORS[c], label=CONDITION_SHORT_LABELS[c])
            for c in conditions
        ],
        ncol=3, fontsize=6.2, loc="lower center", bbox_to_anchor=(0.5, 1.06),
        handlelength=1.1, columnspacing=0.8,
    )

    ax = axes[1]
    grouped = inflation.groupby("n_trials_used")["inflation_ratio"].median()
    ax.plot(grouped.index, grouped.to_numpy(), marker="o", markersize=4,
            color=CONDITION_COLORS["face_interactive"], linewidth=1.4)
    ax.axhline(1.0, color=MUTED_INK, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Interactive-face trials used", fontsize=7.2)
    ax.set_ylabel("CV inflation\n(x full-data CV)", fontsize=7.2)
    ax.set_title("Same unit, same signal,\nfewer trials", fontsize=8)
    nice_axis(ax, y_ticks=4)
    fig.tight_layout()
    return fig


def plot_corner_example_traces(
    corners: pd.DataFrame,
    condition_traces: pd.DataFrame,
    *,
    conditions: Sequence[str] = CONDITION_ORDER,
    display_window_ms: tuple[float, float] = (-500.0, 500.0),
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.2,
) -> plt.Figure:
    """Condition-average traces of one unit per corner of the metric space.

    Trace-only rather than raster-plus-PSTH: the point is the *shape* of the
    mean response at each corner, and four rasters from four different sessions
    would add loading cost without adding to that argument.
    """
    lookup = {
        (row.unit_key, row.condition): (row.bin_centers_s_rel, row.trace_hz)
        for row in condition_traces.itertuples()
    }
    ordered = [
        corners.loc[corners["corner"] == label].iloc[0]
        for label, _, _ in CORNER_ROLES
        if not corners.loc[corners["corner"] == label].empty
    ]
    fig, axes = plt.subplots(
        1, len(ordered), figsize=(figure_width_in, figure_height_in)
    )
    axes = np.atleast_1d(axes)
    for ax, unit in zip(axes, ordered):
        for condition in conditions:
            entry = lookup.get((unit["unit_key"], condition))
            if entry is None:
                continue
            centers_ms = np.asarray(entry[0], dtype=float) * 1000.0
            trace = np.asarray(entry[1], dtype=float)
            mask = (centers_ms >= display_window_ms[0]) & (centers_ms <= display_window_ms[1])
            ax.plot(
                centers_ms[mask], trace[mask],
                color=CONDITION_COLORS[condition],
                linewidth=1.5 if condition == unit.get("condition") else 0.9,
                alpha=1.0 if condition == unit.get("condition") else 0.55,
                zorder=5 if condition == unit.get("condition") else 3,
            )
        ax.axvline(0.0, color=INK, linestyle="--", linewidth=0.7)
        ax.set_title(
            f"{str(unit['corner'])}\n{region_label(unit['region'])} {unit['uuid']}  "
            f"({unit['response_duration_ms']:.0f} ms, iso {unit['peak_isolation']:.2f})",
            fontsize=7.0,
            color=CORNER_COLORS.get(str(unit["corner"]), INK),
        )
        ax.set_xlabel("Time from fixation (ms)", fontsize=7)
        ax.set_xticks([-500, 0, 500])
        nice_axis(ax, y_ticks=3)
    axes[0].set_ylabel("Firing rate (Hz)", fontsize=7.5)
    fig.legend(
        handles=[
            Line2D([0], [0], color=CONDITION_COLORS[c], linewidth=1.5,
                   label=CONDITION_SHORT_LABELS[c])
            for c in conditions
        ],
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.03), fontsize=6.8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig
