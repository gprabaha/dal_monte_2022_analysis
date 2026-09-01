"""Figures and schematics for the spatial decay of pairwise spike coordination.

The schematics are not decoration.  The analysis exists because a confound is
not visible in the result tables -- within-region pairs share an array, so any
comparison against cross-region pairs is contaminated before it begins -- and
the reason electrode separation resolves it is a two-line argument that is much
easier to see than to read.  ``plot_confound_schematic`` and
``plot_method_schematic`` carry that argument; the rest carry the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from dal_monte_2022_analysis.ephys.analysis.fixation_pair_spatial_decay import (
    METRIC_LABELS,
    PEAK_METRIC,
    REGION_ORDER,
    SEPARATION_LABELS,
    exponential_decay,
)
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    REGION_LABELS,
    ThesisFigureSettings,
    apply_thesis_plot_style,
    nice_axis,
    save_thesis_figure,
)

REGION_COLORS: dict[str, str] = {
    "bla": "#c0392b",
    "accg": "#2c7fb8",
    "dmpfc": "#7b5aa6",
    "ofc": "#d95f0e",
}


@dataclass
class SpatialDecayPlotSettings(ThesisFigureSettings):
    """Figure output settings for the spatial-decay panels."""

    schematic_width_in: float = 7.2
    schematic_height_in: float = 2.9
    panel_width_in: float = 1.9
    panel_height_in: float = 1.8


def region_label(region: object) -> str:
    return REGION_LABELS.get(str(region), str(region).upper())


def condition_label(condition: object) -> str:
    return CONDITION_SHORT_LABELS.get(str(condition), str(condition).replace("_", " "))


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_confound_schematic(
    settings: SpatialDecayPlotSettings,
    *,
    stem: str = "fig01_confound_schematic",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Why "within beats across" cannot be read as biology, and what settles it.

    Left: the confound.  Within-region pairs sit on one array and cross-region
    pairs span two, so array identity is perfectly aligned with the comparison.
    Right: the two hypotheses make opposite predictions about the same
    measurement -- a shared reference contaminates every pair on an array
    equally and so is flat with separation, while local circuitry decays.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(settings.schematic_width_in, settings.schematic_height_in),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    # --- left: the confound -------------------------------------------------
    ax = axes[0]
    for x0, name, colour in ((0.06, "Region A", REGION_COLORS["bla"]),
                             (0.58, "Region B", REGION_COLORS["ofc"])):
        ax.add_patch(Rectangle((x0, 0.30), 0.36, 0.46, fill=False,
                               edgecolor=colour, linewidth=1.2))
        ax.text(x0 + 0.18, 0.80, f"{name}\narray", ha="center", va="bottom",
                fontsize=6.5, color=colour)
        for index in range(4):
            ax.add_patch(Circle((x0 + 0.07 + 0.073 * index, 0.53), 0.018,
                                color=colour, alpha=0.85))
    # within-region pair
    ax.annotate("", xy=(0.213, 0.53), xytext=(0.13, 0.53),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text(0.17, 0.44, "within\nregion", ha="center", va="top", fontsize=6, color=INK)
    # cross-region pair
    ax.annotate("", xy=(0.72, 0.53), xytext=(0.36, 0.53),
                arrowprops=dict(arrowstyle="<->", color=MUTED_INK, lw=1.3,
                                linestyle="dashed"))
    ax.text(0.50, 0.60, "across regions", ha="center", va="bottom", fontsize=6, color=MUTED_INK)
    ax.text(0.5, 0.18,
            "Within-region pairs share an array; cross-region pairs do not.\n"
            "A shared reference or common noise would raise the within-region\n"
            "measurement on its own, with no biology involved.",
            ha="center", va="top", fontsize=6, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0.05, 1.0); _bare(ax)
    ax.set_title("The confound", fontsize=8, color=INK)

    # --- right: the discriminating prediction --------------------------------
    ax = axes[1]
    x = np.linspace(0, 25, 200)
    ax.plot(x, np.full_like(x, 0.55), color="#c0392b", lw=1.5, linestyle="--",
            label="shared reference\n(flat)")
    ax.plot(x, 0.9 * np.exp(-x / 4.0) + 0.05, color="#2c7fb8", lw=1.5,
            label="local circuitry\n(decays)")
    ax.set_xlabel("Electrode separation", fontsize=7)
    ax.set_ylabel("Coordination", fontsize=7)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    ax.set_title("Two hypotheses, opposite predictions", fontsize=8, color=INK)
    nice_axis(ax)

    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_method_schematic(
    settings: SpatialDecayPlotSettings,
    *,
    stem: str = "fig02_method_schematic",
) -> tuple[plt.Figure, dict[str, Path]]:
    """What is measured on each pair, from spike trains to a single number."""
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1, 4, figsize=(settings.schematic_width_in, settings.schematic_height_in * 0.85)
    )
    rng = np.random.default_rng(3)

    # 1. the window and two spike trains
    ax = axes[0]
    for row, (offset, colour) in enumerate(((0.62, INK), (0.30, "#2c7fb8"))):
        times = rng.uniform(-500, 500, 26)
        ax.vlines(times, offset, offset + 0.18, color=colour, lw=0.7)
        ax.text(-560, offset + 0.09, f"unit {row + 1}", fontsize=6, color=colour,
                ha="right", va="center")
    ax.axvline(0, color=MUTED_INK, lw=0.8, linestyle=":")
    ax.text(0, 0.90, "fixation\nonset", ha="center", fontsize=5.5, color=MUTED_INK)
    ax.set_xlim(-620, 560); ax.set_ylim(0.1, 1.0); _bare(ax)
    ax.set_title("1. One fixation,\n±500 ms, 1 ms bins", fontsize=7, color=INK)

    # 2. observed correlation against the shifted null
    ax = axes[1]
    lags = np.linspace(-100, 100, 300)
    null = 0.30 + 0.02 * np.exp(-np.abs(lags) / 220)
    observed = null + 0.16 * np.exp(-np.abs(lags) / 6.0)
    ax.plot(lags, observed, color=INK, lw=1.2, label="observed")
    ax.plot(lags, null, color="#2c7fb8", lw=1.2, linestyle="--", label="circular-shift null")
    ax.fill_between(lags, null, observed, color="#d95f0e", alpha=0.28, linewidth=0)
    ax.set_xlabel("Lag (ms)", fontsize=6.5)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    ax.set_yticks([])
    ax.set_title("2. Average over fixations,\nsubtract the null", fontsize=7, color=INK)
    nice_axis(ax)

    # 3. the two components pulled out of the difference
    ax = axes[2]
    excess = 0.16 * np.exp(-np.abs(lags) / 6.0) + 0.022 * np.exp(-np.abs(lags) / 60.0)
    ax.plot(lags, excess, color="#d95f0e", lw=1.3)
    ax.axvspan(-2, 2, color=INK, alpha=0.16)
    ax.axvspan(20, 100, color=MUTED_INK, alpha=0.12)
    ax.axvspan(-100, -20, color=MUTED_INK, alpha=0.12)
    ax.text(0, excess.max() * 1.02, "peak\n±2 ms", ha="center", fontsize=5.5, color=INK)
    ax.text(60, excess.max() * 0.42, "shoulder\n20–100 ms", ha="center", fontsize=5.5,
            color=MUTED_INK)
    ax.set_xlabel("Lag (ms)", fontsize=6.5)
    ax.set_yticks([])
    ax.set_title("3. Two components,\nmeasured separately", fontsize=7, color=INK)
    nice_axis(ax)

    # 4. binned by electrode separation
    ax = axes[3]
    for index in range(6):
        ax.add_patch(Circle((0.14 + 0.145 * index, 0.72), 0.032,
                            color=MUTED_INK, alpha=0.75))
    for offset, (start, end, colour) in enumerate(
        ((0, 1, "#2c7fb8"), (0, 3, "#7b5aa6"), (0, 5, "#c0392b"))
    ):
        y = 0.52 - 0.145 * offset
        ax.annotate("", xy=(0.14 + 0.145 * end, y), xytext=(0.14 + 0.145 * start, y),
                    arrowprops=dict(arrowstyle="<->", color=colour, lw=1.1))
        ax.text(0.97, y, f"{end - start}", fontsize=5.5, color=colour, va="center")
    ax.text(0.55, 0.90, "channels", ha="center", fontsize=6, color=MUTED_INK)
    ax.text(0.5, 0.12, "one number per pair,\nbinned by separation",
            ha="center", fontsize=6, color=INK)
    ax.set_xlim(0, 1.06); ax.set_ylim(0.02, 1.0); _bare(ax)
    ax.set_title("4. Group by electrode\nseparation", fontsize=7, color=INK)

    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_decay_curves(
    table: pd.DataFrame,
    fits: pd.DataFrame,
    settings: SpatialDecayPlotSettings,
    *,
    references: Optional[pd.DataFrame] = None,
    metric: str = PEAK_METRIC,
    log_y: bool = True,
    stem: str = "fig03_decay_curves",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Measured decay per region with the fitted exponential, and the reference levels.

    A log y-axis is the default because the range spans two orders of magnitude
    across regions and a linear axis renders ACCg as a flat line at the bottom.
    """
    apply_thesis_plot_style()
    regions = [r for r in REGION_ORDER if r in set(table["region"])]
    fig, axes = plt.subplots(
        1, len(regions) + 1,
        figsize=(settings.panel_width_in * (len(regions) + 1), settings.panel_height_in + 0.5),
        squeeze=False,
    )
    flat = axes[0]

    for ax, region in zip(flat, regions):
        group = table.loc[table["region"] == region].sort_values("separation")
        colour = REGION_COLORS.get(region, MUTED_INK)
        ax.errorbar(
            group["separation"], group["mean"],
            yerr=[group["mean"] - group["ci_low"], group["ci_high"] - group["mean"]],
            fmt="o", markersize=3.4, color=colour, elinewidth=0.8, capsize=1.5,
        )
        row = fits.loc[fits["region"] == region]
        if len(row) and np.isfinite(row["length_constant"].iloc[0]):
            row = row.iloc[0]
            grid = np.linspace(group["separation"].min(), group["separation"].max(), 200)
            ax.plot(
                grid,
                exponential_decay(grid, row["amplitude"], row["length_constant"], row["offset"]),
                color=colour, lw=1.1, alpha=0.85,
            )
            ax.text(
                0.96, 0.94,
                f"$\\lambda$ = {row['length_constant']:.1f} ch\n$R^2$ = {row['r_squared']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=colour,
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Separation (channels)", fontsize=6.5)
        ax.set_title(f"{region_label(region)}  (n={int(group['n_pairs'].sum()):,})",
                     fontsize=7, color=INK)
        nice_axis(ax)
    flat[0].set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=6.5)

    # reference levels, on the same scale
    ax = flat[-1]
    if references is not None and len(references):
        colours = ["#7f8c8d", "#2c7fb8", "#c0392b"]
        positions = np.arange(len(references))
        ax.barh(positions, references["mean"], xerr=references["sem"],
                color=colours[: len(references)], edgecolor=INK, linewidth=0.5,
                error_kw={"elinewidth": 0.8})
        ax.axvline(0.0, color=MUTED_INK, lw=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(list(references["level"]), fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel(METRIC_LABELS.get(metric, metric), fontsize=6.5)
    ax.set_title("Reference levels", fontsize=7, color=INK)
    nice_axis(ax)

    fig.suptitle(
        "Coordination decays with electrode separation in every region",
        fontsize=8, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, save_thesis_figure(fig, settings, stem)


def plot_decay_by_condition(
    table: pd.DataFrame,
    fits: pd.DataFrame,
    settings: SpatialDecayPlotSettings,
    *,
    metric: str = PEAK_METRIC,
    stem: str = "fig04_decay_by_condition",
) -> tuple[plt.Figure, dict[str, Path]]:
    """The same decay, split by what the animal was fixating."""
    apply_thesis_plot_style()
    regions = [r for r in REGION_ORDER if r in set(table["region"])]
    fig, axes = plt.subplots(
        1, len(regions),
        figsize=(settings.panel_width_in * len(regions), settings.panel_height_in + 0.4),
        squeeze=False,
    )
    for ax, region in zip(axes[0], regions):
        for condition in CONDITION_ORDER:
            group = table.loc[
                (table["region"] == region) & (table["condition"] == condition)
            ].sort_values("separation")
            if group.empty:
                continue
            colour = CONDITION_COLORS.get(condition, MUTED_INK)
            ax.errorbar(
                group["separation"], group["mean"],
                yerr=[group["mean"] - group["ci_low"], group["ci_high"] - group["mean"]],
                fmt="o", markersize=3.0, color=colour, elinewidth=0.7, capsize=1.2,
                label=condition_label(condition),
            )
            row = fits.loc[(fits["region"] == region) & (fits["condition"] == condition)]
            if len(row) and np.isfinite(row["length_constant"].iloc[0]):
                row = row.iloc[0]
                grid = np.linspace(group["separation"].min(), group["separation"].max(), 200)
                ax.plot(
                    grid,
                    exponential_decay(
                        grid, row["amplitude"], row["length_constant"], row["offset"]
                    ),
                    color=colour, lw=1.0, alpha=0.8,
                )
        ax.set_yscale("log")
        ax.set_xlabel("Separation (channels)", fontsize=6.5)
        ax.set_title(region_label(region), fontsize=7, color=INK)
        nice_axis(ax)
    axes[0][0].set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=6.5)
    axes[0][-1].legend(frameon=False, fontsize=5.5, loc="upper right")
    fig.suptitle("The decay is the same whatever the animal was looking at",
                 fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, save_thesis_figure(fig, settings, stem)


def plot_fit_parameters(
    fits: pd.DataFrame,
    settings: SpatialDecayPlotSettings,
    *,
    by_condition: bool = False,
    stem: str = "fig05_fit_parameters",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Amplitude and length constant with bootstrap intervals.

    Splitting the fit into these two makes the result readable in one line:
    regions differ enormously in **how strongly** neighbouring neurons
    coordinate and hardly at all in **how far** that coordination reaches.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(settings.schematic_width_in * 0.85, settings.panel_height_in + 0.5)
    )
    regions = [r for r in REGION_ORDER if r in set(fits["region"])]

    for ax, (key, label, log) in zip(
        axes,
        (("amplitude", "Amplitude at zero separation", True),
         ("length_constant", "Length constant $\\lambda$ (channels)", False)),
    ):
        for index, region in enumerate(regions):
            rows = fits.loc[fits["region"] == region]
            if by_condition and "condition" in fits.columns:
                for offset, condition in enumerate(CONDITION_ORDER):
                    row = rows.loc[rows["condition"] == condition]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    x = index + (offset - 1) * 0.22
                    ax.plot([x, x], [row[f"{key}_low"], row[f"{key}_high"]],
                            color=CONDITION_COLORS.get(condition, MUTED_INK), lw=1.2)
                    ax.plot([x], [row[key]], marker="o", markersize=3.6,
                            color=CONDITION_COLORS.get(condition, MUTED_INK),
                            label=condition_label(condition) if index == 0 else None)
            else:
                row = rows.iloc[0]
                colour = REGION_COLORS.get(region, MUTED_INK)
                ax.plot([index, index], [row[f"{key}_low"], row[f"{key}_high"]],
                        color=colour, lw=1.4)
                ax.plot([index], [row[key]], marker="o", markersize=4.4, color=colour)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels([region_label(r) for r in regions], fontsize=6.5)
        ax.set_ylabel(label, fontsize=6.5)
        nice_axis(ax)
    if by_condition:
        axes[0].legend(frameon=False, fontsize=5.5, loc="best")
    axes[0].set_title("How strongly", fontsize=7, color=INK)
    axes[1].set_title("How far", fontsize=7, color=INK)
    fig.tight_layout()
    suffix = "_by_condition" if by_condition else ""
    return fig, save_thesis_figure(fig, settings, f"{stem}{suffix}")


def plot_condition_by_separation(
    table: pd.DataFrame,
    settings: SpatialDecayPlotSettings,
    *,
    stem: str = "fig06_condition_by_separation",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Does the condition difference grow or shrink with separation?

    Effect sizes, not means: at these sample sizes a mean difference of a few
    ten-thousandths reaches significance and says nothing.
    """
    apply_thesis_plot_style()
    regions = [r for r in REGION_ORDER if r in set(table["region"])]
    fig, ax = plt.subplots(figsize=(settings.schematic_width_in * 0.62,
                                    settings.panel_height_in + 0.3))
    for region in regions:
        group = table.loc[table["region"] == region].sort_values("separation")
        colour = REGION_COLORS.get(region, MUTED_INK)
        ax.plot(group["separation"], group["effect_size_rank_biserial"],
                marker="o", markersize=3.4, lw=1.0, color=colour, label=region_label(region))
        marked = group.loc[group["significant"]]
        if len(marked):
            ax.plot(marked["separation"], marked["effect_size_rank_biserial"],
                    marker="o", markersize=5.6, linestyle="none",
                    markerfacecolor="none", markeredgecolor=colour, markeredgewidth=1.0)
    ax.axhline(0.0, color=MUTED_INK, lw=0.9, linestyle="--")
    ax.set_xlabel("Electrode separation (channels)", fontsize=7)
    ax.set_ylabel("Interactive face − object\n(rank-biserial, trial-matched)", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6, loc="best")
    ax.set_title("Condition difference does not track the decay", fontsize=8, color=INK)
    nice_axis(ax)
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)
