"""Combined figures for noise and signal correlation in simultaneous pairs.

The two analyses answer different questions on the same pairs, and the point of
putting them together is that neither is interpretable alone: noise correlation
without signal correlation cannot say whether co-firing reflects shared tuning,
and signal correlation without noise correlation cannot say whether shared
tuning is accompanied by trial-by-trial coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    REGION_LABELS,
    ThesisFigureSettings,
    add_significance_bracket,
    apply_thesis_plot_style,
    nice_axis,
    save_thesis_figure,
)

REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")

#: One colour per analysis, used consistently across every figure so a reader
#: can tell at a glance which of the two a panel belongs to.
NOISE_COLOUR = "#c0392b"
SIGNAL_COLOUR = "#2c7fb8"


@dataclass
class PairOverviewPlotSettings(ThesisFigureSettings):
    schematic_width_in: float = 7.2
    schematic_height_in: float = 4.4
    panel_width_in: float = 1.95
    panel_height_in: float = 1.75


def significance_stars(p_value: float) -> str:
    """Conventional star notation for a corrected p-value."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "n.s."


def condition_label(condition: object) -> str:
    return CONDITION_SHORT_LABELS.get(str(condition), str(condition).replace("_", " "))


def region_label(value: object) -> str:
    text = str(value)
    if "-" in text:
        return " × ".join(REGION_LABELS.get(p, p.upper()) for p in text.split("-"))
    return REGION_LABELS.get(text, text.upper())


def _finish(ax, *, xlabel="", ylabel="", title="", title_size=7) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=title_size, color=INK)
    nice_axis(ax)


def _bare(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_method_schematic(
    settings: PairOverviewPlotSettings,
    *,
    stem: str = "fig01_method_schematic",
) -> tuple[plt.Figure, dict[str, Path]]:
    """One set of trials in the centre, the two orders of operation either side.

    Reading outward from the middle: to the right, correlate within each trial
    and average the correlograms; to the left, average the trials into rate
    timelines and correlate those.  Averaging first is the only difference, and
    it is what removes trial-by-trial covariation.

    Deliberately unlabelled on the axes.  Nothing here is a measurement -- every
    trace is drawn from a formula -- so tick values would invite reading
    quantities off a diagram.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1, 5,
        figsize=(settings.schematic_width_in * 1.22, 1.95),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.05, 1.0, 1.0], "wspace": 0.62},
    )
    rng = np.random.default_rng(11)
    lags = np.linspace(-200, 200, 260)
    time = np.linspace(-500, 500, 140)
    unit_colours = ("#1a1a1a", "#5b8fb9")

    def strip(ax) -> None:
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["bottom"].set_color(MUTED_INK)

    # ---- centre: the trials both sides start from --------------------------
    ax = axes[2]
    for row, label in enumerate(("trial 1", "trial 2", "trial n")):
        y = 2 - row
        for offset, colour in zip((0.46, 0.08), unit_colours):
            ax.vlines(rng.uniform(-460, 460, 12), y + offset, y + offset + 0.30,
                      color=colour, lw=0.9)
        ax.text(-500, y + 0.40, label, fontsize=5.2, color=MUTED_INK,
                ha="right", va="center")
    ax.text(-500, 0.92, "⋮", ha="right", va="center", fontsize=9, color=MUTED_INK)
    ax.axvline(0, color=MUTED_INK, lw=0.6, ls=(0, (1, 2)))
    ax.set_xlim(-660, 520); ax.set_ylim(-0.30, 3.05)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("spike trains\nper fixation", fontsize=6.8, color=INK, pad=3)

    # ---- left of centre: average first -------------------------------------
    ax = axes[1]
    ax.plot(time, 0.95 + 0.80 * np.exp(-0.5 * ((time - 45) / 135.0) ** 2),
            color=unit_colours[0], lw=1.5)
    ax.plot(time, 0.02 + 0.66 * np.exp(-0.5 * ((time + 5) / 155.0) ** 2),
            color=unit_colours[1], lw=1.5)
    ax.axvline(0, color=MUTED_INK, lw=0.6, ls=(0, (1, 2)))
    ax.set_ylim(-0.10, 2.05)
    strip(ax)
    ax.set_title("mean rate\ntimelines", fontsize=6.8, color=SIGNAL_COLOUR, pad=3)

    ax = axes[0]
    null_s = np.full_like(lags, 0.006)
    obs_s = null_s + 0.085 * np.exp(-0.5 * ((lags - 35) / 88.0) ** 2)
    ax.fill_between(lags, null_s, obs_s, color=SIGNAL_COLOUR, alpha=0.22, lw=0)
    ax.plot(lags, obs_s, color=INK, lw=1.5)
    ax.plot(lags, null_s, color=MUTED_INK, lw=1.0, ls=(0, (3, 2)))
    ax.set_ylim(-0.004, 0.115)
    strip(ax)
    ax.set_title("SIGNAL\ncorrelation", fontsize=7.6, color=SIGNAL_COLOUR, pad=3)

    # ---- right of centre: correlate first ----------------------------------
    ax = axes[3]
    for trial in range(3):
        trace = 0.42 * np.exp(-np.abs(lags) / 20.0) * rng.uniform(0.5, 1.6)
        ax.plot(lags, trace + rng.normal(0, 0.05, lags.size) + (2 - trial) * 0.62,
                color=MUTED_INK, lw=0.6)
    ax.text(0, -0.30, "⋮", ha="center", va="center", fontsize=9, color=MUTED_INK)
    ax.set_ylim(-0.62, 1.95)
    strip(ax)
    ax.set_title("one correlogram\nper trial", fontsize=6.8, color=NOISE_COLOUR, pad=3)

    ax = axes[4]
    null_n = 0.052 + 0.004 * np.exp(-np.abs(lags) / 170)
    obs_n = null_n + 0.021 * np.exp(-np.abs(lags) / 9.0)
    ax.fill_between(lags, null_n, obs_n, color=NOISE_COLOUR, alpha=0.22, lw=0)
    ax.plot(lags, obs_n, color=INK, lw=1.5)
    ax.plot(lags, null_n, color=MUTED_INK, lw=1.0, ls=(0, (3, 2)))
    ax.set_ylim(0.0495, 0.080)
    strip(ax)
    ax.set_title("NOISE\ncorrelation", fontsize=7.6, color=NOISE_COLOUR, pad=3)

    fig.tight_layout(rect=(0, 0.13, 1, 0.87))
    boxes = [ax.get_position() for ax in axes]

    def arrow(left_index: int, right_index: int, pointing: str, colour, label):
        gap_left, gap_right = boxes[left_index].x1, boxes[right_index].x0
        pad = 0.16 * (gap_right - gap_left)
        y = 0.42
        start, end = (
            (gap_right - pad, gap_left + pad) if pointing == "left"
            else (gap_left + pad, gap_right - pad)
        )
        fig.add_artist(
            FancyArrowPatch((start, y), (end, y), transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=7, color=colour, lw=1.1)
        )
        fig.text(0.5 * (gap_left + gap_right), 0.035, label, ha="center",
                 va="bottom", fontsize=5.6, color=colour, linespacing=1.2)

    arrow(1, 2, "left", SIGNAL_COLOUR, "average\nacross trials")
    arrow(0, 1, "left", SIGNAL_COLOUR, "then\ncross-correlate")
    arrow(2, 3, "right", NOISE_COLOUR, "correlate\nwithin trials")
    arrow(3, 4, "right", NOISE_COLOUR, "then\naverage")
    return fig, save_thesis_figure(fig, settings, stem)


def plot_noise_above_null(
    traces: Mapping,
    settings: PairOverviewPlotSettings,
    *,
    conditions: Sequence[str] = CONDITION_ORDER,
    max_lag_ms: float = 150.0,
    regions: Sequence[str] = REGION_ORDER,
    stem: str = "fig04_noise_above_null",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Observed noise correlation against its null, every region and condition.

    Rows are fixation conditions, columns are regions.  The gap between the two
    curves in each panel is the coordination; the observed curve alone is not,
    since a coincidence count scales with the product of the two firing rates.
    """
    apply_thesis_plot_style()
    lags = np.asarray(traces["lags_ms"], dtype=float)
    frame = traces["traces"]
    keep = np.abs(lags) <= float(max_lag_ms)
    present = [
        r for r in regions
        if ((frame["region_pair"] == r) & (frame["scope"] == "within_region")).any()
    ]

    fig, axes = plt.subplots(
        len(conditions), len(present),
        figsize=(settings.panel_width_in * len(present),
                 settings.panel_height_in * 0.92 * len(conditions) + 0.35),
        squeeze=False,
    )
    for row_index, condition in enumerate(conditions):
        colour = CONDITION_COLORS.get(condition, INK)
        for column, region in enumerate(present):
            ax = axes[row_index][column]
            row = frame.loc[
                (frame["scope"] == "within_region")
                & (frame["region_pair"] == region)
                & (frame["condition"] == condition)
            ]
            if row.empty:
                ax.set_visible(False)
                continue
            row = row.iloc[0]
            for channel, line_colour, dash, label in (
                ("observed", colour, "-", "Observed"),
                ("null", MUTED_INK, "--", "Circular-shift null"),
            ):
                mean = np.asarray(row[f"{channel}_mean"], dtype=float)[keep]
                sem = np.asarray(row[f"{channel}_sem"], dtype=float)[keep]
                ax.fill_between(lags[keep], mean - sem, mean + sem,
                                color=line_colour, alpha=0.22, lw=0)
                ax.plot(lags[keep], mean, color=line_colour, lw=1.15, ls=dash, label=label)
            ax.axvline(0, color=MUTED_INK, lw=0.5, ls=":")
            _finish(
                ax,
                xlabel="Lag (ms)" if row_index == len(conditions) - 1 else "",
                title=(f"{region_label(region)}  (n={int(row['n_pairs']):,})"
                       if row_index == 0 else ""),
            )
        axes[row_index][0].set_ylabel(
            f"{condition_label(condition)}\ncoincidences per fixation", fontsize=6
        )
    axes[0][-1].legend(frameon=False, fontsize=5.5, loc="upper right")
    fig.suptitle(
        "Noise correlation sits above the null in every region and condition",
        fontsize=8, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, save_thesis_figure(fig, settings, stem)


def plot_excess_by_condition(
    traces: Mapping,
    settings: PairOverviewPlotSettings,
    *,
    contrasts: Optional[pd.DataFrame] = None,
    max_lag_ms: float = 150.0,
    regions: Sequence[str] = REGION_ORDER,
    ylabel: str = "Observed − null",
    title: str = "",
    stem: str = "fig03_excess_by_condition",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Null-corrected correlation for all three conditions, per region.

    When ``contrasts`` is supplied the largest between-condition effect size in
    each region is printed on the panel.  At these sample sizes almost any
    difference reaches significance, so the effect size is the number that
    decides whether the conditions differ in any way worth reporting.
    """
    apply_thesis_plot_style()
    lags = np.asarray(traces["lags_ms"], dtype=float)
    frame = traces["traces"]
    keep = np.abs(lags) <= float(max_lag_ms)
    present = [r for r in regions if ((frame["region_pair"] == r) & (frame["scope"] == "within_region")).any()]

    fig, axes = plt.subplots(
        1, len(present),
        figsize=(settings.panel_width_in * len(present), settings.panel_height_in + 0.35),
        squeeze=False,
    )
    for ax, region in zip(axes[0], present):
        n_pairs = 0
        for condition in CONDITION_ORDER:
            row = frame.loc[
                (frame["scope"] == "within_region")
                & (frame["region_pair"] == region)
                & (frame["condition"] == condition)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            mean = np.asarray(row["excess_mean"], dtype=float)[keep]
            sem = np.asarray(row["excess_sem"], dtype=float)[keep]
            colour = CONDITION_COLORS.get(condition, MUTED_INK)
            ax.fill_between(lags[keep], mean - sem, mean + sem, color=colour, alpha=0.2, lw=0)
            ax.plot(lags[keep], mean, color=colour, lw=1.1, label=condition_label(condition))
            n_pairs = max(n_pairs, int(row["n_pairs"]))
        ax.axhline(0, color=MUTED_INK, lw=0.8, ls="--")
        ax.axvline(0, color=MUTED_INK, lw=0.5, ls=":")
        if contrasts is not None and len(contrasts):
            local = contrasts.loc[contrasts["region_pair"] == region]
            if len(local):
                biggest = local["effect_size_rank_biserial"].abs().max()
                ax.text(0.96, 0.94, f"max |effect|\n{biggest:.3f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=5.5, color=MUTED_INK)
        _finish(ax, xlabel="Lag (ms)", title=f"{region_label(region)}  (n={n_pairs:,})")
    axes[0][0].set_ylabel(ylabel, fontsize=6.5)
    axes[0][-1].legend(frameon=False, fontsize=5.5, loc="upper right")
    if title:
        fig.suptitle(title, fontsize=8, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_peak_comparison(
    summary: pd.DataFrame,
    settings: PairOverviewPlotSettings,
    *,
    contrasts: Optional[pd.DataFrame] = None,
    measure: str = "window_excess_pm100ms",
    ylabel: str = "Signal correlation, mean ±100 ms\n(observed − null)",
    title: str = "",
    stem: str = "fig03_peak_comparison",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Peak null-corrected correlation per region and fixation condition.

    The peak is a maximum over many noisy lags, so its absolute level is
    inflated -- but identically for every condition, since each is a maximum
    over the same lags on the same pairs.  Comparisons between conditions
    therefore hold even though the level should not be quoted on its own.

    When ``contrasts`` is supplied, contrasts that survive FDR correction are
    marked, and the rank-biserial effect size is printed rather than a p-value:
    with thousands of pairs per region, significance is a statement about sample
    size and the effect size is the one about the neurons.
    """
    apply_thesis_plot_style()
    regions = [r for r in REGION_ORDER if r in set(summary["region_pair"])]
    rows = summary.loc[summary["measure"] == measure] if "measure" in summary.columns else summary

    fig, ax = plt.subplots(
        figsize=(settings.panel_width_in * 0.95 * max(len(regions), 1) + 0.8,
                 settings.panel_height_in + 0.9)
    )
    width = 0.8 / max(len(CONDITION_ORDER), 1)
    for index, condition in enumerate(CONDITION_ORDER):
        table = rows.loc[rows["condition"] == condition].set_index("region_pair")
        positions, values, errors = [], [], []
        for position, region in enumerate(regions):
            if region not in table.index:
                continue
            positions.append(position + (index - 1) * width)
            values.append(float(table.loc[region, "mean"]))
            errors.append(float(table.loc[region, "sem"]))
        if positions:
            ax.bar(positions, values, width=width, yerr=errors,
                   color=CONDITION_COLORS.get(condition, MUTED_INK),
                   edgecolor=INK, linewidth=0.5, error_kw={"elinewidth": 0.8, "capsize": 1.6},
                   label=condition_label(condition))
    ax.axhline(0, color=MUTED_INK, lw=0.8)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels([region_label(r) for r in regions], fontsize=7)
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    if contrasts is not None and len(contrasts):
        local = contrasts
        if "measure" in local.columns:
            local = local.loc[local["measure"] == measure]
        # Only contrasts that survive FDR get a bracket.  Annotating the rest
        # fills the panel with numbers that all say "no difference", and the
        # reader has to check each one to learn that.
        local = local.loc[local["significant"].fillna(False)]
        offsets = {c: (i - 1) * width for i, c in enumerate(CONDITION_ORDER)}
        low, high = ax.get_ylim()
        step = 0.085 * (high - low)
        headroom = high
        for position, region in enumerate(regions):
            block = local.loc[local["region_pair"] == region]
            for level, row in enumerate(block.itertuples()):
                if row.condition_a not in offsets or row.condition_b not in offsets:
                    continue
                y = high + step * (0.35 + level)
                headroom = max(headroom, y + step * 0.7)
                add_significance_bracket(
                    ax,
                    position + offsets[row.condition_a],
                    position + offsets[row.condition_b],
                    y,
                    significance_stars(row.p_value_corrected),
                    fontsize=7.0,
                    color=INK,
                    tick_frac=0.012,
                )
        ax.set_ylim(low, headroom)

    _finish(ax, ylabel=ylabel, title=title, title_size=8)
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_signal_noise_correlation_bars(
    correlations: pd.DataFrame,
    settings: PairOverviewPlotSettings,
    *,
    alpha: float = 0.05,
    stem: str = "fig06_signal_noise_correlation",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Spearman correlation between per-pair signal and noise correlation.

    One bar per region and fixation condition.  A positive value means pairs
    whose mean responses resemble each other also tend to co-fire trial to
    trial -- which is not guaranteed, since the two are computed by different
    operations and one can exist without the other.
    """
    apply_thesis_plot_style()
    regions = [r for r in REGION_ORDER if r in set(correlations["region_pair"])]
    fig, ax = plt.subplots(
        figsize=(settings.panel_width_in * 0.95 * max(len(regions), 1) + 0.8,
                 settings.panel_height_in + 0.5)
    )
    width = 0.8 / max(len(CONDITION_ORDER), 1)
    for index, condition in enumerate(CONDITION_ORDER):
        rows = correlations.loc[correlations["condition"] == condition].set_index("region_pair")
        for position, region in enumerate(regions):
            if region not in rows.index:
                continue
            row = rows.loc[region]
            value = float(row["spearman_rho"])
            x = position + (index - 1) * width
            ax.bar(x, value, width=width,
                   color=CONDITION_COLORS.get(condition, MUTED_INK),
                   edgecolor=INK, linewidth=0.45,
                   label=condition_label(condition) if position == 0 else None)
            stars = significance_stars(float(row["p_value"]))
            if stars and stars != "n.s.":
                ax.text(x, value + (0.012 if value >= 0 else -0.030), stars,
                        ha="center", va="bottom" if value >= 0 else "top",
                        fontsize=7, color=INK)
    ax.axhline(0, color=MUTED_INK, lw=0.8)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels([region_label(r) for r in regions], fontsize=7)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    _finish(ax, ylabel="Spearman ρ\n(signal vs noise correlation)",
            title="Pairs that share a response profile also co-fire", title_size=8)
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)
