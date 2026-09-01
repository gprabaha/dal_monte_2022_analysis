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
    it is what removes trial-by-trial covariation -- which is why the two sides
    measure independent things.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1, 5,
        figsize=(settings.schematic_width_in * 1.32, settings.schematic_height_in * 0.72),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.15, 1.0, 1.0], "wspace": 0.75},
    )
    rng = np.random.default_rng(11)
    lags = np.linspace(-200, 200, 240)
    time = np.linspace(-500, 500, 120)
    unit_colours = ("#111111", "#2c7fb8")
    interactive = CONDITION_COLORS["face_interactive"]

    # ---- centre: the trials both sides start from --------------------------
    ax = axes[2]
    for row, label in enumerate(["trial 1", "trial 2", "", "trial n"]):
        y = 3 - row
        if label == "":
            ax.text(0, y + 0.40, "⋮", ha="center", va="center", fontsize=13, color=INK)
            continue
        for offset, colour in zip((0.44, 0.06), unit_colours):
            ax.vlines(rng.uniform(-500, 500, 13), y + offset, y + offset + 0.30,
                      color=colour, lw=0.85)
        ax.text(-505, y + 0.72, label, fontsize=5.5, color=MUTED_INK, ha="left", va="center")
    ax.axvline(0, color=MUTED_INK, lw=0.7, ls=":")
    ax.text(0, 4.72, "fixation onset", ha="center", fontsize=5.5, color=MUTED_INK)
    ax.text(545, 3.59, "unit 1", fontsize=5.5, color=unit_colours[0], va="center")
    ax.text(545, 3.21, "unit 2", fontsize=5.5, color=unit_colours[1], va="center")
    ax.set_xlim(-540, 660); ax.set_ylim(-0.35, 5.05); _bare(ax)
    ax.set_title("1 ms spike trains\n±500 ms per fixation", fontsize=7.5, color=INK)

    # ---- left of centre: average first -------------------------------------
    ax = axes[1]
    ax.plot(time, 1.05 + 0.85 * np.exp(-0.5 * ((time - 45) / 130.0) ** 2),
            color=unit_colours[0], lw=1.6)
    ax.plot(time, 0.05 + 0.72 * np.exp(-0.5 * ((time + 5) / 150.0) ** 2),
            color=unit_colours[1], lw=1.6)
    ax.axvline(0, color=MUTED_INK, lw=0.6, ls=":")
    ax.set_xlabel("Time (ms)", fontsize=6)
    ax.set_yticks([])
    _finish(ax, title="mean rate timeline\nper unit", title_size=7)
    ax.title.set_color(SIGNAL_COLOUR)

    ax = axes[0]
    null_s = 0.002 + 0.001 * np.exp(-np.abs(lags) / 160)
    obs_s = null_s + 0.085 * np.exp(-0.5 * ((lags - 35) / 85.0) ** 2)
    ax.plot(lags, obs_s, color=interactive, lw=1.4)
    ax.plot(lags, null_s, color=MUTED_INK, lw=1.1, ls="--")
    ax.fill_between(lags, null_s, obs_s, color=SIGNAL_COLOUR, alpha=0.28, lw=0)
    ax.set_xlabel("Lag (ms)", fontsize=6); ax.set_yticks([])
    _finish(ax, title="SIGNAL\ncorrelation", title_size=8.5)
    ax.title.set_color(SIGNAL_COLOUR)

    # ---- right of centre: correlate first ----------------------------------
    ax = axes[3]
    for trial in range(3):
        trace = 0.5 * np.exp(-np.abs(lags) / 22.0) * rng.uniform(0.4, 1.7)
        trace = trace + rng.normal(0, 0.06, lags.size)
        ax.plot(lags, trace + (2 - trial) * 0.8, color=MUTED_INK, lw=0.7)
    ax.text(0, -0.42, "⋮", ha="center", va="center", fontsize=13, color=INK)
    ax.set_ylim(-0.75, 2.35)
    ax.set_xlabel("Lag (ms)", fontsize=6)
    _bare(ax)
    ax.set_title("one correlogram\nper trial", fontsize=7, color=NOISE_COLOUR)

    ax = axes[4]
    null_n = 0.055 + 0.004 * np.exp(-np.abs(lags) / 180)
    obs_n = null_n + 0.020 * np.exp(-np.abs(lags) / 9.0)
    ax.plot(lags, obs_n, color=interactive, lw=1.4)
    ax.plot(lags, null_n, color=MUTED_INK, lw=1.1, ls="--")
    ax.fill_between(lags, null_n, obs_n, color=NOISE_COLOUR, alpha=0.28, lw=0)
    ax.set_xlabel("Lag (ms)", fontsize=6); ax.set_yticks([])
    _finish(ax, title="NOISE\ncorrelation", title_size=8.5)
    ax.title.set_color(NOISE_COLOUR)

    # ---- arrows outward from the centre ------------------------------------
    # Placed from the axes' actual positions after layout, so they land in the
    # gaps between panels rather than on top of them.
    fig.tight_layout(rect=(0, 0.02, 1, 0.82))
    boxes = [ax.get_position() for ax in axes]

    def arrow(left_index: int, right_index: int, pointing: str, colour, label):
        gap_left = boxes[left_index].x1
        gap_right = boxes[right_index].x0
        pad = 0.12 * (gap_right - gap_left)
        y = 0.42
        start, end = (
            (gap_right - pad, gap_left + pad)
            if pointing == "left"
            else (gap_left + pad, gap_right - pad)
        )
        fig.add_artist(
            FancyArrowPatch((start, y), (end, y), transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=9, color=colour, lw=1.3)
        )
        fig.text(0.5 * (gap_left + gap_right), y + 0.045, label, ha="center",
                 va="bottom", fontsize=6.0, color=colour, linespacing=1.25)

    arrow(1, 2, "left", SIGNAL_COLOUR, "average\nacross trials")
    arrow(0, 1, "left", SIGNAL_COLOUR, "then\ncross-correlate")
    arrow(2, 3, "right", NOISE_COLOUR, "correlate\nwithin trials")
    arrow(3, 4, "right", NOISE_COLOUR, "then\naverage")
    # The rightmost panel's tick labels reach into its gap; nudge that one label.
    fig.texts[-1].set_position((fig.texts[-1].get_position()[0] - 0.008,
                                fig.texts[-1].get_position()[1]))

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig, save_thesis_figure(fig, settings, stem)


def plot_noise_above_null(
    traces: Mapping,
    settings: PairOverviewPlotSettings,
    *,
    condition: str = "face_interactive",
    max_lag_ms: float = 150.0,
    regions: Sequence[str] = REGION_ORDER,
    stem: str = "fig02_noise_above_null",
) -> tuple[plt.Figure, dict[str, Path]]:
    """One condition's observed noise correlation against its null, per region."""
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
        row = frame.loc[
            (frame["scope"] == "within_region")
            & (frame["region_pair"] == region)
            & (frame["condition"] == condition)
        ]
        if row.empty:
            _bare(ax); continue
        row = row.iloc[0]
        for channel, colour, dash, label in (
            ("observed", CONDITION_COLORS.get(condition, INK), "-", "Observed"),
            ("null", MUTED_INK, "--", "Circular-shift null"),
        ):
            mean = np.asarray(row[f"{channel}_mean"], dtype=float)[keep]
            sem = np.asarray(row[f"{channel}_sem"], dtype=float)[keep]
            ax.fill_between(lags[keep], mean - sem, mean + sem, color=colour, alpha=0.22, lw=0)
            ax.plot(lags[keep], mean, color=colour, lw=1.2, ls=dash, label=label)
        ax.axvline(0, color=MUTED_INK, lw=0.5, ls=":")
        _finish(ax, xlabel="Lag (ms)",
                title=f"{region_label(region)}  (n={int(row['n_pairs']):,})")
    # Coincidence counts, not a correlation coefficient: at each 1 ms lag this
    # is the number of spike pairs separated by that lag, per fixation.  Chance
    # is roughly rate_1 * rate_2 * bin width, so ~0.05 for two 7 Hz units, which
    # is why the values sit where they do and why they are not comparable in
    # magnitude with the signal correlation.
    axes[0][0].set_ylabel("Coincidences per fixation\n(1 ms lag bins)", fontsize=6.5)
    axes[0][-1].legend(frameon=False, fontsize=5.5, loc="upper right")
    fig.suptitle(
        f"Noise correlation sits above the null — {condition_label(condition)}, "
        "FDR-selective pairs",
        fontsize=8, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig, save_thesis_figure(fig, settings, f"{stem}_{condition}")


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
        # Brackets carry the effect size, not a p-value.  With hundreds to
        # thousands of pairs per region almost any difference clears an alpha,
        # so the star says "survived FDR" and the number says whether it matters.
        offsets = {c: (i - 1) * width for i, c in enumerate(CONDITION_ORDER)}
        low, high = ax.get_ylim()
        step = 0.085 * (high - low)
        headroom = high
        for position, region in enumerate(regions):
            block = local.loc[local["region_pair"] == region]
            level = 0
            for row in block.itertuples():
                if row.condition_a not in offsets or row.condition_b not in offsets:
                    continue
                if not np.isfinite(row.effect_size_rank_biserial):
                    continue
                y = high + step * (0.35 + level)
                headroom = max(headroom, y + step * 0.6)
                star = "*" if bool(row.significant) else ""
                add_significance_bracket(
                    ax,
                    position + offsets[row.condition_a],
                    position + offsets[row.condition_b],
                    y,
                    f"{row.effect_size_rank_biserial:+.2f}{star}",
                    fontsize=5.2,
                    color=MUTED_INK,
                    tick_frac=0.012,
                )
                level += 1
        ax.set_ylim(low, headroom)

    _finish(ax, ylabel=ylabel, title=title, title_size=8)
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_peak_signal_vs_noise(
    joined: pd.DataFrame,
    correlations: pd.DataFrame,
    settings: PairOverviewPlotSettings,
    *,
    stem: str = "fig05_peak_signal_vs_noise",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Peak signal against peak noise correlation, per region, per condition.

    One panel per region because the two quantities differ by an order of
    magnitude between regions and a pooled scatter would show mostly that.
    Points are coloured by fixation condition; the Spearman correlation per
    condition is printed on the panel, since the question is whether pairs that
    share a response profile also co-fire, not where the cloud sits.
    """
    apply_thesis_plot_style()
    regions = [r for r in REGION_ORDER if r in set(joined["region_pair"])]
    if not regions:
        fig, ax = plt.subplots(figsize=(3.2, 1.8))
        ax.text(0.5, 0.5, "No matched pairs", ha="center", va="center",
                fontsize=7, color=MUTED_INK)
        _bare(ax)
        return fig, save_thesis_figure(fig, settings, stem)

    fig, axes = plt.subplots(
        1, len(regions),
        figsize=(settings.panel_width_in * 1.15 * len(regions), settings.panel_height_in + 0.8),
        squeeze=False,
    )
    for ax, region in zip(axes[0], regions):
        block = joined.loc[joined["region_pair"] == region]
        lines = []
        for condition in CONDITION_ORDER:
            part = block.loc[block["condition"] == condition].dropna(subset=["signal", "noise"])
            if part.empty:
                continue
            colour = CONDITION_COLORS.get(condition, MUTED_INK)
            ax.scatter(part["signal"], part["noise"], s=3.0, alpha=0.35,
                       color=colour, linewidths=0, label=condition_label(condition))
            row = correlations.loc[
                (correlations["region_pair"] == region)
                & (correlations["condition"] == condition)
            ]
            if len(row):
                rho = float(row["spearman_rho"].iloc[0])
                star = "*" if float(row["p_value"].iloc[0]) < 0.05 else ""
                lines.append((colour, f"ρ={rho:+.2f}{star}"))
        for index, (colour, text) in enumerate(lines):
            ax.text(0.03, 0.97 - 0.10 * index, text, transform=ax.transAxes,
                    ha="left", va="top", fontsize=5.8, color=colour)
        ax.axhline(0, color=MUTED_INK, lw=0.6)
        ax.axvline(0, color=MUTED_INK, lw=0.6)
        # Robust limits.  A handful of pairs carry noise-correlation peaks an
        # order of magnitude above the rest, and letting them set the axis
        # flattens every other point onto the zero line.
        values = block["noise"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            low, high = np.percentile(values, [0.5, 99.0])
            pad = 0.08 * (high - low or 1.0)
            ax.set_ylim(low - pad, high + pad)
        _finish(ax, xlabel="Peak signal correlation",
                title=f"{region_label(region)}  (n={len(block) // 3:,})")
    axes[0][0].set_ylabel("Peak noise correlation\n(coincidences per fixation)", fontsize=6.5)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6, markerscale=3,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Do pairs that share a response profile also co-fire?", fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return fig, save_thesis_figure(fig, settings, stem)
