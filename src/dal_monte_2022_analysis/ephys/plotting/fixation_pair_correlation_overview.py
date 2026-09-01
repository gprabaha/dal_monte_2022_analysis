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

#: Cross-region combinations reported.  Every well-populated one pairs BLA with
#: a frontal region; ACCg and OFC were never recorded together and dmPFC x OFC
#: comes from a handful of sessions, so neither is reportable.
CROSS_REGION_ORDER: tuple[str, ...] = ("accg-bla", "bla-dmpfc", "bla-ofc")

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


def _trim_spines(ax, *, bottom: bool = True) -> None:
    """End each spine at its outermost tick rather than at the axes corner.

    An untrimmed spine runs the full width of the axes and reads as a frame; a
    trimmed one reads as a scale, and the tick at each end tells you where the
    scale stops rather than leaving the reader to guess.
    """
    ticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    if len(ticks) >= 2:
        ax.spines["left"].set_bounds(min(ticks), max(ticks))
    if bottom:
        ticks = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        if len(ticks) >= 2:
            ax.spines["bottom"].set_bounds(min(ticks), max(ticks))


def _bare(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _group_order(values: Sequence[str], *, scope: str = "within_region") -> list[str]:
    """Groups in reporting order, dropping anything not reportable.

    Cross-region is restricted to the BLA-anchored combinations: the others are
    either impossible (ACCg and OFC were never recorded together) or come from
    too few sessions to interpret, and showing them beside the populated ones
    invites reading a hundred-pair estimate as though it were a result.
    """
    available = set(str(v) for v in values)
    order = CROSS_REGION_ORDER if scope == "cross_region" else REGION_ORDER
    return [group for group in order if group in available]


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
        figsize=(settings.schematic_width_in * 0.96, 2.35),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.05, 1.0, 1.0], "wspace": 0.42},
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
    ax.set_title("spike trains\nper fixation", fontsize=6.8, color=INK, pad=9)

    # ---- left of centre: average first -------------------------------------
    ax = axes[1]
    ax.plot(time, 0.95 + 0.80 * np.exp(-0.5 * ((time - 45) / 135.0) ** 2),
            color=unit_colours[0], lw=1.5)
    ax.plot(time, 0.02 + 0.66 * np.exp(-0.5 * ((time + 5) / 155.0) ** 2),
            color=unit_colours[1], lw=1.5)
    ax.axvline(0, color=MUTED_INK, lw=0.6, ls=(0, (1, 2)))
    ax.set_ylim(-0.10, 2.05)
    strip(ax)
    ax.set_title("mean rate\ntimelines", fontsize=6.8, color=SIGNAL_COLOUR, pad=9)

    ax = axes[0]
    null_s = np.full_like(lags, 0.006)
    obs_s = null_s + 0.085 * np.exp(-0.5 * ((lags - 35) / 88.0) ** 2)
    ax.fill_between(lags, null_s, obs_s, color=SIGNAL_COLOUR, alpha=0.22, lw=0)
    ax.plot(lags, obs_s, color=INK, lw=1.5)
    ax.plot(lags, null_s, color=MUTED_INK, lw=1.0, ls=(0, (3, 2)))
    ax.set_ylim(-0.004, 0.115)
    strip(ax)
    ax.set_title("SIGNAL\ncorrelation", fontsize=7.6, color=SIGNAL_COLOUR, pad=9)

    # ---- right of centre: correlate first ----------------------------------
    ax = axes[3]
    for trial in range(3):
        trace = 0.42 * np.exp(-np.abs(lags) / 20.0) * rng.uniform(0.5, 1.6)
        ax.plot(lags, trace + rng.normal(0, 0.05, lags.size) + (2 - trial) * 0.62,
                color=MUTED_INK, lw=0.6)
    ax.text(0, -0.30, "⋮", ha="center", va="center", fontsize=9, color=MUTED_INK)
    ax.set_ylim(-0.62, 1.95)
    strip(ax)
    ax.set_title("one correlogram\nper trial", fontsize=6.8, color=NOISE_COLOUR, pad=9)

    ax = axes[4]
    null_n = 0.052 + 0.004 * np.exp(-np.abs(lags) / 170)
    obs_n = null_n + 0.021 * np.exp(-np.abs(lags) / 9.0)
    ax.fill_between(lags, null_n, obs_n, color=NOISE_COLOUR, alpha=0.22, lw=0)
    ax.plot(lags, obs_n, color=INK, lw=1.5)
    ax.plot(lags, null_n, color=MUTED_INK, lw=1.0, ls=(0, (3, 2)))
    ax.set_ylim(0.0495, 0.080)
    strip(ax)
    ax.set_title("NOISE\ncorrelation", fontsize=7.6, color=NOISE_COLOUR, pad=9)

    # Set the margins directly.  These axes carry no tick labels and no y
    # labels, so tight_layout reserves a wide band on each flank for nothing.
    fig.subplots_adjust(left=0.015, right=0.985, top=0.775, bottom=0.145, wspace=0.42)
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
    scope: str = "within_region",
    conditions: Sequence[str] = CONDITION_ORDER,
    max_lag_ms: float = 150.0,
    regions: Optional[Sequence[str]] = None,
    stem: str = "fig03_noise_above_null",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Observed noise cross-correlation against its null, per region and condition.

    Rows are fixation conditions, columns are regions, and only the bottom row
    carries an x-axis: the lag axis is the same in every panel, so repeating it
    twelve times spends height on nothing.  Panels are deliberately short --
    what has to be legible is the gap between the two curves, not the shape of
    either one in isolation.
    """
    apply_thesis_plot_style()
    lags = np.asarray(traces["lags_ms"], dtype=float)
    frame = traces["traces"]
    keep = np.abs(lags) <= float(max_lag_ms)
    subset = frame.loc[frame["scope"].astype(str) == scope]
    if regions is None:
        regions = _group_order(list(subset["region_pair"].astype(str).unique()), scope=scope)
    present = [r for r in regions if (subset["region_pair"] == r).any()]
    if not present:
        fig, ax = plt.subplots(figsize=(3.2, 1.4))
        ax.text(0.5, 0.5, f"No {scope.replace('_', ' ')} groups", ha="center",
                va="center", fontsize=7, color=MUTED_INK)
        _bare(ax)
        return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")

    fig, axes = plt.subplots(
        len(conditions), len(present),
        figsize=(settings.panel_width_in * len(present), 0.62 * len(conditions) + 0.72),
        squeeze=False, sharex=True,
    )
    for row_index, condition in enumerate(conditions):
        colour = CONDITION_COLORS.get(condition, INK)
        for column, region in enumerate(present):
            ax = axes[row_index][column]
            row = subset.loc[
                (subset["region_pair"] == region) & (subset["condition"] == condition)
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
                ax.plot(lags[keep], mean, color=line_colour, lw=1.0, ls=dash, label=label)
            ax.axvline(0, color=MUTED_INK, lw=0.4, ls=":")
            ax.tick_params(labelsize=5.5, pad=1.5)
            ax.locator_params(axis="y", nbins=3)
            nice_axis(ax)
            # Detach the axes from each other and from the data, and give the
            # lag axis only to the bottom row: it is identical in every panel,
            # so drawing it twelve times spends ink and height on nothing.
            ax.spines["left"].set_position(("outward", 4))
            if row_index == len(conditions) - 1:
                ax.spines["bottom"].set_position(("outward", 4))
            else:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", length=0)
            _trim_spines(ax, bottom=row_index == len(conditions) - 1)
            if row_index == 0:
                ax.set_title(f"{region_label(region)}  (n={int(row['n_pairs']):,})",
                             fontsize=6.5, color=INK, pad=2.5)
            if row_index == len(conditions) - 1:
                ax.set_xlabel("Lag (ms)", fontsize=6.5)
        axes[row_index][0].set_ylabel(condition_label(condition), fontsize=6,
                                      color=CONDITION_COLORS.get(condition, INK))
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=5.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.015))
    fig.supylabel("Cross-correlation", fontsize=6.5, x=0.005)
    fig.suptitle(
        "Noise correlation: observed against the circular-shift null",
        fontsize=8, color=INK,
    )
    fig.tight_layout(rect=(0.012, 0.075, 1, 0.95))
    return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")


def plot_excess_by_condition(
    traces: Mapping,
    settings: PairOverviewPlotSettings,
    *,
    contrasts: Optional[pd.DataFrame] = None,
    scope: str = "within_region",
    max_lag_ms: float = 150.0,
    regions: Optional[Sequence[str]] = None,
    ylabel: str = "Cross-correlation − null",
    title: str = "",
    stem: str = "fig02_excess_by_condition",
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
    subset = frame.loc[frame["scope"].astype(str) == scope]
    if regions is None:
        regions = _group_order(list(subset["region_pair"].astype(str).unique()), scope=scope)
    present = [r for r in regions if (subset["region_pair"] == r).any()]
    if not present:
        fig, ax = plt.subplots(figsize=(3.2, 1.4))
        ax.text(0.5, 0.5, f"No {scope.replace('_', ' ')} groups", ha="center",
                va="center", fontsize=7, color=MUTED_INK)
        _bare(ax)
        return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")

    fig, axes = plt.subplots(
        1, len(present),
        figsize=(settings.panel_width_in * len(present), settings.panel_height_in + 0.35),
        squeeze=False,
    )
    for ax, region in zip(axes[0], present):
        n_pairs = 0
        for condition in CONDITION_ORDER:
            row = subset.loc[
                (subset["region_pair"] == region) & (subset["condition"] == condition)
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
    return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")


def _bar_panel(
    ax,
    summary: pd.DataFrame,
    groups: Sequence[str],
    *,
    value: str = "mean",
    error: Optional[str] = "sem",
    contrasts: Optional[pd.DataFrame] = None,
    star_column: Optional[str] = None,
) -> None:
    """One grouped bar panel: regions on x, one bar per fixation condition."""
    width = 0.8 / max(len(CONDITION_ORDER), 1)
    offsets = {c: (i - 1) * width for i, c in enumerate(CONDITION_ORDER)}
    for condition in CONDITION_ORDER:
        rows = summary.loc[summary["condition"] == condition].set_index("region_pair")
        positions, values, errors = [], [], []
        for position, group in enumerate(groups):
            if group not in rows.index:
                continue
            positions.append(position + offsets[condition])
            values.append(float(rows.loc[group, value]))
            errors.append(float(rows.loc[group, error]) if error else 0.0)
        if positions:
            ax.bar(positions, values, width=width,
                   yerr=errors if error else None,
                   color=CONDITION_COLORS.get(condition, MUTED_INK),
                   edgecolor=INK, linewidth=0.45,
                   error_kw={"elinewidth": 0.7, "capsize": 1.4},
                   label=condition_label(condition))
            if star_column is not None:
                for position, group in enumerate(groups):
                    if group not in rows.index:
                        continue
                    stars = significance_stars(float(rows.loc[group, star_column]))
                    if stars and stars != "n.s.":
                        height = float(rows.loc[group, value])
                        ax.text(position + offsets[condition],
                                height + (0.02 if height >= 0 else -0.05) * abs(ax.get_ylim()[1]),
                                stars, ha="center",
                                va="bottom" if height >= 0 else "top",
                                fontsize=6.5, color=INK)
    ax.axhline(0, color=MUTED_INK, lw=0.8)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([region_label(g) for g in groups], fontsize=6.5)

    if contrasts is not None and len(contrasts):
        # Only contrasts that survive FDR get a bracket.  Annotating the rest
        # fills the panel with marks that all say "no difference".
        local = contrasts.loc[contrasts["significant"].fillna(False)]
        low, high = ax.get_ylim()
        step = 0.10 * (high - low)
        headroom = high
        for position, group in enumerate(groups):
            block = local.loc[local["region_pair"] == group]
            for level, row in enumerate(block.itertuples()):
                if row.condition_a not in offsets or row.condition_b not in offsets:
                    continue
                y = high + step * (0.30 + level)
                headroom = max(headroom, y + step * 0.85)
                add_significance_bracket(
                    ax, position + offsets[row.condition_a],
                    position + offsets[row.condition_b], y,
                    significance_stars(row.p_value_corrected),
                    fontsize=6.5, color=INK, tick_frac=0.012,
                )
        ax.set_ylim(low, headroom)


def plot_summary_bars(
    signal_summary: pd.DataFrame,
    signal_contrasts: pd.DataFrame,
    correlations: pd.DataFrame,
    settings: PairOverviewPlotSettings,
    *,
    measure: str = "window_excess_pm250ms",
    scope: str = "within_region",
    stem: str = "fig04_summary_bars",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Signal correlation and the signal/noise relationship, side by side.

    Left: null-corrected signal correlation per region and fixation condition,
    with brackets on the contrasts that survive FDR.  Right: the Spearman
    correlation between each pair's signal and noise correlation, starred where
    it differs from zero.

    The legend sits below both panels rather than inside either: the brackets
    grow upward from the tallest bar, and any in-axes legend ends up underneath
    them in whichever region happens to have the largest effect.
    """
    apply_thesis_plot_style()
    rows = (
        signal_summary.loc[signal_summary["measure"] == measure]
        if "measure" in signal_summary.columns else signal_summary
    )
    groups = _group_order(list(rows["region_pair"].astype(str).unique()), scope=scope)
    contrasts = signal_contrasts
    if contrasts is not None and "measure" in contrasts.columns:
        contrasts = contrasts.loc[contrasts["measure"] == measure]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(settings.panel_width_in * 1.05 * max(len(groups), 1) + 1.6,
                 settings.panel_height_in + 1.0),
    )
    _bar_panel(axes[0], rows, groups, contrasts=contrasts)
    _finish(axes[0], ylabel="Signal correlation, mean ±250 ms\n(observed − null)",
            title="Signal correlation by fixation type", title_size=7.5)

    if len(correlations):
        spearman = correlations.copy()
        spearman = spearman.rename(columns={"spearman_rho": "mean"})
        groups_r = _group_order(list(spearman["region_pair"].astype(str).unique()), scope=scope)
        _bar_panel(axes[1], spearman, groups_r, value="mean", error=None,
                   star_column="p_value")
    # Descriptive, not a claim.  The relationship holds in some region and
    # condition combinations and not others, so a title asserting that it holds
    # would be contradicted by half its own panel.
    _finish(axes[1], ylabel="Spearman ρ across pairs",
            title="Signal against noise correlation", title_size=7.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")
