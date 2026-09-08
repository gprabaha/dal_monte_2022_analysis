"""Figures for mean signal and per-trial spike correlation in simultaneous pairs.

The two analyses answer different questions on the same pairs, and the point of
putting them together is that neither is interpretable alone: per-trial spike
correlation without mean signal correlation cannot say whether co-firing
reflects shared tuning, and mean signal correlation without it cannot say
whether shared tuning is accompanied by trial-by-trial coupling.

The two names are used exactly as written throughout -- **mean signal
correlation** for the correlation of the two units' condition-averaged rate
timelines, **per-trial spike correlation** for the correlation of their spike
trains within single fixations.  Each name says which operation came first,
which is the only thing that separates them.

The second measure is deliberately **not** called noise correlation here.  The
classical noise correlation is a single number per pair -- the correlation of
their spike *counts* across trials, after each unit's condition mean is removed.
What is computed here is a full cross-correlogram of the two spike trains within
each fixation, averaged across fixations, which resolves the timing that a count
correlation integrates away and, being unnormalised, is not on the [-1, 1] scale
that name implies.  Calling it what it is avoids inviting a comparison of
magnitudes with a literature that measured something else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

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

REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")

#: Cross-region combinations reported.  Every well-populated one pairs BLA with
#: a frontal region; ACCg and OFC were never recorded together and dmPFC x OFC
#: comes from a handful of sessions, so neither is reportable.
CROSS_REGION_ORDER: tuple[str, ...] = ("accg-bla", "bla-dmpfc", "bla-ofc")

#: One colour per analysis, used consistently across every figure so a reader
#: can tell at a glance which of the two a panel belongs to.
SPIKE_COLOUR = "#c0392b"   # per-trial spike correlation
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


def _significance_bar(ax, x_left: float, x_right: float, y: float, text: str,
                      *, linewidth: float = 1.6, fontsize: float = 7.0) -> None:
    """A plain thick bar between two categories with stars above it.

    No end ticks.  The serifs on a conventional bracket are there to say which
    two categories are being compared, but with bars this narrow the span is
    already unambiguous and the ticks only add clutter -- and drawn downward they
    read as error bars belonging to the bars underneath.
    """
    ax.plot([x_left, x_right], [y, y], color=INK, linewidth=linewidth,
            solid_capstyle="butt", clip_on=False, zorder=6)
    span = float(ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(0.5 * (x_left + x_right), y + 0.012 * span, text, ha="center",
            va="bottom", fontsize=fontsize, color=INK, clip_on=False, zorder=6)


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

    A trial here is one fixation, and the diagram says "trial" throughout because
    that is what the measure is named after.  Reading outward from the middle: to
    the right, correlate within each trial and average the correlograms; to the
    left, average the trials into rate timelines and correlate those.  Averaging first is the only difference
    between the two measures, and it is what removes trial-by-trial covariation.
    Naming the right-hand path *per-trial spike correlation* rather than noise
    correlation keeps the diagram honest: what is drawn is a correlogram per
    fixation, not a correlation of per-fixation spike counts.

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
    ax.set_title("spike trains\nper trial", fontsize=6.8, color=INK, pad=9)

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
    ax.set_title("MEAN SIGNAL\ncorrelation", fontsize=7.4, color=SIGNAL_COLOUR, pad=9)

    # ---- right of centre: correlate first ----------------------------------
    ax = axes[3]
    for trial in range(3):
        trace = 0.42 * np.exp(-np.abs(lags) / 20.0) * rng.uniform(0.5, 1.6)
        ax.plot(lags, trace + rng.normal(0, 0.05, lags.size) + (2 - trial) * 0.62,
                color=MUTED_INK, lw=0.6)
    ax.text(0, -0.30, "⋮", ha="center", va="center", fontsize=9, color=MUTED_INK)
    ax.set_ylim(-0.62, 1.95)
    strip(ax)
    ax.set_title("one correlogram\nper trial", fontsize=6.8, color=SPIKE_COLOUR, pad=9)

    ax = axes[4]
    null_n = 0.052 + 0.004 * np.exp(-np.abs(lags) / 170)
    obs_n = null_n + 0.021 * np.exp(-np.abs(lags) / 9.0)
    ax.fill_between(lags, null_n, obs_n, color=SPIKE_COLOUR, alpha=0.22, lw=0)
    ax.plot(lags, obs_n, color=INK, lw=1.5)
    ax.plot(lags, null_n, color=MUTED_INK, lw=1.0, ls=(0, (3, 2)))
    ax.set_ylim(0.0495, 0.080)
    strip(ax)
    ax.set_title("PER-TRIAL SPIKE\ncorrelation", fontsize=7.4, color=SPIKE_COLOUR, pad=9)

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
    arrow(2, 3, "right", SPIKE_COLOUR, "correlate\nwithin trials")
    arrow(3, 4, "right", SPIKE_COLOUR, "then\naverage")
    return fig, save_thesis_figure(fig, settings, stem)


def plot_spike_correlation_above_null(
    traces: Mapping,
    settings: PairOverviewPlotSettings,
    *,
    scope: str = "within_region",
    conditions: Sequence[str] = CONDITION_ORDER,
    max_lag_ms: float = 150.0,
    regions: Optional[Sequence[str]] = None,
    stem: str = "fig03_spike_correlation_above_null",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Observed per-trial spike correlation against its null, per region and condition.

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
        "Per-trial spike correlation: observed against the circular-shift null",
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


_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _decade(values: Sequence[float]) -> int:
    """Exponent ``n`` that puts the largest plotted magnitude in [1, 10).

    Bar panels in this chapter cannot share a y-axis -- one is a Pearson
    coefficient and another counts spike pairs per fixation, and within one
    measure the cross-region values are a tenth of the within-region ones.  Left
    to matplotlib each panel then gets tick labels like ``0.0005`` beside another
    panel's ``0.08``, and the order-of-magnitude difference between them is
    carried by counting zeros.

    Normalising every panel to its own decade and naming that decade in the axis
    label puts the same three or four digits on every panel and moves the
    comparison to one symbol: a reader sees ×10⁻³ against ×10⁻⁴ rather than
    inferring it.  The panel is still scaled to its own data, which is the point
    -- the alternative, a shared axis, would flatten the small panels to a line.
    """
    magnitudes = np.abs(np.asarray(list(values), dtype=float))
    magnitudes = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
    if not magnitudes.size:
        return 0
    return int(np.floor(np.log10(magnitudes.max())))


def _decade_suffix(exponent: int) -> str:
    """``", ×10⁻³"`` for an exponent of -3, and nothing at all for 0."""
    if exponent == 0:
        return ""
    return f", ×10{str(exponent).translate(_SUPERSCRIPTS)}"


def _rescale_to_decade(rows: pd.DataFrame, columns: Sequence[str] = ("mean", "sem")) -> tuple[pd.DataFrame, int]:
    """Divide the plotted columns by their own decade, and report which."""
    rows = rows.copy()
    present = [c for c in columns if c in rows.columns]
    if not present:
        return rows, 0
    spread = rows[present[0]].astype(float).abs()
    if len(present) > 1:
        spread = spread + rows[present[1]].astype(float).abs()
    exponent = _decade(spread.to_numpy())
    for column in present:
        rows[column] = rows[column].astype(float) / (10.0 ** exponent)
    return rows, exponent


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
    tops: dict[str, float] = {}
    for condition in CONDITION_ORDER:
        rows = summary.loc[summary["condition"] == condition].set_index("region_pair")
        positions, values, errors = [], [], []
        for position, group in enumerate(groups):
            if group not in rows.index:
                continue
            positions.append(position + offsets[condition])
            values.append(float(rows.loc[group, value]))
            errors.append(float(rows.loc[group, error]) if error else 0.0)
            tops[group] = max(tops.get(group, -np.inf), values[-1] + errors[-1])
        if positions:
            # No bar outline and no error-bar caps.  An outline at this size
            # reads as a second, darker bar behind the first, and caps put two
            # horizontal marks per bar into a panel whose only other horizontal
            # marks are the significance bars.
            ax.bar(positions, values, width=width,
                   yerr=errors if error else None,
                   color=CONDITION_COLORS.get(condition, MUTED_INK),
                   edgecolor="none", linewidth=0.0,
                   error_kw={"elinewidth": 0.9, "capsize": 0.0, "ecolor": INK},
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
        # Each group's bars sit just above that group's own tallest bar, not at
        # a shared height near the top of the panel.  When one region is five
        # times another, a shared height leaves the short region's marks
        # floating in empty space with nothing to attach them to.
        local = contrasts.loc[contrasts["significant"].fillna(False)]
        low, high = ax.get_ylim()
        step = 0.075 * (high - low)
        headroom = high
        for position, group in enumerate(groups):
            block = local.loc[local["region_pair"] == group]
            block = block.loc[[
                index for index in block.index
                if block.loc[index, "condition_a"] in offsets
                and block.loc[index, "condition_b"] in offsets
            ]]
            if len(block):
                # Narrow spans underneath wide ones, so no bar crosses another
                # bar's stars.
                spans = (block["condition_b"].map(offsets)
                         - block["condition_a"].map(offsets)).abs()
                block = block.loc[spans.sort_values().index]
            base = tops.get(group, high)
            for level, row in enumerate(block.itertuples()):
                y = base + step * (0.55 + 1.25 * level)
                headroom = max(headroom, y + step * 0.95)
                _significance_bar(
                    ax, position + offsets[row.condition_a],
                    position + offsets[row.condition_b], y,
                    significance_stars(row.p_value_corrected),
                    fontsize=6.5,
                )
        ax.set_ylim(low, headroom)


def plot_analysed_pair_pies(
    summary: pd.DataFrame,
    settings: PairOverviewPlotSettings,
    *,
    scope: str = "within_region",
    stem: str = "fig02_analysed_pairs",
) -> tuple[plt.Figure, dict[str, Path]]:
    """One donut per group: pairs recorded, and the share the chapter analyses.

    A pair enters the analysis only when **both** its units are selective --
    significant for at least one fixation-type contrast after correction.
    Requiring that of two units at once is roughly the square of requiring it of
    one, so the analysed share is far smaller than the selective share of units,
    and the figure exists to make that cost explicit rather than leave it in a
    methods sentence.

    Drawn as donuts rather than filled pies so the recorded total can sit in the
    hole: the two numbers a reader needs are the denominator and the share of it
    that survives, and a filled pie can show only the second.  Both the count and
    the percentage are printed, so nothing has to be estimated from an angle --
    the usual and fair objection to a pie chart.

    One scope per call.  The within-region and cross-region donuts open their own
    sections of the chapter, and a reader arriving at the cross-region results
    should meet that section's denominators there rather than having to page back
    to a combined figure whose top half belongs to the previous section.
    """
    apply_thesis_plot_style()
    rows = summary.loc[summary["scope"].astype(str) == scope].copy()
    rows["region_pair"] = rows["region_pair"].astype(str)
    groups = _group_order(list(rows["region_pair"]), scope=scope)
    if not groups:
        fig, ax = plt.subplots(figsize=(3.2, 1.4))
        ax.text(0.5, 0.5, f"No {scope.replace('_', ' ')} groups", ha="center",
                va="center", fontsize=7, color=MUTED_INK)
        _bare(ax)
        return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")

    fig, axes = plt.subplots(
        1, len(groups), figsize=(1.62 * len(groups) + 0.4, 2.30), squeeze=False,
    )
    lookup = rows.set_index("region_pair")
    for ax, group in zip(axes[0], groups):
        record = lookup.loc[group]
        n_recorded = int(record["n_recorded"])
        n_analysed = int(record["n_analysed"])
        percent = 100.0 * n_analysed / n_recorded if n_recorded else 0.0

        ax.pie(
            [max(n_analysed, 0), max(n_recorded - n_analysed, 0)],
            colors=[INK, "#e3e3e3"], startangle=90, counterclock=False,
            wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 0.7},
        )
        ax.set_aspect("equal")
        ax.text(0, 0.12, f"{n_recorded:,}", ha="center", va="center",
                fontsize=8.5, color=MUTED_INK)
        ax.text(0, -0.14, "recorded", ha="center", va="center",
                fontsize=6.0, color=MUTED_INK)
        ax.text(0, 1.32, f"{n_analysed:,} analysed  ({percent:.0f}%)", ha="center",
                va="center", fontsize=6.8, color=INK)
        ax.set_title(region_label(group), fontsize=7.2, color=INK, pad=13)

    scope_label = "Within region" if scope == "within_region" else "Across regions"
    fig.suptitle(f"{scope_label}: pairs recorded, and the pairs analysed\n"
                 "(both units selective for at least one fixation-type contrast)",
                 fontsize=8.0, color=INK, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")


def plot_correlation_bars(
    signal_summary: pd.DataFrame,
    signal_contrasts: pd.DataFrame,
    spike_summary: pd.DataFrame,
    spike_contrasts: pd.DataFrame,
    settings: PairOverviewPlotSettings,
    *,
    signal_measure: str = "window_excess_pm250ms",
    scope: str = "within_region",
    stem: str = "fig05_correlation_bars",
) -> tuple[plt.Figure, dict[str, Path]]:
    """The two measures side by side, each reduced to one number per condition.

    Left: mean signal correlation, null-corrected, averaged over +-250 ms.
    Right: per-trial spike correlation over the same window, on the
    trial-count-matched recomputation.  A bar sits over every contrast that
    survives FDR and nowhere else.

    Putting them on adjacent axes rather than in separate figures is the point of
    the chapter: they are computed from the same spikes, on the same pairs, over
    the same lags, and differ only in whether the trials were averaged before or
    after correlating.  The axes cannot be shared -- one is a Pearson coefficient
    and the other counts spike pairs per fixation -- so only the *pattern across
    conditions* is comparable, which is what the figure is for.

    Each panel is scaled to its own decade and says which one in its axis label,
    so the magnitudes stay comparable *between* figures too: the cross-region
    version of this figure reads x10^-4 beside this one's x10^-3, and the
    ten-fold difference is a symbol rather than a count of leading zeros.
    """
    apply_thesis_plot_style()
    signal_rows = (
        signal_summary.loc[signal_summary["measure"] == signal_measure]
        if "measure" in signal_summary.columns else signal_summary
    ).copy()
    signal_local = signal_contrasts
    if signal_local is not None and "measure" in signal_local.columns:
        signal_local = signal_local.loc[signal_local["measure"] == signal_measure]

    spike_rows = spike_summary.copy()
    if "scope" in spike_rows.columns:
        spike_rows = spike_rows.loc[spike_rows["scope"].astype(str) == scope]
    signal_rows, signal_exponent = _rescale_to_decade(signal_rows)
    spike_rows, spike_exponent = _rescale_to_decade(spike_rows)
    spike_local = spike_contrasts
    if spike_local is not None and len(spike_local) and "scope" in spike_local.columns:
        spike_local = spike_local.loc[spike_local["scope"].astype(str) == scope]

    groups = _group_order(list(signal_rows["region_pair"].astype(str).unique()), scope=scope)
    if not groups:
        groups = _group_order(list(spike_rows["region_pair"].astype(str).unique()), scope=scope)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(settings.panel_width_in * 1.05 * max(len(groups), 1) + 1.6,
                 settings.panel_height_in + 1.0),
    )
    _bar_panel(axes[0], signal_rows, groups, contrasts=signal_local)
    _finish(axes[0],
            ylabel="Mean signal correlation\n"
                   f"(observed − null, ±250 ms{_decade_suffix(signal_exponent)})",
            title="Mean signal correlation", title_size=7.5)

    spike_groups = _group_order(list(spike_rows["region_pair"].astype(str).unique()), scope=scope)
    _bar_panel(axes[1], spike_rows, spike_groups, contrasts=spike_local)
    _finish(axes[1],
            ylabel="Per-trial spike correlation\n"
                   f"(observed − null, ±250 ms{_decade_suffix(spike_exponent)})",
            title="Per-trial spike correlation", title_size=7.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")
