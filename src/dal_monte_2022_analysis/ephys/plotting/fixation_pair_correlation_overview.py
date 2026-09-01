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
from matplotlib.patches import Rectangle

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
    """Both computations, step by step, on one page.

    The two rows share their first panel and diverge immediately after: the
    noise row correlates *within* each fixation and averages the results, the
    signal row averages *first* and correlates the averages.  Averaging first is
    what removes trial-by-trial covariation, which is why the two measure
    independent things.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        2, 4, figsize=(settings.schematic_width_in, settings.schematic_height_in)
    )
    rng = np.random.default_rng(7)
    time = np.linspace(-500, 500, 100)
    lags = np.linspace(-250, 250, 200)

    # ---------------- row 1: noise correlation --------------------------------
    ax = axes[0][0]
    for trial in range(5):
        for offset, colour in ((0.0, INK), (0.42, "#2c7fb8")):
            ax.vlines(rng.uniform(-500, 500, 11), trial + offset,
                      trial + offset + 0.3, color=colour, lw=0.6)
    ax.axvline(0, color=MUTED_INK, lw=0.6, linestyle=":")
    ax.set_ylabel("fixations", fontsize=6)
    ax.set_xlim(-560, 560); ax.set_ylim(-0.4, 5.3); _bare(ax)
    ax.set_title("1 ms spike trains,\n±500 ms per fixation", fontsize=7, color=INK)

    ax = axes[0][1]
    for trial in range(4):
        trace = 0.4 * np.exp(-np.abs(lags) / 30.0) * rng.uniform(0.5, 1.6) + rng.normal(0, 0.05, lags.size)
        ax.plot(lags, trace + trial * 0.6, color=INK, lw=0.6, alpha=0.7)
    ax.set_xlabel("Lag (ms)", fontsize=6)
    _bare(ax)
    ax.set_title("correlate WITHIN\neach fixation", fontsize=7, color="#c0392b")

    ax = axes[0][2]
    null = 0.30 + 0.015 * np.exp(-np.abs(lags) / 200)
    observed = null + 0.16 * np.exp(-np.abs(lags) / 8.0)
    ax.plot(lags, observed, color=INK, lw=1.2, label="observed")
    ax.plot(lags, null, color="#2c7fb8", lw=1.1, ls="--", label="circular-shift null")
    ax.fill_between(lags, null, observed, color="#d95f0e", alpha=0.3, lw=0)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    ax.set_xlabel("Lag (ms)", fontsize=6); ax.set_yticks([])
    _finish(ax, title="average, then subtract\nthe null")

    ax = axes[0][3]
    ax.text(0.5, 0.92, "NOISE correlation", ha="center", fontsize=8, color="#c0392b", va="top")
    ax.text(0.5, 0.72,
            "Do these two units fire together\n"
            "on the SAME fixation, more than\n"
            "chance?\n\n"
            "Null rotates one train within its\n"
            "own fixation, so each fixation keeps\n"
            "its spike count and slow envelope.",
            ha="center", va="top", fontsize=6, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _bare(ax)

    # ---------------- row 2: signal correlation -------------------------------
    ax = axes[1][0]
    for trial in range(5):
        for offset, colour in ((0.0, INK), (0.42, "#2c7fb8")):
            ax.vlines(rng.uniform(-500, 500, 11), trial + offset,
                      trial + offset + 0.3, color=colour, lw=0.6)
    ax.axvline(0, color=MUTED_INK, lw=0.6, linestyle=":")
    ax.set_ylabel("fixations", fontsize=6)
    ax.set_xlim(-560, 560); ax.set_ylim(-0.4, 5.3); _bare(ax)
    ax.set_title("the same trains", fontsize=7, color=INK)

    ax = axes[1][1]
    ax.plot(time, 1.0 + 0.85 * np.exp(-0.5 * ((time - 40) / 120.0) ** 2), color=INK, lw=1.5)
    ax.plot(time, 0.1 + 0.7 * np.exp(-0.5 * ((time - 5) / 140.0) ** 2), color="#2c7fb8", lw=1.5)
    ax.axvline(0, color=MUTED_INK, lw=0.6, linestyle=":")
    ax.set_xlabel("Time (ms)", fontsize=6); ax.set_yticks([])
    _finish(ax, title="average FIRST, into\nmean rate timelines", title_size=7)
    ax.title.set_color("#2c7fb8")

    ax = axes[1][2]
    sig_null = 0.002 + 0.001 * np.exp(-np.abs(lags) / 150)
    sig_obs = sig_null + 0.09 * np.exp(-0.5 * ((lags - 30) / 90.0) ** 2)
    ax.plot(lags, sig_obs, color=INK, lw=1.2, label="observed")
    ax.plot(lags, sig_null, color="#2c7fb8", lw=1.1, ls="--", label="cross-session null")
    ax.fill_between(lags, sig_null, sig_obs, color="#d95f0e", alpha=0.3, lw=0)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    ax.set_xlabel("Lag (ms)", fontsize=6); ax.set_yticks([])
    _finish(ax, title="correlate the averages,\nsubtract the null")

    ax = axes[1][3]
    ax.text(0.5, 0.92, "SIGNAL correlation", ha="center", fontsize=8, color="#2c7fb8", va="top")
    ax.text(0.5, 0.72,
            "Do their MEAN responses have\n"
            "the same shape, and at what lag?\n\n"
            "Null is a unit of the same region\n"
            "from a different session: real and\n"
            "fixation-locked, but sharing no\n"
            "session, array or behaviour.",
            ha="center", va="top", fontsize=6, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _bare(ax)

    fig.suptitle(
        "Averaging first is the whole difference: it removes trial-by-trial covariation",
        fontsize=8, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
            ("observed", INK, "-", "Observed"),
            ("null", "#2c7fb8", "--", "Circular-shift null"),
        ):
            mean = np.asarray(row[f"{channel}_mean"], dtype=float)[keep]
            sem = np.asarray(row[f"{channel}_sem"], dtype=float)[keep]
            ax.fill_between(lags[keep], mean - sem, mean + sem, color=colour, alpha=0.22, lw=0)
            ax.plot(lags[keep], mean, color=colour, lw=1.2, ls=dash, label=label)
        ax.axvline(0, color=MUTED_INK, lw=0.5, ls=":")
        _finish(ax, xlabel="Lag (ms)",
                title=f"{region_label(region)}  (n={int(row['n_pairs']):,})")
    axes[0][0].set_ylabel("Coincidences\nper fixation", fontsize=6.5)
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


def plot_lag_band_summary(
    summary: pd.DataFrame,
    settings: PairOverviewPlotSettings,
    *,
    measures: Sequence[tuple[str, str]] = (
        ("peak_excess", "Peak (max over ±100 ms)"),
        ("positive_lag_excess", "Mean, +20 to +200 ms"),
        ("negative_lag_excess", "Mean, −20 to −200 ms"),
    ),
    stem: str = "fig05_lag_band_summary",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Signal correlation summarised away from zero lag.

    Zero lag is one bin of fifty and a poor summary: two units whose responses
    have the same shape but different latencies correlate strongly at a non-zero
    lag and weakly at zero.  The peak asks how similar the shapes are at their
    best alignment; the two flanking bands ask whether that alignment is
    symmetric.

    Within a region the ordering of the two units in a pair is arbitrary, so the
    *sign* of a lead/lag difference carries no meaning — the two bands are
    expected to be similar, and it is a departure that would be notable.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1, len(measures),
        figsize=(settings.panel_width_in * 1.25 * len(measures), settings.panel_height_in + 0.6),
        squeeze=False,
    )
    regions = [r for r in REGION_ORDER if r in set(summary["region_pair"])]
    for ax, (key, label) in zip(axes[0], measures):
        width = 0.8 / max(len(CONDITION_ORDER), 1)
        for index, condition in enumerate(CONDITION_ORDER):
            rows = summary.loc[
                (summary["condition"] == condition) & (summary["measure"] == key)
            ].set_index("region_pair")
            positions, values, errors = [], [], []
            for position, region in enumerate(regions):
                if region not in rows.index:
                    continue
                positions.append(position + (index - 1) * width)
                values.append(float(rows.loc[region, "mean"]))
                errors.append(float(rows.loc[region, "sem"]))
            if positions:
                ax.bar(positions, values, width=width, yerr=errors,
                       color=CONDITION_COLORS.get(condition, MUTED_INK),
                       edgecolor=INK, linewidth=0.4, error_kw={"elinewidth": 0.7},
                       label=condition_label(condition) if ax is axes[0][0] else None)
        ax.axhline(0, color=MUTED_INK, lw=0.8)
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels([region_label(r) for r in regions], fontsize=6)
        _finish(ax, title=label)
    axes[0][0].set_ylabel("Signal correlation\n(observed − null)", fontsize=6.5)
    axes[0][0].legend(frameon=False, fontsize=5.5, loc="best")
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)
