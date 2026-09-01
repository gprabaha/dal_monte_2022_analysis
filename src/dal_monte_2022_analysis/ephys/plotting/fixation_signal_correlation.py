"""Figures for signal correlation between condition-averaged rate timelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from dal_monte_2022_analysis.ephys.analysis.fixation_signal_correlation import (
    CONDITION_ORDER,
    REGION_ORDER,
)
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    REGION_LABELS,
    ThesisFigureSettings,
    apply_thesis_plot_style,
    nice_axis,
    save_thesis_figure,
)

SCOPE_LABELS = {"within_region": "Within region", "cross_region": "Across regions"}


@dataclass
class SignalCorrelationPlotSettings(ThesisFigureSettings):
    schematic_width_in: float = 7.2
    schematic_height_in: float = 2.6
    panel_width_in: float = 1.95
    panel_height_in: float = 1.75


def condition_label(condition: object) -> str:
    return CONDITION_SHORT_LABELS.get(str(condition), str(condition).replace("_", " "))


def region_label(value: object) -> str:
    text = str(value)
    if "-" in text:
        return " × ".join(REGION_LABELS.get(p, p.upper()) for p in text.split("-"))
    return REGION_LABELS.get(text, text.upper())


def _finish(ax, *, xlabel="", ylabel="", title="") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7, color=INK)
    nice_axis(ax)


def _bare(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_signal_vs_noise_schematic(
    settings: SignalCorrelationPlotSettings,
    *,
    stem: str = "fig01_signal_vs_noise_schematic",
) -> tuple[plt.Figure, dict[str, Path]]:
    """What this analysis measures, and how it differs from spike coordination.

    Left: noise correlation, computed on single-fixation spike trains -- do the
    two units fire together on the *same* fixation.  Right: signal correlation,
    computed on the condition average -- do their mean responses have the same
    shape.  Averaging removes trial-by-trial covariation entirely, so the two
    are independent quantities and a pair can have either without the other.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(settings.schematic_width_in, settings.schematic_height_in),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.85]},
    )
    rng = np.random.default_rng(5)
    time = np.linspace(-500, 500, 100)

    ax = axes[0]
    for trial in range(4):
        for offset, colour in ((0.0, INK), (0.5, "#2c7fb8")):
            times = rng.uniform(-500, 500, 14)
            ax.vlines(times, trial + offset, trial + offset + 0.36, color=colour, lw=0.6)
    ax.text(-560, 4.2, "unit 1", fontsize=6, color=INK, ha="right")
    ax.text(-560, 3.9, "unit 2", fontsize=6, color="#2c7fb8", ha="right")
    ax.set_ylabel("fixations", fontsize=6.5)
    ax.text(0, -0.7, "same fixation, same millisecond?", ha="center", fontsize=6, color=INK)
    ax.set_xlim(-620, 560); ax.set_ylim(-1.1, 4.6); _bare(ax)
    ax.set_title("Noise correlation\n(spike coordination)", fontsize=7.5, color=INK)

    ax = axes[1]
    bump = np.exp(-0.5 * ((time - 40) / 110.0) ** 2)
    ax.plot(time, 1.0 + 0.9 * bump, color=INK, lw=1.4)
    ax.plot(time, 0.15 + 0.75 * np.exp(-0.5 * ((time - 10) / 130.0) ** 2),
            color="#2c7fb8", lw=1.4)
    ax.axvline(0, color=MUTED_INK, lw=0.7, linestyle=":")
    ax.text(0, -0.28, "same response shape, at what lag?", ha="center", fontsize=6, color=INK)
    ax.set_xlabel("Time from fixation (ms)", fontsize=6.5)
    ax.set_ylabel("Mean firing rate", fontsize=6.5)
    ax.set_yticks([])
    ax.set_title("Signal correlation\n(this analysis)", fontsize=7.5, color=INK)
    nice_axis(ax)

    ax = axes[2]
    ax.text(0.5, 0.94, "The null has to be another unit", ha="center",
            fontsize=7.5, color=INK, va="top")
    ax.text(0.5, 0.74,
            "Every unit is fixation-locked, so any two\n"
            "mean timelines correlate before shared\n"
            "tuning is involved. A null that only\n"
            "scrambles time would confirm that.",
            ha="center", va="top", fontsize=6, color=MUTED_INK)
    ax.add_patch(Rectangle((0.08, 0.16), 0.84, 0.34, fill=False,
                           edgecolor=INK, linewidth=0.8))
    ax.text(0.5, 0.42,
            "Null: unit A against a unit of the\n"
            "same region from a different session.\n"
            "Fixation-locked and real, but shares\n"
            "no session, array or behaviour.",
            ha="center", va="top", fontsize=6, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _bare(ax)

    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_correlation_traces(
    traces: pd.DataFrame,
    lags_ms: np.ndarray,
    settings: SignalCorrelationPlotSettings,
    *,
    scope: str = "within_region",
    max_lag_ms: float = 250.0,
    stem: str = "fig02_correlation_traces",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Observed signal cross-correlation against the cross-session null."""
    apply_thesis_plot_style()
    keep = np.abs(lags_ms) <= float(max_lag_ms)
    subset = traces.loc[traces["scope"].astype(str) == scope]
    groups = [r for r in REGION_ORDER if r in set(subset["region_pair"])]
    groups += sorted(set(subset["region_pair"]) - set(groups))
    if not groups:
        fig, ax = plt.subplots(figsize=(3.4, 1.8))
        ax.text(0.5, 0.5, f"No {SCOPE_LABELS[scope].lower()} groups", ha="center",
                va="center", fontsize=7, color=MUTED_INK)
        _bare(ax)
        return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")

    fig, axes = plt.subplots(
        2, len(groups),
        figsize=(settings.panel_width_in * len(groups), settings.panel_height_in * 2),
        squeeze=False,
    )
    for column, group in enumerate(groups):
        for row_index, channel in enumerate(("observed", "excess")):
            ax = axes[row_index][column]
            for condition in CONDITION_ORDER:
                row = subset.loc[
                    (subset["region_pair"] == group) & (subset["condition"] == condition)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                mean = np.asarray(row[f"{channel}_mean"], dtype=float)[keep]
                sem = np.asarray(row[f"{channel}_sem"], dtype=float)[keep]
                colour = CONDITION_COLORS.get(condition, MUTED_INK)
                ax.fill_between(lags_ms[keep], mean - sem, mean + sem,
                                color=colour, alpha=0.2, linewidth=0)
                ax.plot(lags_ms[keep], mean, color=colour, lw=1.1,
                        label=condition_label(condition))
                if channel == "observed":
                    null = np.asarray(row["null_mean"], dtype=float)[keep]
                    ax.plot(lags_ms[keep], null, color=colour, lw=0.7,
                            linestyle="--", alpha=0.7)
            ax.axhline(0.0, color=MUTED_INK, lw=0.8, linestyle="--")
            ax.axvline(0.0, color=MUTED_INK, lw=0.5, linestyle=":")
            title = (
                f"{region_label(group)}  (n={int(row['n_pairs']):,})"
                if row_index == 0 else ""
            )
            _finish(ax, xlabel="Lag (ms)" if row_index == 1 else "", title=title)
    axes[0][0].set_ylabel("Signal correlation\n(dashed: null)", fontsize=6.5)
    axes[1][0].set_ylabel("Observed − null", fontsize=6.5)
    axes[0][-1].legend(frameon=False, fontsize=5.5, loc="upper right")
    fig.suptitle(f"{SCOPE_LABELS[scope]} — signal cross-correlation", fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, save_thesis_figure(fig, settings, f"{stem}_{scope}")


def plot_trial_count_confound(
    strata: pd.DataFrame,
    settings: SignalCorrelationPlotSettings,
    *,
    target: str = "face_interactive",
    reference: str = "object",
    stem: str = "fig03_trial_count_confound",
) -> tuple[plt.Figure, dict[str, Path]]:
    """The control that decides whether the condition effect is real.

    Shared tuning predicts a difference flat across strata of the trial-count
    ratio; an estimation-noise artifact predicts one that grows with the ratio
    and vanishes as it approaches one.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(settings.schematic_width_in * 0.82, settings.panel_height_in + 0.6)
    )

    ax = axes[0]
    x = np.arange(len(strata))
    for condition in CONDITION_ORDER:
        if condition not in strata.columns:
            continue
        ax.plot(x, strata[condition], marker="o", markersize=3.6, lw=1.1,
                color=CONDITION_COLORS.get(condition, MUTED_INK),
                label=condition_label(condition))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.1f}×" for r in strata["median_trial_ratio"]], fontsize=6)
    ax.legend(frameon=False, fontsize=5.5, loc="best")
    _finish(ax, xlabel="Trial-count ratio (interactive / object)",
            ylabel="Signal correlation\n(observed − null)",
            title="Each condition, by trial-count ratio")

    ax = axes[1]
    key = f"{target}_minus_{reference}"
    ax.plot(x, strata[key], marker="o", markersize=4.4, lw=1.3, color=INK)
    ax.axhline(0.0, color="#c0392b", lw=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.1f}×" for r in strata["median_trial_ratio"]], fontsize=6)
    _finish(ax, xlabel="Trial-count ratio (interactive / object)",
            ylabel="Interactive face − object",
            title="The difference tracks the ratio")
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_condition_summary(
    summary: pd.DataFrame,
    settings: SignalCorrelationPlotSettings,
    *,
    stem: str = "fig04_condition_summary",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Zero-lag signal correlation per region and condition."""
    apply_thesis_plot_style()
    scopes = [s for s in ("within_region", "cross_region") if (summary["scope"] == s).any()]
    fig, axes = plt.subplots(
        1, len(scopes),
        figsize=(settings.schematic_width_in * 0.85, settings.panel_height_in + 0.5),
        squeeze=False,
    )
    for ax, scope in zip(axes[0], scopes):
        subset = summary.loc[summary["scope"] == scope]
        groups = [r for r in REGION_ORDER if r in set(subset["region_pair"])]
        groups += sorted(set(subset["region_pair"]) - set(groups))
        width = 0.8 / max(len(CONDITION_ORDER), 1)
        for index, condition in enumerate(CONDITION_ORDER):
            rows = subset.loc[subset["condition"] == condition].set_index("region_pair")
            positions, values, errors = [], [], []
            for position, group in enumerate(groups):
                if group not in rows.index:
                    continue
                positions.append(position + (index - 1) * width)
                values.append(float(rows.loc[group, "mean"]))
                errors.append(float(rows.loc[group, "sem"]))
            if positions:
                ax.bar(positions, values, width=width, yerr=errors,
                       color=CONDITION_COLORS.get(condition, MUTED_INK),
                       edgecolor=INK, linewidth=0.4,
                       error_kw={"elinewidth": 0.7},
                       label=condition_label(condition) if ax is axes[0][0] else None)
        ax.axhline(0.0, color=MUTED_INK, lw=0.8)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([region_label(g) for g in groups], fontsize=6,
                           rotation=25, ha="right")
        _finish(ax, title=SCOPE_LABELS[scope])
    axes[0][0].set_ylabel("Zero-lag signal correlation\n(observed − null)", fontsize=6.5)
    axes[0][0].legend(frameon=False, fontsize=5.5, loc="best")
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_signal_vs_noise(
    joined: pd.DataFrame,
    correlations: pd.DataFrame,
    settings: SignalCorrelationPlotSettings,
    *,
    stem: str = "fig05_signal_vs_noise",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Do pairs with similar mean responses also covary trial to trial?"""
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(settings.schematic_width_in * 0.8, settings.panel_height_in + 0.6)
    )

    ax = axes[0]
    subset = joined.loc[joined["scope"] == "within_region"].dropna(subset=["signal", "noise"])
    if len(subset):
        sample = subset.sample(min(len(subset), 6000), random_state=0)
        for condition in CONDITION_ORDER:
            part = sample.loc[sample["condition"] == condition]
            ax.scatter(part["signal"], part["noise"], s=1.6, alpha=0.25,
                       color=CONDITION_COLORS.get(condition, MUTED_INK),
                       label=condition_label(condition), linewidths=0)
    ax.axhline(0.0, color=MUTED_INK, lw=0.6); ax.axvline(0.0, color=MUTED_INK, lw=0.6)
    ax.legend(frameon=False, fontsize=5.5, markerscale=4, loc="best")
    _finish(ax, xlabel="Signal correlation (zero lag)",
            ylabel="Noise correlation\n(sharp peak)", title="Within region")

    ax = axes[1]
    if len(correlations):
        groups = list(dict.fromkeys(correlations["region_pair"]))
        width = 0.8 / max(len(CONDITION_ORDER), 1)
        for index, condition in enumerate(CONDITION_ORDER):
            rows = correlations.loc[correlations["condition"] == condition].set_index("region_pair")
            positions, values = [], []
            for position, group in enumerate(groups):
                if group not in rows.index:
                    continue
                positions.append(position + (index - 1) * width)
                values.append(float(rows.loc[group, "spearman_rho"]))
            if positions:
                ax.bar(positions, values, width=width,
                       color=CONDITION_COLORS.get(condition, MUTED_INK),
                       edgecolor=INK, linewidth=0.4)
        ax.axhline(0.0, color=MUTED_INK, lw=0.8)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([region_label(g) for g in groups], fontsize=6,
                           rotation=25, ha="right")
    _finish(ax, ylabel="Spearman ρ\n(signal vs noise)", title="Per region and condition")
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)
