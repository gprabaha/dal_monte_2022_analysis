"""Figures for neural pair spike coordination across fixation conditions.

Every panel here reads the aggregates written by
``build_fixation_pair_spike_coordination_summary.py``; nothing recomputes.  The
figures are ordered the way the argument has to be made:

1. the nulls behave (``plot_null_validation``),
2. coordination exists at all (``plot_group_z_traces``, ``plot_excess_vs_null``),
3. it differs by condition (``plot_condition_effects``, ``plot_condition_contrasts``),
4. it is not a zero-lag artifact (``plot_zero_lag_diagnostics``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_LABELS,
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

SCOPE_ORDER: tuple[str, ...] = ("within_region", "cross_region")
SCOPE_LABELS: dict[str, str] = {
    "within_region": "Within region",
    "cross_region": "Across regions",
}
NULL_LABELS: dict[str, str] = {
    "trial_shuffle": "vs trial-shuffle null",
    "circular_shift": "vs circular-shift null",
}
#: What each null licenses you to claim, kept next to the figure that shows it.
NULL_MEANINGS: dict[str, str] = {
    "trial_shuffle": "co-fluctuation across fixations",
    "circular_shift": "fine-timing alignment within a fixation",
}


def condition_label(condition: object, *, short: bool = False) -> str:
    key = str(condition)
    table = CONDITION_SHORT_LABELS if short else CONDITION_LABELS
    return table.get(key, key.replace("_", " "))


def scope_label(scope: object) -> str:
    return SCOPE_LABELS.get(str(scope), str(scope).replace("_", " "))


def region_pair_label(value: object) -> str:
    text = str(value)
    if "-" in text:
        return " x ".join(REGION_LABELS.get(part, part.upper()) for part in text.split("-"))
    return REGION_LABELS.get(text, text.upper())


@dataclass
class PairCoordinationPlotSettings(ThesisFigureSettings):
    """Figure output settings for the pair-coordination panels."""

    trace_figure_width_in: float = 7.2
    trace_figure_height_in: float = 2.6
    summary_figure_width_in: float = 7.2
    summary_figure_height_in: float = 2.8


def _finish(ax, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=8, color=INK)
    nice_axis(ax)


def plot_null_validation(
    identities: pd.DataFrame,
    sensitivity: pd.DataFrame,
    settings: PairCoordinationPlotSettings,
    *,
    stem: str = "fig00_null_validation",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Show that the fast null construction is exact and that each null is selective.

    The right panel is the one that matters for interpretation: an uncoupled
    pair sits at zero for both nulls, a pair sharing only a per-fixation gain
    moves the trial-shuffle null but not the circular-shift null, and a pair
    with genuine fine-timing structure moves both.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(settings.summary_figure_width_in, settings.summary_figure_height_in),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )

    ax = axes[0]
    errors = np.asarray(identities["max_abs_error"], dtype=float)
    floor = 1e-18
    positions = np.arange(len(errors))
    ax.barh(positions, np.maximum(errors, floor), color="#8fa8bf", edgecolor=INK, linewidth=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        ["mean identity", "trial-shuffle identity", "circular-shift identity"][: len(errors)],
        fontsize=6,
    )
    ax.set_xscale("log")
    ax.axvline(1e-9, color="#c0392b", linestyle="--", linewidth=0.8)
    ax.text(1e-9, len(errors) - 0.4, " tolerance", fontsize=6, color="#c0392b", va="top")
    _finish(ax, xlabel="Max absolute error vs brute force", title="Fast path is exact")

    ax = axes[1]
    order = ["independent", "shared_rate", "synchronous", "common_zero_lag"]
    labels = ["Independent", "Shared rate\n(per fixation)", "Synchronous\n(4 ms lag)", "Common\nzero lag"]
    frame = sensitivity.set_index("scenario")
    width = 0.38
    positions = np.arange(len(order))
    for offset, (null_name, colour) in enumerate(
        (("trial_shuffle", "#2c7fb8"), ("circular_shift", "#d95f0e"))
    ):
        values = [float(frame.loc[name, f"{null_name}_mean_z_pm10ms"]) for name in order]
        ax.bar(
            positions + (offset - 0.5) * width,
            values,
            width=width,
            color=colour,
            edgecolor=INK,
            linewidth=0.5,
            label=NULL_LABELS[null_name],
        )
    ax.axhline(0.0, color=MUTED_INK, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    _finish(ax, ylabel="Mean z, ±10 ms", title="Each null responds only to its own structure")

    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_group_z_traces(
    payload: Mapping,
    settings: PairCoordinationPlotSettings,
    *,
    null_name: str = "trial_shuffle",
    scopes: Sequence[str] = SCOPE_ORDER,
    conditions: Sequence[str] = CONDITION_ORDER,
    max_lag_ms: float = 100.0,
    stem: str = "fig01_group_z_traces",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Mean per-lag excess over the null, one panel per scope.

    A value of zero is the null's own expectation, so the distance of each curve
    from zero *is* the coordination.  Shaded bands are the standard error across
    pairs.
    """
    apply_thesis_plot_style()
    lags = np.asarray(payload["lags_ms"], dtype=float)
    traces = payload["traces"]
    keep = np.abs(lags) <= float(max_lag_ms)

    fig, axes = plt.subplots(
        1,
        len(scopes),
        figsize=(settings.trace_figure_width_in, settings.trace_figure_height_in),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, scope in zip(axes, scopes):
        for condition in conditions:
            row = traces.loc[
                (traces["scope"] == scope)
                & (traces["condition"] == condition)
                & (traces["null_name"] == null_name)
            ]
            if row.empty:
                continue
            mean = np.asarray(row["mean_z"].iloc[0], dtype=float)[keep]
            sem = np.asarray(row["sem_z"].iloc[0], dtype=float)[keep]
            colour = CONDITION_COLORS.get(condition, MUTED_INK)
            ax.fill_between(
                lags[keep], mean - sem, mean + sem, color=colour, alpha=0.22, linewidth=0
            )
            ax.plot(
                lags[keep],
                mean,
                color=colour,
                linewidth=1.2,
                label=condition_label(condition),
            )
        ax.axhline(0.0, color=MUTED_INK, linewidth=0.8, linestyle="--")
        ax.axvline(0.0, color=MUTED_INK, linewidth=0.6, linestyle=":")
        _finish(ax, xlabel="Lag (ms)", title=scope_label(scope))
    axes[0].set_ylabel(f"Mean z\n({NULL_LABELS[null_name]})")
    axes[-1].legend(frameon=False, fontsize=6, loc="upper right")
    fig.suptitle(
        f"Excess coordination: {NULL_MEANINGS[null_name]}",
        fontsize=8,
        color=INK,
    )
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, f"{stem}_{null_name}")


def plot_excess_vs_null(
    vs_null: pd.DataFrame,
    settings: PairCoordinationPlotSettings,
    *,
    stem: str = "fig02_excess_vs_null",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Per-group excess over null with its significance, answering "is anything there".

    The bar is the mean per-pair excess in single-fixation null units; the
    annotation is the Wilcoxon test of that population against zero.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1,
        len(SCOPE_ORDER),
        figsize=(settings.summary_figure_width_in, settings.summary_figure_height_in),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, scope in zip(axes, SCOPE_ORDER):
        subset = vs_null.loc[vs_null["scope"] == scope]
        positions = np.arange(len(CONDITION_ORDER))
        for position, condition in zip(positions, CONDITION_ORDER):
            row = subset.loc[subset["condition"] == condition]
            if row.empty:
                continue
            value = float(row["mean_excess"].iloc[0])
            ax.bar(
                position,
                value,
                width=0.62,
                color=CONDITION_COLORS.get(condition, MUTED_INK),
                edgecolor=INK,
                linewidth=0.5,
            )
            p_value = float(row["p_value"].iloc[0])
            marker = "n.s." if not np.isfinite(p_value) or p_value > 0.05 else (
                "***" if p_value < 1e-3 else "**" if p_value < 1e-2 else "*"
            )
            ax.text(
                position,
                value,
                marker,
                ha="center",
                va="bottom",
                fontsize=6,
                color=INK,
            )
        ax.axhline(0.0, color=MUTED_INK, linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [condition_label(c, short=True) for c in CONDITION_ORDER], fontsize=6
        )
        _finish(ax, title=scope_label(scope))
    axes[0].set_ylabel("Mean excess over null\n(single-fixation null SD)")
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_condition_effects(
    summary: pd.DataFrame,
    settings: PairCoordinationPlotSettings,
    *,
    group_column: str = "region_pair",
    stem: str = "fig03_condition_effects",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Condition effects broken out by region pair, with bootstrap intervals."""
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1,
        len(SCOPE_ORDER),
        figsize=(settings.summary_figure_width_in, settings.summary_figure_height_in + 0.4),
    )
    axes = np.atleast_1d(axes)
    for ax, scope in zip(axes, SCOPE_ORDER):
        subset = summary.loc[summary["scope"] == scope]
        groups = sorted(subset[group_column].astype(str).unique())
        width = 0.8 / max(len(CONDITION_ORDER), 1)
        for index, condition in enumerate(CONDITION_ORDER):
            rows = subset.loc[subset["condition"] == condition].set_index(group_column)
            values, lows, highs, positions = [], [], [], []
            for position, group in enumerate(groups):
                if str(group) not in rows.index:
                    continue
                row = rows.loc[str(group)]
                row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                values.append(float(row["mean"]))
                lows.append(float(row["mean"]) - float(row["ci_low"]))
                highs.append(float(row["ci_high"]) - float(row["mean"]))
                positions.append(position + (index - 1) * width)
            if not positions:
                continue
            ax.bar(
                positions,
                values,
                width=width,
                yerr=np.vstack([lows, highs]),
                error_kw={"elinewidth": 0.7, "capsize": 1.5},
                color=CONDITION_COLORS.get(condition, MUTED_INK),
                edgecolor=INK,
                linewidth=0.4,
                label=condition_label(condition) if ax is axes[0] else None,
            )
        ax.axhline(0.0, color=MUTED_INK, linewidth=0.8)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([region_pair_label(g) for g in groups], fontsize=6, rotation=30, ha="right")
        _finish(ax, title=scope_label(scope))
    axes[0].set_ylabel("Coordination effect\n(single-fixation null SD)")
    axes[0].legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_condition_contrasts(
    comparisons: pd.DataFrame,
    settings: PairCoordinationPlotSettings,
    *,
    label: str = "All pairs",
    stem: str = "fig04_condition_contrasts",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Within-pair condition contrasts: the same two neurons, different fixations.

    Every contrast is paired, so pair identity, firing rate and recording
    quality cannot explain a difference.  Points are the mean within-pair
    difference; filled markers survived FDR correction.
    """
    apply_thesis_plot_style()
    fig, ax = plt.subplots(
        figsize=(settings.summary_figure_width_in * 0.62, settings.summary_figure_height_in)
    )
    frame = comparisons.copy()
    frame["contrast"] = [
        f"{condition_label(a, short=True)} - {condition_label(b, short=True)}"
        for a, b in zip(frame["condition_a"], frame["condition_b"])
    ]
    contrasts = list(dict.fromkeys(frame["contrast"]))
    offsets = {"within_region": -0.16, "cross_region": 0.16}
    colours = {"within_region": "#2c7fb8", "cross_region": "#d95f0e"}
    for scope in SCOPE_ORDER:
        subset = frame.loc[frame["scope"] == scope]
        for _, row in subset.iterrows():
            position = contrasts.index(row["contrast"]) + offsets[scope]
            significant = bool(row.get("significant", False))
            ax.plot(
                [row["mean_difference"]],
                [position],
                marker="o",
                markersize=4.5,
                color=colours[scope],
                markerfacecolor=colours[scope] if significant else "white",
                markeredgewidth=0.9,
                linestyle="none",
            )
    ax.axvline(0.0, color=MUTED_INK, linewidth=0.9, linestyle="--")
    ax.set_yticks(np.arange(len(contrasts)))
    ax.set_yticklabels(contrasts, fontsize=6)
    ax.invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=colours[s], label=scope_label(s))
        for s in SCOPE_ORDER
    ]
    handles.append(
        plt.Line2D(
            [], [], marker="o", linestyle="none", color=INK,
            markerfacecolor="white", label="not significant (FDR)",
        )
    )
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="best")
    _finish(ax, xlabel="Within-pair difference in coordination effect", title=label)
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_zero_lag_diagnostics(
    diagnostics: pd.DataFrame,
    settings: PairCoordinationPlotSettings,
    *,
    stem: str = "fig05_zero_lag_diagnostics",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Per-date zero-lag prevalence, for spotting a contaminated recording day.

    Two randomly sampled neurons are essentially never monosynaptically
    connected, so a sharp zero-lag peak shared by most pairs on a day is common
    input or a recording artifact, not a pairwise interaction.  A day carrying
    it shows up here as an outlier rather than disappearing into the average.
    """
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        1,
        len(SCOPE_ORDER),
        figsize=(settings.summary_figure_width_in, settings.summary_figure_height_in),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, scope in zip(axes, SCOPE_ORDER):
        subset = diagnostics.loc[diagnostics["scope"] == scope].sort_values("date")
        if subset.empty:
            _finish(ax, title=scope_label(scope))
            continue
        values = subset["frac_pairs_zero_lag_above"].to_numpy(dtype=float)
        flagged = subset.get("suspected_zero_lag_artifact", pd.Series(False, index=subset.index))
        flagged = np.asarray(flagged, dtype=bool)
        positions = np.arange(len(subset))
        ax.bar(
            positions[~flagged],
            values[~flagged],
            color="#8fa8bf",
            edgecolor=INK,
            linewidth=0.4,
            label="typical day",
        )
        if flagged.any():
            ax.bar(
                positions[flagged],
                values[flagged],
                color="#c0392b",
                edgecolor=INK,
                linewidth=0.4,
                label="flagged as artifact",
            )
        median = float(np.nanmedian(values))
        ax.axhline(median, color=MUTED_INK, linestyle="--", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(subset["date"].astype(str), fontsize=4.5, rotation=90)
        _finish(ax, title=scope_label(scope))
        if flagged.any() and ax is axes[0]:
            ax.legend(frameon=False, fontsize=6, loc="upper left")
    axes[0].set_ylabel("Fraction of pairs with\nzero-lag z > 3")
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_selectivity_comparison(
    all_payload: Mapping,
    selective_payload: Mapping,
    settings: PairCoordinationPlotSettings,
    *,
    null_name: str = "trial_shuffle",
    max_lag_ms: float = 100.0,
    stem: str = "fig06_selective_pairs",
) -> tuple[plt.Figure, dict[str, Path]]:
    """All pairs against pairs where both units are FDR-selective.

    Selecting units by a condition contrast and then asking whether coordination
    differs by condition is not independent evidence -- it is a sensitivity
    check.  What it can legitimately show is whether any condition effect is
    carried by the selective subset or is present across the population.
    """
    apply_thesis_plot_style()
    lags = np.asarray(all_payload["lags_ms"], dtype=float)
    keep = np.abs(lags) <= float(max_lag_ms)

    fig, axes = plt.subplots(
        1,
        len(SCOPE_ORDER),
        figsize=(settings.trace_figure_width_in, settings.trace_figure_height_in),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, scope in zip(axes, SCOPE_ORDER):
        for payload, style, tag in (
            (all_payload, {"linestyle": "-", "alpha": 1.0}, "all pairs"),
            (selective_payload, {"linestyle": "--", "alpha": 0.95}, "both selective"),
        ):
            traces = payload["traces"]
            for condition in CONDITION_ORDER:
                row = traces.loc[
                    (traces["scope"] == scope)
                    & (traces["condition"] == condition)
                    & (traces["null_name"] == null_name)
                ]
                if row.empty:
                    continue
                mean = np.asarray(row["mean_z"].iloc[0], dtype=float)[keep]
                ax.plot(
                    lags[keep],
                    mean,
                    color=CONDITION_COLORS.get(condition, MUTED_INK),
                    linewidth=1.1,
                    label=f"{condition_label(condition, short=True)} ({tag})",
                    **style,
                )
        ax.axhline(0.0, color=MUTED_INK, linewidth=0.8, linestyle=":")
        _finish(ax, xlabel="Lag (ms)", title=scope_label(scope))
    axes[0].set_ylabel(f"Mean z\n({NULL_LABELS[null_name]})")
    axes[-1].legend(frameon=False, fontsize=5, loc="upper right", ncol=1)
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)
