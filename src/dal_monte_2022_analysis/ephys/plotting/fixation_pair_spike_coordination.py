"""Figures for neural pair spike coordination across fixation conditions.

Every panel reads the aggregates written by ``run_summary_build``; nothing
recomputes.  Two figure shapes carry the analysis:

``plot_observed_and_null_grid``
    Regions down, fixation conditions across.  Each panel shows the observed
    cross-correlation and the cross-trial shuffle null on the same axes, so the
    excess is visible rather than inferred.

``plot_null_corrected_grid``
    One panel per region, the three conditions overlaid, showing observed minus
    null.  This is the comparison figure.

Nothing here averages across regions.  With this dataset's recordings a pooled
curve would be a composition of very unequal region contributions rather than a
summary of them, so regions stay separate and are compared by eye and by the
statistics beside them.
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

#: Lag half-width the figures show.  The correlation is computed over
#: -999..+999 ms, but the overlap taper makes the far lags progressively noisier
#: -- at 900 ms only 100 of 1000 bins contribute -- and every peak in this data
#: has decayed to its baseline well before 250 ms.
DEFAULT_PLOT_MAX_LAG_MS = 250.0

SCOPE_LABELS: dict[str, str] = {
    "within_region": "Within region",
    "cross_region": "Across regions",
}

#: Everything is in coincidences per fixation.  No normalisation is applied:
#: the trial-shuffle null already carries both units' firing rates and exact
#: spike counts, so ``observed - null`` is rate-controlled by construction.
MEASURE_LABEL = "Coincidences\nper fixation"
EXCESS_LABEL = "Cross-correlation − null\n(coincidences per fixation)"


@dataclass
class PairCoordinationPlotSettings(ThesisFigureSettings):
    """Figure output settings for the pair-coordination panels."""

    panel_width_in: float = 1.85
    panel_height_in: float = 1.55
    contrast_figure_width_in: float = 5.0
    contrast_figure_height_in: float = 2.8


def condition_label(condition: object, *, short: bool = False) -> str:
    key = str(condition)
    table = CONDITION_SHORT_LABELS if short else CONDITION_LABELS
    return table.get(key, key.replace("_", " "))


def scope_label(scope: object) -> str:
    return SCOPE_LABELS.get(str(scope), str(scope).replace("_", " "))


def region_pair_label(value: object) -> str:
    text = str(value)
    if "-" in text:
        return " × ".join(REGION_LABELS.get(part, part.upper()) for part in text.split("-"))
    return REGION_LABELS.get(text, text.upper())


def _row(traces: pd.DataFrame, **filters) -> Optional[pd.Series]:
    mask = np.ones(len(traces), dtype=bool)
    for column, value in filters.items():
        if column not in traces.columns:
            return None
        mask &= traces[column].astype(str).to_numpy() == str(value)
    subset = traces.loc[mask]
    return None if subset.empty else subset.iloc[0]


def region_groups(
    traces: pd.DataFrame,
    scope: str,
    *,
    min_pairs: int = 0,
    order: Sequence[str] = ("bla", "accg", "dmpfc", "ofc"),
) -> list[str]:
    """Region or region-pair groups present in one scope, in reporting order."""
    subset = traces.loc[traces["scope"].astype(str) == str(scope)]
    if subset.empty:
        return []
    groups = set(subset["region_pair"].astype(str))
    if min_pairs > 0:
        keep = set()
        for group in groups:
            counts = [
                int(row["n_pairs"])
                for condition in CONDITION_ORDER
                if (row := _row(traces, scope=scope, region_pair=group, condition=condition))
                is not None
            ]
            if counts and min(counts) >= int(min_pairs):
                keep.add(group)
        groups = keep
    ranked = [region for region in order if region in groups]
    return ranked + sorted(groups - set(ranked))


def _finish(ax, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7, color=INK)
    nice_axis(ax)


def _empty_figure(
    settings: PairCoordinationPlotSettings, note: str, stem: str
) -> tuple[plt.Figure, dict[str, Path]]:
    fig, ax = plt.subplots(figsize=(3.6, 1.8))
    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=7, color=MUTED_INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)


def plot_observed_and_null_grid(
    payload: Mapping,
    settings: PairCoordinationPlotSettings,
    *,
    scope: str = "within_region",
    max_lag_ms: Optional[float] = DEFAULT_PLOT_MAX_LAG_MS,
    min_pairs: int = 0,
    label: str = "",
    stem: str = "observed_and_null",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Observed correlation and the trial-shuffle null, regions × conditions.

    One row per region (or region pair), one column per fixation condition.
    Reading across a row compares conditions within a region; reading down a
    column compares regions within a condition.  The gap between the two curves
    is the coordination; the curves themselves are not, because a correlation
    scales with the product of the two firing rates.
    """
    apply_thesis_plot_style()
    lags = np.asarray(payload["lags_ms"], dtype=float)
    traces = payload["traces"]
    # ``None`` shows every lag that was saved.  The default shows +/-250 ms,
    # which contains the whole peak and its return to baseline in every region.
    keep = (
        np.ones(lags.shape, dtype=bool)
        if max_lag_ms is None
        else np.abs(lags) <= float(max_lag_ms)
    )
    groups = region_groups(traces, scope, min_pairs=min_pairs)
    if not groups:
        return _empty_figure(
            settings,
            f"No {scope_label(scope).lower()} group reaches\n{min_pairs:,} pairs per condition",
            f"{stem}_{scope}",
        )

    n_rows, n_cols = len(groups), len(CONDITION_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(settings.panel_width_in * n_cols, settings.panel_height_in * n_rows),
        sharex=True,
        squeeze=False,
    )
    for row_index, group in enumerate(groups):
        # Every panel autoscales independently.  Sharing a scale even within a
        # row clipped the taller peaks and squashed the shorter ones, and the
        # comparison that matters here is between the two curves inside a panel,
        # not between panels -- that comparison is what the null-corrected
        # figure is for.
        row_axes = axes[row_index]
        for column_index, condition in enumerate(CONDITION_ORDER):
            ax = row_axes[column_index]
            row = _row(traces, scope=scope, region_pair=group, condition=condition)
            if row is None:
                ax.set_visible(False)
                continue
            for channel, colour, dash, name in (
                ("observed", INK, "-", "Observed"),
                ("null", "#2c7fb8", "--", "Circular-shift null"),
            ):
                mean = np.asarray(row[f"{channel}_mean"], dtype=float)[keep]
                sem = np.asarray(row[f"{channel}_sem"], dtype=float)[keep]
                ax.fill_between(
                    lags[keep], mean - sem, mean + sem, color=colour, alpha=0.20, linewidth=0
                )
                ax.plot(lags[keep], mean, color=colour, linewidth=1.05, linestyle=dash, label=name)
            ax.axvline(0.0, color=MUTED_INK, linewidth=0.5, linestyle=":")
            title = (
                f"{condition_label(condition, short=True)}  (n={int(row['n_pairs']):,})"
                if row_index == 0
                else f"n={int(row['n_pairs']):,}"
            )
            _finish(
                ax,
                xlabel="Lag (ms)" if row_index == n_rows - 1 else "",
                title=title,
            )
        row_axes[0].set_ylabel(f"{region_pair_label(group)}\n{MEASURE_LABEL}", fontsize=6)
    axes[0][-1].legend(frameon=False, fontsize=5.5, loc="upper right")
    heading = f"{scope_label(scope)} — observed vs circular-shift null"
    if label:
        heading = f"{heading} — {label}"
    # Reserve the strip the title needs before tight_layout packs the axes, or
    # the top row's panel titles collide with it.
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.suptitle(heading, fontsize=8, color=INK, y=0.99)
    suffix = scope + (f"_{label.replace(' ', '_').lower()}" if label else "")
    return fig, save_thesis_figure(fig, settings, f"{stem}_{suffix}")


def plot_null_corrected_grid(
    payload: Mapping,
    settings: PairCoordinationPlotSettings,
    *,
    scope: str = "within_region",
    max_lag_ms: Optional[float] = DEFAULT_PLOT_MAX_LAG_MS,
    min_pairs: int = 0,
    label: str = "",
    stem: str = "null_corrected",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Observed minus null, one panel per region, conditions overlaid.

    This is the comparison figure: the null has been subtracted, so zero is "no
    coordination beyond what each unit's own fixation-locked rate profile
    predicts", and the three conditions can be read against each other directly.
    """
    apply_thesis_plot_style()
    lags = np.asarray(payload["lags_ms"], dtype=float)
    traces = payload["traces"]
    # ``None`` shows every lag that was saved.  The default shows +/-250 ms,
    # which contains the whole peak and its return to baseline in every region.
    keep = (
        np.ones(lags.shape, dtype=bool)
        if max_lag_ms is None
        else np.abs(lags) <= float(max_lag_ms)
    )
    groups = region_groups(traces, scope, min_pairs=min_pairs)
    if not groups:
        return _empty_figure(
            settings,
            f"No {scope_label(scope).lower()} group reaches\n{min_pairs:,} pairs per condition",
            f"{stem}_{scope}",
        )

    n_cols = min(len(groups), 4)
    n_rows = int(np.ceil(len(groups) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(settings.panel_width_in * 1.25 * n_cols, settings.panel_height_in * 1.2 * n_rows),
        squeeze=False,
    )
    flat = axes.reshape(-1)
    for ax, group in zip(flat, groups):
        n_pairs = 0
        for condition in CONDITION_ORDER:
            row = _row(traces, scope=scope, region_pair=group, condition=condition)
            if row is None:
                continue
            mean = np.asarray(row["excess_mean"], dtype=float)[keep]
            sem = np.asarray(row["excess_sem"], dtype=float)[keep]
            colour = CONDITION_COLORS.get(condition, MUTED_INK)
            ax.fill_between(
                lags[keep], mean - sem, mean + sem, color=colour, alpha=0.20, linewidth=0
            )
            ax.plot(
                lags[keep], mean, color=colour, linewidth=1.1,
                label=condition_label(condition, short=True),
            )
            n_pairs = max(n_pairs, int(row["n_pairs"]))
        ax.axhline(0.0, color=MUTED_INK, linewidth=0.8, linestyle="--")
        ax.axvline(0.0, color=MUTED_INK, linewidth=0.5, linestyle=":")
        _finish(ax, xlabel="Lag (ms)", title=f"{region_pair_label(group)}  (n={n_pairs:,})")
    for ax in flat[len(groups):]:
        ax.set_visible(False)
    axes[0][0].set_ylabel(EXCESS_LABEL, fontsize=6)
    flat[len(groups) - 1].legend(frameon=False, fontsize=5.5, loc="upper right")
    heading = f"{scope_label(scope)} — null-corrected cross-correlation"
    if label:
        heading = f"{heading} — {label}"
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle(heading, fontsize=8, color=INK, y=0.99)
    suffix = scope + (f"_{label.replace(' ', '_').lower()}" if label else "")
    return fig, save_thesis_figure(fig, settings, f"{stem}_{suffix}")


def plot_condition_contrasts(
    comparisons: pd.DataFrame,
    settings: PairCoordinationPlotSettings,
    *,
    scope: str = "within_region",
    label: str = "",
    stem: str = "condition_contrasts",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Within-pair condition contrasts, one row per region and contrast.

    Every contrast is paired: the same two neurons, the same electrodes, the
    same session, differing only in which fixations were used.  Filled markers
    survived FDR correction.
    """
    apply_thesis_plot_style()
    frame = comparisons.loc[comparisons["scope"].astype(str) == str(scope)].copy()
    if frame.empty:
        return _empty_figure(settings, f"No {scope_label(scope).lower()} contrasts", f"{stem}_{scope}")

    frame["contrast"] = [
        f"{condition_label(a, short=True)} − {condition_label(b, short=True)}"
        for a, b in zip(frame["condition_a"], frame["condition_b"])
    ]
    groups = sorted(frame["region_pair"].astype(str).unique())
    contrasts = list(dict.fromkeys(frame["contrast"]))
    palette = plt.get_cmap("tab10")

    fig, ax = plt.subplots(
        figsize=(settings.contrast_figure_width_in, settings.contrast_figure_height_in)
    )
    for group_index, group in enumerate(groups):
        subset = frame.loc[frame["region_pair"].astype(str) == group]
        offset = (group_index - (len(groups) - 1) / 2) * 0.18
        colour = palette(group_index % 10)
        for _, row in subset.iterrows():
            position = contrasts.index(row["contrast"]) + offset
            significant = bool(row.get("significant", False))
            ax.plot(
                [row["mean_difference"]], [position],
                marker="o", markersize=4.5, linestyle="none",
                color=colour,
                markerfacecolor=colour if significant else "white",
                markeredgewidth=0.9,
                label=region_pair_label(group) if row is subset.iloc[0] else None,
            )
    ax.axvline(0.0, color=MUTED_INK, linewidth=0.9, linestyle="--")
    ax.set_yticks(np.arange(len(contrasts)))
    ax.set_yticklabels(contrasts, fontsize=6)
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    unique["not significant (FDR)"] = plt.Line2D(
        [], [], marker="o", linestyle="none", color=INK, markerfacecolor="white"
    )
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=6, loc="best")
    heading = f"{scope_label(scope)}"
    if label:
        heading = f"{heading} — {label}"
    _finish(ax, xlabel="Within-pair difference in null-corrected cross-correlation", title=heading)
    fig.tight_layout()
    suffix = scope + (f"_{label.replace(' ', '_').lower()}" if label else "")
    return fig, save_thesis_figure(fig, settings, f"{stem}_{suffix}")


def plot_zero_lag_diagnostics(
    diagnostics: pd.DataFrame,
    settings: PairCoordinationPlotSettings,
    *,
    stem: str = "zero_lag_diagnostics",
) -> tuple[plt.Figure, dict[str, Path]]:
    """Per-date zero-lag prevalence, for spotting a contaminated recording day.

    Two randomly sampled neurons are essentially never monosynaptically
    connected, so a sharp zero-lag peak shared by most pairs on a day is common
    input or a recording artifact, not a pairwise interaction.  Flagged days are
    removed from every result; this shows which and why.
    """
    apply_thesis_plot_style()
    scopes = [s for s in ("within_region", "cross_region") if (diagnostics["scope"] == s).any()]
    if not scopes:
        return _empty_figure(settings, "No diagnostics available", stem)
    fig, axes = plt.subplots(
        1, len(scopes), figsize=(3.4 * len(scopes), 2.4), sharey=True, squeeze=False
    )
    for ax, scope in zip(axes[0], scopes):
        subset = diagnostics.loc[diagnostics["scope"] == scope].sort_values("date")
        values = subset["frac_pairs_zero_lag_above"].to_numpy(dtype=float)
        flagged = np.asarray(
            subset.get("suspected_zero_lag_artifact", pd.Series(False, index=subset.index)),
            dtype=bool,
        )
        positions = np.arange(len(subset))
        ax.bar(positions[~flagged], values[~flagged], color="#8fa8bf", edgecolor=INK, linewidth=0.4)
        if flagged.any():
            ax.bar(
                positions[flagged], values[flagged], color="#c0392b",
                edgecolor=INK, linewidth=0.4, label="removed",
            )
            ax.legend(frameon=False, fontsize=6, loc="upper left")
        ax.set_xticks(positions)
        ax.set_xticklabels(subset["date"].astype(str), fontsize=4.5, rotation=90)
        _finish(ax, title=scope_label(scope))
    axes[0][0].set_ylabel("Fraction of pairs with\nzero-lag z > 3", fontsize=6)
    fig.tight_layout()
    return fig, save_thesis_figure(fig, settings, stem)
