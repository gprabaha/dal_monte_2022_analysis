"""Figures for the circuit-level readouts of a fitted ensemble.

Every panel here is built so that a number cannot be read without its floor. Agreement
between fits is plotted against the agreement between *untrained* models of the same
architecture, because most of these measures are high before any training happens, and a
figure that showed only the fitted value would invite exactly the wrong conclusion.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    NEUTRAL_EDGE,
    NEUTRAL_FILL,
    nice_axis,
    region_label,
)

FITTED_COLOR = "#2a6f4e"
NULL_COLOR = "#b0b0b0"


def _panel(ax, letter: str) -> None:
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def plot_agreement_against_null(
    comparison: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 3.0),
):
    """Fitted agreement beside its architectural floor, per invariant.

    ``comparison`` comes from ``agreement_against_null``. The left panel is the raw
    numbers, which is how these were reported before the floor was measured; the right is
    the margin in units of the floor's own spread, which is what the numbers actually
    support. Measures are ordered strictest first.
    """
    table = comparison.dropna(subset=["fitted_mean"]).copy()
    positions = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    width = 0.36
    ax.bar(positions - width / 2, table["fitted_mean"], width,
           yerr=table["fitted_sd"].fillna(0.0), capsize=2,
           color=FITTED_COLOR, edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="fitted seeds")
    ax.bar(positions + width / 2, table["null_mean"], width,
           yerr=table["null_sd"].fillna(0.0), capsize=2,
           color=NULL_COLOR, edgecolor=NEUTRAL_EDGE, linewidth=0.6, label="untrained, same architecture")
    ax.set_xticks(positions)
    ax.set_xticklabels(table["measure"], rotation=28, ha="right", fontsize=6)
    ax.set_ylabel("agreement between two fits")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("the number, and what it is worth", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _panel(ax, "A")

    ax = axes[1]
    margin = table["margin_in_null_sd"].to_numpy(dtype=float)
    colours = [FITTED_COLOR if value > 2 else NULL_COLOR for value in margin]
    ax.bar(positions, margin, 0.6, color=colours, edgecolor=NEUTRAL_EDGE, linewidth=0.6)
    ax.axhline(0.0, color=INK, linewidth=0.9)
    ax.axhline(2.0, color=MUTED_INK, linewidth=0.8, linestyle="--")
    ax.text(len(table) - 0.5, 2.0, " above the floor", fontsize=5.8, color=MUTED_INK,
            va="bottom", ha="right")
    ax.set_xticks(positions)
    ax.set_xticklabels(table["measure"], rotation=28, ha="right", fontsize=6)
    ax.set_ylabel("margin over the floor\n(null standard deviations)")
    ax.set_title("what training actually bought", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _panel(ax, "B")

    fig.tight_layout()
    return fig


def plot_current_condition_alignment(
    alignment: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 2.9),
):
    """Whether two fixation types drive a pathway through the same channel.

    The cosine is computed within one model, so it needs no alignment step and is exact
    under the model's own symmetry. One column per condition pair, one row per pathway.
    """
    pairs = (alignment[["condition_a", "condition_b"]].drop_duplicates()
             .itertuples(index=False, name=None))
    pairs = list(pairs)
    pathways = sorted(alignment["pathway"].unique())
    grid = np.full((len(pathways), len(pairs)), np.nan)
    for column, (a, b) in enumerate(pairs):
        block = alignment[(alignment["condition_a"] == a) & (alignment["condition_b"] == b)]
        lookup = block.groupby("pathway")["cosine"].mean()
        for row, pathway in enumerate(pathways):
            if pathway in lookup.index:
                grid[row, column] = float(lookup[pathway])

    fig, ax = plt.subplots(figsize=figsize)
    limit = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    image = ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels(
        [f"{CONDITION_SHORT_LABELS.get(a, a)}\nvs {CONDITION_SHORT_LABELS.get(b, b)}"
         for a, b in pairs], fontsize=6)
    ax.set_yticks(np.arange(len(pathways)))
    ax.set_yticklabels(pathways, fontsize=6)
    ax.set_title("do two fixation types use the same inter-regional channel?",
                 fontsize=7, color=MUTED_INK, pad=6)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("cosine between currents", fontsize=6)
    bar.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    return fig


def plot_source_share(
    share: pd.DataFrame,
    *,
    regions: Sequence[str] | None = None,
    figsize: tuple[float, float] = (7.6, 4.4),
):
    """Share of each region's recurrent input by source, resolved in time and condition.

    Shares rather than magnitudes, because a share is a ratio of quantities in the same
    space and so survives the arbitrary scaling of any one fit.
    """
    targets = list(regions) if regions is not None else sorted(share["target"].unique())
    conditions = [c for c in CONDITION_ORDER if c in set(share["condition"])]
    fig, axes = plt.subplots(len(targets), len(conditions), figsize=figsize,
                             squeeze=False, sharex=True, sharey=True)
    sources = sorted(share["source"].unique())
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(sources)))

    for row, target in enumerate(targets):
        for column, condition in enumerate(conditions):
            ax = axes[row][column]
            block = share[(share["target"] == target) & (share["condition"] == condition)]
            for source, colour in zip(sources, colours):
                line = block[block["source"] == source].sort_values("time_s")
                style = "-" if source != target else ":"
                ax.plot(line["time_s"], line["share"], linewidth=1.0, color=colour,
                        linestyle=style, label=region_label(source) if row == 0 and column == 0 else None)
            ax.axvline(0.0, color=MUTED_INK, linewidth=0.7, linestyle="--")
            if row == 0:
                ax.set_title(CONDITION_SHORT_LABELS.get(condition, condition),
                             fontsize=7, color=MUTED_INK, pad=4)
            if column == 0:
                ax.set_ylabel(f"into {region_label(target)}", fontsize=6.5)
            if row == len(targets) - 1:
                ax.set_xlabel("time from fixation (s)")
            nice_axis(ax)
    axes[0][0].legend(fontsize=5.5, loc="upper left", ncol=2, title="source",
                      title_fontsize=5.5)
    fig.tight_layout()
    return fig


def plot_jacobian_spectra(
    spectra: dict[str, np.ndarray],
    *,
    bin_size_s: float = 0.01,
    figsize: tuple[float, float] = (7.4, 3.0),
):
    """Local dynamics at each condition's slow point, in the complex plane.

    The input is a constant one-hot per condition, so each condition is its own autonomous
    system and its Jacobian is well defined. Modulus above one is locally expanding, and
    the argument of a complex pair is an oscillation frequency -- which is how a claim
    about high-frequency structure in one fixation type becomes a claim about that
    condition's dynamical regime rather than about the fit.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    angles = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(angles), np.sin(angles), color=INK, linewidth=0.9, linestyle="--")
    for condition, eigenvalues in spectra.items():
        ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=16, alpha=0.8,
                   color=CONDITION_COLORS.get(condition, NEUTRAL_FILL),
                   edgecolor=NEUTRAL_EDGE, linewidth=0.4,
                   label=CONDITION_SHORT_LABELS.get(condition, condition))
    ax.set_xlabel("real")
    ax.set_ylabel("imaginary")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("Jacobian at the slow point", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _panel(ax, "A")

    ax = axes[1]
    for condition, eigenvalues in spectra.items():
        oscillating = eigenvalues[np.abs(np.imag(eigenvalues)) > 1e-9]
        if not len(oscillating):
            continue
        frequency = np.abs(np.angle(oscillating)) / (2.0 * np.pi * float(bin_size_s))
        ax.scatter(frequency, np.abs(oscillating), s=16, alpha=0.8,
                   color=CONDITION_COLORS.get(condition, NEUTRAL_FILL),
                   edgecolor=NEUTRAL_EDGE, linewidth=0.4,
                   label=CONDITION_SHORT_LABELS.get(condition, condition))
    ax.axhline(1.0, color=INK, linewidth=0.9, linestyle="--")
    ax.text(0.0, 1.005, "neither decaying nor growing", fontsize=5.8, color=INK, va="bottom")
    ax.set_xlabel("mode frequency (Hz)")
    ax.set_ylabel("|eigenvalue|")
    ax.set_title("which frequencies persist", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _panel(ax, "B")

    fig.tight_layout()
    return fig


__all__ = [
    "plot_agreement_against_null",
    "plot_current_condition_alignment",
    "plot_jacobian_spectra",
    "plot_source_share",
]
