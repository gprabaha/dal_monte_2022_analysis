"""Figures for the training-protocol notebook.

The subject of these panels is the optimiser rather than the brain, so they are built to
answer one question each: is the trajectory converging, and can the saved model be
trusted as the model the run found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib as mpl
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

MODEL_REGION_ORDER: tuple[str, ...] = ("ofc", "bla", "dmpfc", "accg")

PASS_COLOR = "#2a6f4e"
FAIL_COLOR = "#c1121f"
PENDING_COLOR = "#c7c7c7"


def _panel(ax, letter: str) -> None:
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def _log_y(ax) -> None:
    """Restore decade ticks after ``nice_axis`` installs a linear locator."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())


def plot_instability_problem(
    trajectories: Mapping[str, np.ndarray],
    spike_positions: np.ndarray,
    *,
    figsize: tuple[float, float] = (7.4, 2.7),
):
    """The problem: the saved iterate is not the best one, and the spikes never stop.

    ``trajectories`` maps a short label to one run's loss history. ``spike_positions``
    pools the within-run position of every upward jump across many runs.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1.0]})

    ax = axes[0]
    colors = ["#1b4965", "#5fa8d3", FAIL_COLOR, "#e07a5f"]
    for (label, losses), color in zip(trajectories.items(), colors):
        losses = np.asarray(losses, dtype=float)
        iterations = np.arange(1, len(losses) + 1)
        ax.plot(iterations, losses, color=color, linewidth=0.6, alpha=0.9, label=label)
        best = int(np.nanargmin(losses))
        ax.scatter([iterations[best]], [losses[best]], s=34, facecolor="white",
                   edgecolor=color, linewidth=1.2, zorder=5)
        ax.scatter([iterations[-1]], [losses[-1]], s=44, marker="X", color=color,
                   edgecolor=INK, linewidth=0.5, zorder=5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("training loss")
    ax.legend(fontsize=6, loc="upper right", ncol=1)
    ax.set_title("open circle = best iterate,  ✕ = what was saved", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _log_y(ax)
    _panel(ax, "A")

    ax = axes[1]
    ax.hist(np.asarray(spike_positions, dtype=float), bins=np.linspace(0, 1, 11),
            color=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE, linewidth=0.8)
    ax.axvspan(0.5, 1.0, color=FAIL_COLOR, alpha=0.10, zorder=0)
    late = float(np.mean(np.asarray(spike_positions, dtype=float) > 0.5))
    ax.text(0.75, ax.get_ylim()[1] * 0.94, f"{late:.0%} of spikes\nland here", fontsize=6.4,
            color=FAIL_COLOR, ha="center", va="top")
    ax.set_xlabel("position within training")
    ax.set_ylabel("upward loss jumps")
    ax.set_title("spikes do not stop as training proceeds", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _panel(ax, "B")

    fig.tight_layout()
    return fig


def plot_gradient_norm_diagnosis(
    gradient_norms: pd.DataFrame,
    binding: pd.DataFrame,
    *,
    historical_clip: float = 1.0,
    proposed_clip: float = 0.05,
    figsize: tuple[float, float] = (7.4, 2.6),
):
    """Why clipping never helped: the threshold sat far above any gradient it saw."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    values = gradient_norms["gradient_norm"].to_numpy(dtype=float)

    ax = axes[0]
    positive = values[values > 0]
    ax.hist(positive, bins=np.geomspace(positive.min(), max(positive.max(), historical_clip * 1.2), 50),
            color=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE, linewidth=0.5)
    ax.set_xscale("log")
    for threshold, color, label in (
        (proposed_clip, PASS_COLOR, f"proposed {proposed_clip:g}"),
        (historical_clip, FAIL_COLOR, f"historical {historical_clip:g}"),
    ):
        ax.axvline(threshold, color=color, linewidth=1.2, linestyle="--")
        ax.text(threshold, ax.get_ylim()[1] * 0.96, f" {label}", fontsize=6.2, color=color,
                ha="left", va="top", rotation=90)
    ax.axvline(float(np.median(positive)), color=INK, linewidth=1.0, linestyle=":")
    ax.text(float(np.median(positive)), ax.get_ylim()[1] * 0.5, " median", fontsize=6.2,
            color=INK, ha="left", va="center", rotation=90)
    ax.set_xlabel("gradient norm")
    ax.set_ylabel("steps")
    nice_axis(ax)
    _panel(ax, "A")

    ax = axes[1]
    ax.plot(binding["clip_norm"], 100 * binding["fraction_of_steps_clipped"],
            marker="o", markersize=4, color=NEUTRAL_EDGE, linewidth=1.2)
    for threshold, color in ((proposed_clip, PASS_COLOR), (historical_clip, FAIL_COLOR)):
        row = binding.loc[(binding["clip_norm"] - threshold).abs().idxmin()]
        ax.scatter([row["clip_norm"]], [100 * row["fraction_of_steps_clipped"]],
                   s=52, color=color, edgecolor=INK, linewidth=0.6, zorder=5)
        ax.annotate(f"{100 * row['fraction_of_steps_clipped']:.1f}% of steps",
                    (row["clip_norm"], 100 * row["fraction_of_steps_clipped"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=6.2, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clip threshold")
    ax.set_ylabel("% of steps actually clipped")
    ax.set_title("a threshold above the gradients is not a constraint", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _panel(ax, "B")

    fig.tight_layout()
    return fig


def plot_legacy_learning_rate_survey(survey: pd.DataFrame, *, figsize: tuple[float, float] = (7.4, 2.6)):
    """The one axis the legacy tree shows moving the instability, and by how much."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    grouped = (
        survey.groupby("lr")
        .agg(n=("run", "size"), spikes=("spikes_per_10k", "median"), final_is_best=("final_is_best", "mean"))
        .reset_index()
        .sort_values("lr")
    )

    ax = axes[0]
    ax.plot(grouped["lr"], grouped["spikes"], marker="o", markersize=5, color=FAIL_COLOR, linewidth=1.3)
    for _, row in grouped.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["lr"], row["spikes"]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=5.8, color=MUTED_INK)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("upward loss jumps\nper 10k iterations")
    nice_axis(ax)
    _panel(ax, "A")

    ax = axes[1]
    ax.plot(grouped["lr"], 100 * grouped["final_is_best"], marker="o", markersize=5,
            color=PASS_COLOR, linewidth=1.3)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("% of runs whose final iterate\nis their best")
    ax.set_ylim(-4, 104)
    ax.set_title("observational, and confounded — which is what the sweep is for",
                 fontsize=6.6, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _panel(ax, "B")

    fig.tight_layout()
    return fig


def plot_pilot_schedule_comparison(
    per_seed: pd.DataFrame,
    trajectories: Mapping[str, Sequence[np.ndarray]],
    *,
    figsize: tuple[float, float] = (7.4, 2.7),
):
    """The pilot: what a decaying rate does to the end of training, and what it costs.

    ``per_seed`` needs ``lr_schedule``, ``final_over_best`` and ``best_loss``;
    ``trajectories`` maps a schedule name to that schedule's loss histories.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
    colors = {"constant": FAIL_COLOR, "cosine": PASS_COLOR}

    ax = axes[0]
    for schedule, runs in trajectories.items():
        for index, losses in enumerate(runs):
            losses = np.asarray(losses, dtype=float)
            ax.plot(np.arange(1, len(losses) + 1), losses, color=colors.get(schedule, INK),
                    linewidth=0.5, alpha=0.75, label=schedule if index == 0 else None)
    ax.set_xlabel("iteration")
    ax.set_ylabel("training loss")
    ax.legend(fontsize=6.5, loc="upper right")
    nice_axis(ax)
    _log_y(ax)
    _panel(ax, "A")

    ax = axes[1]
    schedules = [s for s in ("constant", "cosine") if s in set(per_seed["lr_schedule"])]
    rng = np.random.default_rng(0)
    for index, schedule in enumerate(schedules):
        values = per_seed.loc[per_seed["lr_schedule"] == schedule, "final_over_best"].to_numpy(dtype=float)
        ax.scatter(index + rng.uniform(-0.1, 0.1, size=len(values)), values, s=22,
                   color=colors.get(schedule, INK), edgecolor=INK, linewidth=0.4, zorder=3)
    ax.axhline(1.0, color=INK, linewidth=0.9, linestyle=":")
    ax.text(len(schedules) - 0.5, 1.0, " final = best", fontsize=6.2, color=INK, va="center", ha="left")
    ax.set_xticks(range(len(schedules)))
    ax.set_xticklabels(schedules, fontsize=7)
    ax.set_ylabel("final loss / best loss")
    ax.set_yscale("log")
    ax.set_title("one dot per seed", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=6))
    _panel(ax, "B")

    ax = axes[2]
    for index, schedule in enumerate(schedules):
        values = per_seed.loc[per_seed["lr_schedule"] == schedule, "best_loss"].to_numpy(dtype=float)
        ax.scatter(index + rng.uniform(-0.1, 0.1, size=len(values)), values, s=22,
                   color=colors.get(schedule, INK), edgecolor=INK, linewidth=0.4, zorder=3)
        ax.scatter([index], [np.median(values)], s=70, marker="_", color=INK, zorder=4)
    ax.set_xticks(range(len(schedules)))
    ax.set_xticklabels(schedules, fontsize=7)
    ax.set_ylabel("best loss reached")
    ax.set_title("and what the smoothness costs", fontsize=7, color=MUTED_INK, pad=5)
    nice_axis(ax)
    _panel(ax, "C")

    fig.tight_layout()
    return fig


def plot_sweep_state(inventory: pd.DataFrame, *, figsize: tuple[float, float] = (7.4, 4.0)):
    """Which cells of the sweep exist, which failed, and which have not run.

    A run-state panel is not decoration here: the sweep is submitted from the notebook
    and completes over hours, so the notebook has to be readable while it is half done.
    """
    labels = list(dict.fromkeys(inventory["label"]))
    seeds = list(dict.fromkeys(inventory["seed"]))
    state = np.zeros((len(labels), len(seeds)))
    lookup = inventory.set_index(["label", "seed"])
    for row, label in enumerate(labels):
        for column, seed in enumerate(seeds):
            if (label, seed) not in lookup.index:
                continue
            cell = lookup.loc[(label, seed)]
            state[row, column] = 2.0 if bool(cell["complete"]) else (1.0 if bool(cell["diverged"]) else 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = mpl.colors.ListedColormap([PENDING_COLOR, FAIL_COLOR, PASS_COLOR])
    ax.imshow(state, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s)[:5] for s in seeds], fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_xlabel("seed")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(seeds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=7, color=PASS_COLOR, label="complete"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=7, color=FAIL_COLOR, label="diverged"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=7, color=PENDING_COLOR, label="not run"),
    ]
    ax.legend(handles=handles, fontsize=6.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.07), frameon=False)
    fig.tight_layout()
    return fig


def plot_protocol_results(
    results: pd.DataFrame,
    *,
    bar_label: str = "meets the convergence bar",
    figsize: tuple[float, float] = (7.4, 3.0),
):
    """Fit against convergence, with the bar drawn on it.

    The choice is a trade-off, not a maximisation: the smoothest configuration is not the
    one that fits best, so the panel shows both axes and lets the bar do the selecting.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    for passes, block in results.groupby("passes_bar"):
        ax.scatter(
            block["max_final_over_best"],
            block["median_best_loss"],
            s=34,
            color=PASS_COLOR if passes else FAIL_COLOR,
            edgecolor=INK,
            linewidth=0.4,
            label=bar_label if passes else "fails it",
            zorder=3,
        )
    winners = results[results["passes_bar"]]
    if len(winners):
        best = winners.loc[winners["median_best_loss"].idxmin()]
        ax.annotate(str(best["label"]), (best["max_final_over_best"], best["median_best_loss"]),
                    textcoords="offset points", xytext=(8, 0), fontsize=5.8, color=INK, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("worst final / best across seeds")
    ax.set_ylabel("median best loss")
    ax.legend(fontsize=6.2, loc="upper right")
    nice_axis(ax)
    _panel(ax, "A")

    ax = axes[1]
    ordered = results.sort_values("median_best_loss")
    y = np.arange(len(ordered))
    ax.barh(y, ordered["median_best_loss"],
            color=[PASS_COLOR if p else FAIL_COLOR for p in ordered["passes_bar"]],
            edgecolor=INK, linewidth=0.5, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["label"], fontsize=5.2)
    ax.set_xscale("log")
    ax.set_xlabel("median best loss")
    ax.invert_yaxis()
    nice_axis(ax)
    _panel(ax, "B")

    fig.tight_layout()
    return fig


def plot_reconstruction_quality(
    quality: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 3.0),
):
    """R^2 per region and condition, in both spaces, for the runs scored.

    Two spaces because they can disagree. The PC target weights all 42 components
    equally; firing-rate space is dominated by the high-variance ones. A model that
    loses the small PCs is visible on the left and invisible on the right.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    regions = [r for r in MODEL_REGION_ORDER if r in set(quality["region"])]
    conditions = [c for c in CONDITION_ORDER if c in set(quality["condition"])]
    width = 0.8 / max(len(conditions), 1)

    for ax, space, title in zip(axes, ("pc", "fr"), ("region PC space", "PC-backprojected firing rates")):
        block = quality[quality["space"] == space]
        for condition_index, condition in enumerate(conditions):
            values, errors = [], []
            for region in regions:
                cell = block[(block["region"] == region) & (block["condition"] == condition)]["r2"]
                values.append(float(cell.median()) if len(cell) else np.nan)
                errors.append(float(cell.max() - cell.min()) / 2 if len(cell) > 1 else 0.0)
            x = np.arange(len(regions)) + condition_index * width - 0.4 + width / 2
            ax.bar(x, values, yerr=errors, width=width * 0.9,
                   color=CONDITION_COLORS[condition], edgecolor=INK, linewidth=0.6,
                   error_kw=dict(elinewidth=0.7, capsize=1.5),
                   label=CONDITION_SHORT_LABELS[condition], zorder=2)
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels([region_label(r) for r in regions], fontsize=7)
        ax.set_title(title, fontsize=7.5, pad=5)
        ax.set_ylim(0, 1.04)
        ax.axhline(1.0, color=MUTED_INK, linewidth=0.6, linestyle=":")
        nice_axis(ax)

    axes[0].set_ylabel("$R^2$")
    axes[0].legend(fontsize=6, loc="lower left", ncol=3, columnspacing=0.7, handletextpad=0.4)
    _panel(axes[0], "A")
    _panel(axes[1], "B")
    fig.tight_layout()
    return fig


def plot_loss_versus_reconstruction(
    comparison: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 2.6),
):
    """Is the training loss a good proxy for reconstruction quality on these fits?

    Selection is made on the loss, which is only legitimate if the runs the loss prefers
    are the runs that reconstruct best. Scatter rather than assert it.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, column, label in zip(axes, ("mean_pc_r2", "mean_fr_r2"), ("mean PC $R^2$", "mean firing-rate $R^2$")):
        ax.scatter(comparison["best_loss"], comparison[column], s=30,
                   color=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE, linewidth=0.7, zorder=3)
        finite = comparison[np.isfinite(comparison["best_loss"]) & np.isfinite(comparison[column])]
        if len(finite) > 2:
            rho = finite["best_loss"].corr(finite[column], method="spearman")
            ax.set_title(f"Spearman ρ = {rho:+.2f}  (n = {len(finite)})", fontsize=7, color=MUTED_INK, pad=5)
        ax.set_xscale("log")
        ax.set_xlabel("best training loss")
        ax.set_ylabel(label)
        nice_axis(ax)
    _panel(axes[0], "A")
    _panel(axes[1], "B")
    fig.tight_layout()
    return fig


def _overlay_panels(traces: pd.DataFrame, value_columns: tuple[str, str], title: str,
                    figsize: tuple[float, float], series_column: str | None):
    regions = [r for r in MODEL_REGION_ORDER if r in set(traces["region"])]
    conditions = [c for c in CONDITION_ORDER if c in set(traces["condition"])]
    fig, axes = plt.subplots(len(regions), len(conditions), figsize=figsize,
                             sharex=True, squeeze=False)
    observed_column, predicted_column = value_columns
    for row, region in enumerate(regions):
        for column, condition in enumerate(conditions):
            ax = axes[row][column]
            block = traces[(traces["region"] == region) & (traces["condition"] == condition)]
            series = sorted(block[series_column].unique()) if series_column else [None]
            for index, key in enumerate(series):
                panel = block if key is None else block[block[series_column] == key]
                panel = panel.sort_values("time_s")
                alpha = 1.0 - 0.28 * index
                ax.plot(panel["time_s"], panel[observed_column],
                        color=CONDITION_COLORS[condition], linewidth=1.1, alpha=alpha, zorder=3)
                ax.plot(panel["time_s"], panel[predicted_column],
                        color=INK, linewidth=0.9, linestyle="--", alpha=alpha, zorder=4)
            ax.axvline(0.0, color=MUTED_INK, linewidth=0.6, linestyle=":", zorder=1)
            if row == 0:
                ax.set_title(CONDITION_SHORT_LABELS[condition], fontsize=7.5, pad=4)
            if column == 0:
                ax.set_ylabel(region_label(region), fontsize=7.5)
            if row == len(regions) - 1:
                ax.set_xlabel("time from fixation (s)", fontsize=7)
            nice_axis(ax, y_ticks=3)
    handles = [
        plt.Line2D([], [], color=INK, linewidth=1.2, label="observed"),
        plt.Line2D([], [], color=INK, linewidth=1.0, linestyle="--", label="mRNN"),
    ]
    fig.legend(handles=handles, fontsize=6.5, ncol=2, loc="upper right",
               bbox_to_anchor=(0.99, 1.0), frameon=False)
    fig.suptitle(title, fontsize=8.5, y=1.015)
    fig.tight_layout()
    return fig


def plot_pc_trace_overlay(
    traces: pd.DataFrame,
    *,
    title: str = "Region PC trajectories — observed against mRNN",
    figsize: tuple[float, float] = (7.4, 6.2),
):
    """Observed and reconstructed PC trajectories, region by condition.

    Coloured solid lines are observed, dark dashed are the model, and successive
    components fade. This is the check the loss cannot give: whether a well-scoring fit
    tracks the shape of the transients or has smoothed them into a slow drift.
    """
    return _overlay_panels(traces, ("observed", "predicted"), title, figsize, "component")


def plot_firing_rate_trace_overlay(
    traces: pd.DataFrame,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (7.4, 6.2),
):
    """One example unit per region, in PC-backprojected firing-rate space."""
    if title is None:
        rank = str(traces["unit_rank"].iloc[0]) if len(traces) else "example"
        title = f"Backprojected firing rate — {rank}-fitting unit per region"
    labelled = traces.copy()
    figure = _overlay_panels(labelled, ("observed", "predicted"), title, figsize, None)
    regions = [r for r in MODEL_REGION_ORDER if r in set(traces["region"])]
    for row, region in enumerate(regions):
        cell = traces[traces["region"] == region]
        if not len(cell):
            continue
        figure.axes[row * 3].text(
            0.02, 0.94, f"unit {int(cell['unit_index'].iloc[0])}, $R^2$ = {float(cell['unit_r2'].iloc[0]):.3f}",
            transform=figure.axes[row * 3].transAxes, fontsize=5.8, color=MUTED_INK,
            va="top", ha="left",
        )
    return figure


__all__ = [
    "plot_reconstruction_quality",
    "plot_pc_trace_overlay",
    "plot_loss_versus_reconstruction",
    "plot_firing_rate_trace_overlay",
    "plot_gradient_norm_diagnosis",
    "plot_instability_problem",
    "plot_legacy_learning_rate_survey",
    "plot_pilot_schedule_comparison",
    "plot_protocol_results",
    "plot_sweep_state",
]
