"""Small plotting helpers for fixation mRNN notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
    compute_global_flow_field,
    extract_region_currents,
    output_pc_scores,
)


DEFAULT_CONDITION_COLORS = {
    "face_interactive": "#b64198",
    "face_non_interactive": "#4c9a2a",
    "object": "#6f4e37",
}
DEFAULT_REGION_COLORS = {
    "ofc": "#4c78a8",
    "bla": "#f58518",
    "dmpfc": "#54a24b",
    "accg": "#e45756",
}


@dataclass
class FixationMRNNDiagnosticPlotSettings:
    condition_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONDITION_COLORS))
    region_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_REGION_COLORS))
    figsize: tuple[float, float] = (11.0, 7.0)
    linewidth: float = 1.5
    dpi: int = 150
    flow_scale: float = 2.5


def plot_fixation_mrnn_activation_pc_timeseries(
    replay: Mapping[str, object],
    *,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot PC1-PC3 of reconstructed output time series."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    regions = tuple(region_order or replay["region_order"])
    conditions = tuple(condition_order or replay["condition_order"])
    timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
    fig, axes = plt.subplots(
        len(regions),
        len(conditions),
        figsize=settings.figsize,
        dpi=settings.dpi,
        squeeze=False,
        sharex=True,
    )
    for row, region in enumerate(regions):
        scores = output_pc_scores(replay, region=region, n_components=3)
        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            cond_idx = replay["condition_order"].index(condition)
            color = settings.condition_colors.get(condition, "black")
            for pc_idx, linestyle in enumerate(("-", "--", ":")):
                ax.plot(
                    timeline,
                    scores[cond_idx, :, pc_idx],
                    color=color,
                    linestyle=linestyle,
                    linewidth=settings.linewidth,
                    label=f"PC{pc_idx + 1}",
                )
            if row == 0:
                ax.set_title(condition)
            if col == 0:
                ax.set_ylabel(f"{region}\\nscore")
            if row == len(regions) - 1:
                ax.set_xlabel("Time (s)")
    axes[0, -1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig, axes


def plot_fixation_mrnn_activation_trajectories_3d(
    replay: Mapping[str, object],
    *,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot reconstructed output PC trajectories in 3D."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    regions = tuple(region_order or replay["region_order"])
    conditions = tuple(condition_order or replay["condition_order"])
    fig = plt.figure(figsize=settings.figsize, dpi=settings.dpi)
    axes = []
    for idx, region in enumerate(regions):
        ax = fig.add_subplot(2, int(np.ceil(len(regions) / 2)), idx + 1, projection="3d")
        axes.append(ax)
        scores = output_pc_scores(replay, region=region, n_components=3)
        for condition in conditions:
            cond_idx = replay["condition_order"].index(condition)
            color = settings.condition_colors.get(condition, "black")
            xyz = scores[cond_idx]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, label=condition)
        ax.set_title(region)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig, axes


def plot_fixation_mrnn_current_influence(
    replay: Mapping[str, object],
    *,
    current_vectors: Mapping[tuple[str, str], object] | None = None,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot signed projection-based current contributions over time."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    current_df, _ = extract_region_currents(replay)
    regions = tuple(region_order or replay["region_order"])
    conditions = tuple(condition_order or replay["condition_order"])
    timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
    fig, axes = plt.subplots(
        len(regions),
        len(conditions),
        figsize=settings.figsize,
        dpi=settings.dpi,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, target_region in enumerate(regions):
        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            subset = current_df[
                (current_df["target_region"] == target_region)
                & (current_df["condition"] == condition)
            ]
            values = []
            for source_region in regions:
                source = subset.loc[subset["source_region"] == source_region].sort_values("time_idx")
                values.append(
                    (
                        source_region,
                        source["relative_contribution"].to_numpy(dtype=float),
                    )
                )
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.45)
            for source_region, contribution in values:
                ax.plot(
                    timeline,
                    contribution,
                    color=settings.region_colors.get(source_region, "gray"),
                    linewidth=settings.linewidth,
                    label=source_region,
                )
                ax.fill_between(
                    timeline,
                    0.0,
                    contribution,
                    color=settings.region_colors.get(source_region, "gray"),
                    alpha=0.18,
                )
            ax.set_ylim(-1, 1)
            if row == 0:
                ax.set_title(condition)
            if col == 0:
                ax.set_ylabel(f"{target_region}\nsigned rel.")
            if row == len(regions) - 1:
                ax.set_xlabel("Time (s)")
    axes[0, -1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig, axes


def plot_fixation_mrnn_average_current_influence_bars(
    replay: Mapping[str, object],
    *,
    current_vectors: Mapping[tuple[str, str], object] | None = None,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot time-averaged signed current contributions as diverging bars."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    current_df, _ = extract_region_currents(replay)
    regions = tuple(region_order or replay["region_order"])
    conditions = tuple(condition_order or replay["condition_order"])
    fig, axes = plt.subplots(len(regions), len(conditions), figsize=settings.figsize, dpi=settings.dpi, squeeze=False)
    colors = [settings.region_colors.get(region, "gray") for region in regions]
    for row, target_region in enumerate(regions):
        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            subset = current_df[
                (current_df["target_region"] == target_region)
                & (current_df["condition"] == condition)
            ]
            means = [
                subset.loc[subset["source_region"] == region, "relative_contribution"].mean()
                for region in regions
            ]
            y_pos = np.arange(len(regions))
            ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.5)
            ax.barh(y_pos, means, color=colors, alpha=0.86)
            ax.set_yticks(y_pos, labels=regions, fontsize=7)
            ax.set_xlim(-1.0, 1.0)
            ax.tick_params(axis="x", labelsize=7)
            if row == 0:
                ax.set_title(condition)
            if col == 0:
                ax.set_ylabel(target_region)
            if row == len(regions) - 1:
                ax.set_xlabel("mean signed rel.")
    fig.tight_layout()
    return fig, axes


def plot_fixation_mrnn_average_current_influence_pies(*args, **kwargs):
    """Deprecated alias: signed contributions should be shown as bars."""
    return plot_fixation_mrnn_average_current_influence_bars(*args, **kwargs)


def plot_fixation_mrnn_signal_evolution(
    replay: Mapping[str, object],
    *,
    current_vectors: Mapping[tuple[str, str], object] | None = None,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot output PC1 plus signed current contributions for each region."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    current_df, _ = extract_region_currents(replay)
    regions = tuple(region_order or replay["region_order"])
    conditions = tuple(condition_order or replay["condition_order"])
    timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
    fig = plt.figure(figsize=(settings.figsize[0], settings.figsize[1] * 1.35), dpi=settings.dpi)
    grid = fig.add_gridspec(len(regions), len(conditions), hspace=0.42, wspace=0.22)
    axes = {}
    for row, region in enumerate(regions):
        scores = output_pc_scores(replay, region=region, n_components=1)
        for col, condition in enumerate(conditions):
            inner = grid[row, col].subgridspec(2, 1, height_ratios=(1.0, 1.1), hspace=0.08)
            ax_signal = fig.add_subplot(inner[0, 0])
            ax_current = fig.add_subplot(inner[1, 0], sharex=ax_signal)
            axes[(region, condition)] = (ax_signal, ax_current)
            cond_idx = replay["condition_order"].index(condition)
            ax_signal.plot(
                timeline,
                scores[cond_idx, :, 0],
                color=settings.condition_colors.get(condition, "black"),
                linewidth=settings.linewidth,
            )
            ax_signal.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
            ax_signal.tick_params(axis="x", labelbottom=False)
            if row == 0:
                ax_signal.set_title(condition)
            if col == 0:
                ax_signal.set_ylabel(f"{region}\noutput PC1")

            subset = current_df[
                (current_df["target_region"] == region)
                & (current_df["condition"] == condition)
            ]
            ax_current.axhline(0.0, color="black", linewidth=0.6, alpha=0.45)
            for source_region in regions:
                source = subset.loc[subset["source_region"] == source_region].sort_values("time_idx")
                contribution = source["relative_contribution"].to_numpy(dtype=float)
                color = settings.region_colors.get(source_region, "gray")
                ax_current.plot(timeline, contribution, color=color, linewidth=settings.linewidth, label=source_region)
                ax_current.fill_between(timeline, 0.0, contribution, color=color, alpha=0.16)
            ax_current.set_ylim(-1, 1)
            if col == 0:
                ax_current.set_ylabel("signed rel.")
            if row == len(regions) - 1:
                ax_current.set_xlabel("Time (s)")
    axes[(regions[0], conditions[-1])][1].legend(frameon=False, fontsize=7)
    return fig, axes


def plot_fixation_mrnn_flow_fields_at_time(
    replay: Mapping[str, object],
    *,
    time_idx: int | None = None,
    time_s: float = 0.0,
    grid_points: int = 15,
    radius: float = 1.0,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot 2D Elman flow fields in each region's hidden PC space."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    conditions = tuple(condition_order or replay["condition_order"])
    timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
    selected_idx = int(np.argmin(np.abs(timeline - float(time_s)))) if time_idx is None else int(time_idx)
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=settings.figsize,
        dpi=settings.dpi,
        squeeze=False,
    )
    for col, condition in enumerate(conditions):
        ax = axes[0, col]
        field = compute_global_flow_field(
            replay,
            condition=condition,
            time_idx=selected_idx,
            grid_points=grid_points,
            radius=radius,
        )
        ax.quiver(
            field["grid_x"],
            field["grid_y"],
            field["u"] * settings.flow_scale,
            field["v"] * settings.flow_scale,
            field["speed"],
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            pivot="mid",
            width=0.006,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(condition)
        ax.set_xlabel("global hidden PC1")
        if col == 0:
            ax.set_ylabel("global hidden PC2")
    fig.suptitle(f"Global Elman mRNN Flow Fields (t={timeline[selected_idx]:.3f}s)", y=1.01)
    fig.tight_layout()
    return fig, axes


__all__ = [
    "FixationMRNNDiagnosticPlotSettings",
    "plot_fixation_mrnn_activation_pc_timeseries",
    "plot_fixation_mrnn_activation_trajectories_3d",
    "plot_fixation_mrnn_average_current_influence_bars",
    "plot_fixation_mrnn_average_current_influence_pies",
    "plot_fixation_mrnn_current_influence",
    "plot_fixation_mrnn_flow_fields_at_time",
    "plot_fixation_mrnn_signal_evolution",
]
