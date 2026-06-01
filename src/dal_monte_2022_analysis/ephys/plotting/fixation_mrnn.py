"""Small plotting helpers for fixation mRNN notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
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
    """Stackplot relative recurrent current contributions."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    current_df, _ = extract_region_currents(replay) if current_vectors is None else (None, current_vectors)
    if current_df is None:
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
            colors = []
            for source_region in regions:
                source = subset.loc[subset["source_region"] == source_region].sort_values("time_idx")
                values.append(source["relative_contribution"].to_numpy())
                colors.append(settings.region_colors.get(source_region, "gray"))
            ax.stackplot(timeline, values, colors=colors)
            ax.set_ylim(0, 1)
            if row == 0:
                ax.set_title(condition)
            if col == 0:
                ax.set_ylabel(target_region)
            if row == len(regions) - 1:
                ax.set_xlabel("Time (s)")
    fig.tight_layout()
    return fig, axes


def plot_fixation_mrnn_average_current_influence_pies(
    replay: Mapping[str, object],
    *,
    current_vectors: Mapping[tuple[str, str], object] | None = None,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot time-averaged current contributions."""
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
            ax.pie(means, colors=colors, autopct="%1.0f%%", textprops={"fontsize": 7})
            if row == 0:
                ax.set_title(condition)
            if col == 0:
                ax.set_ylabel(target_region)
    fig.tight_layout()
    return fig, axes


def plot_fixation_mrnn_flow_fields_at_time(*args, **kwargs):
    """Flow fields were intentionally removed from the minimal Elman pipeline."""
    raise NotImplementedError("Flow fields are not part of the minimal Elman mRNN pipeline.")


__all__ = [
    "FixationMRNNDiagnosticPlotSettings",
    "plot_fixation_mrnn_activation_pc_timeseries",
    "plot_fixation_mrnn_activation_trajectories_3d",
    "plot_fixation_mrnn_average_current_influence_pies",
    "plot_fixation_mrnn_current_influence",
    "plot_fixation_mrnn_flow_fields_at_time",
]
