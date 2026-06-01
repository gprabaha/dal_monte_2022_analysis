"""Diagnostic plots for fixation mRNN replay outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
    checkpoint_region_order,
    compute_fixation_mrnn_currents,
    compute_fixation_mrnn_flow_fields,
)


DEFAULT_MRNN_CONDITION_LABELS = {
    "face_interactive": "Face interactive",
    "face_non_interactive": "Face non-interactive",
    "object": "Object",
}
DEFAULT_MRNN_CONDITION_COLORS = {
    "face_interactive": "#b64198",
    "face_non_interactive": "#97ca3d",
    "object": "#754c29",
}
DEFAULT_MRNN_REGION_COLORS = {
    "ofc": "#4c78a8",
    "bla": "#f58518",
    "dmpfc": "#54a24b",
    "accg": "#e45756",
}


@dataclass
class FixationMRNNDiagnosticPlotSettings:
    """Display settings for fixation mRNN diagnostic figures."""

    condition_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_MRNN_CONDITION_LABELS)
    )
    condition_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_MRNN_CONDITION_COLORS)
    )
    region_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_MRNN_REGION_COLORS)
    )
    activation_state: str = "h"
    activation_figsize: tuple[float, float] = (9.5, 7.2)
    pc_timeseries_figsize: tuple[float, float] = (12.0, 8.2)
    current_figsize: tuple[float, float] = (12.0, 10.0)
    pie_figsize: tuple[float, float] = (11.0, 8.5)
    flow_figsize: tuple[float, float] = (11.0, 8.5)
    flow_arrow_scale: float = 2
    line_width: float = 1.8
    marker_size: float = 22.0
    alpha: float = 0.94
    pc_alphas: tuple[float, float, float] = (0.98, 0.66, 0.38)
    pc_line_styles: tuple[str, str, str] = ("-", "--", ":")
    dpi: int = 150


def _region_display(region: str) -> str:
    labels = {
        "ofc": "OFC",
        "bla": "BLA",
        "dmpfc": "dmPFC",
        "accg": "ACCg",
    }
    return labels.get(str(region), str(region))


def _condition_display(
    condition: str,
    settings: FixationMRNNDiagnosticPlotSettings,
) -> str:
    return settings.condition_labels.get(str(condition), str(condition))


def _timeline(replay: Mapping[str, object], n_time: int) -> np.ndarray:
    checkpoint = replay.get("checkpoint", {})
    raw = checkpoint.get("timeline_s_rel") if isinstance(checkpoint, Mapping) else None
    if raw is None:
        return np.arange(int(n_time), dtype=float)
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size != int(n_time):
        return np.arange(int(n_time), dtype=float)
    return arr


def _state_sequence(replay: Mapping[str, object], state: str) -> torch.Tensor:
    token = str(state).strip().lower()
    if token in {"h", "hidden", "activation", "activations"}:
        return replay["h_seq"]
    if token in {"x", "state", "preactivation", "pre_activation"}:
        return replay["x_seq"]
    raise ValueError("state must be one of 'h' or 'x'.")


def _region_activity_np(
    replay: Mapping[str, object],
    region: str,
    *,
    state: str,
) -> np.ndarray:
    model = replay["model"]
    seq = _state_sequence(replay, state)
    activity = model.mrnn.get_region_activity(seq, str(region))
    return activity.detach().cpu().numpy().astype(float, copy=False)


def _fit_condition_pca_trajectories(
    activity_by_condition: np.ndarray,
    *,
    n_components: int,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    arr = np.asarray(activity_by_condition, dtype=float)
    if arr.ndim != 3:
        raise ValueError("activity_by_condition must have shape condition x time x feature.")
    n_conditions, n_time, n_features = arr.shape
    flat = arr.reshape(n_conditions * n_time, n_features)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_fit = min(int(n_components), vt.shape[0])
    scores = np.zeros((flat.shape[0], int(n_components)), dtype=float)
    if n_fit > 0:
        scores[:, :n_fit] = centered @ vt[:n_fit].T
    projected = scores.reshape(n_conditions, n_time, int(n_components))
    denom = max(flat.shape[0] - 1, 1)
    explained = (singular_values**2) / denom
    total = float(explained.sum())
    ratios = explained / total if total > 0 else np.zeros_like(explained)
    return (
        {idx: projected[idx] for idx in range(n_conditions)},
        {
            "mean": mean.squeeze(0),
            "components": vt[:n_fit],
            "explained_variance_ratio": ratios[:n_fit],
        },
    )


def _apply_equal_3d_limits(ax, points: list[np.ndarray]) -> None:
    if not points:
        return
    stack = np.vstack(points)
    finite = np.isfinite(stack).all(axis=1)
    if not finite.any():
        return
    stack = stack[finite]
    mins = np.min(stack, axis=0)
    maxs = np.max(stack, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(np.maximum(maxs - mins, 1e-6)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_fixation_mrnn_activation_trajectories_3d(
    replay: Mapping[str, object],
    *,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot per-region hidden activation trajectories in the first three PCs."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    regions = tuple(region_order or checkpoint_region_order(replay))
    conditions = tuple(condition_order or replay["condition_names"])
    condition_names = tuple(replay["condition_names"])
    condition_indices = [condition_names.index(condition) for condition in conditions]

    fig = plt.figure(figsize=settings.activation_figsize, dpi=int(settings.dpi))
    n_cols = 2
    n_rows = int(np.ceil(len(regions) / n_cols))
    axes = []
    for idx, region in enumerate(regions):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        axes.append(ax)
        activity = _region_activity_np(
            replay,
            str(region),
            state=settings.activation_state,
        )[condition_indices]
        projected, _ = _fit_condition_pca_trajectories(activity, n_components=3)
        all_points: list[np.ndarray] = []
        for local_idx, condition in enumerate(conditions):
            xyz = projected[local_idx]
            all_points.append(xyz)
            color = settings.condition_colors.get(str(condition), "#777777")
            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                color=color,
                linewidth=float(settings.line_width),
                alpha=float(settings.alpha),
                label=_condition_display(str(condition), settings),
            )
            ax.scatter(
                xyz[0:1, 0],
                xyz[0:1, 1],
                xyz[0:1, 2],
                color=color,
                edgecolors="black",
                linewidths=0.45,
                s=float(settings.marker_size),
                marker="o",
            )
            ax.scatter(
                xyz[-1:, 0],
                xyz[-1:, 1],
                xyz[-1:, 2],
                color=color,
                edgecolors="black",
                linewidths=0.45,
                s=float(settings.marker_size),
                marker="^",
            )
        _apply_equal_3d_limits(ax, all_points)
        ax.set_title(_region_display(str(region)), fontsize=10, pad=5)
        ax.set_xlabel("PC1", labelpad=-2)
        ax.set_ylabel("PC2", labelpad=-2)
        ax.set_zlabel("PC3", labelpad=-2)
        ax.tick_params(axis="both", labelsize=7, pad=0)
        ax.tick_params(axis="z", labelsize=7, pad=0)
        ax.view_init(elev=24, azim=-58)
        ax.grid(True, linewidth=0.35, alpha=0.35)

    handles = [
        Line2D(
            [0],
            [0],
            color=settings.condition_colors.get(str(condition), "#777777"),
            linewidth=float(settings.line_width),
            label=_condition_display(str(condition), settings),
        )
        for condition in conditions
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle("mRNN Activation Trajectories", y=0.985, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig, axes


def _plot_pc_timeseries(
    ax,
    *,
    time: np.ndarray,
    scores_time_by_pc: np.ndarray,
    color: str,
    settings: FixationMRNNDiagnosticPlotSettings,
    show_legend: bool = False,
) -> None:
    for pc_idx in range(min(3, scores_time_by_pc.shape[1])):
        alpha = settings.pc_alphas[min(pc_idx, len(settings.pc_alphas) - 1)]
        linestyle = settings.pc_line_styles[min(pc_idx, len(settings.pc_line_styles) - 1)]
        ax.plot(
            time,
            scores_time_by_pc[:, pc_idx],
            color=color,
            alpha=float(alpha),
            linestyle=str(linestyle),
            linewidth=float(settings.line_width),
            label=f"PC{pc_idx + 1}",
        )
    ax.axhline(0.0, color="#333333", linewidth=0.45, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="both", labelsize=6, pad=1)
    ax.grid(axis="y", linewidth=0.3, alpha=0.23)
    if show_legend:
        ax.legend(frameon=False, fontsize=6, loc="upper right", handlelength=2.0)


def plot_fixation_mrnn_activation_pc_timeseries(
    replay: Mapping[str, object],
    *,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot PC1-PC3 activation scores over time for each region and condition."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    regions = tuple(region_order or checkpoint_region_order(replay))
    conditions = tuple(condition_order or replay["condition_names"])
    condition_names = tuple(replay["condition_names"])
    condition_indices = [condition_names.index(condition) for condition in conditions]

    n_rows = len(regions)
    n_cols = len(conditions)
    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=settings.pc_timeseries_figsize,
        dpi=int(settings.dpi),
        squeeze=False,
        sharex=True,
    )
    axes: dict[tuple[str, str], plt.Axes] = {}
    for row_idx, region in enumerate(regions):
        activity = _region_activity_np(
            replay,
            str(region),
            state=settings.activation_state,
        )
        projected, _ = _fit_condition_pca_trajectories(activity, n_components=3)
        time = _timeline(replay, activity.shape[1])
        for col_idx, condition in enumerate(conditions):
            ax = axes_arr[row_idx, col_idx]
            axes[(str(region), str(condition))] = ax
            cond_idx = condition_indices[col_idx]
            color = settings.condition_colors.get(str(condition), "#777777")
            _plot_pc_timeseries(
                ax,
                time=time,
                scores_time_by_pc=projected[cond_idx],
                color=color,
                settings=settings,
                show_legend=(row_idx == 0 and col_idx == n_cols - 1),
            )
            if row_idx == 0:
                ax.set_title(_condition_display(str(condition), settings), fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{_region_display(str(region))}\nPC score", fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)
    fig.suptitle("mRNN Activation PC Timecourses", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return fig, axes


def _current_relative_influence(
    current_vectors: Mapping[tuple[str, str], torch.Tensor],
    *,
    target_region: str,
    source_regions: Sequence[str],
    condition_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    norms: list[np.ndarray] = []
    for source_region in source_regions:
        vector = current_vectors[(str(target_region), str(source_region))]
        source_norm = torch.linalg.vector_norm(vector[condition_idx], dim=-1)
        norms.append(source_norm.detach().cpu().numpy().astype(float, copy=False))
    norm_arr = np.vstack(norms)
    denom = np.sum(norm_arr, axis=0, keepdims=True)
    relative = np.divide(norm_arr, denom, out=np.zeros_like(norm_arr), where=denom > 0)
    return norm_arr, relative


def plot_fixation_mrnn_current_influence(
    replay: Mapping[str, object],
    *,
    current_vectors: Mapping[tuple[str, str], torch.Tensor] | None = None,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot PC timecourses with stacked relative source-region current norms."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    regions = tuple(region_order or checkpoint_region_order(replay))
    source_regions = regions
    conditions = tuple(condition_order or replay["condition_names"])
    condition_names = tuple(replay["condition_names"])
    condition_indices = [condition_names.index(condition) for condition in conditions]
    if current_vectors is None:
        _, current_vectors = compute_fixation_mrnn_currents(replay)

    n_rows = len(regions)
    n_cols = len(conditions)
    fig = plt.figure(figsize=settings.current_figsize, dpi=int(settings.dpi))
    outer = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.055,
        right=0.985,
        top=0.93,
        bottom=0.07,
        wspace=0.22,
        hspace=0.36,
    )
    axes: dict[tuple[str, str], tuple[plt.Axes, plt.Axes]] = {}

    for row_idx, target_region in enumerate(regions):
        activity = _region_activity_np(
            replay,
            str(target_region),
            state=settings.activation_state,
        )
        projected, _ = _fit_condition_pca_trajectories(activity, n_components=3)
        time = _timeline(replay, activity.shape[1])
        for col_idx, condition in enumerate(conditions):
            cond_idx = condition_indices[col_idx]
            inner = outer[row_idx, col_idx].subgridspec(
                2,
                1,
                height_ratios=(1.15, 0.85),
                hspace=0.08,
            )
            ax_traj = fig.add_subplot(inner[0, 0])
            ax_area = fig.add_subplot(inner[1, 0])
            axes[(str(target_region), str(condition))] = (ax_traj, ax_area)

            pc_scores = projected[cond_idx]
            color = settings.condition_colors.get(str(condition), "#777777")
            _plot_pc_timeseries(
                ax_traj,
                time=time,
                scores_time_by_pc=pc_scores,
                color=color,
                settings=settings,
                show_legend=(row_idx == 0 and col_idx == n_cols - 1),
            )
            ax_traj.set_xlim(float(time[0]), float(time[-1]))
            ax_traj.set_xticklabels([])
            ax_traj.set_ylabel("PC score", fontsize=7)
            if row_idx == 0:
                ax_traj.set_title(_condition_display(str(condition), settings), fontsize=9)
            if col_idx == 0:
                ax_traj.text(
                    -0.28,
                    0.5,
                    _region_display(str(target_region)),
                    transform=ax_traj.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

            _, relative = _current_relative_influence(
                current_vectors,
                target_region=str(target_region),
                source_regions=source_regions,
                condition_idx=cond_idx,
            )
            colors = [
                settings.region_colors.get(str(source_region), "#777777")
                for source_region in source_regions
            ]
            ax_area.stackplot(
                time,
                relative,
                colors=colors,
                alpha=0.84,
                linewidth=0.0,
            )
            ax_area.set_ylim(0.0, 1.0)
            ax_area.set_xlim(float(time[0]), float(time[-1]))
            ax_area.set_ylabel("Rel.", fontsize=7)
            ax_area.set_xlabel("Time (s)", fontsize=7)
            ax_area.tick_params(axis="both", labelsize=6, pad=1)
            ax_area.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax_area.grid(axis="y", linewidth=0.3, alpha=0.25)

    region_handles = [
        Line2D(
            [0],
            [0],
            color=settings.region_colors.get(str(region), "#777777"),
            linewidth=5,
            label=_region_display(str(region)),
        )
        for region in source_regions
    ]
    fig.legend(
        handles=region_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        ncol=len(region_handles),
        fontsize=8,
    )
    fig.suptitle(
        "mRNN Signal Evolution and Relative Recurrent Current Influence",
        y=0.985,
        fontsize=13,
    )
    return fig, axes


def plot_fixation_mrnn_average_current_influence_pies(
    replay: Mapping[str, object],
    *,
    current_vectors: Mapping[tuple[str, str], torch.Tensor] | None = None,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot time-averaged relative source-region current influence as pie charts."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    regions = tuple(region_order or checkpoint_region_order(replay))
    source_regions = regions
    conditions = tuple(condition_order or replay["condition_names"])
    condition_names = tuple(replay["condition_names"])
    condition_indices = [condition_names.index(condition) for condition in conditions]
    if current_vectors is None:
        _, current_vectors = compute_fixation_mrnn_currents(replay)

    n_rows = len(regions)
    n_cols = len(conditions)
    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=settings.pie_figsize,
        dpi=int(settings.dpi),
        squeeze=False,
        subplot_kw={"aspect": "equal"},
    )
    axes: dict[tuple[str, str], plt.Axes] = {}
    colors = [
        settings.region_colors.get(str(source_region), "#777777")
        for source_region in source_regions
    ]
    for row_idx, target_region in enumerate(regions):
        for col_idx, condition in enumerate(conditions):
            ax = axes_arr[row_idx, col_idx]
            axes[(str(target_region), str(condition))] = ax
            cond_idx = condition_indices[col_idx]
            _, relative = _current_relative_influence(
                current_vectors,
                target_region=str(target_region),
                source_regions=source_regions,
                condition_idx=cond_idx,
            )
            mean_relative = np.nanmean(relative, axis=1)
            if not np.isfinite(mean_relative).any() or float(np.nansum(mean_relative)) <= 0.0:
                mean_relative = np.ones((len(source_regions),), dtype=float) / len(source_regions)
            ax.pie(
                mean_relative,
                colors=colors,
                autopct="%1.0f%%",
                pctdistance=0.68,
                startangle=90,
                counterclock=False,
                textprops={"fontsize": 6, "color": "white", "fontweight": "bold"},
                wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
            )
            if row_idx == 0:
                ax.set_title(_condition_display(str(condition), settings), fontsize=9)
            if col_idx == 0:
                ax.text(
                    -0.22,
                    0.5,
                    _region_display(str(target_region)),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

    region_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=settings.region_colors.get(str(region), "#777777"),
            markeredgecolor="white",
            markersize=8,
            label=_region_display(str(region)),
        )
        for region in source_regions
    ]
    fig.legend(
        handles=region_handles,
        loc="upper center",
        ncol=len(region_handles),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Time-Averaged Relative Recurrent Current Influence",
        y=0.985,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig, axes


def _as_flow_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(float, copy=False)
    return np.asarray(value, dtype=float)


def _flow_field_arrays(flow_field: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = _as_flow_numpy(flow_field.grid)
    x_vels = _as_flow_numpy(flow_field.x_vels)
    y_vels = _as_flow_numpy(flow_field.y_vels)
    speeds = _as_flow_numpy(flow_field.speeds)
    if grid.ndim != 3 or grid.shape[-1] < 2:
        raise ValueError("FlowField.grid must have shape rows x columns x 2.")
    return grid[..., 0], grid[..., 1], x_vels, y_vels, speeds


def plot_fixation_mrnn_flow_fields_at_time(
    replay: Mapping[str, object],
    *,
    time_s: float = 0.0,
    num_points: int = 9,
    x_offset: float = 1.0,
    y_offset: float = 1.0,
    cancel_other_regions: bool = False,
    settings: FixationMRNNDiagnosticPlotSettings | None = None,
    region_order: Sequence[str] | None = None,
    condition_order: Sequence[str] | None = None,
):
    """Plot 2D nonlinear flow fields near one replay time for each region/condition."""
    settings = settings or FixationMRNNDiagnosticPlotSettings()
    regions = tuple(region_order or checkpoint_region_order(replay))
    conditions = tuple(condition_order or replay["condition_names"])
    timeline = _timeline(replay, int(replay["inp"].shape[1]))
    time_idx = int(np.argmin(np.abs(timeline - float(time_s))))
    selected_time = float(timeline[time_idx])

    n_rows = len(regions)
    n_cols = len(conditions)
    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=settings.flow_figsize,
        dpi=int(settings.dpi),
        squeeze=False,
    )
    axes: dict[tuple[str, str], plt.Axes] = {}
    for row_idx, region in enumerate(regions):
        for col_idx, condition in enumerate(conditions):
            ax = axes_arr[row_idx, col_idx]
            axes[(str(region), str(condition))] = ax
            flow = compute_fixation_mrnn_flow_fields(
                replay,
                region=str(region),
                condition=str(condition),
                time_indices=(time_idx,),
                num_points=int(num_points),
                x_offset=float(x_offset),
                y_offset=float(y_offset),
                cancel_other_regions=bool(cancel_other_regions),
            )
            field = flow["flow_fields"][0]
            grid_x, grid_y, x_vels, y_vels, speeds = _flow_field_arrays(field)
            ax.quiver(
                grid_x,
                grid_y,
                x_vels * float(settings.flow_arrow_scale),
                y_vels * float(settings.flow_arrow_scale),
                speeds,
                cmap="viridis",
                angles="xy",
                scale_units="xy",
                pivot="mid",
                width=0.0045,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis="both", labelsize=6, pad=1)
            ax.grid(linewidth=0.3, alpha=0.25)
            if row_idx == 0:
                ax.set_title(_condition_display(str(condition), settings), fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{_region_display(str(region))}\nFlow PC2", fontsize=8)
            else:
                ax.set_ylabel("Flow PC2", fontsize=7)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Flow PC1", fontsize=8)
    fig.suptitle(
        f"mRNN Flow Fields Near Fixation Onset (t = {selected_time:.3f}s)",
        y=0.985,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig, axes


__all__ = [
    "DEFAULT_MRNN_CONDITION_COLORS",
    "DEFAULT_MRNN_CONDITION_LABELS",
    "DEFAULT_MRNN_REGION_COLORS",
    "FixationMRNNDiagnosticPlotSettings",
    "plot_fixation_mrnn_activation_trajectories_3d",
    "plot_fixation_mrnn_activation_pc_timeseries",
    "plot_fixation_mrnn_average_current_influence_pies",
    "plot_fixation_mrnn_current_influence",
    "plot_fixation_mrnn_flow_fields_at_time",
]
