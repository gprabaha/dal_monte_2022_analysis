"""Plotting helpers for fixation population PCA outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import apply_plotting_config
from dal_monte_2022_analysis.runtime.io.plot_output import (
    normalize_extension,
    save_figure,
)
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_CONDITION_ORDER: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
DEFAULT_CONDITION_LABELS: dict[str, str] = {
    "face_interactive": "Interactive Face",
    "face_non_interactive": "Non-Interactive Face",
    "object": "Object",
}
DEFAULT_CONDITION_COLORS: dict[str, str] = {
    "face_interactive": "#d62728",
    "face_non_interactive": "#1f77b4",
    "object": "#2ca02c",
}


@dataclass
class FixationPopulationPCAPlotSettings:
    """Configuration for fixation population PCA plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_population_pca"
    input_filename: str = "results.pkl"
    output_subdir: str = "ephys/psth/fixation_population_pca/plots"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 300
    conditions: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_CONDITION_ORDER),
    )
    condition_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONDITION_LABELS),
    )
    condition_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONDITION_COLORS),
    )
    trajectory_n_pcs: int = 3
    trajectory_n_columns: int = 4
    trajectory_letter_width_in: float = 8.5
    trajectory_letter_height_frac: float = 0.2
    variance_letter_height_frac: float = 0.4
    max_components_display: int = 20


def _resolve_output_ext(settings: FixationPopulationPCAPlotSettings) -> str:
    return normalize_extension(settings.output_extension, fallback="pdf")


def _load_population_pca_result(
    settings: FixationPopulationPCAPlotSettings,
) -> tuple[dict, Path]:
    cfg = load_config(settings.cfg_path)
    in_root = build_analysis_output_dir(cfg, settings.input_subdir)
    in_path = in_root / ensure_filename(settings.input_filename, ".pkl")
    if not in_path.exists():
        raise FileNotFoundError(f"Population PCA results not found: {in_path}")
    obj = load_pickle_path(in_path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict payload in PCA results: {in_path}")
    return obj, in_path


def _resolve_region_order(
    result_obj: dict,
    *,
    regions: Optional[Sequence[str]],
) -> list[str]:
    region_payloads = result_obj.get("regions", {}) if isinstance(result_obj, dict) else {}
    if not isinstance(region_payloads, dict):
        region_payloads = {}
    available = sorted((str(key) for key in region_payloads.keys()), key=lambda token: token.lower())
    if regions is None:
        return available
    wanted = [str(region) for region in regions]
    return [region for region in wanted if region in set(available)]


def _resolve_condition_colors(settings: FixationPopulationPCAPlotSettings) -> dict[str, str]:
    out = dict(DEFAULT_CONDITION_COLORS)
    for cond, color in settings.condition_colors.items():
        key = str(cond).strip()
        if key:
            out[key] = str(color).strip()
    return out


def _coerce_scores_pc_by_time(
    raw_scores: object,
    *,
    n_time_bins: int,
) -> np.ndarray:
    arr = np.asarray(raw_scores, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.asarray([], dtype=float)
    # Preferred orientation is PCs x time.
    if arr.shape[1] == int(n_time_bins):
        return arr
    # Backward-compat orientation (time x PCs).
    if arr.shape[0] == int(n_time_bins):
        return arr.T
    return np.asarray([], dtype=float)


def _compose_white_to_color_gradient(
    color_hex: str,
    n_segments: int,
) -> np.ndarray:
    if int(n_segments) <= 0:
        return np.asarray([], dtype=float).reshape(0, 3)
    base = np.asarray(to_rgb(color_hex), dtype=float).reshape(1, 3)
    ramp = np.linspace(0.0, 1.0, int(n_segments), dtype=float).reshape(-1, 1)
    white = np.ones((int(n_segments), 3), dtype=float)
    return (1.0 - ramp) * white + ramp * np.tile(base, (int(n_segments), 1))


def _line_segments_3d(points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        return np.asarray([], dtype=float).reshape(0, 2, 3)
    return np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)


def _nearest_marker_indices(bin_centers_s: np.ndarray) -> list[int]:
    centers = np.asarray(bin_centers_s, dtype=float).reshape(-1)
    if centers.size == 0:
        return []
    targets = (-0.5, 0.0, 0.5)
    return [int(np.argmin(np.abs(centers - target))) for target in targets]


def _apply_axis_limits_3d(ax, all_points: list[np.ndarray]) -> None:
    if not all_points:
        return
    stack = np.vstack(all_points)
    mins = np.nanmin(stack, axis=0)
    maxs = np.nanmax(stack, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pads = 0.08 * spans
    ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
    ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
    ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])


def _apply_plotting_style(plotting_cfg_path: str) -> None:
    if plotting_cfg_path and Path(plotting_cfg_path).exists():
        cfg = load_config(plotting_cfg_path)
        apply_plotting_config(cfg)


def plot_fixation_population_pca_trajectories(
    settings: FixationPopulationPCAPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    output_filename: str = "population_pca_pc_trajectories",
) -> Optional[dict]:
    """Plot 3D trajectories (PC1-3) by region for fixation-condition projections."""
    _apply_plotting_style(settings.plotting_cfg_path)
    result_obj, input_path = _load_population_pca_result(settings)
    region_payloads = result_obj.get("regions", {})
    if not isinstance(region_payloads, dict) or not region_payloads:
        print("[plot] no region payloads available in fixation population PCA results")
        return None

    region_order = _resolve_region_order(result_obj, regions=regions)
    if not region_order:
        print("[plot] no regions available after filtering for PCA trajectory plotting")
        return None

    n_cols = max(1, int(settings.trajectory_n_columns))
    n_rows = int(np.ceil(len(region_order) / float(n_cols)))
    fig_w = float(settings.trajectory_letter_width_in)
    fig_h = float(8.5 * settings.trajectory_letter_height_frac * max(1, n_rows))
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=settings.output_dpi)

    color_map = _resolve_condition_colors(settings)
    cond_order = [cond for cond in settings.conditions if cond in color_map]
    marker_indices_cache: dict[str, list[int]] = {}

    for idx, region in enumerate(region_order):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        payload = region_payloads.get(region, {})
        scores_map = payload.get("concatenated_condition_scores_pc_by_time")
        if not isinstance(scores_map, dict):
            scores_map = payload.get("concatenated_condition_scores", {})
        if not isinstance(scores_map, dict):
            scores_map = {}
        bin_centers_s = np.asarray(payload.get("bin_centers_s_window", np.asarray([], dtype=float)), dtype=float).reshape(-1)
        marker_indices = _nearest_marker_indices(bin_centers_s)
        marker_indices_cache[str(region)] = marker_indices

        all_xyz: list[np.ndarray] = []
        for condition in cond_order:
            raw_scores = scores_map.get(condition, np.asarray([], dtype=float))
            scores = _coerce_scores_pc_by_time(raw_scores, n_time_bins=bin_centers_s.size)
            if scores.size == 0 or scores.shape[0] < int(settings.trajectory_n_pcs):
                continue
            xyz = np.asarray(scores[:3, :], dtype=float).T
            if xyz.shape[0] < 2:
                continue
            all_xyz.append(xyz)
            segments = _line_segments_3d(xyz)
            if segments.size == 0:
                continue

            border_color = str(color_map.get(condition, DEFAULT_CONDITION_COLORS[condition]))
            border = Line3DCollection(
                segments,
                colors=[border_color],
                linewidths=1.6,
                alpha=1.0,
            )
            gradient = _compose_white_to_color_gradient(border_color, segments.shape[0])
            main = Line3DCollection(
                segments,
                colors=gradient,
                linewidths=1.0,
                alpha=1.0,
            )
            ax.add_collection3d(border)
            ax.add_collection3d(main)

            for marker_idx in marker_indices:
                marker_idx = max(0, min(int(marker_idx), xyz.shape[0] - 1))
                ax.scatter(
                    [float(xyz[marker_idx, 0])],
                    [float(xyz[marker_idx, 1])],
                    [float(xyz[marker_idx, 2])],
                    s=9.0,
                    c=[border_color],
                    edgecolors="black",
                    linewidths=0.25,
                    alpha=0.95,
                )

        _apply_axis_limits_3d(ax, all_xyz)
        ax.view_init(elev=22, azim=-58)
        ax.grid(True, linewidth=0.35, alpha=0.28)
        ax.set_title(str(region), fontsize=8, pad=2.0)
        ax.set_xlabel("PC1", labelpad=-4, fontsize=7)
        ax.set_ylabel("PC2", labelpad=-4, fontsize=7)
        ax.set_zlabel("PC3", labelpad=-3, fontsize=7)
        ax.tick_params(axis="both", which="major", labelsize=6, pad=0.5)
        ax.tick_params(axis="z", which="major", labelsize=6, pad=0.5)

    n_axes_total = n_rows * n_cols
    for idx in range(len(region_order), n_axes_total):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        ax.set_axis_off()

    handles = [
        Line2D([0], [0], color=str(color_map[cond]), lw=1.8, label=settings.condition_labels.get(cond, cond))
        for cond in cond_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(handles)),
        fontsize=7,
        frameon=False,
    )
    fig.subplots_adjust(left=0.02, right=0.995, top=0.80, bottom=0.07, wspace=0.02, hspace=0.18)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    ext = _resolve_output_ext(settings)
    out_name = ensure_filename(output_filename, f".{ext}")
    out_path = out_root / out_name
    save_figure(
        fig,
        out_path,
        ext=ext,
        dpi=settings.output_dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)

    return {
        "output_path": str(out_path),
        "input_path": str(input_path),
        "regions": list(region_order),
        "conditions": list(cond_order),
        "marker_indices": marker_indices_cache,
    }

