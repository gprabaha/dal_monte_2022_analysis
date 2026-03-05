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
DEFAULT_REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")
DEFAULT_REGION_LABELS: dict[str, str] = {
    "bla": "BLA",
    "accg": "ACCg",
    "dmpfc": "dmPFC",
    "ofc": "OFC",
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
    available = [str(key) for key in region_payloads.keys()]
    return _ordered_region_tokens(available=available, requested=regions)


def _normalize_region_token(region: object) -> str:
    token = str(region).strip()
    return token.lower()


def _ordered_region_tokens(
    *,
    available: Sequence[str],
    requested: Optional[Sequence[str]],
) -> list[str]:
    available_map = {str(region).lower(): str(region) for region in available}
    if requested is not None:
        requested_tokens = [str(region).strip().lower() for region in requested]
        ordered = [available_map[token] for token in requested_tokens if token in available_map]
        return ordered

    ordered: list[str] = []
    for token in DEFAULT_REGION_ORDER:
        if token in available_map:
            ordered.append(available_map[token])
    for token in sorted(available_map.keys()):
        if token not in set(DEFAULT_REGION_ORDER):
            ordered.append(available_map[token])
    return ordered


def _region_display_label(region: object) -> str:
    token = _normalize_region_token(region)
    return DEFAULT_REGION_LABELS.get(token, str(region))


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


def _extract_cross_condition_explained_variance_df(result_obj: dict) -> pd.DataFrame:
    raw = result_obj.get("cross_condition_explained_variance")
    if isinstance(raw, pd.DataFrame):
        out = raw.copy()
    elif isinstance(raw, dict):
        out = pd.DataFrame(raw)
    else:
        out = pd.DataFrame()
    if out.empty:
        return pd.DataFrame()
    required = {
        "region",
        "fit_condition",
        "eval_condition",
        "n_components",
        "explained_variance_fraction",
    }
    if not required.issubset(out.columns):
        return pd.DataFrame()
    out = out.copy()
    out["region"] = out["region"].astype(str)
    out["fit_condition"] = out["fit_condition"].astype(str)
    out["eval_condition"] = out["eval_condition"].astype(str)
    out["n_components"] = pd.to_numeric(out["n_components"], errors="coerce")
    out["explained_variance_fraction"] = pd.to_numeric(
        out["explained_variance_fraction"],
        errors="coerce",
    )
    if "explained_variance_per_pc_fraction" in out.columns:
        out["explained_variance_per_pc_fraction"] = pd.to_numeric(
            out["explained_variance_per_pc_fraction"],
            errors="coerce",
        )
    if "explained_variance_cumulative_fraction" in out.columns:
        out["explained_variance_cumulative_fraction"] = pd.to_numeric(
            out["explained_variance_cumulative_fraction"],
            errors="coerce",
        )
    out = out.loc[out["n_components"].notna()].copy()
    out["n_components"] = out["n_components"].astype(int)

    group_cols = ["region", "fit_condition", "eval_condition"]
    out = out.sort_values(group_cols + ["n_components"]).reset_index(drop=True)

    if "explained_variance_per_pc_fraction" in out.columns:
        per_pc = np.asarray(out["explained_variance_per_pc_fraction"], dtype=float)
    else:
        raw = np.asarray(out["explained_variance_fraction"], dtype=float)
        # Backward-compat: older files stored cumulative values in explained_variance_fraction.
        # Detect monotonic curves and convert cumulative -> per-PC via finite differences.
        is_likely_cumulative = True
        grouped = out.groupby(group_cols, dropna=False, sort=False)
        for _, grp in grouped:
            values = np.asarray(grp["explained_variance_fraction"], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size <= 1:
                continue
            diffs = np.diff(finite)
            if np.any(diffs < -1e-8):
                is_likely_cumulative = False
                break
        if is_likely_cumulative:
            per_pc = np.full(raw.shape, np.nan, dtype=float)
            for _, idx in grouped.indices.items():
                values = np.asarray(out.loc[idx, "explained_variance_fraction"], dtype=float)
                diffs = np.full(values.shape, np.nan, dtype=float)
                prev = 0.0
                for i, value in enumerate(values):
                    if np.isfinite(value):
                        diffs[i] = max(0.0, float(value) - float(prev))
                        prev = float(value)
                per_pc[np.asarray(idx, dtype=int)] = diffs
        else:
            per_pc = raw
    out["explained_variance_per_pc_fraction"] = per_pc
    out["explained_variance_fraction"] = per_pc

    if "explained_variance_cumulative_fraction" not in out.columns:
        out["explained_variance_cumulative_fraction"] = np.nan
    for _, idx in out.groupby(group_cols, dropna=False, sort=False).indices.items():
        values = np.asarray(out.loc[idx, "explained_variance_per_pc_fraction"], dtype=float)
        cumulative = np.full(values.shape, np.nan, dtype=float)
        running = 0.0
        for i, value in enumerate(values):
            if np.isfinite(value):
                running += float(value)
                cumulative[i] = running
        out.loc[idx, "explained_variance_cumulative_fraction"] = cumulative
    return out


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
        ax.set_title(_region_display_label(region), fontsize=8, pad=2.0)
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


def plot_fixation_population_pca_explained_variance_bars(
    settings: FixationPopulationPCAPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    output_filename: str = "population_pca_explained_variance_bars",
) -> Optional[dict]:
    """Plot per-PC explained variance bars across eval fixation types."""
    _apply_plotting_style(settings.plotting_cfg_path)
    result_obj, input_path = _load_population_pca_result(settings)
    explained_df = _extract_cross_condition_explained_variance_df(result_obj)
    if explained_df.empty:
        print("[plot] no cross-condition explained variance rows found")
        return None

    cond_order = [cond for cond in settings.conditions if cond in set(explained_df["fit_condition"].unique())]
    if not cond_order:
        print("[plot] no matching fit conditions for explained variance plotting")
        return None

    region_order = _ordered_region_tokens(
        available=explained_df["region"].astype(str).unique().tolist(),
        requested=regions,
    )
    if not region_order:
        print("[plot] no matching regions for explained variance plotting")
        return None

    max_comp = max(1, int(settings.max_components_display))
    n_rows = len(cond_order)
    n_cols = max(4, len(region_order))
    fig = plt.figure(
        figsize=(8.5, 8.5 * float(settings.variance_letter_height_frac)),
        dpi=settings.output_dpi,
    )
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    color_map = _resolve_condition_colors(settings)
    eval_order = [cond for cond in settings.conditions if cond in set(explained_df["eval_condition"].unique())]
    bar_width = 0.84 / max(1, len(eval_order))
    x = np.arange(1, max_comp + 1, dtype=float)

    for row_idx, fit_condition in enumerate(cond_order):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(region_order):
                ax.set_axis_off()
                continue
            region = region_order[col_idx]
            sub = explained_df.loc[
                (explained_df["region"] == str(region))
                & (explained_df["fit_condition"] == str(fit_condition))
                & (explained_df["n_components"] >= 1)
                & (explained_df["n_components"] <= max_comp)
            ].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            for eval_idx, eval_condition in enumerate(eval_order):
                eval_sub = sub.loc[sub["eval_condition"] == str(eval_condition)].copy()
                if eval_sub.empty:
                    continue
                eval_sub = eval_sub.sort_values("n_components")
                y = np.full((max_comp,), np.nan, dtype=float)
                comp_idx = np.asarray(eval_sub["n_components"], dtype=int) - 1
                valid = (comp_idx >= 0) & (comp_idx < max_comp)
                if np.any(valid):
                    y[comp_idx[valid]] = np.asarray(
                        eval_sub.loc[valid, "explained_variance_fraction"],
                        dtype=float,
                    )
                offset = (float(eval_idx) - (len(eval_order) - 1.0) / 2.0) * bar_width
                finite = np.isfinite(y)
                if np.any(finite):
                    ax.bar(
                        x[finite] + offset,
                        y[finite],
                        width=bar_width * 0.95,
                        color=str(color_map.get(eval_condition, DEFAULT_CONDITION_COLORS.get(eval_condition, "#444444"))),
                        edgecolor="black",
                        linewidth=0.15,
                        alpha=0.95,
                    )

            ax.axhline(0.0, color="#222222", linewidth=0.35, alpha=0.7)
            ax.set_xlim(0.3, max_comp + 0.7)
            ax.set_xticks([1, 5, 10, 15, 20] if max_comp >= 20 else list(range(1, max_comp + 1)))
            ax.tick_params(axis="both", which="major", labelsize=5, pad=1.0)
            ax.grid(axis="y", linewidth=0.25, alpha=0.3)
            if row_idx == 0:
                ax.set_title(_region_display_label(region), fontsize=6, pad=1.5)
            if row_idx == n_rows - 1:
                ax.set_xlabel("PC", fontsize=6, labelpad=0.8)
            if col_idx == 0:
                fit_label = settings.condition_labels.get(fit_condition, fit_condition)
                ax.set_ylabel(f"{fit_label}\nExplained", fontsize=6, labelpad=1.0)

    handles = [
        Line2D(
            [0],
            [0],
            color=str(color_map.get(cond, DEFAULT_CONDITION_COLORS.get(cond, "#444444"))),
            lw=2.0,
            label=settings.condition_labels.get(cond, cond),
        )
        for cond in eval_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(handles)),
        fontsize=6,
        frameon=False,
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.12, wspace=0.18, hspace=0.34)

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
        "fit_conditions": list(cond_order),
        "eval_conditions": list(eval_order),
        "max_components": int(max_comp),
    }


def plot_fixation_population_pca_explained_variance_cumulative(
    settings: FixationPopulationPCAPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    output_filename: str = "population_pca_explained_variance_cumulative",
) -> Optional[dict]:
    """Plot cumulative explained-variance curves across PCs."""
    _apply_plotting_style(settings.plotting_cfg_path)
    result_obj, input_path = _load_population_pca_result(settings)
    explained_df = _extract_cross_condition_explained_variance_df(result_obj)
    if explained_df.empty:
        print("[plot] no cross-condition explained variance rows found")
        return None

    cond_order = [cond for cond in settings.conditions if cond in set(explained_df["fit_condition"].unique())]
    if not cond_order:
        print("[plot] no matching fit conditions for cumulative explained variance plotting")
        return None

    region_order = _ordered_region_tokens(
        available=explained_df["region"].astype(str).unique().tolist(),
        requested=regions,
    )
    if not region_order:
        print("[plot] no matching regions for cumulative explained variance plotting")
        return None

    max_comp = max(1, int(settings.max_components_display))
    n_rows = len(cond_order)
    n_cols = max(4, len(region_order))
    fig = plt.figure(
        figsize=(8.5, 8.5 * float(settings.variance_letter_height_frac)),
        dpi=settings.output_dpi,
    )
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    color_map = _resolve_condition_colors(settings)
    eval_order = [cond for cond in settings.conditions if cond in set(explained_df["eval_condition"].unique())]
    x = np.arange(1, max_comp + 1, dtype=float)

    for row_idx, fit_condition in enumerate(cond_order):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(region_order):
                ax.set_axis_off()
                continue
            region = region_order[col_idx]
            sub = explained_df.loc[
                (explained_df["region"] == str(region))
                & (explained_df["fit_condition"] == str(fit_condition))
                & (explained_df["n_components"] >= 1)
                & (explained_df["n_components"] <= max_comp)
            ].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            for eval_condition in eval_order:
                eval_sub = sub.loc[sub["eval_condition"] == str(eval_condition)].copy()
                if eval_sub.empty:
                    continue
                eval_sub = eval_sub.sort_values("n_components")
                per_pc = np.full((max_comp,), np.nan, dtype=float)
                comp_idx = np.asarray(eval_sub["n_components"], dtype=int) - 1
                valid = (comp_idx >= 0) & (comp_idx < max_comp)
                if np.any(valid):
                    per_pc[comp_idx[valid]] = np.asarray(
                        eval_sub.loc[valid, "explained_variance_fraction"],
                        dtype=float,
                    )
                cumulative = np.full((max_comp,), np.nan, dtype=float)
                running = 0.0
                for i, value in enumerate(per_pc):
                    if np.isfinite(value):
                        running += float(value)
                        cumulative[i] = running
                finite = np.isfinite(cumulative)
                if np.any(finite):
                    ax.plot(
                        x[finite],
                        cumulative[finite],
                        color=str(color_map.get(eval_condition, DEFAULT_CONDITION_COLORS.get(eval_condition, "#444444"))),
                        linewidth=1.0,
                        marker="o",
                        markersize=1.6,
                        markeredgewidth=0.0,
                        alpha=0.95,
                    )

            ax.axhline(0.0, color="#222222", linewidth=0.35, alpha=0.7)
            ax.set_xlim(0.8, max_comp + 0.2)
            ax.set_xticks([1, 5, 10, 15, 20] if max_comp >= 20 else list(range(1, max_comp + 1)))
            ax.tick_params(axis="both", which="major", labelsize=5, pad=1.0)
            ax.grid(axis="y", linewidth=0.25, alpha=0.3)
            if row_idx == 0:
                ax.set_title(_region_display_label(region), fontsize=6, pad=1.5)
            if row_idx == n_rows - 1:
                ax.set_xlabel("PC", fontsize=6, labelpad=0.8)
            if col_idx == 0:
                fit_label = settings.condition_labels.get(fit_condition, fit_condition)
                ax.set_ylabel(f"{fit_label}\nCumulative", fontsize=6, labelpad=1.0)

    handles = [
        Line2D(
            [0],
            [0],
            color=str(color_map.get(cond, DEFAULT_CONDITION_COLORS.get(cond, "#444444"))),
            lw=1.8,
            marker="o",
            markersize=2.4,
            markeredgewidth=0.0,
            label=settings.condition_labels.get(cond, cond),
        )
        for cond in eval_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(handles)),
        fontsize=6,
        frameon=False,
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.12, wspace=0.18, hspace=0.34)

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
        "fit_conditions": list(cond_order),
        "eval_conditions": list(eval_order),
        "max_components": int(max_comp),
    }
