"""Plot three-way fixation-condition population activity in triangular coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.runtime.io.plot_output import (
    normalize_extension,
    save_figure,
)
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


TRIANGLE_HEIGHT = float(np.sqrt(3.0) / 2.0)
REQUIRED_MEAN_COLUMNS = (
    "mean_fr_face_interactive_hz",
    "mean_fr_face_non_interactive_hz",
    "mean_fr_object_hz",
)
RELATIVE_COLUMNS = (
    "relative_face_interactive",
    "relative_face_non_interactive",
    "relative_object",
)
DEFAULT_COLORS = {
    "face_interactive": "#d62728",
    "face_non_interactive": "#1f77b4",
    "object": "#2ca02c",
}


@dataclass
class FixationThreeWayTriangularPlotSettings:
    """Configuration for triangular three-way selectivity population plots."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    condition_summary_filename: str = "condition_window_means.csv"
    output_subdir: str = "ephys/psth/fixation_psth_selectivity_triangular"
    output_filename: str = "population_triangular"
    output_extension: str = "png"
    output_dpi: Optional[int] = 220
    min_units_per_panel: int = 1
    point_size: float = 18.0
    point_alpha: float = 0.68
    marker_edge_width: float = 0.28
    draw_centroid: bool = True
    condition_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COLORS))


def _resolve_figsize_and_dpi(
    settings: FixationThreeWayTriangularPlotSettings,
    *,
    n_rows: int,
    n_cols: int,
) -> tuple[list[float], Optional[int]]:
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        figsize, cfg_dpi = resolve_figsize(plot_cfg)
    else:
        figsize, cfg_dpi = (None, None)

    if figsize is None:
        figsize = [max(3.0 * float(n_cols), 6.5), max(2.8 * float(n_rows), 5.4)]
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    return [float(figsize[0]), float(figsize[1])], dpi


def _load_condition_summary_df(settings: FixationThreeWayTriangularPlotSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.condition_summary_filename, ".csv")
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Three-way condition summary CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        return df

    missing_mean_cols = [col for col in REQUIRED_MEAN_COLUMNS if col not in df.columns]
    if missing_mean_cols:
        raise ValueError(f"condition summary is missing required columns: {missing_mean_cols}")

    if "region" not in df.columns:
        df["region"] = "unknown"
    else:
        df["region"] = df["region"].fillna("unknown").astype(str).replace({"": "unknown"})
    if "window_name" not in df.columns:
        raise ValueError("condition summary is missing required column 'window_name'.")

    return df


def _compute_relative_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    means = out.loc[:, list(REQUIRED_MEAN_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    arr = means.to_numpy(dtype=float)
    valid = np.all(np.isfinite(arr), axis=1)
    totals = np.sum(arr, axis=1)
    valid = valid & np.isfinite(totals) & (totals > 0.0)

    rel = np.full_like(arr, np.nan, dtype=float)
    rel[valid] = arr[valid] / totals[valid, None]
    out["relative_face_interactive"] = rel[:, 0]
    out["relative_face_non_interactive"] = rel[:, 1]
    out["relative_object"] = rel[:, 2]
    return out


def _window_order(df: pd.DataFrame, requested_windows: Optional[Sequence[str]]) -> list[str]:
    if requested_windows is not None:
        allowed = {str(w) for w in requested_windows}
        return [str(w) for w in requested_windows if str(w) in set(df["window_name"].astype(str)) and str(w) in allowed]

    if "window_start_ms" not in df.columns:
        return sorted(df["window_name"].astype(str).unique().tolist())

    meta = (
        df.loc[:, ["window_name", "window_start_ms", "window_stop_ms"]]
        .dropna(subset=["window_name"])
        .copy()
    )
    if meta.empty:
        return sorted(df["window_name"].astype(str).unique().tolist())
    meta["window_name"] = meta["window_name"].astype(str)
    meta["window_start_ms"] = pd.to_numeric(meta["window_start_ms"], errors="coerce")
    meta["window_stop_ms"] = pd.to_numeric(meta["window_stop_ms"], errors="coerce")
    grouped = (
        meta.groupby("window_name", as_index=False)
        .agg(
            window_start_ms=("window_start_ms", "median"),
            window_stop_ms=("window_stop_ms", "median"),
        )
        .sort_values(["window_start_ms", "window_stop_ms", "window_name"], na_position="last")
    )
    return grouped["window_name"].astype(str).tolist()


def _window_label_map(df: pd.DataFrame) -> dict[str, str]:
    if "window_start_ms" not in df.columns or "window_stop_ms" not in df.columns:
        return {str(name): str(name) for name in sorted(df["window_name"].astype(str).unique())}

    meta = (
        df.loc[:, ["window_name", "window_start_ms", "window_stop_ms"]]
        .dropna(subset=["window_name"])
        .copy()
    )
    meta["window_name"] = meta["window_name"].astype(str)
    meta["window_start_ms"] = pd.to_numeric(meta["window_start_ms"], errors="coerce")
    meta["window_stop_ms"] = pd.to_numeric(meta["window_stop_ms"], errors="coerce")
    grouped = (
        meta.groupby("window_name", as_index=False)
        .agg(
            window_start_ms=("window_start_ms", "median"),
            window_stop_ms=("window_stop_ms", "median"),
        )
    )
    labels: dict[str, str] = {}
    for row in grouped.itertuples(index=False):
        if np.isfinite(row.window_start_ms) and np.isfinite(row.window_stop_ms):
            labels[str(row.window_name)] = (
                f"{row.window_name}\n[{float(row.window_start_ms):.0f}, {float(row.window_stop_ms):.0f}] ms"
            )
        else:
            labels[str(row.window_name)] = str(row.window_name)
    return labels


def _to_triangle_xy(rel_int: np.ndarray, rel_obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = rel_obj + 0.5 * rel_int
    y = TRIANGLE_HEIGHT * rel_int
    return x, y


def _draw_triangle_frame(ax) -> None:
    ax.plot([0.0, 1.0], [0.0, 0.0], color="#2e2e2e", linewidth=1.1)
    ax.plot([0.0, 0.5], [0.0, TRIANGLE_HEIGHT], color="#2e2e2e", linewidth=1.1)
    ax.plot([1.0, 0.5], [0.0, TRIANGLE_HEIGHT], color="#2e2e2e", linewidth=1.1)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.09, TRIANGLE_HEIGHT + 0.12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _dominant_colors(df: pd.DataFrame, color_map: dict[str, str]) -> list[str]:
    if "dominant_condition" in df.columns:
        out = [color_map.get(str(token), "#5e5e5e") for token in df["dominant_condition"].astype(str).tolist()]
        return out

    rel = df.loc[:, list(RELATIVE_COLUMNS)].to_numpy(dtype=float)
    idx = np.nanargmax(rel, axis=1)
    labels = np.asarray(["face_interactive", "face_non_interactive", "object"])[idx]
    return [color_map.get(str(token), "#5e5e5e") for token in labels.tolist()]


def _output_path(settings: FixationThreeWayTriangularPlotSettings, ext: str, root: Path) -> Path:
    stem = Path(str(settings.output_filename).strip()).stem
    if not stem:
        stem = "population_triangular"
    return root / f"{stem}.{ext}"


def plot_fixation_three_way_selectivity_triangular(
    settings: FixationThreeWayTriangularPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    windows: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Render one triangular population figure with regions-by-windows subplots."""
    df = _load_condition_summary_df(settings)
    if df.empty:
        print("[plot] no three-way condition summary rows found")
        return None

    if not set(RELATIVE_COLUMNS).issubset(df.columns):
        df = _compute_relative_columns(df)

    required_rel = list(RELATIVE_COLUMNS)
    rel_df = df.loc[:, required_rel].apply(pd.to_numeric, errors="coerce")
    valid = np.all(np.isfinite(rel_df.to_numpy(dtype=float)), axis=1)
    df = df.loc[valid].copy()
    if df.empty:
        print("[plot] no rows with valid relative three-way firing components")
        return None

    if regions is not None:
        allowed_regions = {str(region) for region in regions}
        df = df.loc[df["region"].astype(str).isin(allowed_regions)].copy()
    if windows is not None:
        allowed_windows = {str(window) for window in windows}
        df = df.loc[df["window_name"].astype(str).isin(allowed_windows)].copy()
    if df.empty:
        print("[plot] no rows remain after region/window filtering")
        return None

    region_order = sorted(df["region"].astype(str).unique().tolist())
    window_order = _window_order(df, windows)
    if not region_order or not window_order:
        print("[plot] no region/window panels available to render")
        return None

    n_rows = len(window_order)
    n_cols = len(region_order)
    figsize, dpi = _resolve_figsize_and_dpi(settings, n_rows=n_rows, n_cols=n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)

    win_labels = _window_label_map(df)
    panel_counts: list[dict] = []
    for row_idx, window_name in enumerate(window_order):
        for col_idx, region in enumerate(region_order):
            ax = axes[row_idx][col_idx]
            _draw_triangle_frame(ax)

            panel = df.loc[
                (df["window_name"].astype(str) == str(window_name))
                & (df["region"].astype(str) == str(region))
            ].copy()
            n_units = int(panel["unit_key"].astype(str).nunique()) if "unit_key" in panel.columns else int(len(panel))
            panel_counts.append(
                {
                    "region": str(region),
                    "window_name": str(window_name),
                    "n_units": int(n_units),
                }
            )
            if n_units >= int(settings.min_units_per_panel):
                rel = panel.loc[:, list(RELATIVE_COLUMNS)].to_numpy(dtype=float)
                x, y = _to_triangle_xy(rel_int=rel[:, 0], rel_obj=rel[:, 2])
                colors = _dominant_colors(panel, settings.condition_colors)
                ax.scatter(
                    x,
                    y,
                    s=float(settings.point_size),
                    c=colors,
                    alpha=float(settings.point_alpha),
                    linewidths=float(settings.marker_edge_width),
                    edgecolors="#1b1b1b",
                )
                if settings.draw_centroid and rel.shape[0] > 0:
                    centroid = np.nanmean(rel, axis=0)
                    cx, cy = _to_triangle_xy(
                        rel_int=np.asarray([centroid[0]], dtype=float),
                        rel_obj=np.asarray([centroid[2]], dtype=float),
                    )
                    ax.scatter(
                        cx,
                        cy,
                        marker="X",
                        s=max(20.0, float(settings.point_size) * 1.8),
                        c="#111111",
                        alpha=0.95,
                        linewidths=0.45,
                        edgecolors="#ffffff",
                        zorder=6,
                    )

            ax.text(
                0.02,
                0.97,
                f"n={n_units}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
            )
            ax.text(
                0.5,
                1.02,
                "Int Face",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=settings.condition_colors.get("face_interactive", "#111111"),
            )
            ax.text(
                -0.03,
                -0.02,
                "Non-Int Face",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                color=settings.condition_colors.get("face_non_interactive", "#111111"),
            )
            ax.text(
                1.03,
                -0.02,
                "Object",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.6,
                color=settings.condition_colors.get("object", "#111111"),
            )

            if row_idx == 0:
                ax.set_title(str(region), fontsize=11.2, pad=10.0)
            if col_idx == 0:
                ax.set_ylabel(str(win_labels.get(str(window_name), str(window_name))), fontsize=9.0)

    fig.suptitle(
        "Three-way fixation-condition population activity",
        fontsize=13.2,
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.97], w_pad=0.8, h_pad=1.0)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    ext = normalize_extension(settings.output_extension, fallback="png")
    out_path = _output_path(settings, ext, out_root)

    save_figure(
        fig,
        out_path,
        ext=ext,
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)

    return {
        "output_path": out_path,
        "regions": region_order,
        "windows": window_order,
        "panel_counts": panel_counts,
    }

