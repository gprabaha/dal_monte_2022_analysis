"""Plot three-way fixation-condition population activity in triangular coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


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
@dataclass
class FixationThreeWayTriangularPlotSettings:
    """Configuration for triangular three-way selectivity population plots."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    condition_summary_filename: str = "condition_window_means.csv"
    unit_summary_filename: str = "unit_selectivity.csv"
    output_subdir: str = "ephys/psth/fixation_psth_selectivity_triangular"
    output_filename: str = "population_triangular"
    output_extension: str = "png"
    output_dpi: Optional[int] = 220
    min_units_per_panel: int = 1
    point_size: float = 26.0
    point_color: str = "#1f1f1f"
    point_alpha_significant: float = 1.0
    point_alpha_non_significant: float = 0.5
    marker_edge_width: float = 0.28
    draw_centroid: bool = True
    include_only_selective_units: bool = False
    region_order: Optional[Sequence[str]] = None
    default_windows: Optional[Sequence[str]] = None
    figure_width_in: Optional[float] = None
    figure_height_in: Optional[float] = None
    show_suptitle: bool = True
    left_margin: float = 0.035
    right_margin: float = 0.995
    top_margin: float = 0.95
    bottom_margin: float = 0.04
    panel_wspace: float = 0.12
    panel_hspace: float = 0.22


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

    if settings.figure_width_in is not None or settings.figure_height_in is not None:
        if figsize is None:
            figsize = [max(3.0 * float(n_cols), 7.5), max(4.2 * float(n_rows), 8.6)]
        width_in = float(settings.figure_width_in) if settings.figure_width_in is not None else float(figsize[0])
        height_in = float(settings.figure_height_in) if settings.figure_height_in is not None else float(figsize[1])
        figsize = [width_in, height_in]
    elif figsize is None:
        # Favor taller panels so equal-aspect triangles fill more of each subplot.
        figsize = [max(3.0 * float(n_cols), 7.5), max(4.2 * float(n_rows), 8.6)]
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    return [float(figsize[0]), float(figsize[1])], dpi


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return float(value) != 0.0
    if value is None:
        return False
    token = str(value).strip().lower()
    return token in {"1", "true", "t", "yes", "y"}


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


def _load_unit_summary_df(settings: FixationThreeWayTriangularPlotSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.unit_summary_filename, ".csv")
    )
    if not in_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(in_path)
    if df.empty:
        return df
    if "is_selective_unit" not in df.columns:
        return pd.DataFrame()
    if "unit_key" not in df.columns and {"date", "unit_uuid"}.issubset(df.columns):
        df["unit_key"] = df["date"].astype(str) + "|" + df["unit_uuid"].astype(str)
    if "unit_key" not in df.columns:
        return pd.DataFrame()

    out = df.loc[:, ["unit_key", "is_selective_unit"]].copy()
    out["unit_key"] = out["unit_key"].astype(str)
    out["is_selective_unit"] = out["is_selective_unit"].map(_as_bool)
    out = out.drop_duplicates(subset=["unit_key"], keep="last")
    return out


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


def _normalize_region(region: object) -> str:
    return str(region).strip().lower()


def _dedupe_preserve_order(values: Sequence[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def _draw_triangle_frame(ax) -> None:
    ax.plot([0.0, 1.0], [0.0, 0.0], color="#2e2e2e", linewidth=1.5)
    ax.plot([0.0, 0.5], [0.0, TRIANGLE_HEIGHT], color="#2e2e2e", linewidth=1.5)
    ax.plot([1.0, 0.5], [0.0, TRIANGLE_HEIGHT], color="#2e2e2e", linewidth=1.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.035, TRIANGLE_HEIGHT + 0.07)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _point_rgba_for_panel(
    panel: pd.DataFrame,
    *,
    point_color: str,
    alpha_sig: float,
    alpha_nonsig: float,
) -> np.ndarray:
    n = int(len(panel))
    rgba = np.tile(np.asarray(to_rgba(point_color), dtype=float), (n, 1))
    if "is_selective_unit" in panel.columns:
        selective = panel["is_selective_unit"].map(_as_bool).to_numpy(dtype=bool)
    else:
        selective = np.zeros(n, dtype=bool)
    rgba[:, 3] = np.where(selective, float(alpha_sig), float(alpha_nonsig))
    return rgba


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

    unit_summary_df = _load_unit_summary_df(settings)
    if not unit_summary_df.empty and "unit_key" in df.columns:
        df = df.merge(unit_summary_df, on="unit_key", how="left")
    if "is_selective_unit" not in df.columns:
        df["is_selective_unit"] = False
    df["is_selective_unit"] = df["is_selective_unit"].map(_as_bool)

    if not set(RELATIVE_COLUMNS).issubset(df.columns):
        df = _compute_relative_columns(df)

    required_rel = list(RELATIVE_COLUMNS)
    rel_df = df.loc[:, required_rel].apply(pd.to_numeric, errors="coerce")
    valid = np.all(np.isfinite(rel_df.to_numpy(dtype=float)), axis=1)
    df = df.loc[valid].copy()
    if df.empty:
        print("[plot] no rows with valid relative three-way firing components")
        return None

    df_total = df.copy()

    requested_regions = regions if regions is not None else settings.region_order
    requested_windows = windows if windows is not None else settings.default_windows

    df_total["region"] = df_total["region"].astype(str).map(lambda text: text.strip())
    df_total["region_norm"] = df_total["region"].map(_normalize_region)
    df_total["window_name"] = df_total["window_name"].astype(str).map(lambda text: text.strip())

    if requested_regions is not None:
        region_order = _dedupe_preserve_order(requested_regions)
        allowed_norm = {_normalize_region(region) for region in region_order}
        df_total = df_total.loc[df_total["region_norm"].isin(allowed_norm)].copy()
    else:
        region_order = sorted(df_total["region"].astype(str).unique().tolist())

    if requested_windows is not None:
        window_order = _dedupe_preserve_order(requested_windows)
        allowed_windows = {str(window) for window in window_order}
        df_total = df_total.loc[df_total["window_name"].astype(str).isin(allowed_windows)].copy()
    else:
        window_order = _window_order(df_total, None)

    if df_total.empty:
        print("[plot] no rows remain after region/window filtering")
        return None

    if settings.include_only_selective_units:
        df = df_total.loc[df_total["is_selective_unit"]].copy()
    else:
        df = df_total.copy()

    if not region_order or not window_order:
        print("[plot] no region/window panels available to render")
        return None

    n_rows = len(window_order)
    n_cols = len(region_order)
    figsize, dpi = _resolve_figsize_and_dpi(settings, n_rows=n_rows, n_cols=n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)
    compact = float(figsize[1]) <= 2.4

    win_labels = _window_label_map(df_total)
    panel_counts: list[dict] = []
    for row_idx, window_name in enumerate(window_order):
        for col_idx, region in enumerate(region_order):
            ax = axes[row_idx][col_idx]
            _draw_triangle_frame(ax)

            panel_total = df_total.loc[
                (df_total["window_name"].astype(str) == str(window_name))
                & (df_total["region_norm"] == _normalize_region(region))
            ].copy()
            n_total_units = (
                int(panel_total["unit_key"].astype(str).nunique())
                if "unit_key" in panel_total.columns
                else int(len(panel_total))
            )

            panel = df.loc[
                (df["window_name"].astype(str) == str(window_name))
                & (df["region_norm"] == _normalize_region(region))
            ].copy()
            n_units = int(panel["unit_key"].astype(str).nunique()) if "unit_key" in panel.columns else int(len(panel))
            panel_counts.append(
                {
                    "region": str(region),
                    "window_name": str(window_name),
                    "n_units": int(n_units),
                    "n_total_units": int(n_total_units),
                }
            )
            if n_units >= int(settings.min_units_per_panel):
                rel = panel.loc[:, list(RELATIVE_COLUMNS)].to_numpy(dtype=float)
                x, y = _to_triangle_xy(rel_int=rel[:, 0], rel_obj=rel[:, 2])
                rgba = _point_rgba_for_panel(
                    panel,
                    point_color=settings.point_color,
                    alpha_sig=settings.point_alpha_significant,
                    alpha_nonsig=settings.point_alpha_non_significant,
                )
                ax.scatter(
                    x,
                    y,
                    s=float(settings.point_size),
                    c=rgba,
                    alpha=None,
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
                (
                    f"n_sel={n_units}/{n_total_units}"
                    if settings.include_only_selective_units
                    else f"n={n_units}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5 if compact else 8.5,
            )
            ax.text(
                0.5,
                1.02,
                "Int Face",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.2 if compact else 7.8,
                color="#111111",
            )
            ax.text(
                -0.03,
                -0.02,
                "Non-Int Face",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.6 if compact else 7.0,
                color="#111111",
            )
            ax.text(
                1.03,
                -0.02,
                "Object",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.0 if compact else 7.6,
                color="#111111",
            )

            if row_idx == 0:
                ax.set_title(str(region), fontsize=9.0 if compact else 11.2, pad=5.0 if compact else 10.0)
            if col_idx == 0 and n_rows > 1:
                ax.set_ylabel(
                    str(win_labels.get(str(window_name), str(window_name))),
                    fontsize=7.3 if compact else 9.0,
                )

    if settings.show_suptitle:
        fig.suptitle(
            "Three-way fixation-condition population activity",
            fontsize=10.2 if compact else 13.2,
            y=0.992 if compact else 0.995,
        )
    fig.subplots_adjust(
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=float(settings.top_margin),
        bottom=float(settings.bottom_margin),
        wspace=float(settings.panel_wspace),
        hspace=float(settings.panel_hspace),
    )

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
