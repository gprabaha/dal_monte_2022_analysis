"""Plot region-comparison summaries for three-way fixation compositions."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class FixationThreeWayRegionComparisonPlotSettings:
    """Configuration for region-comparison heatmap plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_psth_selectivity_region_comparison"
    pairwise_summary_filename: str = "pairwise_region_comparisons.csv"
    window_summary_filename: str = "window_region_comparisons.csv"
    output_subdir: str = "ephys/psth/fixation_psth_selectivity_region_comparison/plots"
    output_filename: str = "region_comparison_heatmaps"
    output_extension: str = "png"
    output_dpi: Optional[int] = 220
    alpha: float = 0.05
    pvalue_floor: float = 1e-6
    annotation_max_regions: int = 10


def _load_pairwise_and_window_summaries(
    settings: FixationThreeWayRegionComparisonPlotSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(settings.cfg_path)
    root = build_analysis_output_dir(cfg, settings.input_subdir)
    pair_path = root / ensure_filename(settings.pairwise_summary_filename, ".csv")
    win_path = root / ensure_filename(settings.window_summary_filename, ".csv")
    if not pair_path.exists():
        raise FileNotFoundError(f"Pairwise region comparison CSV not found: {pair_path}")
    if not win_path.exists():
        raise FileNotFoundError(f"Window region comparison CSV not found: {win_path}")

    pair_df = pd.read_csv(pair_path)
    win_df = pd.read_csv(win_path)
    return pair_df, win_df


def _resolve_figsize_and_dpi(
    settings: FixationThreeWayRegionComparisonPlotSettings,
    *,
    n_rows: int,
) -> tuple[list[float], Optional[int]]:
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        figsize, cfg_dpi = resolve_figsize(plot_cfg)
    else:
        figsize, cfg_dpi = (None, None)

    if figsize is None:
        figsize = [14.0, max(3.4 * float(n_rows), 6.8)]
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    return [float(figsize[0]), float(figsize[1])], dpi


def _window_order(win_df: pd.DataFrame, pair_df: pd.DataFrame) -> list[str]:
    if not win_df.empty and "window_name" in win_df.columns:
        df = win_df.copy()
        if "window_start_ms" in df.columns:
            df["window_start_ms"] = pd.to_numeric(df["window_start_ms"], errors="coerce")
        else:
            df["window_start_ms"] = np.nan
        if "window_stop_ms" in df.columns:
            df["window_stop_ms"] = pd.to_numeric(df["window_stop_ms"], errors="coerce")
        else:
            df["window_stop_ms"] = np.nan
        return (
            df.sort_values(["window_start_ms", "window_stop_ms", "window_name"], na_position="last")
            ["window_name"]
            .astype(str)
            .tolist()
        )
    if pair_df.empty:
        return []
    return sorted(pair_df["window_name"].astype(str).unique().tolist())


def _region_order(pair_df: pd.DataFrame) -> list[str]:
    if pair_df.empty:
        return []
    regions = set(pair_df["region_a"].astype(str).tolist()) | set(pair_df["region_b"].astype(str).tolist())
    return sorted(regions)


def _build_pairwise_matrices(
    pair_df: pd.DataFrame,
    *,
    window_name: str,
    regions: list[str],
    pvalue_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(regions)
    stat_mat = np.full((n, n), np.nan, dtype=float)
    p_adj_mat = np.full((n, n), np.nan, dtype=float)
    sig_mat = np.zeros((n, n), dtype=bool)
    if pair_df.empty or n <= 0:
        return stat_mat, p_adj_mat, sig_mat

    r_index = {region: idx for idx, region in enumerate(regions)}
    df = pair_df.loc[pair_df["window_name"].astype(str) == str(window_name)].copy()
    for row in df.itertuples(index=False):
        ra = str(getattr(row, "region_a"))
        rb = str(getattr(row, "region_b"))
        if ra not in r_index or rb not in r_index:
            continue
        i = r_index[ra]
        j = r_index[rb]
        stat = float(getattr(row, "centroid_distance_ilr", np.nan))
        p_adj = getattr(row, "p_value_adjusted", np.nan)
        p_adj = float(p_adj) if np.isfinite(float(p_adj)) else np.nan
        sig = bool(getattr(row, "significant", False))
        stat_mat[i, j] = stat
        stat_mat[j, i] = stat
        p_adj_mat[i, j] = p_adj
        p_adj_mat[j, i] = p_adj
        sig_mat[i, j] = sig
        sig_mat[j, i] = sig

    for k in range(n):
        stat_mat[k, k] = 0.0
        p_adj_mat[k, k] = 1.0
        sig_mat[k, k] = False

    p_adj_mat = np.clip(p_adj_mat, max(float(pvalue_floor), 1e-12), 1.0)
    return stat_mat, p_adj_mat, sig_mat


def plot_fixation_three_way_region_comparison_heatmaps(
    settings: FixationThreeWayRegionComparisonPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    windows: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Plot per-window region-comparison heatmaps (effect + adjusted p-values)."""
    pair_df, win_df = _load_pairwise_and_window_summaries(settings)
    if pair_df.empty:
        print("[plot] no pairwise region comparison rows found")
        return None

    if regions is not None:
        allowed = {str(region) for region in regions}
        pair_df = pair_df.loc[
            pair_df["region_a"].astype(str).isin(allowed)
            & pair_df["region_b"].astype(str).isin(allowed)
        ].copy()
        if not win_df.empty and "window_name" in win_df.columns:
            valid_windows = set(pair_df["window_name"].astype(str))
            win_df = win_df.loc[win_df["window_name"].astype(str).isin(valid_windows)].copy()
    if windows is not None:
        allowed_windows = {str(window) for window in windows}
        pair_df = pair_df.loc[pair_df["window_name"].astype(str).isin(allowed_windows)].copy()
        if not win_df.empty and "window_name" in win_df.columns:
            win_df = win_df.loc[win_df["window_name"].astype(str).isin(allowed_windows)].copy()
    if pair_df.empty:
        print("[plot] no pairwise rows remain after filters")
        return None

    win_order = _window_order(win_df, pair_df)
    region_order = _region_order(pair_df)
    if not win_order or not region_order:
        print("[plot] unable to resolve window/region order for plotting")
        return None

    n_rows = len(win_order)
    figsize, dpi = _resolve_figsize_and_dpi(settings, n_rows=n_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, dpi=dpi, squeeze=False)

    stat_max = 0.0
    logp_max = 0.0
    matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for window_name in win_order:
        stat_mat, p_adj_mat, sig_mat = _build_pairwise_matrices(
            pair_df,
            window_name=window_name,
            regions=region_order,
            pvalue_floor=float(settings.pvalue_floor),
        )
        matrices[str(window_name)] = (stat_mat, p_adj_mat, sig_mat)
        stat_max = max(stat_max, float(np.nanmax(stat_mat)) if np.any(np.isfinite(stat_mat)) else 0.0)
        logp = -np.log10(p_adj_mat)
        logp_max = max(logp_max, float(np.nanmax(logp)) if np.any(np.isfinite(logp)) else 0.0)

    stat_max = stat_max if stat_max > 0 else 1.0
    logp_max = logp_max if logp_max > 0 else 1.0
    stat_im = None
    logp_im = None

    show_numeric = len(region_order) <= int(settings.annotation_max_regions)
    for row_idx, window_name in enumerate(win_order):
        ax_stat = axes[row_idx, 0]
        ax_logp = axes[row_idx, 1]
        stat_mat, p_adj_mat, sig_mat = matrices[str(window_name)]
        logp_mat = -np.log10(p_adj_mat)

        stat_im = ax_stat.imshow(stat_mat, cmap="magma", vmin=0.0, vmax=stat_max)
        logp_im = ax_logp.imshow(logp_mat, cmap="viridis", vmin=0.0, vmax=logp_max)

        for ax in (ax_stat, ax_logp):
            ax.set_xticks(np.arange(len(region_order)))
            ax.set_yticks(np.arange(len(region_order)))
            ax.set_xticklabels(region_order, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(region_order, fontsize=8)
            ax.set_aspect("equal")

        ax_stat.set_title("Centroid Distance (ILR)", fontsize=10)
        ax_logp.set_title("-log10(adjusted p)", fontsize=10)

        win_row = (
            win_df.loc[win_df["window_name"].astype(str) == str(window_name)].iloc[0]
            if (not win_df.empty and str(window_name) in set(win_df["window_name"].astype(str)))
            else None
        )
        if win_row is not None:
            gp = win_row.get("global_p_value_adjusted", np.nan)
            gp_text = "nan" if not np.isfinite(float(gp)) else f"{float(gp):.3g}"
            y_label = f"{window_name}\nglobal p_adj={gp_text}"
        else:
            y_label = str(window_name)
        ax_stat.set_ylabel(y_label, fontsize=9)

        if show_numeric:
            for i in range(len(region_order)):
                for j in range(len(region_order)):
                    if i == j:
                        continue
                    stat_val = stat_mat[i, j]
                    if np.isfinite(stat_val):
                        ax_stat.text(j, i, f"{stat_val:.2f}", ha="center", va="center", fontsize=6, color="white")
                    if sig_mat[i, j] and i < j:
                        ax_logp.text(
                            j,
                            i,
                            "*",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white",
                            fontweight="bold",
                        )

        thr = -np.log10(float(settings.alpha))
        for i in range(len(region_order)):
            for j in range(len(region_order)):
                if i >= j:
                    continue
                if np.isfinite(logp_mat[i, j]) and logp_mat[i, j] >= thr:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.1)
                    ax_logp.add_patch(rect)
                    rect_mirror = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.1)
                    ax_logp.add_patch(rect_mirror)

    if stat_im is not None:
        fig.colorbar(stat_im, ax=axes[:, 0], fraction=0.02, pad=0.02, label="Distance")
    if logp_im is not None:
        fig.colorbar(logp_im, ax=axes[:, 1], fraction=0.02, pad=0.02, label="-log10(p_adj)")

    fig.suptitle("Region Comparison of Three-Way Fixation Composition", fontsize=13.5, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.08, wspace=0.25, hspace=0.35)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    ext = normalize_extension(settings.output_extension, fallback="png")
    stem = Path(str(settings.output_filename).strip()).stem or "region_comparison_heatmaps"
    out_path = out_root / f"{stem}.{ext}"
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
        "window_order": win_order,
        "region_order": region_order,
    }

