"""Plot region-level fixation-condition dominance summaries."""

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
from dal_monte_2022_analysis.ephys.analysis.fixation_condition_dominance import (
    DOMINANCE_CONDITIONS,
    DOMINANCE_UNIT_SUBSETS,
)
from dal_monte_2022_analysis.ephys.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.plot_output import normalize_extension, save_figure
from dal_monte_2022_analysis.utils.filenames import ensure_filename


DEFAULT_CONDITION_LABELS: dict[str, str] = {
    "face_interactive": "Int Face",
    "face_non_interactive": "Non-Int Face",
    "object": "Object",
}
DEFAULT_CONDITION_COLORS: dict[str, str] = {
    "face_interactive": "#b64198",
    "face_non_interactive": "#97ca3d",
    "object": "#754c29",
}
DEFAULT_SUBSET_LABELS: dict[str, str] = {
    "all_units": "All Units",
    "raw_selective_units": "Raw Selective",
    "corrected_selective_units": "Corrected Selective",
}


@dataclass
class FixationConditionDominancePlotSettings:
    """Configuration for fixation-condition dominance bar plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_condition_dominance"
    region_summary_filename: str = "region_condition_dominance_summary.csv"
    output_subdir: str = "ephys/psth/fixation_condition_dominance/plots"
    output_filename: str = "condition_dominance_by_region"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    region_order: Optional[Sequence[str]] = None
    condition_order: tuple[str, ...] = field(default_factory=lambda: DOMINANCE_CONDITIONS)
    unit_subset_order: tuple[str, ...] = field(default_factory=lambda: DOMINANCE_UNIT_SUBSETS)
    condition_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONDITION_LABELS))
    condition_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONDITION_COLORS))
    subset_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SUBSET_LABELS))
    figure_width_in: Optional[float] = None
    figure_height_in: Optional[float] = None
    show_suptitle: bool = False
    left_margin: float = 0.06
    right_margin: float = 0.995
    top_margin: float = 0.88
    bottom_margin: float = 0.18
    panel_wspace: float = 0.18
    panel_hspace: float = 0.32
    bar_width: float = 0.72


def _load_region_summary(settings: FixationConditionDominancePlotSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.region_summary_filename, ".csv")
    )
    if not path.exists():
        raise FileNotFoundError(f"Dominance region summary CSV not found: {path}")
    return pd.read_csv(path)


def _dedupe(values: Sequence[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _resolve_regions(
    df: pd.DataFrame,
    configured: Optional[Sequence[str]],
    requested: Optional[Sequence[str]],
) -> list[str]:
    if requested is not None:
        return _dedupe(requested)
    if configured is not None:
        configured_order = _dedupe(configured)
        available = set(df["region"].dropna().astype(str)) if "region" in df.columns else set()
        ordered = [region for region in configured_order if region in available]
        if ordered:
            return ordered
    if df.empty or "region" not in df.columns:
        return []
    return sorted(df["region"].dropna().astype(str).unique().tolist())


def _resolve_figsize_and_dpi(
    settings: FixationConditionDominancePlotSettings,
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
            figsize = [max(2.0 * float(n_cols), 7.5), max(1.8 * float(n_rows), 4.0)]
        width = float(settings.figure_width_in) if settings.figure_width_in is not None else float(figsize[0])
        height = float(settings.figure_height_in) if settings.figure_height_in is not None else float(figsize[1])
        figsize = [width, height]
    elif figsize is None:
        figsize = [max(2.0 * float(n_cols), 7.5), max(1.8 * float(n_rows), 4.0)]
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    return [float(figsize[0]), float(figsize[1])], dpi


def plot_fixation_condition_dominance_by_region(
    settings: FixationConditionDominancePlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    unit_subsets: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Plot region columns x unit-subset rows with dominant-condition counts."""
    df = _load_region_summary(settings)
    if df.empty:
        print("[plot] no dominance region-summary rows found")
        return None

    region_order = _resolve_regions(df, settings.region_order, regions)
    subset_order = _dedupe(unit_subsets) if unit_subsets is not None else _dedupe(settings.unit_subset_order)
    condition_order = _dedupe(settings.condition_order)
    if not region_order or not subset_order or not condition_order:
        print("[plot] no dominance panels available to render")
        return None

    df = df.loc[
        df["region"].astype(str).isin(set(region_order))
        & df["unit_subset"].astype(str).isin(set(subset_order))
        & df["dominant_condition"].astype(str).isin(set(condition_order))
    ].copy()
    if df.empty:
        print("[plot] no dominance rows remain after filters")
        return None

    n_rows = len(subset_order)
    n_cols = len(region_order)
    figsize, dpi = _resolve_figsize_and_dpi(settings, n_rows=n_rows, n_cols=n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False, sharey=True)
    max_y = max(1.0, float(pd.to_numeric(df["n_units"], errors="coerce").max()))

    x = np.arange(len(condition_order), dtype=float)
    colors = [settings.condition_colors.get(condition, "#4c4c4c") for condition in condition_order]
    labels = [settings.condition_labels.get(condition, condition) for condition in condition_order]

    panel_counts: list[dict] = []
    for row_idx, subset in enumerate(subset_order):
        for col_idx, region in enumerate(region_order):
            ax = axes[row_idx, col_idx]
            panel = df.loc[
                (df["unit_subset"].astype(str) == str(subset))
                & (df["region"].astype(str) == str(region))
            ].copy()
            count_by_condition = {
                str(row.dominant_condition): int(row.n_units)
                for row in panel.itertuples(index=False)
            }
            counts = np.asarray(
                [count_by_condition.get(condition, 0) for condition in condition_order],
                dtype=float,
            )
            ax.bar(
                x,
                counts,
                width=float(settings.bar_width),
                color=colors,
                edgecolor="#202020",
                linewidth=0.6,
            )
            ax.set_ylim(0.0, max_y * 1.15)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
            if row_idx == 0:
                ax.set_title(str(region), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(settings.subset_labels.get(str(subset), str(subset)), fontsize=9)
            for xpos, value in zip(x, counts):
                if value <= 0:
                    continue
                ax.text(xpos, value + max_y * 0.03, f"{int(value)}", ha="center", va="bottom", fontsize=7)
            n_total = int(panel["n_units_subset_total"].max()) if not panel.empty else 0
            n_classified = int(panel["n_units_classified"].max()) if not panel.empty else 0
            ax.text(
                0.98,
                0.96,
                f"n={n_classified}/{n_total}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
            panel_counts.append(
                {
                    "unit_subset": str(subset),
                    "region": str(region),
                    "n_units_subset_total": n_total,
                    "n_units_classified": n_classified,
                }
            )

    if settings.show_suptitle:
        fig.suptitle("Fixation Condition Dominance by Region", fontsize=12)
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
    ext = normalize_extension(settings.output_extension, fallback="pdf")
    stem = Path(str(settings.output_filename).strip()).stem or "condition_dominance_by_region"
    out_path = out_root / f"{stem}.{ext}"
    save_figure(fig, out_path, ext=ext, dpi=dpi, facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return {
        "output_path": out_path,
        "regions": region_order,
        "unit_subsets": subset_order,
        "conditions": condition_order,
        "panel_counts": panel_counts,
    }
