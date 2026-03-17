"""Plot comparison-group summaries for fixation selectivity pair sets."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
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


DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_ORDER: tuple[str, ...] = (
    "face_interactive__vs__face_non_interactive",
    "face_interactive__vs__object_interactive",
    "face_non_interactive__vs__object_non_interactive",
    "object_interactive__vs__object_non_interactive",
)

DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_LABELS: dict[str, str] = {
    "face_interactive__vs__face_non_interactive": "Int Face\nvs Non-Int Face",
    "face_interactive__vs__object_interactive": "Int Face\nvs Int Object",
    "face_non_interactive__vs__object_non_interactive": "Non-Int Face\nvs Non-Int Object",
    "object_interactive__vs__object_non_interactive": "Int Object\nvs Non-Int Object",
}

DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_COLORS: dict[str, str] = {
    "face_interactive__vs__face_non_interactive": "#C44E52",
    "face_interactive__vs__object_interactive": "#4C72B0",
    "face_non_interactive__vs__object_non_interactive": "#55A868",
    "object_interactive__vs__object_non_interactive": "#8172B3",
}


@dataclass
class FixationSelectivityComparisonGroupPlotSettings:
    """Configuration for comparison-group summary figures."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    pair_summary_filename: str = "pair_selectivity.csv"
    output_subdir: str = "ephys/psth/fixation_psth_selectivity_comparison_group_plots"
    comparison_label: str = "interactive_state_matched"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    selective_windows: Optional[Sequence[str]] = ("pre_fix", "peri_fix", "post_fix")
    region_order: Optional[Sequence[str]] = ("BLA", "ACCg", "dmPFC", "OFC")
    pair_order: Sequence[str] = field(
        default_factory=lambda: tuple(DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_ORDER),
    )
    pair_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_LABELS),
    )
    pair_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_COLORS),
    )
    min_units_per_region: int = 1
    fraction_bar_output_filename: Optional[str] = None
    overlap_matrix_output_filename: Optional[str] = None
    fraction_bar_figure_width_in: float = 8.5
    fraction_bar_figure_height_in: float = 2.9
    fraction_bar_left_margin: float = 0.05
    fraction_bar_right_margin: float = 0.995
    fraction_bar_top_margin: float = 0.86
    fraction_bar_bottom_margin: float = 0.26
    fraction_bar_wspace: float = 0.22
    overlap_figure_width_in: float = 9.2
    overlap_figure_height_in: float = 5.6
    overlap_left_margin: float = 0.08
    overlap_right_margin: float = 0.995
    overlap_top_margin: float = 0.90
    overlap_bottom_margin: float = 0.10
    overlap_wspace: float = 0.18
    overlap_hspace: float = 0.28
    overlap_bar_height_ratio: float = 2.6
    overlap_matrix_height_ratio: float = 1.5


def _comparison_label_token(label: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(label).strip()).strip("_").lower()
    return token or "comparison"


def _filename_with_comparison_label(filename: str, comparison_label: str) -> str:
    token = ensure_filename(filename, ".csv")
    stem = token[:-4]
    return f"{stem}__{_comparison_label_token(comparison_label)}.csv"


def _normalize_region_token(region: object) -> str:
    return str(region).strip().lower()


def _region_sort_key(region: str, region_order: Optional[Sequence[str]]) -> tuple[int, int | str]:
    text = str(region).strip()
    norm = _normalize_region_token(text)
    if region_order is None:
        return (1, norm)
    for idx, expected in enumerate(region_order):
        if norm == _normalize_region_token(expected):
            return (0, idx)
    return (1, norm)


def _ordered_regions(
    region_names: Sequence[str],
    region_order: Optional[Sequence[str]],
) -> list[str]:
    return sorted({str(region).strip() for region in region_names if str(region).strip()}, key=lambda x: _region_sort_key(x, region_order))


def _as_bool(val) -> bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val) != 0
    if isinstance(val, (float, np.floating)):
        return float(val) != 0.0
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "t", "yes", "y"}


def _split_sig_windows(raw: object) -> set[str]:
    if raw is None:
        return set()
    text = str(raw).strip()
    if not text:
        return set()
    return {token.strip() for token in text.split("|") if token.strip()}


def _pair_is_selective_for_windows(
    row: pd.Series,
    *,
    selective_windows: Optional[Sequence[str]],
) -> bool:
    fallback = _as_bool(row.get("is_selective_pair"))
    if selective_windows is None:
        return fallback
    allowed = {str(name).strip() for name in selective_windows if str(name).strip()}
    if not allowed:
        return fallback
    sig_windows = _split_sig_windows(row.get("significant_windows"))
    if sig_windows:
        return bool(sig_windows.intersection(allowed))
    return fallback


def _resolve_pair_order(
    settings: FixationSelectivityComparisonGroupPlotSettings,
    pair_df: pd.DataFrame,
) -> list[str]:
    configured = [str(pair).strip() for pair in settings.pair_order if str(pair).strip()]
    available = []
    if not pair_df.empty and "pair_label" in pair_df.columns:
        available = sorted(pair_df["pair_label"].dropna().astype(str).unique().tolist())
    ordered: list[str] = []
    for pair in configured:
        if pair not in ordered and (not available or pair in available):
            ordered.append(pair)
    for pair in available:
        if pair not in ordered:
            ordered.append(pair)
    return ordered


def _load_pair_summary_df(
    settings: FixationSelectivityComparisonGroupPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    root = build_analysis_output_dir(cfg, settings.input_subdir)
    base_filename = ensure_filename(settings.pair_summary_filename, ".csv")
    comparison_filename = _filename_with_comparison_label(base_filename, settings.comparison_label)

    in_path = root / comparison_filename
    if in_path.exists():
        df = pd.read_csv(in_path)
    else:
        fallback_path = root / base_filename
        if not fallback_path.exists():
            raise FileNotFoundError(f"Pair selectivity CSV not found: {in_path}")
        df = pd.read_csv(fallback_path)
        if "comparison_label" in df.columns:
            df = df.loc[df["comparison_label"].astype(str) == str(settings.comparison_label)].copy()

    if df.empty:
        return df
    if "region" not in df.columns:
        df["region"] = "unknown"
    else:
        df["region"] = df["region"].fillna("unknown").astype(str).replace({"": "unknown"})
    if "unit_key" not in df.columns:
        if {"date", "unit_uuid"}.issubset(df.columns):
            df["unit_key"] = df["date"].astype(str) + "|" + df["unit_uuid"].astype(str)
        else:
            raise ValueError("pair selectivity CSV must include 'unit_key' or ('date' and 'unit_uuid').")
    if "pair_label" not in df.columns:
        raise ValueError("pair selectivity CSV missing required column 'pair_label'.")
    if "is_selective_pair" not in df.columns:
        raise ValueError("pair selectivity CSV missing required column 'is_selective_pair'.")

    df["is_selective_pair"] = df["is_selective_pair"].map(_as_bool)
    df["is_selective_pair_for_plot"] = df.apply(
        lambda row: _pair_is_selective_for_windows(row, selective_windows=settings.selective_windows),
        axis=1,
    )
    if regions is not None:
        allowed = {str(region) for region in regions}
        df = df.loc[df["region"].astype(str).isin(allowed)].copy()
    return df


def _compute_region_summary(
    region: str,
    df_region: pd.DataFrame,
    *,
    pair_order: Sequence[str],
) -> Optional[dict]:
    if df_region.empty:
        return None

    total_units = int(df_region["unit_key"].astype(str).nunique())
    if total_units <= 0:
        return None

    selective_df = df_region.loc[df_region["is_selective_pair_for_plot"].map(_as_bool)].copy()
    pair_to_units: dict[str, set[str]] = {}
    for pair in pair_order:
        pair_to_units[str(pair)] = set(
            selective_df.loc[
                selective_df["pair_label"].astype(str) == str(pair),
                "unit_key",
            ].astype(str).tolist()
        )

    any_selective_units = set()
    for units in pair_to_units.values():
        any_selective_units.update(units)

    pattern_counts: dict[tuple[str, ...], int] = {}
    for unit_key in sorted(any_selective_units):
        active_pairs = tuple(pair for pair in pair_order if unit_key in pair_to_units.get(str(pair), set()))
        if not active_pairs:
            continue
        pattern_counts[active_pairs] = pattern_counts.get(active_pairs, 0) + 1

    pair_counts = {pair: int(len(pair_to_units.get(str(pair), set()))) for pair in pair_order}
    pair_fractions = {
        pair: (float(pair_counts[pair]) / float(total_units)) if total_units > 0 else np.nan
        for pair in pair_order
    }

    return {
        "region": str(region),
        "total_units": int(total_units),
        "any_selective_units": int(len(any_selective_units)),
        "pair_counts": pair_counts,
        "pair_fractions": pair_fractions,
        "pattern_counts": pattern_counts,
    }


def _resolve_plot_cfg(
    settings: FixationSelectivityComparisonGroupPlotSettings,
) -> tuple[Optional[dict], Optional[int]]:
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        _, cfg_dpi = resolve_figsize(plot_cfg)
        return plot_cfg, cfg_dpi
    return None, None


def _build_output_path(
    settings: FixationSelectivityComparisonGroupPlotSettings,
    *,
    filename: Optional[str],
    suffix: str,
) -> Path:
    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    token = str(filename).strip() if filename is not None else ""
    if not token:
        token = f"{_comparison_label_token(settings.comparison_label)}_{suffix}"
    ext = normalize_extension(settings.output_extension, fallback="pdf")
    return out_root / f"{token}.{ext}"


def _draw_fraction_bar_panel(
    ax,
    *,
    summary: dict,
    pair_order: Sequence[str],
    settings: FixationSelectivityComparisonGroupPlotSettings,
) -> None:
    total_units = int(summary["total_units"])
    region = str(summary["region"])
    x = np.arange(len(pair_order), dtype=float)
    fracs = [float(summary["pair_fractions"].get(pair, np.nan)) for pair in pair_order]
    colors = [settings.pair_colors.get(pair, "#4c4c4c") for pair in pair_order]

    ax.bar(
        x,
        np.ones_like(x, dtype=float),
        width=0.78,
        color="#ebebeb",
        edgecolor="#c8c8c8",
        linewidth=0.8,
        zorder=1,
    )
    ax.bar(
        x,
        fracs,
        width=0.78,
        color=colors,
        edgecolor="#222222",
        linewidth=0.7,
        zorder=2,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [settings.pair_labels.get(pair, str(pair)) for pair in pair_order],
        fontsize=7.0,
    )
    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    ax.set_yticklabels([f"{val:.2f}" for val in np.linspace(0.0, 1.0, 5)], fontsize=7.0)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"{region}\nN={total_units}", fontsize=9.2, pad=4.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")


def _sorted_patterns(pattern_counts: dict[tuple[str, ...], int], pair_order: Sequence[str]) -> list[tuple[tuple[str, ...], int]]:
    def _pattern_key(item):
        active_pairs, count = item
        binary = tuple(1 if pair in active_pairs else 0 for pair in pair_order)
        return (-int(count), -sum(binary), binary)

    return sorted(pattern_counts.items(), key=_pattern_key)


def _draw_overlap_matrix_panel(
    fig,
    parent_spec,
    *,
    summary: dict,
    pair_order: Sequence[str],
    settings: FixationSelectivityComparisonGroupPlotSettings,
) -> tuple[plt.Axes, plt.Axes]:
    total_units = int(summary["total_units"])
    region = str(summary["region"])
    patterns = _sorted_patterns(summary["pattern_counts"], pair_order)
    if not patterns:
        patterns = []

    inner = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=parent_spec,
        height_ratios=[
            float(settings.overlap_bar_height_ratio),
            float(settings.overlap_matrix_height_ratio),
        ],
        hspace=0.05,
    )
    ax_bar = fig.add_subplot(inner[0, 0])
    ax_mat = fig.add_subplot(inner[1, 0], sharex=ax_bar)

    if not patterns:
        ax_bar.text(0.5, 0.5, f"{region}\nNo selective units", ha="center", va="center", fontsize=9.0)
        ax_bar.axis("off")
        ax_mat.axis("off")
        return ax_bar, ax_mat

    x = np.arange(len(patterns), dtype=float)
    counts = np.asarray([int(count) for _, count in patterns], dtype=float)
    colors = []
    for active_pairs, _ in patterns:
        if len(active_pairs) == 1:
            colors.append(settings.pair_colors.get(active_pairs[0], "#4c4c4c"))
        else:
            colors.append("#6f6f6f")

    ax_bar.bar(x, counts, width=0.72, color=colors, edgecolor="#222222", linewidth=0.7)
    y_max = max(1.0, float(np.nanmax(counts)) if counts.size > 0 else 1.0)
    ax_bar.set_ylim(0.0, y_max * 1.18)
    ax_bar.set_ylabel("n", fontsize=7.2)
    ax_bar.set_title(
        f"{region}\nN={total_units}, selective={int(summary['any_selective_units'])}",
        fontsize=9.0,
        pad=4.0,
    )
    ax_bar.tick_params(axis="y", labelsize=7.0)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.grid(axis="y", color="#e0e0e0", linewidth=0.7, alpha=0.8)
    ax_bar.set_axisbelow(True)
    for xi, count in zip(x, counts):
        ax_bar.text(xi, float(count) + 0.03 * y_max, f"{int(count)}", ha="center", va="bottom", fontsize=6.8)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    row_y = np.arange(len(pair_order), dtype=float)
    ax_mat.scatter(
        np.repeat(x, len(pair_order)),
        np.tile(row_y, len(patterns)),
        s=18.0,
        color="#d0d0d0",
        zorder=1,
    )
    for xi, (active_pairs, _) in zip(x, patterns):
        active_idx = [idx for idx, pair in enumerate(pair_order) if pair in active_pairs]
        if not active_idx:
            continue
        if len(active_idx) >= 2:
            ax_mat.plot(
                [xi, xi],
                [float(min(active_idx)), float(max(active_idx))],
                color="#4a4a4a",
                linewidth=1.0,
                zorder=2,
            )
        for idx in active_idx:
            pair = pair_order[idx]
            ax_mat.scatter(
                [xi],
                [float(idx)],
                s=28.0,
                color=settings.pair_colors.get(pair, "#4c4c4c"),
                edgecolors="#1f1f1f",
                linewidths=0.5,
                zorder=3,
            )

    ax_mat.set_yticks(row_y)
    ax_mat.set_yticklabels([settings.pair_labels.get(pair, str(pair)) for pair in pair_order], fontsize=6.8)
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels([])
    ax_mat.set_ylim(float(len(pair_order)) - 0.5, -0.5)
    ax_mat.set_xlim(-0.6, float(len(patterns)) - 0.4)
    ax_mat.tick_params(axis="x", length=0)
    for spine in ("top", "right", "bottom"):
        ax_mat.spines[spine].set_visible(False)
    ax_mat.spines["left"].set_color("#666666")
    return ax_bar, ax_mat


def _render_fraction_bars(
    settings: FixationSelectivityComparisonGroupPlotSettings,
    *,
    summaries: Sequence[dict],
    pair_order: Sequence[str],
) -> Path:
    if not summaries:
        raise ValueError("No region summaries available for fraction-bar plotting.")

    _, cfg_dpi = _resolve_plot_cfg(settings)
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    out_path = _build_output_path(
        settings,
        filename=settings.fraction_bar_output_filename,
        suffix="fraction_bars",
    )

    n_cols = len(summaries)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=[float(settings.fraction_bar_figure_width_in), float(settings.fraction_bar_figure_height_in)],
        dpi=dpi,
        squeeze=False,
    )
    axes_flat = list(axes[0])
    for ax, summary in zip(axes_flat, summaries):
        _draw_fraction_bar_panel(ax, summary=summary, pair_order=pair_order, settings=settings)
    for ax in axes_flat[1:]:
        ax.tick_params(axis="y", labelleft=False)
    axes_flat[0].set_ylabel("Fraction of Units", fontsize=8.0)

    fig.subplots_adjust(
        left=float(settings.fraction_bar_left_margin),
        right=float(settings.fraction_bar_right_margin),
        top=float(settings.fraction_bar_top_margin),
        bottom=float(settings.fraction_bar_bottom_margin),
        wspace=float(settings.fraction_bar_wspace),
    )
    fig.patch.set_facecolor("white")
    save_figure(
        fig,
        out_path,
        ext=normalize_extension(settings.output_extension, fallback="pdf"),
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)
    return out_path


def _render_overlap_matrix(
    settings: FixationSelectivityComparisonGroupPlotSettings,
    *,
    summaries: Sequence[dict],
    pair_order: Sequence[str],
) -> Path:
    if not summaries:
        raise ValueError("No region summaries available for overlap-matrix plotting.")

    _, cfg_dpi = _resolve_plot_cfg(settings)
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    out_path = _build_output_path(
        settings,
        filename=settings.overlap_matrix_output_filename,
        suffix="overlap_matrix",
    )

    n_panels = len(summaries)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(math.ceil(float(n_panels) / float(n_cols)))
    fig = plt.figure(
        figsize=[float(settings.overlap_figure_width_in), float(settings.overlap_figure_height_in)],
        dpi=dpi,
    )
    outer = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    for idx, summary in enumerate(summaries):
        _draw_overlap_matrix_panel(
            fig,
            outer[idx // n_cols, idx % n_cols],
            summary=summary,
            pair_order=pair_order,
            settings=settings,
        )

    for idx in range(n_panels, n_rows * n_cols):
        ax = fig.add_subplot(outer[idx // n_cols, idx % n_cols])
        ax.axis("off")

    fig.subplots_adjust(
        left=float(settings.overlap_left_margin),
        right=float(settings.overlap_right_margin),
        top=float(settings.overlap_top_margin),
        bottom=float(settings.overlap_bottom_margin),
        wspace=float(settings.overlap_wspace),
        hspace=float(settings.overlap_hspace),
    )
    fig.patch.set_facecolor("white")
    save_figure(
        fig,
        out_path,
        ext=normalize_extension(settings.output_extension, fallback="pdf"),
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)
    return out_path


def plot_fixation_selectivity_comparison_group_summaries(
    settings: FixationSelectivityComparisonGroupPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Plot comparison-group fraction bars and overlap matrices across regions."""
    pair_df = _load_pair_summary_df(settings, regions=regions)
    if pair_df.empty:
        print("[plot] no pair selectivity rows found for comparison-group plotting")
        return None

    pair_order = _resolve_pair_order(settings, pair_df)
    summaries: list[dict] = []
    for region, df_region in pair_df.groupby("region", sort=True, dropna=False):
        summary = _compute_region_summary(str(region), df_region.copy(), pair_order=pair_order)
        if summary is None:
            continue
        if int(summary["total_units"]) < int(settings.min_units_per_region):
            continue
        summaries.append(summary)

    if not summaries:
        print("[plot] no regions passed filters for comparison-group plotting")
        return None

    summaries = sorted(
        summaries,
        key=lambda row: _region_sort_key(str(row["region"]), settings.region_order),
    )

    bar_out = _render_fraction_bars(settings, summaries=summaries, pair_order=pair_order)
    overlap_out = _render_overlap_matrix(settings, summaries=summaries, pair_order=pair_order)
    return {
        "comparison_label": str(settings.comparison_label),
        "pair_order": list(pair_order),
        "region_summaries": summaries,
        "fraction_bar_output_path": bar_out,
        "overlap_matrix_output_path": overlap_out,
    }
