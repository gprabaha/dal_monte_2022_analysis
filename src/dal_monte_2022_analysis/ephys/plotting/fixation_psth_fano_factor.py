"""Plot region-wise fixation PSTH Fano-factor summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.runtime.io.plot_output import normalize_extension, save_figure
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


DEFAULT_REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")
DEFAULT_REGION_LABELS: dict[str, str] = {
    "bla": "BLA",
    "acc": "ACC",
    "accg": "ACC",
    "dmpfc": "dmPFC",
    "ofc": "OFC",
}
DEFAULT_CONDITION_ORDER: tuple[str, ...] = (
    "object",
    "face_non_interactive",
    "face_interactive",
)
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
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "region",
    "condition",
    "bin_index",
    "bin_center_s_rel",
    "mean_fano_factor",
    "sem_fano_factor",
    "n_units",
)


@dataclass
class FixationPSTHFanoFactorPlotSettings:
    """Configuration for region-level fixation PSTH Fano plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_psth_fano_factor"
    region_summary_filename: str = "region_fano_factor_summary.csv"
    output_subdir: str = "ephys/psth/fixation_psth_fano_factor/plots"
    output_filename: str = "fixation_psth_fano_factor_by_region"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    region_order: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_REGION_ORDER))
    region_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_REGION_LABELS))
    condition_order: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_CONDITION_ORDER))
    condition_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONDITION_LABELS))
    condition_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONDITION_COLORS))
    subplot_ncols: int = 4
    figure_width_in: float = 8.25
    figure_height_in: float = 2.6
    left_margin: float = 0.06
    right_margin: float = 0.995
    top_margin: float = 0.80
    bottom_margin: float = 0.22
    wspace: float = 0.22
    hspace: float = 0.26
    line_width: float = 1.8
    shade_alpha: float = 0.18
    min_units_per_condition: int = 1
    show_zero_line: bool = True
    show_unity_line: bool = True
    time_window_ms: Optional[tuple[float, float]] = (-500.0, 500.0)
    x_label: str = "Time from fixation onset (ms)"
    y_label: str = "Fano Factor"


def _normalize_region_token(value: object) -> str:
    token = str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "bla": "bla",
        "acc": "accg",
        "accg": "accg",
        "dmpfc": "dmpfc",
        "dmpf": "dmpfc",
        "ofc": "ofc",
    }
    return aliases.get(token, token)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        raise ValueError(f"Unable to read CSV '{path}': {exc}") from exc


def _load_region_summary_df(settings: FixationPSTHFanoFactorPlotSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.region_summary_filename, ".csv")
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Fixation PSTH Fano region summary not found: {in_path}")
    df = _read_csv_or_empty(in_path)
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=list(_REQUIRED_COLUMNS) + ["region_norm"])

    missing = sorted(set(_REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            "Fixation PSTH Fano region summary CSV missing required columns: "
            + ", ".join(missing)
        )

    out = df.loc[:, list(_REQUIRED_COLUMNS)].copy()
    out["region"] = out["region"].astype(str).map(str.strip)
    out["condition"] = out["condition"].astype(str).map(str.strip)
    out["bin_index"] = pd.to_numeric(out["bin_index"], errors="coerce").astype("Int64")
    for col in ("bin_center_s_rel", "mean_fano_factor", "sem_fano_factor"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["n_units"] = pd.to_numeric(out["n_units"], errors="coerce").fillna(0).astype(int)
    out["region_norm"] = out["region"].map(_normalize_region_token)
    return out


def _resolve_regions(
    settings: FixationPSTHFanoFactorPlotSettings,
    df: pd.DataFrame,
    *,
    regions: Optional[Sequence[str]] = None,
) -> list[str]:
    requested = list(regions) if regions is not None else list(settings.region_order)
    resolved: list[str] = []
    seen: set[str] = set()
    available = {
        _normalize_region_token(token)
        for token in df["region"].astype(str).tolist()
    }

    for token in requested:
        norm = _normalize_region_token(token)
        if not norm or norm in seen or norm not in available:
            continue
        resolved.append(norm)
        seen.add(norm)

    for token in df["region_norm"].astype(str).tolist():
        norm = _normalize_region_token(token)
        if not norm or norm in seen:
            continue
        resolved.append(norm)
        seen.add(norm)

    return resolved


def _display_region(region: str, settings: FixationPSTHFanoFactorPlotSettings) -> str:
    return settings.region_labels.get(_normalize_region_token(region), str(region).strip())


def _resolve_plot_cfg(
    settings: FixationPSTHFanoFactorPlotSettings,
) -> tuple[Optional[dict], Optional[int]]:
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        _, cfg_dpi = resolve_figsize(plot_cfg)
        return plot_cfg, cfg_dpi
    return None, None


def _apply_time_window(
    df: pd.DataFrame,
    *,
    time_window_ms: Optional[tuple[float, float]],
) -> pd.DataFrame:
    if time_window_ms is None:
        return df
    start_ms = float(time_window_ms[0])
    stop_ms = float(time_window_ms[1])
    lo_s = min(start_ms, stop_ms) / 1000.0
    hi_s = max(start_ms, stop_ms) / 1000.0
    mask = (
        pd.to_numeric(df["bin_center_s_rel"], errors="coerce").to_numpy(dtype=float) >= lo_s
    ) & (
        pd.to_numeric(df["bin_center_s_rel"], errors="coerce").to_numpy(dtype=float) <= hi_s
    )
    if not np.any(mask):
        raise ValueError(
            "Requested Fano-factor plot time window contains no bins: "
            f"[{start_ms}, {stop_ms}] ms"
        )
    return df.loc[mask].copy()


def plot_fixation_psth_fano_factor_by_region(
    settings: FixationPSTHFanoFactorPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Render a region-panel Fano-factor figure with mean +/- SEM traces."""

    summary_df = _load_region_summary_df(settings)
    summary_df = _apply_time_window(summary_df, time_window_ms=settings.time_window_ms)
    if summary_df.empty:
        print("[plot] no fixation PSTH Fano rows available for plotting")
        return None

    region_order = _resolve_regions(settings, summary_df, regions=regions)
    condition_order = [
        condition
        for condition in settings.condition_order
        if condition in set(summary_df["condition"].astype(str).tolist())
    ]
    if not region_order or not condition_order:
        print("[plot] unable to resolve regions or conditions for fixation PSTH Fano plot")
        return None

    _, cfg_dpi = _resolve_plot_cfg(settings)
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    ext = normalize_extension(settings.output_extension, fallback="pdf")

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / ensure_filename(settings.output_filename, f".{ext}")

    n_cols = max(1, min(int(settings.subplot_ncols), len(region_order)))
    n_rows = int(np.ceil(len(region_order) / float(n_cols)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(float(settings.figure_width_in), float(settings.figure_height_in)),
        dpi=dpi,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = list(axes.reshape(-1))

    legend_handles = []
    legend_labels = []
    region_summaries: list[dict] = []

    for ax_idx, region in enumerate(region_order):
        ax = axes_flat[ax_idx]
        region_df = summary_df.loc[summary_df["region_norm"] == region].copy()
        if region_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10.0)
            ax.set_title(_display_region(region, settings), fontsize=11.0)
            ax.set_axis_off()
            region_summaries.append({"region": region, "conditions_plotted": []})
            continue

        plotted_conditions: list[str] = []
        n_units_region = int(region_df["n_units"].max()) if not region_df.empty else 0
        for condition in condition_order:
            cond_df = region_df.loc[region_df["condition"].astype(str) == str(condition)].copy()
            if cond_df.empty:
                continue
            cond_df = cond_df.sort_values("bin_index")
            x_ms = cond_df["bin_center_s_rel"].to_numpy(dtype=float) * 1000.0
            mean_vec = cond_df["mean_fano_factor"].to_numpy(dtype=float)
            sem_vec = cond_df["sem_fano_factor"].to_numpy(dtype=float)
            n_units_max = int(cond_df["n_units"].max()) if not cond_df.empty else 0
            if n_units_max < int(settings.min_units_per_condition):
                continue
            if not np.isfinite(mean_vec).any():
                continue

            lower = np.maximum(mean_vec - sem_vec, 0.0)
            upper = mean_vec + sem_vec
            color = settings.condition_colors.get(str(condition), "#666666")
            label = settings.condition_labels.get(str(condition), str(condition))
            line, = ax.plot(
                x_ms,
                mean_vec,
                color=color,
                linewidth=float(settings.line_width),
                label=label,
            )
            ax.fill_between(
                x_ms,
                lower,
                upper,
                color=color,
                alpha=float(settings.shade_alpha),
                linewidth=0.0,
            )
            plotted_conditions.append(str(condition))
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        if settings.show_zero_line:
            ax.axvline(0.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.7)
        if settings.show_unity_line:
            ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle=":", alpha=0.8)

        title = _display_region(region, settings)
        if n_units_region > 0:
            title = f"{title} (N={n_units_region})"
        ax.set_title(title, fontsize=11.0)
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        if settings.time_window_ms is not None:
            start_ms = float(settings.time_window_ms[0])
            stop_ms = float(settings.time_window_ms[1])
            ax.set_xlim(min(start_ms, stop_ms), max(start_ms, stop_ms))
            ax.set_xticks([min(start_ms, stop_ms), 0.0, max(start_ms, stop_ms)])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        region_summaries.append(
            {
                "region": str(region),
                "conditions_plotted": plotted_conditions,
            }
        )

    for ax in axes_flat[len(region_order):]:
        ax.set_axis_off()

    for row_axes in axes:
        row_axes[0].set_ylabel(str(settings.y_label), fontsize=10.0)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(str(settings.x_label), fontsize=10.0)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=max(1, len(legend_labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )

    fig.subplots_adjust(
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=float(settings.top_margin),
        bottom=float(settings.bottom_margin),
        wspace=float(settings.wspace),
        hspace=float(settings.hspace),
    )
    fig.patch.set_facecolor("white")
    save_figure(fig, out_path, ext=ext, dpi=dpi, facecolor="white")
    plt.close(fig)

    return {
        "output_path": str(out_path),
        "region_order": list(region_order),
        "condition_order": list(condition_order),
        "region_summaries": region_summaries,
    }
