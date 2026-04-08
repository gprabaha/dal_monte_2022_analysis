"""Plot saved face-fixation gap distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.behav.plotting.common import apply_plotting_config
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure


@dataclass
class FaceFixationGapDistributionPlotSettings:
    """Configuration for face-fixation gap distribution plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "face_fixation_gap_distributions"
    m1_input_filename: str = "within_session_m1_face_fixation_gap_distribution.csv"
    m1_m2_input_filename: str = (
        "within_session_interactive_m1_m2_face_fixation_gap_distribution.csv"
    )
    m1_output_filename: str = "m1_face_fixation_gap_distributions.pdf"
    m1_m2_output_filename: str = "interactive_m1_m2_face_fixation_gap_distributions.pdf"
    interactive_state_label: str = "interactive"
    non_interactive_state_label: str = "non_interactive"
    histogram_bins: int = 60


_DEFAULT_STYLE = {
    "m1_figsize": [9.2, 3.8],
    "m1_m2_figsize": [9.2, 3.8],
    "histogram_bins": 60,
    "density": True,
    "alpha": 0.42,
    "edgecolor": "#1F1F1F",
    "linewidth": 0.8,
    "colors": {
        "interactive": "#1b9e77",
        "non_interactive": "#7f7f7f",
        "cross": "#d95f02",
    },
    "zero_line": {
        "color": "#1A1A1A",
        "linewidth": 0.9,
        "linestyle": "--",
        "alpha": 0.8,
    },
}


def _resolve_style(plot_cfg: dict) -> dict:
    """Resolve plotting style with optional config overrides."""
    style = dict(_DEFAULT_STYLE)
    style["colors"] = dict(_DEFAULT_STYLE["colors"])
    style["zero_line"] = dict(_DEFAULT_STYLE["zero_line"])

    override = plot_cfg.get("face_fixation_gap_distribution", {})
    if not isinstance(override, dict):
        return style

    for key in (
        "m1_figsize",
        "m1_m2_figsize",
        "histogram_bins",
        "density",
        "alpha",
        "edgecolor",
        "linewidth",
    ):
        if key in override:
            style[key] = override[key]

    for key in ("colors", "zero_line"):
        if key in override and isinstance(override[key], dict):
            merged = dict(style[key])
            merged.update(override[key])
            style[key] = merged
    return style


def _finite_values(df: pd.DataFrame, *, gap_metric: str) -> np.ndarray:
    """Return finite gap values for one gap metric."""
    if df.empty or "gap_metric" not in df.columns or "gap_ms" not in df.columns:
        return np.asarray([], dtype=float)
    values = pd.to_numeric(
        df.loc[df["gap_metric"] == gap_metric, "gap_ms"],
        errors="coerce",
    ).to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _build_histogram_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Build stable histogram bins for one set of values."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.linspace(-0.5, 0.5, 2)

    value_min = float(np.min(arr))
    value_max = float(np.max(arr))
    if np.isclose(value_min, value_max):
        pad = max(0.5, abs(value_min) * 0.05)
        return np.linspace(value_min - pad, value_max + pad, 3)
    return np.linspace(value_min, value_max, int(max(2, n_bins)) + 1)


def _format_panel(
    ax,
    *,
    title: str,
    x_label: str,
    y_label: str,
    zero_line_cfg: dict,
) -> None:
    """Apply common axis formatting."""
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axvline(
        0.0,
        color=str(zero_line_cfg.get("color", "#1A1A1A")),
        linewidth=float(zero_line_cfg.get("linewidth", 0.9)),
        linestyle=str(zero_line_cfg.get("linestyle", "--")),
        alpha=float(zero_line_cfg.get("alpha", 0.8)),
        zorder=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_m1_panel(
    ax,
    *,
    df: pd.DataFrame,
    gap_metric: str,
    settings: FaceFixationGapDistributionPlotSettings,
    style: dict,
    title: str,
) -> None:
    """Plot one m1 distribution panel with interactive-state overlays."""
    if "gap_metric" not in df.columns:
        panel_df = pd.DataFrame(columns=df.columns)
    else:
        panel_df = df[df["gap_metric"] == gap_metric].copy()
    values = _finite_values(panel_df, gap_metric=gap_metric)
    bins = _build_histogram_bins(values, int(style["histogram_bins"]))

    state_order = [
        str(settings.interactive_state_label),
        str(settings.non_interactive_state_label),
    ]
    state_labels = {
        str(settings.interactive_state_label): "Interactive",
        str(settings.non_interactive_state_label): "Non-interactive",
    }
    colors = style["colors"]
    plotted = False
    for state in state_order:
        if "period_state" not in panel_df.columns:
            subset = pd.DataFrame(columns=panel_df.columns)
        else:
            subset = panel_df[panel_df["period_state"] == state]
        subset_vals = _finite_values(subset, gap_metric=gap_metric)
        if subset_vals.size == 0:
            continue
        plotted = True
        ax.hist(
            subset_vals,
            bins=bins,
            density=bool(style["density"]),
            alpha=float(style["alpha"]),
            color=colors.get(state, "#808080"),
            edgecolor=str(style["edgecolor"]),
            linewidth=float(style["linewidth"]),
            label=state_labels.get(state, state),
        )

    if not plotted:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    _format_panel(
        ax,
        title=title,
        x_label="Gap (ms)",
        y_label="Density" if bool(style["density"]) else "Count",
        zero_line_cfg=style["zero_line"],
    )


def _plot_cross_panel(
    ax,
    *,
    df: pd.DataFrame,
    gap_metric: str,
    style: dict,
    title: str,
) -> None:
    """Plot one interactive cross-monkey distribution panel."""
    values = _finite_values(df, gap_metric=gap_metric)
    bins = _build_histogram_bins(values, int(style["histogram_bins"]))
    if values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(
            values,
            bins=bins,
            density=bool(style["density"]),
            alpha=float(style["alpha"]),
            color=style["colors"].get("cross", "#d95f02"),
            edgecolor=str(style["edgecolor"]),
            linewidth=float(style["linewidth"]),
        )

    _format_panel(
        ax,
        title=title,
        x_label="Gap (ms)",
        y_label="Density" if bool(style["density"]) else "Count",
        zero_line_cfg=style["zero_line"],
    )


def plot_face_fixation_gap_distribution_figures(
    settings: FaceFixationGapDistributionPlotSettings,
) -> tuple[Path, Path]:
    """Plot the two requested face-fixation gap distribution figures."""
    cfg = load_config(settings.cfg_path)
    plot_cfg = apply_plotting_config(load_config(settings.plotting_cfg_path))
    style = _resolve_style(plot_cfg)
    analysis_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)

    m1_path = analysis_dir / settings.m1_input_filename
    m1_m2_path = analysis_dir / settings.m1_m2_input_filename
    if not m1_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {m1_path}")
    if not m1_m2_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {m1_m2_path}")

    m1_df = pd.read_csv(m1_path)
    m1_m2_df = pd.read_csv(m1_m2_path)
    dpi = plot_cfg.get("figure", {}).get("dpi")

    fig_m1, axes_m1 = plt.subplots(
        1,
        2,
        figsize=tuple(float(value) for value in style["m1_figsize"]),
        constrained_layout=True,
    )
    _plot_m1_panel(
        axes_m1[0],
        df=m1_df,
        gap_metric="start_to_start",
        settings=settings,
        style=style,
        title="m1 face: start to start",
    )
    _plot_m1_panel(
        axes_m1[1],
        df=m1_df,
        gap_metric="stop_to_start",
        settings=settings,
        style=style,
        title="m1 face: prev stop to next start",
    )
    handles, labels = axes_m1[0].get_legend_handles_labels()
    if handles:
        axes_m1[0].legend(frameon=False, loc="upper right")
    m1_out = save_figure(
        fig_m1,
        analysis_dir / settings.m1_output_filename,
        dpi=dpi,
    )
    plt.close(fig_m1)

    fig_m1_m2, axes_m1_m2 = plt.subplots(
        1,
        2,
        figsize=tuple(float(value) for value in style["m1_m2_figsize"]),
        constrained_layout=True,
    )
    _plot_cross_panel(
        axes_m1_m2[0],
        df=m1_m2_df,
        gap_metric="start_to_start",
        style=style,
        title="Interactive m1/m2 face: start to start",
    )
    _plot_cross_panel(
        axes_m1_m2[1],
        df=m1_m2_df,
        gap_metric="stop_to_start",
        style=style,
        title="Interactive m1/m2 face: prev stop to next start",
    )
    m1_m2_out = save_figure(
        fig_m1_m2,
        analysis_dir / settings.m1_m2_output_filename,
        dpi=dpi,
    )
    plt.close(fig_m1_m2)
    return m1_out, m1_m2_out
