"""Plot m1-m2 cross-correlation traces vs controls across scopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from dal_monte_2022_analysis.behav.plotting.cross_correlation_common import (
    downsample_indices,
    downsample_significance_mask,
    limit_true_markers,
    load_df_for_scope,
    load_lags_for_scope,
    nanmean_sem,
    paired_session_matrices,
    significance_mask_per_lag,
    scope_y_bounds,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.behav.plotting.common import apply_plotting_config, resolve_figsize
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    normalize_fix_cross_correlation_time_scope,
)


@dataclass
class M1M2CrossCorrComparisonPlotSettings:
    """Configuration for observed-vs-control m1-m2 cross-correlation plots."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "cross_correlation_outputs"
    fixation_label: str = "face"
    scopes: tuple[str, ...] = ("whole", "interactive", "non_interactive")
    significance_alpha: float = 0.05
    lag_sampling_rate_hz: float = 1000.0
    max_plot_points: int = 4000
    max_sig_markers: int = 1000
    rasterize_bands: bool = True
    rasterize_sig_markers: bool = True
    ttest_parallel: bool = True
    ttest_parallel_workers: int | None = None
    ttest_parallel_min_lags: int = 4000
    ttest_parallel_chunk_size: int = 4096
    output_subdir: str = "plots/m1-m2"
    observed_vs_cross_filename: str = "observed_vs_cross_session_face_m1_m2_crosscorr.pdf"
    observed_vs_shuffle_filename: str = "observed_vs_shuffle_face_m1_m2_crosscorr.pdf"


def _pretty_scope(scope: str) -> str:
    """Format scope names for subplot titles."""
    if scope == "non_interactive":
        return "non-interactive"
    return scope


def _append_filename_suffix(filename: str, suffix: str) -> str:
    """Append a suffix before extension in a filename."""
    path = Path(str(filename))
    extension = path.suffix if path.suffix else ".pdf"
    return f"{path.stem}_{suffix}{extension}"


def _resolve_single_panel_figsize(plot_cfg: dict) -> list[float]:
    """Resolve single-panel figure size defaults for m1-m2 plots."""
    figsize = plot_cfg.get("crosscorr_m1_m2_single_panel_figsize", [4.25, 2.6])
    if not isinstance(figsize, (list, tuple)) or len(figsize) < 2:
        raise RuntimeError(
            "crosscorr_m1_m2_single_panel_figsize must be a 2-element list/tuple [width, height]."
        )
    width = float(figsize[0])
    height = float(figsize[1])
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"crosscorr_m1_m2_single_panel_figsize must be positive, got [{width}, {height}]."
        )
    if width <= height:
        height = 0.72 * width
    return [width, height]


def _scale_figsize_2d(figsize: list[float], width_scale: float, height_scale: float) -> list[float]:
    """Scale figure width and height."""
    return [float(figsize[0]) * float(width_scale), float(figsize[1]) * float(height_scale)]


def _fill_sem_band(
    ax,
    *,
    x: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    """Draw SEM band as explicit polygon(s) for cleaner vector editing."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    sem = np.asarray(sem, dtype=np.float64).reshape(-1)
    if x.size == 0 or mean.size != x.size or sem.size != x.size:
        return

    y_lo = mean - sem
    y_hi = mean + sem
    finite_mask = np.isfinite(x) & np.isfinite(y_lo) & np.isfinite(y_hi)
    finite_idx = np.flatnonzero(finite_mask)
    if finite_idx.size < 2:
        return

    split_points = np.where(np.diff(finite_idx) > 1)[0]
    run_starts = np.r_[0, split_points + 1]
    run_stops = np.r_[split_points + 1, finite_idx.size]

    drew_any = False
    for start, stop in zip(run_starts, run_stops):
        run = finite_idx[int(start) : int(stop)]
        if run.size < 2:
            continue
        xs = x[run]
        upper = y_hi[run]
        lower = y_lo[run]
        poly_x = np.r_[xs, xs[::-1]]
        poly_y = np.r_[upper, lower[::-1]]
        ax.fill(
            poly_x,
            poly_y,
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=float(alpha),
            rasterized=False,
            zorder=1.6,
        )
        drew_any = True

    if drew_any:
        return

    # Fallback path for sparse/degenerate finite masks where polygon runs do not
    # meet the length threshold; keeps shading visible rather than dropping it.
    valid = np.isfinite(x) & np.isfinite(y_lo) & np.isfinite(y_hi)
    if np.count_nonzero(valid) >= 2:
        ax.fill_between(
            x,
            y_lo,
            y_hi,
            where=valid,
            color=color,
            alpha=float(alpha),
            linewidth=0.0,
            interpolate=False,
            rasterized=False,
            zorder=1.6,
        )


def _plot_scope_trace(
    *,
    ax,
    lags: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    observed_label: str,
    control_label: str,
    alpha: float,
    color_observed: str,
    color_control: str,
    max_plot_points: int,
    max_sig_markers: int,
    rasterize_bands: bool,
    rasterize_sig_markers: bool,
    ttest_parallel: bool,
    ttest_parallel_workers: int | None,
    ttest_parallel_min_lags: int,
    ttest_parallel_chunk_size: int,
    y_sig: float,
    observed_lw: float = 1.7,
    control_lw: float = 1.5,
    observed_ls: str = "-",
    control_ls: str = ":",
) -> None:
    """Render one observed-vs-control trace pair on one axis."""
    obs_mean, obs_sem = nanmean_sem(observed)
    ctl_mean, ctl_sem = nanmean_sem(control)
    sig = significance_mask_per_lag(
        observed,
        control,
        alpha=alpha,
        parallel=ttest_parallel,
        workers=ttest_parallel_workers,
        min_lags_for_parallel=ttest_parallel_min_lags,
        chunk_size=ttest_parallel_chunk_size,
    )

    idx = downsample_indices(int(lags.size), int(max_plot_points))
    lags_plot = lags[idx]
    obs_mean_plot = obs_mean[idx]
    obs_sem_plot = obs_sem[idx]
    ctl_mean_plot = ctl_mean[idx]
    ctl_sem_plot = ctl_sem[idx]
    sig_plot = downsample_significance_mask(sig, idx)
    sig_plot = limit_true_markers(sig_plot, int(max_sig_markers))

    ax.plot(
        lags_plot,
        obs_mean_plot,
        color=color_observed,
        lw=float(observed_lw),
        ls=observed_ls,
        label=observed_label,
    )
    ax.plot(
        lags_plot,
        ctl_mean_plot,
        color=color_control,
        lw=float(control_lw),
        ls=control_ls,
        label=control_label,
    )
    _fill_sem_band(
        ax,
        x=lags_plot,
        mean=obs_mean_plot,
        sem=obs_sem_plot,
        color=color_observed,
        alpha=0.18,
    )
    _fill_sem_band(
        ax,
        x=lags_plot,
        mean=ctl_mean_plot,
        sem=ctl_sem_plot,
        color=color_control,
        alpha=0.14,
    )
    if np.any(sig_plot):
        sig_x = lags_plot[sig_plot]
        sig_label = f"paired t-test p<{alpha:.02f}"
        # Draw one marker artist per significant bin so each marker is independently
        # editable/manipulable in vector editors (e.g., Illustrator).
        for idx, x_val in enumerate(sig_x):
            ax.plot(
                [float(x_val)],
                [float(y_sig)],
                marker="D",
                ms=3.8,
                mfc="white",
                mec=color_observed,
                mew=0.9,
                linestyle="None",
                label=sig_label if idx == 0 else "_nolegend_",
                zorder=5,
                rasterized=False,
            )


def _plot_observed_vs_control(
    settings: M1M2CrossCorrComparisonPlotSettings,
    *,
    control_kind: str,
    control_col: str,
    control_label: str,
    output_filename: str,
) -> list[Path]:
    """Create two figures: whole-only and interactive/non-interactive overlay."""
    cfg = load_config(settings.cfg_path)
    plot_cfg = load_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)

    out_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)
    scopes = tuple(dict.fromkeys(normalize_fix_cross_correlation_time_scope(scope) for scope in settings.scopes))
    required_scopes = ("whole", "interactive", "non_interactive")
    missing_scopes = sorted(set(required_scopes).difference(scopes))
    if missing_scopes:
        raise RuntimeError(
            f"Missing required scopes for plotting: {missing_scopes}. "
            f"Found scopes={list(scopes)}."
        )
    if float(settings.lag_sampling_rate_hz) <= 0:
        raise RuntimeError(
            f"lag_sampling_rate_hz must be > 0, got {settings.lag_sampling_rate_hz}."
        )

    _, dpi = resolve_figsize(plot_cfg)
    single_panel_figsize = _resolve_single_panel_figsize(plot_cfg)
    if dpi is None:
        dpi = 300

    colors = plot_cfg.get("crosscorr_m1_m2_colors", {})
    whole_color = colors.get("whole", "#d62728")
    whole_control_color = colors.get("whole_control", "#4A4A4A")
    interactive_color = colors.get("interactive", "#6A3D9A")
    non_interactive_color = colors.get("non_interactive", "#B8860B")

    panel_rows: dict[str, dict] = {}
    for scope in required_scopes:
        lags = load_lags_for_scope(out_dir, fixation_label=settings.fixation_label, scope=scope)
        lags_seconds = np.asarray(lags, dtype=np.float64) / float(settings.lag_sampling_rate_hz)
        within_df = load_df_for_scope(
            out_dir,
            fixation_label=settings.fixation_label,
            scope=scope,
            kind="within",
        )
        control_df = load_df_for_scope(
            out_dir,
            fixation_label=settings.fixation_label,
            scope=scope,
            kind=control_kind,
        )
        observed, control, n_obs_total, n_ctl_total, n_paired = paired_session_matrices(
            within_df,
            control_df,
            control_col=control_col,
        )
        if observed.shape[1] != lags_seconds.size:
            raise RuntimeError(
                f"Lag length mismatch for scope='{scope}': "
                f"lags={lags_seconds.size}, observed={observed.shape[1]}"
            )
        panel_rows[scope] = {
            "scope": scope,
            "lags_seconds": lags_seconds,
            "observed": observed,
            "control": control,
            "n_obs_total": n_obs_total,
            "n_ctl_total": n_ctl_total,
            "n_paired": n_paired,
        }

    plot_dir = out_dir / settings.output_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    group_specs: list[tuple[str, list[str], str]] = [
        (output_filename, ["whole"], "Whole Session"),
        (
            _append_filename_suffix(output_filename, "interactive_non_interactive"),
            ["interactive", "non_interactive"],
            "Interactive + Non-Interactive",
        ),
    ]
    observed_colors = {
        "whole": whole_color,
        "interactive": interactive_color,
        "non_interactive": non_interactive_color,
    }
    control_colors = {
        "whole": whole_control_color,
        "interactive": interactive_color,
        "non_interactive": non_interactive_color,
    }

    for group_filename, group_scopes, title_prefix in group_specs:
        is_overlay_group = len(group_scopes) > 1
        group_panels = [panel_rows[scope] for scope in group_scopes]
        group_y_min = np.inf
        group_y_max = -np.inf
        for panel in group_panels:
            y_lo, y_hi = scope_y_bounds(panel["observed"], panel["control"])
            group_y_min = min(group_y_min, y_lo)
            group_y_max = max(group_y_max, y_hi)
        if not np.isfinite(group_y_min) or not np.isfinite(group_y_max):
            group_y_min, group_y_max = -1.0, 1.0
        if group_y_max <= group_y_min:
            group_y_max = group_y_min + 1e-6

        group_figsize = (
            _scale_figsize_2d(single_panel_figsize, 1.30, 1.12)
            if is_overlay_group
            else list(single_panel_figsize)
        )
        fig, ax = plt.subplots(1, 1, figsize=group_figsize, dpi=dpi)
        y_range = group_y_max - group_y_min
        y_pad = 0.10 * y_range
        sig_step = 0.14 * y_range
        for idx, panel in enumerate(group_panels):
            scope = panel["scope"]
            scope_pretty = _pretty_scope(scope)
            obs_color = observed_colors[scope]
            ctl_color = control_colors[scope]
            is_overlay_scope = scope in ("interactive", "non_interactive")
            _plot_scope_trace(
                ax=ax,
                lags=panel["lags_seconds"],
                observed=panel["observed"],
                control=panel["control"],
                observed_label=f"{scope_pretty} observed",
                control_label=f"{scope_pretty} {control_label}",
                alpha=settings.significance_alpha,
                color_observed=obs_color,
                color_control=ctl_color,
                max_plot_points=settings.max_plot_points,
                max_sig_markers=settings.max_sig_markers,
                rasterize_bands=settings.rasterize_bands,
                rasterize_sig_markers=settings.rasterize_sig_markers,
                ttest_parallel=settings.ttest_parallel,
                ttest_parallel_workers=settings.ttest_parallel_workers,
                ttest_parallel_min_lags=settings.ttest_parallel_min_lags,
                ttest_parallel_chunk_size=settings.ttest_parallel_chunk_size,
                y_sig=group_y_max + 0.40 * y_pad + idx * sig_step,
                observed_lw=1.8,
                control_lw=1.0 if is_overlay_scope else 1.4,
                observed_ls="-",
                control_ls=":",
            )
        ax.axvline(0, color="#444444", lw=0.8, ls="--", alpha=0.8)
        ax.set_ylim(group_y_min - y_pad, group_y_max + 2.2 * y_pad + (len(group_panels) - 1) * sig_step)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Normalized cross-correlation")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if is_overlay_group:
            # Keep ~7 major x ticks (including 0) and 3-4 y ticks for cleaner readability.
            max_abs_lag = max(abs(float(panel["lags_seconds"][0])) for panel in group_panels)
            max_abs_lag = max(
                max_abs_lag,
                max(abs(float(panel["lags_seconds"][-1])) for panel in group_panels),
            )
            if np.isfinite(max_abs_lag) and max_abs_lag > 0:
                ax.set_xlim(-max_abs_lag, max_abs_lag)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=7))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        if len(group_panels) == 1:
            panel = group_panels[0]
            subtitle = (
                f"n_paired={panel['n_paired']} "
                f"(obs={panel['n_obs_total']}, ctrl={panel['n_ctl_total']})"
            )
        else:
            subtitle = "; ".join(
                f"{_pretty_scope(panel['scope'])}: n_paired={panel['n_paired']}"
                for panel in group_panels
            )
        ax.set_title(f"{title_prefix}\n{subtitle}", fontsize=10)

        handles, labels = ax.get_legend_handles_labels()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label in unique_labels:
                continue
            unique_labels.append(label)
            unique_handles.append(handle)
        if unique_handles:
            legend_ncols = min(3 if is_overlay_group else 2, len(unique_handles))
            ax.legend(
                unique_handles,
                unique_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.28 if is_overlay_group else 1.23),
                ncol=legend_ncols,
                frameon=False,
                fontsize=8,
            )

        top_rect = 0.86 if is_overlay_group else 0.90
        fig.tight_layout(rect=[0, 0, 1, top_rect])
        out_path = plot_dir / group_filename
        save_figure(fig, out_path, ext="pdf")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def plot_observed_vs_cross_session_m1_m2(
    settings: M1M2CrossCorrComparisonPlotSettings,
) -> list[Path]:
    """Plot observed-vs-cross-session comparison as whole-only and interactive overlays."""
    return _plot_observed_vs_control(
        settings,
        control_kind="cross",
        control_col="cross_correlation_mean",
        control_label="Cross-session control",
        output_filename=settings.observed_vs_cross_filename,
    )


def plot_observed_vs_shuffle_m1_m2(
    settings: M1M2CrossCorrComparisonPlotSettings,
) -> list[Path]:
    """Plot observed-vs-shuffle comparison as whole-only and interactive overlays."""
    return _plot_observed_vs_control(
        settings,
        control_kind="shuffle",
        control_col="cross_correlation_shuffle_mean",
        control_label="Shuffle control",
        output_filename=settings.observed_vs_shuffle_filename,
    )
