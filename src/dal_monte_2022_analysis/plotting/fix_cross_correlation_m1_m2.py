"""Plot m1-m2 cross-correlation traces vs controls across scopes."""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from dal_monte_2022_analysis.config.load import load_dataset_config, load_plotting_config
from dal_monte_2022_analysis.plotting.common import apply_plotting_config, resolve_figsize
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    build_fix_crosscorr_output_filename,
    normalize_fix_crosscorr_time_scope,
)


@dataclass
class M1M2CrossCorrComparisonPlotSettings:
    """Configuration for observed-vs-control m1-m2 cross-correlation plots."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "crosscorr_outputs"
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


def _load_pickle(path: Path):
    """Load pickle object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_lags_for_scope(
    out_dir: Path,
    *,
    fixation_label: str,
    scope: str,
) -> np.ndarray:
    """Load lag axis for one scope."""
    lags_path = out_dir / build_fix_crosscorr_output_filename(
        fixation_label,
        "lags",
        time_scope=scope,
    )
    if not lags_path.exists():
        raise FileNotFoundError(f"Missing lag file for scope='{scope}': {lags_path}")
    lags = np.asarray(_load_pickle(lags_path), dtype=np.int64).reshape(-1)
    if lags.size == 0:
        raise RuntimeError(f"Lag file is empty for scope='{scope}': {lags_path}")
    return lags


def _load_df_for_scope(
    out_dir: Path,
    *,
    fixation_label: str,
    scope: str,
    kind: str,
) -> pd.DataFrame:
    """Load cross-correlation dataframe for one scope and output kind."""
    data_path = out_dir / build_fix_crosscorr_output_filename(
        fixation_label,
        kind,
        time_scope=scope,
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {kind} file for scope='{scope}': {data_path}")
    return pd.read_pickle(data_path)


def _as_1d_float(arr) -> np.ndarray:
    """Coerce array-like to 1D float ndarray."""
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _stack_column_arrays(df: pd.DataFrame, col: str) -> np.ndarray:
    """Stack fixed-length arrays from one dataframe column into 2D matrix."""
    mats = [_as_1d_float(v) for v in df[col].to_list()]
    if not mats:
        return np.empty((0, 0), dtype=np.float64)
    n_lags = mats[0].size
    for idx, row in enumerate(mats):
        if row.size != n_lags:
            raise RuntimeError(
                f"Array-length mismatch at row={idx} col={col}: {row.size} != {n_lags}"
            )
    return np.vstack(mats)


def _paired_session_matrices(
    within_df: pd.DataFrame,
    control_df: pd.DataFrame,
    *,
    control_col: str,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Align within/control by (date, session) and return paired matrices."""
    within_cols = ["date", "session", "cross_correlation"]
    control_cols = ["date", "session", control_col]
    missing_within = set(within_cols).difference(within_df.columns)
    missing_control = set(control_cols).difference(control_df.columns)
    if missing_within:
        raise RuntimeError(f"Within-session table missing columns: {sorted(missing_within)}")
    if missing_control:
        raise RuntimeError(f"Control table missing columns: {sorted(missing_control)}")

    merged = within_df[within_cols].merge(
        control_df[control_cols],
        how="inner",
        on=["date", "session"],
    )
    if merged.empty:
        raise RuntimeError("No overlapping (date, session) rows between observed and control tables.")

    observed = _stack_column_arrays(merged, "cross_correlation")
    control = _stack_column_arrays(merged, control_col)
    if observed.shape != control.shape:
        raise RuntimeError(
            "Observed/control paired matrices have different shapes: "
            f"{observed.shape} vs {control.shape}"
        )
    return observed, control, int(len(within_df)), int(len(control_df)), int(len(merged))


def _nanmean_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-column mean and SEM with NaN handling."""
    if mat.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    mean = np.nanmean(mat, axis=0)
    finite_counts = np.sum(np.isfinite(mat), axis=0)
    std = np.nanstd(mat, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = std / np.sqrt(finite_counts)
    sem[finite_counts < 2] = np.nan
    return mean, sem


def _downsample_indices(n_points: int, max_points: int) -> np.ndarray:
    """Return monotonic indices for plotting downsampling."""
    n = int(max(0, n_points))
    if n == 0:
        return np.asarray([], dtype=np.int64)
    cap = int(max(1, max_points))
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    step = int(np.ceil(n / float(cap)))
    return np.arange(0, n, step, dtype=np.int64)


def _downsample_significance_mask(sig_full: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Collapse a full-resolution significance mask onto downsampled indices."""
    if idx.size == 0:
        return np.asarray([], dtype=bool)
    out = np.zeros(idx.size, dtype=bool)
    n = int(sig_full.size)
    for i, start in enumerate(idx):
        stop = int(idx[i + 1]) if i + 1 < idx.size else n
        out[i] = bool(np.any(sig_full[int(start) : stop]))
    return out


def _limit_true_markers(mask: np.ndarray, max_true: int) -> np.ndarray:
    """Cap number of True markers by uniform subsampling over True positions."""
    out = np.asarray(mask, dtype=bool).copy()
    cap = int(max_true)
    if cap <= 0:
        out[:] = False
        return out
    true_idx = np.flatnonzero(out)
    if true_idx.size <= cap:
        return out
    keep = np.linspace(0, true_idx.size - 1, num=cap, dtype=int)
    keep_idx = true_idx[keep]
    out[:] = False
    out[keep_idx] = True
    return out


def _paired_ttest_per_lag_chunk(
    observed: np.ndarray,
    control: np.ndarray,
    *,
    start: int,
    stop: int,
) -> tuple[int, np.ndarray]:
    """Compute paired t-test p-values for one [start:stop) lag chunk."""
    x = observed[:, start:stop]
    y = control[:, start:stop]
    pvals = np.asarray(
        ttest_rel(x, y, axis=0, nan_policy="omit").pvalue,
        dtype=np.float64,
    ).reshape(-1)
    valid_counts = np.sum(np.isfinite(x) & np.isfinite(y), axis=0)
    pvals[valid_counts < 2] = np.nan
    return start, pvals


def _paired_ttest_per_lag(
    observed: np.ndarray,
    control: np.ndarray,
    *,
    parallel: bool,
    workers: int | None,
    min_lags_for_parallel: int,
    chunk_size: int,
) -> np.ndarray:
    """Compute per-lag paired t-test p-values (optionally in parallel chunks)."""
    if observed.shape != control.shape:
        raise ValueError("Observed and control matrices must have same shape.")
    n_lags = observed.shape[1]
    if n_lags <= 0:
        return np.array([], dtype=np.float64)

    if (
        not parallel
        or n_lags < int(max(1, min_lags_for_parallel))
        or int(max(1, chunk_size)) >= n_lags
    ):
        pvals = np.asarray(
            ttest_rel(observed, control, axis=0, nan_policy="omit").pvalue,
            dtype=np.float64,
        ).reshape(-1)
        valid_counts = np.sum(np.isfinite(observed) & np.isfinite(control), axis=0)
        pvals[valid_counts < 2] = np.nan
        return pvals

    chunk = int(max(1, chunk_size))
    starts = list(range(0, n_lags, chunk))
    auto_workers = os.cpu_count() or 1
    n_workers = int(max(1, workers if workers is not None else auto_workers))
    n_workers = min(n_workers, len(starts))
    if n_workers <= 1:
        pvals = np.full(n_lags, np.nan, dtype=np.float64)
        for start in starts:
            stop = min(start + chunk, n_lags)
            _, chunk_p = _paired_ttest_per_lag_chunk(
                observed,
                control,
                start=start,
                stop=stop,
            )
            pvals[start:stop] = chunk_p
        return pvals

    pvals = np.full(n_lags, np.nan, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for start in starts:
            stop = min(start + chunk, n_lags)
            futures.append(
                executor.submit(
                    _paired_ttest_per_lag_chunk,
                    observed,
                    control,
                    start=start,
                    stop=stop,
                )
            )
        for future in futures:
            start, chunk_p = future.result()
            stop = start + int(chunk_p.size)
            pvals[start:stop] = chunk_p
    return pvals


def _pretty_scope(scope: str) -> str:
    """Format scope names for subplot titles."""
    if scope == "non_interactive":
        return "non-interactive"
    return scope


def _plot_one_scope(
    *,
    ax,
    lags: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    scope: str,
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
    n_obs_total: int,
    n_ctl_total: int,
    n_paired: int,
) -> None:
    """Render one scope panel."""
    obs_mean, obs_sem = _nanmean_sem(observed)
    ctl_mean, ctl_sem = _nanmean_sem(control)
    pvals = _paired_ttest_per_lag(
        observed,
        control,
        parallel=ttest_parallel,
        workers=ttest_parallel_workers,
        min_lags_for_parallel=ttest_parallel_min_lags,
        chunk_size=ttest_parallel_chunk_size,
    )
    sig = np.isfinite(pvals) & (pvals < float(alpha))

    idx = _downsample_indices(int(lags.size), int(max_plot_points))
    lags_plot = lags[idx]
    obs_mean_plot = obs_mean[idx]
    obs_sem_plot = obs_sem[idx]
    ctl_mean_plot = ctl_mean[idx]
    ctl_sem_plot = ctl_sem[idx]
    sig_plot = _downsample_significance_mask(sig, idx)
    sig_plot = _limit_true_markers(sig_plot, int(max_sig_markers))

    ax.plot(
        lags_plot,
        obs_mean_plot,
        color=color_observed,
        lw=1.7,
        label="Observed (within-session)",
    )
    ax.plot(lags_plot, ctl_mean_plot, color=color_control, lw=1.5, label=control_label)
    ax.fill_between(
        lags_plot,
        obs_mean_plot - obs_sem_plot,
        obs_mean_plot + obs_sem_plot,
        color=color_observed,
        alpha=0.20,
        rasterized=bool(rasterize_bands),
    )
    ax.fill_between(
        lags_plot,
        ctl_mean_plot - ctl_sem_plot,
        ctl_mean_plot + ctl_sem_plot,
        color=color_control,
        alpha=0.20,
        rasterized=bool(rasterize_bands),
    )
    ax.axvline(0, color="#444444", lw=0.8, ls="--", alpha=0.8)

    y_lo = float(np.nanmin(np.r_[obs_mean - obs_sem, ctl_mean - ctl_sem]))
    y_hi = float(np.nanmax(np.r_[obs_mean + obs_sem, ctl_mean + ctl_sem]))
    if not np.isfinite(y_lo) or not np.isfinite(y_hi):
        y_lo, y_hi = -1.0, 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-6
    y_pad = 0.08 * (y_hi - y_lo)
    y_sig = y_hi + 0.02 * (y_hi - y_lo)
    if np.any(sig_plot):
        ax.scatter(
            lags_plot[sig_plot],
            np.full(int(np.count_nonzero(sig_plot)), y_sig),
            marker="|",
            s=36,
            c="#111111",
            linewidths=0.9,
            label=f"paired t-test p<{alpha:.02f}",
            zorder=5,
            rasterized=bool(rasterize_sig_markers),
        )
    ax.set_ylim(y_lo - y_pad, y_hi + 2.5 * y_pad)

    ax.set_title(
        f"{_pretty_scope(scope)}\n"
        f"n_paired={n_paired} (obs={n_obs_total}, ctrl={n_ctl_total})",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Lag (s)")


def _plot_observed_vs_control(
    settings: M1M2CrossCorrComparisonPlotSettings,
    *,
    control_kind: str,
    control_col: str,
    control_label: str,
    output_filename: str,
) -> Path:
    """Create one 3-panel figure comparing observed trace to one control type."""
    cfg = load_dataset_config(settings.cfg_path)
    plot_cfg = load_plotting_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)

    out_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)
    scopes = tuple(normalize_fix_crosscorr_time_scope(scope) for scope in settings.scopes)
    if len(scopes) != 3:
        raise RuntimeError("Expected exactly 3 scopes for plotting (whole, interactive, non_interactive).")
    if float(settings.lag_sampling_rate_hz) <= 0:
        raise RuntimeError(
            f"lag_sampling_rate_hz must be > 0, got {settings.lag_sampling_rate_hz}."
        )

    figsize, dpi = resolve_figsize(plot_cfg)
    custom_figsize = plot_cfg.get("crosscorr_m1_m2_figsize")
    if custom_figsize is not None:
        figsize = custom_figsize
    if figsize is None or (len(figsize) >= 1 and float(figsize[0]) < 12.0):
        figsize = [16, 4.6]
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharey=False)

    colors = plot_cfg.get("crosscorr_m1_m2_colors", {})
    observed_color = colors.get("observed", "#1f77b4")
    control_color = colors.get("control", "#d62728")

    for ax, scope in zip(axes, scopes):
        lags = _load_lags_for_scope(out_dir, fixation_label=settings.fixation_label, scope=scope)
        lags_seconds = np.asarray(lags, dtype=np.float64) / float(settings.lag_sampling_rate_hz)
        within_df = _load_df_for_scope(
            out_dir,
            fixation_label=settings.fixation_label,
            scope=scope,
            kind="within",
        )
        control_df = _load_df_for_scope(
            out_dir,
            fixation_label=settings.fixation_label,
            scope=scope,
            kind=control_kind,
        )
        observed, control, n_obs_total, n_ctl_total, n_paired = _paired_session_matrices(
            within_df,
            control_df,
            control_col=control_col,
        )
        if observed.shape[1] != lags_seconds.size:
            raise RuntimeError(
                f"Lag length mismatch for scope='{scope}': "
                f"lags={lags_seconds.size}, observed={observed.shape[1]}"
            )
        _plot_one_scope(
            ax=ax,
            lags=lags_seconds,
            observed=observed,
            control=control,
            scope=scope,
            control_label=control_label,
            alpha=settings.significance_alpha,
            color_observed=observed_color,
            color_control=control_color,
            max_plot_points=settings.max_plot_points,
            max_sig_markers=settings.max_sig_markers,
            rasterize_bands=settings.rasterize_bands,
            rasterize_sig_markers=settings.rasterize_sig_markers,
            ttest_parallel=settings.ttest_parallel,
            ttest_parallel_workers=settings.ttest_parallel_workers,
            ttest_parallel_min_lags=settings.ttest_parallel_min_lags,
            ttest_parallel_chunk_size=settings.ttest_parallel_chunk_size,
            n_obs_total=n_obs_total,
            n_ctl_total=n_ctl_total,
            n_paired=n_paired,
        )

    axes[0].set_ylabel("Normalized cross-correlation")
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label in labels:
                continue
            labels.append(label)
            handles.append(handle)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(3, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    plot_dir = out_dir / settings.output_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / output_filename
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return out_path


def plot_observed_vs_cross_session_m1_m2(settings: M1M2CrossCorrComparisonPlotSettings) -> Path:
    """Plot observed-vs-cross-session comparison across whole/interactive/non-interactive."""
    return _plot_observed_vs_control(
        settings,
        control_kind="cross",
        control_col="cross_correlation_mean",
        control_label="Cross-session control",
        output_filename=settings.observed_vs_cross_filename,
    )


def plot_observed_vs_shuffle_m1_m2(settings: M1M2CrossCorrComparisonPlotSettings) -> Path:
    """Plot observed-vs-shuffle comparison across whole/interactive/non-interactive."""
    return _plot_observed_vs_control(
        settings,
        control_kind="shuffle",
        control_col="cross_correlation_shuffle_mean",
        control_label="Shuffle control",
        output_filename=settings.observed_vs_shuffle_filename,
    )
