"""Plot m1-m2 cross-correlation traces vs controls across scopes."""

from __future__ import annotations

import pickle
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


def _paired_ttest_per_lag(observed: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Compute per-lag paired t-test p-values."""
    if observed.shape != control.shape:
        raise ValueError("Observed and control matrices must have same shape.")
    n_lags = observed.shape[1]
    pvals = np.full(n_lags, np.nan, dtype=np.float64)
    for idx in range(n_lags):
        x = observed[:, idx]
        y = control[:, idx]
        valid = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid) < 2:
            continue
        pvals[idx] = float(ttest_rel(x[valid], y[valid]).pvalue)
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
    n_obs_total: int,
    n_ctl_total: int,
    n_paired: int,
) -> None:
    """Render one scope panel."""
    obs_mean, obs_sem = _nanmean_sem(observed)
    ctl_mean, ctl_sem = _nanmean_sem(control)
    pvals = _paired_ttest_per_lag(observed, control)
    sig = np.isfinite(pvals) & (pvals < float(alpha))

    ax.plot(lags, obs_mean, color=color_observed, lw=1.7, label="Observed (within-session)")
    ax.plot(lags, ctl_mean, color=color_control, lw=1.5, label=control_label)
    ax.fill_between(lags, obs_mean - obs_sem, obs_mean + obs_sem, color=color_observed, alpha=0.20)
    ax.fill_between(lags, ctl_mean - ctl_sem, ctl_mean + ctl_sem, color=color_control, alpha=0.20)
    ax.axvline(0, color="#444444", lw=0.8, ls="--", alpha=0.8)

    y_lo = float(np.nanmin(np.r_[obs_mean - obs_sem, ctl_mean - ctl_sem]))
    y_hi = float(np.nanmax(np.r_[obs_mean + obs_sem, ctl_mean + ctl_sem]))
    if not np.isfinite(y_lo) or not np.isfinite(y_hi):
        y_lo, y_hi = -1.0, 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-6
    y_pad = 0.08 * (y_hi - y_lo)
    y_sig = y_hi + 0.02 * (y_hi - y_lo)
    if np.any(sig):
        ax.scatter(
            lags[sig],
            np.full(int(np.count_nonzero(sig)), y_sig),
            marker="|",
            s=36,
            c="#111111",
            linewidths=0.9,
            label=f"paired t-test p<{alpha:.02f}",
            zorder=5,
        )
    ax.set_ylim(y_lo - y_pad, y_hi + 2.5 * y_pad)

    ax.set_title(
        f"{_pretty_scope(scope)}\n"
        f"n_paired={n_paired} (obs={n_obs_total}, ctrl={n_ctl_total})",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Lag (samples)")


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

    figsize, dpi = resolve_figsize(plot_cfg)
    custom_figsize = plot_cfg.get("crosscorr_m1_m2_figsize")
    if custom_figsize is not None:
        figsize = custom_figsize
    if figsize is None or (len(figsize) >= 1 and float(figsize[0]) < 12.0):
        figsize = [16, 4.6]
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharey=True)

    colors = plot_cfg.get("crosscorr_m1_m2_colors", {})
    observed_color = colors.get("observed", "#1f77b4")
    control_color = colors.get("control", "#d62728")

    for ax, scope in zip(axes, scopes):
        lags = _load_lags_for_scope(out_dir, fixation_label=settings.fixation_label, scope=scope)
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
        if observed.shape[1] != lags.size:
            raise RuntimeError(
                f"Lag length mismatch for scope='{scope}': "
                f"lags={lags.size}, observed={observed.shape[1]}"
            )
        _plot_one_scope(
            ax=ax,
            lags=lags,
            observed=observed,
            control=control,
            scope=scope,
            control_label=control_label,
            alpha=settings.significance_alpha,
            color_observed=observed_color,
            color_control=control_color,
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
