"""Plot leader-aligned cross-correlation traces vs controls across scopes."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.behav.plotting.common import apply_plotting_config, resolve_figsize
from dal_monte_2022_analysis.utils.io import load_pickle
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    build_fix_crosscorr_output_filename,
    normalize_fix_crosscorr_time_scope,
)

_VALID_LEADER_BASES = ("session", "date", "pair")


@dataclass
class LeaderFollowerCrossCorrComparisonPlotSettings:
    """Configuration for leader-aligned cross-correlation comparison plots."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    crosscorr_analysis_subdir: str = "crosscorr_outputs"
    leader_follower_subdir: str = "crosscorr_outputs/leader_follower"
    fixation_label: str = "face"
    scopes: tuple[str, ...] = ("whole", "interactive", "non_interactive")
    leader_bases: tuple[str, ...] = ("session", "date", "pair")
    leader_reference_scope: str = "whole"
    leader_session_filename: str = "within_session_face_fix_crosscorr_leader_follower.pkl"
    leader_date_filename: str = "date_summary_face_fix_crosscorr_leader_follower.pkl"
    leader_pair_filename: str = "pair_summary_face_fix_crosscorr_leader_follower.pkl"
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
    output_subdir: str = "plots/leader_follower"
    observed_vs_cross_filename_template: str = (
        "observed_vs_cross_session_face_leader_follower_basis={basis}.pdf"
    )
    observed_vs_shuffle_filename_template: str = (
        "observed_vs_shuffle_face_leader_follower_basis={basis}.pdf"
    )


_load_pickle = load_pickle


def _load_lags_for_scope(
    out_dir: Path,
    *,
    fixation_label: str,
    scope: str,
) -> np.ndarray:
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
    data_path = out_dir / build_fix_crosscorr_output_filename(
        fixation_label,
        kind,
        time_scope=scope,
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {kind} file for scope='{scope}': {data_path}")
    return pd.read_pickle(data_path)


def _as_1d_float(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _paired_ttest_per_lag_chunk(
    observed: np.ndarray,
    control: np.ndarray,
    *,
    start: int,
    stop: int,
) -> tuple[int, np.ndarray]:
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
    pvals = np.full(n_lags, np.nan, dtype=np.float64)
    if n_workers <= 1:
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


def _nanmean_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    n = int(max(0, n_points))
    if n == 0:
        return np.asarray([], dtype=np.int64)
    cap = int(max(1, max_points))
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    step = int(np.ceil(n / float(cap)))
    return np.arange(0, n, step, dtype=np.int64)


def _downsample_significance_mask(sig_full: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        return np.asarray([], dtype=bool)
    out = np.zeros(idx.size, dtype=bool)
    n = int(sig_full.size)
    for i, start in enumerate(idx):
        stop = int(idx[i + 1]) if i + 1 < idx.size else n
        out[i] = bool(np.any(sig_full[int(start) : stop]))
    return out


def _limit_true_markers(mask: np.ndarray, max_true: int) -> np.ndarray:
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


def _scope_y_bounds(observed: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    """Return y-bounds from mean +/- SEM envelopes for one scope."""
    obs_mean, obs_sem = _nanmean_sem(observed)
    ctl_mean, ctl_sem = _nanmean_sem(control)
    y_lo = float(np.nanmin(np.r_[obs_mean - obs_sem, ctl_mean - ctl_sem]))
    y_hi = float(np.nanmax(np.r_[obs_mean + obs_sem, ctl_mean + ctl_sem]))
    if not np.isfinite(y_lo) or not np.isfinite(y_hi):
        return -1.0, 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-6
    return y_lo, y_hi


def _resolve_pair_key(df: pd.DataFrame) -> pd.Series:
    if "pair_key" in df.columns:
        return df["pair_key"].astype(str)
    needed = {"monkey_name_m1", "monkey_name_m2"}
    missing = needed.difference(df.columns)
    if missing:
        raise RuntimeError(
            "Cannot derive pair_key; missing columns: "
            f"{sorted(missing)}"
        )
    return df["monkey_name_m1"].astype(str) + "__" + df["monkey_name_m2"].astype(str)


def _load_leader_tables(
    leader_dir: Path,
    settings: LeaderFollowerCrossCorrComparisonPlotSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_path = leader_dir / settings.leader_session_filename
    date_path = leader_dir / settings.leader_date_filename
    pair_path = leader_dir / settings.leader_pair_filename
    for path in (session_path, date_path, pair_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing leader-follower file: {path}")
    session_df = pd.read_pickle(session_path)
    date_df = pd.read_pickle(date_path)
    pair_df = pd.read_pickle(pair_path)
    return session_df, date_df, pair_df


def _build_leader_maps(
    session_df: pd.DataFrame,
    date_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> dict[str, dict]:
    session_map: dict[tuple[str, str], str] = {}
    for _, row in session_df.iterrows():
        session_map[(str(row.get("date")), str(row.get("session")))] = str(
            row.get("leader_agent", "")
        )

    date_map: dict[tuple[str, str], str] = {}
    for _, row in date_df.iterrows():
        date_map[(str(row.get("date")), str(row.get("pair_key")))] = str(
            row.get("leader_agent", "")
        )

    pair_map: dict[str, str] = {}
    for _, row in pair_df.iterrows():
        pair_map[str(row.get("pair_key"))] = str(row.get("leader_agent", ""))

    return {
        "session": session_map,
        "date": date_map,
        "pair": pair_map,
    }


def _lookup_leader_agent(
    row: pd.Series,
    *,
    basis: str,
    leader_maps: dict[str, dict],
) -> str:
    date = str(row.get("date"))
    session = str(row.get("session"))
    pair_key = str(row.get("pair_key"))
    if basis == "session":
        return str(leader_maps["session"].get((date, session), ""))
    if basis == "date":
        return str(leader_maps["date"].get((date, pair_key), ""))
    if basis == "pair":
        return str(leader_maps["pair"].get(pair_key, ""))
    raise ValueError(f"Unsupported leader basis: {basis}")


def _align_leader_oriented_matrices(
    within_df: pd.DataFrame,
    control_df: pd.DataFrame,
    *,
    basis: str,
    comparison_kind: str,
    leader_maps: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    within_cols = ["date", "session", "cross_correlation"]
    missing_within = set(within_cols).difference(within_df.columns)
    if missing_within:
        raise RuntimeError(f"Within-session table missing columns: {sorted(missing_within)}")

    if comparison_kind == "cross":
        required_control = {
            "date",
            "session",
            "cross_correlation_mean_m1_source",
            "cross_correlation_mean_m2_source",
        }
        missing_control = required_control.difference(control_df.columns)
        if missing_control:
            raise RuntimeError(
                "Cross-session table is missing directional source columns required for "
                "leader-aligned cross-session comparison. Re-run observed cross-correlation "
                "build to regenerate cross outputs. Missing: "
                f"{sorted(missing_control)}"
            )
    elif comparison_kind == "shuffle":
        required_control = {"date", "session", "cross_correlation_shuffle_mean"}
        missing_control = required_control.difference(control_df.columns)
        if missing_control:
            raise RuntimeError(f"Shuffle table missing columns: {sorted(missing_control)}")
    else:
        raise ValueError(f"Unsupported comparison kind: {comparison_kind}")

    within_tmp = within_df.copy()
    control_tmp = control_df.copy()
    within_tmp["pair_key"] = _resolve_pair_key(within_tmp)
    control_tmp["pair_key"] = _resolve_pair_key(control_tmp)
    merged = within_tmp.merge(
        control_tmp,
        how="inner",
        on=["date", "session"],
        suffixes=("_obs", "_ctl"),
    )
    if merged.empty:
        raise RuntimeError("No overlapping (date, session) rows between observed and control tables.")

    observed_rows: list[np.ndarray] = []
    control_rows: list[np.ndarray] = []
    n_oriented = 0
    for _, row in merged.iterrows():
        pair_key = row.get("pair_key_obs", row.get("pair_key_ctl"))
        row_with_key = pd.Series({
            "date": row.get("date"),
            "session": row.get("session"),
            "pair_key": pair_key,
        })
        leader_agent = _lookup_leader_agent(row_with_key, basis=basis, leader_maps=leader_maps)
        if leader_agent not in {"m1", "m2"}:
            continue

        obs = _as_1d_float(row["cross_correlation"])
        if leader_agent == "m2":
            obs = obs[::-1]

        if comparison_kind == "cross":
            if leader_agent == "m1":
                ctl = _as_1d_float(row["cross_correlation_mean_m1_source"])
            else:
                ctl = _as_1d_float(row["cross_correlation_mean_m2_source"])
        else:
            ctl = _as_1d_float(row["cross_correlation_shuffle_mean"])
            if leader_agent == "m2":
                ctl = ctl[::-1]

        if obs.size != ctl.size:
            raise RuntimeError(
                "Observed/control array-length mismatch after leader alignment: "
                f"{obs.size} vs {ctl.size}"
            )
        observed_rows.append(obs)
        control_rows.append(ctl)
        n_oriented += 1

    if not observed_rows:
        raise RuntimeError(
            "No leader-oriented rows available after applying leader basis "
            f"'{basis}'. Check leader-follower outputs for this basis."
        )
    observed = np.vstack(observed_rows)
    control = np.vstack(control_rows)
    return observed, control, int(len(within_df)), int(len(control_df)), int(n_oriented)


def _plot_one_scope(
    *,
    ax,
    lags_seconds: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    scope: str,
    basis: str,
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
    y_min_global: float,
    y_max_global: float,
    n_obs_total: int,
    n_ctl_total: int,
    n_oriented: int,
) -> None:
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

    idx = _downsample_indices(int(lags_seconds.size), int(max_plot_points))
    lags_plot = lags_seconds[idx]
    obs_mean_plot = obs_mean[idx]
    obs_sem_plot = obs_sem[idx]
    ctl_mean_plot = ctl_mean[idx]
    ctl_sem_plot = ctl_sem[idx]
    sig_plot = _downsample_significance_mask(sig, idx)
    sig_plot = _limit_true_markers(sig_plot, int(max_sig_markers))

    ax.plot(lags_plot, obs_mean_plot, color=color_observed, lw=1.7, label="Observed (leader-aligned)")
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
    ax.axvline(0.0, color="#444444", lw=0.8, ls="--", alpha=0.8)

    y_lo = float(y_min_global)
    y_hi = float(y_max_global)
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
    pretty_scope = "non-interactive" if scope == "non_interactive" else scope
    ax.set_title(
        f"{pretty_scope} ({basis} basis)\n"
        f"n_oriented={n_oriented} (obs={n_obs_total}, ctrl={n_ctl_total})",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Lag (s)")


def _plot_observed_vs_control_for_basis(
    settings: LeaderFollowerCrossCorrComparisonPlotSettings,
    *,
    basis: str,
    leader_maps: dict[str, dict],
    comparison_kind: str,
    control_label: str,
    output_filename: str,
) -> Path:
    cfg = load_config(settings.cfg_path)
    plot_cfg = load_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)

    out_dir = build_analysis_output_dir(cfg, settings.crosscorr_analysis_subdir)
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
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharey=True)

    colors = plot_cfg.get("crosscorr_m1_m2_colors", {})
    observed_color = colors.get("observed", "#1f77b4")
    control_color = colors.get("control", "#d62728")

    panel_rows: list[dict] = []
    global_y_min = np.inf
    global_y_max = -np.inf

    for scope in scopes:
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
            kind="cross" if comparison_kind == "cross" else "shuffle",
        )
        observed, control, n_obs_total, n_ctl_total, n_oriented = _align_leader_oriented_matrices(
            within_df,
            control_df,
            basis=basis,
            comparison_kind=comparison_kind,
            leader_maps=leader_maps,
        )
        if observed.shape[1] != lags_seconds.size:
            raise RuntimeError(
                f"Lag length mismatch for scope='{scope}': "
                f"lags={lags_seconds.size}, observed={observed.shape[1]}"
            )
        y_lo, y_hi = _scope_y_bounds(observed, control)
        global_y_min = min(global_y_min, y_lo)
        global_y_max = max(global_y_max, y_hi)
        panel_rows.append(
            {
                "scope": scope,
                "lags_seconds": lags_seconds,
                "observed": observed,
                "control": control,
                "n_obs_total": n_obs_total,
                "n_ctl_total": n_ctl_total,
                "n_oriented": n_oriented,
            }
        )

    if not np.isfinite(global_y_min) or not np.isfinite(global_y_max):
        global_y_min, global_y_max = -1.0, 1.0
    if global_y_max <= global_y_min:
        global_y_max = global_y_min + 1e-6

    for ax, panel in zip(axes, panel_rows):
        _plot_one_scope(
            ax=ax,
            lags_seconds=panel["lags_seconds"],
            observed=panel["observed"],
            control=panel["control"],
            scope=panel["scope"],
            basis=basis,
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
            y_min_global=global_y_min,
            y_max_global=global_y_max,
            n_obs_total=panel["n_obs_total"],
            n_ctl_total=panel["n_ctl_total"],
            n_oriented=panel["n_oriented"],
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


def plot_leader_follower_crosscorr_comparisons(
    settings: LeaderFollowerCrossCorrComparisonPlotSettings,
) -> list[Path]:
    """Create leader-aligned observed-vs-cross and observed-vs-shuffle figures."""
    cfg = load_config(settings.cfg_path)
    leader_dir = build_analysis_output_dir(cfg, settings.leader_follower_subdir)
    session_df, date_df, pair_df = _load_leader_tables(leader_dir, settings)

    ref_scope = normalize_fix_crosscorr_time_scope(settings.leader_reference_scope)
    if "time_scope" in session_df.columns:
        session_df = session_df[session_df["time_scope"].astype(str) == ref_scope]
    if "time_scope" in date_df.columns:
        date_df = date_df[date_df["time_scope"].astype(str) == ref_scope]
    if "time_scope" in pair_df.columns:
        pair_df = pair_df[pair_df["time_scope"].astype(str) == ref_scope]
    leader_maps = _build_leader_maps(session_df, date_df, pair_df)

    bases = []
    for basis in settings.leader_bases:
        token = str(basis).strip().lower()
        if token not in _VALID_LEADER_BASES:
            raise ValueError(
                f"Unsupported leader basis '{basis}'. Allowed: {', '.join(_VALID_LEADER_BASES)}."
            )
        bases.append(token)

    outputs: list[Path] = []
    for basis in bases:
        outputs.append(
            _plot_observed_vs_control_for_basis(
                settings,
                basis=basis,
                leader_maps=leader_maps,
                comparison_kind="cross",
                control_label="Cross-session control (leader-aligned)",
                output_filename=settings.observed_vs_cross_filename_template.format(basis=basis),
            )
        )
        outputs.append(
            _plot_observed_vs_control_for_basis(
                settings,
                basis=basis,
                leader_maps=leader_maps,
                comparison_kind="shuffle",
                control_label="Shuffle control (leader-aligned)",
                output_filename=settings.observed_vs_shuffle_filename_template.format(basis=basis),
            )
        )
    return outputs
