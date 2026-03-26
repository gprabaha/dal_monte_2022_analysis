"""Plot leader-aligned cross-correlation traces vs controls across scopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.core.behav.analysis_filenames import (
    normalize_fix_cross_correlation_time_scope,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.behav.plotting.common import apply_plotting_config, resolve_figsize
from dal_monte_2022_analysis.behav.plotting.cross_correlation_common import (
    as_1d_float,
    downsample_indices,
    downsample_significance_mask,
    limit_true_markers,
    load_df_for_scope,
    load_lags_for_scope,
    nanmean_sem,
    significance_mask_per_lag,
    scope_y_bounds,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure

_VALID_LEADER_BASES = ("session", "date", "pair")


@dataclass
class LeaderFollowerCrossCorrComparisonPlotSettings:
    """Configuration for leader-aligned cross-correlation comparison plots."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    cross_correlation_analysis_subdir: str = "cross_correlation_outputs"
    leader_follower_subdir: str = "cross_correlation_outputs/leader_follower"
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

    @property
    def crosscorr_analysis_subdir(self) -> str:
        """Backward-compatible alias for legacy setting name."""
        return str(self.cross_correlation_analysis_subdir)

    @crosscorr_analysis_subdir.setter
    def crosscorr_analysis_subdir(self, value: str) -> None:
        self.cross_correlation_analysis_subdir = str(value)


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

        obs = as_1d_float(row["cross_correlation"])
        if leader_agent == "m2":
            obs = obs[::-1]

        if comparison_kind == "cross":
            if leader_agent == "m1":
                ctl = as_1d_float(row["cross_correlation_mean_m1_source"])
            else:
                ctl = as_1d_float(row["cross_correlation_mean_m2_source"])
        else:
            ctl = as_1d_float(row["cross_correlation_shuffle_mean"])
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

    idx = downsample_indices(int(lags_seconds.size), int(max_plot_points))
    lags_plot = lags_seconds[idx]
    obs_mean_plot = obs_mean[idx]
    obs_sem_plot = obs_sem[idx]
    ctl_mean_plot = ctl_mean[idx]
    ctl_sem_plot = ctl_sem[idx]
    sig_plot = downsample_significance_mask(sig, idx)
    sig_plot = limit_true_markers(sig_plot, int(max_sig_markers))

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

    out_dir = build_analysis_output_dir(cfg, settings.cross_correlation_analysis_subdir)
    scopes = tuple(normalize_fix_cross_correlation_time_scope(scope) for scope in settings.scopes)
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
        y_lo, y_hi = scope_y_bounds(observed, control)
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
    save_figure(fig, out_path, ext="pdf")
    plt.close(fig)
    return out_path


def plot_leader_follower_cross_correlation_comparisons(
    settings: LeaderFollowerCrossCorrComparisonPlotSettings,
) -> list[Path]:
    """Create leader-aligned observed-vs-cross and observed-vs-shuffle figures."""
    cfg = load_config(settings.cfg_path)
    leader_dir = build_analysis_output_dir(cfg, settings.leader_follower_subdir)
    session_df, date_df, pair_df = _load_leader_tables(leader_dir, settings)

    ref_scope = normalize_fix_cross_correlation_time_scope(settings.leader_reference_scope)
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
