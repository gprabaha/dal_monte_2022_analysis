"""Plot leader-follower monkey-role summaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_plotting_config,
)
from dal_monte_2022_analysis.plotting.common import (
    apply_plotting_config,
    format_p_value,
    resolve_figsize,
)
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


@dataclass
class LeaderFollowerMonkeyRolePupilPlotSettings:
    """Configuration for monkey-level leader/follower pupil violin plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "fix_cross_correlation"
    monkey_role_session_filename: str = (
        "within_session_face_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv"
    )
    monkey_role_summary_filename: str = (
        "summary_face_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv"
    )
    output_filename: str = "summary_face_fix_crosscorr_leader_follower_pupil_by_monkey_role_violin.pdf"
    max_samples_per_role: int = 20000
    n_cols: int = 4
    value_column: str = "mean_pupil"
    y_label: str = "Session mean pupil size"


@dataclass
class LeaderFollowerMonkeyRoleFixationCountPlotSettings:
    """Configuration for monkey-level leader/follower fixation-count violin plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "fix_cross_correlation"
    monkey_role_session_filename: str = (
        "within_session_face_fix_crosscorr_leader_follower_fixation_count_by_monkey_role.csv"
    )
    monkey_role_summary_filename: str = (
        "summary_face_fix_crosscorr_leader_follower_fixation_count_by_monkey_role.csv"
    )
    output_filename: str = (
        "summary_face_fix_crosscorr_leader_follower_fixation_count_by_monkey_role_violin.pdf"
    )
    max_samples_per_role: int = 20000
    n_cols: int = 4
    value_column: str = "fixation_count"
    y_label: str = "Session fixation count"


def _subsample_for_plot(
    values: np.ndarray,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample for display speed when distributions are very large."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if max_samples <= 0 or values.size <= max_samples:
        return values
    idx = rng.choice(values.size, size=int(max_samples), replace=False)
    return values[idx]


def _load_monkey_role_frames(
    *,
    cfg: dict,
    settings,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load session and summary CSVs for monkey-role plotting."""
    out_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)
    session_path = out_dir / settings.monkey_role_session_filename
    summary_path = out_dir / settings.monkey_role_summary_filename
    out_path = out_dir / settings.output_filename

    if not session_path.exists():
        raise FileNotFoundError(f"Missing monkey-role session file: {session_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing monkey-role summary file: {summary_path}")

    session_df = pd.read_csv(session_path)
    summary_df = pd.read_csv(summary_path)
    return session_df, summary_df, out_path


def _plot_single_monkey_violin(
    *,
    ax,
    monkey_rows: pd.DataFrame,
    summary_row: pd.Series,
    value_column: str,
    max_samples_per_role: int,
    seed: int,
    color_leader: str,
    color_follower: str,
    y_label: str,
) -> None:
    """Render one monkey panel with leader/follower violins."""
    leader_values = monkey_rows.loc[monkey_rows["role"] == "leader", value_column].to_numpy(dtype=float)
    follower_values = monkey_rows.loc[monkey_rows["role"] == "follower", value_column].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    leader_values = _subsample_for_plot(leader_values, max_samples=max_samples_per_role, rng=rng)
    follower_values = _subsample_for_plot(follower_values, max_samples=max_samples_per_role, rng=rng)

    datasets: list[np.ndarray] = []
    positions: list[int] = []
    if leader_values.size > 0:
        datasets.append(leader_values)
        positions.append(1)
    if follower_values.size > 0:
        datasets.append(follower_values)
        positions.append(2)

    if datasets:
        parts = ax.violinplot(
            datasets,
            positions=positions,
            widths=0.75,
            showmedians=True,
            showextrema=False,
        )
        for body, pos in zip(parts["bodies"], positions):
            body.set_facecolor(color_leader if pos == 1 else color_follower)
            body.set_edgecolor("#222222")
            body.set_alpha(0.82)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#111111")
            parts["cmedians"].set_linewidth(1.0)
    else:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )

    monkey_name = str(summary_row["monkey_name"])
    p_value = pd.to_numeric(summary_row.get("p"), errors="coerce")
    mean_diff = pd.to_numeric(summary_row.get("mean_diff"), errors="coerce")
    sig_raw = summary_row.get("sig", False)
    if isinstance(sig_raw, str):
        sig = sig_raw.strip().lower() in {"1", "true", "t", "yes", "y"}
    else:
        sig = bool(sig_raw)
    sig_label = " *" if sig else ""
    ax.set_title(
        f"{monkey_name}\nmean_diff={mean_diff:.3f}, p={format_p_value(float(p_value))}{sig_label}"
        if np.isfinite(mean_diff)
        else f"{monkey_name}\nmean_diff=n/a, p={format_p_value(float(p_value))}{sig_label}",
        fontsize=10,
    )
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["leader", "follower"])
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


def _plot_leader_follower_monkey_role_violin(settings) -> Path:
    """Plot monkey-level leader-vs-follower role violins and save PDF."""
    cfg = load_dataset_config(settings.cfg_path)
    plot_cfg = load_plotting_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)

    session_df, summary_df, out_path = _load_monkey_role_frames(cfg=cfg, settings=settings)
    if session_df.empty or summary_df.empty:
        raise RuntimeError("No monkey-role data found to plot.")

    required_session_cols = {"monkey_name", "role", settings.value_column}
    missing_session = required_session_cols.difference(session_df.columns)
    if missing_session:
        raise RuntimeError(
            f"Monkey-role session table missing required columns: {sorted(missing_session)}"
        )
    required_summary_cols = {"monkey_name", "p", "mean_diff", "sig"}
    missing_summary = required_summary_cols.difference(summary_df.columns)
    if missing_summary:
        raise RuntimeError(
            f"Monkey-role summary table missing required columns: {sorted(missing_summary)}"
        )

    monkeys = summary_df["monkey_name"].astype(str).drop_duplicates().tolist()
    if not monkeys:
        raise RuntimeError("No monkey names found in monkey-role summary.")

    n_cols = max(1, min(int(settings.n_cols), len(monkeys)))
    n_rows = int(ceil(len(monkeys) / n_cols))
    figsize, dpi = resolve_figsize(plot_cfg)
    if figsize is None:
        figsize = [4.2 * n_cols, 3.6 * n_rows]

    violin_cfg = plot_cfg.get("violin", {})
    colors = violin_cfg.get("colors", {})
    color_leader = str(colors.get("leader", "#4C72B0"))
    color_follower = str(colors.get("follower", "#DD8452"))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)
    for i, monkey_name in enumerate(monkeys):
        ax = axes.flat[i]
        monkey_rows = session_df[session_df["monkey_name"].astype(str) == monkey_name]
        summary_row = summary_df[summary_df["monkey_name"].astype(str) == monkey_name].iloc[0]
        _plot_single_monkey_violin(
            ax=ax,
            monkey_rows=monkey_rows,
            summary_row=summary_row,
            value_column=settings.value_column,
            max_samples_per_role=int(settings.max_samples_per_role),
            seed=13 + i,
            color_leader=color_leader,
            color_follower=color_follower,
            y_label=str(settings.y_label),
        )

    for j in range(len(monkeys), n_rows * n_cols):
        axes.flat[j].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return out_path


def plot_leader_follower_monkey_role_pupil_violin(
    settings: LeaderFollowerMonkeyRolePupilPlotSettings,
) -> Path:
    """Plot monkey-level leader-vs-follower pupil violins and save PDF."""
    return _plot_leader_follower_monkey_role_violin(settings)


def plot_leader_follower_monkey_role_fixation_count_violin(
    settings: LeaderFollowerMonkeyRoleFixationCountPlotSettings,
) -> Path:
    """Plot monkey-level leader-vs-follower fixation-count violins and save PDF."""
    return _plot_leader_follower_monkey_role_violin(settings)
