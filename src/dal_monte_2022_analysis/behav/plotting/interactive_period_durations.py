"""Plot interactive/non-interactive period duration histograms by monkey pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.behav.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    scan_processed_data_paths,
)


@dataclass
class InteractivePeriodDurationDistributionPlotSettings:
    """Configuration for monkey-pair interactive-period duration histogram plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    interactive_periods_cfg_path: str = "configs/interactive_periods.yaml"
    analysis_subdir: str = "interactive_periods"
    output_subdir: str = "duration_distributions"
    output_filename: str = "interactive_period_duration_distributions_histogram.pdf"
    m1_output_filename: str = "interactive_period_duration_distributions_histogram_by_m1.pdf"
    aggregate_output_filename: str = (
        "interactive_period_duration_distributions_histogram_all_pairs_aggregate.pdf"
    )
    interactive_periods_modality: str = "interactive_periods"
    high_label: str = "interactive"
    low_label: str = "non_interactive"
    state_column: str = "state"
    start_column: str = "start"
    stop_column: str = "stop"
    duration_column: str = "duration_seconds"
    histogram_bins: int = 60
    x_label: str = "Period duration (s)"
    y_label: str = "Count"


def _canonical_pair_label(monkey_a: object, monkey_b: object) -> str:
    """Build an order-insensitive monkey pair label."""
    a = str(monkey_a).strip()
    b = str(monkey_b).strip()
    if not a:
        a = "unknown"
    if not b:
        b = "unknown"
    ordered = sorted([a, b], key=lambda item: item.casefold())
    return f"{ordered[0]} + {ordered[1]}"


def _load_pair_lookup(cfg: dict) -> pd.DataFrame:
    """Load date->pair and m1 mapping from ephys metadata."""
    ephys_path = Path(cfg["raw_data_root"]) / "ephys_days_and_monkeys.pkl"
    if not ephys_path.exists():
        raise FileNotFoundError(f"Missing ephys metadata file: {ephys_path}")

    ephys_df = pd.read_pickle(ephys_path)
    date_col = "date" if "date" in ephys_df.columns else "session_name"
    required_cols = {date_col, "m1", "m2"}
    missing = required_cols.difference(ephys_df.columns)
    if missing:
        raise RuntimeError(
            "Ephys metadata missing required columns "
            f"(missing: {sorted(missing)}, found: {list(ephys_df.columns)})"
        )

    session_str = ephys_df[date_col].astype(str).str.strip()
    session_str = session_str.apply(lambda value: value.zfill(8) if len(value) == 7 else value)
    bad_vals = session_str[~session_str.str.fullmatch(r"\d{8}")].head(5).tolist()
    if bad_vals:
        raise RuntimeError(
            "Session names must be 7 or 8 digits in ephys metadata; "
            f"examples: {bad_vals}"
        )

    pairs_df = pd.DataFrame(
        {
            "date": session_str,
            "m1_label": [str(m1).strip() if str(m1).strip() else "unknown_m1" for m1 in ephys_df["m1"]],
            "pair_label": [
                _canonical_pair_label(m1, m2)
                for m1, m2 in zip(ephys_df["m1"], ephys_df["m2"])
            ],
        }
    )
    per_date_unique = pairs_df.groupby("date")["pair_label"].nunique()
    ambiguous_dates = per_date_unique[per_date_unique > 1]
    if not ambiguous_dates.empty:
        bad_dates = ambiguous_dates.index[:5].tolist()
        raise RuntimeError(
            "Found multiple monkey-pair labels for the same date in ephys metadata; "
            f"examples: {bad_dates}"
        )

    return pairs_df.drop_duplicates(subset=["date"], keep="first")


def _load_interactive_period_durations(
    *,
    cfg: dict,
    modality: str,
    state_column: str,
    start_column: str,
    stop_column: str,
    duration_column: str,
) -> pd.DataFrame:
    """Load shared interactive-period pickles and compute durations in seconds."""
    rows = scan_processed_data_paths(cfg, modality, agents=[None])
    if not rows:
        raise RuntimeError(f"No processed files found for modality '{modality}'")

    frames: list[pd.DataFrame] = []
    for row in rows:
        table = pd.read_pickle(row["path"])
        if not isinstance(table, pd.DataFrame):
            table = pd.DataFrame(table)
        if table.empty:
            continue

        required_cols = {state_column, start_column, stop_column}
        missing = required_cols.difference(table.columns)
        if missing:
            raise RuntimeError(
                f"Interactive periods table missing required columns {sorted(missing)}: {row['path']}"
            )

        starts = pd.to_numeric(table[start_column], errors="coerce")
        stops = pd.to_numeric(table[stop_column], errors="coerce")
        duration_seconds = (stops - starts + 1.0) / 1000.0
        valid = duration_seconds.notna() & (duration_seconds > 0)
        if not valid.any():
            continue

        frame = pd.DataFrame(
            {
                "date": str(row["date"]),
                "session": str(row["session"]),
                "state": table.loc[valid, state_column].astype(str).to_numpy(),
                duration_column: duration_seconds.loc[valid].to_numpy(dtype=float),
            }
        )
        frames.append(frame)

    if not frames:
        raise RuntimeError("No valid interactive-period durations were found.")
    return pd.concat(frames, ignore_index=True)


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics for one distribution."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
        }

    q1, median, q3 = np.percentile(arr, [25.0, 50.0, 75.0])
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
    }


def _plot_histogram_with_stats(
    *,
    ax,
    values: np.ndarray,
    bins: np.ndarray,
    hist_color: str,
    mean_color: str,
    summary_color: str,
) -> None:
    """Draw one histogram plus mean, q1, median, q3 vertical lines and callouts."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        return

    ax.hist(
        arr,
        bins=bins,
        color=hist_color,
        alpha=0.78,
        edgecolor="#222222",
        linewidth=0.55,
    )

    stats = _summary_stats(arr)
    summary_specs = [
        ("q1", stats["q1"], summary_color, 1.6, "--"),
        ("median", stats["median"], summary_color, 2.0, "-"),
        ("q3", stats["q3"], summary_color, 1.6, "--"),
        ("mean", stats["mean"], mean_color, 3.2, "-"),
    ]
    for _, x_val, color, width, linestyle in summary_specs:
        ax.axvline(
            x=x_val,
            color=color,
            linewidth=width,
            linestyle=linestyle,
            alpha=0.96,
            zorder=5,
        )

    y_max = ax.get_ylim()[1]
    # Place readable callout labels in a fixed, non-overlapping stack with
    # connector lines back to each vertical statistic line.
    x_text = float(bins[-1]) * 0.985
    label_specs = [
        ("mean", stats["mean"], mean_color, 0.93),
        ("median", stats["median"], summary_color, 0.82),
        ("q1", stats["q1"], summary_color, 0.71),
        ("q3", stats["q3"], summary_color, 0.60),
    ]
    for label, x_val, color, y_frac in label_specs:
        xy = (x_val, y_max * 0.97)
        xytext = (x_text, y_max * y_frac)
        ax.scatter(
            [x_val],
            [y_max * 0.97],
            s=22 if label == "mean" else 16,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=6,
        )
        ax.annotate(
            f"{label}={x_val:.3f}s",
            xy=xy,
            xytext=xytext,
            textcoords="data",
            xycoords="data",
            color=color,
            fontsize=8.2,
            fontweight="bold" if label == "mean" else "normal",
            ha="right",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.96,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 1.0 if label == "mean" else 0.85,
                "alpha": 0.9,
            },
            zorder=7,
        )


def _resolve_common_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, float]:
    """Resolve shared histogram bins and x-axis maximum."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        raise RuntimeError("No positive interactive-period durations available for plotting.")
    x_max = float(np.max(arr))
    if x_max <= 0:
        raise RuntimeError("All interactive-period durations are non-positive.")
    n_bins = max(10, int(n_bins))
    bins = np.linspace(0.0, x_max, n_bins + 1)
    return bins, x_max


def _resolve_figure_size(
    *,
    n_rows: int,
    plot_cfg: dict,
) -> tuple[list[float], int | None]:
    """Resolve dynamic figure size for rows=monkey pairs, cols=interactive states."""
    figsize_cfg, dpi = resolve_figsize(plot_cfg)
    if figsize_cfg is None:
        return [12.0, max(2.7 * n_rows, 4.5)], dpi
    width = max(float(figsize_cfg[0]), 10.0)
    height = max(float(figsize_cfg[1]), 2.7 * n_rows)
    return [width, height], dpi


def _plot_grouped_state_histograms(
    *,
    grouped_df: pd.DataFrame,
    group_column: str,
    group_values: list[str],
    duration_column: str,
    state_specs: list[tuple[str, str]],
    bins: np.ndarray,
    x_max: float,
    figsize: list[float],
    dpi: int | None,
    x_label: str,
    y_label: str,
    mean_color: str,
    summary_color: str,
) -> plt.Figure:
    """Plot one two-column histogram grid for grouped interactive states."""
    fig, axes = plt.subplots(
        len(group_values),
        2,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharex=True,
    )
    for row_idx, group_label in enumerate(group_values):
        group_df = grouped_df[grouped_df[group_column].astype(str) == str(group_label)]
        for col_idx, (state_label, hist_color) in enumerate(state_specs):
            ax = axes[row_idx, col_idx]
            state_values = group_df.loc[
                group_df["state"] == state_label,
                duration_column,
            ].to_numpy(dtype=float)
            _plot_histogram_with_stats(
                ax=ax,
                values=state_values,
                bins=bins,
                hist_color=hist_color,
                mean_color=mean_color,
                summary_color=summary_color,
            )
            ax.set_xlim(0.0, x_max)
            ax.grid(axis="y", alpha=0.24, linewidth=0.6)
            if row_idx == 0:
                ax.set_title(state_label.replace("_", " "))
            if col_idx == 0:
                ax.text(
                    0.98,
                    0.06,
                    str(group_label),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.20",
                        "facecolor": "white",
                        "edgecolor": "#DDDDDD",
                        "alpha": 0.9,
                    },
                )

    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    fig.tight_layout()
    return fig


def plot_interactive_period_duration_distributions(
    settings: InteractivePeriodDurationDistributionPlotSettings,
) -> list[Path]:
    """Plot monkey-pair grid, m1-only grid, and all-pairs aggregate histograms."""
    cfg = load_config(settings.cfg_path)
    interactive_cfg = load_config(settings.interactive_periods_cfg_path)
    plot_cfg = load_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)

    modality = str(interactive_cfg.get("output_modality", settings.interactive_periods_modality))
    high_label = str(interactive_cfg.get("high_label", settings.high_label))
    low_label = str(interactive_cfg.get("low_label", settings.low_label))
    resolved_settings = InteractivePeriodDurationDistributionPlotSettings(
        cfg_path=settings.cfg_path,
        plotting_cfg_path=settings.plotting_cfg_path,
        interactive_periods_cfg_path=settings.interactive_periods_cfg_path,
        analysis_subdir=settings.analysis_subdir,
        output_subdir=settings.output_subdir,
        output_filename=settings.output_filename,
        m1_output_filename=settings.m1_output_filename,
        aggregate_output_filename=settings.aggregate_output_filename,
        interactive_periods_modality=modality,
        high_label=high_label,
        low_label=low_label,
        state_column=settings.state_column,
        start_column=settings.start_column,
        stop_column=settings.stop_column,
        duration_column=settings.duration_column,
        histogram_bins=int(settings.histogram_bins),
        x_label=settings.x_label,
        y_label=settings.y_label,
    )

    durations_df = _load_interactive_period_durations(
        cfg=cfg,
        modality=resolved_settings.interactive_periods_modality,
        state_column=resolved_settings.state_column,
        start_column=resolved_settings.start_column,
        stop_column=resolved_settings.stop_column,
        duration_column=resolved_settings.duration_column,
    )
    pair_lookup = _load_pair_lookup(cfg)
    durations_df = durations_df.merge(pair_lookup, on="date", how="left")
    durations_df["pair_label"] = durations_df["pair_label"].fillna(
        durations_df["date"].map(lambda date: f"unknown_pair ({date})")
    )
    durations_df["m1_label"] = durations_df["m1_label"].fillna(
        durations_df["date"].map(lambda date: f"unknown_m1 ({date})")
    )

    pair_labels = sorted(durations_df["pair_label"].astype(str).drop_duplicates(), key=str.casefold)
    if not pair_labels:
        raise RuntimeError("No monkey pairs found after loading interactive-period durations.")
    m1_labels = sorted(durations_df["m1_label"].astype(str).drop_duplicates(), key=str.casefold)
    if not m1_labels:
        raise RuntimeError("No m1 labels found after loading interactive-period durations.")

    bins, x_max = _resolve_common_bins(
        durations_df[resolved_settings.duration_column].to_numpy(dtype=float),
        resolved_settings.histogram_bins,
    )
    figsize, dpi = _resolve_figure_size(n_rows=len(pair_labels), plot_cfg=plot_cfg)

    violin_cfg = plot_cfg.get("violin", {})
    colors = violin_cfg.get("colors", {})
    interactive_color = str(colors.get("joint", "#2A9D8F"))
    non_interactive_color = str(colors.get("product", "#E76F51"))
    mean_color = "#C62828"
    summary_color = "#1A1A1A"

    state_specs = [
        (resolved_settings.high_label, interactive_color),
        (resolved_settings.low_label, non_interactive_color),
    ]
    fig = _plot_grouped_state_histograms(
        grouped_df=durations_df,
        group_column="pair_label",
        group_values=pair_labels,
        duration_column=resolved_settings.duration_column,
        state_specs=state_specs,
        bins=bins,
        x_max=x_max,
        figsize=figsize,
        dpi=dpi,
        x_label=resolved_settings.x_label,
        y_label=resolved_settings.y_label,
        mean_color=mean_color,
        summary_color=summary_color,
    )

    out_dir = build_analysis_output_dir(cfg, resolved_settings.analysis_subdir) / resolved_settings.output_subdir
    out_path = out_dir / resolved_settings.output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

    # Per-m1 grid (each row is one unique m1 monkey, pooled across its sessions/pairs).
    m1_figsize, _ = _resolve_figure_size(n_rows=len(m1_labels), plot_cfg=plot_cfg)
    fig_m1 = _plot_grouped_state_histograms(
        grouped_df=durations_df,
        group_column="m1_label",
        group_values=m1_labels,
        duration_column=resolved_settings.duration_column,
        state_specs=state_specs,
        bins=bins,
        x_max=x_max,
        figsize=m1_figsize,
        dpi=dpi,
        x_label=resolved_settings.x_label,
        y_label=resolved_settings.y_label,
        mean_color=mean_color,
        summary_color=summary_color,
    )

    m1_out_path = out_dir / resolved_settings.m1_output_filename
    m1_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_m1.savefig(m1_out_path, format="pdf")
    plt.close(fig_m1)

    agg_height = min(4.0, max(2.2, 0.22 * float(figsize[1])))
    fig_agg, axes_agg = plt.subplots(
        1,
        2,
        figsize=[max(figsize[0], 10.0), agg_height],
        dpi=dpi,
        squeeze=False,
        sharex=True,
    )
    for col_idx, (state_label, hist_color) in enumerate(state_specs):
        ax = axes_agg[0, col_idx]
        state_values = durations_df.loc[
            durations_df["state"] == state_label,
            resolved_settings.duration_column,
        ].to_numpy(dtype=float)
        _plot_histogram_with_stats(
            ax=ax,
            values=state_values,
            bins=bins,
            hist_color=hist_color,
            mean_color=mean_color,
            summary_color=summary_color,
        )
        ax.set_xlim(0.0, x_max)
        ax.grid(axis="y", alpha=0.24, linewidth=0.6)
        ax.set_title(state_label.replace("_", " "))

    fig_agg.suptitle("All Sessions Aggregated Across Monkey Pairs", fontsize=12)
    fig_agg.supxlabel(resolved_settings.x_label)
    fig_agg.supylabel(resolved_settings.y_label)
    fig_agg.tight_layout()

    aggregate_out_path = out_dir / resolved_settings.aggregate_output_filename
    aggregate_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_agg.savefig(aggregate_out_path, format="pdf")
    plt.close(fig_agg)
    return [out_path, m1_out_path, aggregate_out_path]
