"""Plot interactive/non-interactive period duration distributions by monkey pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_interactive_periods_config,
    load_plotting_config,
)
from dal_monte_2022_analysis.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    scan_processed_data_paths,
)


@dataclass
class InteractivePeriodDurationDistributionPlotSettings:
    """Configuration for per-monkey-pair interactive-period duration plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    interactive_periods_cfg_path: str = "configs/interactive_periods.yaml"
    analysis_subdir: str = "interactive_periods"
    output_subdir: str = "duration_distributions"
    output_filename_prefix: str = "interactive_period_duration_distribution"
    interactive_periods_modality: str = "interactive_periods"
    high_label: str = "interactive"
    low_label: str = "non_interactive"
    state_column: str = "state"
    start_column: str = "start"
    stop_column: str = "stop"
    duration_column: str = "duration_bins"
    y_label: str = "Period duration (bins)"
    max_samples_per_state: int = 0


def _sanitize_slug(text: str) -> str:
    """Convert free text to a safe filename slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    slug = slug.strip("_").lower()
    return slug or "unknown_pair"


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
    """Load date->pair mapping from ephys metadata."""
    ephys_path = Path(cfg["raw_data_root"]) / "ephys_days_and_monkeys.pkl"
    if not ephys_path.exists():
        raise FileNotFoundError(f"Missing ephys metadata file: {ephys_path}")

    ephys_df = pd.read_pickle(ephys_path)
    required_cols = {"session_name", "m1", "m2"}
    missing = required_cols.difference(ephys_df.columns)
    if missing:
        raise RuntimeError(
            "Ephys metadata missing required columns "
            f"(missing: {sorted(missing)}, found: {list(ephys_df.columns)})"
        )

    session_str = ephys_df["session_name"].astype(str).str.strip()
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
    """Load all shared interactive-period pickles and compute durations."""
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
        durations = stops - starts + 1.0
        valid = durations.notna() & (durations > 0)
        if not valid.any():
            continue

        frame = pd.DataFrame(
            {
                "date": str(row["date"]),
                "session": str(row["session"]),
                "state": table.loc[valid, state_column].astype(str).to_numpy(),
                duration_column: durations.loc[valid].to_numpy(dtype=float),
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


def _subsample_for_plot(
    values: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Subsample values for plotting speed while preserving summary stats from full data."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if max_samples <= 0 or arr.size <= max_samples:
        return arr
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(arr.size, size=int(max_samples), replace=False)
    return arr[idx]


def _format_stats_line(label: str, stats: dict[str, float]) -> str:
    """Format one label's summary stats for plot annotation."""
    pretty_label = label.replace("_", " ")
    n = int(stats["n"])
    if n <= 0:
        return f"{pretty_label}: n=0"
    return (
        f"{pretty_label}: n={n}, mean={stats['mean']:.2f}, med={stats['median']:.2f}, "
        f"q1={stats['q1']:.2f}, q3={stats['q3']:.2f}"
    )


def _draw_single_distribution(
    *,
    ax,
    values: np.ndarray,
    position: int,
    color: str,
    max_samples: int,
    seed: int,
) -> dict[str, float]:
    """Draw one violin with quartile and mean overlays."""
    stats = _summary_stats(values)
    if int(stats["n"]) <= 0:
        return stats

    plot_values = _subsample_for_plot(values, max_samples=max_samples, seed=seed)
    parts = ax.violinplot(
        [plot_values],
        positions=[position],
        widths=0.75,
        showmedians=False,
        showextrema=False,
    )
    body = parts["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("#222222")
    body.set_alpha(0.84)

    ax.vlines(
        position,
        stats["q1"],
        stats["q3"],
        color="#111111",
        linewidth=2.0,
        zorder=4,
    )
    ax.hlines(
        stats["median"],
        position - 0.10,
        position + 0.10,
        color="#111111",
        linewidth=1.8,
        zorder=5,
    )
    ax.scatter(
        [position],
        [stats["mean"]],
        marker="D",
        s=28,
        color="#111111",
        zorder=6,
    )
    return stats


def _plot_pair_duration_distribution(
    *,
    pair_df: pd.DataFrame,
    pair_label: str,
    settings: InteractivePeriodDurationDistributionPlotSettings,
    figsize: list[float] | None,
    dpi: int | None,
    color_interactive: str,
    color_non_interactive: str,
    seed: int,
    out_path: Path,
) -> None:
    """Plot and save one monkey pair's interactive/non-interactive duration distributions."""
    interactive_values = pair_df.loc[
        pair_df["state"] == settings.high_label,
        settings.duration_column,
    ].to_numpy(dtype=float)
    non_interactive_values = pair_df.loc[
        pair_df["state"] == settings.low_label,
        settings.duration_column,
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    stats_interactive = _draw_single_distribution(
        ax=ax,
        values=interactive_values,
        position=1,
        color=color_interactive,
        max_samples=int(settings.max_samples_per_state),
        seed=seed,
    )
    stats_non_interactive = _draw_single_distribution(
        ax=ax,
        values=non_interactive_values,
        position=2,
        color=color_non_interactive,
        max_samples=int(settings.max_samples_per_state),
        seed=seed + 1,
    )
    if int(stats_interactive["n"]) <= 0 and int(stats_non_interactive["n"]) <= 0:
        ax.text(
            0.5,
            0.5,
            "No matching states in this pair.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    n_sessions = pair_df[["date", "session"]].drop_duplicates().shape[0]
    ax.set_title(f"{pair_label} (sessions={n_sessions})")
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [
            settings.high_label.replace("_", " "),
            settings.low_label.replace("_", " "),
        ]
    )
    ax.set_ylabel(settings.y_label)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    stats_text = "\n".join(
        [
            _format_stats_line(settings.high_label, stats_interactive),
            _format_stats_line(settings.low_label, stats_non_interactive),
            "Markers: diamond=mean, bar=median, whisker=q1-q3",
        ]
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.72, "edgecolor": "#DDDDDD"},
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_interactive_period_duration_distributions(
    settings: InteractivePeriodDurationDistributionPlotSettings,
) -> list[Path]:
    """Plot duration distributions for each monkey pair and save one PDF per pair."""
    cfg = load_dataset_config(settings.cfg_path)
    interactive_cfg = load_interactive_periods_config(settings.interactive_periods_cfg_path)
    plot_cfg = load_plotting_config(settings.plotting_cfg_path)
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
        output_filename_prefix=settings.output_filename_prefix,
        interactive_periods_modality=modality,
        high_label=high_label,
        low_label=low_label,
        state_column=settings.state_column,
        start_column=settings.start_column,
        stop_column=settings.stop_column,
        duration_column=settings.duration_column,
        y_label=settings.y_label,
        max_samples_per_state=int(settings.max_samples_per_state),
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

    pair_labels = sorted(durations_df["pair_label"].astype(str).drop_duplicates(), key=str.casefold)
    if not pair_labels:
        raise RuntimeError("No monkey pairs found after loading interactive-period durations.")

    out_dir = build_analysis_output_dir(cfg, resolved_settings.analysis_subdir) / resolved_settings.output_subdir
    figsize, dpi = resolve_figsize(plot_cfg)
    if figsize is None:
        figsize = [6.0, 4.2]

    violin_cfg = plot_cfg.get("violin", {})
    colors = violin_cfg.get("colors", {})
    color_interactive = str(colors.get("joint", "#2A9D8F"))
    color_non_interactive = str(colors.get("product", "#E76F51"))

    written_paths: list[Path] = []
    for idx, pair_label in enumerate(pair_labels):
        pair_df = durations_df[durations_df["pair_label"].astype(str) == pair_label]
        slug = _sanitize_slug(pair_label)
        out_path = out_dir / f"{resolved_settings.output_filename_prefix}_{slug}.pdf"
        _plot_pair_duration_distribution(
            pair_df=pair_df,
            pair_label=pair_label,
            settings=resolved_settings,
            figsize=figsize,
            dpi=dpi,
            color_interactive=color_interactive,
            color_non_interactive=color_non_interactive,
            seed=211 + (idx * 2),
            out_path=out_path,
        )
        written_paths.append(out_path)

    return written_paths
