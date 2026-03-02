"""QC plotting for raw vs smoothed pupil timecourses."""

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
from dal_monte_2022_analysis.runtime.io.processed_data import scan_processed_paths
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
)


@dataclass
class SmoothedPupilQCPlotSettings:
    """Configuration for smoothed pupil QC plots."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "pupil_smoothing"
    output_filename: str = "smoothed_pupil_timecourse_qc.pdf"
    raw_pupil_modality: str = "pupil_size"
    smoothed_pupil_modality: str = "smoothed_pupil_size"
    n_sessions: int = 5
    random_seed: int = 13
    agents: tuple[str, str] = ("m1", "m2")
    max_points_per_trace: int = 30000


def _load_pupil_trace(path: Path) -> np.ndarray:
    """Load a 1D pupil trace from a pickle path."""
    obj = pd.read_pickle(path)
    if hasattr(obj, "d"):
        arr = np.asarray(obj.d, dtype=float).reshape(-1)
    elif isinstance(obj, dict) and "d" in obj:
        arr = np.asarray(obj["d"], dtype=float).reshape(-1)
    else:
        arr = np.asarray(obj, dtype=float).reshape(-1)
    return arr


def _subsample_for_plot(values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a trace by stride for faster plotting while preserving shape."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if max_points <= 0 or arr.size <= max_points:
        idx = np.arange(arr.size, dtype=int)
        return idx, arr
    stride = int(np.ceil(arr.size / float(max_points)))
    idx = np.arange(0, arr.size, stride, dtype=int)
    return idx, arr[idx]


def _session_agent_map(rows: list[dict]) -> dict[tuple[str, str, str], Path]:
    """Map (date, session, agent) to path."""
    out: dict[tuple[str, str, str], Path] = {}
    for row in rows:
        agent = row.get("agent")
        if agent is None:
            continue
        key = (str(row["date"]), str(row["session"]), str(agent))
        out[key] = Path(row["path"])
    return out


def _select_sessions(
    *,
    raw_map: dict[tuple[str, str, str], Path],
    smoothed_map: dict[tuple[str, str, str], Path],
    agents: tuple[str, str],
    n_sessions: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Choose random sessions that have raw+smoothed data for all agents."""
    raw_sessions = {(date, session) for date, session, _ in raw_map}
    smooth_sessions = {(date, session) for date, session, _ in smoothed_map}
    candidate_sessions = sorted(raw_sessions.intersection(smooth_sessions))

    eligible: list[tuple[str, str]] = []
    for date, session in candidate_sessions:
        has_all = True
        for agent in agents:
            if (date, session, agent) not in raw_map or (date, session, agent) not in smoothed_map:
                has_all = False
                break
        if has_all:
            eligible.append((date, session))

    if not eligible:
        raise RuntimeError(
            "No sessions found with both raw and smoothed pupil data for all agents."
        )

    n_pick = min(int(n_sessions), len(eligible))
    rng = np.random.default_rng(int(seed))
    chosen_idx = rng.choice(len(eligible), size=n_pick, replace=False)
    return [eligible[int(i)] for i in chosen_idx]


def _resolve_dynamic_figsize(
    *,
    n_rows: int,
    plot_cfg: dict,
) -> tuple[list[float], int | None]:
    """Resolve figure size with one row per sampled session and two columns."""
    figsize, dpi = resolve_figsize(plot_cfg)
    if figsize is None:
        return [12.0, max(2.4 * n_rows, 4.8)], dpi
    width = max(float(figsize[0]), 10.0)
    height = max(float(figsize[1]), 2.2 * n_rows)
    return [width, height], dpi


def plot_smoothed_pupil_timecourse_qc(settings: SmoothedPupilQCPlotSettings) -> Path:
    """Plot random-session raw vs smoothed pupil timecourses for m1/m2."""
    cfg = load_config(settings.cfg_path)
    plot_cfg = load_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)

    raw_rows = scan_processed_paths(
        cfg,
        settings.raw_pupil_modality,
        agents=list(settings.agents),
    )
    smoothed_rows = scan_processed_paths(
        cfg,
        settings.smoothed_pupil_modality,
        agents=list(settings.agents),
    )
    raw_map = _session_agent_map(raw_rows)
    smoothed_map = _session_agent_map(smoothed_rows)
    sessions = _select_sessions(
        raw_map=raw_map,
        smoothed_map=smoothed_map,
        agents=tuple(settings.agents),
        n_sessions=int(settings.n_sessions),
        seed=int(settings.random_seed),
    )

    figsize, dpi = _resolve_dynamic_figsize(n_rows=len(sessions), plot_cfg=plot_cfg)
    fig, axes = plt.subplots(
        len(sessions),
        len(settings.agents),
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharex=False,
    )

    agent_colors = {
        "m1": "#4C72B0",
        "m2": "#DD8452",
    }
    raw_color = "#707070"

    for row_idx, (date, session) in enumerate(sessions):
        for col_idx, agent in enumerate(settings.agents):
            ax = axes[row_idx, col_idx]
            raw_path = raw_map[(date, session, agent)]
            smooth_path = smoothed_map[(date, session, agent)]

            raw = _load_pupil_trace(raw_path)
            smoothed = _load_pupil_trace(smooth_path)
            n = min(raw.size, smoothed.size)
            if n == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            raw = raw[:n]
            smoothed = smoothed[:n]
            raw_idx, raw_vals = _subsample_for_plot(raw, int(settings.max_points_per_trace))
            smooth_idx, smooth_vals = _subsample_for_plot(smoothed, int(settings.max_points_per_trace))

            ax.plot(raw_idx / 1000.0, raw_vals, color=raw_color, alpha=0.42, linewidth=0.6, label="raw")
            ax.plot(
                smooth_idx / 1000.0,
                smooth_vals,
                color=agent_colors.get(agent, "#1F77B4"),
                alpha=0.95,
                linewidth=1.1,
                label="smoothed",
            )
            ax.grid(alpha=0.25, linewidth=0.6)

            if row_idx == 0:
                ax.set_title(agent)
            if col_idx == 0:
                ax.set_ylabel(f"{date}\nS{session}\nPupil")
            if row_idx == len(sessions) - 1:
                ax.set_xlabel("Time (s)")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", frameon=True)

    fig.suptitle(
        f"Raw vs Smoothed Pupil QC ({len(sessions)} random sessions, seed={int(settings.random_seed)})",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)
    out_path = out_dir / settings.output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, ext="pdf")
    plt.close(fig)
    return out_path
