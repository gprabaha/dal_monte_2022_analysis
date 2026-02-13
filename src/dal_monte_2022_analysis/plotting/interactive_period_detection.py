"""Plot per-session interactive-period detection traces and timelines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_interactive_periods_config,
    load_plotting_config,
)
from dal_monte_2022_analysis.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    build_processed_data_path,
    scan_processed_data_paths,
)


@dataclass
class InteractivePeriodDetectionPlotSettings:
    """Configuration for interactive-period detection plots."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    interactive_periods_cfg_path: str = "configs/interactive_periods.yaml"
    analysis_subdir: str = "interactive_periods"
    output_subdir: str = "period_detection"
    output_extension: str = "pdf"
    fixation_vectors_modality: str = "fixation_binary_vectors"
    fixation_density_modality: str = "fixation_density_vectors"
    joint_density_modality: str = "joint_face_fixation_density"
    interactive_periods_modality: str = "interactive_periods"
    face_label: str = "face"
    object_label: str = "object"
    interactive_label: str = "interactive"
    threshold_factor: float = 0.34
    state_column: str = "state"
    start_column: str = "start"
    stop_column: str = "stop"
    threshold_column: str = "threshold"
    sample_rate_hz: float = 1000.0
    session_parallel: bool = True
    test_single: bool = False
    max_parallel_workers: int = 32
    max_density_points: int = 50000
    rasterize_density_traces: bool = True
    rasterize_interactive_blocks: bool = True
    pdf_compression: int = 6


def _load_pickle(path: Path):
    """Load a pickle object from disk."""
    return pd.read_pickle(path)


def _extract_vectors(obj) -> dict[str, np.ndarray]:
    """Extract vector dictionary from supported serialized objects."""
    vectors = None
    if hasattr(obj, "vectors"):
        vectors = getattr(obj, "vectors")
    elif isinstance(obj, dict) and "vectors" in obj:
        vectors = obj["vectors"]
    elif isinstance(obj, dict):
        vectors = obj

    if not isinstance(vectors, dict):
        return {}

    out: dict[str, np.ndarray] = {}
    for key, values in vectors.items():
        out[str(key)] = np.asarray(values, dtype=float).reshape(-1)
    return out


def _extract_joint_density(obj) -> Optional[np.ndarray]:
    """Extract joint density from supported serialized objects."""
    if hasattr(obj, "density"):
        arr = np.asarray(getattr(obj, "density"), dtype=float).reshape(-1)
        return arr
    if isinstance(obj, dict) and "density" in obj:
        arr = np.asarray(obj["density"], dtype=float).reshape(-1)
        return arr
    if isinstance(obj, np.ndarray):
        return np.asarray(obj, dtype=float).reshape(-1)
    return None


def _coerce_period_table(obj) -> pd.DataFrame:
    """Convert loaded interactive-period object to a DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)
    return pd.DataFrame()


def _binary_segments(binary: np.ndarray, sample_rate_hz: float) -> list[tuple[float, float]]:
    """Convert binary vector to (start_sec, duration_sec) segments."""
    arr = np.asarray(binary, dtype=float).reshape(-1)
    if arr.size == 0:
        return []
    mask = arr > 0.5
    if not np.any(mask):
        return []

    transitions = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    starts = np.concatenate(([0], transitions))
    stops = np.concatenate((transitions, [arr.size]))

    segments: list[tuple[float, float]] = []
    for start, stop in zip(starts, stops):
        if not mask[start]:
            continue
        width = (float(stop) - float(start)) / sample_rate_hz
        if width <= 0:
            continue
        segments.append((float(start) / sample_rate_hz, width))
    return segments


def _interactive_intervals(
    periods_df: pd.DataFrame,
    *,
    interactive_label: str,
    state_column: str,
    start_column: str,
    stop_column: str,
    n_samples: int,
) -> list[tuple[int, int]]:
    """Extract and clip interactive intervals to [0, n_samples-1]."""
    required = {state_column, start_column, stop_column}
    if periods_df.empty or required.difference(periods_df.columns):
        return []

    states = periods_df[state_column].astype(str)
    selected = periods_df.loc[states == str(interactive_label)].copy()
    if selected.empty:
        return []

    starts = pd.to_numeric(selected[start_column], errors="coerce")
    stops = pd.to_numeric(selected[stop_column], errors="coerce")
    valid = starts.notna() & stops.notna()
    if not valid.any():
        return []

    intervals: list[tuple[int, int]] = []
    for start_val, stop_val in zip(starts.loc[valid], stops.loc[valid]):
        start = max(0, int(start_val))
        stop = min(n_samples - 1, int(stop_val))
        if start <= stop:
            intervals.append((start, stop))
    return intervals


def _resolve_threshold(
    periods_df: pd.DataFrame,
    *,
    threshold_column: str,
    threshold_factor: float,
    joint_density: np.ndarray,
) -> float:
    """Resolve threshold from period table when available, else from factor*mean."""
    if not periods_df.empty and threshold_column in periods_df.columns:
        vals = pd.to_numeric(periods_df[threshold_column], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if not vals.empty:
            return float(vals.iloc[0])
    return float(threshold_factor) * float(np.mean(joint_density))


def _resolve_period_figsize(plot_cfg: dict) -> tuple[list[float], int | None]:
    """Resolve a readable figure size for two-row session traces."""
    figsize, dpi = resolve_figsize(plot_cfg)
    if figsize is None:
        return [12.0, 6.0], dpi
    width = max(float(figsize[0]), 9.0)
    height = max(float(figsize[1]), 5.0)
    return [width, height], dpi


def _downsample_xy(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample dense traces by striding to reduce PDF size."""
    x_arr = np.asarray(x_values, dtype=float).reshape(-1)
    y_arr = np.asarray(y_values, dtype=float).reshape(-1)
    n = min(x_arr.size, y_arr.size)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    if max_points <= 0 or n <= max_points:
        return x_arr, y_arr

    stride = int(np.ceil(float(n) / float(max_points)))
    return x_arr[::stride], y_arr[::stride]


def _build_tasks(
    cfg: dict,
    settings: InteractivePeriodDetectionPlotSettings,
) -> list[dict]:
    """Build per-session plotting tasks with required input paths."""
    rows = scan_processed_data_paths(
        cfg,
        settings.joint_density_modality,
        agents=[None],
    )
    rows = sorted(rows, key=lambda row: (str(row["date"]), str(row["session"])))

    tasks: list[dict] = []
    for row in rows:
        base_row = {"date": str(row["date"]), "session": str(row["session"])}
        m1_fix_vec = build_processed_data_path(
            cfg,
            base_row,
            settings.fixation_vectors_modality,
            "m1",
        )
        m2_fix_vec = build_processed_data_path(
            cfg,
            base_row,
            settings.fixation_vectors_modality,
            "m2",
        )
        m1_density = build_processed_data_path(
            cfg,
            base_row,
            settings.fixation_density_modality,
            "m1",
        )
        m2_density = build_processed_data_path(
            cfg,
            base_row,
            settings.fixation_density_modality,
            "m2",
        )
        joint_density = build_processed_data_path(
            cfg,
            base_row,
            settings.joint_density_modality,
            None,
        )
        periods = build_processed_data_path(
            cfg,
            base_row,
            settings.interactive_periods_modality,
            None,
        )

        required = [m1_fix_vec, m2_fix_vec, m1_density, m2_density, joint_density, periods]
        if any(not path.exists() for path in required):
            continue

        tasks.append(
            {
                "date": base_row["date"],
                "session": base_row["session"],
                "m1_fix_vec_path": m1_fix_vec,
                "m2_fix_vec_path": m2_fix_vec,
                "m1_density_path": m1_density,
                "m2_density_path": m2_density,
                "joint_density_path": joint_density,
                "periods_path": periods,
            }
        )

    if settings.test_single and tasks:
        return [tasks[0]]
    return tasks


def _plot_single_period(
    task: dict,
    *,
    settings: InteractivePeriodDetectionPlotSettings,
    out_dir: Path,
    plot_cfg: dict,
    figsize: list[float],
    dpi: int | None,
) -> Optional[Path]:
    """Render one session plot and return output path."""
    apply_plotting_config(plot_cfg)

    m1_fix_vecs = _extract_vectors(_load_pickle(task["m1_fix_vec_path"]))
    m2_fix_vecs = _extract_vectors(_load_pickle(task["m2_fix_vec_path"]))
    m1_densities = _extract_vectors(_load_pickle(task["m1_density_path"]))
    m2_densities = _extract_vectors(_load_pickle(task["m2_density_path"]))
    joint_density = _extract_joint_density(_load_pickle(task["joint_density_path"]))
    periods_df = _coerce_period_table(_load_pickle(task["periods_path"]))

    if joint_density is None or joint_density.size == 0:
        return None

    arrays = [
        np.asarray(joint_density, dtype=float).reshape(-1),
    ]
    for values in (
        m1_fix_vecs.get(settings.face_label),
        m2_fix_vecs.get(settings.face_label),
        m1_fix_vecs.get(settings.object_label),
        m1_densities.get(settings.face_label),
        m2_densities.get(settings.face_label),
        m1_densities.get(settings.object_label),
    ):
        if values is not None:
            arrays.append(np.asarray(values, dtype=float).reshape(-1))

    n_samples = min((arr.size for arr in arrays if arr.size > 0), default=0)
    if n_samples <= 0:
        return None

    joint_density = joint_density[:n_samples]
    m1_face_density = np.asarray(
        m1_densities.get(settings.face_label, np.zeros(n_samples)),
        dtype=float,
    )[:n_samples]
    m2_face_density = np.asarray(
        m2_densities.get(settings.face_label, np.zeros(n_samples)),
        dtype=float,
    )[:n_samples]
    m1_object_density = np.asarray(
        m1_densities.get(settings.object_label, np.zeros(n_samples)),
        dtype=float,
    )[:n_samples]
    m1_face_binary = np.asarray(
        m1_fix_vecs.get(settings.face_label, np.zeros(n_samples)),
        dtype=float,
    )[:n_samples]
    m2_face_binary = np.asarray(
        m2_fix_vecs.get(settings.face_label, np.zeros(n_samples)),
        dtype=float,
    )[:n_samples]
    m1_object_binary = np.asarray(
        m1_fix_vecs.get(settings.object_label, np.zeros(n_samples)),
        dtype=float,
    )[:n_samples]

    threshold = _resolve_threshold(
        periods_df,
        threshold_column=settings.threshold_column,
        threshold_factor=settings.threshold_factor,
        joint_density=joint_density,
    )
    interactive_periods = _interactive_intervals(
        periods_df,
        interactive_label=settings.interactive_label,
        state_column=settings.state_column,
        start_column=settings.start_column,
        stop_column=settings.stop_column,
        n_samples=n_samples,
    )

    x_seconds = np.arange(n_samples, dtype=float) / float(settings.sample_rate_hz)
    duration_seconds = float(n_samples) / float(settings.sample_rate_hz)
    line_rasterized = bool(settings.rasterize_density_traces)
    block_rasterized = bool(settings.rasterize_interactive_blocks)

    x_m1_face, y_m1_face = _downsample_xy(
        x_seconds,
        m1_face_density,
        max_points=int(settings.max_density_points),
    )
    x_m2_face, y_m2_face = _downsample_xy(
        x_seconds,
        m2_face_density,
        max_points=int(settings.max_density_points),
    )
    x_m1_object, y_m1_object = _downsample_xy(
        x_seconds,
        m1_object_density,
        max_points=int(settings.max_density_points),
    )
    x_joint, y_joint = _downsample_xy(
        x_seconds,
        joint_density,
        max_points=int(settings.max_density_points),
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 3]},
    )
    ax_top = axes[0, 0]
    ax_bottom = axes[1, 0]

    colors = {
        "m1_face": "#4C72B0",
        "m2_face": "#DD8452",
        "m1_object": "#55A868",
        "joint": "#C62828",
        "threshold": "#1A1A1A",
        "interactive": "#F2A65A",
    }

    lane_specs = [
        ("m1 face", m1_face_binary, colors["m1_face"], 2),
        ("m2 face", m2_face_binary, colors["m2_face"], 1),
        ("m1 object", m1_object_binary, colors["m1_object"], 0),
    ]
    for _, binary, color, lane in lane_specs:
        segments = _binary_segments(binary, float(settings.sample_rate_hz))
        if segments:
            ax_top.broken_barh(
                segments,
                (lane + 0.10, 0.80),
                facecolors=color,
                edgecolors=color,
                linewidth=0.8,
                alpha=0.95,
                zorder=3,
            )

    ax_top.set_ylim(0.0, 3.0)
    ax_top.set_yticks([2.5, 1.5, 0.5])
    ax_top.set_yticklabels([spec[0] for spec in lane_specs])
    ax_top.grid(axis="x", alpha=0.24, linewidth=0.6)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_xlim(0.0, duration_seconds)
    ax_top.set_ylabel("Fixation")

    for start, stop in interactive_periods:
        start_sec = float(start) / float(settings.sample_rate_hz)
        stop_sec = float(stop + 1) / float(settings.sample_rate_hz)
        ax_bottom.axvspan(
            start_sec,
            stop_sec,
            facecolor=colors["interactive"],
            edgecolor="none",
            alpha=0.60,
            zorder=1,
            rasterized=block_rasterized,
        )

    line_m1_face = ax_bottom.plot(
        x_m1_face,
        y_m1_face,
        linestyle=":",
        linewidth=1.2,
        color=colors["m1_face"],
        label="m1 face density",
        zorder=2,
        rasterized=line_rasterized,
    )[0]
    line_m2_face = ax_bottom.plot(
        x_m2_face,
        y_m2_face,
        linestyle=":",
        linewidth=1.2,
        color=colors["m2_face"],
        label="m2 face density",
        zorder=2,
        rasterized=line_rasterized,
    )[0]
    line_m1_object = ax_bottom.plot(
        x_m1_object,
        y_m1_object,
        linestyle=":",
        linewidth=1.2,
        color=colors["m1_object"],
        label="m1 object density",
        zorder=2,
        rasterized=line_rasterized,
    )[0]
    line_joint = ax_bottom.plot(
        x_joint,
        y_joint,
        linestyle="-",
        linewidth=2.4,
        color=colors["joint"],
        label="joint face density",
        zorder=3,
        rasterized=line_rasterized,
    )[0]
    line_threshold = ax_bottom.axhline(
        y=float(threshold),
        linestyle="--",
        linewidth=1.4,
        color=colors["threshold"],
        label=f"threshold={float(threshold):.3f}",
        zorder=4,
    )

    ax_bottom.grid(alpha=0.24, linewidth=0.6)
    ax_bottom.set_xlim(0.0, duration_seconds)
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Density")

    interactive_patch = Patch(
        facecolor=colors["interactive"],
        edgecolor="none",
        alpha=0.20,
        label=settings.interactive_label.replace("_", " "),
    )
    ax_bottom.legend(
        handles=[
            interactive_patch,
            line_m1_face,
            line_m2_face,
            line_m1_object,
            line_joint,
            line_threshold,
        ],
        loc="upper right",
        fontsize=8.8,
        frameon=True,
        ncol=2,
    )

    fig.suptitle(
        f"Interactive period detection | date={task['date']} session={task['session']}",
        fontsize=12,
    )
    fig.tight_layout()

    ext = str(settings.output_extension).lstrip(".").lower() or "pdf"
    date_out_dir = out_dir / f"date={task['date']}"
    out_path = date_out_dir / (
        f"interactive_period_detection_date={task['date']}_session={task['session']}.{ext}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if ext == "pdf":
        with mpl.rc_context({"pdf.compression": int(settings.pdf_compression)}):
            fig.savefig(out_path, format=ext)
    else:
        fig.savefig(out_path, format=ext)
    plt.close(fig)
    return out_path


def _plot_single_period_worker(args) -> Optional[Path]:
    """Worker wrapper for parallel session plotting."""
    task, settings, out_dir, plot_cfg, figsize, dpi = args
    return _plot_single_period(
        task,
        settings=settings,
        out_dir=out_dir,
        plot_cfg=plot_cfg,
        figsize=figsize,
        dpi=dpi,
    )


def plot_interactive_period_detection(
    settings: InteractivePeriodDetectionPlotSettings,
) -> list[Path]:
    """Generate one interactive-period detection plot per session."""
    cfg = load_dataset_config(settings.cfg_path)
    plot_cfg = load_plotting_config(settings.plotting_cfg_path)
    interactive_cfg = load_interactive_periods_config(settings.interactive_periods_cfg_path)
    apply_plotting_config(plot_cfg)

    resolved_settings = replace(
        settings,
        interactive_periods_modality=str(
            interactive_cfg.get("output_modality", settings.interactive_periods_modality)
        ),
        interactive_label=str(interactive_cfg.get("high_label", settings.interactive_label)),
        threshold_factor=float(interactive_cfg.get("threshold_factor", settings.threshold_factor)),
    )

    tasks = _build_tasks(cfg, resolved_settings)
    if not tasks:
        raise RuntimeError("No sessions found with all required plotting inputs.")

    figsize, dpi = _resolve_period_figsize(plot_cfg)
    out_dir = (
        build_analysis_output_dir(cfg, resolved_settings.analysis_subdir)
        / resolved_settings.output_subdir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if resolved_settings.test_single:
        out_path = _plot_single_period(
            tasks[0],
            settings=resolved_settings,
            out_dir=out_dir,
            plot_cfg=plot_cfg,
            figsize=figsize,
            dpi=dpi,
        )
        return [out_path] if out_path is not None else []

    worker_args = [
        (task, resolved_settings, out_dir, plot_cfg, figsize, dpi)
        for task in tasks
    ]

    out_paths: list[Path] = []
    if resolved_settings.session_parallel and len(worker_args) > 1:
        n_proc = get_n_processes(max_procs=int(resolved_settings.max_parallel_workers))
        with Pool(processes=n_proc) as pool:
            iterator = pool.imap_unordered(_plot_single_period_worker, worker_args)
            for out_path in tqdm(
                iterator,
                total=len(worker_args),
                desc=f"Plotting interactive period detection ({n_proc} workers)",
                unit="session",
            ):
                if out_path is not None:
                    out_paths.append(Path(out_path))
    else:
        for args in tqdm(worker_args, desc="Plotting interactive period detection", unit="session"):
            out_path = _plot_single_period_worker(args)
            if out_path is not None:
                out_paths.append(Path(out_path))

    out_paths = sorted(set(out_paths), key=lambda path: str(path))
    return out_paths
