"""Plot fixation-level neural cross-correlation summaries by date and region grouping."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm

from dal_monte_2022_analysis.behav.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation import (
    CROSS_ANALYSIS_KIND,
    WITHIN_ANALYSIS_KIND,
    FixationNeuralCrossCorrelationPlotAggregationSettings,
    build_fixation_neural_cross_correlation_plot_payload,
)
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


_CONDITION_ORDER = ("face_interactive", "face_non_interactive", "object")
_CONDITION_LABELS = {
    "face_interactive": "Face (interactive)",
    "face_non_interactive": "Face (non-interactive)",
    "object": "Object",
}
_CONDITION_COLORS = {
    "face_interactive": "#d62728",
    "face_non_interactive": "#1f77b4",
    "object": "#2ca02c",
}
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_PLOT_MODES = ("pairs", "mean")
_PLOT_NORMALIZATION_METHODS = ("none", "max_abs", "zscore")


@dataclass
class FixationNeuralCrossCorrelationPlotSettings:
    """Configuration for fixation neural cross-correlation summary plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    within_input_subdir: str = "ephys/psth/fixation_neural_crosscorr/within_region"
    cross_input_subdir: str = "ephys/psth/fixation_neural_crosscorr/cross_region"
    within_input_filename: str = "fixations.pkl"
    cross_input_filename: str = "fixations.pkl"
    within_pair_average_input_filename: str = "pair_averages.pkl"
    cross_pair_average_input_filename: str = "pair_averages.pkl"
    output_subdir: str = "ephys/psth/fixation_neural_crosscorr/plots"
    date_output_extension: str = "png"
    region_output_extension: str = "pdf"
    date_output_dpi: Optional[int] = 220
    region_output_dpi: Optional[int] = 220
    date_figsize: Optional[Sequence[float]] = None
    region_figsize: Optional[Sequence[float]] = None
    face_label: str = "face"
    object_label: str = "object"
    interactive_label: str = "interactive"
    condition_order: Sequence[str] = field(default_factory=lambda: tuple(_CONDITION_ORDER))
    condition_labels: dict[str, str] = field(default_factory=lambda: dict(_CONDITION_LABELS))
    condition_colors: dict[str, str] = field(default_factory=lambda: dict(_CONDITION_COLORS))
    pair_trace_alpha: float = 0.12
    pair_trace_linewidth: float = 0.75
    mean_trace_linewidth: float = 2.2
    max_pair_traces_per_plot: Optional[int] = None
    max_points_per_pdf_trace: Optional[int] = None
    normalize_traces: bool = False
    normalization_method: str = "max_abs"
    subplot_ncols: int = 3
    use_parallel: bool = True
    max_procs: int = 16
    parallelize_date_plots: bool = True
    parallelize_global_plots: bool = True
    random_seed: int = 13
    test_single: bool = False


def _ensure_ext(ext: str, *, fallback: str) -> str:
    token = str(ext).strip().lower()
    if not token:
        return fallback
    return token[1:] if token.startswith(".") else token


def _slug(text: str) -> str:
    token = _SLUG_RE.sub("_", str(text).strip())
    token = token.strip("_")
    return token or "unknown"


def _stable_seed(base_seed: int, *parts: str) -> int:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**32 - 1)


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None:
        return x, y
    n = int(x.size)
    if max_points <= 0 or n <= int(max_points):
        return x, y
    idx = np.linspace(0, n - 1, int(max_points), dtype=int)
    idx = np.unique(idx)
    return x[idx], y[idx]


def _resolve_figsize_and_dpi(
    settings: FixationNeuralCrossCorrelationPlotSettings,
    *,
    figure_kind: str,
    cfg_figsize: Optional[Sequence[float]],
    cfg_dpi: Optional[int],
) -> tuple[list[float], Optional[int]]:
    if figure_kind == "date":
        figsize = list(settings.date_figsize) if settings.date_figsize is not None else None
        dpi = settings.date_output_dpi if settings.date_output_dpi is not None else cfg_dpi
    else:
        figsize = list(settings.region_figsize) if settings.region_figsize is not None else None
        dpi = settings.region_output_dpi if settings.region_output_dpi is not None else cfg_dpi

    if figsize is None:
        figsize = list(cfg_figsize) if cfg_figsize is not None else [12.0, 4.0]
    return [float(figsize[0]), float(figsize[1])], dpi


def _build_output_path(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationPlotSettings,
    *,
    analysis_kind: str,
    level: str,
    date: Optional[str],
    plot_mode: str,
    ext: str,
) -> Path:
    root = build_analysis_output_dir(cfg, settings.output_subdir)
    if level == "date":
        if date is None:
            raise ValueError("date output path requires a date value.")
        file_name = f"date={_slug(str(date))}__plot={_slug(str(plot_mode))}.{ext}"
        return root / "date_level" / analysis_kind / file_name
    return root / "global_level" / analysis_kind / f"all_dates__plot={_slug(str(plot_mode))}.{ext}"


def _resolve_normalization_method(settings: FixationNeuralCrossCorrelationPlotSettings) -> str:
    token = str(settings.normalization_method).strip().lower()
    if token not in _PLOT_NORMALIZATION_METHODS:
        allowed = ", ".join(_PLOT_NORMALIZATION_METHODS)
        raise ValueError(f"Unsupported normalization_method='{settings.normalization_method}'. Expected: {allowed}.")
    return token


def _normalize_trace(trace: np.ndarray, method: str) -> np.ndarray:
    vec = np.asarray(trace, dtype=np.float64).reshape(-1)
    if vec.size == 0:
        return vec
    if not np.isfinite(vec).all():
        vec = np.where(np.isfinite(vec), vec, 0.0)

    if method == "none":
        return vec
    if method == "max_abs":
        denom = float(np.max(np.abs(vec)))
        if denom <= 0.0 or not np.isfinite(denom):
            return np.zeros_like(vec)
        return vec / denom
    if method == "zscore":
        mean = float(np.mean(vec))
        std = float(np.std(vec))
        if std <= 0.0 or not np.isfinite(std):
            return np.zeros_like(vec)
        return (vec - mean) / std
    raise ValueError(f"Unsupported normalization method '{method}'.")


def _normalize_plot_map(
    plot_map: dict,
    settings: FixationNeuralCrossCorrelationPlotSettings,
) -> dict:
    method = _resolve_normalization_method(settings)
    if not settings.normalize_traces or method == "none":
        return plot_map

    out: dict = {}
    for key, traces_by_condition in plot_map.items():
        cond_map: dict[str, list[np.ndarray]] = {}
        for condition, traces in traces_by_condition.items():
            cond_map[condition] = [_normalize_trace(np.asarray(trace, dtype=np.float64), method) for trace in traces]
        out[key] = cond_map
    return out


def _pick_pair_traces(
    traces: list[np.ndarray],
    *,
    max_pair_traces_per_plot: Optional[int],
    seed: int,
) -> list[np.ndarray]:
    if max_pair_traces_per_plot is None:
        return traces
    if len(traces) <= int(max_pair_traces_per_plot):
        return traces
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(traces), size=int(max_pair_traces_per_plot), replace=False))
    return [traces[int(idx)] for idx in picked]


def _build_subplot_grid(n_panels: int, ncols: int) -> tuple[int, int]:
    cols = max(1, min(int(max(1, ncols)), int(max(1, n_panels))))
    rows = int(np.ceil(float(n_panels) / float(cols)))
    return rows, cols


def _plot_group_grid_figure(
    *,
    group_items: Sequence[tuple[str, dict[str, list[np.ndarray]]]],
    x_axis: np.ndarray,
    x_label: str,
    title: str,
    plot_mode: str,
    settings: FixationNeuralCrossCorrelationPlotSettings,
    output_path: Path,
    figure_kind: str,
    cfg_figsize: Optional[Sequence[float]],
    cfg_dpi: Optional[int],
) -> None:
    if not group_items:
        return

    figsize, dpi = _resolve_figsize_and_dpi(
        settings,
        figure_kind=figure_kind,
        cfg_figsize=cfg_figsize,
        cfg_dpi=cfg_dpi,
    )
    n_rows, n_cols = _build_subplot_grid(len(group_items), settings.subplot_ncols)
    panel_w = max(3.8, float(figsize[0]) / max(1.0, float(settings.subplot_ncols)))
    panel_h = max(3.0, float(figsize[1]))
    fig_w = panel_w * float(n_cols)
    fig_h = panel_h * float(n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=[fig_w, fig_h],
        squeeze=False,
        sharex=True,
        sharey=(plot_mode == "mean"),
        facecolor="white",
    )
    axes_flat = list(np.ravel(axes))

    is_pdf = str(output_path.suffix).lower() == ".pdf"
    max_points = settings.max_points_per_pdf_trace if is_pdf else None

    for axis_idx, axis in enumerate(axes_flat):
        if axis_idx >= len(group_items):
            axis.set_visible(False)
            continue

        group_label, traces_by_condition = group_items[axis_idx]
        count_parts: list[str] = []
        any_data = False

        for condition in settings.condition_order:
            cond_label = settings.condition_labels.get(condition, condition)
            cond_color = settings.condition_colors.get(condition, "#333333")
            traces = [np.asarray(trace, dtype=np.float64) for trace in traces_by_condition.get(condition, [])]
            count_parts.append(f"{cond_label}: {len(traces)}")

            if not traces:
                continue
            any_data = True

            if plot_mode == "pairs":
                traces_to_plot = _pick_pair_traces(
                    traces,
                    max_pair_traces_per_plot=settings.max_pair_traces_per_plot,
                    seed=_stable_seed(settings.random_seed, title, group_label, condition),
                )
                for trace in traces_to_plot:
                    x_plot, y_plot = _downsample_xy(x_axis, trace, max_points)
                    axis.plot(
                        x_plot,
                        y_plot,
                        color=cond_color,
                        alpha=float(settings.pair_trace_alpha),
                        linewidth=float(settings.pair_trace_linewidth),
                        zorder=1,
                    )
            elif plot_mode == "mean":
                stacked = np.vstack(traces)
                mean_trace = np.mean(stacked, axis=0)
                x_plot, y_plot = _downsample_xy(x_axis, mean_trace, max_points)
                axis.plot(
                    x_plot,
                    y_plot,
                    color=cond_color,
                    alpha=1.0,
                    linewidth=float(settings.mean_trace_linewidth),
                    zorder=3,
                )
            else:
                raise ValueError(f"Unsupported plot_mode='{plot_mode}'.")

        if any_data:
            axis.text(
                0.02,
                0.96,
                " | ".join(count_parts),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#222222",
            )
        else:
            axis.text(
                0.5,
                0.5,
                "no data",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )

        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)
        axis.set_title(str(group_label))
        axis.set_xlabel(x_label)
        if axis_idx % int(max(1, n_cols)) == 0:
            axis.set_ylabel("Normalized XCorr" if settings.normalize_traces else "Cross-correlation")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=settings.condition_colors.get(condition, "#333333"),
            linewidth=float(settings.mean_trace_linewidth if plot_mode == "mean" else settings.pair_trace_linewidth),
            alpha=(1.0 if plot_mode == "mean" else max(0.35, float(settings.pair_trace_alpha))),
            label=settings.condition_labels.get(condition, condition),
        )
        for condition in settings.condition_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=max(1, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"format": str(output_path.suffix).lstrip(".")}
    if dpi is not None:
        save_kwargs["dpi"] = int(dpi)
    save_kwargs["facecolor"] = "white"
    save_kwargs["transparent"] = False
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


_GLOBAL_PLOT_STATE: dict = {}


def _init_plot_worker(state: dict) -> None:
    global _GLOBAL_PLOT_STATE
    _GLOBAL_PLOT_STATE = state
    plot_cfg = state.get("plot_cfg")
    if isinstance(plot_cfg, dict):
        apply_plotting_config(plot_cfg)


def _plot_group_grid_worker(task: tuple[str, str, Optional[str], str, Path]) -> Optional[Path]:
    level, kind, date, plot_mode, output_path = task
    state = _GLOBAL_PLOT_STATE
    settings = state["settings"]
    cfg_figsize = state.get("cfg_figsize")
    cfg_dpi = state.get("cfg_dpi")

    if level == "date":
        traces_map = state["date_plot_map"]
        group_items = [
            (str(group_label), traces)
            for (kind_key, date_key, group_label), traces in traces_map.items()
            if str(kind_key) == str(kind) and str(date_key) == str(date)
        ]
    else:
        traces_map = state["global_plot_map"]
        group_items = [
            (str(group_label), traces)
            for (kind_key, group_label), traces in traces_map.items()
            if str(kind_key) == str(kind)
        ]
    group_items.sort(key=lambda item: item[0])
    if not group_items:
        return None

    x_axis = state["x_axes"].get(kind)
    x_label = state["x_labels"].get(kind)
    if x_axis is None or x_axis.size == 0:
        return None

    if level == "date":
        title = f"{date} | {kind} | {plot_mode}"
        figure_kind = "date"
    else:
        title = f"All dates | {kind} | {plot_mode}"
        figure_kind = "region"

    _plot_group_grid_figure(
        group_items=group_items,
        x_axis=x_axis,
        x_label=x_label or "Lag",
        title=title,
        plot_mode=plot_mode,
        settings=settings,
        output_path=output_path,
        figure_kind=figure_kind,
        cfg_figsize=cfg_figsize,
        cfg_dpi=cfg_dpi,
    )
    return output_path


def _run_plot_jobs(
    settings: FixationNeuralCrossCorrelationPlotSettings,
    *,
    desc: str,
    tasks: Sequence[tuple[str, str, Optional[str], str, Path]],
    state: dict,
    allow_parallel: bool,
) -> list[Path]:
    if not tasks:
        return []

    outputs: list[Path] = []
    if allow_parallel and settings.use_parallel and len(tasks) > 1:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        with Pool(processes=n_proc, initializer=_init_plot_worker, initargs=(state,)) as pool:
            iterator = pool.imap_unordered(_plot_group_grid_worker, tasks)
            for out_path in tqdm(iterator, total=len(tasks), desc=f"{desc} ({n_proc} workers)", unit="plot"):
                if out_path is not None:
                    outputs.append(Path(out_path))
    else:
        _init_plot_worker(state)
        for task in tqdm(tasks, total=len(tasks), desc=desc, unit="plot"):
            out_path = _plot_group_grid_worker(task)
            if out_path is not None:
                outputs.append(Path(out_path))
    outputs.sort()
    return outputs


def plot_fixation_neural_cross_correlation_summaries(
    settings: FixationNeuralCrossCorrelationPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    analysis_kinds: Optional[Sequence[str]] = None,
) -> dict:
    """Build date-level and global region-group neural xcorr summary plots."""
    plot_cfg = None
    cfg_figsize = None
    cfg_dpi = None
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        cfg_figsize, cfg_dpi = resolve_figsize(plot_cfg)

    aggregation_settings = FixationNeuralCrossCorrelationPlotAggregationSettings(
        cfg_path=settings.cfg_path,
        within_input_subdir=settings.within_input_subdir,
        cross_input_subdir=settings.cross_input_subdir,
        within_input_filename=settings.within_input_filename,
        cross_input_filename=settings.cross_input_filename,
        within_pair_average_input_filename=settings.within_pair_average_input_filename,
        cross_pair_average_input_filename=settings.cross_pair_average_input_filename,
        face_label=settings.face_label,
        object_label=settings.object_label,
        interactive_label=settings.interactive_label,
        condition_order=tuple(settings.condition_order),
    )
    payload = build_fixation_neural_cross_correlation_plot_payload(
        aggregation_settings,
        dates=dates,
        sessions=sessions,
        analysis_kinds=analysis_kinds,
    )
    cfg = payload["cfg"]
    date_plot_map = payload["date_plot_map"]
    global_plot_map = payload["global_plot_map"]
    x_axes = payload["x_axes"]
    x_labels = payload["x_labels"]
    within_counts = payload["within_counts"]
    cross_counts = payload["cross_counts"]

    if settings.normalize_traces:
        date_plot_map = _normalize_plot_map(date_plot_map, settings)
        global_plot_map = _normalize_plot_map(global_plot_map, settings)

    date_ext = _ensure_ext(settings.date_output_extension, fallback="png")
    region_ext = _ensure_ext(settings.region_output_extension, fallback="pdf")

    outputs_date: list[Path] = []
    outputs_global: list[Path] = []

    plot_state = {
        "settings": settings,
        "plot_cfg": plot_cfg,
        "cfg_figsize": cfg_figsize,
        "cfg_dpi": cfg_dpi,
        "date_plot_map": date_plot_map,
        "global_plot_map": global_plot_map,
        "x_axes": x_axes,
        "x_labels": x_labels,
    }

    date_units = sorted({(kind, str(date)) for kind, date, _group_label in date_plot_map.keys()})
    if settings.test_single and date_units:
        rng = np.random.default_rng(_stable_seed(settings.random_seed, "date_keys"))
        date_units = [date_units[int(rng.integers(0, len(date_units)))]]

    date_tasks: list[tuple[str, str, Optional[str], str, Path]] = []
    for kind, date in date_units:
        x_axis = x_axes.get(kind)
        if x_axis is None or x_axis.size == 0:
            continue
        for plot_mode in _PLOT_MODES:
            out_path = _build_output_path(
                cfg,
                settings,
                analysis_kind=kind,
                level="date",
                date=date,
                plot_mode=plot_mode,
                ext=date_ext,
            )
            date_tasks.append(("date", kind, date, plot_mode, out_path))
    outputs_date = _run_plot_jobs(
        settings,
        desc="Date-level neural xcorr plots",
        tasks=date_tasks,
        state=plot_state,
        allow_parallel=settings.parallelize_date_plots,
    )

    global_units = sorted({kind for kind, _group_label in global_plot_map.keys()})
    if settings.test_single and global_units:
        rng = np.random.default_rng(_stable_seed(settings.random_seed, "global_keys"))
        global_units = [global_units[int(rng.integers(0, len(global_units)))]]

    global_tasks: list[tuple[str, str, Optional[str], str, Path]] = []
    for kind in global_units:
        x_axis = x_axes.get(kind)
        if x_axis is None or x_axis.size == 0:
            continue
        for plot_mode in _PLOT_MODES:
            out_path = _build_output_path(
                cfg,
                settings,
                analysis_kind=kind,
                level="global",
                date=None,
                plot_mode=plot_mode,
                ext=region_ext,
            )
            global_tasks.append(("global", kind, None, plot_mode, out_path))
    outputs_global = _run_plot_jobs(
        settings,
        desc="Global neural xcorr plots",
        tasks=global_tasks,
        state=plot_state,
        allow_parallel=settings.parallelize_global_plots,
    )

    return {
        "date_outputs": outputs_date,
        "global_outputs": outputs_global,
        "analysis_kinds": payload.get("analysis_kinds"),
        "within_counts": within_counts,
        "cross_counts": cross_counts,
    }


def plot_within_region_fixation_neural_cross_correlation_summaries(
    settings: FixationNeuralCrossCorrelationPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> dict:
    """Build within-region fixation neural xcorr summary plots."""
    return plot_fixation_neural_cross_correlation_summaries(
        settings,
        dates=dates,
        sessions=sessions,
        analysis_kinds=(WITHIN_ANALYSIS_KIND,),
    )


def plot_cross_region_fixation_neural_cross_correlation_summaries(
    settings: FixationNeuralCrossCorrelationPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> dict:
    """Build cross-region fixation neural xcorr summary plots."""
    return plot_fixation_neural_cross_correlation_summaries(
        settings,
        dates=dates,
        sessions=sessions,
        analysis_kinds=(CROSS_ANALYSIS_KIND,),
    )
