"""Plot pair-condition mean neural xcorr summaries and condition-comparison statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_helpers import (
    CROSS_ANALYSIS_KIND,
    WITHIN_ANALYSIS_KIND,
    _PLOT_ALLOWED_ANALYSIS_KINDS,
    _PLOT_CONDITION_ORDER,
    _build_plot_x_axis,
    _resolve_signal_output_filename,
    _signal_output_label,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_pair_condition_means import (
    iter_fixation_neural_cross_correlation_pair_condition_mean_output_paths,
)
from dal_monte_2022_analysis.ephys.plotting.common import (
    apply_plotting_config,
    ensure_ext as _ensure_ext_shared,
    resolve_figsize,
)
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


_CONDITION_LABELS = {
    "face_interactive": "Face (interactive)",
    "face_non_interactive": "Face (non-interactive)",
    "object": "Object",
}
_CONDITION_SHORT_LABELS = {
    "face_interactive": "FI",
    "face_non_interactive": "FNI",
    "object": "OBJ",
}
_CONDITION_COLORS = {
    "face_interactive": "#b64198",
    "face_non_interactive": "#97ca3d",
    "object": "#754c29",
}
_SIGNIFICANCE_CORRECTIONS = ("none", "bonferroni", "holm", "fdr_bh")


@dataclass
class FixationNeuralCrossCorrelationPairConditionMeanPlotSettings:
    """Configuration for pair-condition mean xcorr plotting and condition comparisons."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    within_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/within_region"
    cross_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/cross_region"
    within_input_filename: str = "pair_condition_means.pkl"
    cross_input_filename: str = "pair_condition_means.pkl"
    signal_input_column: str = "spike_train_counts"
    signal_input_columns: Optional[Sequence[str]] = None
    output_subdir: str = "ephys/psth/fixation_neural_cross_correlation/pair_condition_mean_plots"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    figsize: Optional[Sequence[float]] = None
    within_figsize: Optional[Sequence[float]] = None
    cross_figsize: Optional[Sequence[float]] = None
    condition_order: Sequence[str] = field(default_factory=lambda: tuple(_PLOT_CONDITION_ORDER))
    condition_labels: dict[str, str] = field(default_factory=lambda: dict(_CONDITION_LABELS))
    condition_short_labels: dict[str, str] = field(default_factory=lambda: dict(_CONDITION_SHORT_LABELS))
    condition_colors: dict[str, str] = field(default_factory=lambda: dict(_CONDITION_COLORS))
    significance_alpha: float = 0.05
    mean_lag_significance_correction: str = "bonferroni"
    per_lag_significance_correction: str = "bonferroni"
    min_pairs_for_significance: int = 3
    mean_trace_linewidth: float = 2.2
    sem_alpha: float = 0.12
    between_condition_marker_size: float = 5.0
    between_condition_marker_alpha: float = 0.95
    mean_lag_annotation_fontsize: float = 8.0
    fit_annotation_fontsize: float = 7.0
    plot_lag_half_window_ms: float = 100.0
    lag_tick_step_ms: float = 50.0
    fit_r2_good_threshold: float = 0.8
    fit_min_points_per_side: int = 5
    fit_linear_linewidth: float = 1.3
    fit_exponential_linewidth: float = 1.5
    fit_line_alpha: float = 0.9
    use_parallel: bool = True
    max_procs: Optional[int] = None
    lag_test_min_lags_for_parallel: int = 256
    lag_test_chunk_size: int = 256



def _ensure_ext(ext: str, *, fallback: str) -> str:
    return _ensure_ext_shared(ext, fallback=fallback)



def _resolve_significance_correction(value: str) -> str:
    token = str(value).strip().lower()
    if token not in _SIGNIFICANCE_CORRECTIONS:
        allowed = ", ".join(_SIGNIFICANCE_CORRECTIONS)
        raise ValueError(f"Unsupported significance correction '{value}'. Expected one of: {allowed}.")
    return token



def _apply_pvalue_correction(
    p_vals: np.ndarray,
    *,
    alpha: float,
    correction: str,
) -> np.ndarray:
    vec = np.asarray(p_vals, dtype=np.float64).reshape(-1)
    sig = np.zeros(vec.shape, dtype=bool)
    finite = np.isfinite(vec)
    if not finite.any():
        return sig.reshape(np.asarray(p_vals).shape)

    if correction == "none":
        sig = finite & (vec < float(alpha))
        return sig.reshape(np.asarray(p_vals).shape)

    if correction == "bonferroni":
        m = int(np.sum(finite))
        if m <= 0:
            return sig.reshape(np.asarray(p_vals).shape)
        sig = finite & (vec < (float(alpha) / float(m)))
        return sig.reshape(np.asarray(p_vals).shape)

    if correction == "holm":
        idx = np.flatnonzero(finite)
        vals = vec[idx]
        order = np.argsort(vals)
        ranked = vals[order]
        m = int(ranked.size)
        if m <= 0:
            return sig.reshape(np.asarray(p_vals).shape)
        reject = np.zeros(m, dtype=bool)
        for i, p_value in enumerate(ranked):
            threshold = float(alpha) / float(m - i)
            if p_value <= threshold:
                reject[i] = True
            else:
                break
        if np.any(reject):
            max_i = int(np.max(np.flatnonzero(reject)))
            keep_sorted = np.zeros(m, dtype=bool)
            keep_sorted[: max_i + 1] = True
            keep_original = np.zeros(m, dtype=bool)
            keep_original[order] = keep_sorted
            sig[idx] = keep_original
        return sig.reshape(np.asarray(p_vals).shape)

    idx = np.flatnonzero(finite)
    vals = vec[idx]
    order = np.argsort(vals)
    ranked = vals[order]
    m = int(ranked.size)
    if m <= 0:
        return sig.reshape(np.asarray(p_vals).shape)
    thresholds = float(alpha) * (np.arange(1, m + 1, dtype=np.float64) / float(m))
    passed = ranked <= thresholds
    if np.any(passed):
        cutoff = ranked[int(np.max(np.flatnonzero(passed)))]
        sig[idx] = vals <= cutoff
    return sig.reshape(np.asarray(p_vals).shape)



def _safe_paired_ttest(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_pairs: int,
) -> dict[str, object]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n_pairs = int(x.size)
    mean_left = float(np.mean(x)) if n_pairs else None
    mean_right = float(np.mean(y)) if n_pairs else None
    mean_difference = float(np.mean(x - y)) if n_pairs else None

    min_required = int(max(2, min_pairs))
    if n_pairs < min_required:
        return {
            "n_pairs": n_pairs,
            "mean_left": mean_left,
            "mean_right": mean_right,
            "mean_difference": mean_difference,
            "t_statistic": None,
            "p_value": None,
            "tested": False,
        }

    diff = x - y
    if np.allclose(diff, diff[0]):
        if float(diff[0]) > 0.0:
            t_stat = float("inf")
            p_value = 0.0
        elif float(diff[0]) < 0.0:
            t_stat = float("-inf")
            p_value = 0.0
        else:
            t_stat = 0.0
            p_value = 1.0
    else:
        res = ttest_rel(x, y, nan_policy="omit")
        t_stat = float(np.asarray(res.statistic, dtype=np.float64).reshape(()))
        p_raw = float(np.asarray(res.pvalue, dtype=np.float64).reshape(()))
        p_value = p_raw if np.isfinite(p_raw) else None

    return {
        "n_pairs": n_pairs,
        "mean_left": mean_left,
        "mean_right": mean_right,
        "mean_difference": mean_difference,
        "t_statistic": t_stat,
        "p_value": p_value,
        "tested": True,
    }



def _build_pair_id(row) -> tuple[str, str, str, str, str]:
    return (
        str(getattr(row, "date", "unknown_date")),
        str(getattr(row, "region_1", "unknown_1")),
        str(getattr(row, "region_2", "unknown_2")),
        str(getattr(row, "unit_uuid_1", "unknown_unit_1")),
        str(getattr(row, "unit_uuid_2", "unknown_unit_2")),
    )



def _resolve_pair_condition_plot_paths(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    analysis_kind: str,
) -> tuple[str, str]:
    if analysis_kind == WITHIN_ANALYSIS_KIND:
        return settings.within_input_subdir, settings.within_input_filename
    if analysis_kind == CROSS_ANALYSIS_KIND:
        return settings.cross_input_subdir, settings.cross_input_filename
    raise ValueError(f"Unsupported analysis_kind={analysis_kind!r}.")



def _build_subset_label(dates: Optional[Sequence[str]]) -> str:
    if dates is None:
        return "all_dates"
    normalized = [str(value) for value in dates if str(value)]
    if not normalized:
        return "all_dates"
    if len(normalized) == 1:
        return f"date={normalized[0]}"
    return f"n_dates={len(normalized)}"


def _extract_date_from_output_path(path: Path) -> Optional[str]:
    for part in path.parts:
        if part.startswith("date=") and len(part) > 5:
            return str(part.split("=", 1)[1])
    return None


def _discover_available_dates(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    analysis_kinds: Sequence[str],
    signal_columns: Sequence[str],
) -> tuple[str, ...]:
    dates: set[str] = set()
    for analysis_kind in analysis_kinds:
        input_subdir, input_filename = _resolve_pair_condition_plot_paths(settings, analysis_kind=analysis_kind)
        for signal_column in signal_columns:
            for path in iter_fixation_neural_cross_correlation_pair_condition_mean_output_paths(
                dataset_cfg_path=settings.cfg_path,
                output_subdir=input_subdir,
                output_filename=input_filename,
                signal_input_column=signal_column,
                date=None,
            ):
                date = _extract_date_from_output_path(path)
                if date:
                    dates.add(str(date))
    return tuple(sorted(dates))


def _format_mean_lag_annotation(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    mean_rows: pd.DataFrame,
) -> str:
    if mean_rows.empty:
        return ""
    lines: list[str] = []
    for row in mean_rows.itertuples(index=False):
        left = settings.condition_short_labels.get(str(row.condition_a), str(row.condition_a))
        right = settings.condition_short_labels.get(str(row.condition_b), str(row.condition_b))
        if getattr(row, "p_value", None) is None or not np.isfinite(float(row.p_value)):
            p_label = "n/a"
        else:
            p_label = f"{float(row.p_value):.3g}"
        suffix = " *" if bool(getattr(row, "significant", False)) else ""
        lines.append(f"{left} vs {right}: p={p_label}{suffix}")
    return "\n".join(lines)


def _linear_decay_model(distance: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return (np.asarray(distance, dtype=np.float64) * float(slope)) + float(intercept)


def _exponential_decay_model(distance: np.ndarray, amplitude: float, rate: float, baseline: float) -> np.ndarray:
    return float(baseline) + (float(amplitude) * np.exp(-float(rate) * np.asarray(distance, dtype=np.float64)))


def _compute_r_squared(observed: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(predicted, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(y_hat)
    y = y[valid]
    y_hat = y_hat[valid]
    if y.size < 2:
        return None
    residual = y - y_hat
    ss_res = float(np.sum(residual * residual))
    centered = y - float(np.mean(y))
    ss_tot = float(np.sum(centered * centered))
    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.allclose(y, y_hat) else 0.0
    return float(1.0 - (ss_res / ss_tot))


def _fit_linear_decay_segment(distance: np.ndarray, values: np.ndarray) -> dict[str, object]:
    x = np.asarray(distance, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return {"predicted": None, "r_squared": None, "params": None}
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = _linear_decay_model(x, slope, intercept)
    return {
        "predicted": predicted.astype(np.float64),
        "r_squared": _compute_r_squared(y, predicted),
        "params": (float(slope), float(intercept)),
    }


def _fit_exponential_decay_segment(distance: np.ndarray, values: np.ndarray) -> dict[str, object]:
    x = np.asarray(distance, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return {"predicted": None, "r_squared": None, "params": None}
    amplitude0 = float(y[0] - y[-1])
    baseline0 = float(y[-1])
    max_x = float(np.max(np.abs(x))) if x.size else 1.0
    rate0 = 1.0 / max(max_x, 1.0)
    try:
        params, _ = curve_fit(
            _exponential_decay_model,
            x,
            y,
            p0=(amplitude0, rate0, baseline0),
            bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
    except Exception:
        return {"predicted": None, "r_squared": None, "params": None}
    predicted = _exponential_decay_model(x, *params)
    return {
        "predicted": np.asarray(predicted, dtype=np.float64),
        "r_squared": _compute_r_squared(y, predicted),
        "params": tuple(float(value) for value in params),
    }


def _fit_decay_models_for_side(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    x_axis: np.ndarray,
    values: np.ndarray,
    side: str,
) -> dict[str, object]:
    x = np.asarray(x_axis, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    if side == "negative":
        side_mask = x < 0.0
    elif side == "positive":
        side_mask = x > 0.0
    else:
        raise ValueError(f"Unsupported fit side '{side}'.")
    x_side = x[side_mask]
    y_side = y[side_mask]
    if x_side.size < int(max(2, settings.fit_min_points_per_side)):
        return {
            "side": side,
            "x_axis": x_side,
            "linear": {"predicted": None, "r_squared": None, "params": None, "good": False},
            "exponential": {"predicted": None, "r_squared": None, "params": None, "good": False},
            "selection": "none",
        }

    distance = np.abs(x_side)
    linear = _fit_linear_decay_segment(distance, y_side)
    exponential = _fit_exponential_decay_segment(distance, y_side)
    linear_r2 = linear.get("r_squared")
    exp_r2 = exponential.get("r_squared")
    linear_good = linear_r2 is not None and float(linear_r2) >= float(settings.fit_r2_good_threshold)
    exp_good = exp_r2 is not None and float(exp_r2) >= float(settings.fit_r2_good_threshold)
    linear["good"] = bool(linear_good)
    exponential["good"] = bool(exp_good)
    if linear_good and exp_good:
        selection = "both"
    elif linear_good:
        selection = "linear"
    elif exp_good:
        selection = "exponential"
    else:
        selection = "none"

    linear_pred = linear.get("predicted")
    if linear_pred is not None:
        slope, intercept = linear.get("params") or (0.0, 0.0)
        linear["predicted"] = _linear_decay_model(distance, slope, intercept).astype(np.float64)
    exp_pred = exponential.get("predicted")
    if exp_pred is not None:
        amplitude, rate, baseline = exponential.get("params") or (0.0, 0.0, 0.0)
        exponential["predicted"] = _exponential_decay_model(distance, amplitude, rate, baseline).astype(np.float64)

    return {
        "side": side,
        "x_axis": x_side.astype(np.float64),
        "linear": linear,
        "exponential": exponential,
        "selection": selection,
    }


def _format_fit_choice(model_result: dict[str, object]) -> str:
    selection = str(model_result.get("selection", "none"))
    linear_r2 = model_result.get("linear", {}).get("r_squared")
    exp_r2 = model_result.get("exponential", {}).get("r_squared")
    if selection == "linear":
        return f"lin({float(linear_r2):.2f})" if linear_r2 is not None else "lin"
    if selection == "exponential":
        return f"exp({float(exp_r2):.2f})" if exp_r2 is not None else "exp"
    if selection == "both":
        lin_label = f"{float(linear_r2):.2f}" if linear_r2 is not None else "n/a"
        exp_label = f"{float(exp_r2):.2f}" if exp_r2 is not None else "n/a"
        return f"both(l={lin_label},e={exp_label})"
    return "none"


def _format_fit_annotation(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    fit_map: dict[str, dict[str, dict[str, object]]],
) -> str:
    if not fit_map:
        return ""
    lines: list[str] = []
    for condition in settings.condition_order:
        condition = str(condition)
        side_map = fit_map.get(condition) or {}
        negative = _format_fit_choice(side_map.get("negative") or {})
        positive = _format_fit_choice(side_map.get("positive") or {})
        short = settings.condition_short_labels.get(condition, condition)
        lines.append(f"{short}: - {negative} | + {positive}")
    return "\n".join(lines)


def build_pair_condition_mean_fit_summary(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    result: dict[str, object],
) -> pd.DataFrame:
    group_plot_map = result.get("group_plot_map", {}) or {}
    if not group_plot_map:
        return pd.DataFrame()
    x_axis = np.asarray(result.get("x_axis", []), dtype=np.float64).reshape(-1)
    if x_axis.size <= 0:
        return pd.DataFrame()

    x_label = str(result.get("x_label") or "Lag")
    if x_label == "Lag (ms)":
        half_window_ms = float(max(1.0, settings.plot_lag_half_window_ms))
        plot_mask = np.isfinite(x_axis) & (x_axis >= -half_window_ms) & (x_axis <= half_window_ms)
        if not np.any(plot_mask):
            plot_mask = np.ones(x_axis.shape, dtype=bool)
        plot_x_axis = x_axis[plot_mask]
    else:
        plot_mask = np.ones(x_axis.shape, dtype=bool)
        plot_x_axis = x_axis

    rows: list[dict[str, object]] = []
    for group_label, traces_by_condition in sorted(group_plot_map.items(), key=lambda item: item[0]):
        for condition in settings.condition_order:
            condition = str(condition)
            traces = [
                np.asarray(trace, dtype=np.float64).reshape(-1)
                for trace in traces_by_condition.get(condition, [])
                if np.asarray(trace, dtype=np.float64).reshape(-1).shape == x_axis.shape
            ]
            if not traces:
                continue
            stacked = np.vstack([trace[plot_mask] for trace in traces])
            if stacked.size <= 0 or stacked.shape[1] <= 0:
                continue
            mean_trace = np.nanmean(stacked, axis=0)
            negative_fit = _fit_decay_models_for_side(
                settings,
                x_axis=plot_x_axis,
                values=mean_trace,
                side="negative",
            )
            positive_fit = _fit_decay_models_for_side(
                settings,
                x_axis=plot_x_axis,
                values=mean_trace,
                side="positive",
            )
            rows.append(
                {
                    "group_label": str(group_label),
                    "condition": condition,
                    "n_pairs": int(stacked.shape[0]),
                    "negative_fit": str(negative_fit.get("selection", "none")),
                    "negative_fit_label": _format_fit_choice(negative_fit),
                    "negative_linear_r2": negative_fit.get("linear", {}).get("r_squared"),
                    "negative_exponential_r2": negative_fit.get("exponential", {}).get("r_squared"),
                    "positive_fit": str(positive_fit.get("selection", "none")),
                    "positive_fit_label": _format_fit_choice(positive_fit),
                    "positive_linear_r2": positive_fit.get("linear", {}).get("r_squared"),
                    "positive_exponential_r2": positive_fit.get("exponential", {}).get("r_squared"),
                }
            )
    return pd.DataFrame(rows)



def _build_output_paths(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    analysis_kind: str,
    signal_column: str,
    subset_label: str,
    ext: str,
    is_per_day: bool = False,
) -> tuple[Path, Path, Path]:
    root = build_analysis_output_dir(cfg, settings.output_subdir)
    if is_per_day:
        root = root / "per_day" / analysis_kind
    else:
        root = root / analysis_kind
    figure_filename = _resolve_signal_output_filename(
        f"{subset_label}__pair_condition_mean_xcorr.{ext}",
        signal_column,
    )
    mean_lag_filename = _resolve_signal_output_filename(
        f"{subset_label}__pair_condition_mean_lag_stats.csv",
        signal_column,
    )
    per_lag_filename = _resolve_signal_output_filename(
        f"{subset_label}__pair_condition_per_lag_stats.pkl",
        signal_column,
    )
    return root / figure_filename, root / mean_lag_filename, root / per_lag_filename



def _collect_pair_condition_mean_rows(
    output_paths: Sequence[Path],
    *,
    condition_order: Sequence[str],
) -> dict[str, object]:
    lag_axis: Optional[np.ndarray] = None
    bin_size_ms: Optional[float] = None
    group_pair_condition_map: dict[str, dict[tuple[str, str, str, str, str], dict[str, np.ndarray]]] = {}
    counts = {
        "files": 0,
        "rows": 0,
        "used_rows": 0,
        "skipped_rows": 0,
    }

    valid_conditions = {str(name) for name in condition_order}
    signal_input_column: Optional[str] = None
    signal_variant: Optional[str] = None
    loaded_paths: list[str] = []

    for path in output_paths:
        counts["files"] += 1
        loaded_paths.append(str(path))
        obj = load_pickle_path(path)
        meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        if signal_input_column is None and meta.get("signal_input_column") is not None:
            signal_input_column = str(meta.get("signal_input_column"))
        if signal_variant is None and meta.get("signal_variant") is not None:
            signal_variant = str(meta.get("signal_variant"))

        file_lags = meta.get("lags")
        if file_lags is not None:
            file_lag_axis = np.asarray(file_lags, dtype=np.int64).reshape(-1)
            if file_lag_axis.size > 0:
                if lag_axis is None:
                    lag_axis = file_lag_axis
                elif lag_axis.shape != file_lag_axis.shape or not np.array_equal(lag_axis, file_lag_axis):
                    raise ValueError(f"Lag-axis mismatch while loading pair-condition means: {path}")
        raw_bin_size_ms = meta.get("bin_size_ms")
        if raw_bin_size_ms is not None:
            try:
                candidate = float(raw_bin_size_ms)
            except Exception:
                candidate = np.nan
            if np.isfinite(candidate) and candidate > 0.0:
                if bin_size_ms is None:
                    bin_size_ms = float(candidate)
                elif not np.isclose(bin_size_ms, candidate):
                    bin_size_ms = None

        df = obj.get("pair_condition_means") if isinstance(obj, dict) else None
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if df.empty:
            continue

        for row in df.itertuples(index=False):
            counts["rows"] += 1
            condition = str(getattr(row, "condition", ""))
            if condition not in valid_conditions:
                counts["skipped_rows"] += 1
                continue
            trace = np.asarray(getattr(row, "cross_correlation", []), dtype=np.float64).reshape(-1)
            if trace.size <= 0:
                counts["skipped_rows"] += 1
                continue
            if lag_axis is not None and trace.shape != lag_axis.shape:
                counts["skipped_rows"] += 1
                continue
            trace = np.where(np.isfinite(trace), trace, np.nan)
            if not np.isfinite(trace).any():
                counts["skipped_rows"] += 1
                continue

            group_label = str(getattr(row, "group_label", "unknown_group"))
            pair_id = _build_pair_id(row)
            pair_bucket = group_pair_condition_map.setdefault(group_label, {}).setdefault(pair_id, {})
            pair_bucket[condition] = trace.astype(np.float64)
            counts["used_rows"] += 1

    return {
        "signal_input_column": signal_input_column,
        "signal_variant": signal_variant,
        "lags": lag_axis,
        "bin_size_ms": bin_size_ms,
        "group_pair_condition_map": group_pair_condition_map,
        "counts": counts,
        "loaded_paths": loaded_paths,
    }



def _compute_group_pair_condition_comparisons(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    group_label: str,
    pair_condition_map: dict[tuple[str, str, str, str, str], dict[str, np.ndarray]],
    n_lag_workers: Optional[int],
) -> tuple[dict[str, list[np.ndarray]], list[dict[str, object]], pd.DataFrame]:
    del n_lag_workers
    traces_by_condition: dict[str, list[np.ndarray]] = {str(name): [] for name in settings.condition_order}
    pair_ids_by_condition: dict[str, dict[tuple[str, str, str, str, str], np.ndarray]] = {
        str(name): {} for name in settings.condition_order
    }
    for pair_id, cond_map in pair_condition_map.items():
        for condition in settings.condition_order:
            condition = str(condition)
            trace = cond_map.get(condition)
            if trace is None:
                continue
            arr = np.asarray(trace, dtype=np.float64).reshape(-1)
            traces_by_condition[condition].append(arr)
            pair_ids_by_condition[condition][pair_id] = arr

    mean_lag_rows: list[dict[str, object]] = []
    pairwise_keys = list(combinations([str(name) for name in settings.condition_order], 2))
    mean_lag_pvals = np.full(len(pairwise_keys), np.nan, dtype=np.float64)

    for idx, (condition_a, condition_b) in enumerate(pairwise_keys):
        common_pair_ids = sorted(
            set(pair_ids_by_condition[condition_a].keys()) & set(pair_ids_by_condition[condition_b].keys())
        )
        if not common_pair_ids:
            paired_a = np.empty((0, 0), dtype=np.float64)
            paired_b = np.empty((0, 0), dtype=np.float64)
        else:
            paired_a = np.vstack([pair_ids_by_condition[condition_a][pair_id] for pair_id in common_pair_ids])
            paired_b = np.vstack([pair_ids_by_condition[condition_b][pair_id] for pair_id in common_pair_ids])

        mean_stats = _safe_paired_ttest(
            np.nanmean(paired_a, axis=1) if paired_a.size else np.array([], dtype=np.float64),
            np.nanmean(paired_b, axis=1) if paired_b.size else np.array([], dtype=np.float64),
            min_pairs=int(max(2, settings.min_pairs_for_significance)),
        )
        mean_lag_pvals[idx] = (
            float(mean_stats["p_value"]) if mean_stats.get("p_value") is not None else np.nan
        )
        mean_lag_rows.append(
            {
                "group_label": str(group_label),
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_pairs": int(mean_stats["n_pairs"]),
                "mean_condition_a": mean_stats["mean_left"],
                "mean_condition_b": mean_stats["mean_right"],
                "mean_difference": mean_stats["mean_difference"],
                "t_statistic": mean_stats["t_statistic"],
                "p_value": mean_stats["p_value"],
                "tested": bool(mean_stats["tested"]),
                "significant": False,
            }
        )

    mean_sig = _apply_pvalue_correction(
        mean_lag_pvals,
        alpha=float(settings.significance_alpha),
        correction=_resolve_significance_correction(settings.mean_lag_significance_correction),
    )
    for idx, is_sig in enumerate(np.asarray(mean_sig, dtype=bool).reshape(-1)):
        mean_lag_rows[idx]["significant"] = bool(is_sig)

    return traces_by_condition, mean_lag_rows, pd.DataFrame()



def build_fixation_neural_cross_correlation_pair_condition_mean_plot_payload(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    analysis_kinds: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Load pair-condition mean outputs and compute group-level condition-comparison statistics."""
    cfg = load_config(settings.cfg_path)
    selected_kinds = tuple(analysis_kinds) if analysis_kinds is not None else tuple(_PLOT_ALLOWED_ANALYSIS_KINDS)
    for kind in selected_kinds:
        if kind not in _PLOT_ALLOWED_ANALYSIS_KINDS:
            raise ValueError(
                f"Unsupported analysis kind '{kind}'. Expected one of: {', '.join(_PLOT_ALLOWED_ANALYSIS_KINDS)}."
            )

    signal_columns = tuple(
        str(value)
        for value in (
            settings.signal_input_columns
            if settings.signal_input_columns is not None
            else (settings.signal_input_column,)
        )
    )
    subset_label = _build_subset_label(dates)
    results: dict[tuple[str, str], dict[str, object]] = {}

    for analysis_kind in selected_kinds:
        input_subdir, input_filename = _resolve_pair_condition_plot_paths(settings, analysis_kind=analysis_kind)
        for signal_column in signal_columns:
            output_paths = iter_fixation_neural_cross_correlation_pair_condition_mean_output_paths(
                dataset_cfg_path=settings.cfg_path,
                output_subdir=input_subdir,
                output_filename=input_filename,
                signal_input_column=signal_column,
                date=dates[0] if dates is not None and len(dates) == 1 else None,
            )
            if dates is not None and len(dates) != 1:
                allowed_dates = {str(value) for value in dates}
                output_paths = [
                    path for path in output_paths if any(part == f"date={date}" for part in path.parts for date in allowed_dates)
                ]

            collected = _collect_pair_condition_mean_rows(
                output_paths,
                condition_order=settings.condition_order,
            )
            lag_axis = collected["lags"]
            x_axis, x_label = _build_plot_x_axis(lag_axis, collected["bin_size_ms"])
            group_plot_map: dict[str, dict[str, list[np.ndarray]]] = {}
            mean_lag_rows: list[dict[str, object]] = []
            per_lag_rows: list[dict[str, object]] = []
            for group_label, pair_condition_map in sorted(collected["group_pair_condition_map"].items()):
                traces_by_condition, group_mean_rows, group_lag_rows = _compute_group_pair_condition_comparisons(
                    settings,
                    group_label=group_label,
                    pair_condition_map=pair_condition_map,
                    n_lag_workers=None,
                )
                group_plot_map[group_label] = traces_by_condition
                mean_lag_rows.extend(group_mean_rows)
                per_lag_rows.extend(group_lag_rows)

            results[(analysis_kind, signal_column)] = {
                "analysis_kind": analysis_kind,
                "signal_input_column": collected["signal_input_column"] or signal_column,
                "signal_variant": collected["signal_variant"] or _signal_output_label(signal_column),
                "group_plot_map": group_plot_map,
                "mean_lag_comparisons": pd.DataFrame(mean_lag_rows),
                "per_lag_comparisons": pd.DataFrame(per_lag_rows),
                "counts": collected["counts"],
                "lags": lag_axis,
                "bin_size_ms": collected["bin_size_ms"],
                "x_axis": x_axis,
                "x_label": x_label,
                "subset_label": subset_label,
                "loaded_paths": collected["loaded_paths"],
            }

    return {
        "cfg": cfg,
        "analysis_kinds": selected_kinds,
        "signal_input_columns": signal_columns,
        "results": results,
        "subset_label": subset_label,
    }



def _build_between_condition_marker_map(
    condition_order: Sequence[str],
) -> dict[tuple[str, str], str]:
    marker_cycle = ("|", "x", "+", "1", "2", "3", "4", "o", "s", "^", "v", "d")
    out: dict[tuple[str, str], str] = {}
    for idx, pair_key in enumerate(combinations(condition_order, 2)):
        out[pair_key] = marker_cycle[idx % len(marker_cycle)]
    return out



def _resolve_figsize_and_dpi(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    analysis_kind: str,
    cfg_figsize: Optional[Sequence[float]],
    cfg_dpi: Optional[int],
    n_groups: int,
) -> tuple[list[float], Optional[int]]:
    if analysis_kind == CROSS_ANALYSIS_KIND and settings.cross_figsize is not None:
        figsize = list(settings.cross_figsize)
    elif analysis_kind == WITHIN_ANALYSIS_KIND and settings.within_figsize is not None:
        figsize = list(settings.within_figsize)
    else:
        figsize = list(settings.figsize) if settings.figsize is not None else None
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    if figsize is None:
        figsize = list(cfg_figsize) if cfg_figsize is not None else [max(4.0, 4.2 * max(1, n_groups)), 4.6]
    if len(figsize) != 2:
        figsize = [max(4.0, 4.2 * max(1, n_groups)), 4.6]
    return [float(figsize[0]), float(figsize[1])], dpi



def _plot_result(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    result: dict[str, object],
    output_path: Path,
    cfg_figsize: Optional[Sequence[float]],
    cfg_dpi: Optional[int],
) -> None:
    group_plot_map = result.get("group_plot_map", {}) or {}
    if not group_plot_map:
        return
    analysis_kind = str(result.get("analysis_kind") or "")
    is_cross_region = analysis_kind == CROSS_ANALYSIS_KIND
    x_axis = np.asarray(result.get("x_axis", []), dtype=np.float64).reshape(-1)
    if x_axis.size <= 0:
        return
    x_label = str(result.get("x_label") or "Lag")
    half_window_ms = float(max(1.0, settings.plot_lag_half_window_ms))
    tick_step_ms = float(max(1.0, settings.lag_tick_step_ms))
    if x_label == "Lag (ms)":
        plot_mask = np.isfinite(x_axis) & (x_axis >= -half_window_ms) & (x_axis <= half_window_ms)
        if not np.any(plot_mask):
            plot_mask = np.ones(x_axis.shape, dtype=bool)
        plot_x_axis = x_axis[plot_mask]
        tick_values = np.arange(
            -half_window_ms,
            half_window_ms + (0.5 * tick_step_ms),
            tick_step_ms,
            dtype=np.float64,
        )
    else:
        plot_mask = np.ones(x_axis.shape, dtype=bool)
        plot_x_axis = x_axis
        tick_values = None

    mean_lag_df = result.get("mean_lag_comparisons")
    mean_lag_df = mean_lag_df if isinstance(mean_lag_df, pd.DataFrame) else pd.DataFrame()

    group_items = sorted(group_plot_map.items(), key=lambda item: item[0])
    figsize, dpi = _resolve_figsize_and_dpi(
        settings,
        analysis_kind=analysis_kind,
        cfg_figsize=cfg_figsize,
        cfg_dpi=cfg_dpi,
        n_groups=len(group_items),
    )
    title_fontsize = 8.5 if is_cross_region else 10.0
    axis_label_fontsize = 8.0 if is_cross_region else 9.0
    tick_label_fontsize = 7.0 if is_cross_region else 8.0
    legend_fontsize = 8.0 if is_cross_region else 9.0
    suptitle_fontsize = 8.5 if is_cross_region else 10.0
    annotation_fontsize = min(float(settings.mean_lag_annotation_fontsize), 6.5) if is_cross_region else float(settings.mean_lag_annotation_fontsize)
    fig, axes = plt.subplots(
        1,
        len(group_items),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=False,
        facecolor="white",
    )
    plot_axes = list(np.ravel(axes))

    for plot_idx, (axis, (group_label, traces_by_condition)) in enumerate(zip(plot_axes, group_items)):
        for condition in settings.condition_order:
            condition = str(condition)
            traces = [
                np.asarray(trace, dtype=np.float64).reshape(-1)
                for trace in traces_by_condition.get(condition, [])
                if np.asarray(trace, dtype=np.float64).reshape(-1).shape == x_axis.shape
            ]
            if not traces:
                continue
            stacked = np.vstack([trace[plot_mask] for trace in traces])
            if stacked.size <= 0 or stacked.shape[1] <= 0:
                continue
            mean_trace = np.nanmean(stacked, axis=0)
            if stacked.shape[0] >= 2:
                sem_trace = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(float(stacked.shape[0]))
            else:
                sem_trace = np.zeros_like(mean_trace)
            color = settings.condition_colors.get(condition, "#333333")
            axis.fill_between(
                plot_x_axis,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                color=color,
                alpha=float(settings.sem_alpha),
                linewidth=0.0,
                zorder=1,
            )
            axis.plot(
                plot_x_axis,
                mean_trace,
                color=color,
                linewidth=float(settings.mean_trace_linewidth),
                zorder=2,
            )

        mean_rows = (
            mean_lag_df.loc[mean_lag_df["group_label"].astype(str) == str(group_label)].copy()
            if not mean_lag_df.empty
            else pd.DataFrame()
        )
        annotation = _format_mean_lag_annotation(settings, mean_rows)
        if annotation:
            axis.text(
                0.02,
                0.98,
                annotation,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=annotation_fontsize,
                color="#222222",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
            )
        axis.set_title(str(group_label), fontsize=title_fontsize)
        axis.set_xlabel(x_label, fontsize=axis_label_fontsize)
        axis.set_ylabel("Cross-correlation" if (not is_cross_region or plot_idx == 0) else "", fontsize=axis_label_fontsize)
        axis.tick_params(axis="x", which="both", labelbottom=True, labelsize=tick_label_fontsize)
        axis.tick_params(axis="y", which="both", labelsize=tick_label_fontsize)
        axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        if is_cross_region:
            sci_formatter = mticker.ScalarFormatter(useMathText=True)
            sci_formatter.set_scientific(True)
            sci_formatter.set_powerlimits((0, 0))
            axis.yaxis.set_major_formatter(sci_formatter)
            axis.yaxis.get_offset_text().set_fontsize(max(6.0, tick_label_fontsize - 0.5))
        if x_label == "Lag (ms)":
            axis.set_xlim(-half_window_ms, half_window_ms)
            axis.set_xticks(tick_values)
            axis.set_xticklabels([f"{int(round(value))}" for value in tick_values])

    for axis in plot_axes[len(group_items):]:
        axis.set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=settings.condition_colors.get(str(condition), "#333333"),
            linewidth=float(settings.mean_trace_linewidth),
            label=settings.condition_labels.get(str(condition), str(condition)),
        )
        for condition in settings.condition_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93 if is_cross_region else 0.94),
        ncol=max(1, len(legend_handles)),
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.suptitle(
        f"{result.get('subset_label')} | {analysis_kind} | {result.get('signal_variant')}",
        y=0.975 if is_cross_region else 0.985,
        fontsize=suptitle_fontsize,
    )
    if is_cross_region:
        fig.subplots_adjust(left=0.10, right=0.995, top=0.79, bottom=0.18, wspace=0.46)
    else:
        fig.subplots_adjust(left=0.06, right=0.995, top=0.82, bottom=0.16, wspace=0.24)
    save_figure(
        fig,
        output_path,
        ext=str(output_path.suffix).lstrip('.'),
        dpi=dpi,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)



def _plot_single_subset(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    dates: Optional[Sequence[str]],
    analysis_kinds: Optional[Sequence[str]],
    cfg_figsize: Optional[Sequence[float]],
    cfg_dpi: Optional[int],
    output_ext: str,
    is_per_day: bool,
) -> dict[str, object]:
    payload = build_fixation_neural_cross_correlation_pair_condition_mean_plot_payload(
        settings,
        dates=dates,
        analysis_kinds=analysis_kinds,
    )
    cfg = payload["cfg"]
    subset_label = str(payload.get("subset_label") or "all_dates")
    ext = _ensure_ext(output_ext, fallback="pdf")

    figure_outputs: list[str] = []
    mean_lag_stat_outputs: list[str] = []

    for key in sorted(payload["results"].keys()):
        result = payload["results"][key]
        analysis_kind, signal_column = key
        output_path, mean_lag_path, _ = _build_output_paths(
            cfg,
            settings,
            analysis_kind=analysis_kind,
            signal_column=signal_column,
            subset_label=subset_label,
            ext=ext,
            is_per_day=is_per_day,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mean_lag_path.parent.mkdir(parents=True, exist_ok=True)
        mean_lag_df = result.get("mean_lag_comparisons")
        mean_lag_df = mean_lag_df if isinstance(mean_lag_df, pd.DataFrame) else pd.DataFrame()
        mean_lag_df.to_csv(mean_lag_path, index=False)
        _plot_result(
            settings,
            result=result,
            output_path=output_path,
            cfg_figsize=cfg_figsize,
            cfg_dpi=cfg_dpi,
        )
        figure_outputs.append(str(output_path))
        mean_lag_stat_outputs.append(str(mean_lag_path))

    return {
        "figure_outputs": figure_outputs,
        "mean_lag_stat_outputs": mean_lag_stat_outputs,
        "per_lag_stat_outputs": [],
        "results": payload["results"],
        "analysis_kinds": payload.get("analysis_kinds"),
        "subset_label": subset_label,
    }



def plot_fixation_neural_cross_correlation_pair_condition_mean_summaries(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    analysis_kinds: Optional[Sequence[str]] = None,
    include_per_day_when_dates_unspecified: bool = True,
) -> dict[str, object]:
    """Compute pair-condition mean stats and generate summary plots."""
    cfg_figsize = None
    cfg_dpi = None
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        cfg_figsize, cfg_dpi = resolve_figsize(plot_cfg)

    selected_kinds = tuple(analysis_kinds) if analysis_kinds is not None else tuple(_PLOT_ALLOWED_ANALYSIS_KINDS)
    signal_columns = tuple(
        str(value)
        for value in (
            settings.signal_input_columns
            if settings.signal_input_columns is not None
            else (settings.signal_input_column,)
        )
    )

    subset_dates_list: list[Optional[Sequence[str]]] = []
    if dates is not None:
        subset_dates_list.append(tuple(str(value) for value in dates if str(value)))
    else:
        subset_dates_list.append(None)
        if include_per_day_when_dates_unspecified:
            for date in _discover_available_dates(
                settings,
                analysis_kinds=selected_kinds,
                signal_columns=signal_columns,
            ):
                subset_dates_list.append((str(date),))

    figure_outputs: list[str] = []
    mean_lag_stat_outputs: list[str] = []
    results_by_subset: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
    subset_labels: list[str] = []

    for subset_dates in subset_dates_list:
        is_per_day = subset_dates is not None and len(subset_dates) == 1
        subset_result = _plot_single_subset(
            settings,
            dates=subset_dates,
            analysis_kinds=selected_kinds,
            cfg_figsize=cfg_figsize,
            cfg_dpi=cfg_dpi,
            output_ext=("png" if is_per_day else str(settings.output_extension)),
            is_per_day=is_per_day,
        )
        subset_label = str(subset_result.get("subset_label") or _build_subset_label(subset_dates))
        results_by_subset[subset_label] = subset_result.get("results", {}) or {}
        subset_labels.append(subset_label)
        figure_outputs.extend(list(subset_result.get("figure_outputs", [])))
        mean_lag_stat_outputs.extend(list(subset_result.get("mean_lag_stat_outputs", [])))

    primary_subset = subset_labels[0] if subset_labels else _build_subset_label(dates)
    return {
        "figure_outputs": figure_outputs,
        "mean_lag_stat_outputs": mean_lag_stat_outputs,
        "per_lag_stat_outputs": [],
        "results": results_by_subset.get(primary_subset, {}),
        "results_by_subset": results_by_subset,
        "analysis_kinds": selected_kinds,
        "subset_label": primary_subset,
        "subset_labels": subset_labels,
    }



def plot_within_region_fixation_neural_cross_correlation_pair_condition_mean_summaries(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Generate within-region pair-condition mean xcorr summary plots."""
    return plot_fixation_neural_cross_correlation_pair_condition_mean_summaries(
        settings,
        dates=dates,
        analysis_kinds=(WITHIN_ANALYSIS_KIND,),
    )



def plot_cross_region_fixation_neural_cross_correlation_pair_condition_mean_summaries(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Generate cross-region pair-condition mean xcorr summary plots."""
    return plot_fixation_neural_cross_correlation_pair_condition_mean_summaries(
        settings,
        dates=dates,
        analysis_kinds=(CROSS_ANALYSIS_KIND,),
    )
