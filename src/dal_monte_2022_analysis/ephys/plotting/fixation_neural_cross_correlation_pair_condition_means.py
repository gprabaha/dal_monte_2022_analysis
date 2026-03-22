"""Plot pair-condition mean neural xcorr summaries and condition-comparison statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.stats import ttest_rel

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.stats.hypothesis import paired_ttest_per_lag
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
from dal_monte_2022_analysis.runtime.execution.parallel import get_n_processes
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
)
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
    "face_interactive": "#d62728",
    "face_non_interactive": "#1f77b4",
    "object": "#2ca02c",
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
    lag_tick_step_ms: float = 20.0
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



def _build_output_paths(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    analysis_kind: str,
    signal_column: str,
    subset_label: str,
    ext: str,
) -> tuple[Path, Path, Path]:
    root = build_analysis_output_dir(cfg, settings.output_subdir) / analysis_kind
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
) -> tuple[dict[str, list[np.ndarray]], list[dict[str, object]], list[dict[str, object]]]:
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
    per_lag_rows: list[dict[str, object]] = []
    pairwise_keys = list(combinations([str(name) for name in settings.condition_order], 2))
    mean_lag_pvals = np.full(len(pairwise_keys), np.nan, dtype=np.float64)
    raw_lag_pvals: list[np.ndarray] = []
    paired_mats: list[tuple[np.ndarray, np.ndarray]] = []

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

        if (
            paired_a.ndim != 2
            or paired_b.ndim != 2
            or paired_a.shape != paired_b.shape
            or paired_a.shape[0] < int(max(2, settings.min_pairs_for_significance))
        ):
            raw_lag_pvals.append(np.full(paired_a.shape[1] if paired_a.ndim == 2 else 0, np.nan, dtype=np.float64))
            paired_mats.append((paired_a, paired_b))
            continue

        lag_p = paired_ttest_per_lag(
            paired_a,
            paired_b,
            parallel=bool(settings.use_parallel),
            workers=n_lag_workers,
            min_lags_for_parallel=int(max(1, settings.lag_test_min_lags_for_parallel)),
            chunk_size=int(max(1, settings.lag_test_chunk_size)),
        )
        raw_lag_pvals.append(np.asarray(lag_p, dtype=np.float64).reshape(-1))
        paired_mats.append((paired_a, paired_b))

    mean_sig = _apply_pvalue_correction(
        mean_lag_pvals,
        alpha=float(settings.significance_alpha),
        correction=_resolve_significance_correction(settings.mean_lag_significance_correction),
    )
    for idx, is_sig in enumerate(np.asarray(mean_sig, dtype=bool).reshape(-1)):
        mean_lag_rows[idx]["significant"] = bool(is_sig)

    if raw_lag_pvals:
        max_lags = max((arr.size for arr in raw_lag_pvals), default=0)
    else:
        max_lags = 0
    lag_p_mat = np.full((len(raw_lag_pvals), max_lags), np.nan, dtype=np.float64)
    for idx, arr in enumerate(raw_lag_pvals):
        lag_p_mat[idx, : arr.size] = arr
    lag_sig_mat = _apply_pvalue_correction(
        lag_p_mat,
        alpha=float(settings.significance_alpha),
        correction=_resolve_significance_correction(settings.per_lag_significance_correction),
    ) if lag_p_mat.size else np.zeros_like(lag_p_mat, dtype=bool)

    for idx, (condition_a, condition_b) in enumerate(pairwise_keys):
        paired_a, paired_b = paired_mats[idx]
        mean_diff_trace = (
            np.nanmean(paired_a - paired_b, axis=0).astype(np.float32)
            if paired_a.ndim == 2 and paired_a.size and paired_b.ndim == 2 and paired_b.size
            else np.asarray([], dtype=np.float32)
        )
        per_lag_rows.append(
            {
                "group_label": str(group_label),
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_pairs": int(mean_lag_rows[idx]["n_pairs"]),
                "p_values": lag_p_mat[idx, :].astype(np.float32),
                "significant_mask": np.asarray(lag_sig_mat[idx, :], dtype=bool),
                "mean_difference_trace": mean_diff_trace,
            }
        )

    return traces_by_condition, mean_lag_rows, per_lag_rows



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
    n_lag_workers = get_n_processes(max_procs=settings.max_procs) if settings.use_parallel else None
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
                    n_lag_workers=n_lag_workers,
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
    cfg_figsize: Optional[Sequence[float]],
    cfg_dpi: Optional[int],
    n_groups: int,
) -> tuple[list[float], Optional[int]]:
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
    x_axis = np.asarray(result.get("x_axis", []), dtype=np.float64).reshape(-1)
    if x_axis.size <= 0:
        return
    mean_lag_df = result.get("mean_lag_comparisons")
    mean_lag_df = mean_lag_df if isinstance(mean_lag_df, pd.DataFrame) else pd.DataFrame()
    per_lag_df = result.get("per_lag_comparisons")
    per_lag_df = per_lag_df if isinstance(per_lag_df, pd.DataFrame) else pd.DataFrame()

    group_items = sorted(group_plot_map.items(), key=lambda item: item[0])
    figsize, dpi = _resolve_figsize_and_dpi(
        settings,
        cfg_figsize=cfg_figsize,
        cfg_dpi=cfg_dpi,
        n_groups=len(group_items),
    )
    fig, axes = plt.subplots(
        2,
        len(group_items),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
        facecolor="white",
        gridspec_kw={"height_ratios": [1.0, 0.2]},
    )
    plot_axes = list(np.ravel(axes[0, :]))
    legend_axes = list(np.ravel(axes[1, :]))
    for legend_axis in legend_axes:
        legend_axis.axis("off")

    marker_map = _build_between_condition_marker_map(settings.condition_order)
    for axis, (group_label, traces_by_condition) in zip(plot_axes, group_items):
        mean_sig_lines: list[str] = []
        lag_sig_rows = per_lag_df.loc[per_lag_df["group_label"].astype(str) == str(group_label)].copy() if not per_lag_df.empty else pd.DataFrame()
        for condition in settings.condition_order:
            condition = str(condition)
            traces = [np.asarray(trace, dtype=np.float64) for trace in traces_by_condition.get(condition, [])]
            if not traces:
                continue
            stacked = np.vstack(traces)
            mean_trace = np.nanmean(stacked, axis=0)
            if stacked.shape[0] >= 2:
                sem_trace = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(float(stacked.shape[0]))
            else:
                sem_trace = np.zeros_like(mean_trace)
            color = settings.condition_colors.get(condition, "#333333")
            axis.fill_between(
                x_axis,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                color=color,
                alpha=float(settings.sem_alpha),
                linewidth=0.0,
                zorder=1,
            )
            axis.plot(
                x_axis,
                mean_trace,
                color=color,
                linewidth=float(settings.mean_trace_linewidth),
                zorder=2,
            )

        if not mean_lag_df.empty:
            mean_rows = mean_lag_df.loc[mean_lag_df["group_label"].astype(str) == str(group_label)].copy()
            for row in mean_rows.itertuples(index=False):
                if bool(getattr(row, "significant", False)):
                    left = settings.condition_short_labels.get(str(row.condition_a), str(row.condition_a))
                    right = settings.condition_short_labels.get(str(row.condition_b), str(row.condition_b))
                    mean_sig_lines.append(f"{left} vs {right}")

        sig_pair_rows = []
        if not lag_sig_rows.empty:
            for row in lag_sig_rows.itertuples(index=False):
                sig_mask = np.asarray(getattr(row, "significant_mask", []), dtype=bool).reshape(-1)
                if sig_mask.any():
                    sig_pair_rows.append((row.condition_a, row.condition_b, sig_mask))
        if sig_pair_rows:
            y_min, y_max = axis.get_ylim()
            span = float(y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 1.0
            top_pad = 0.08 * span * float(len(sig_pair_rows))
            axis.set_ylim(y_min, y_max + top_pad)
            y_min2, y_max2 = axis.get_ylim()
            span2 = float(y_max2 - y_min2) if np.isfinite(y_max2 - y_min2) and (y_max2 > y_min2) else 1.0
            for idx, (condition_a, condition_b, sig_mask) in enumerate(sig_pair_rows):
                marker = marker_map.get((str(condition_a), str(condition_b)), "|")
                x_marks = x_axis[sig_mask[: x_axis.size]]
                if x_marks.size == 0:
                    continue
                y_level = y_max2 - ((idx + 1) * 0.04 * span2)
                axis.plot(
                    x_marks,
                    np.full(x_marks.shape, y_level, dtype=np.float64),
                    linestyle="None",
                    marker=marker,
                    color="#111111",
                    alpha=float(settings.between_condition_marker_alpha),
                    markersize=float(settings.between_condition_marker_size),
                    zorder=3,
                )

        if mean_sig_lines:
            axis.text(
                0.02,
                0.98,
                "Mean-lag sig: " + ", ".join(mean_sig_lines),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=float(settings.mean_lag_annotation_fontsize),
                color="#222222",
            )

        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)
        axis.set_title(str(group_label))
        axis.set_xlabel(str(result.get("x_label") or "Lag"))
        axis.set_ylabel("Cross-correlation")
        axis.tick_params(axis="x", which="both", labelbottom=True)
        if str(result.get("x_label") or "") == "Lag (ms)":
            tick_step_ms = float(max(1.0, settings.lag_tick_step_ms))
            axis.xaxis.set_major_locator(MultipleLocator(tick_step_ms))

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
    for pair_key in combinations([str(name) for name in settings.condition_order], 2):
        left = settings.condition_labels.get(pair_key[0], pair_key[0])
        right = settings.condition_labels.get(pair_key[1], pair_key[1])
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#111111",
                marker=marker_map.get(pair_key, "|"),
                linestyle="None",
                markersize=float(settings.between_condition_marker_size),
                alpha=float(settings.between_condition_marker_alpha),
                label=(
                    f"{left} vs {right} "
                    f"(lag p < {float(settings.significance_alpha):g}, "
                    f"{_resolve_significance_correction(settings.per_lag_significance_correction)})"
                ),
            )
        )
    legend_anchor = legend_axes[len(legend_axes) // 2]
    legend_anchor.legend(handles=legend_handles, loc="center", ncol=1, frameon=False)

    fig.suptitle(
        f"{result.get('subset_label')} | {result.get('analysis_kind')} | {result.get('signal_variant')}",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.08, wspace=0.24, hspace=0.18)
    save_figure(
        fig,
        output_path,
        ext=str(output_path.suffix).lstrip("."),
        dpi=dpi,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)



def plot_fixation_neural_cross_correlation_pair_condition_mean_summaries(
    settings: FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    analysis_kinds: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Compute pair-condition comparison stats and generate summary plots."""
    plot_cfg = None
    cfg_figsize = None
    cfg_dpi = None
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        cfg_figsize, cfg_dpi = resolve_figsize(plot_cfg)

    payload = build_fixation_neural_cross_correlation_pair_condition_mean_plot_payload(
        settings,
        dates=dates,
        analysis_kinds=analysis_kinds,
    )
    cfg = payload["cfg"]
    subset_label = str(payload.get("subset_label") or "all_dates")
    ext = _ensure_ext(settings.output_extension, fallback="pdf")

    figure_outputs: list[str] = []
    mean_lag_stat_outputs: list[str] = []
    per_lag_stat_outputs: list[str] = []

    for key in sorted(payload["results"].keys()):
        result = payload["results"][key]
        analysis_kind, signal_column = key
        output_path, mean_lag_path, per_lag_path = _build_output_paths(
            cfg,
            settings,
            analysis_kind=analysis_kind,
            signal_column=signal_column,
            subset_label=subset_label,
            ext=ext,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mean_lag_path.parent.mkdir(parents=True, exist_ok=True)
        mean_lag_df = result.get("mean_lag_comparisons")
        mean_lag_df = mean_lag_df if isinstance(mean_lag_df, pd.DataFrame) else pd.DataFrame()
        mean_lag_df.to_csv(mean_lag_path, index=False)
        save_pickle_path(
            {
                "meta": {
                    "analysis_kind": analysis_kind,
                    "signal_input_column": str(result.get("signal_input_column")),
                    "signal_variant": str(result.get("signal_variant")),
                    "subset_label": subset_label,
                    "condition_order": [str(name) for name in settings.condition_order],
                    "significance_alpha": float(settings.significance_alpha),
                    "mean_lag_significance_correction": _resolve_significance_correction(
                        settings.mean_lag_significance_correction
                    ),
                    "per_lag_significance_correction": _resolve_significance_correction(
                        settings.per_lag_significance_correction
                    ),
                    "loaded_paths": result.get("loaded_paths") or [],
                    "counts": result.get("counts") or {},
                    "lags": result.get("lags"),
                    "bin_size_ms": result.get("bin_size_ms"),
                },
                "mean_lag_comparisons": mean_lag_df,
                "per_lag_comparisons": result.get("per_lag_comparisons"),
            },
            per_lag_path,
        )
        _plot_result(
            settings,
            result=result,
            output_path=output_path,
            cfg_figsize=cfg_figsize,
            cfg_dpi=cfg_dpi,
        )
        figure_outputs.append(str(output_path))
        mean_lag_stat_outputs.append(str(mean_lag_path))
        per_lag_stat_outputs.append(str(per_lag_path))

    return {
        "figure_outputs": figure_outputs,
        "mean_lag_stat_outputs": mean_lag_stat_outputs,
        "per_lag_stat_outputs": per_lag_stat_outputs,
        "results": payload["results"],
        "analysis_kinds": payload.get("analysis_kinds"),
        "subset_label": subset_label,
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
