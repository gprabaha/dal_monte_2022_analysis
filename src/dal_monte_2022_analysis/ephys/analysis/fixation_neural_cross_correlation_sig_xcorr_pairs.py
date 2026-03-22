"""Aggregate fixation neural xcorr outputs into date-level significant-pair summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_optional_str as _as_optional_str,
    ensure_filename as _ensure_filename,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_helpers import (
    CROSS_ANALYSIS_KIND,
    DEFAULT_FIXATION_ROI_GROUPS,
    WITHIN_ANALYSIS_KIND,
    _PLOT_ALLOWED_ANALYSIS_KINDS,
    _PLOT_CONDITION_ORDER,
    _assert_lag_axis_match,
    _extract_xcorr_dataframes_and_meta,
    _normalize_region_pair_label,
    _normalize_roi_groups,
    _resolve_plot_condition_from_row,
    _resolve_signal_input_columns,
    _resolve_signal_output_filename,
    _signal_output_label,
    _summarize_cross_correlation,
)
from dal_monte_2022_analysis.runtime.execution.parallel import get_n_processes
from dal_monte_2022_analysis.runtime.io.analysis_index import (
    scan_analysis_date_paths,
    scan_analysis_paths,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
)
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


@dataclass
class FixationNeuralCrossCorrelationPairMetaAnalysisSettings:
    """Configuration for date-level neural xcorr significant-pair analysis."""

    cfg_path: str
    signal_input_column: str = "spike_train_counts"
    signal_input_columns: Optional[Sequence[str]] = None
    within_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/within_region"
    cross_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/cross_region"
    within_input_filename: str = "fixations.pkl"
    cross_input_filename: str = "fixations.pkl"
    within_output_subdir: str = "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/within_region"
    cross_output_subdir: str = "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/cross_region"
    within_output_filename: str = "pair_fixation_lag_mean_significance.pkl"
    cross_output_filename: str = "pair_fixation_lag_mean_significance.pkl"
    within_output_csv_filename: str = "pair_fixation_lag_mean_significance.csv"
    cross_output_csv_filename: str = "pair_fixation_lag_mean_significance.csv"
    face_label: str = "face"
    object_label: str = "object"
    interactive_label: str = "interactive"
    roi_groups: dict[str, Sequence[str]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_FIXATION_ROI_GROUPS.items()},
    )
    condition_order: Sequence[str] = field(default_factory=lambda: tuple(_PLOT_CONDITION_ORDER))
    alpha: float = 0.05
    min_fixations: int = 2
    use_parallel: bool = True
    max_procs: Optional[int] = None
    parallelize_across_dates: bool = True


def resolve_fixation_neural_cross_correlation_pair_meta_analysis_signal_columns(
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
) -> tuple[str, ...]:
    """Resolve the configured signal columns in run order."""
    return _resolve_signal_input_columns(settings)


def build_fixation_neural_cross_correlation_pair_meta_analysis_settings_from_config(
    *,
    dataset_cfg_path: str,
    ephys_fixation_neural_cross_correlation_cfg_path: str | None = None,
    ephys_fixation_neural_crosscorr_cfg_path: str | None = None,
) -> FixationNeuralCrossCorrelationPairMetaAnalysisSettings:
    """Build sig-xcorr-pair settings from dataset + task config paths."""
    if (
        ephys_fixation_neural_cross_correlation_cfg_path is None
        and ephys_fixation_neural_crosscorr_cfg_path is not None
    ):
        warnings.warn(
            (
                "ephys_fixation_neural_crosscorr_cfg_path is deprecated; "
                "use ephys_fixation_neural_cross_correlation_cfg_path instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
    cfg_path = (
        ephys_fixation_neural_cross_correlation_cfg_path
        or ephys_fixation_neural_crosscorr_cfg_path
    )
    if cfg_path is None:
        raise ValueError(
            "Expected one of ephys_fixation_neural_cross_correlation_cfg_path "
            "or ephys_fixation_neural_crosscorr_cfg_path.",
        )

    cfg = load_config(cfg_path)
    condition_order = cfg.get(
        "pair_meta_condition_order",
        cfg.get("plot_condition_order", tuple(_PLOT_CONDITION_ORDER)),
    )
    return FixationNeuralCrossCorrelationPairMetaAnalysisSettings(
        cfg_path=dataset_cfg_path,
        signal_input_column=cfg.get("signal_input_column", "spike_train_counts"),
        signal_input_columns=cfg.get("signal_input_columns"),
        within_input_subdir=cfg.get(
            "within_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/within_region",
        ),
        cross_input_subdir=cfg.get(
            "cross_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/cross_region",
        ),
        within_input_filename=cfg.get("within_output_filename", "fixations.pkl"),
        cross_input_filename=cfg.get("cross_output_filename", "fixations.pkl"),
        within_output_subdir=cfg.get(
            "pair_meta_within_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/within_region",
        ),
        cross_output_subdir=cfg.get(
            "pair_meta_cross_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/cross_region",
        ),
        within_output_filename=cfg.get(
            "pair_meta_within_output_filename",
            "pair_fixation_lag_mean_significance.pkl",
        ),
        cross_output_filename=cfg.get(
            "pair_meta_cross_output_filename",
            "pair_fixation_lag_mean_significance.pkl",
        ),
        within_output_csv_filename=cfg.get(
            "pair_meta_within_output_csv_filename",
            "pair_fixation_lag_mean_significance.csv",
        ),
        cross_output_csv_filename=cfg.get(
            "pair_meta_cross_output_csv_filename",
            "pair_fixation_lag_mean_significance.csv",
        ),
        face_label=cfg.get("pair_meta_face_label", "face"),
        object_label=cfg.get("pair_meta_object_label", "object"),
        interactive_label=cfg.get("pair_meta_interactive_label", "interactive"),
        roi_groups=cfg.get("roi_groups"),
        condition_order=tuple(str(value) for value in condition_order),
        alpha=float(cfg.get("pair_meta_alpha", 0.05)),
        min_fixations=max(1, int(cfg.get("pair_meta_min_fixations", 2))),
        use_parallel=cfg.get("pair_meta_use_parallel", cfg.get("use_parallel", True)),
        max_procs=cfg.get("pair_meta_max_procs", cfg.get("max_procs")),
        parallelize_across_dates=cfg.get("pair_meta_parallelize_across_dates", True),
    )


def _resolve_pair_meta_paths(
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    *,
    analysis_kind: str,
) -> tuple[str, str, str, str, str]:
    if analysis_kind == WITHIN_ANALYSIS_KIND:
        return (
            settings.within_input_subdir,
            settings.within_input_filename,
            settings.within_output_subdir,
            settings.within_output_filename,
            settings.within_output_csv_filename,
        )
    if analysis_kind == CROSS_ANALYSIS_KIND:
        return (
            settings.cross_input_subdir,
            settings.cross_input_filename,
            settings.cross_output_subdir,
            settings.cross_output_filename,
            settings.cross_output_csv_filename,
        )
    raise ValueError(f"Unsupported analysis_kind={analysis_kind!r}.")


def _join_tokens(values: Sequence[str]) -> Optional[str]:
    tokens = sorted({str(value).strip() for value in values if str(value).strip()})
    if not tokens:
        return None
    return "|".join(tokens)


def _sort_pair_summary_dataframe(
    df: pd.DataFrame,
    *,
    condition_order: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    order_map = {str(name): idx for idx, name in enumerate(condition_order)}
    work = df.copy()
    work["_condition_sort"] = work["condition"].map(
        lambda value: order_map.get(str(value), len(order_map)),
    )
    sort_cols = [
        "date",
        "group_label",
        "region_1",
        "unit_uuid_1",
        "region_2",
        "unit_uuid_2",
        "_condition_sort",
    ]
    available = [col for col in sort_cols if col in work.columns]
    if available:
        work = work.sort_values(available).reset_index(drop=True)
    return work.drop(columns=["_condition_sort"], errors="ignore")


def _build_pair_summary_key(
    row,
    *,
    analysis_kind: str,
    condition: str,
) -> tuple[str, str, str, str, str, str]:
    region_1 = _as_optional_str(getattr(row, "region_1", None)) or "unknown_1"
    region_2 = _as_optional_str(getattr(row, "region_2", None)) or region_1
    unit_uuid_1 = _as_optional_str(getattr(row, "unit_uuid_1", None)) or "unknown_unit_1"
    unit_uuid_2 = _as_optional_str(getattr(row, "unit_uuid_2", None)) or "unknown_unit_2"
    if analysis_kind == WITHIN_ANALYSIS_KIND:
        group_label = region_1
        region_2 = region_1
    elif analysis_kind == CROSS_ANALYSIS_KIND:
        group_label = _normalize_region_pair_label(region_1, region_2)
    else:
        raise ValueError(f"Unsupported analysis_kind={analysis_kind!r}.")
    return (group_label, region_1, region_2, unit_uuid_1, unit_uuid_2, str(condition))


def _compute_lag_mean_significance(
    values: np.ndarray,
    *,
    alpha: float,
    min_fixations: int,
) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n_fixations = int(arr.size)
    mean_value = float(np.mean(arr)) if n_fixations else None
    if n_fixations >= 2:
        lag_mean_std = float(np.std(arr, ddof=1))
        lag_mean_sem = float(lag_mean_std / np.sqrt(float(n_fixations)))
    else:
        lag_mean_std = None
        lag_mean_sem = None

    min_required = int(max(2, min_fixations))
    if n_fixations < min_required:
        return {
            "n_fixations_tested": n_fixations,
            "lag_mean": mean_value,
            "lag_mean_std": lag_mean_std,
            "lag_mean_sem": lag_mean_sem,
            "t_statistic": None,
            "p_value": None,
            "tested": False,
            "significant_above_zero": False,
        }

    if np.allclose(arr, arr[0]):
        if float(arr[0]) > 0.0:
            t_stat = float("inf")
            p_value = 0.0
        elif float(arr[0]) < 0.0:
            t_stat = float("-inf")
            p_value = 1.0
        else:
            t_stat = 0.0
            p_value = 1.0
    else:
        res = ttest_1samp(arr, popmean=0.0, nan_policy="omit")
        t_stat = float(np.asarray(res.statistic, dtype=np.float64).reshape(()))
        p_two = float(np.asarray(res.pvalue, dtype=np.float64).reshape(()))
        if not np.isfinite(t_stat) or not np.isfinite(p_two):
            p_value = None
        elif t_stat > 0.0:
            p_value = p_two / 2.0
        else:
            p_value = 1.0 - (p_two / 2.0)

    significant = bool(
        mean_value is not None
        and mean_value > 0.0
        and p_value is not None
        and np.isfinite(p_value)
        and float(p_value) < float(alpha)
    )
    return {
        "n_fixations_tested": n_fixations,
        "lag_mean": mean_value,
        "lag_mean_std": lag_mean_std,
        "lag_mean_sem": lag_mean_sem,
        "t_statistic": t_stat,
        "p_value": p_value,
        "tested": True,
        "significant_above_zero": significant,
    }


def _build_empty_trace_summary(n_lags: int) -> dict[str, object]:
    return {
        "n_lags": int(max(0, n_lags)),
        "zero_lag_correlation": None,
        "peak_lag": None,
        "peak_correlation": None,
    }




def _group_summary_count_column(condition: str) -> str:
    return f"n_sig_{str(condition)}_pairs"



def _sort_group_summary_dataframe(
    df: pd.DataFrame,
    *,
    condition_order: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    sort_cols = ["date", "group_label", "region_1", "region_2"]
    available = [col for col in sort_cols if col in work.columns]
    if available:
        work = work.sort_values(available).reset_index(drop=True)

    ordered = [
        col
        for col in (
            "date",
            "group_label",
            "region_1",
            "region_2",
            "n_total_pairs",
            *[_group_summary_count_column(condition) for condition in condition_order],
            "n_sig_any_condition_pairs",
        )
        if col in work.columns
    ]
    remaining = [col for col in work.columns if col not in ordered]
    return work.loc[:, ordered + remaining]



def _build_group_sig_xcorr_pair_summary_dataframe(
    pair_summaries: pd.DataFrame,
    *,
    condition_order: Sequence[str],
) -> pd.DataFrame:
    count_cols = [_group_summary_count_column(condition) for condition in condition_order]
    base_columns = [
        "date",
        "group_label",
        "region_1",
        "region_2",
        "n_total_pairs",
        *count_cols,
        "n_sig_any_condition_pairs",
    ]
    if pair_summaries.empty:
        return pd.DataFrame(columns=base_columns)

    required_cols = {
        "group_label",
        "region_1",
        "region_2",
        "unit_uuid_1",
        "unit_uuid_2",
        "condition",
        "significant_above_zero",
    }
    if not required_cols.issubset(set(pair_summaries.columns)):
        return pd.DataFrame(columns=base_columns)

    work = pair_summaries.copy()
    pair_id_cols = ["group_label", "region_1", "region_2", "unit_uuid_1", "unit_uuid_2"]
    group_cols = ["group_label", "region_1", "region_2"]
    if "date" in work.columns:
        pair_id_cols = ["date", *pair_id_cols]
        group_cols = ["date", *group_cols]

    unique_pairs = work.loc[:, pair_id_cols].drop_duplicates().reset_index(drop=True)
    total_counts = {
        tuple(key): int(count)
        for key, count in unique_pairs.groupby(group_cols, dropna=False).size().items()
    }

    significant_rows = work.loc[work["significant_above_zero"].fillna(False).astype(bool)].copy()
    any_counts = {
        tuple(key): int(count)
        for key, count in significant_rows.loc[:, pair_id_cols]
        .drop_duplicates()
        .groupby(group_cols, dropna=False)
        .size()
        .items()
    } if not significant_rows.empty else {}

    condition_counts: dict[str, dict[tuple, int]] = {}
    for condition in condition_order:
        cond_rows = significant_rows.loc[
            significant_rows["condition"].astype(str) == str(condition),
            pair_id_cols,
        ]
        if cond_rows.empty:
            condition_counts[str(condition)] = {}
            continue
        condition_counts[str(condition)] = {
            tuple(key): int(count)
            for key, count in cond_rows.drop_duplicates().groupby(group_cols, dropna=False).size().items()
        }

    group_rows = unique_pairs.loc[:, group_cols].drop_duplicates().reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for group_row in group_rows.itertuples(index=False, name=None):
        key = tuple(group_row)
        row = {col: value for col, value in zip(group_cols, group_row)}
        row["n_total_pairs"] = int(total_counts.get(key, 0))
        for condition in condition_order:
            row[_group_summary_count_column(condition)] = int(
                condition_counts.get(str(condition), {}).get(key, 0)
            )
        row["n_sig_any_condition_pairs"] = int(any_counts.get(key, 0))
        rows.append(row)

    return _sort_group_summary_dataframe(pd.DataFrame(rows), condition_order=condition_order)



def _build_date_pair_meta_result(
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    *,
    analysis_kind: str,
    date: str,
    signal_column: str,
    rows: Sequence[dict],
) -> dict[str, object]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, list[object]]] = {}
    lag_axis: Optional[np.ndarray] = None
    trace_n_lags: Optional[int] = None
    signal_bin_size_ms: Optional[float] = None
    source_sessions: set[str] = set()
    input_paths: list[str] = []
    source_row_count = 0
    used_row_count = 0
    skipped_row_count = 0
    valid_conditions = {str(name) for name in settings.condition_order}
    roi_groups = _normalize_roi_groups(settings.roi_groups)

    for session_row in rows:
        input_paths.append(str(session_row["path"]))
        source_sessions.add(str(session_row["session"]))
        obj = load_pickle_path(session_row["path"])
        xcorr_df, _pair_avg_df, meta = _extract_xcorr_dataframes_and_meta(obj)
        if xcorr_df.empty or "cross_correlation" not in xcorr_df.columns:
            continue

        file_lags = meta.get("lags")
        if file_lags is not None:
            file_lag_axis = np.asarray(file_lags, dtype=np.int64).reshape(-1)
            if file_lag_axis.size > 0:
                if lag_axis is None:
                    lag_axis = file_lag_axis
                else:
                    _assert_lag_axis_match(lag_axis, file_lag_axis)
        raw_bin_size_ms = meta.get("bin_size_ms", meta.get("signal_bin_size_ms"))
        if signal_bin_size_ms is None and raw_bin_size_ms is not None:
            try:
                candidate = float(raw_bin_size_ms)
            except Exception:
                candidate = np.nan
            if np.isfinite(candidate) and candidate > 0.0:
                signal_bin_size_ms = float(candidate)

        for xrow in xcorr_df.itertuples(index=False):
            source_row_count += 1
            condition = _resolve_plot_condition_from_row(
                xrow,
                face_label=settings.face_label,
                object_label=settings.object_label,
                interactive_label=settings.interactive_label,
                roi_groups=roi_groups,
            )
            if condition is None or str(condition) not in valid_conditions:
                skipped_row_count += 1
                continue

            trace = np.asarray(getattr(xrow, "cross_correlation", []), dtype=np.float64).reshape(-1)
            if trace.size <= 0:
                skipped_row_count += 1
                continue
            if lag_axis is not None:
                if lag_axis.size != trace.size:
                    raise ValueError(
                        "Encountered cross-correlation trace length mismatch while aggregating "
                        f"date={date} sig-pair analysis for {signal_column}."
                    )
            else:
                if trace_n_lags is None:
                    trace_n_lags = int(trace.size)
                elif int(trace.size) != int(trace_n_lags):
                    raise ValueError(
                        "Encountered cross-correlation trace length mismatch while aggregating "
                        f"date={date} sig-pair analysis for {signal_column}."
                    )

            trace = np.where(np.isfinite(trace), trace, np.nan)
            if not np.isfinite(trace).any():
                skipped_row_count += 1
                continue

            key = _build_pair_summary_key(
                xrow,
                analysis_kind=analysis_kind,
                condition=str(condition),
            )
            bucket = grouped.setdefault(key, {"traces": []})
            bucket["traces"].append(trace)
            used_row_count += 1

    summary_rows: list[dict[str, object]] = []
    for key, bucket in grouped.items():
        stacked = np.asarray(bucket["traces"], dtype=np.float64)
        if stacked.ndim != 2 or stacked.shape[0] <= 0 or stacked.shape[1] <= 0:
            continue

        lag_mean_values = np.nanmean(stacked, axis=1)
        valid_fixation_mask = np.isfinite(lag_mean_values)
        if not np.any(valid_fixation_mask):
            continue
        stacked = stacked[valid_fixation_mask]
        lag_mean_values = lag_mean_values[valid_fixation_mask]

        mean_trace = np.nanmean(stacked, axis=0)
        mean_trace = np.where(np.isfinite(mean_trace), mean_trace, np.nan)
        if not np.isfinite(mean_trace).any():
            continue

        if lag_axis is not None and lag_axis.size == mean_trace.size:
            trace_summary = _summarize_cross_correlation(lag_axis, mean_trace)
        else:
            trace_summary = _build_empty_trace_summary(int(mean_trace.size))

        lag_mean_stats = _compute_lag_mean_significance(
            lag_mean_values,
            alpha=float(settings.alpha),
            min_fixations=int(settings.min_fixations),
        )
        group_label, region_1, region_2, unit_uuid_1, unit_uuid_2, condition = key
        summary_rows.append(
            {
                "date": str(date),
                "group_label": group_label,
                "condition": condition,
                "region_1": region_1,
                "region_2": region_2,
                "unit_uuid_1": unit_uuid_1,
                "unit_uuid_2": unit_uuid_2,
                "n_fixations": int(stacked.shape[0]),
                "mean_cross_correlation_across_lags": lag_mean_stats["lag_mean"],
                "lag_mean_std": lag_mean_stats["lag_mean_std"],
                "lag_mean_sem": lag_mean_stats["lag_mean_sem"],
                "n_fixations_tested": lag_mean_stats["n_fixations_tested"],
                "tested": lag_mean_stats["tested"],
                "t_statistic": lag_mean_stats["t_statistic"],
                "p_value": lag_mean_stats["p_value"],
                "significant_above_zero": lag_mean_stats["significant_above_zero"],
                **trace_summary,
            }
        )

    pair_summaries = _sort_pair_summary_dataframe(
        pd.DataFrame(summary_rows),
        condition_order=settings.condition_order,
    )
    group_summaries = _build_group_sig_xcorr_pair_summary_dataframe(
        pair_summaries,
        condition_order=settings.condition_order,
    )
    if lag_axis is None and trace_n_lags is not None:
        lag_axis = np.arange(trace_n_lags, dtype=np.int64)

    return {
        "meta": {
            "analysis_kind": analysis_kind,
            "date": str(date),
            "signal_input_column": str(signal_column),
            "signal_variant": _signal_output_label(signal_column),
            "alpha": float(settings.alpha),
            "min_fixations": int(settings.min_fixations),
            "condition_order": [str(name) for name in settings.condition_order],
            "n_input_session_files": int(len(rows)),
            "n_input_sessions": int(len(source_sessions)),
            "source_sessions": sorted(source_sessions),
            "source_paths": input_paths,
            "source_row_count": int(source_row_count),
            "used_row_count": int(used_row_count),
            "skipped_row_count": int(skipped_row_count),
            "n_pair_summaries": int(len(pair_summaries)),
            "n_group_summaries": int(len(group_summaries)),
            "lags": lag_axis,
            "bin_size_ms": signal_bin_size_ms,
        },
        "pair_summaries": pair_summaries,
        "group_summaries": group_summaries,
    }

def _build_pair_meta_output_paths(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    *,
    analysis_kind: str,
    date: str,
    signal_column: str,
) -> tuple[Path, Path]:
    _input_subdir, _input_filename, output_subdir, output_filename, output_csv_filename = _resolve_pair_meta_paths(
        settings,
        analysis_kind=analysis_kind,
    )
    out_root = build_analysis_output_dir(cfg, str(output_subdir).rstrip("/")) / f"date={date}"
    resolved_pkl = _resolve_signal_output_filename(output_filename, signal_column)
    resolved_csv = _resolve_signal_output_filename(output_csv_filename, signal_column)
    return (
        out_root / _ensure_filename(resolved_pkl, ".pkl"),
        out_root / _ensure_filename(resolved_csv, ".csv"),
    )


def _build_and_save_date_pair_meta_result(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    *,
    analysis_kind: str,
    date: str,
    signal_column: str,
    rows: Sequence[dict],
) -> dict[str, object]:
    result = _build_date_pair_meta_result(
        settings,
        analysis_kind=analysis_kind,
        date=str(date),
        signal_column=str(signal_column),
        rows=rows,
    )
    out_pkl, out_csv = _build_pair_meta_output_paths(
        cfg,
        settings,
        analysis_kind=analysis_kind,
        date=str(date),
        signal_column=str(signal_column),
    )
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    save_pickle_path(result, out_pkl)
    pair_summaries = result.get("pair_summaries")
    csv_df = pair_summaries if isinstance(pair_summaries, pd.DataFrame) else pd.DataFrame()
    csv_df.to_csv(out_csv, index=False)
    group_summaries = result.get("group_summaries")
    group_csv_df = group_summaries if isinstance(group_summaries, pd.DataFrame) else pd.DataFrame()
    group_out_csv = out_csv.with_name(f"{out_csv.stem}_group_summary.csv")
    group_csv_df.to_csv(group_out_csv, index=False)
    return {
        "date": str(date),
        "output_path": str(out_pkl),
        "csv_output_path": str(out_csv),
        "group_summary_csv_output_path": str(group_out_csv),
        "n_summary_rows": int(len(csv_df)),
        "n_group_summary_rows": int(len(group_csv_df)),
    }


def _build_and_save_date_pair_meta_result_worker(
    task: tuple[
        dict,
        FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
        str,
        str,
        str,
        Sequence[dict],
    ],
) -> dict[str, object]:
    cfg, settings, analysis_kind, date, signal_column, rows = task
    return _build_and_save_date_pair_meta_result(
        cfg,
        settings,
        analysis_kind=analysis_kind,
        date=str(date),
        signal_column=str(signal_column),
        rows=rows,
    )


def _run_fixation_neural_cross_correlation_pair_meta_analysis(
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    *,
    analysis_kind: str,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict[str, object]:
    if analysis_kind not in _PLOT_ALLOWED_ANALYSIS_KINDS:
        raise ValueError(
            f"Unsupported analysis_kind={analysis_kind!r}. "
            f"Expected one of: {', '.join(_PLOT_ALLOWED_ANALYSIS_KINDS)}."
        )

    cfg = load_config(settings.cfg_path)
    signal_columns = resolve_fixation_neural_cross_correlation_pair_meta_analysis_signal_columns(settings)
    signal_summaries: dict[str, dict[str, object]] = {}
    n_date_signal_runs_total = 0

    for signal_column in signal_columns:
        input_subdir, input_filename, _output_subdir, output_filename, output_csv_filename = _resolve_pair_meta_paths(
            settings,
            analysis_kind=analysis_kind,
        )
        resolved_input_filename = _resolve_signal_output_filename(input_filename, signal_column)
        session_rows = scan_analysis_paths(
            cfg,
            str(input_subdir).rstrip("/"),
            filename=_ensure_filename(resolved_input_filename, ".pkl"),
            dates=[str(value) for value in dates] if dates is not None else None,
            sessions=[str(value) for value in sessions] if sessions is not None else None,
        )
        date_to_rows: dict[str, list[dict]] = {}
        for row in session_rows:
            date_to_rows.setdefault(str(row["date"]), []).append(row)

        output_paths: list[str] = []
        csv_output_paths: list[str] = []
        group_summary_csv_output_paths: list[str] = []
        n_summary_rows_total = 0
        n_group_summary_rows_total = 0
        date_results: list[dict[str, object]] = []
        date_items = sorted(date_to_rows.items())
        run_date_pool = bool(
            settings.use_parallel
            and settings.parallelize_across_dates
            and len(date_items) > 1
        )
        date_pool_n_procs = 1

        if run_date_pool:
            date_pool_n_procs = get_n_processes(max_procs=settings.max_procs)
            worker_tasks = [
                (cfg, settings, analysis_kind, str(date), str(signal_column), date_rows)
                for date, date_rows in date_items
            ]
            with Pool(processes=date_pool_n_procs) as pool:
                iterator = pool.imap_unordered(
                    _build_and_save_date_pair_meta_result_worker,
                    worker_tasks,
                    chunksize=1,
                )
                if show_progress:
                    iterator = tqdm(
                        iterator,
                        total=len(worker_tasks),
                        desc=f"{analysis_kind} sig xcorr pairs {signal_column} ({date_pool_n_procs} workers)",
                        unit="date",
                    )
                for date_result in iterator:
                    date_results.append(dict(date_result))
        else:
            iterator = date_items
            if show_progress and date_items:
                iterator = tqdm(
                    iterator,
                    desc=f"{analysis_kind} sig xcorr pairs {signal_column}",
                    unit="date",
                )
            for date, date_rows in iterator:
                date_results.append(
                    _build_and_save_date_pair_meta_result(
                        cfg,
                        settings,
                        analysis_kind=analysis_kind,
                        date=str(date),
                        signal_column=str(signal_column),
                        rows=date_rows,
                    )
                )

        date_results = sorted(date_results, key=lambda row: str(row.get("date", "")))
        for date_result in date_results:
            output_paths.append(str(date_result.get("output_path")))
            csv_output_paths.append(str(date_result.get("csv_output_path")))
            group_summary_csv_output_paths.append(str(date_result.get("group_summary_csv_output_path")))
            n_summary_rows_total += int(date_result.get("n_summary_rows", 0))
            n_group_summary_rows_total += int(date_result.get("n_group_summary_rows", 0))
        n_date_signal_runs_total += int(len(date_results))

        resolved_output_filename = _resolve_signal_output_filename(output_filename, signal_column)
        resolved_output_csv_filename = _resolve_signal_output_filename(output_csv_filename, signal_column)
        signal_summaries[str(signal_column)] = {
            "analysis_kind": analysis_kind,
            "signal_input_column": str(signal_column),
            "signal_variant": _signal_output_label(signal_column),
            "input_filename": _ensure_filename(resolved_input_filename, ".pkl"),
            "output_filename": _ensure_filename(resolved_output_filename, ".pkl"),
            "output_csv_filename": _ensure_filename(resolved_output_csv_filename, ".csv"),
            "n_session_files_total": int(len(session_rows)),
            "n_dates_total": int(len(date_to_rows)),
            "n_dates_written": int(len(output_paths)),
            "n_summary_rows_total": int(n_summary_rows_total),
            "n_group_summary_rows_total": int(n_group_summary_rows_total),
            "parallelized_across_dates": bool(run_date_pool),
            "date_pool_n_procs": int(date_pool_n_procs),
            "output_paths": output_paths,
            "csv_output_paths": csv_output_paths,
            "group_summary_csv_output_paths": group_summary_csv_output_paths,
        }

    return {
        "analysis_kind": analysis_kind,
        "signal_input_columns": list(signal_columns),
        "signal_summaries": signal_summaries,
        "n_date_signal_runs_total": int(n_date_signal_runs_total),
    }


def run_within_region_fixation_neural_cross_correlation_pair_meta_analysis(
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict[str, object]:
    """Aggregate within-region fixation neural xcorr outputs into sig-pair summaries."""
    return _run_fixation_neural_cross_correlation_pair_meta_analysis(
        settings,
        analysis_kind=WITHIN_ANALYSIS_KIND,
        dates=dates,
        sessions=sessions,
        show_progress=show_progress,
    )


def run_cross_region_fixation_neural_cross_correlation_pair_meta_analysis(
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict[str, object]:
    """Aggregate cross-region fixation neural xcorr outputs into sig-pair summaries."""
    return _run_fixation_neural_cross_correlation_pair_meta_analysis(
        settings,
        analysis_kind=CROSS_ANALYSIS_KIND,
        dates=dates,
        sessions=sessions,
        show_progress=show_progress,
    )




def iter_fixation_neural_cross_correlation_pair_meta_analysis_output_paths(
    *,
    dataset_cfg_path: str,
    output_subdir: str,
    output_filename: str,
    signal_input_column: Optional[str] = None,
    date: Optional[str] = None,
) -> list[Path]:
    """List date-level sig-pair output files for one analysis kind."""
    cfg = load_config(dataset_cfg_path)
    resolved_filename = _ensure_filename(output_filename, ".pkl")
    if signal_input_column is not None:
        resolved_filename = _resolve_signal_output_filename(resolved_filename, signal_input_column)
    rows = scan_analysis_date_paths(
        cfg,
        str(output_subdir).rstrip("/"),
        filename=resolved_filename,
        dates=[str(date)] if date is not None else None,
    )
    return [Path(row["path"]) for row in rows]



def build_fixation_neural_cross_correlation_sig_xcorr_pair_group_summary_table(
    output_paths: Sequence[str | Path],
) -> pd.DataFrame:
    """Concatenate per-date group summary counts from saved sig-pair outputs."""
    frames: list[pd.DataFrame] = []
    condition_order: Sequence[str] = tuple(_PLOT_CONDITION_ORDER)

    for path in output_paths:
        obj = load_pickle_path(path)
        meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        if isinstance(meta.get("condition_order"), (list, tuple)) and meta.get("condition_order"):
            condition_order = tuple(str(value) for value in meta["condition_order"])

        group_summaries = obj.get("group_summaries") if isinstance(obj, dict) else None
        if not isinstance(group_summaries, pd.DataFrame):
            pair_summaries = obj.get("pair_summaries") if isinstance(obj, dict) else None
            if isinstance(pair_summaries, pd.DataFrame):
                group_summaries = _build_group_sig_xcorr_pair_summary_dataframe(
                    pair_summaries,
                    condition_order=condition_order,
                )
            else:
                group_summaries = pd.DataFrame()
        if group_summaries.empty:
            continue

        work = group_summaries.copy()
        if "date" not in work.columns and meta.get("date") is not None:
            work["date"] = str(meta.get("date"))
        if meta.get("signal_input_column") is not None:
            work["signal_input_column"] = str(meta.get("signal_input_column"))
        if meta.get("signal_variant") is not None:
            work["signal_variant"] = str(meta.get("signal_variant"))
        frames.append(work)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    out = _sort_group_summary_dataframe(out, condition_order=condition_order)
    ordered = [
        col
        for col in (
            "signal_variant",
            "signal_input_column",
            "date",
            "group_label",
            "region_1",
            "region_2",
            "n_total_pairs",
            *[_group_summary_count_column(condition) for condition in condition_order],
            "n_sig_any_condition_pairs",
        )
        if col in out.columns
    ]
    remaining = [col for col in out.columns if col not in ordered]
    return out.loc[:, ordered + remaining]



def print_fixation_neural_cross_correlation_pair_meta_analysis_example(
    path: str | Path,
) -> None:
    """Print a compact preview from one saved sig-pair output."""
    obj = load_pickle_path(path)
    pair_summaries = obj.get("pair_summaries") if isinstance(obj, dict) else None
    group_summaries = obj.get("group_summaries") if isinstance(obj, dict) else None
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    df = pair_summaries if isinstance(pair_summaries, pd.DataFrame) else pd.DataFrame()
    group_df = group_summaries if isinstance(group_summaries, pd.DataFrame) else pd.DataFrame()
    path = Path(path)

    if df.empty:
        print(f"[example] Sig-pair output exists but is empty: {path}")
        return

    row = df.iloc[0]
    print("\nExample fixation neural xcorr sig-pair output:")
    print(f"  file: {path}")
    print(f"  analysis_kind: {meta.get('analysis_kind')}")
    print(f"  date: {meta.get('date')}")
    print(f"  signal_input_column: {meta.get('signal_input_column')}")
    print(f"  n_pair_summaries: {len(df)}")
    print(
        "  first_row: "
        f"group_label={row.get('group_label')}, "
        f"condition={row.get('condition')}, "
        f"n_fixations={row.get('n_fixations')}, "
        f"mean_xcorr_across_lags={row.get('mean_cross_correlation_across_lags')}, "
        f"p_value={row.get('p_value')}, "
        f"significant_above_zero={row.get('significant_above_zero')}"
    )
    if not group_df.empty:
        print("  group_summary_table:")
        print(group_df.to_string(index=False))


print_fixation_neural_cross_correlation_sig_xcorr_pairs_example = (
    print_fixation_neural_cross_correlation_pair_meta_analysis_example
)


# Backward-compatible aliases with clearer sig-pair naming for scripts.
FixationNeuralCrossCorrelationSigXcorrPairsSettings = FixationNeuralCrossCorrelationPairMetaAnalysisSettings


def resolve_fixation_neural_cross_correlation_sig_xcorr_pairs_signal_columns(
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
) -> tuple[str, ...]:
    return resolve_fixation_neural_cross_correlation_pair_meta_analysis_signal_columns(settings)



def build_fixation_neural_cross_correlation_sig_xcorr_pairs_settings_from_config(
    *,
    dataset_cfg_path: str,
    ephys_fixation_neural_cross_correlation_cfg_path: str | None = None,
    ephys_fixation_neural_crosscorr_cfg_path: str | None = None,
) -> FixationNeuralCrossCorrelationPairMetaAnalysisSettings:
    return build_fixation_neural_cross_correlation_pair_meta_analysis_settings_from_config(
        dataset_cfg_path=dataset_cfg_path,
        ephys_fixation_neural_cross_correlation_cfg_path=ephys_fixation_neural_cross_correlation_cfg_path,
        ephys_fixation_neural_crosscorr_cfg_path=ephys_fixation_neural_crosscorr_cfg_path,
    )



def run_within_region_fixation_neural_cross_correlation_sig_xcorr_pairs(
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict[str, object]:
    return run_within_region_fixation_neural_cross_correlation_pair_meta_analysis(
        settings,
        dates=dates,
        sessions=sessions,
        show_progress=show_progress,
    )



def run_cross_region_fixation_neural_cross_correlation_sig_xcorr_pairs(
    settings: FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict[str, object]:
    return run_cross_region_fixation_neural_cross_correlation_pair_meta_analysis(
        settings,
        dates=dates,
        sessions=sessions,
        show_progress=show_progress,
    )



def iter_fixation_neural_cross_correlation_sig_xcorr_pairs_output_paths(
    *,
    dataset_cfg_path: str,
    output_subdir: str,
    output_filename: str,
    signal_input_column: Optional[str] = None,
    date: Optional[str] = None,
) -> list[Path]:
    return iter_fixation_neural_cross_correlation_pair_meta_analysis_output_paths(
        dataset_cfg_path=dataset_cfg_path,
        output_subdir=output_subdir,
        output_filename=output_filename,
        signal_input_column=signal_input_column,
        date=date,
    )
