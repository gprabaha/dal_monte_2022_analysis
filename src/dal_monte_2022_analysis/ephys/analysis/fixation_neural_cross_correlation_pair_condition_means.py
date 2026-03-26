"""Aggregate session-level neural xcorr pair averages into date-level pair-condition means."""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_optional_str as _as_optional_str,
    ensure_filename as _ensure_filename,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_helpers import (
    CROSS_ANALYSIS_KIND,
    WITHIN_ANALYSIS_KIND,
    _PLOT_ALLOWED_ANALYSIS_KINDS,
    _PLOT_CONDITION_ORDER,
    _append_trace_sum,
    _assert_lag_axis_match,
    _extract_xcorr_dataframes_and_meta,
    _normalize_region_pair_label,
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
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


@dataclass
class FixationNeuralCrossCorrelationPairConditionMeanSettings:
    """Configuration for date-level pair-condition mean xcorr aggregation."""

    cfg_path: str
    signal_input_column: str = "spike_train_counts"
    signal_input_columns: Optional[Sequence[str]] = None
    within_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/within_region"
    cross_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/cross_region"
    within_input_filename: str = "pair_averages.pkl"
    cross_input_filename: str = "pair_averages.pkl"
    within_output_subdir: str = (
        "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/within_region"
    )
    cross_output_subdir: str = (
        "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/cross_region"
    )
    within_output_filename: str = "pair_condition_means.pkl"
    cross_output_filename: str = "pair_condition_means.pkl"
    within_output_csv_filename: str = "pair_condition_means.csv"
    cross_output_csv_filename: str = "pair_condition_means.csv"
    condition_order: Sequence[str] = field(default_factory=lambda: tuple(_PLOT_CONDITION_ORDER))
    use_parallel: bool = True
    parallelize_across_dates: bool = True


def resolve_fixation_neural_cross_correlation_pair_condition_mean_signal_columns(
    settings: FixationNeuralCrossCorrelationPairConditionMeanSettings,
) -> tuple[str, ...]:
    """Resolve the configured signal columns in run order."""
    return _resolve_signal_input_columns(settings)


def build_fixation_neural_cross_correlation_pair_condition_mean_settings_from_config(
    *,
    dataset_cfg_path: str,
    ephys_fixation_neural_cross_correlation_cfg_path: str | None = None,
    ephys_fixation_neural_crosscorr_cfg_path: str | None = None,
) -> FixationNeuralCrossCorrelationPairConditionMeanSettings:
    """Build pair-condition mean settings from dataset + task config paths."""
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
        "pair_condition_mean_condition_order",
        cfg.get("plot_condition_order", tuple(_PLOT_CONDITION_ORDER)),
    )
    return FixationNeuralCrossCorrelationPairConditionMeanSettings(
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
        within_input_filename=cfg.get("within_pair_average_output_filename", "pair_averages.pkl"),
        cross_input_filename=cfg.get("cross_pair_average_output_filename", "pair_averages.pkl"),
        within_output_subdir=cfg.get(
            "pair_condition_mean_within_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/within_region",
        ),
        cross_output_subdir=cfg.get(
            "pair_condition_mean_cross_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/cross_region",
        ),
        within_output_filename=cfg.get(
            "pair_condition_mean_within_output_filename",
            "pair_condition_means.pkl",
        ),
        cross_output_filename=cfg.get(
            "pair_condition_mean_cross_output_filename",
            "pair_condition_means.pkl",
        ),
        within_output_csv_filename=cfg.get(
            "pair_condition_mean_within_output_csv_filename",
            "pair_condition_means.csv",
        ),
        cross_output_csv_filename=cfg.get(
            "pair_condition_mean_cross_output_csv_filename",
            "pair_condition_means.csv",
        ),
        condition_order=tuple(str(value) for value in condition_order),
        use_parallel=cfg.get("pair_condition_mean_use_parallel", cfg.get("use_parallel", True)),
        parallelize_across_dates=cfg.get("pair_condition_mean_parallelize_across_dates", True),
    )


def _resolve_pair_condition_mean_paths(
    settings: FixationNeuralCrossCorrelationPairConditionMeanSettings,
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


def _build_pair_condition_mean_key(
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


def _sort_pair_condition_mean_dataframe(
    df: pd.DataFrame,
    *,
    condition_order: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    order_map = {str(name): idx for idx, name in enumerate(condition_order)}
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


def _build_date_pair_condition_means_result(
    settings: FixationNeuralCrossCorrelationPairConditionMeanSettings,
    *,
    analysis_kind: str,
    date: str,
    signal_column: str,
    rows: Sequence[dict],
) -> dict[str, object]:
    trace_accum: dict[tuple[str, str, str, str, str, str], list[object]] = {}
    lag_axis: Optional[np.ndarray] = None
    signal_bin_size_ms: Optional[float] = None
    source_sessions: set[str] = set()
    input_paths: list[str] = []
    source_row_count = 0
    used_row_count = 0
    skipped_row_count = 0
    valid_conditions = {str(name) for name in settings.condition_order}

    for session_row in rows:
        input_paths.append(str(session_row["path"]))
        source_sessions.add(str(session_row["session"]))
        obj = load_pickle_path(session_row["path"])
        _xcorr_df, pair_avg_df, meta = _extract_xcorr_dataframes_and_meta(obj)
        if pair_avg_df.empty or "cross_correlation" not in pair_avg_df.columns:
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

        for pair_row in pair_avg_df.itertuples(index=False):
            source_row_count += 1
            condition = _as_optional_str(getattr(pair_row, "condition", None))
            if condition is None or str(condition) not in valid_conditions:
                skipped_row_count += 1
                continue

            trace = np.asarray(getattr(pair_row, "cross_correlation", []), dtype=np.float64).reshape(-1)
            if trace.size <= 0:
                skipped_row_count += 1
                continue
            if lag_axis is not None and lag_axis.size != trace.size:
                raise ValueError(
                    "Encountered pair-condition mean trace length mismatch while aggregating "
                    f"date={date} for {signal_column}."
                )
            trace = np.where(np.isfinite(trace), trace, np.nan)
            if not np.isfinite(trace).any():
                skipped_row_count += 1
                continue

            n_fixations = getattr(pair_row, "n_fixations", None)
            try:
                weight = float(n_fixations)
            except Exception:
                weight = np.nan
            if not np.isfinite(weight) or weight <= 0.0:
                skipped_row_count += 1
                continue

            key = _build_pair_condition_mean_key(
                pair_row,
                analysis_kind=analysis_kind,
                condition=str(condition),
            )
            _append_trace_sum(trace_accum, key, trace, weight=weight)
            used_row_count += 1

    rows_out: list[dict[str, object]] = []
    for key, (trace_sum, weight_sum) in trace_accum.items():
        total_fixations = float(weight_sum)
        if total_fixations <= 0.0:
            continue
        mean_trace = np.asarray(trace_sum, dtype=np.float64) / total_fixations
        if not np.isfinite(mean_trace).any():
            continue

        if lag_axis is not None and lag_axis.size == mean_trace.size:
            trace_summary = _summarize_cross_correlation(lag_axis, mean_trace)
        else:
            trace_summary = {
                "n_lags": int(mean_trace.size),
                "zero_lag_correlation": None,
                "peak_lag": None,
                "peak_correlation": float(np.nanmax(mean_trace)) if mean_trace.size else None,
            }

        group_label, region_1, region_2, unit_uuid_1, unit_uuid_2, condition = key
        rows_out.append(
            {
                "date": str(date),
                "group_label": group_label,
                "condition": condition,
                "region_1": region_1,
                "region_2": region_2,
                "unit_uuid_1": unit_uuid_1,
                "unit_uuid_2": unit_uuid_2,
                "n_fixations": int(round(total_fixations)),
                "mean_cross_correlation_across_lags": float(np.nanmean(mean_trace)),
                "cross_correlation": mean_trace.astype(np.float32),
                **trace_summary,
            }
        )

    pair_condition_means = _sort_pair_condition_mean_dataframe(
        pd.DataFrame(rows_out),
        condition_order=settings.condition_order,
    )
    return {
        "meta": {
            "analysis_kind": analysis_kind,
            "date": str(date),
            "signal_input_column": str(signal_column),
            "signal_variant": _signal_output_label(signal_column),
            "condition_order": [str(name) for name in settings.condition_order],
            "n_input_session_files": int(len(rows)),
            "n_input_sessions": int(len(source_sessions)),
            "source_sessions": sorted(source_sessions),
            "source_paths": input_paths,
            "source_row_count": int(source_row_count),
            "used_row_count": int(used_row_count),
            "skipped_row_count": int(skipped_row_count),
            "n_pair_condition_mean_rows": int(len(pair_condition_means)),
            "lags": lag_axis,
            "bin_size_ms": signal_bin_size_ms,
        },
        "pair_condition_means": pair_condition_means,
    }


def _build_pair_condition_mean_output_paths(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationPairConditionMeanSettings,
    *,
    analysis_kind: str,
    date: str,
    signal_column: str,
) -> tuple[Path, Path]:
    _input_subdir, _input_filename, output_subdir, output_filename, output_csv_filename = (
        _resolve_pair_condition_mean_paths(
            settings,
            analysis_kind=analysis_kind,
        )
    )
    out_root = build_analysis_output_dir(cfg, str(output_subdir).rstrip("/")) / f"date={date}"
    resolved_pkl = _resolve_signal_output_filename(output_filename, signal_column)
    resolved_csv = _resolve_signal_output_filename(output_csv_filename, signal_column)
    return (
        out_root / _ensure_filename(resolved_pkl, ".pkl"),
        out_root / _ensure_filename(resolved_csv, ".csv"),
    )


def _build_and_save_date_pair_condition_means_result(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationPairConditionMeanSettings,
    *,
    analysis_kind: str,
    date: str,
    signal_column: str,
    rows: Sequence[dict],
) -> dict[str, object]:
    result = _build_date_pair_condition_means_result(
        settings,
        analysis_kind=analysis_kind,
        date=str(date),
        signal_column=str(signal_column),
        rows=rows,
    )
    out_pkl, out_csv = _build_pair_condition_mean_output_paths(
        cfg,
        settings,
        analysis_kind=analysis_kind,
        date=str(date),
        signal_column=str(signal_column),
    )
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    save_pickle_path(result, out_pkl)
    pair_condition_means = result.get("pair_condition_means")
    csv_df = pair_condition_means if isinstance(pair_condition_means, pd.DataFrame) else pd.DataFrame()
    csv_df.drop(columns=["cross_correlation"], errors="ignore").to_csv(out_csv, index=False)
    return {
        "date": str(date),
        "output_path": str(out_pkl),
        "csv_output_path": str(out_csv),
        "n_pair_condition_mean_rows": int(len(csv_df)),
    }


def _build_and_save_date_pair_condition_means_result_worker(
    task: tuple[
        dict,
        FixationNeuralCrossCorrelationPairConditionMeanSettings,
        str,
        str,
        str,
        Sequence[dict],
    ],
) -> dict[str, object]:
    cfg, settings, analysis_kind, date, signal_column, rows = task
    return _build_and_save_date_pair_condition_means_result(
        cfg,
        settings,
        analysis_kind=analysis_kind,
        date=str(date),
        signal_column=str(signal_column),
        rows=rows,
    )


def _run_fixation_neural_cross_correlation_pair_condition_means(
    settings: FixationNeuralCrossCorrelationPairConditionMeanSettings,
    *,
    analysis_kind: str,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict[str, object]:
    if analysis_kind not in _PLOT_ALLOWED_ANALYSIS_KINDS:
        raise ValueError(
            f"Unsupported analysis_kind={analysis_kind!r}. "
            f"Expected one of: {', '.join(_PLOT_ALLOWED_ANALYSIS_KINDS)}.",
        )

    cfg = load_config(settings.cfg_path)
    signal_columns = resolve_fixation_neural_cross_correlation_pair_condition_mean_signal_columns(settings)
    signal_summaries: dict[str, dict[str, object]] = {}
    n_date_signal_runs_total = 0

    for signal_column in signal_columns:
        input_subdir, input_filename, _output_subdir, output_filename, output_csv_filename = (
            _resolve_pair_condition_mean_paths(
                settings,
                analysis_kind=analysis_kind,
            )
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
        n_pair_condition_mean_rows_total = 0
        date_results: list[dict[str, object]] = []
        date_items = sorted(date_to_rows.items())
        run_date_pool = bool(
            settings.use_parallel
            and settings.parallelize_across_dates
            and len(date_items) > 1
        )
        date_pool_n_procs = 1

        if run_date_pool:
            date_pool_n_procs = get_n_processes(max_procs=None)
            worker_tasks = [
                (cfg, settings, analysis_kind, str(date), str(signal_column), date_rows)
                for date, date_rows in date_items
            ]
            with Pool(processes=date_pool_n_procs) as pool:
                iterator = pool.imap_unordered(
                    _build_and_save_date_pair_condition_means_result_worker,
                    worker_tasks,
                    chunksize=1,
                )
                if show_progress:
                    iterator = tqdm(
                        iterator,
                        total=len(worker_tasks),
                        desc=(
                            f"{analysis_kind} pair-condition means {signal_column} "
                            f"({date_pool_n_procs} workers)"
                        ),
                        unit="date",
                    )
                for date_result in iterator:
                    date_results.append(dict(date_result))
        else:
            iterator = date_items
            if show_progress and date_items:
                iterator = tqdm(
                    iterator,
                    desc=f"{analysis_kind} pair-condition means {signal_column}",
                    unit="date",
                )
            for date, date_rows in iterator:
                date_results.append(
                    _build_and_save_date_pair_condition_means_result(
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
            n_pair_condition_mean_rows_total += int(date_result.get("n_pair_condition_mean_rows", 0))
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
            "n_pair_condition_mean_rows_total": int(n_pair_condition_mean_rows_total),
            "parallelized_across_dates": bool(run_date_pool),
            "date_pool_n_procs": int(date_pool_n_procs),
            "output_paths": output_paths,
            "csv_output_paths": csv_output_paths,
        }

    return {
        "analysis_kind": analysis_kind,
        "signal_input_columns": list(signal_columns),
        "signal_summaries": signal_summaries,
        "n_date_signal_runs_total": int(n_date_signal_runs_total),
    }


def run_within_region_fixation_neural_cross_correlation_pair_condition_means(
    settings: FixationNeuralCrossCorrelationPairConditionMeanSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict[str, object]:
    """Aggregate within-region session pair averages into date-level pair-condition means."""
    return _run_fixation_neural_cross_correlation_pair_condition_means(
        settings,
        analysis_kind=WITHIN_ANALYSIS_KIND,
        dates=dates,
        sessions=sessions,
        show_progress=show_progress,
    )


def run_cross_region_fixation_neural_cross_correlation_pair_condition_means(
    settings: FixationNeuralCrossCorrelationPairConditionMeanSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict[str, object]:
    """Aggregate cross-region session pair averages into date-level pair-condition means."""
    return _run_fixation_neural_cross_correlation_pair_condition_means(
        settings,
        analysis_kind=CROSS_ANALYSIS_KIND,
        dates=dates,
        sessions=sessions,
        show_progress=show_progress,
    )


def iter_fixation_neural_cross_correlation_pair_condition_mean_output_paths(
    *,
    dataset_cfg_path: str,
    output_subdir: str,
    output_filename: str,
    signal_input_column: Optional[str] = None,
    date: Optional[str] = None,
) -> list[Path]:
    """List date-level pair-condition mean output files for one analysis kind."""
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


def print_fixation_neural_cross_correlation_pair_condition_mean_example(path: str | Path) -> None:
    """Print a compact preview from one saved pair-condition mean output."""
    obj = load_pickle_path(path)
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    df = obj.get("pair_condition_means") if isinstance(obj, dict) else None
    df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    path = Path(path)

    if df.empty:
        print(f"[example] Pair-condition mean output exists but is empty: {path}")
        return

    row = df.iloc[0]
    print("\nExample fixation neural xcorr pair-condition mean output:")
    print(f"  file: {path}")
    print(f"  analysis_kind: {meta.get('analysis_kind')}")
    print(f"  date: {meta.get('date')}")
    print(f"  signal_input_column: {meta.get('signal_input_column')}")
    print(f"  n_pair_condition_mean_rows: {len(df)}")
    print(
        "  first_row: "
        f"group_label={row.get('group_label')}, "
        f"condition={row.get('condition')}, "
        f"n_fixations={row.get('n_fixations')}, "
        f"mean_xcorr_across_lags={row.get('mean_cross_correlation_across_lags')}, "
        f"peak_lag={row.get('peak_lag')}, "
        f"peak_corr={row.get('peak_correlation')}"
    )
