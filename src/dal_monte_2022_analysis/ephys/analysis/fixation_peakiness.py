"""Score unit peakiness from average fixation PSTHs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
    resolve_bin_centers_from_meta,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import (
    build_analysis_output_dir,
    scan_analysis_date_paths,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
)
from dal_monte_2022_analysis.utils.filenames import ensure_filename


PEAKINESS_CONDITIONS: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)


@dataclass
class FixationPeakinessSettings:
    """Configuration for peakiness scoring from average fixation PSTHs."""

    cfg_path: str
    average_input_subdir: str = "ephys/psth/fixation_psth_averages"
    average_input_filename: str = "fixations_psth_10ms.pkl"
    output_subdir: str = "ephys/psth/fixation_peakiness"
    unit_output_filename: str = "unit_peakiness.csv"
    condition_output_filename: str = "unit_condition_peakiness.csv"
    region_summary_filename: str = "region_peakiness_summary.csv"
    output_pickle_filename: str = "results.pkl"
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    condition_order: tuple[str, ...] = field(default_factory=lambda: PEAKINESS_CONDITIONS)
    min_trials_per_condition: int = 1
    mean_rate_floor_hz: float = 0.5
    peak_distance_ms: float = 30.0
    peak_prominence_floor: float = 0.0
    competition_penalty_lambda: float = 0.5
    prominence_epsilon: float = 1.0e-12
    bin_size_ms_fallback: float = 10.0
    region_order: Optional[Sequence[str]] = None


def _norm_token(value: object) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def _date_token(value: object) -> str:
    token = _as_optional_str(value)
    if token is None:
        return ""
    if len(token) == 7 and token.isdigit():
        return token.zfill(8)
    return token


def _extract_average_partitions(obj) -> list[tuple[str, pd.DataFrame, dict]]:
    if isinstance(obj, dict):
        meta = obj.get("meta", {}) or {}
        meta_dict = meta if isinstance(meta, dict) else {}
        out: list[tuple[str, pd.DataFrame, dict]] = []
        for partition, df_key, meta_key in (
            ("split", "averages_split_by_interactive_state", "split_meta"),
            ("unsplit", "averages_unsplit_by_interactive_state", "unsplit_meta"),
        ):
            df = obj.get(df_key)
            if not isinstance(df, pd.DataFrame):
                continue
            merged_meta = dict(meta_dict)
            partition_meta = meta_dict.get(meta_key, {})
            if isinstance(partition_meta, dict):
                merged_meta.update(partition_meta)
            merged_meta["selected_partition"] = partition
            out.append((partition, df, merged_meta))
        if out:
            return out

        df = obj.get("averages")
        if isinstance(df, pd.DataFrame):
            partition = "split" if bool(meta_dict.get("split_by_interactive_state")) else "unsplit"
            merged_meta = dict(meta_dict)
            merged_meta["selected_partition"] = partition
            return [(partition, df, merged_meta)]

    if isinstance(obj, pd.DataFrame):
        return [("unsplit", obj, {})]
    return []


def _average_row_condition(
    row: pd.Series,
    *,
    partition: str,
    settings: FixationPeakinessSettings,
) -> Optional[str]:
    category = _norm_token(row.get("fixation_category"))
    face = _norm_token(settings.face_label)
    obj = _norm_token(settings.object_label)

    if partition == "unsplit":
        return "object" if category == obj else None
    if partition != "split" or category != face:
        return None

    interactive_state = row.get("interactive_state")
    is_interactive = row.get("is_interactive")
    if is_interactive is not None and not pd.isna(is_interactive):
        interactive = bool(_as_bool(is_interactive, settings.interactive_label))
    else:
        interactive = _norm_token(interactive_state) == _norm_token(settings.interactive_label)
    return "face_interactive" if interactive else "face_non_interactive"


def _resolve_bin_duration_s(
    meta: dict,
    centers_s: np.ndarray,
    settings: FixationPeakinessSettings,
) -> float:
    for key in ("target_bin_size_s", "output_bin_size_s", "bin_size_s"):
        value = meta.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except Exception:
            continue
        if np.isfinite(out) and out > 0.0:
            return out
    if centers_s.size > 1:
        out = float(np.median(np.diff(centers_s)))
        if np.isfinite(out) and out > 0.0:
            return out
    return float(settings.bin_size_ms_fallback) / 1000.0


def _average_values_are_rate(meta: dict) -> bool:
    value_kind = str(meta.get("psth_value_kind", "")).strip().lower()
    return bool(
        value_kind == "firing_rate_hz"
        or (
            meta.get("convert_to_firing_rate_before_average") is True
            and value_kind != "counts"
        )
    )


def _trace_to_rate_hz(
    trace: object,
    *,
    centers_s: np.ndarray,
    meta: dict,
    settings: FixationPeakinessSettings,
    source_path: Path,
) -> np.ndarray:
    values = np.asarray(trace, dtype=float).reshape(-1)
    if values.size != centers_s.size:
        raise ValueError(
            "Average PSTH row length does not match bin centers: "
            f"path={source_path}, n_values={values.size}, n_centers={centers_s.size}"
        )
    if _average_values_are_rate(meta):
        return values
    return values / _resolve_bin_duration_s(meta, centers_s, settings)


def _coerce_n_trials(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(out):
        return np.nan
    return out


def _load_condition_trace_rows(
    settings: FixationPeakinessSettings,
    *,
    dates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    rows = scan_analysis_date_paths(
        cfg,
        settings.average_input_subdir,
        filename=ensure_filename(settings.average_input_filename, ".pkl"),
        dates=dates,
    )
    records: list[dict] = []
    for row in rows:
        path = Path(row["path"])
        obj = load_pickle_path(path)
        for partition, avg_df, meta in _extract_average_partitions(obj):
            if avg_df.empty or "psth_mean" not in avg_df.columns:
                continue
            centers = resolve_bin_centers_from_meta(meta)
            if centers is None:
                raise ValueError(f"Unable to resolve average PSTH bin centers: {path}")
            centers_s = np.asarray(centers, dtype=float).reshape(-1)
            for _, avg_row in avg_df.iterrows():
                condition = _average_row_condition(avg_row, partition=partition, settings=settings)
                if condition is None:
                    continue
                n_trials = _coerce_n_trials(avg_row.get("n_trials", np.nan))
                if np.isfinite(n_trials) and n_trials < float(settings.min_trials_per_condition):
                    continue
                date = _date_token(avg_row.get("date")) or _date_token(row.get("date"))
                unit_uuid = _as_optional_str(avg_row.get("unit_uuid"))
                if not date or unit_uuid is None:
                    continue
                records.append(
                    {
                        "unit_key": f"{date}|{unit_uuid}",
                        "date": date,
                        "unit_uuid": unit_uuid,
                        "region": _as_optional_str(avg_row.get("region")) or "unknown",
                        "spike_channel": _as_optional_str(avg_row.get("spike_channel")),
                        "recorded_agent": _as_optional_str(avg_row.get("recorded_agent")),
                        "recorded_monkey": _as_optional_str(avg_row.get("recorded_monkey")),
                        "area": _as_optional_str(avg_row.get("area")),
                        "condition": condition,
                        "n_trials": n_trials,
                        "bin_centers_s_rel": centers_s,
                        "trace_hz": _trace_to_rate_hz(
                            avg_row.get("psth_mean"),
                            centers_s=centers_s,
                            meta=meta,
                            settings=settings,
                            source_path=path,
                        ),
                        "source_path": str(path),
                    }
                )
    return pd.DataFrame(records)


def _first_non_null(series: pd.Series) -> Optional[str]:
    for value in series:
        token = _as_optional_str(value)
        if token is not None:
            return token
    return None


def _aggregate_condition_trace_rows(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame(
            columns=[
                "unit_key",
                "date",
                "unit_uuid",
                "region",
                "spike_channel",
                "recorded_agent",
                "recorded_monkey",
                "area",
                "condition",
                "n_trials",
                "n_source_rows",
                "bin_centers_s_rel",
                "trace_hz",
            ]
        )

    rows: list[dict] = []
    group_cols = ["unit_key", "condition"]
    for (_, _), group in trace_df.groupby(group_cols, dropna=False, sort=False):
        first = group.iloc[0]
        centers_ref = np.asarray(first["bin_centers_s_rel"], dtype=float).reshape(-1)
        traces: list[np.ndarray] = []
        weights: list[float] = []
        for _, row in group.iterrows():
            centers = np.asarray(row["bin_centers_s_rel"], dtype=float).reshape(-1)
            if centers.shape != centers_ref.shape or not np.allclose(centers, centers_ref):
                raise ValueError(
                    "Mismatched bin centers within the same unit-condition average traces: "
                    f"unit_key={first['unit_key']}, condition={first['condition']}"
                )
            trace = np.asarray(row["trace_hz"], dtype=float).reshape(-1)
            if trace.shape != centers_ref.shape:
                raise ValueError(
                    "Mismatched trace length within the same unit-condition average traces: "
                    f"unit_key={first['unit_key']}, condition={first['condition']}"
                )
            traces.append(trace)
            n_trials = _coerce_n_trials(row.get("n_trials", np.nan))
            weights.append(n_trials if np.isfinite(n_trials) and n_trials > 0.0 else 1.0)
        stacked = np.vstack(traces)
        weight_arr = np.asarray(weights, dtype=float)
        trace_hz = np.average(stacked, axis=0, weights=weight_arr)

        n_trials_values = pd.to_numeric(group["n_trials"], errors="coerce")
        n_trials_total = float(n_trials_values.dropna().sum()) if n_trials_values.notna().any() else np.nan
        rows.append(
            {
                "unit_key": str(first["unit_key"]),
                "date": str(first["date"]),
                "unit_uuid": str(first["unit_uuid"]),
                "region": _first_non_null(group["region"]) or "unknown",
                "spike_channel": _first_non_null(group["spike_channel"]),
                "recorded_agent": _first_non_null(group["recorded_agent"]),
                "recorded_monkey": _first_non_null(group["recorded_monkey"]),
                "area": _first_non_null(group["area"]),
                "condition": str(first["condition"]),
                "n_trials": n_trials_total,
                "n_source_rows": int(len(group)),
                "bin_centers_s_rel": centers_ref,
                "trace_hz": trace_hz,
            }
        )
    return pd.DataFrame(rows)


def _resolve_bin_step_ms(centers_s: np.ndarray, settings: FixationPeakinessSettings) -> float:
    centers = np.asarray(centers_s, dtype=float).reshape(-1)
    if centers.size > 1:
        diffs = np.diff(centers)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size > 0:
            out = float(np.median(diffs) * 1000.0)
            if np.isfinite(out) and out > 0.0:
                return out
    return float(settings.bin_size_ms_fallback)


def _score_trace(
    trace_hz: np.ndarray,
    centers_s: np.ndarray,
    settings: FixationPeakinessSettings,
) -> dict[str, float | int]:
    values_hz = np.asarray(trace_hz, dtype=float).reshape(-1)
    centers = np.asarray(centers_s, dtype=float).reshape(-1)
    if values_hz.size != centers.size:
        raise ValueError(
            "Trace length must match bin centers when scoring peakiness. "
            f"n_values={values_hz.size}, n_centers={centers.size}"
        )
    if values_hz.size == 0:
        return {
            "mean_fr_hz": np.nan,
            "normalization_denom_hz": np.nan,
            "bin_step_ms": float(settings.bin_size_ms_fallback),
            "peak_distance_bins": 1,
            "n_detected_peaks": 0,
            "best_peak_latency_ms": np.nan,
            "best_peak_value_hz": np.nan,
            "best_peak_value_norm": np.nan,
            "best_peak_prominence": 0.0,
            "second_peak_prominence": 0.0,
            "competition_ratio": 0.0,
            "dominance": 0.0,
            "peakiness_score": 0.0,
        }

    finite_mask = np.isfinite(values_hz)
    mean_fr_hz = float(np.mean(values_hz[finite_mask])) if np.any(finite_mask) else np.nan
    floor_hz = float(settings.mean_rate_floor_hz)
    denom_hz = max(mean_fr_hz, floor_hz) if np.isfinite(mean_fr_hz) else floor_hz

    if np.any(finite_mask):
        fill_value = float(np.min(values_hz[finite_mask]))
    else:
        fill_value = 0.0
    safe_values_hz = np.where(finite_mask, values_hz, fill_value)
    norm_values = safe_values_hz / max(denom_hz, float(settings.prominence_epsilon))

    bin_step_ms = _resolve_bin_step_ms(centers, settings)
    distance_bins = max(1, int(round(float(settings.peak_distance_ms) / max(bin_step_ms, 1.0e-12))))

    peaks, props = find_peaks(
        norm_values,
        distance=distance_bins,
        prominence=max(float(settings.peak_prominence_floor), 0.0),
    )
    prominences = np.asarray(props.get("prominences", []), dtype=float).reshape(-1)
    if peaks.size == 0 or prominences.size == 0:
        return {
            "mean_fr_hz": mean_fr_hz,
            "normalization_denom_hz": denom_hz,
            "bin_step_ms": bin_step_ms,
            "peak_distance_bins": distance_bins,
            "n_detected_peaks": 0,
            "best_peak_latency_ms": np.nan,
            "best_peak_value_hz": np.nan,
            "best_peak_value_norm": np.nan,
            "best_peak_prominence": 0.0,
            "second_peak_prominence": 0.0,
            "competition_ratio": 0.0,
            "dominance": 0.0,
            "peakiness_score": 0.0,
        }

    order = np.argsort(-prominences)
    best_rank = int(order[0])
    best_peak_idx = int(peaks[best_rank])
    p1 = float(prominences[best_rank])
    p2 = float(prominences[order[1]]) if order.size > 1 else 0.0
    eps = float(settings.prominence_epsilon)
    competition_ratio = float(p2 / (p1 + eps)) if p1 > 0.0 else 0.0
    dominance = float(p1 / (p1 + p2 + eps)) if p1 > 0.0 else 0.0
    peakiness_score = float(
        p1 / (1.0 + float(settings.competition_penalty_lambda) * competition_ratio)
    ) if p1 > 0.0 else 0.0

    return {
        "mean_fr_hz": mean_fr_hz,
        "normalization_denom_hz": denom_hz,
        "bin_step_ms": bin_step_ms,
        "peak_distance_bins": distance_bins,
        "n_detected_peaks": int(peaks.size),
        "best_peak_latency_ms": float(centers[best_peak_idx] * 1000.0),
        "best_peak_value_hz": float(values_hz[best_peak_idx]) if np.isfinite(values_hz[best_peak_idx]) else np.nan,
        "best_peak_value_norm": float(norm_values[best_peak_idx]) if np.isfinite(norm_values[best_peak_idx]) else np.nan,
        "best_peak_prominence": p1,
        "second_peak_prominence": p2,
        "competition_ratio": competition_ratio,
        "dominance": dominance,
        "peakiness_score": peakiness_score,
    }


def _build_condition_peakiness_table(
    trace_df: pd.DataFrame,
    settings: FixationPeakinessSettings,
) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame(
            columns=[
                "unit_key",
                "date",
                "unit_uuid",
                "region",
                "spike_channel",
                "recorded_agent",
                "recorded_monkey",
                "area",
                "condition",
                "n_trials",
                "n_source_rows",
                "mean_fr_hz",
                "normalization_denom_hz",
                "bin_step_ms",
                "peak_distance_bins",
                "n_detected_peaks",
                "best_peak_latency_ms",
                "best_peak_value_hz",
                "best_peak_value_norm",
                "best_peak_prominence",
                "second_peak_prominence",
                "competition_ratio",
                "dominance",
                "peakiness_score",
            ]
        )

    rows: list[dict] = []
    for _, row in trace_df.iterrows():
        score = _score_trace(
            np.asarray(row["trace_hz"], dtype=float),
            np.asarray(row["bin_centers_s_rel"], dtype=float),
            settings,
        )
        out_row = {
            "unit_key": str(row["unit_key"]),
            "date": str(row["date"]),
            "unit_uuid": str(row["unit_uuid"]),
            "region": _as_optional_str(row["region"]) or "unknown",
            "spike_channel": _as_optional_str(row["spike_channel"]),
            "recorded_agent": _as_optional_str(row["recorded_agent"]),
            "recorded_monkey": _as_optional_str(row["recorded_monkey"]),
            "area": _as_optional_str(row["area"]),
            "condition": str(row["condition"]),
            "n_trials": _coerce_n_trials(row["n_trials"]),
            "n_source_rows": int(row["n_source_rows"]),
        }
        out_row.update(score)
        rows.append(out_row)
    return pd.DataFrame(rows)


def _condition_column(condition: str, suffix: str) -> str:
    return f"{condition}_{suffix}"


def _build_unit_peakiness_table(
    condition_df: pd.DataFrame,
    settings: FixationPeakinessSettings,
) -> pd.DataFrame:
    if condition_df.empty:
        columns = [
            "unit_key",
            "date",
            "unit_uuid",
            "region",
            "spike_channel",
            "recorded_agent",
            "recorded_monkey",
            "area",
            "n_conditions_observed",
            "all_target_conditions_present",
            "peakiness_score",
            "best_condition",
            "best_peak_latency_ms",
            "best_peak_prominence",
            "best_peak_dominance",
            "best_peak_competition_ratio",
        ]
        for condition in settings.condition_order:
            for suffix in (
                "peakiness_score",
                "best_peak_prominence",
                "second_peak_prominence",
                "dominance",
                "competition_ratio",
                "best_peak_latency_ms",
                "n_detected_peaks",
                "mean_fr_hz",
                "n_trials",
            ):
                columns.append(_condition_column(condition, suffix))
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for _, group in condition_df.groupby("unit_key", dropna=False, sort=False):
        group = group.copy()
        score_values = pd.to_numeric(group["peakiness_score"], errors="coerce")
        max_score = float(score_values.max()) if score_values.notna().any() else np.nan
        observed = [str(value) for value in group["condition"].astype(str).tolist()]
        row = {
            "unit_key": str(group.iloc[0]["unit_key"]),
            "date": str(group.iloc[0]["date"]),
            "unit_uuid": str(group.iloc[0]["unit_uuid"]),
            "region": _first_non_null(group["region"]) or "unknown",
            "spike_channel": _first_non_null(group["spike_channel"]),
            "recorded_agent": _first_non_null(group["recorded_agent"]),
            "recorded_monkey": _first_non_null(group["recorded_monkey"]),
            "area": _first_non_null(group["area"]),
            "n_conditions_observed": int(len(set(observed))),
            "all_target_conditions_present": bool(
                set(str(condition) for condition in settings.condition_order).issubset(set(observed))
            ),
            "peakiness_score": max_score,
            "best_condition": None,
            "best_peak_latency_ms": np.nan,
            "best_peak_prominence": 0.0,
            "best_peak_dominance": 0.0,
            "best_peak_competition_ratio": 0.0,
        }
        for condition in settings.condition_order:
            condition_row = group.loc[group["condition"].astype(str) == str(condition)]
            if condition_row.empty:
                row[_condition_column(condition, "peakiness_score")] = np.nan
                row[_condition_column(condition, "best_peak_prominence")] = np.nan
                row[_condition_column(condition, "second_peak_prominence")] = np.nan
                row[_condition_column(condition, "dominance")] = np.nan
                row[_condition_column(condition, "competition_ratio")] = np.nan
                row[_condition_column(condition, "best_peak_latency_ms")] = np.nan
                row[_condition_column(condition, "n_detected_peaks")] = np.nan
                row[_condition_column(condition, "mean_fr_hz")] = np.nan
                row[_condition_column(condition, "n_trials")] = np.nan
                continue
            picked = condition_row.iloc[0]
            row[_condition_column(condition, "peakiness_score")] = float(picked["peakiness_score"])
            row[_condition_column(condition, "best_peak_prominence")] = float(picked["best_peak_prominence"])
            row[_condition_column(condition, "second_peak_prominence")] = float(picked["second_peak_prominence"])
            row[_condition_column(condition, "dominance")] = float(picked["dominance"])
            row[_condition_column(condition, "competition_ratio")] = float(picked["competition_ratio"])
            row[_condition_column(condition, "best_peak_latency_ms")] = float(picked["best_peak_latency_ms"])
            row[_condition_column(condition, "n_detected_peaks")] = int(picked["n_detected_peaks"])
            row[_condition_column(condition, "mean_fr_hz")] = float(picked["mean_fr_hz"])
            row[_condition_column(condition, "n_trials")] = _coerce_n_trials(picked["n_trials"])

        if np.isfinite(max_score) and max_score > 0.0:
            condition_priority = {str(name): idx for idx, name in enumerate(settings.condition_order)}
            best_rows = group.loc[np.isclose(score_values, max_score, equal_nan=False)].copy()
            best_rows["_priority"] = best_rows["condition"].map(
                lambda value: condition_priority.get(str(value), len(condition_priority))
            )
            best_rows = best_rows.sort_values("_priority", kind="stable")
            best = best_rows.iloc[0]
            row["best_condition"] = str(best["condition"])
            row["best_peak_latency_ms"] = float(best["best_peak_latency_ms"])
            row["best_peak_prominence"] = float(best["best_peak_prominence"])
            row["best_peak_dominance"] = float(best["dominance"])
            row["best_peak_competition_ratio"] = float(best["competition_ratio"])

        rows.append(row)
    return pd.DataFrame(rows)


def _ordered_regions(
    unit_df: pd.DataFrame,
    settings: FixationPeakinessSettings,
) -> list[str]:
    observed = [str(value) for value in unit_df["region"].astype(str).dropna().unique().tolist()]
    if settings.region_order is None:
        return observed
    ordered: list[str] = []
    wanted = [str(region) for region in settings.region_order]
    for region in wanted:
        if region in observed and region not in ordered:
            ordered.append(region)
    for region in observed:
        if region not in ordered:
            ordered.append(region)
    return ordered


def _summary_row(
    *,
    region: str,
    score_scope: str,
    values: np.ndarray,
) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "region": region,
            "score_scope": score_scope,
            "n_units": 0,
            "n_nonzero_score": 0,
            "fraction_nonzero_score": np.nan,
            "mean_peakiness": np.nan,
            "std_peakiness": np.nan,
            "median_peakiness": np.nan,
            "q10_peakiness": np.nan,
            "q25_peakiness": np.nan,
            "q75_peakiness": np.nan,
            "q90_peakiness": np.nan,
            "min_peakiness": np.nan,
            "max_peakiness": np.nan,
        }
    return {
        "region": region,
        "score_scope": score_scope,
        "n_units": int(finite.size),
        "n_nonzero_score": int((finite > 0.0).sum()),
        "fraction_nonzero_score": float((finite > 0.0).sum()) / float(finite.size),
        "mean_peakiness": float(np.mean(finite)),
        "std_peakiness": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "median_peakiness": float(np.median(finite)),
        "q10_peakiness": float(np.quantile(finite, 0.10)),
        "q25_peakiness": float(np.quantile(finite, 0.25)),
        "q75_peakiness": float(np.quantile(finite, 0.75)),
        "q90_peakiness": float(np.quantile(finite, 0.90)),
        "min_peakiness": float(np.min(finite)),
        "max_peakiness": float(np.max(finite)),
    }


def _build_region_summary(
    unit_df: pd.DataFrame,
    settings: FixationPeakinessSettings,
) -> pd.DataFrame:
    if unit_df.empty:
        return pd.DataFrame(
            columns=[
                "region",
                "score_scope",
                "n_units",
                "n_nonzero_score",
                "fraction_nonzero_score",
                "mean_peakiness",
                "std_peakiness",
                "median_peakiness",
                "q10_peakiness",
                "q25_peakiness",
                "q75_peakiness",
                "q90_peakiness",
                "min_peakiness",
                "max_peakiness",
            ]
        )

    rows: list[dict[str, object]] = []
    regions = _ordered_regions(unit_df, settings)
    scope_to_column = {"any_condition_max": "peakiness_score"}
    for condition in settings.condition_order:
        scope_to_column[str(condition)] = _condition_column(str(condition), "peakiness_score")

    for region in regions:
        region_df = unit_df.loc[unit_df["region"].astype(str) == region]
        for scope, column in scope_to_column.items():
            values = pd.to_numeric(region_df[column], errors="coerce").to_numpy(dtype=float)
            rows.append(_summary_row(region=region, score_scope=scope, values=values))
    return pd.DataFrame(rows)


def run_fixation_peakiness_analysis(
    settings: FixationPeakinessSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    unit_uuids: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Run peakiness scoring from date-level average fixation PSTHs."""
    trace_rows = _load_condition_trace_rows(settings, dates=dates)
    trace_rows = _aggregate_condition_trace_rows(trace_rows)
    condition_df = _build_condition_peakiness_table(trace_rows, settings)
    if regions is not None and not condition_df.empty:
        allowed = {str(region) for region in regions}
        condition_df = condition_df.loc[condition_df["region"].astype(str).isin(allowed)].copy()
    unit_df = _build_unit_peakiness_table(condition_df, settings)
    region_summary_df = _build_region_summary(unit_df, settings)

    queried_df = pd.DataFrame()
    if unit_uuids is not None:
        requested = {str(unit_uuid).strip() for unit_uuid in unit_uuids if str(unit_uuid).strip()}
        queried_df = unit_df.loc[unit_df["unit_uuid"].astype(str).isin(requested)].copy()

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    unit_path = out_root / ensure_filename(settings.unit_output_filename, ".csv")
    condition_path = out_root / ensure_filename(settings.condition_output_filename, ".csv")
    summary_path = out_root / ensure_filename(settings.region_summary_filename, ".csv")
    result_path = out_root / ensure_filename(settings.output_pickle_filename, ".pkl")
    unit_df.to_csv(unit_path, index=False)
    condition_df.to_csv(condition_path, index=False)
    region_summary_df.to_csv(summary_path, index=False)

    result: dict[str, object] = {
        "meta": {
            "average_input_subdir": str(settings.average_input_subdir),
            "average_input_filename": ensure_filename(settings.average_input_filename, ".pkl"),
            "condition_order": list(settings.condition_order),
            "min_trials_per_condition": int(settings.min_trials_per_condition),
            "mean_rate_floor_hz": float(settings.mean_rate_floor_hz),
            "peak_distance_ms": float(settings.peak_distance_ms),
            "peak_prominence_floor": float(settings.peak_prominence_floor),
            "competition_penalty_lambda": float(settings.competition_penalty_lambda),
            "prominence_epsilon": float(settings.prominence_epsilon),
            "n_units": int(len(unit_df)),
            "n_condition_rows": int(len(condition_df)),
        },
        "unit_peakiness": unit_df,
        "condition_peakiness": condition_df,
        "region_summary": region_summary_df,
    }
    if unit_uuids is not None:
        result["queried_units"] = queried_df
        result["meta"]["queried_unit_uuids"] = [str(unit_uuid) for unit_uuid in unit_uuids]

    save_pickle_path(result, result_path)
    return result
