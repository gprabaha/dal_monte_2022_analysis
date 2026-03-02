"""Shared helper functions for fixation neural cross-correlation analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.behav.roi_groups import (
    DEFAULT_FIXATION_ROI_GROUPS as DEFAULT_SHARED_FIXATION_ROI_GROUPS,
    canonical_fixation_category,
    categorize_locations,
    normalize_roi_groups,
)
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
    ensure_filename as _ensure_filename,
    extract_trials_df_and_meta as _extract_trials_df_and_meta,
)
from dal_monte_2022_analysis.core.signal.cross_correlation import (
    assert_lag_axis_match as assert_lag_axis_match_shared,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_paths
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
)

WITHIN_ANALYSIS_KIND = "within_region"
CROSS_ANALYSIS_KIND = "cross_region"
_PLOT_ALLOWED_ANALYSIS_KINDS = (WITHIN_ANALYSIS_KIND, CROSS_ANALYSIS_KIND)
_ALLOWED_SIGNAL_TRANSFORMS = {"none", "demean", "zscore"}
_ALLOWED_XCORR_NORMALIZATIONS = {"none", "energy"}
_PLOT_CONDITION_ORDER = ("face_interactive", "face_non_interactive", "object")
_REGION_TOKEN_RE = re.compile(r"[^a-z0-9]+")
DEFAULT_FIXATION_ROI_GROUPS: dict[str, tuple[str, ...]] = DEFAULT_SHARED_FIXATION_ROI_GROUPS


@dataclass
class FixationNeuralCrossCorrelationPlotAggregationSettings:
    """Configuration for analysis-side aggregation used by plotting."""

    cfg_path: str
    within_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/within_region"
    cross_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/cross_region"
    within_input_filename: str = "fixations.pkl"
    cross_input_filename: str = "fixations.pkl"
    within_pair_average_input_filename: str = "pair_averages.pkl"
    cross_pair_average_input_filename: str = "pair_averages.pkl"
    face_label: str = "face"
    object_label: str = "object"
    interactive_label: str = "interactive"
    condition_order: Sequence[str] = field(default_factory=lambda: tuple(_PLOT_CONDITION_ORDER))

def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _coerce_location_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        out = []
        for item in value:
            token = _as_optional_str(item)
            if token is not None:
                out.append(token)
        return tuple(out)
    token = _as_optional_str(value)
    return tuple() if token is None else (token,)


def _canonical_region_name(value: Optional[str]) -> Optional[str]:
    token = _as_optional_str(value)
    if token is None:
        return None
    canonical = _REGION_TOKEN_RE.sub("", token.lower())
    return canonical or None


def _normalize_region_keys(regions: Optional[Sequence[str]]) -> Optional[set[str]]:
    if regions is None:
        return None
    keys: set[str] = set()
    for region in regions:
        key = _canonical_region_name(region)
        if key is not None:
            keys.add(key)
    return keys


def _validate_signal_transform(transform: str) -> str:
    token = str(transform).strip().lower()
    if token not in _ALLOWED_SIGNAL_TRANSFORMS:
        allowed = ", ".join(sorted(_ALLOWED_SIGNAL_TRANSFORMS))
        raise ValueError(f"Unsupported signal_transform='{transform}'. Expected one of: {allowed}.")
    return token


def _validate_xcorr_normalization(normalization: str) -> str:
    token = str(normalization).strip().lower()
    if token not in _ALLOWED_XCORR_NORMALIZATIONS:
        allowed = ", ".join(sorted(_ALLOWED_XCORR_NORMALIZATIONS))
        raise ValueError(
            f"Unsupported xcorr_normalization='{normalization}'. Expected one of: {allowed}.",
        )
    return token


def _normalize_roi_groups(groups: Optional[dict[str, Sequence[str]]]) -> dict[str, list[str]]:
    return normalize_roi_groups(
        groups,
        include_defaults=True,
        default_groups=DEFAULT_FIXATION_ROI_GROUPS,
    )


def _canonical_fixation_category(value: Optional[str]) -> Optional[str]:
    return canonical_fixation_category(value)


def _infer_fixation_category_from_locations(
    locations: tuple[str, ...],
    roi_groups: dict[str, list[str]],
) -> Optional[str]:
    return categorize_locations(
        list(locations),
        roi_groups,
        ordered_groups=("face", "object", "out_of_roi"),
        allowed_categories=None,
    )


def _select_preferred_rows(
    preferred_rows: Sequence[dict],
    fallback_rows: Sequence[dict],
) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for row in fallback_rows:
        key = (str(row["date"]), str(row["session"]))
        by_key[key] = row
    for row in preferred_rows:
        key = (str(row["date"]), str(row["session"]))
        by_key[key] = row
    out = list(by_key.values())
    out.sort(key=lambda row: (row["date"], row["session"]))
    return out


def _extract_xcorr_dataframes_and_meta(obj) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if isinstance(obj, dict):
        xcorr_df = obj.get("cross_correlations")
        pair_avg_df = obj.get("pair_averages")
        meta = obj.get("meta", {}) or {}
        return (
            xcorr_df if isinstance(xcorr_df, pd.DataFrame) else pd.DataFrame(),
            pair_avg_df if isinstance(pair_avg_df, pd.DataFrame) else pd.DataFrame(),
            meta,
        )
    if isinstance(obj, pd.DataFrame):
        return obj, pd.DataFrame(), {}
    return pd.DataFrame(), pd.DataFrame(), {}


def _resolve_plot_condition_from_row(
    row,
    *,
    face_label: str,
    object_label: str,
    interactive_label: str,
) -> Optional[str]:
    category = _canonical_fixation_category(getattr(row, "fixation_category", None))
    if category is None:
        return None

    if category == str(object_label):
        return "object"
    if category != str(face_label):
        return None

    if hasattr(row, "is_interactive"):
        interactive = _as_bool(getattr(row, "is_interactive"), interactive_label)
    else:
        interactive = _as_bool(getattr(row, "interactive_state", None), interactive_label)
    return "face_interactive" if interactive else "face_non_interactive"


def _normalize_region_pair_label(region_1: Optional[str], region_2: Optional[str]) -> str:
    left = _as_optional_str(region_1) or "unknown_1"
    right = _as_optional_str(region_2) or "unknown_2"
    return f"{left}__{right}"


def _append_trace_sum(
    accum: dict[tuple, list[object]],
    key: tuple,
    trace: np.ndarray,
    *,
    weight: float = 1.0,
) -> None:
    if key not in accum:
        accum[key] = [np.zeros(trace.size, dtype=np.float64), 0.0]
    bucket = accum[key]
    if bucket[0].shape != trace.shape:
        raise ValueError("Encountered mismatched cross-correlation trace lengths while aggregating.")
    scalar = float(weight)
    if not np.isfinite(scalar) or scalar <= 0.0:
        return
    bucket[0] += np.asarray(trace, dtype=np.float64) * scalar
    bucket[1] = float(bucket[1]) + scalar


def _resolve_region_for_row(row) -> Optional[str]:
    region = _as_optional_str(getattr(row, "region", None))
    if region is not None:
        return region
    return _as_optional_str(getattr(row, "area", None))


def _build_fixation_key_and_meta(
    row,
    row_index: int,
    *,
    default_date: str,
    default_session: str,
    roi_groups: dict[str, list[str]],
) -> tuple[tuple, dict]:
    date = _as_optional_str(getattr(row, "date", None)) or str(default_date)
    session = _as_optional_str(getattr(row, "session", None)) or str(default_session)
    fixation_agent = _as_optional_str(getattr(row, "fixation_agent", None))
    fixation_monkey_name = _as_optional_str(getattr(row, "fixation_monkey_name", None))
    fixation_location = _coerce_location_tuple(getattr(row, "fixation_location", None))
    fixation_category = _canonical_fixation_category(getattr(row, "fixation_category", None))
    if fixation_category is None:
        fixation_category = _infer_fixation_category_from_locations(fixation_location, roi_groups)
    fixation_start_idx = _safe_int(getattr(row, "fixation_start_idx", None))
    fixation_stop_idx = _safe_int(getattr(row, "fixation_stop_idx", None))
    fixation_start_time_s = _safe_float(getattr(row, "fixation_start_time_s", None))
    interactive_state = _as_optional_str(getattr(row, "interactive_state", None))
    is_interactive = _as_bool(getattr(row, "is_interactive", None))

    unique_row_idx = int(row_index) if fixation_start_idx is None else int(fixation_start_idx)
    key = (
        str(date),
        str(session),
        fixation_agent,
        int(unique_row_idx),
        fixation_stop_idx,
        fixation_start_time_s,
        fixation_category,
        fixation_location,
        interactive_state,
        bool(is_interactive),
    )

    meta = {
        "date": str(date),
        "session": str(session),
        "fixation_agent": fixation_agent,
        "fixation_monkey_name": fixation_monkey_name,
        "fixation_category": fixation_category,
        "fixation_location": fixation_location,
        "fixation_start_idx": fixation_start_idx,
        "fixation_stop_idx": fixation_stop_idx,
        "fixation_start_time_s": fixation_start_time_s,
        "interactive_state": interactive_state,
        "is_interactive": bool(is_interactive),
    }
    return key, meta


def _collect_fixation_groups(
    trial_df: pd.DataFrame,
    *,
    default_date: str,
    default_session: str,
    include_region_keys: Optional[set[str]],
    roi_groups: dict[str, list[str]],
) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    order: list[tuple] = []

    for row_index, row in enumerate(trial_df.itertuples(index=False)):
        counts = np.asarray(getattr(row, "psth_counts", []), dtype=np.float64).reshape(-1)
        if counts.size == 0:
            continue
        if not np.isfinite(counts).all():
            counts = np.where(np.isfinite(counts), counts, 0.0)

        unit_uuid = _as_optional_str(getattr(row, "unit_uuid", None))
        if unit_uuid is None:
            continue

        region_raw = _resolve_region_for_row(row)
        region_key = _canonical_region_name(region_raw)
        if region_key is None:
            continue
        if include_region_keys is not None and region_key not in include_region_keys:
            continue

        key, fixation_meta = _build_fixation_key_and_meta(
            row,
            row_index,
            default_date=default_date,
            default_session=default_session,
            roi_groups=roi_groups,
        )
        if key not in grouped:
            fixation_meta["fixation_id"] = int(len(order))
            grouped[key] = {"meta": fixation_meta, "units": {}}
            order.append(key)

        unit_key = (str(unit_uuid), str(region_key))
        if unit_key in grouped[key]["units"]:
            continue

        grouped[key]["units"][unit_key] = {
            "unit_uuid": str(unit_uuid),
            "region": region_raw,
            "region_key": str(region_key),
            "spike_channel": _as_optional_str(getattr(row, "spike_channel", None)),
            "session_name": _as_optional_str(getattr(row, "session_name", None)),
            "recorded_agent": _as_optional_str(getattr(row, "recorded_agent", None)),
            "recorded_monkey": _as_optional_str(getattr(row, "recorded_monkey", None)),
            "area": _as_optional_str(getattr(row, "area", None)),
            "psth_counts": counts,
        }

    out: list[dict] = []
    for key in order:
        payload = grouped[key]
        units = list(payload["units"].values())
        if not units:
            continue
        units.sort(key=lambda unit: (unit["region_key"], unit["unit_uuid"]))
        out.append({"meta": payload["meta"], "units": units})
    return out


def _apply_signal_transform(signal: np.ndarray, transform: str) -> np.ndarray:
    vec = np.asarray(signal, dtype=np.float64).reshape(-1)
    if vec.size == 0:
        return vec
    if not np.isfinite(vec).all():
        vec = np.where(np.isfinite(vec), vec, 0.0)

    if transform == "none":
        return vec

    centered = vec - float(np.mean(vec))
    if transform == "demean":
        return centered

    std = float(np.std(centered))
    if std <= 0.0 or not np.isfinite(std):
        return np.zeros_like(centered)
    return centered / std


def _fft_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_lag: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    return fft_cross_correlation(x, y, max_lag=max_lag, round_to_int=False)


def _normalize_cross_correlation(
    corr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str,
) -> np.ndarray:
    vec = np.asarray(corr, dtype=np.float64).reshape(-1)
    token = _validate_xcorr_normalization(normalization)
    if token == "none":
        return vec
    if token == "energy":
        return normalize_cross_correlation_energy(vec, x, y)
    raise ValueError(f"Unsupported xcorr normalization '{normalization}'.")


def _summarize_cross_correlation(lags: np.ndarray, corr: np.ndarray) -> dict:
    return summarize_cross_correlation(lags, corr)


_GLOBAL_FIXATION_META: list[dict] = []
_GLOBAL_SIGNAL_ENTRIES: list[dict] = []
_GLOBAL_SIGNAL_TRANSFORM: str = "none"
_GLOBAL_MAX_LAG: Optional[int] = None
_GLOBAL_XCORR_NORMALIZATION: str = "energy"


def _init_pair_worker(
    fixation_meta: list[dict],
    signal_entries: list[dict],
    signal_transform: str,
    max_lag: Optional[int],
    xcorr_normalization: str,
) -> None:
    global _GLOBAL_FIXATION_META, _GLOBAL_SIGNAL_ENTRIES
    global _GLOBAL_SIGNAL_TRANSFORM, _GLOBAL_MAX_LAG, _GLOBAL_XCORR_NORMALIZATION
    _GLOBAL_FIXATION_META = fixation_meta
    _GLOBAL_SIGNAL_ENTRIES = signal_entries
    _GLOBAL_SIGNAL_TRANSFORM = signal_transform
    _GLOBAL_MAX_LAG = max_lag
    _GLOBAL_XCORR_NORMALIZATION = _validate_xcorr_normalization(xcorr_normalization)


def _compute_pair_xcorr_worker(task: tuple[int, int, int]) -> Optional[dict]:
    fixation_idx, signal_idx_1, signal_idx_2 = task
    if signal_idx_1 == signal_idx_2:
        return None

    fixation_meta = _GLOBAL_FIXATION_META[fixation_idx]
    unit_1 = _GLOBAL_SIGNAL_ENTRIES[signal_idx_1]
    unit_2 = _GLOBAL_SIGNAL_ENTRIES[signal_idx_2]

    signal_1 = _apply_signal_transform(unit_1["psth_counts"], _GLOBAL_SIGNAL_TRANSFORM)
    signal_2 = _apply_signal_transform(unit_2["psth_counts"], _GLOBAL_SIGNAL_TRANSFORM)
    lags, corr = _fft_cross_correlation(signal_1, signal_2, max_lag=_GLOBAL_MAX_LAG)
    if corr.size == 0:
        return None
    corr = _normalize_cross_correlation(
        corr,
        signal_1,
        signal_2,
        normalization=_GLOBAL_XCORR_NORMALIZATION,
    )

    row = {
        **fixation_meta,
        "unit_uuid_1": unit_1["unit_uuid"],
        "region_1": unit_1["region"],
        "spike_channel_1": unit_1["spike_channel"],
        "session_name_1": unit_1["session_name"],
        "recorded_agent_1": unit_1["recorded_agent"],
        "recorded_monkey_1": unit_1["recorded_monkey"],
        "area_1": unit_1["area"],
        "unit_uuid_2": unit_2["unit_uuid"],
        "region_2": unit_2["region"],
        "spike_channel_2": unit_2["spike_channel"],
        "session_name_2": unit_2["session_name"],
        "recorded_agent_2": unit_2["recorded_agent"],
        "recorded_monkey_2": unit_2["recorded_monkey"],
        "area_2": unit_2["area"],
        "cross_correlation": corr.astype(np.float32),
        "lags": lags,
        "signal_bins_1": int(signal_1.size),
        "signal_bins_2": int(signal_2.size),
        "signal_mean_1": float(np.mean(signal_1)),
        "signal_mean_2": float(np.mean(signal_2)),
        "signal_std_1": float(np.std(signal_1)),
        "signal_std_2": float(np.std(signal_2)),
    }
    row.update(_summarize_cross_correlation(lags, corr))
    return row


def _build_pair_tasks(
    fixation_groups: Sequence[dict],
    *,
    analysis_kind: str,
    anchor_region_key: Optional[str],
    partner_region_keys: Optional[set[str]],
) -> tuple[list[dict], list[dict], list[tuple[int, int, int]], int]:
    fixation_meta: list[dict] = []
    signal_entries: list[dict] = []
    tasks: list[tuple[int, int, int]] = []
    n_fixations_with_pairs = 0

    for fixation_payload in fixation_groups:
        meta = dict(fixation_payload["meta"])
        units = list(fixation_payload["units"])
        if not units:
            continue

        fixation_idx = len(fixation_meta)
        fixation_meta.append(meta)

        local_signal_indices: list[int] = []
        for unit in units:
            signal_entries.append(unit)
            local_signal_indices.append(len(signal_entries) - 1)

        before = len(tasks)
        if analysis_kind == WITHIN_ANALYSIS_KIND:
            region_to_signal_indices: dict[str, list[int]] = {}
            for signal_idx in local_signal_indices:
                region_key = str(signal_entries[signal_idx]["region_key"])
                region_to_signal_indices.setdefault(region_key, []).append(signal_idx)

            for region_key in sorted(region_to_signal_indices):
                signal_ids = sorted(region_to_signal_indices[region_key])
                if len(signal_ids) < 2:
                    continue
                for i in range(len(signal_ids) - 1):
                    for j in range(i + 1, len(signal_ids)):
                        tasks.append((fixation_idx, signal_ids[i], signal_ids[j]))

        elif analysis_kind == CROSS_ANALYSIS_KIND:
            if anchor_region_key is None:
                raise ValueError("anchor_region must be defined for cross-region analysis.")

            anchor_signal_ids = [
                signal_idx
                for signal_idx in local_signal_indices
                if signal_entries[signal_idx]["region_key"] == anchor_region_key
            ]

            if partner_region_keys is None:
                partner_signal_ids = [
                    signal_idx
                    for signal_idx in local_signal_indices
                    if signal_entries[signal_idx]["region_key"] != anchor_region_key
                ]
            else:
                partner_signal_ids = [
                    signal_idx
                    for signal_idx in local_signal_indices
                    if signal_entries[signal_idx]["region_key"] in partner_region_keys
                ]

            if anchor_signal_ids and partner_signal_ids:
                for signal_idx_anchor in sorted(anchor_signal_ids):
                    for signal_idx_partner in sorted(partner_signal_ids):
                        tasks.append((fixation_idx, signal_idx_anchor, signal_idx_partner))
        else:
            raise ValueError(f"Unsupported analysis_kind='{analysis_kind}'.")

        if len(tasks) > before:
            n_fixations_with_pairs += 1

    return fixation_meta, signal_entries, tasks, n_fixations_with_pairs


def _assert_lag_axis_match(reference_lags: np.ndarray, lags: np.ndarray) -> None:
    assert_lag_axis_match_shared(
        reference_lags,
        lags,
        message=(
            "Encountered inconsistent lag vectors across fixation-neuron pairs. "
            "Use consistent PSTH windows/binning and a fixed max_lag."
        ),
    )


def _sort_result_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [
        "fixation_id",
        "fixation_start_idx",
        "region_1",
        "unit_uuid_1",
        "region_2",
        "unit_uuid_2",
    ]
    available = [col for col in sort_cols if col in df.columns]
    if not available:
        return df.reset_index(drop=True)
    return df.sort_values(available).reset_index(drop=True)


def _build_session_pair_average_dataframe(
    xcorr_df: pd.DataFrame,
    *,
    analysis_kind: str,
    lags: Optional[np.ndarray],
    face_label: str,
    object_label: str,
    interactive_label: str,
) -> pd.DataFrame:
    if xcorr_df.empty or "cross_correlation" not in xcorr_df.columns:
        return pd.DataFrame()

    lags_ref = np.asarray(lags, dtype=np.int64).reshape(-1) if lags is not None else None
    lags_shape = None if lags_ref is None else lags_ref.shape
    trace_accum: dict[tuple, list[object]] = {}
    for xrow in xcorr_df.itertuples(index=False):
        condition = _resolve_plot_condition_from_row(
            xrow,
            face_label=face_label,
            object_label=object_label,
            interactive_label=interactive_label,
        )
        if condition is None:
            continue

        trace = np.asarray(getattr(xrow, "cross_correlation"), dtype=np.float64).reshape(-1)
        if trace.size == 0:
            continue
        if lags_shape is not None and lags_shape != trace.shape:
            continue

        region_1 = _as_optional_str(getattr(xrow, "region_1", None)) or "unknown_1"
        region_2 = _as_optional_str(getattr(xrow, "region_2", None)) or "unknown_2"
        unit_uuid_1 = _as_optional_str(getattr(xrow, "unit_uuid_1", None)) or "unknown_unit_1"
        unit_uuid_2 = _as_optional_str(getattr(xrow, "unit_uuid_2", None)) or "unknown_unit_2"
        date = str(_as_optional_str(getattr(xrow, "date", None)) or "unknown_date")
        session = str(_as_optional_str(getattr(xrow, "session", None)) or "unknown_session")

        if analysis_kind == WITHIN_ANALYSIS_KIND:
            group_label = region_1
            pair_key = (date, session, group_label, region_1, region_1, unit_uuid_1, unit_uuid_2, condition)
        elif analysis_kind == CROSS_ANALYSIS_KIND:
            group_label = _normalize_region_pair_label(region_1, region_2)
            pair_key = (date, session, group_label, region_1, region_2, unit_uuid_1, unit_uuid_2, condition)
        else:
            raise ValueError(f"Unsupported analysis kind '{analysis_kind}' for session pair averages.")

        _append_trace_sum(trace_accum, pair_key, trace, weight=1.0)

    rows: list[dict] = []
    for key, (trace_sum, n_fix) in trace_accum.items():
        if float(n_fix) <= 0.0:
            continue

        date, session, group_label, region_1, region_2, unit_uuid_1, unit_uuid_2, condition = key
        pair_avg = np.asarray(trace_sum, dtype=np.float64) / float(n_fix)
        if lags_ref is not None and lags_ref.size == pair_avg.size:
            summary = _summarize_cross_correlation(lags_ref, pair_avg)
        else:
            summary = {
                "n_lags": int(pair_avg.size),
                "zero_lag_correlation": None,
                "peak_lag": None,
                "peak_correlation": float(np.max(pair_avg)) if pair_avg.size else None,
            }

        rows.append(
            {
                "analysis_kind": analysis_kind,
                "date": date,
                "session": session,
                "group_label": group_label,
                "condition": condition,
                "region_1": region_1,
                "region_2": region_2,
                "unit_uuid_1": unit_uuid_1,
                "unit_uuid_2": unit_uuid_2,
                "n_fixations": int(round(float(n_fix))),
                "cross_correlation": pair_avg.astype(np.float32),
                **summary,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    sort_cols = [
        "date",
        "session",
        "group_label",
        "region_1",
        "unit_uuid_1",
        "region_2",
        "unit_uuid_2",
        "condition",
    ]
    available = [col for col in sort_cols if col in out.columns]
    if available:
        out = out.sort_values(available).reset_index(drop=True)
    return out


def _aggregate_pair_averages_for_plotting(
    settings: FixationNeuralCrossCorrelationPlotAggregationSettings,
    *,
    kind: str,
    rows: Sequence[dict],
    show_progress: bool,
) -> tuple[dict, dict, Optional[np.ndarray], Optional[float], dict[str, int]]:
    if not rows:
        return {}, {}, None, None, {
            "files": 0,
            "files_using_pair_averages": 0,
            "rows": 0,
            "used_rows": 0,
            "skipped_rows": 0,
        }

    lag_axis_ref: Optional[np.ndarray] = None
    bin_size_ms_ref: Optional[float] = None
    date_pair_accum: dict[tuple, list[object]] = {}
    kind_label = "within-region" if kind == WITHIN_ANALYSIS_KIND else "cross-region"
    valid_conditions = set(settings.condition_order)

    counts = {
        "files": 0,
        "files_using_pair_averages": 0,
        "rows": 0,
        "used_rows": 0,
        "skipped_rows": 0,
    }

    file_iter = rows
    if show_progress:
        file_iter = tqdm(rows, total=len(rows), desc=f"Aggregate {kind_label} files", unit="file")
    for row in file_iter:
        counts["files"] += 1
        obj = load_pickle_path(Path(row["path"]))
        xcorr_df, pair_avg_df, meta = _extract_xcorr_dataframes_and_meta(obj)
        has_pair_averages = bool(
            not pair_avg_df.empty
            and "cross_correlation" in pair_avg_df.columns
            and "condition" in pair_avg_df.columns
            and "n_fixations" in pair_avg_df.columns
        )
        if has_pair_averages:
            counts["files_using_pair_averages"] += 1
        source_df = pair_avg_df if has_pair_averages else xcorr_df
        if source_df.empty or "cross_correlation" not in source_df.columns:
            continue

        local_lags = np.asarray(meta.get("lags", []), dtype=np.int64).reshape(-1)
        if local_lags.size == 0 and "lags" in xcorr_df.columns and len(xcorr_df):
            local_lags = np.asarray(xcorr_df.iloc[0]["lags"], dtype=np.int64).reshape(-1)
        if local_lags.size == 0:
            continue

        if lag_axis_ref is None:
            lag_axis_ref = local_lags
        elif lag_axis_ref.shape != local_lags.shape or not np.array_equal(lag_axis_ref, local_lags):
            print(f"[plot-xcorr] skipping file due to lag mismatch: {row['path']}")
            continue

        local_bin_size_ms = meta.get("bin_size_ms")
        if local_bin_size_ms is not None:
            try:
                local_bin_size_ms = float(local_bin_size_ms)
                if np.isfinite(local_bin_size_ms):
                    if bin_size_ms_ref is None:
                        bin_size_ms_ref = local_bin_size_ms
                    elif not np.isclose(bin_size_ms_ref, local_bin_size_ms):
                        bin_size_ms_ref = None
            except Exception:
                pass

        date = str(row["date"])
        for xrow in source_df.itertuples(index=False):
            counts["rows"] += 1
            trace = np.asarray(getattr(xrow, "cross_correlation"), dtype=np.float64).reshape(-1)
            if lag_axis_ref is None or trace.shape != lag_axis_ref.shape:
                counts["skipped_rows"] += 1
                continue

            if has_pair_averages:
                condition = _as_optional_str(getattr(xrow, "condition", None))
                if condition is None or condition not in valid_conditions:
                    counts["skipped_rows"] += 1
                    continue
                weight = _safe_float(getattr(xrow, "n_fixations", None))
                if weight is None or weight <= 0.0:
                    counts["skipped_rows"] += 1
                    continue
            else:
                condition = _resolve_plot_condition_from_row(
                    xrow,
                    face_label=settings.face_label,
                    object_label=settings.object_label,
                    interactive_label=settings.interactive_label,
                )
                weight = 1.0
            if condition is None:
                counts["skipped_rows"] += 1
                continue

            if kind == WITHIN_ANALYSIS_KIND:
                region = _as_optional_str(getattr(xrow, "region_1", None))
                group_label = region or "unknown_region"
                pair_id = (
                    _as_optional_str(getattr(xrow, "unit_uuid_1", None)) or "unknown_unit_1",
                    _as_optional_str(getattr(xrow, "unit_uuid_2", None)) or "unknown_unit_2",
                )
            else:
                region_1 = _as_optional_str(getattr(xrow, "region_1", None))
                region_2 = _as_optional_str(getattr(xrow, "region_2", None))
                group_label = _normalize_region_pair_label(region_1, region_2)
                pair_id = (
                    region_1 or "unknown_1",
                    region_2 or "unknown_2",
                    _as_optional_str(getattr(xrow, "unit_uuid_1", None)) or "unknown_unit_1",
                    _as_optional_str(getattr(xrow, "unit_uuid_2", None)) or "unknown_unit_2",
                )

            key = (kind, date, group_label, condition, pair_id)
            _append_trace_sum(date_pair_accum, key, trace, weight=float(weight))
            counts["used_rows"] += 1

        if show_progress and counts["files"] % 16 == 0 and hasattr(file_iter, "set_postfix"):
            file_iter.set_postfix(
                rows=counts["rows"],
                used=counts["used_rows"],
                skipped=counts["skipped_rows"],
                refresh=False,
            )

    if lag_axis_ref is None:
        return {}, {}, None, None, counts

    date_plot_map: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = {}
    global_plot_map: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    pair_items = date_pair_accum.items()
    if show_progress:
        pair_items = tqdm(
            pair_items,
            total=len(date_pair_accum),
            desc=f"Assemble {kind_label} pair averages",
            unit="pair",
        )
    for key, (trace_sum, n) in pair_items:
        if int(n) <= 0:
            continue
        kind_key, date, group_label, condition, _pair_id = key
        pair_avg = (np.asarray(trace_sum, dtype=np.float64) / float(n)).astype(np.float64)

        date_bucket = date_plot_map.setdefault(
            (kind_key, date, group_label),
            {name: [] for name in settings.condition_order},
        )
        date_bucket.setdefault(condition, []).append(pair_avg)

        global_bucket = global_plot_map.setdefault(
            (kind_key, group_label),
            {name: [] for name in settings.condition_order},
        )
        global_bucket.setdefault(condition, []).append(pair_avg)

    return date_plot_map, global_plot_map, lag_axis_ref, bin_size_ms_ref, counts


def _build_plot_x_axis(
    lags: Optional[np.ndarray],
    bin_size_ms: Optional[float],
) -> tuple[np.ndarray, str]:
    if lags is None:
        return np.array([], dtype=float), "Lag"
    if bin_size_ms is None:
        return np.asarray(lags, dtype=float), "Lag (bins)"
    return np.asarray(lags, dtype=float) * float(bin_size_ms) / 1000.0, "Lag (s)"


def _normalize_plot_analysis_kinds(
    analysis_kinds: Optional[Sequence[str]],
) -> tuple[str, ...]:
    if analysis_kinds is None:
        return tuple(_PLOT_ALLOWED_ANALYSIS_KINDS)
    normalized: list[str] = []
    for kind in analysis_kinds:
        token = str(kind).strip()
        if token not in _PLOT_ALLOWED_ANALYSIS_KINDS:
            allowed = ", ".join(_PLOT_ALLOWED_ANALYSIS_KINDS)
            raise ValueError(
                f"Unsupported analysis kind for plot payload '{kind}'. Expected one of: {allowed}.",
            )
        normalized.append(token)
    if not normalized:
        raise ValueError("analysis_kinds cannot be empty.")
    return tuple(dict.fromkeys(normalized))


def build_fixation_neural_cross_correlation_plot_payload(
    settings: FixationNeuralCrossCorrelationPlotAggregationSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    analysis_kinds: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict:
    """Aggregate fixation neural xcorr outputs into plot-ready date/global maps."""
    cfg = load_config(settings.cfg_path)
    selected_kinds = set(_normalize_plot_analysis_kinds(analysis_kinds))

    empty_counts = {
        "files": 0,
        "files_using_pair_averages": 0,
        "rows": 0,
        "used_rows": 0,
        "skipped_rows": 0,
    }

    within_date_map: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = {}
    within_global_map: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    within_lags: Optional[np.ndarray] = None
    within_bin_ms: Optional[float] = None
    within_counts = dict(empty_counts)

    cross_date_map: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = {}
    cross_global_map: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    cross_lags: Optional[np.ndarray] = None
    cross_bin_ms: Optional[float] = None
    cross_counts = dict(empty_counts)

    if WITHIN_ANALYSIS_KIND in selected_kinds:
        within_xcorr_rows = scan_analysis_paths(
            cfg,
            settings.within_input_subdir,
            filename=_ensure_filename(settings.within_input_filename, ".pkl"),
            dates=dates,
            sessions=sessions,
        )
        within_pair_rows = scan_analysis_paths(
            cfg,
            settings.within_input_subdir,
            filename=_ensure_filename(settings.within_pair_average_input_filename, ".pkl"),
            dates=dates,
            sessions=sessions,
        )
        within_rows = _select_preferred_rows(within_pair_rows, within_xcorr_rows)
        within_date_map, within_global_map, within_lags, within_bin_ms, within_counts = (
            _aggregate_pair_averages_for_plotting(
                settings,
                kind=WITHIN_ANALYSIS_KIND,
                rows=within_rows,
                show_progress=show_progress,
            )
        )

    if CROSS_ANALYSIS_KIND in selected_kinds:
        cross_xcorr_rows = scan_analysis_paths(
            cfg,
            settings.cross_input_subdir,
            filename=_ensure_filename(settings.cross_input_filename, ".pkl"),
            dates=dates,
            sessions=sessions,
        )
        cross_pair_rows = scan_analysis_paths(
            cfg,
            settings.cross_input_subdir,
            filename=_ensure_filename(settings.cross_pair_average_input_filename, ".pkl"),
            dates=dates,
            sessions=sessions,
        )
        cross_rows = _select_preferred_rows(cross_pair_rows, cross_xcorr_rows)
        cross_date_map, cross_global_map, cross_lags, cross_bin_ms, cross_counts = (
            _aggregate_pair_averages_for_plotting(
                settings,
                kind=CROSS_ANALYSIS_KIND,
                rows=cross_rows,
                show_progress=show_progress,
            )
        )

    x_axes: dict[str, np.ndarray] = {}
    x_labels: dict[str, str] = {}
    if WITHIN_ANALYSIS_KIND in selected_kinds:
        within_x, within_x_label = _build_plot_x_axis(within_lags, within_bin_ms)
        x_axes[WITHIN_ANALYSIS_KIND] = within_x
        x_labels[WITHIN_ANALYSIS_KIND] = within_x_label
    if CROSS_ANALYSIS_KIND in selected_kinds:
        cross_x, cross_x_label = _build_plot_x_axis(cross_lags, cross_bin_ms)
        x_axes[CROSS_ANALYSIS_KIND] = cross_x
        x_labels[CROSS_ANALYSIS_KIND] = cross_x_label

    date_plot_map: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = {}
    if WITHIN_ANALYSIS_KIND in selected_kinds:
        date_plot_map.update(within_date_map)
    if CROSS_ANALYSIS_KIND in selected_kinds:
        date_plot_map.update(cross_date_map)

    global_plot_map: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    if WITHIN_ANALYSIS_KIND in selected_kinds:
        global_plot_map.update(within_global_map)
    if CROSS_ANALYSIS_KIND in selected_kinds:
        global_plot_map.update(cross_global_map)

    return {
        "cfg": cfg,
        "analysis_kinds": tuple(sorted(selected_kinds)),
        "date_plot_map": date_plot_map,
        "global_plot_map": global_plot_map,
        "x_axes": x_axes,
        "x_labels": x_labels,
        "within_counts": within_counts,
        "cross_counts": cross_counts,
    }


def build_within_region_fixation_neural_cross_correlation_plot_payload(
    settings: FixationNeuralCrossCorrelationPlotAggregationSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict:
    return build_fixation_neural_cross_correlation_plot_payload(
        settings,
        dates=dates,
        sessions=sessions,
        analysis_kinds=(WITHIN_ANALYSIS_KIND,),
        show_progress=show_progress,
    )


def build_cross_region_fixation_neural_cross_correlation_plot_payload(
    settings: FixationNeuralCrossCorrelationPlotAggregationSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> dict:
    return build_fixation_neural_cross_correlation_plot_payload(
        settings,
        dates=dates,
        sessions=sessions,
        analysis_kinds=(CROSS_ANALYSIS_KIND,),
        show_progress=show_progress,
    )
