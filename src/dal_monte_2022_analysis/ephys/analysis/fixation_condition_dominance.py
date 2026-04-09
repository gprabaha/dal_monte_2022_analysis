"""Summarize dominant fixation-condition firing by region."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
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


DOMINANCE_CONDITIONS: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
DOMINANCE_UNIT_SUBSETS: tuple[str, ...] = (
    "all_units",
    "raw_selective_units",
    "corrected_selective_units",
)


@dataclass
class FixationConditionDominanceSettings:
    """Configuration for region-level fixation-condition dominance summaries."""

    cfg_path: str
    average_input_subdir: str = "ephys/psth/fixation_psth_averages"
    average_input_filename: str = "fixations_psth_10ms.pkl"
    selectivity_input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    selectivity_unit_summary_filename: str = "unit_selectivity.csv"
    output_subdir: str = "ephys/psth/fixation_condition_dominance"
    unit_output_filename: str = "unit_condition_dominance.csv"
    region_summary_filename: str = "region_condition_dominance_summary.csv"
    output_pickle_filename: str = "results.pkl"
    window_start_ms: float = -500.0
    window_stop_ms: float = 500.0
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    min_trials_per_condition: int = 1
    tie_tolerance_hz: float = 1e-12
    bin_size_ms_fallback: float = 10.0
    condition_order: tuple[str, ...] = field(default_factory=lambda: DOMINANCE_CONDITIONS)
    unit_subset_order: tuple[str, ...] = field(default_factory=lambda: DOMINANCE_UNIT_SUBSETS)
    region_order: Optional[Sequence[str]] = None


def _optional_str(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _coerce_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return False
        return float(value) != 0.0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _norm_token(value: object) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def _date_token(value: object) -> str:
    token = _optional_str(value)
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
    settings: FixationConditionDominanceSettings,
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
        interactive = _coerce_bool(is_interactive)
    else:
        interactive = _norm_token(interactive_state) == _norm_token(settings.interactive_label)
    return "face_interactive" if interactive else "face_non_interactive"


def _resolve_bin_duration_s(meta: dict, centers_s: np.ndarray, settings: FixationConditionDominanceSettings) -> float:
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


def _mean_fr_in_window(
    trace: object,
    *,
    centers_s: np.ndarray,
    meta: dict,
    settings: FixationConditionDominanceSettings,
    source_path: Path,
) -> float:
    values = np.asarray(trace, dtype=float).reshape(-1)
    if values.size != centers_s.size:
        raise ValueError(
            "Average PSTH row length does not match bin centers: "
            f"path={source_path}, n_values={values.size}, n_centers={centers_s.size}"
        )
    start_s = float(settings.window_start_ms) / 1000.0
    stop_s = float(settings.window_stop_ms) / 1000.0
    if stop_s <= start_s:
        raise ValueError("Dominance window stop must be greater than start.")
    mask = (centers_s >= start_s) & (centers_s < stop_s)
    if not np.any(mask):
        raise ValueError(
            "Dominance window does not overlap average PSTH bin centers: "
            f"path={source_path}, window_ms=({settings.window_start_ms}, {settings.window_stop_ms})"
        )
    if not _average_values_are_rate(meta):
        values = values / _resolve_bin_duration_s(meta, centers_s, settings)
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    return float(np.mean(selected)) if selected.size > 0 else np.nan


def _load_condition_mean_rows(
    settings: FixationConditionDominanceSettings,
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
        partitions = _extract_average_partitions(obj)
        for partition, avg_df, meta in partitions:
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
                date = _date_token(avg_row.get("date")) or _date_token(row.get("date"))
                unit_uuid = _optional_str(avg_row.get("unit_uuid"))
                if not date or unit_uuid is None:
                    continue
                n_trials = avg_row.get("n_trials", np.nan)
                try:
                    n_trials_float = float(n_trials)
                except Exception:
                    n_trials_float = np.nan
                mean_fr_hz = _mean_fr_in_window(
                    avg_row.get("psth_mean"),
                    centers_s=centers_s,
                    meta=meta,
                    settings=settings,
                    source_path=path,
                )
                records.append(
                    {
                        "unit_key": f"{date}|{unit_uuid}",
                        "date": date,
                        "unit_uuid": unit_uuid,
                        "region": _optional_str(avg_row.get("region")) or "unknown",
                        "spike_channel": _optional_str(avg_row.get("spike_channel")),
                        "recorded_agent": _optional_str(avg_row.get("recorded_agent")),
                        "recorded_monkey": _optional_str(avg_row.get("recorded_monkey")),
                        "area": _optional_str(avg_row.get("area")),
                        "condition": condition,
                        "n_trials": n_trials_float,
                        "mean_fr_hz": mean_fr_hz,
                        "average_partition": partition,
                        "source_average_path": str(path),
                    }
                )
    return pd.DataFrame(records)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(vals)
    vals = vals[valid]
    w = w[valid]
    if vals.size == 0:
        return np.nan
    valid_w = np.isfinite(w) & (w > 0.0)
    if np.any(valid_w):
        return float(np.average(vals[valid_w], weights=w[valid_w]))
    return float(np.mean(vals))


def _build_unit_dominance_table(
    condition_df: pd.DataFrame,
    *,
    settings: FixationConditionDominanceSettings,
) -> pd.DataFrame:
    if condition_df.empty:
        return pd.DataFrame()

    condition_rows: list[dict] = []
    for (unit_key, condition), grp in condition_df.groupby(["unit_key", "condition"], dropna=False):
        row0 = grp.iloc[0]
        n_trials = pd.to_numeric(grp["n_trials"], errors="coerce").to_numpy(dtype=float)
        condition_rows.append(
            {
                "unit_key": str(unit_key),
                "date": str(row0.get("date")),
                "unit_uuid": str(row0.get("unit_uuid")),
                "region": _optional_str(row0.get("region")) or "unknown",
                "spike_channel": _optional_str(row0.get("spike_channel")),
                "recorded_agent": _optional_str(row0.get("recorded_agent")),
                "recorded_monkey": _optional_str(row0.get("recorded_monkey")),
                "area": _optional_str(row0.get("area")),
                "condition": str(condition),
                "mean_fr_hz": _weighted_mean(grp["mean_fr_hz"], grp["n_trials"]),
                "n_trials": float(np.nansum(n_trials)) if np.any(np.isfinite(n_trials)) else np.nan,
            }
        )
    cond_agg = pd.DataFrame(condition_rows)
    rows: list[dict] = []
    for unit_key, grp in cond_agg.groupby("unit_key", dropna=False):
        row0 = grp.iloc[0]
        mean_by_condition = {
            str(row.condition): float(row.mean_fr_hz)
            for row in grp.itertuples(index=False)
            if str(row.condition) in set(settings.condition_order)
        }
        n_by_condition = {
            str(row.condition): float(row.n_trials)
            for row in grp.itertuples(index=False)
            if str(row.condition) in set(settings.condition_order)
        }
        means = np.asarray(
            [mean_by_condition.get(condition, np.nan) for condition in settings.condition_order],
            dtype=float,
        )
        n_trials = np.asarray(
            [n_by_condition.get(condition, np.nan) for condition in settings.condition_order],
            dtype=float,
        )
        all_observed = bool(np.all(np.isfinite(means)))
        meets_min_trials = bool(
            np.all(np.isfinite(n_trials))
            and np.all(n_trials >= float(settings.min_trials_per_condition))
        )
        dominant_condition = ""
        dominance_status = "missing_condition"
        dominant_mean_fr_hz = np.nan
        dominant_margin_hz = np.nan
        tie_conditions = ""
        if all_observed and meets_min_trials:
            max_val = float(np.max(means))
            winners = [
                condition
                for condition, value in zip(settings.condition_order, means)
                if np.isfinite(value)
                and np.isclose(float(value), max_val, atol=float(settings.tie_tolerance_hz), rtol=0.0)
            ]
            if len(winners) == 1:
                dominant_condition = str(winners[0])
                dominance_status = "dominant"
                dominant_mean_fr_hz = max_val
                sorted_vals = np.sort(means[np.isfinite(means)])
                dominant_margin_hz = (
                    float(sorted_vals[-1] - sorted_vals[-2])
                    if sorted_vals.size >= 2
                    else np.nan
                )
            else:
                dominance_status = "tie"
                dominant_mean_fr_hz = max_val
                tie_conditions = "|".join(winners)

        out_row = {
            "unit_key": str(unit_key),
            "date": str(row0.get("date")),
            "unit_uuid": str(row0.get("unit_uuid")),
            "region": _optional_str(row0.get("region")) or "unknown",
            "spike_channel": _optional_str(row0.get("spike_channel")),
            "recorded_agent": _optional_str(row0.get("recorded_agent")),
            "recorded_monkey": _optional_str(row0.get("recorded_monkey")),
            "area": _optional_str(row0.get("area")),
            "window_start_ms": float(settings.window_start_ms),
            "window_stop_ms": float(settings.window_stop_ms),
            "all_conditions_observed": all_observed,
            "meets_min_trials": meets_min_trials,
            "dominance_status": dominance_status,
            "dominant_condition": dominant_condition,
            "tie_conditions": tie_conditions,
            "dominant_mean_fr_hz": dominant_mean_fr_hz,
            "dominant_margin_hz": dominant_margin_hz,
        }
        for condition in settings.condition_order:
            out_row[f"mean_fr_{condition}_hz"] = mean_by_condition.get(condition, np.nan)
            out_row[f"n_trials_{condition}"] = n_by_condition.get(condition, np.nan)
        rows.append(out_row)
    return pd.DataFrame(rows).sort_values(["date", "region", "unit_uuid"]).reset_index(drop=True)


def _load_selectivity_unit_summary(settings: FixationConditionDominanceSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    path = (
        build_analysis_output_dir(cfg, settings.selectivity_input_subdir)
        / ensure_filename(settings.selectivity_unit_summary_filename, ".csv")
    )
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    if "unit_key" not in df.columns:
        if {"date", "unit_uuid"}.issubset(df.columns):
            df["unit_key"] = df["date"].astype(str) + "|" + df["unit_uuid"].astype(str)
        else:
            return pd.DataFrame()
    out = df.copy()
    out["unit_key"] = out["unit_key"].astype(str)
    if "is_selective_unit_raw" not in out.columns:
        out["is_selective_unit_raw"] = out.get("is_selective_unit", False)
    if "is_selective_unit_corrected" not in out.columns:
        out["is_selective_unit_corrected"] = out["is_selective_unit_raw"]
    keep = [
        "unit_key",
        "is_selective_unit",
        "is_selective_unit_raw",
        "is_selective_unit_corrected",
        "selective_pairs",
        "selective_pairs_raw",
        "selective_pairs_corrected",
    ]
    keep = [column for column in keep if column in out.columns]
    out = out.loc[:, keep].copy()
    for column in ("is_selective_unit", "is_selective_unit_raw", "is_selective_unit_corrected"):
        if column in out.columns:
            out[column] = out[column].map(_coerce_bool)
    return out.drop_duplicates(subset=["unit_key"], keep="last")


def _attach_selectivity(unit_df: pd.DataFrame, settings: FixationConditionDominanceSettings) -> pd.DataFrame:
    out = unit_df.copy()
    selectivity_df = _load_selectivity_unit_summary(settings)
    if not selectivity_df.empty and "unit_key" in out.columns:
        out = out.merge(selectivity_df, on="unit_key", how="left")
    for column in ("is_selective_unit", "is_selective_unit_raw", "is_selective_unit_corrected"):
        if column not in out.columns:
            out[column] = False
        out[column] = out[column].map(_coerce_bool)
    for column in ("selective_pairs", "selective_pairs_raw", "selective_pairs_corrected"):
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)
    return out


def _region_order(unit_df: pd.DataFrame, configured: Optional[Sequence[str]]) -> list[str]:
    if configured is not None:
        ordered = [str(region).strip() for region in configured if str(region).strip()]
        if ordered:
            return ordered
    if unit_df.empty or "region" not in unit_df.columns:
        return []
    return sorted(unit_df["region"].dropna().astype(str).unique().tolist())


def _build_region_summary(
    unit_df: pd.DataFrame,
    *,
    settings: FixationConditionDominanceSettings,
) -> pd.DataFrame:
    if unit_df.empty:
        return pd.DataFrame()
    regions = _region_order(unit_df, settings.region_order)
    rows: list[dict] = []
    subset_masks = {
        "all_units": pd.Series(True, index=unit_df.index, dtype=bool),
        "raw_selective_units": unit_df["is_selective_unit_raw"].map(_coerce_bool),
        "corrected_selective_units": unit_df["is_selective_unit_corrected"].map(_coerce_bool),
    }
    for subset_name in settings.unit_subset_order:
        subset_mask = subset_masks.get(str(subset_name))
        if subset_mask is None:
            continue
        subset_df = unit_df.loc[subset_mask].copy()
        for region in regions:
            region_df = subset_df.loc[subset_df["region"].astype(str) == str(region)].copy()
            classified = region_df.loc[region_df["dominance_status"].astype(str) == "dominant"].copy()
            n_subset = int(len(region_df))
            n_classified = int(len(classified))
            n_ties = int((region_df["dominance_status"].astype(str) == "tie").sum()) if n_subset else 0
            n_missing = int((region_df["dominance_status"].astype(str) == "missing_condition").sum()) if n_subset else 0
            for condition in settings.condition_order:
                n_condition = int((classified["dominant_condition"].astype(str) == str(condition)).sum())
                rows.append(
                    {
                        "unit_subset": str(subset_name),
                        "region": str(region),
                        "dominant_condition": str(condition),
                        "n_units": n_condition,
                        "n_units_subset_total": n_subset,
                        "n_units_classified": n_classified,
                        "fraction_of_classified": (
                            float(n_condition) / float(n_classified)
                            if n_classified > 0
                            else np.nan
                        ),
                        "fraction_of_subset": (
                            float(n_condition) / float(n_subset)
                            if n_subset > 0
                            else np.nan
                        ),
                        "n_ties": n_ties,
                        "n_missing_condition": n_missing,
                        "window_start_ms": float(settings.window_start_ms),
                        "window_stop_ms": float(settings.window_stop_ms),
                    }
                )
    return pd.DataFrame(rows)


def run_fixation_condition_dominance_analysis(
    settings: FixationConditionDominanceSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
) -> dict:
    """Run region-level dominance counts from average fixation PSTHs."""
    condition_df = _load_condition_mean_rows(settings, dates=dates)
    unit_df = _build_unit_dominance_table(condition_df, settings=settings)
    unit_df = _attach_selectivity(unit_df, settings)
    if regions is not None and not unit_df.empty:
        allowed = {str(region) for region in regions}
        unit_df = unit_df.loc[unit_df["region"].astype(str).isin(allowed)].copy()
    region_summary_df = _build_region_summary(unit_df, settings=settings)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    unit_path = out_root / ensure_filename(settings.unit_output_filename, ".csv")
    summary_path = out_root / ensure_filename(settings.region_summary_filename, ".csv")
    result_path = out_root / ensure_filename(settings.output_pickle_filename, ".pkl")
    unit_df.to_csv(unit_path, index=False)
    region_summary_df.to_csv(summary_path, index=False)

    result = {
        "meta": {
            "window_start_ms": float(settings.window_start_ms),
            "window_stop_ms": float(settings.window_stop_ms),
            "condition_order": list(settings.condition_order),
            "unit_subset_order": list(settings.unit_subset_order),
            "average_input_subdir": str(settings.average_input_subdir),
            "average_input_filename": ensure_filename(settings.average_input_filename, ".pkl"),
            "selectivity_input_subdir": str(settings.selectivity_input_subdir),
            "selectivity_unit_summary_filename": ensure_filename(
                settings.selectivity_unit_summary_filename,
                ".csv",
            ),
            "n_units": int(len(unit_df)),
        },
        "unit_dominance": unit_df,
        "region_summary": region_summary_df,
    }
    save_pickle_path(result, result_path)
    return result
