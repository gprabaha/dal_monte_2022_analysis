"""Quantify condition-specific variability in average fixation PSTHs."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
    ensure_filename as _ensure_filename,
    resolve_bin_centers_from_meta as _resolve_bin_centers_from_meta,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_date_paths
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path, save_pickle_path
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_CONDITIONS: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
_ALLOWED_PVALUE_CORRECTIONS: frozenset[str] = frozenset(
    {"none", "bonferroni", "holm", "bh", "fdr_bh"}
)
_SUMMARY_META_COLUMNS: tuple[str, ...] = ("date", "unit_uuid", "unit_key", "region")


@dataclass
class FixationPSTHVariabilitySettings:
    """Configuration for per-unit fixation PSTH variability analysis."""

    cfg_path: str
    input_subdir: str = "ephys/psth/fixation_psth_averages"
    input_filename: str = "fixations_psth_10ms.pkl"
    object_input_subdir: Optional[str] = "ephys/psth/fixation_psth_averages"
    object_input_filename: Optional[str] = "fixations_psth_10ms.pkl"
    output_subdir: str = "ephys/psth/fixation_psth_variability"
    unit_summary_filename: str = "unit_condition_variability.csv"
    within_region_stats_filename: str = "within_region_condition_variability_stats.csv"
    output_pickle_filename: Optional[str] = None
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    conditions: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_CONDITIONS))
    variability_metric_name: str = "std_over_time_bins"
    variability_metric_label: str = "SD of Mean FR"
    variability_metric_unit: str = "Hz"
    variability_window_start_ms: Optional[float] = None
    variability_window_stop_ms: Optional[float] = None
    pvalue_correction: str = "fdr_bh"
    alpha: float = 0.05
    min_paired_units_per_region: int = 2
    bin_size_ms_fallback: float = 10.0
    window_pre_s_fallback: float = 1.0
    window_post_s_fallback: float = 1.0
    verbose_logging: bool = True


def _variability_column(condition: str) -> str:
    return f"{str(condition).strip()}_variability"


def _summary_columns(settings: FixationPSTHVariabilitySettings) -> list[str]:
    return [*_SUMMARY_META_COLUMNS, *[_variability_column(condition) for condition in settings.conditions]]


def _empty_unit_summary_df(settings: FixationPSTHVariabilitySettings) -> pd.DataFrame:
    return pd.DataFrame(columns=_summary_columns(settings))


def _empty_within_region_stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "region",
            "n_units_region_total",
            "condition_a",
            "condition_b",
            "condition_pair",
            "test_name",
            "n_units_paired",
            "statistic",
            "p_value",
            "p_value_adjusted",
            "pvalue_correction",
            "alpha",
            "significant_adjusted",
        ]
    )


def _extract_average_df_and_meta(
    obj,
    *,
    partition: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    partition_token: Optional[str]
    if partition is None:
        partition_token = None
    else:
        token = str(partition).strip().lower()
        if not token or token == "auto":
            partition_token = None
        elif token in {"split", "unsplit"}:
            partition_token = token
        else:
            raise ValueError(
                f"Unsupported average partition '{partition}'. Expected one of: split, unsplit."
            )

    if isinstance(obj, dict):
        meta = obj.get("meta", {}) or {}
        meta_dict = meta if isinstance(meta, dict) else {}

        if partition_token is not None:
            partition_key = (
                "averages_split_by_interactive_state"
                if partition_token == "split"
                else "averages_unsplit_by_interactive_state"
            )
            partition_meta_key = "split_meta" if partition_token == "split" else "unsplit_meta"
            partition_df = obj.get(partition_key)
            if isinstance(partition_df, pd.DataFrame):
                merged_meta = dict(meta_dict)
                partition_meta = meta_dict.get(partition_meta_key, {})
                if isinstance(partition_meta, dict):
                    merged_meta.update(partition_meta)
                merged_meta["selected_partition"] = partition_token
                return partition_df, merged_meta

        if "averages" in obj:
            df = obj.get("averages")
            if isinstance(df, pd.DataFrame):
                split_flag = meta_dict.get("split_by_interactive_state")
                if partition_token == "split" and split_flag is False:
                    raise ValueError(
                        "Requested split average partition but file contains unsplit averages."
                    )
                if partition_token == "unsplit" and split_flag is True:
                    raise ValueError(
                        "Requested unsplit average partition but file contains split averages."
                    )
                if partition_token is not None:
                    meta_dict = dict(meta_dict)
                    meta_dict["selected_partition"] = partition_token
                return df, meta_dict
    if isinstance(obj, pd.DataFrame):
        return obj, {}
    return pd.DataFrame(), {}


def _norm_token(value: object) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def _normalize_date_token(value: object) -> Optional[str]:
    token = _as_optional_str(value)
    if token is None:
        return None
    if len(token) == 7 and token.isdigit():
        return token.zfill(8)
    return token


def _normalize_n_trials(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return 1.0
    if not np.isfinite(out) or out <= 0.0:
        return 1.0
    return out


def _fallback_bin_centers(settings: FixationPSTHVariabilitySettings, n_bins: int) -> np.ndarray:
    if int(n_bins) <= 0:
        return np.asarray([], dtype=float)
    bin_size_s = float(settings.bin_size_ms_fallback) / 1000.0
    start_center_s = -float(settings.window_pre_s_fallback) + 0.5 * bin_size_s
    return start_center_s + np.arange(int(n_bins), dtype=float) * bin_size_s


def _resolve_condition_from_average_row(
    row: pd.Series,
    settings: FixationPSTHVariabilitySettings,
    *,
    require_face_interactive_state: bool,
) -> Optional[str]:
    category_token = _norm_token(row.get("fixation_category"))
    if not category_token or category_token == "nan":
        return None

    if category_token in {_norm_token(settings.object_label), "object", "objects"}:
        return "object"
    if category_token in {"face_interactive", "interactive_face", "int_face", "faceinteractive"}:
        return "face_interactive"
    if category_token in {
        "face_non_interactive",
        "face_noninteractive",
        "non_interactive_face",
        "noninteractive_face",
        "nonint_face",
    }:
        return "face_non_interactive"
    if category_token != _norm_token(settings.face_label):
        return None

    interactive_state = row.get("interactive_state")
    is_interactive = row.get("is_interactive")
    has_interactive_state = interactive_state is not None and not pd.isna(interactive_state)
    has_is_interactive = is_interactive is not None and not pd.isna(is_interactive)
    if not has_interactive_state and not has_is_interactive:
        if require_face_interactive_state:
            raise ValueError(
                "Face rows are missing interactive-state labels. "
                "Build averages with split_by_interactive_state=true."
            )
        return "face_non_interactive"

    if has_is_interactive:
        interactive = _as_bool(is_interactive, settings.interactive_label)
    else:
        interactive = _as_bool(interactive_state, settings.interactive_label)
    return "face_interactive" if interactive else "face_non_interactive"


def _aggregate_average_records(
    records: list[dict],
    *,
    settings: FixationPSTHVariabilitySettings,
    n_bins_ref: Optional[int],
    bin_centers_ref: Optional[np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    if not records:
        return pd.DataFrame(), np.asarray([], dtype=float)

    if bin_centers_ref is None:
        if n_bins_ref is None:
            return pd.DataFrame(), np.asarray([], dtype=float)
        bin_centers_ref = _fallback_bin_centers(settings, int(n_bins_ref))

    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        key = (str(record["unit_key"]), str(record["condition"]))
        weighted = np.asarray(record["psth_mean"], dtype=float) * float(record["n_trials"])
        if key not in grouped:
            grouped[key] = {
                "date": str(record["date"]),
                "unit_uuid": str(record["unit_uuid"]),
                "unit_key": str(record["unit_key"]),
                "region": str(record["region"]),
                "condition": str(record["condition"]),
                "weighted_sum": weighted.copy(),
                "weight": float(record["n_trials"]),
            }
            continue
        bucket = grouped[key]
        if weighted.shape != np.asarray(bucket["weighted_sum"]).shape:
            raise ValueError("Encountered inconsistent PSTH lengths during average aggregation.")
        bucket["weighted_sum"] = np.asarray(bucket["weighted_sum"], dtype=float) + weighted
        bucket["weight"] = float(bucket["weight"]) + float(record["n_trials"])

    out_rows: list[dict] = []
    for bucket in grouped.values():
        weight = max(float(bucket["weight"]), 1e-12)
        out_rows.append(
            {
                "date": bucket["date"],
                "unit_uuid": bucket["unit_uuid"],
                "unit_key": bucket["unit_key"],
                "region": bucket["region"],
                "condition": bucket["condition"],
                "n_trials_total": weight,
                "psth_mean": np.asarray(bucket["weighted_sum"], dtype=float) / weight,
            }
        )

    out_df = pd.DataFrame(out_rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["region", "unit_key", "condition"]).reset_index(drop=True)
    return out_df, np.asarray(bin_centers_ref, dtype=float).reshape(-1)


def _load_average_psth_table(
    settings: FixationPSTHVariabilitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
    input_subdir: Optional[str] = None,
    input_filename: Optional[str] = None,
    require_face_interactive_state: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = load_config(settings.cfg_path)
    subdir = str(settings.input_subdir if input_subdir is None else input_subdir).strip()
    filename = str(settings.input_filename if input_filename is None else input_filename).strip()
    if not subdir or not filename:
        return pd.DataFrame(), np.asarray([], dtype=float)

    rows = scan_analysis_date_paths(
        cfg,
        subdir,
        filename=_ensure_filename(filename, ".pkl"),
        dates=dates,
    )
    if settings.verbose_logging:
        print(
            "[analysis] fixation PSTH variability average-input scan: "
            f"subdir={subdir}, filename={filename}, "
            f"require_face_interactive_state={bool(require_face_interactive_state)}, "
            f"date_filter={list(dates) if dates is not None else 'all'}, matched_files={len(rows)}"
        )
    if not rows:
        return pd.DataFrame(), np.asarray([], dtype=float)

    records: list[dict] = []
    bin_centers_ref = None
    n_bins_ref = None
    for row in rows:
        obj = load_pickle_path(row["path"])
        partition = "split" if bool(require_face_interactive_state) else "unsplit"
        avg_df, meta = _extract_average_df_and_meta(obj, partition=partition)
        if avg_df.empty or "psth_mean" not in avg_df.columns:
            continue

        centers = _resolve_bin_centers_from_meta(meta)
        if centers is not None:
            centers = np.asarray(centers, dtype=float).reshape(-1)
            if bin_centers_ref is None:
                bin_centers_ref = centers
            elif centers.shape != bin_centers_ref.shape or not np.allclose(centers, bin_centers_ref):
                raise ValueError(
                    "Mismatched bin centers across average PSTH files; "
                    f"path={row['path']}"
                )

        has_face_interactive_state = (
            ("interactive_state" in avg_df.columns)
            or ("is_interactive" in avg_df.columns)
        )
        for _, avg_row in avg_df.iterrows():
            if bool(require_face_interactive_state) and not has_face_interactive_state:
                category_token = _norm_token(avg_row.get("fixation_category"))
                if category_token == _norm_token(settings.face_label):
                    raise ValueError(
                        "Face rows are missing interactive-state labels. "
                        "Build averages with split_by_interactive_state=true."
                    )
            condition = _resolve_condition_from_average_row(
                avg_row,
                settings,
                require_face_interactive_state=require_face_interactive_state,
            )
            if condition is None:
                continue

            psth_mean = np.asarray(avg_row.get("psth_mean"), dtype=float).reshape(-1)
            if psth_mean.size == 0 or np.any(~np.isfinite(psth_mean)):
                continue
            if n_bins_ref is None:
                n_bins_ref = int(psth_mean.size)
            elif int(psth_mean.size) != int(n_bins_ref):
                raise ValueError(
                    "Mismatched PSTH length across average rows; "
                    f"expected={n_bins_ref}, got={psth_mean.size}"
                )
            if bin_centers_ref is not None and int(psth_mean.size) != int(bin_centers_ref.size):
                raise ValueError(
                    "PSTH length does not match bin centers for average row; "
                    f"path={row['path']}"
                )

            date = (
                _normalize_date_token(avg_row.get("date"))
                or _normalize_date_token(str(row["date"]))
                or str(row["date"])
            )
            unit_uuid = _as_optional_str(avg_row.get("unit_uuid"))
            if unit_uuid is None:
                continue
            records.append(
                {
                    "date": date,
                    "unit_uuid": unit_uuid,
                    "unit_key": f"{date}|{unit_uuid}",
                    "region": _as_optional_str(avg_row.get("region")) or "unknown",
                    "condition": condition,
                    "n_trials": _normalize_n_trials(avg_row.get("n_trials", 1.0)),
                    "psth_mean": psth_mean,
                }
            )

    return _aggregate_average_records(
        records,
        settings=settings,
        n_bins_ref=n_bins_ref,
        bin_centers_ref=bin_centers_ref,
    )


def _load_combined_average_psth_table(
    settings: FixationPSTHVariabilitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    split_df, split_centers = _load_average_psth_table(
        settings,
        dates=dates,
        input_subdir=settings.input_subdir,
        input_filename=settings.input_filename,
        require_face_interactive_state=True,
    )

    has_object_override = (
        settings.object_input_subdir is not None
        or settings.object_input_filename is not None
    )
    if not has_object_override:
        return split_df, split_centers

    object_subdir = (
        str(settings.object_input_subdir).strip()
        if settings.object_input_subdir is not None
        else str(settings.input_subdir).strip()
    )
    object_filename = (
        str(settings.object_input_filename).strip()
        if settings.object_input_filename is not None
        else str(settings.input_filename).strip()
    )
    object_df, object_centers = _load_average_psth_table(
        settings,
        dates=dates,
        input_subdir=object_subdir,
        input_filename=object_filename,
        require_face_interactive_state=False,
    )
    if object_df.empty:
        return split_df, split_centers
    if split_df.empty:
        return object_df.loc[object_df["condition"].astype(str) == "object"].copy(), object_centers

    if object_centers.size > 0 and (
        split_centers.size != object_centers.size or not np.allclose(split_centers, object_centers)
    ):
        raise ValueError("Split and object-average PSTH inputs have mismatched bin centers.")

    object_df = object_df.loc[object_df["condition"].astype(str) == "object"].copy()
    if object_df.empty:
        return split_df, split_centers

    object_dates = {
        str(token).strip()
        for token in object_df["date"].astype(str).tolist()
    }
    split_object_mask = split_df["condition"].astype(str).map(lambda token: token == "object")
    split_date_mask = split_df["date"].astype(str).map(lambda token: token.strip() in object_dates)
    kept = split_df.loc[~(split_object_mask & split_date_mask)].copy()
    out_df = pd.concat([kept, object_df], axis=0, ignore_index=True)
    out_df = out_df.sort_values(["region", "unit_key", "condition"]).reset_index(drop=True)
    return out_df, split_centers


def _resolve_variability_window_mask(
    settings: FixationPSTHVariabilitySettings,
    bin_centers_s: np.ndarray,
) -> np.ndarray:
    centers = np.asarray(bin_centers_s, dtype=float).reshape(-1)
    if centers.size == 0:
        return np.asarray([], dtype=bool)

    start_ms = settings.variability_window_start_ms
    stop_ms = settings.variability_window_stop_ms
    if start_ms is None and stop_ms is None:
        return np.ones(centers.shape, dtype=bool)
    if start_ms is None or stop_ms is None:
        raise ValueError(
            "variability_window_start_ms and variability_window_stop_ms must both be provided or both be null."
        )
    start_s = float(min(start_ms, stop_ms)) / 1000.0
    stop_s = float(max(start_ms, stop_ms)) / 1000.0
    mask = (centers >= start_s) & (centers <= stop_s)
    if not np.any(mask):
        raise ValueError(
            "Requested variability window contains no bins: "
            f"window=[{float(start_ms)}, {float(stop_ms)}] ms."
        )
    return mask


def _normalize_pvalue_correction(method: object) -> str:
    token = str(method).strip().lower()
    aliases = {
        "fdr": "fdr_bh",
        "benjamini_hochberg": "fdr_bh",
        "benjamini-hochberg": "fdr_bh",
        "bh": "fdr_bh",
    }
    resolved = aliases.get(token, token)
    if resolved not in _ALLOWED_PVALUE_CORRECTIONS:
        raise ValueError(
            f"Unsupported p-value correction '{method}'. "
            f"Expected one of: {sorted(_ALLOWED_PVALUE_CORRECTIONS)}"
        )
    return resolved


def _adjust_pvalues(p_values: Sequence[float], method: str) -> np.ndarray:
    resolved = _normalize_pvalue_correction(method)
    vec = np.asarray(p_values, dtype=float).reshape(-1)
    out = np.full(vec.shape, np.nan, dtype=float)
    finite = np.isfinite(vec)
    if not np.any(finite):
        return out
    vals = vec[finite]
    m = int(vals.size)
    if resolved == "none":
        out[finite] = vals
        return out
    if resolved == "bonferroni":
        out[finite] = np.minimum(vals * float(m), 1.0)
        return out

    order = np.argsort(vals)
    ranked = vals[order]
    if resolved == "holm":
        holm_ranked = (m - np.arange(m, dtype=float)) * ranked
        holm_ranked = np.maximum.accumulate(holm_ranked)
        holm_ranked = np.clip(holm_ranked, 0.0, 1.0)
        adjusted = np.empty(m, dtype=float)
        adjusted[order] = holm_ranked
        out[finite] = adjusted
        return out

    bh_ranked = ranked * (float(m) / np.arange(1.0, float(m) + 1.0))
    bh_ranked = np.minimum.accumulate(bh_ranked[::-1])[::-1]
    bh_ranked = np.clip(bh_ranked, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = bh_ranked
    out[finite] = adjusted
    return out


def _apply_adjusted_pvalues(
    df: pd.DataFrame,
    *,
    p_col: str,
    out_col: str,
    method: str,
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    if out.empty or p_col not in out.columns:
        return out
    if group_cols is None or len(group_cols) == 0:
        out[out_col] = _adjust_pvalues(out[p_col].to_numpy(dtype=float), method)
        return out
    for _, idx in out.groupby(list(group_cols), dropna=False).groups.items():
        out.loc[idx, out_col] = _adjust_pvalues(
            out.loc[idx, p_col].to_numpy(dtype=float),
            method,
        )
    return out


def _safe_ttest_rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, int(mask.sum())
    stat, p_value = ttest_rel(x, y, nan_policy="omit")
    return float(stat), float(p_value), int(x.size)


def _build_unit_variability_summary(
    avg_df: pd.DataFrame,
    *,
    settings: FixationPSTHVariabilitySettings,
    bin_centers_s: np.ndarray,
) -> pd.DataFrame:
    out_columns = _summary_columns(settings)
    if avg_df.empty:
        return _empty_unit_summary_df(settings)

    mask = _resolve_variability_window_mask(settings, bin_centers_s)
    rows: list[dict] = []
    for row in avg_df.itertuples(index=False):
        psth_mean = np.asarray(getattr(row, "psth_mean"), dtype=float).reshape(-1)
        if psth_mean.size != int(np.asarray(bin_centers_s).size):
            raise ValueError(
                "Average PSTH length does not match resolved bin centers for unit variability analysis."
            )
        selected = psth_mean[mask]
        finite = selected[np.isfinite(selected)]
        if finite.size == 0:
            continue
        variability = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        rows.append(
            {
                "date": str(getattr(row, "date", "")),
                "unit_uuid": str(getattr(row, "unit_uuid", "")),
                "unit_key": str(getattr(row, "unit_key", "")),
                "region": str(getattr(row, "region", "unknown")),
                "condition": str(getattr(row, "condition", "")),
                "fr_variability": variability,
            }
        )

    if not rows:
        return _empty_unit_summary_df(settings)

    long_df = pd.DataFrame(rows)
    metadata_df = (
        long_df.loc[:, list(_SUMMARY_META_COLUMNS)]
        .drop_duplicates(subset=["unit_key"], keep="first")
        .set_index("unit_key")
    )
    wide_df = long_df.pivot_table(
        index="unit_key",
        columns="condition",
        values="fr_variability",
        aggfunc="mean",
    )
    wide_df = wide_df.rename(columns={condition: _variability_column(condition) for condition in settings.conditions})
    out_df = metadata_df.join(wide_df, how="left").reset_index()
    for column in out_columns:
        if column not in out_df.columns:
            out_df[column] = np.nan
    out_df = out_df.loc[:, out_columns]
    out_df = out_df.sort_values(["region", "unit_key"]).reset_index(drop=True)
    return out_df


def _build_within_region_stats(
    summary_df: pd.DataFrame,
    *,
    settings: FixationPSTHVariabilitySettings,
) -> pd.DataFrame:
    if summary_df.empty:
        return _empty_within_region_stats_df()

    rows: list[dict] = []
    condition_order = [str(condition) for condition in settings.conditions if str(condition).strip()]
    condition_columns = {condition: _variability_column(condition) for condition in condition_order}
    for region, grp in summary_df.groupby("region", dropna=False, sort=False):
        n_units_region = int(grp["unit_key"].astype(str).nunique())
        available_conditions = [
            condition
            for condition in condition_order
            if condition_columns[condition] in grp.columns
            and np.isfinite(pd.to_numeric(grp[condition_columns[condition]], errors="coerce").to_numpy(dtype=float)).any()
        ]
        if len(available_conditions) < 2:
            continue
        for condition_a, condition_b in combinations(available_conditions, 2):
            arr_a = pd.to_numeric(grp[condition_columns[condition_a]], errors="coerce").to_numpy(dtype=float)
            arr_b = pd.to_numeric(grp[condition_columns[condition_b]], errors="coerce").to_numpy(dtype=float)
            stat, p_value, n_paired = _safe_ttest_rel(arr_a, arr_b)
            if n_paired < int(max(settings.min_paired_units_per_region, 2)):
                continue
            rows.append(
                {
                    "region": str(region),
                    "n_units_region_total": int(n_units_region),
                    "condition_a": str(condition_a),
                    "condition_b": str(condition_b),
                    "condition_pair": f"{condition_a}__vs__{condition_b}",
                    "test_name": "paired_ttest",
                    "n_units_paired": int(n_paired),
                    "statistic": stat,
                    "p_value": p_value,
                }
            )

    if not rows:
        return _empty_within_region_stats_df()

    out_df = pd.DataFrame(rows)
    correction = _normalize_pvalue_correction(settings.pvalue_correction)
    out_df = _apply_adjusted_pvalues(
        out_df,
        p_col="p_value",
        out_col="p_value_adjusted",
        method=correction,
        group_cols=("region",),
    )
    out_df["pvalue_correction"] = str(correction)
    out_df["alpha"] = float(settings.alpha)
    out_df["significant_adjusted"] = (
        pd.to_numeric(out_df["p_value_adjusted"], errors="coerce").to_numpy(dtype=float)
        < float(settings.alpha)
    )
    out_df = out_df.sort_values(["region", "condition_a", "condition_b"]).reset_index(drop=True)
    return out_df.loc[:, list(_empty_within_region_stats_df().columns)]


def run_fixation_psth_variability_analysis(
    settings: FixationPSTHVariabilitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """Compute per-unit condition-specific variability from average PSTHs."""

    avg_df, bin_centers_s = _load_combined_average_psth_table(settings, dates=dates)
    unit_summary_df = _build_unit_variability_summary(avg_df, settings=settings, bin_centers_s=bin_centers_s)
    within_region_stats_df = _build_within_region_stats(unit_summary_df, settings=settings)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    unit_summary_path = out_root / _ensure_filename(settings.unit_summary_filename, ".csv")
    within_region_stats_path = out_root / _ensure_filename(settings.within_region_stats_filename, ".csv")

    unit_summary_df.to_csv(unit_summary_path, index=False)
    within_region_stats_df.to_csv(within_region_stats_path, index=False)

    correction = _normalize_pvalue_correction(settings.pvalue_correction)
    result = {
        "unit_summary": unit_summary_df,
        "within_region_stats": within_region_stats_df,
        "meta": {
            "conditions": [str(condition) for condition in settings.conditions],
            "variability_columns": {
                str(condition): _variability_column(str(condition))
                for condition in settings.conditions
            },
            "variability_metric_name": str(settings.variability_metric_name),
            "variability_metric_label": str(settings.variability_metric_label),
            "variability_metric_unit": str(settings.variability_metric_unit),
            "variability_window_start_ms": (
                None if settings.variability_window_start_ms is None else float(settings.variability_window_start_ms)
            ),
            "variability_window_stop_ms": (
                None if settings.variability_window_stop_ms is None else float(settings.variability_window_stop_ms)
            ),
            "alpha": float(settings.alpha),
            "pvalue_correction": str(correction),
            "n_regions": int(unit_summary_df["region"].astype(str).nunique()) if not unit_summary_df.empty else 0,
            "n_units": int(unit_summary_df["unit_key"].astype(str).nunique()) if not unit_summary_df.empty else 0,
        },
        "unit_summary_path": str(unit_summary_path),
        "within_region_stats_path": str(within_region_stats_path),
        "pickle_path": None,
    }

    if settings.output_pickle_filename is not None and str(settings.output_pickle_filename).strip():
        pickle_path = out_root / _ensure_filename(str(settings.output_pickle_filename), ".pkl")
        save_pickle_path(result, pickle_path)
        result["pickle_path"] = str(pickle_path)

    if settings.verbose_logging:
        variability_counts = {
            str(condition): int(
                np.isfinite(
                    pd.to_numeric(
                        unit_summary_df.get(_variability_column(str(condition)), pd.Series(dtype=float)),
                        errors="coerce",
                    ).to_numpy(dtype=float)
                ).sum()
            )
            for condition in settings.conditions
        }
        print(
            "[analysis] fixation PSTH variability summary: "
            f"units={result['meta']['n_units']}, regions={result['meta']['n_regions']}, "
            f"condition_counts={variability_counts}, stats_rows={len(within_region_stats_df)}"
        )
        print(f"[analysis] wrote unit summary: {unit_summary_path}")
        print(f"[analysis] wrote within-region stats: {within_region_stats_path}")
        if result["pickle_path"] is not None:
            print(f"[analysis] wrote results pickle: {result['pickle_path']}")

    return result
