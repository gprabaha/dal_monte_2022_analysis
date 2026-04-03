"""Population PCA analysis for fixation-conditioned average PSTHs."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
    ensure_filename as _ensure_filename,
    extract_trials_df_and_meta as _extract_trials_df_and_meta,
    resolve_bin_centers_from_meta as _resolve_bin_centers_from_meta,
)
from dal_monte_2022_analysis.core.stats import (
    apply_adjusted_pvalues,
    normalize_pvalue_correction,
    safe_paired_ttest,
)
from dal_monte_2022_analysis.data.transforms.annotate import load_pair_context_table
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_date_paths
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
    scan_processed_paths_for_filename,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


DEFAULT_POPULATION_PCA_CONDITIONS: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
DEFAULT_POPULATION_PCA_ANALYSIS_VARIANT = "three_condition_core"


@dataclass
class FixationPopulationPCASettings:
    """Configuration for population PCA from fixation PSTH averages."""

    cfg_path: str
    analysis_variant: str = DEFAULT_POPULATION_PCA_ANALYSIS_VARIANT
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations.pkl"
    prefer_trial_input: bool = False
    allow_trial_fallback: bool = True
    input_subdir: str = "ephys/psth/fixation_psth_averages"
    input_filename: str = "fixations.pkl"
    object_input_subdir: Optional[str] = None
    object_input_filename: Optional[str] = None
    output_subdir: str = "ephys/psth/fixation_population_pca"
    summary_filename: str = "pca_fit_summary.csv"
    timecourse_filename: str = "concatenated_pc_timecourses.csv"
    explained_variance_filename: str = "cross_condition_explained_variance.csv"
    unit_inventory_filename: str = "region_unit_inventory.csv"
    pairwise_geometry_timecourse_filename: str = "pairwise_geometry_timecourses.csv"
    pairwise_geometry_summary_filename: str = "pairwise_geometry_summary.csv"
    pairwise_geometry_within_region_stats_filename: str = (
        "pairwise_geometry_within_region_stats.csv"
    )
    pairwise_geometry_cross_region_stats_filename: str = (
        "pairwise_geometry_cross_region_stats.csv"
    )
    output_pickle_filename: str = "results.pkl"
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    conditions: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_POPULATION_PCA_CONDITIONS),
    )
    window_start_ms: float = -500.0
    window_stop_ms: float = 500.0
    max_components: Optional[int] = 50
    min_units_per_region: int = 3
    require_all_conditions: bool = True
    require_face_interactive_state: bool = True
    geometry_n_pcs: Optional[int] = 20
    geometry_angle_unit: str = "degrees"
    geometry_alpha: float = 0.05
    geometry_pvalue_correction: str = "fdr_bh"
    smooth_before_average: bool = True
    smoothing_sigma_ms: float = 20.0
    verbose_logging: bool = True
    use_parallel: bool = True
    max_procs: int = 16
    bin_size_ms_fallback: float = 10.0
    window_pre_s_fallback: float = 1.0
    window_post_s_fallback: float = 1.0


def _normalize_condition_sequence(
    raw: object,
    *,
    fallback: Sequence[str],
) -> tuple[str, ...]:
    if raw is None:
        return tuple(str(token) for token in fallback)
    if isinstance(raw, str):
        seq = [raw]
    elif isinstance(raw, (list, tuple)):
        seq = list(raw)
    else:
        seq = [raw]
    out: list[str] = []
    for item in seq:
        token = str(item).strip()
        if token:
            out.append(token)
    return tuple(out) if out else tuple(str(token) for token in fallback)


def _normalize_str_dict(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        key_token = str(key).strip()
        value_token = str(value).strip()
        if key_token and value_token:
            out[key_token] = value_token
    return out


def _normalize_variant_cfg(raw: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, object]] = {}
    for key, value in raw.items():
        token = str(key).strip()
        if token and isinstance(value, dict):
            out[token] = dict(value)
    return out


def resolve_population_pca_variant_config(
    cfg: dict,
    *,
    analysis_variant: Optional[str] = None,
) -> dict[str, object]:
    """Resolve variant-specific PCA analysis and plotting settings."""
    variant = str(
        analysis_variant
        if analysis_variant is not None
        else cfg.get("population_pca_analysis_variant", DEFAULT_POPULATION_PCA_ANALYSIS_VARIANT)
    ).strip() or DEFAULT_POPULATION_PCA_ANALYSIS_VARIANT

    configured_variants = _normalize_variant_cfg(cfg.get("population_pca_analysis_variants"))
    variant_cfg = configured_variants.get(variant, {})
    if variant != DEFAULT_POPULATION_PCA_ANALYSIS_VARIANT and not variant_cfg:
        raise ValueError(
            f"Unknown population PCA analysis variant '{variant}'. "
            "Add it to population_pca_analysis_variants in the config."
        )

    legacy_base: dict[str, object] = {}
    if variant == DEFAULT_POPULATION_PCA_ANALYSIS_VARIANT:
        legacy_base = {
            "conditions": cfg.get("population_pca_conditions"),
            "input_subdir": cfg.get("population_pca_input_subdir"),
            "input_filename": cfg.get(
                "population_pca_input_filename_split",
                cfg.get("population_pca_input_filename"),
            ),
            "object_input_subdir": cfg.get("population_pca_object_input_subdir"),
            "object_input_filename": cfg.get("population_pca_object_input_filename"),
            "output_subdir": cfg.get("population_pca_output_subdir"),
            "plot_input_subdir": cfg.get("population_pca_plot_input_subdir"),
            "plot_output_subdir": cfg.get("population_pca_plot_output_subdir"),
            "plot_trajectory_output_filename": cfg.get("population_pca_plot_trajectory_output_filename"),
            "plot_explained_variance_output_filename": cfg.get(
                "population_pca_plot_explained_variance_output_filename"
            ),
            "plot_cumulative_variance_output_filename": cfg.get(
                "population_pca_plot_cumulative_variance_output_filename"
            ),
            "plot_pairwise_geometry_output_filename": cfg.get(
                "population_pca_plot_pairwise_geometry_output_filename"
            ),
            "plot_condition_labels": cfg.get("population_pca_plot_condition_labels"),
            "plot_condition_colors": cfg.get("population_pca_plot_condition_colors"),
        }

    resolved: dict[str, object] = {}
    for key, value in legacy_base.items():
        if value is not None:
            resolved[key] = value
    for key, value in variant_cfg.items():
        resolved[key] = value

    fallback_conditions = (
        DEFAULT_POPULATION_PCA_CONDITIONS
        if variant == DEFAULT_POPULATION_PCA_ANALYSIS_VARIANT
        else ()
    )
    resolved["analysis_variant"] = variant
    resolved["conditions"] = _normalize_condition_sequence(
        resolved.get("conditions"),
        fallback=fallback_conditions,
    )
    if variant != DEFAULT_POPULATION_PCA_ANALYSIS_VARIANT and not resolved["conditions"]:
        raise ValueError(
            f"population_pca_analysis_variants.{variant}.conditions is required for non-default variants."
        )
    resolved["plot_condition_labels"] = _normalize_str_dict(resolved.get("plot_condition_labels"))
    resolved["plot_condition_colors"] = _normalize_str_dict(resolved.get("plot_condition_colors"))
    return resolved


def _fallback_bin_centers_s(settings: FixationPopulationPCASettings, n_bins: int) -> np.ndarray:
    if int(n_bins) <= 0:
        return np.asarray([], dtype=float)
    bin_size_s = float(settings.bin_size_ms_fallback) / 1000.0
    start_center_s = -float(settings.window_pre_s_fallback) + 0.5 * bin_size_s
    return start_center_s + np.arange(int(n_bins), dtype=float) * bin_size_s


def _extract_average_df_and_meta(obj) -> tuple[pd.DataFrame, dict]:
    if isinstance(obj, dict) and "averages" in obj:
        df = obj.get("averages")
        meta = obj.get("meta", {}) or {}
        return (df if isinstance(df, pd.DataFrame) else pd.DataFrame(), meta if isinstance(meta, dict) else {})
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


def _build_recorded_monkey_lookup(
    settings: FixationPopulationPCASettings,
) -> dict[str, dict[str, Optional[str]]]:
    try:
        pair_df = load_pair_context_table(cfg_path=settings.cfg_path)
    except Exception:
        return {}
    if pair_df.empty or "date" not in pair_df.columns:
        return {}

    out: dict[str, dict[str, Optional[str]]] = {}
    for row in pair_df.itertuples(index=False):
        date_token = _normalize_date_token(getattr(row, "date", None))
        if date_token is None:
            continue
        out[date_token] = {
            "m1": _as_optional_str(getattr(row, "m1_name", None)),
            "m2": _as_optional_str(getattr(row, "m2_name", None)),
        }
    return out


def _resolve_recorded_monkey_name(
    *,
    date: object,
    recorded_agent: object,
    recorded_monkey: object,
    monkey_lookup: dict[str, dict[str, Optional[str]]],
) -> Optional[str]:
    direct = _as_optional_str(recorded_monkey)
    if direct is not None:
        return direct
    if not monkey_lookup:
        return None

    date_token = _normalize_date_token(date)
    if date_token is None:
        return None
    day_names = monkey_lookup.get(date_token)
    if not day_names:
        return None

    agent = _norm_token(recorded_agent if recorded_agent is not None else "m1")
    if agent not in {"m1", "m2"}:
        agent = "m1"
    resolved = _as_optional_str(day_names.get(agent))
    if resolved is not None:
        return resolved
    return _as_optional_str(day_names.get("m1"))


def _resolve_condition_from_row_fields(
    *,
    fixation_category: object,
    interactive_state: object,
    is_interactive: object,
    settings: FixationPopulationPCASettings,
    require_face_interactive_state: Optional[bool] = None,
) -> Optional[str]:
    category_token = _norm_token(fixation_category)
    if not category_token or category_token == "nan":
        return None

    requested_conditions = {str(token).strip() for token in settings.conditions if str(token).strip()}

    object_tokens = {
        _norm_token(settings.object_label),
        "object",
        "objects",
    }
    split_object_requested = bool(
        {"object_interactive", "object_non_interactive"}.intersection(requested_conditions)
    )
    pooled_object_requested = "object" in requested_conditions

    has_interactive_state = interactive_state is not None and not pd.isna(interactive_state)
    has_is_interactive = is_interactive is not None and not pd.isna(is_interactive)

    def _resolve_interactive_flag() -> bool:
        if has_is_interactive:
            return _as_bool(is_interactive, settings.interactive_label)
        if has_interactive_state:
            return _as_bool(interactive_state, settings.interactive_label)
        raise ValueError("Interactive-state label is required for split PCA conditions.")

    if category_token in {
        "object_interactive",
        "interactive_object",
        "int_object",
        "objectinteractive",
    }:
        return "object_interactive" if split_object_requested else ("object" if pooled_object_requested else None)
    if category_token in {
        "object_non_interactive",
        "object_noninteractive",
        "non_interactive_object",
        "noninteractive_object",
        "nonint_object",
    }:
        return "object_non_interactive" if split_object_requested else ("object" if pooled_object_requested else None)
    if category_token in object_tokens:
        if split_object_requested:
            if not has_interactive_state and not has_is_interactive:
                raise ValueError(
                    "Object rows are missing interactive-state labels. "
                    "Build averages with split_by_interactive_state=true "
                    "or provide is_interactive/interactive_state columns."
                )
            return "object_interactive" if _resolve_interactive_flag() else "object_non_interactive"
        if pooled_object_requested or not requested_conditions:
            return "object"
        return None

    if category_token in {
        "face_interactive",
        "interactive_face",
        "int_face",
        "faceinteractive",
    }:
        return "face_interactive"
    if category_token in {
        "face_non_interactive",
        "face_noninteractive",
        "non_interactive_face",
        "noninteractive_face",
        "nonint_face",
    }:
        return "face_non_interactive"

    if category_token == "face":
        if "face" in requested_conditions and "face_interactive" not in requested_conditions and "face_non_interactive" not in requested_conditions:
            return "face"

    face_token = _norm_token(settings.face_label)
    if category_token != face_token:
        return None

    require_face_state = (
        bool(settings.require_face_interactive_state)
        if require_face_interactive_state is None
        else bool(require_face_interactive_state)
    )
    if not has_interactive_state and not has_is_interactive:
        if require_face_state:
            raise ValueError(
                "Face rows are missing interactive-state labels. "
                "Build averages with split_by_interactive_state=true "
                "or provide is_interactive/interactive_state columns."
            )
        if "face" in requested_conditions and "face_interactive" not in requested_conditions and "face_non_interactive" not in requested_conditions:
            return "face"
        return "face_non_interactive"

    interactive = _resolve_interactive_flag()
    if "face_interactive" in requested_conditions or "face_non_interactive" in requested_conditions:
        return "face_interactive" if interactive else "face_non_interactive"
    if "face" in requested_conditions:
        return "face"
    return None


def _resolve_condition_for_average_row(
    row: pd.Series,
    settings: FixationPopulationPCASettings,
    *,
    require_face_interactive_state: Optional[bool] = None,
) -> Optional[str]:
    return _resolve_condition_from_row_fields(
        fixation_category=row.get("fixation_category"),
        interactive_state=row.get("interactive_state"),
        is_interactive=row.get("is_interactive"),
        settings=settings,
        require_face_interactive_state=require_face_interactive_state,
    )


def _resolve_condition_for_trial_row(
    row,
    settings: FixationPopulationPCASettings,
    *,
    require_face_interactive_state: Optional[bool] = None,
) -> Optional[str]:
    return _resolve_condition_from_row_fields(
        fixation_category=getattr(row, "fixation_category", None),
        interactive_state=getattr(row, "interactive_state", None),
        is_interactive=getattr(row, "is_interactive", None),
        settings=settings,
        require_face_interactive_state=require_face_interactive_state,
    )


def _normalize_n_trials(value: object) -> float:
    try:
        n_trials = float(value)
    except Exception:
        n_trials = 1.0
    if not np.isfinite(n_trials) or n_trials <= 0:
        return 1.0
    return n_trials


def _resolve_trial_smoothing_sigma_bins(
    settings: FixationPopulationPCASettings,
    *,
    bin_centers_s: Optional[np.ndarray],
) -> Optional[float]:
    if not settings.smooth_before_average:
        return None
    if float(settings.smoothing_sigma_ms) <= 0:
        raise ValueError("population_pca_smoothing_sigma_ms must be > 0 when smoothing is enabled.")

    bin_size_ms = float(settings.bin_size_ms_fallback)
    if bin_centers_s is not None:
        centers = np.asarray(bin_centers_s, dtype=float).reshape(-1)
        if centers.size > 1:
            diffs = np.diff(centers)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size > 0:
                bin_size_ms = float(np.mean(diffs)) * 1000.0
    if not np.isfinite(bin_size_ms) or bin_size_ms <= 0:
        raise ValueError("Unable to infer positive trial bin size for smoothing.")
    return float(settings.smoothing_sigma_ms) / float(bin_size_ms)


def _aggregate_psth_records(
    records: list[dict],
    *,
    settings: FixationPopulationPCASettings,
    n_bins_ref: Optional[int],
    bin_centers_ref: Optional[np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    if not records:
        return pd.DataFrame(), np.asarray([], dtype=float)

    if bin_centers_ref is None:
        if n_bins_ref is None:
            return pd.DataFrame(), np.asarray([], dtype=float)
        bin_centers_ref = _fallback_bin_centers_s(settings, int(n_bins_ref))

    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in records:
        key = (str(record["region"]), str(record["unit_key"]), str(record["condition"]))
        weighted = np.asarray(record["psth_mean"], dtype=float) * float(record["n_trials"])
        if key not in grouped:
            grouped[key] = {
                "region": str(record["region"]),
                "unit_key": str(record["unit_key"]),
                "condition": str(record["condition"]),
                "date": str(record["date"]),
                "unit_uuid": str(record["unit_uuid"]),
                "spike_channel": record["spike_channel"],
                "recorded_agent": record["recorded_agent"],
                "recorded_monkey": record["recorded_monkey"],
                "area": record["area"],
                "weighted_sum": weighted.copy(),
                "weight": float(record["n_trials"]),
            }
            continue

        bucket = grouped[key]
        for meta_key in ("spike_channel", "recorded_agent", "recorded_monkey", "area"):
            if bucket.get(meta_key) is None and record.get(meta_key) is not None:
                bucket[meta_key] = record.get(meta_key)
        if weighted.shape != np.asarray(bucket["weighted_sum"]).shape:
            raise ValueError("Encountered inconsistent PSTH lengths during average aggregation.")
        bucket["weighted_sum"] = np.asarray(bucket["weighted_sum"], dtype=float) + weighted
        bucket["weight"] = float(bucket["weight"]) + float(record["n_trials"])

    out_rows: list[dict] = []
    for bucket in grouped.values():
        weight = max(float(bucket["weight"]), 1e-12)
        mean_vec = np.asarray(bucket["weighted_sum"], dtype=float) / weight
        out_rows.append(
            {
                "date": bucket["date"],
                "unit_uuid": bucket["unit_uuid"],
                "unit_key": bucket["unit_key"],
                "region": bucket["region"],
                "spike_channel": bucket["spike_channel"],
                "recorded_agent": bucket["recorded_agent"],
                "recorded_monkey": bucket["recorded_monkey"],
                "area": bucket["area"],
                "condition": bucket["condition"],
                "n_trials_total": weight,
                "psth_mean": mean_vec,
            }
        )

    out_df = pd.DataFrame(out_rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["region", "unit_key", "condition"]).reset_index(drop=True)
    return out_df, np.asarray(bin_centers_ref, dtype=float).reshape(-1)


def _load_average_psth_table(
    settings: FixationPopulationPCASettings,
    *,
    dates: Optional[Sequence[str]] = None,
    input_subdir: Optional[str] = None,
    input_filename: Optional[str] = None,
    require_face_interactive_state: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = load_config(settings.cfg_path)
    subdir = str(settings.input_subdir if input_subdir is None else input_subdir).strip()
    if not subdir:
        if settings.verbose_logging:
            print("[analysis] population PCA average-input scan skipped: empty subdir")
        return pd.DataFrame(), np.asarray([], dtype=float)
    filename = str(settings.input_filename if input_filename is None else input_filename).strip()
    if not filename:
        if settings.verbose_logging:
            print("[analysis] population PCA average-input scan skipped: empty filename")
        return pd.DataFrame(), np.asarray([], dtype=float)
    filename_norm = _ensure_filename(filename, ".pkl")
    rows = scan_analysis_date_paths(
        cfg,
        subdir,
        filename=filename_norm,
        dates=dates,
    )
    if settings.verbose_logging:
        print(
            "[analysis] population PCA average-input scan: "
            f"subdir={subdir}, "
            f"require_face_interactive_state={bool(require_face_interactive_state)}, "
            f"date_filter={list(dates) if dates is not None else 'all'}, "
            f"matched_files={len(rows)}"
        )
    if not rows:
        return pd.DataFrame(), np.asarray([], dtype=float)
    monkey_lookup = _build_recorded_monkey_lookup(settings)

    records: list[dict] = []
    bin_centers_ref = None
    n_bins_ref = None

    for row in rows:
        obj = load_pickle_path(row["path"])
        avg_df, meta = _extract_average_df_and_meta(obj)
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
            condition = _resolve_condition_for_average_row(
                avg_row,
                settings,
                require_face_interactive_state=require_face_interactive_state,
            )
            if condition is None:
                continue

            psth_mean = np.asarray(avg_row.get("psth_mean"), dtype=float).reshape(-1)
            if psth_mean.size == 0:
                continue
            if np.any(~np.isfinite(psth_mean)):
                continue

            if n_bins_ref is None:
                n_bins_ref = int(psth_mean.size)
            elif int(psth_mean.size) != int(n_bins_ref):
                raise ValueError(
                    "Mismatched PSTH length across average rows; "
                    f"expected {n_bins_ref}, got {psth_mean.size}"
                )

            if bin_centers_ref is not None and int(psth_mean.size) != int(bin_centers_ref.size):
                raise ValueError(
                    "PSTH length does not match bin centers for average row; "
                    f"path={row['path']}"
                )

            date = (
                _normalize_date_token(_as_optional_str(avg_row.get("date")))
                or _normalize_date_token(str(row["date"]))
                or str(row["date"])
            )
            unit_uuid = _as_optional_str(avg_row.get("unit_uuid"))
            if unit_uuid is None:
                continue
            unit_key = f"{date}|{unit_uuid}"
            region = _as_optional_str(avg_row.get("region")) or "unknown"
            recorded_agent = _as_optional_str(avg_row.get("recorded_agent")) or "m1"
            recorded_monkey = _resolve_recorded_monkey_name(
                date=date,
                recorded_agent=recorded_agent,
                recorded_monkey=_as_optional_str(avg_row.get("recorded_monkey")),
                monkey_lookup=monkey_lookup,
            )

            records.append(
                {
                    "date": date,
                    "unit_uuid": unit_uuid,
                    "unit_key": unit_key,
                    "region": region,
                    "spike_channel": _as_optional_str(avg_row.get("spike_channel")),
                    "recorded_agent": recorded_agent,
                    "recorded_monkey": recorded_monkey,
                    "area": _as_optional_str(avg_row.get("area")),
                    "condition": condition,
                    "n_trials": _normalize_n_trials(avg_row.get("n_trials", 1.0)),
                    "psth_mean": psth_mean,
                }
            )

    out_df, out_centers = _aggregate_psth_records(
        records,
        settings=settings,
        n_bins_ref=n_bins_ref,
        bin_centers_ref=bin_centers_ref,
    )
    if settings.verbose_logging:
        condition_counts = (
            out_df["condition"].astype(str).value_counts().to_dict()
            if not out_df.empty and "condition" in out_df.columns
            else {}
        )
        n_regions = (
            int(out_df["region"].astype(str).nunique())
            if not out_df.empty and "region" in out_df.columns
            else 0
        )
        print(
            "[analysis] population PCA average-input rows prepared: "
            f"rows={len(out_df)}, regions={n_regions}, "
            f"n_bins={int(np.asarray(out_centers).size)}, "
            f"condition_counts={condition_counts}"
        )
    return out_df, out_centers


def _load_combined_average_psth_table(
    settings: FixationPopulationPCASettings,
    *,
    dates: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if settings.verbose_logging:
        print(
            "[analysis] population PCA average-input combine: "
            f"split_subdir={settings.input_subdir}, "
            f"object_subdir={settings.object_input_subdir}"
        )
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
        if settings.verbose_logging:
            print(
                "[analysis] population PCA average-input combine: "
                "no unsplit object override configured; using split input only"
            )
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
    if settings.verbose_logging:
        print(
            "[analysis] population PCA average-input combine: "
            f"split_rows={len(split_df)}, object_rows={len(object_df)}, "
            f"split_bins={int(np.asarray(split_centers).size)}, object_bins={int(np.asarray(object_centers).size)}"
        )
    if object_df.empty:
        return split_df, split_centers

    if split_df.empty:
        split_centers = np.asarray(object_centers, dtype=float)
    elif object_centers.size > 0:
        if split_centers.size != object_centers.size or not np.allclose(split_centers, object_centers):
            raise ValueError("Split and object-average PSTH inputs have mismatched bin centers.")

    object_df = object_df.loc[object_df["condition"].astype(str) == "object"].copy()
    if object_df.empty:
        return split_df, split_centers

    if split_df.empty:
        return object_df, split_centers

    object_dates = {
        str(token).strip()
        for token in object_df["date"].astype(str).tolist()
    }
    split_object_mask = split_df["condition"].astype(str).map(lambda token: token == "object")
    split_date_mask = split_df["date"].astype(str).map(lambda token: token.strip() in object_dates)
    kept = split_df.loc[~(split_object_mask & split_date_mask)].copy()
    out_df = pd.concat([kept, object_df], axis=0, ignore_index=True)
    out_df = out_df.sort_values(["region", "unit_key", "condition"]).reset_index(drop=True)
    if settings.verbose_logging:
        condition_counts = (
            out_df["condition"].astype(str).value_counts().to_dict()
            if not out_df.empty
            else {}
        )
        print(
            "[analysis] population PCA average-input combine complete: "
            f"rows={len(out_df)}, condition_counts={condition_counts}, "
            f"kept_split_rows={len(kept)}, object_override_rows={len(object_df)}"
        )
    return out_df, split_centers


def _load_trial_averaged_psth_table(
    settings: FixationPopulationPCASettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = load_config(settings.cfg_path)
    trial_filename = _ensure_filename(settings.trial_input_filename, ".pkl")
    rows = scan_processed_paths_for_filename(
        cfg,
        settings.trial_input_modality,
        filename=trial_filename,
        dates=dates,
        sessions=sessions,
        agents=(None,),
    )
    if settings.verbose_logging:
        print(
            "[analysis] population PCA trial-input scan: "
            f"modality={settings.trial_input_modality}, "
            f"date_filter={list(dates) if dates is not None else 'all'}, "
            f"session_filter={list(sessions) if sessions is not None else 'all'}, "
            f"matched_files={len(rows)}"
        )
    if not rows:
        return pd.DataFrame(), np.asarray([], dtype=float)
    monkey_lookup = _build_recorded_monkey_lookup(settings)

    records: list[dict] = []
    bin_centers_ref = None
    n_bins_ref = None
    for row in rows:
        obj = load_pickle_path(row["path"])
        trial_df, meta = _extract_trials_df_and_meta(obj)
        if trial_df.empty or "psth_counts" not in trial_df.columns:
            continue

        centers = _resolve_bin_centers_from_meta(meta)
        if centers is not None:
            centers = np.asarray(centers, dtype=float).reshape(-1)
            if bin_centers_ref is None:
                bin_centers_ref = centers
            elif centers.shape != bin_centers_ref.shape or not np.allclose(centers, bin_centers_ref):
                raise ValueError(
                    "Mismatched bin centers across trial PSTH files; "
                    f"path={row['path']}"
                )
        sigma_bins = _resolve_trial_smoothing_sigma_bins(
            settings,
            bin_centers_s=centers if centers is not None else bin_centers_ref,
        )

        df = trial_df.copy()
        if "date" not in df.columns:
            df["date"] = str(row["date"])
        if "session" not in df.columns:
            df["session"] = str(row["session"])

        for trial_row in df.itertuples(index=False):
            condition = _resolve_condition_for_trial_row(trial_row, settings)
            if condition is None:
                continue

            psth_counts = np.asarray(getattr(trial_row, "psth_counts"), dtype=float).reshape(-1)
            if psth_counts.size == 0:
                continue
            if np.any(~np.isfinite(psth_counts)):
                continue
            if sigma_bins is not None:
                psth_counts = gaussian_filter1d(psth_counts, sigma=sigma_bins, mode="nearest")

            if n_bins_ref is None:
                n_bins_ref = int(psth_counts.size)
            elif int(psth_counts.size) != int(n_bins_ref):
                raise ValueError(
                    "Mismatched trial PSTH length across rows; "
                    f"expected={n_bins_ref}, got={psth_counts.size}"
                )
            if bin_centers_ref is not None and int(psth_counts.size) != int(bin_centers_ref.size):
                raise ValueError(
                    "Trial PSTH length does not match bin centers for row; "
                    f"path={row['path']}"
                )

            date = (
                _normalize_date_token(_as_optional_str(getattr(trial_row, "date", None)))
                or _normalize_date_token(str(row["date"]))
                or str(row["date"])
            )
            unit_uuid = _as_optional_str(getattr(trial_row, "unit_uuid", None))
            if unit_uuid is None:
                continue
            unit_key = f"{date}|{unit_uuid}"
            region = _as_optional_str(getattr(trial_row, "region", None)) or "unknown"
            recorded_agent = _as_optional_str(getattr(trial_row, "recorded_agent", None)) or "m1"
            recorded_monkey = _resolve_recorded_monkey_name(
                date=date,
                recorded_agent=recorded_agent,
                recorded_monkey=_as_optional_str(getattr(trial_row, "recorded_monkey", None)),
                monkey_lookup=monkey_lookup,
            )

            records.append(
                {
                    "date": date,
                    "unit_uuid": unit_uuid,
                    "unit_key": unit_key,
                    "region": region,
                    "spike_channel": _as_optional_str(getattr(trial_row, "spike_channel", None)),
                    "recorded_agent": recorded_agent,
                    "recorded_monkey": recorded_monkey,
                    "area": _as_optional_str(getattr(trial_row, "area", None)),
                    "condition": condition,
                    "n_trials": 1.0,
                    "psth_mean": psth_counts,
                }
            )

    out_df, out_centers = _aggregate_psth_records(
        records,
        settings=settings,
        n_bins_ref=n_bins_ref,
        bin_centers_ref=bin_centers_ref,
    )
    if settings.verbose_logging:
        condition_counts = (
            out_df["condition"].astype(str).value_counts().to_dict()
            if not out_df.empty and "condition" in out_df.columns
            else {}
        )
        n_regions = (
            int(out_df["region"].astype(str).nunique())
            if not out_df.empty and "region" in out_df.columns
            else 0
        )
        print(
            "[analysis] population PCA trial-input rows prepared: "
            f"rows={len(out_df)}, regions={n_regions}, "
            f"n_bins={int(np.asarray(out_centers).size)}, "
            f"condition_counts={condition_counts}"
        )
    return out_df, out_centers


def _load_input_psth_table(
    settings: FixationPopulationPCASettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Load PCA input rows from average PSTHs with optional trial fallback."""
    average_source = f"average:{str(settings.input_subdir).strip()}"
    if settings.verbose_logging:
        print(
            "[analysis] population PCA input selection: "
            f"prefer_trial_input={bool(settings.prefer_trial_input)}, "
            f"allow_trial_fallback={bool(settings.allow_trial_fallback)}, "
            f"average_source={average_source}, "
            f"trial_source={settings.trial_input_modality}"
        )

    if settings.prefer_trial_input:
        trial_df, trial_centers = _load_trial_averaged_psth_table(
            settings,
            dates=dates,
            sessions=sessions,
        )
        if not trial_df.empty:
            if settings.verbose_logging:
                print(
                    "[analysis] population PCA input selected: "
                    f"source=trial, rows={len(trial_df)}, bins={int(np.asarray(trial_centers).size)}"
                )
            return trial_df, trial_centers, "trial"
        print("[analysis] no usable trial PSTH rows found for population PCA; trying average inputs")
        avg_df, avg_centers = _load_combined_average_psth_table(settings, dates=dates)
        if not avg_df.empty:
            if settings.verbose_logging:
                print(
                    "[analysis] population PCA input selected: "
                    f"source={average_source}, rows={len(avg_df)}, bins={int(np.asarray(avg_centers).size)}"
                )
            return avg_df, avg_centers, average_source
        return pd.DataFrame(), np.asarray([], dtype=float), "none"

    avg_df, avg_centers = _load_combined_average_psth_table(settings, dates=dates)
    if not avg_df.empty:
        if settings.verbose_logging:
            print(
                "[analysis] population PCA input selected: "
                f"source={average_source}, rows={len(avg_df)}, bins={int(np.asarray(avg_centers).size)}"
            )
        return avg_df, avg_centers, average_source

    if settings.allow_trial_fallback:
        trial_df, trial_centers = _load_trial_averaged_psth_table(
            settings,
            dates=dates,
            sessions=sessions,
        )
        if not trial_df.empty:
            if settings.verbose_logging:
                print(
                    "[analysis] population PCA input selected: "
                    f"source=trial (fallback), rows={len(trial_df)}, bins={int(np.asarray(trial_centers).size)}"
                )
            return trial_df, trial_centers, "trial"

    return pd.DataFrame(), np.asarray([], dtype=float), "none"


def _window_mask_from_centers(
    bin_centers_s: np.ndarray,
    *,
    window_start_ms: float,
    window_stop_ms: float,
) -> np.ndarray:
    centers = np.asarray(bin_centers_s, dtype=float).reshape(-1)
    start_s = float(min(window_start_ms, window_stop_ms)) / 1000.0
    stop_s = float(max(window_start_ms, window_stop_ms)) / 1000.0
    mask = (centers >= start_s) & (centers <= stop_s)
    if not np.any(mask):
        raise ValueError(
            "Requested PCA time window contains no bins: "
            f"window=[{window_start_ms}, {window_stop_ms}] ms."
        )
    return mask


def _condition_pair_token(condition_a: str, condition_b: str) -> str:
    return f"{str(condition_a)}__vs__{str(condition_b)}"


def _resolve_angle_unit(settings: FixationPopulationPCASettings) -> str:
    token = str(settings.geometry_angle_unit).strip().lower()
    aliases = {
        "deg": "degrees",
        "degree": "degrees",
        "degrees": "degrees",
        "rad": "radians",
        "radian": "radians",
        "radians": "radians",
    }
    resolved = aliases.get(token, token)
    if resolved not in {"degrees", "radians"}:
        raise ValueError(
            f"Unsupported geometry angle unit '{settings.geometry_angle_unit}'. "
            "Expected 'degrees' or 'radians'."
        )
    return resolved


def _resolve_geometry_n_pcs_requested(
    settings: FixationPopulationPCASettings,
) -> Optional[int]:
    if settings.geometry_n_pcs is not None:
        return max(1, int(settings.geometry_n_pcs))
    if settings.max_components is not None:
        return max(1, int(settings.max_components))
    return None


def _euclidean_distance_by_time(
    scores_a_pc_by_time: np.ndarray,
    scores_b_pc_by_time: np.ndarray,
    *,
    n_pcs: int,
) -> np.ndarray:
    a = np.asarray(scores_a_pc_by_time, dtype=float)
    b = np.asarray(scores_b_pc_by_time, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        return np.asarray([], dtype=float)
    n_use = min(int(max(1, n_pcs)), int(a.shape[0]), int(b.shape[0]))
    if n_use <= 0 or a.shape[1] != b.shape[1]:
        return np.asarray([], dtype=float)
    diff = a[:n_use, :] - b[:n_use, :]
    return np.sqrt(np.sum(diff * diff, axis=0))


def _angle_between_traces_by_time(
    scores_a_pc_by_time: np.ndarray,
    scores_b_pc_by_time: np.ndarray,
    *,
    n_pcs: int,
    angle_unit: str,
) -> np.ndarray:
    a = np.asarray(scores_a_pc_by_time, dtype=float)
    b = np.asarray(scores_b_pc_by_time, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        return np.asarray([], dtype=float)
    n_use = min(int(max(1, n_pcs)), int(a.shape[0]), int(b.shape[0]))
    if n_use <= 0 or a.shape[1] != b.shape[1]:
        return np.asarray([], dtype=float)
    x = a[:n_use, :].T
    y = b[:n_use, :].T
    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    out = np.full((x.shape[0],), np.nan, dtype=float)
    valid = np.isfinite(norms) & (norms > 1e-12)
    if not np.any(valid):
        return out
    cos_theta = np.sum(x[valid] * y[valid], axis=1) / norms[valid]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    if str(angle_unit) == "degrees":
        angles = np.degrees(angles)
    out[valid] = angles
    return out


def _build_region_pairwise_geometry_rows(
    *,
    region: str,
    condition_scores_pc_by_time: dict[str, np.ndarray],
    bin_centers_s_window: np.ndarray,
    settings: FixationPopulationPCASettings,
) -> tuple[list[dict], list[dict]]:
    angle_unit = _resolve_angle_unit(settings)
    n_bins = int(np.asarray(bin_centers_s_window, dtype=float).size)
    metrics = (
        ("euclidean_distance", "Euclidean Distance", "a.u."),
        (
            "angle_degrees" if angle_unit == "degrees" else "angle_radians",
            f"Angle ({'deg' if angle_unit == 'degrees' else 'rad'})",
            "deg" if angle_unit == "degrees" else "rad",
        ),
    )
    time_rows: list[dict] = []
    summary_rows: list[dict] = []
    requested_n_pcs = _resolve_geometry_n_pcs_requested(settings)

    for condition_a, condition_b in combinations(settings.conditions, 2):
        scores_a = np.asarray(
            condition_scores_pc_by_time.get(str(condition_a), np.asarray([], dtype=float)),
            dtype=float,
        )
        scores_b = np.asarray(
            condition_scores_pc_by_time.get(str(condition_b), np.asarray([], dtype=float)),
            dtype=float,
        )
        if scores_a.ndim != 2 or scores_b.ndim != 2:
            continue
        if scores_a.shape[1] != n_bins or scores_b.shape[1] != n_bins:
            continue
        if requested_n_pcs is None:
            n_pcs_used = min(int(scores_a.shape[0]), int(scores_b.shape[0]))
        else:
            n_pcs_used = min(
                int(requested_n_pcs),
                int(scores_a.shape[0]),
                int(scores_b.shape[0]),
            )
        if n_pcs_used <= 0:
            continue
        pair_token = _condition_pair_token(str(condition_a), str(condition_b))
        values_by_metric = {
            "euclidean_distance": _euclidean_distance_by_time(
                scores_a,
                scores_b,
                n_pcs=n_pcs_used,
            ),
            metrics[1][0]: _angle_between_traces_by_time(
                scores_a,
                scores_b,
                n_pcs=n_pcs_used,
                angle_unit=angle_unit,
            ),
        }
        for metric_name, metric_label, metric_unit in metrics:
            values = np.asarray(values_by_metric.get(metric_name, np.asarray([], dtype=float)), dtype=float)
            if values.size != n_bins:
                continue
            for bin_idx, bin_center_s in enumerate(np.asarray(bin_centers_s_window, dtype=float)):
                time_rows.append(
                    {
                        "region": str(region),
                        "condition_a": str(condition_a),
                        "condition_b": str(condition_b),
                        "condition_pair": str(pair_token),
                        "metric_name": str(metric_name),
                        "metric_label": str(metric_label),
                        "metric_unit": str(metric_unit),
                        "n_pcs_used": int(n_pcs_used),
                        "bin_index": int(bin_idx),
                        "bin_center_s": float(bin_center_s),
                        "value": float(values[bin_idx]) if np.isfinite(values[bin_idx]) else np.nan,
                    }
                )
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            summary_rows.append(
                {
                    "region": str(region),
                    "condition_a": str(condition_a),
                    "condition_b": str(condition_b),
                    "condition_pair": str(pair_token),
                    "metric_name": str(metric_name),
                    "metric_label": str(metric_label),
                    "metric_unit": str(metric_unit),
                    "n_pcs_used": int(n_pcs_used),
                    "n_time_bins_total": int(values.size),
                    "n_time_bins_valid": int(finite.size),
                    "mean_value": float(np.mean(finite)),
                    "median_value": float(np.median(finite)),
                    "std_value": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
                    "min_value": float(np.min(finite)),
                    "max_value": float(np.max(finite)),
                }
            )
    return time_rows, summary_rows


def _build_pairwise_geometry_stat_tables(
    pairwise_geometry_timecourse_df: pd.DataFrame,
    *,
    settings: FixationPopulationPCASettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pairwise_geometry_timecourse_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    required = {
        "region",
        "condition_pair",
        "metric_name",
        "metric_label",
        "metric_unit",
        "bin_index",
        "value",
    }
    if not required.issubset(pairwise_geometry_timecourse_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    df = pairwise_geometry_timecourse_df.copy()
    df["region"] = df["region"].astype(str)
    df["condition_pair"] = df["condition_pair"].astype(str)
    df["metric_name"] = df["metric_name"].astype(str)
    df["metric_label"] = df["metric_label"].astype(str)
    df["metric_unit"] = df["metric_unit"].astype(str)
    df["bin_index"] = pd.to_numeric(df["bin_index"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.loc[df["bin_index"].notna()].copy()

    within_rows: list[dict] = []
    for (metric_name, metric_label, metric_unit, region), grp in df.groupby(
        ["metric_name", "metric_label", "metric_unit", "region"],
        dropna=False,
        sort=False,
    ):
        pair_tokens = [str(token) for token in grp["condition_pair"].dropna().astype(str).unique().tolist()]
        pivot = grp.pivot_table(
            index="bin_index",
            columns="condition_pair",
            values="value",
            aggfunc="mean",
        )
        for pair_a, pair_b in combinations(pair_tokens, 2):
            if pair_a not in pivot.columns or pair_b not in pivot.columns:
                continue
            arr_a = pivot[pair_a].to_numpy(dtype=float)
            arr_b = pivot[pair_b].to_numpy(dtype=float)
            stat, p_value, n_paired = safe_paired_ttest(arr_a, arr_b)
            if n_paired < 2:
                continue
            valid_mask = np.isfinite(arr_a) & np.isfinite(arr_b)
            vals_a = arr_a[valid_mask]
            vals_b = arr_b[valid_mask]
            within_rows.append(
                {
                    "metric_name": str(metric_name),
                    "metric_label": str(metric_label),
                    "metric_unit": str(metric_unit),
                    "region": str(region),
                    "condition_pair_a": str(pair_a),
                    "condition_pair_b": str(pair_b),
                    "test_name": "paired_ttest",
                    "n_time_bins_paired": int(n_paired),
                    "mean_a": float(np.mean(vals_a)),
                    "mean_b": float(np.mean(vals_b)),
                    "median_a": float(np.median(vals_a)),
                    "median_b": float(np.median(vals_b)),
                    "delta_mean_a_minus_b": float(np.mean(vals_a) - np.mean(vals_b)),
                    "delta_median_a_minus_b": float(np.median(vals_a) - np.median(vals_b)),
                    "statistic": stat,
                    "p_value": p_value,
                }
            )
    within_df = pd.DataFrame(within_rows)
    if not within_df.empty:
        correction = normalize_pvalue_correction(settings.geometry_pvalue_correction)
        within_df = apply_adjusted_pvalues(
            within_df,
            p_col="p_value",
            out_col="p_value_adjusted",
            method=correction,
            group_cols=("metric_name", "region"),
        )
        within_df["pvalue_correction"] = str(correction)
        within_df["alpha"] = float(settings.geometry_alpha)
        within_df["significant_adjusted"] = (
            pd.to_numeric(within_df["p_value_adjusted"], errors="coerce").to_numpy(dtype=float)
            < float(settings.geometry_alpha)
        )
        within_df = within_df.sort_values(
            ["metric_name", "region", "condition_pair_a", "condition_pair_b"]
        ).reset_index(drop=True)

    cross_rows: list[dict] = []
    for (metric_name, metric_label, metric_unit, condition_pair), grp in df.groupby(
        ["metric_name", "metric_label", "metric_unit", "condition_pair"],
        dropna=False,
        sort=False,
    ):
        region_tokens = [str(token) for token in grp["region"].dropna().astype(str).unique().tolist()]
        pivot = grp.pivot_table(
            index="bin_index",
            columns="region",
            values="value",
            aggfunc="mean",
        )
        for region_a, region_b in combinations(region_tokens, 2):
            if region_a not in pivot.columns or region_b not in pivot.columns:
                continue
            arr_a = pivot[region_a].to_numpy(dtype=float)
            arr_b = pivot[region_b].to_numpy(dtype=float)
            stat, p_value, n_paired = safe_paired_ttest(arr_a, arr_b)
            if n_paired < 2:
                continue
            valid_mask = np.isfinite(arr_a) & np.isfinite(arr_b)
            vals_a = arr_a[valid_mask]
            vals_b = arr_b[valid_mask]
            cross_rows.append(
                {
                    "metric_name": str(metric_name),
                    "metric_label": str(metric_label),
                    "metric_unit": str(metric_unit),
                    "condition_pair": str(condition_pair),
                    "region_a": str(region_a),
                    "region_b": str(region_b),
                    "test_name": "paired_ttest",
                    "n_time_bins_paired": int(n_paired),
                    "mean_a": float(np.mean(vals_a)),
                    "mean_b": float(np.mean(vals_b)),
                    "median_a": float(np.median(vals_a)),
                    "median_b": float(np.median(vals_b)),
                    "delta_mean_a_minus_b": float(np.mean(vals_a) - np.mean(vals_b)),
                    "delta_median_a_minus_b": float(np.median(vals_a) - np.median(vals_b)),
                    "statistic": stat,
                    "p_value": p_value,
                }
            )
    cross_df = pd.DataFrame(cross_rows)
    if not cross_df.empty:
        correction = normalize_pvalue_correction(settings.geometry_pvalue_correction)
        cross_df = apply_adjusted_pvalues(
            cross_df,
            p_col="p_value",
            out_col="p_value_adjusted",
            method=correction,
            group_cols=("metric_name", "condition_pair"),
        )
        cross_df["pvalue_correction"] = str(correction)
        cross_df["alpha"] = float(settings.geometry_alpha)
        cross_df["significant_adjusted"] = (
            pd.to_numeric(cross_df["p_value_adjusted"], errors="coerce").to_numpy(dtype=float)
            < float(settings.geometry_alpha)
        )
        cross_df = cross_df.sort_values(
            ["metric_name", "condition_pair", "region_a", "region_b"]
        ).reset_index(drop=True)
    return within_df, cross_df


def _fit_pca_units_by_time(
    matrix_units_by_time: np.ndarray,
    *,
    max_components: Optional[int],
) -> dict:
    matrix = np.asarray(matrix_units_by_time, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("PCA input must have shape (n_units, n_time_bins) with non-zero dimensions.")

    X = matrix.T  # samples=time bins, features=units
    n_samples, n_features = X.shape
    max_rank = min(n_samples, n_features)
    if max_components is None:
        n_components = max_rank
    else:
        n_components = min(max_rank, max(1, int(max_components)))

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    U, singular_values_all, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components, :]
    scores_fit_time_by_pc = X_centered @ components.T
    scores_fit_pc_by_time = scores_fit_time_by_pc.T
    singular_values = singular_values_all[:n_components]

    if n_samples > 1:
        explained_variance = (singular_values**2) / float(n_samples - 1)
        total_variance = float(np.sum((singular_values_all**2) / float(n_samples - 1)))
    else:
        explained_variance = np.zeros((n_components,), dtype=float)
        total_variance = 0.0

    if total_variance > 0.0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros((n_components,), dtype=float)
    cumulative_ratio = np.cumsum(explained_variance_ratio)

    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_components": int(n_components),
        "mean": mean.reshape(-1),
        "components": components,
        # Primary orientation keeps consistency with input matrices:
        # input = units x time, projected = PCs x time.
        "scores_fit": scores_fit_pc_by_time,
        "scores_fit_pc_by_time": scores_fit_pc_by_time,
        # Legacy orientation retained for compatibility with older outputs.
        "scores_fit_time_by_pc": scores_fit_time_by_pc,
        "singular_values": singular_values,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance_ratio": cumulative_ratio,
    }


def _project_units_by_time_with_model(
    matrix_units_by_time: np.ndarray,
    model: dict,
) -> np.ndarray:
    matrix = np.asarray(matrix_units_by_time, dtype=float)
    X = matrix.T
    mean = np.asarray(model["mean"], dtype=float).reshape(1, -1)
    components = np.asarray(model["components"], dtype=float)
    if X.shape[1] != mean.shape[1] or X.shape[1] != components.shape[1]:
        raise ValueError(
            "Projection feature dimension mismatch; "
            f"input_features={X.shape[1]}, model_features={components.shape[1]}."
        )
    # Return projected trajectories as PCs x time to mirror units x time input shape.
    return ((X - mean) @ components.T).T


def _reconstruct_samples_with_model(
    X_samples_by_features: np.ndarray,
    model: dict,
    *,
    n_components: int,
) -> np.ndarray:
    X = np.asarray(X_samples_by_features, dtype=float)
    mean = np.asarray(model["mean"], dtype=float).reshape(1, -1)
    components = np.asarray(model["components"], dtype=float)
    k = min(max(1, int(n_components)), int(components.shape[0]))
    basis = components[:k, :]
    scores = (X - mean) @ basis.T
    return (scores @ basis) + mean


def _explained_variance_profile_for_eval(
    matrix_units_by_time_eval: np.ndarray,
    model: dict,
) -> dict[str, np.ndarray | float]:
    matrix = np.asarray(matrix_units_by_time_eval, dtype=float)
    X = matrix.T  # samples=time bins, features=units
    n_components = int(model["n_components"])
    if n_components <= 0:
        empty = np.asarray([], dtype=float)
        return {
            "per_pc_variance": empty,
            "cumulative_variance": empty,
            "total_projected_variance": float("nan"),
            "per_pc_fraction": empty,
            "cumulative_fraction": empty,
        }

    mean = np.asarray(model["mean"], dtype=float).reshape(1, -1)
    components = np.asarray(model["components"], dtype=float)
    if X.shape[1] != mean.shape[1] or X.shape[1] != components.shape[1]:
        raise ValueError(
            "Explained-variance evaluation feature mismatch; "
            f"eval_features={X.shape[1]}, model_features={components.shape[1]}."
        )

    X_centered = X - mean
    scores = X_centered @ components.T  # time bins x PCs
    if X.shape[0] > 1:
        per_pc_var = np.var(scores, axis=0, ddof=1)
    else:
        per_pc_var = np.zeros((n_components,), dtype=float)

    per_pc_var = np.asarray(per_pc_var, dtype=float).reshape(-1)
    finite = np.isfinite(per_pc_var)
    if not np.any(finite):
        values = np.full((n_components,), np.nan, dtype=float)
        return {
            "per_pc_variance": values,
            "cumulative_variance": values.copy(),
            "total_projected_variance": float("nan"),
            "per_pc_fraction": values.copy(),
            "cumulative_fraction": values.copy(),
        }

    per_pc_var_nonneg = np.full((n_components,), np.nan, dtype=float)
    per_pc_var_nonneg[finite] = np.maximum(per_pc_var[finite], 0.0)

    cumulative_variance = np.full((n_components,), np.nan, dtype=float)
    running_variance = 0.0
    for idx, value in enumerate(per_pc_var_nonneg):
        if np.isfinite(value):
            running_variance += float(value)
            cumulative_variance[idx] = running_variance

    total_projected_var = float(np.sum(per_pc_var_nonneg[np.isfinite(per_pc_var_nonneg)]))
    if total_projected_var <= 0.0:
        values = np.full((n_components,), np.nan, dtype=float)
        return {
            "per_pc_variance": per_pc_var_nonneg,
            "cumulative_variance": cumulative_variance,
            "total_projected_variance": float(total_projected_var),
            "per_pc_fraction": values,
            "cumulative_fraction": values.copy(),
        }

    per_pc_fraction = np.full((n_components,), np.nan, dtype=float)
    per_pc_fraction[finite] = per_pc_var_nonneg[finite] / total_projected_var
    per_pc_fraction = np.clip(per_pc_fraction, 0.0, 1.0)

    cumulative_fraction = np.full((n_components,), np.nan, dtype=float)
    running = 0.0
    for idx, frac in enumerate(per_pc_fraction):
        if np.isfinite(frac):
            running += float(frac)
            cumulative_fraction[idx] = running

    return {
        "per_pc_variance": per_pc_var_nonneg,
        "cumulative_variance": cumulative_variance,
        "total_projected_variance": float(total_projected_var),
        "per_pc_fraction": per_pc_fraction,
        "cumulative_fraction": cumulative_fraction,
    }


def _build_fit_summary_row(
    *,
    region: str,
    fit_scope: str,
    fit_condition: str,
    model: dict,
    n_units_common: int,
    n_time_bins_window: int,
    condition_unit_counts: dict[str, int],
) -> dict:
    ratio = np.asarray(model["explained_variance_ratio"], dtype=float).reshape(-1)
    cum = np.asarray(model["cumulative_explained_variance_ratio"], dtype=float).reshape(-1)
    row = {
        "region": str(region),
        "fit_scope": str(fit_scope),
        "fit_condition": str(fit_condition),
        "n_units_common": int(n_units_common),
        "n_time_bins_window": int(n_time_bins_window),
        "n_samples_fit": int(model["n_samples"]),
        "n_components": int(model["n_components"]),
        "explained_variance_ratio_pc1": float(ratio[0]) if ratio.size >= 1 else np.nan,
        "explained_variance_ratio_pc2": float(ratio[1]) if ratio.size >= 2 else np.nan,
        "explained_variance_ratio_pc3": float(ratio[2]) if ratio.size >= 3 else np.nan,
        "cumulative_explained_variance_ratio_pc3": float(cum[2]) if cum.size >= 3 else np.nan,
    }
    for condition, count in condition_unit_counts.items():
        token = _norm_token(condition)
        if token:
            row[f"n_units_{token}"] = int(count)
    return row


def _build_region_analysis(args) -> dict:
    region, region_df, bin_centers_s_window, settings = args
    if settings.verbose_logging:
        print(
            "\n[analysis] ===== region PCA start =====\n"
            "[analysis] region PCA start: "
            f"region={region}, input_rows={len(region_df)}, "
            f"fixation_types={list(settings.conditions)}"
        )

    condition_maps: dict[str, dict[str, np.ndarray]] = {
        condition: {} for condition in settings.conditions
    }
    unit_meta: dict[str, dict[str, object]] = {}

    for row in region_df.to_dict(orient="records"):
        condition = str(row.get("condition"))
        if condition not in condition_maps:
            continue
        unit_key = str(row.get("unit_key"))
        vec = np.asarray(row.get("psth_window"), dtype=float).reshape(-1)
        if vec.size == 0 or np.any(~np.isfinite(vec)):
            continue
        condition_maps[condition][unit_key] = vec
        unit_meta.setdefault(
            unit_key,
            {
                "region": str(region),
                "date": _as_optional_str(row.get("date")),
                "unit_uuid": _as_optional_str(row.get("unit_uuid")),
                "spike_channel": _as_optional_str(row.get("spike_channel")),
                "recorded_agent": _as_optional_str(row.get("recorded_agent")),
                "recorded_monkey": _as_optional_str(row.get("recorded_monkey")),
                "area": _as_optional_str(row.get("area")),
            },
        )

    condition_unit_counts = {
        condition: int(len(unit_map)) for condition, unit_map in condition_maps.items()
    }
    if settings.verbose_logging:
        print(
            "[analysis] region condition inventory: "
            f"region={region}, unit_counts={condition_unit_counts}"
        )
    if not condition_maps:
        if settings.verbose_logging:
            print(
                "[analysis] region PCA skipped: "
                f"region={region}, reason=no_condition_maps"
            )
        return {
            "region": str(region),
            "skipped_reason": "no_condition_maps",
            "condition_unit_counts": condition_unit_counts,
            "summary_rows": [],
            "timecourse_rows": [],
            "explained_rows": [],
            "geometry_time_rows": [],
            "geometry_summary_rows": [],
            "unit_rows": [],
            "payload": None,
        }

    # Cross-condition projections require a shared neuron basis across all
    # conditions, so all downstream fits are computed on the intersection.
    unit_sets = [set(unit_map.keys()) for unit_map in condition_maps.values()]
    common_unit_keys = sorted(set.intersection(*unit_sets)) if unit_sets else []
    if settings.verbose_logging:
        print(
            "[analysis] region common-unit intersection: "
            f"region={region}, n_common_units={len(common_unit_keys)}, "
            f"min_units_required={int(settings.min_units_per_region)}"
        )

    if int(len(common_unit_keys)) < int(settings.min_units_per_region):
        if settings.verbose_logging:
            print(
                "[analysis] region PCA skipped: "
                f"region={region}, reason=insufficient_units, "
                f"n_common_units={len(common_unit_keys)}"
            )
        return {
            "region": str(region),
            "skipped_reason": "insufficient_units",
            "condition_unit_counts": condition_unit_counts,
            "summary_rows": [],
            "timecourse_rows": [],
            "explained_rows": [],
            "geometry_time_rows": [],
            "geometry_summary_rows": [],
            "unit_rows": [],
            "payload": None,
        }

    condition_matrices: dict[str, np.ndarray] = {}
    n_bins_window = int(np.asarray(bin_centers_s_window, dtype=float).size)
    for condition in settings.conditions:
        unit_map = condition_maps.get(condition, {})
        mat = np.vstack([np.asarray(unit_map[unit_key], dtype=float) for unit_key in common_unit_keys])
        if mat.shape[1] != n_bins_window:
            raise ValueError(
                f"Unexpected windowed PSTH length for region={region}, condition={condition}. "
                f"expected={n_bins_window}, got={mat.shape[1]}"
            )
        condition_matrices[condition] = mat
        if settings.verbose_logging:
            print(
                "[analysis] region PCA matrix (individual fixation type): "
                f"region={region}, fixation_type={condition}, "
                f"shape_units_by_time={mat.shape}"
            )

    per_condition_fits: dict[str, dict] = {}
    fit_summary_rows: list[dict] = []
    for condition in settings.conditions:
        fit = _fit_pca_units_by_time(
            condition_matrices[condition],
            max_components=settings.max_components,
        )
        fit["unit_keys"] = np.asarray(common_unit_keys, dtype=object)
        fit["fit_condition"] = str(condition)
        per_condition_fits[condition] = fit
        if settings.verbose_logging:
            print(
                "[analysis] region PCA fit complete (individual fixation type): "
                f"region={region}, fixation_type={condition}, "
                f"fit_input_shape_units_by_time={condition_matrices[condition].shape}, "
                f"components_shape_pc_by_unit={np.asarray(fit['components']).shape}, "
                f"fit_scores_shape_pc_by_time={np.asarray(fit['scores_fit']).shape}, "
                "reduced_dimension=units_to_pcs"
            )
        fit_summary_rows.append(
            _build_fit_summary_row(
                region=str(region),
                fit_scope="per_condition",
                fit_condition=str(condition),
                model=fit,
                n_units_common=len(common_unit_keys),
                n_time_bins_window=n_bins_window,
                condition_unit_counts=condition_unit_counts,
            )
        )

    concat_units_by_time = np.concatenate(
        [condition_matrices[condition] for condition in settings.conditions],
        axis=1,
    )
    concat_fit = _fit_pca_units_by_time(
        concat_units_by_time,
        max_components=settings.max_components,
    )
    if settings.verbose_logging:
        print(
            "[analysis] region PCA fit complete (concatenated fixation matrix): "
            f"region={region}, concatenated_order={list(settings.conditions)}, "
            f"fit_input_shape_units_by_time={concat_units_by_time.shape}, "
            f"components_shape_pc_by_unit={np.asarray(concat_fit['components']).shape}, "
            f"fit_scores_shape_pc_by_time={np.asarray(concat_fit['scores_fit']).shape}, "
            "reduced_dimension=units_to_pcs"
        )
    concat_fit["unit_keys"] = np.asarray(common_unit_keys, dtype=object)
    concat_fit["fit_condition"] = "concatenated"
    concat_fit["sample_conditions"] = np.concatenate(
        [
            np.asarray([str(condition)] * n_bins_window, dtype=object)
            for condition in settings.conditions
        ],
        axis=0,
    )
    concat_fit["sample_bin_centers_s"] = np.tile(
        np.asarray(bin_centers_s_window, dtype=float).reshape(-1),
        len(settings.conditions),
    )
    sample_condition_slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for condition in settings.conditions:
        start_idx = int(cursor)
        stop_idx = int(cursor + n_bins_window)
        sample_condition_slices[str(condition)] = (start_idx, stop_idx)
        cursor = stop_idx
    concat_fit["sample_condition_slices"] = sample_condition_slices
    fit_summary_rows.append(
        _build_fit_summary_row(
            region=str(region),
            fit_scope="concatenated",
            fit_condition="concatenated",
            model=concat_fit,
            n_units_common=len(common_unit_keys),
            n_time_bins_window=n_bins_window,
            condition_unit_counts=condition_unit_counts,
        )
    )

    timecourse_rows: list[dict] = []
    concatenated_condition_scores_pc_by_time: dict[str, np.ndarray] = {}
    for condition in settings.conditions:
        scores = _project_units_by_time_with_model(condition_matrices[condition], concat_fit)
        concatenated_condition_scores_pc_by_time[condition] = scores
        if settings.verbose_logging:
            print(
                "[analysis] region PCA projection complete: "
                f"region={region}, fixation_type={condition}, "
                f"projection_input_shape_units_by_time={condition_matrices[condition].shape}, "
                f"projection_output_shape_pc_by_time={scores.shape}"
            )
        n_components = int(scores.shape[0])
        n_time_bins_proj = int(scores.shape[1])
        if n_time_bins_proj != n_bins_window:
            raise ValueError(
                "Projected PCA score matrix has unexpected time dimension; "
                f"region={region}, fixation_type={condition}, "
                f"expected={n_bins_window}, got={n_time_bins_proj}"
            )
        for pc_idx in range(n_components):
            for bin_idx, bin_center_s in enumerate(np.asarray(bin_centers_s_window, dtype=float)):
                timecourse_rows.append(
                    {
                        "region": str(region),
                        "fit_scope": "concatenated",
                        "fit_condition": "concatenated",
                        "eval_condition": str(condition),
                        "pc_index": int(pc_idx + 1),
                        "bin_index": int(bin_idx),
                        "bin_center_s": float(bin_center_s),
                        "pc_score": float(scores[pc_idx, bin_idx]),
                        "n_units_common": int(len(common_unit_keys)),
                    }
                )

    geometry_time_rows, geometry_summary_rows = _build_region_pairwise_geometry_rows(
        region=str(region),
        condition_scores_pc_by_time=concatenated_condition_scores_pc_by_time,
        bin_centers_s_window=np.asarray(bin_centers_s_window, dtype=float),
        settings=settings,
    )

    explained_rows: list[dict] = []
    fit_models: dict[str, dict] = {
        str(condition): per_condition_fits[str(condition)] for condition in settings.conditions
    }
    fit_models["concatenated"] = concat_fit
    for fit_condition, fit_model in fit_models.items():
        fit_scope = "concatenated" if str(fit_condition) == "concatenated" else "per_condition"
        for eval_condition in settings.conditions:
            profile = _explained_variance_profile_for_eval(
                condition_matrices[eval_condition],
                fit_model,
            )
            per_pc_var = np.asarray(profile.get("per_pc_variance", np.asarray([], dtype=float)), dtype=float)
            cumulative_var = np.asarray(profile.get("cumulative_variance", np.asarray([], dtype=float)), dtype=float)
            total_projected_var = profile.get("total_projected_variance", np.nan)
            per_pc = np.asarray(profile.get("per_pc_fraction", np.asarray([], dtype=float)), dtype=float)
            cumulative = np.asarray(profile.get("cumulative_fraction", np.asarray([], dtype=float)), dtype=float)
            n_rows_profile = min(
                int(per_pc_var.size),
                int(cumulative_var.size),
                int(per_pc.size),
                int(cumulative.size),
            )
            for comp_idx in range(1, n_rows_profile + 1):
                var = float(per_pc_var[comp_idx - 1]) if np.isfinite(per_pc_var[comp_idx - 1]) else np.nan
                var_cum = (
                    float(cumulative_var[comp_idx - 1])
                    if np.isfinite(cumulative_var[comp_idx - 1])
                    else np.nan
                )
                frac = float(per_pc[comp_idx - 1]) if np.isfinite(per_pc[comp_idx - 1]) else np.nan
                frac_cum = (
                    float(cumulative[comp_idx - 1])
                    if np.isfinite(cumulative[comp_idx - 1])
                    else np.nan
                )
                explained_rows.append(
                    {
                        "region": str(region),
                        "fit_scope": str(fit_scope),
                        "fit_condition": str(fit_condition),
                        "eval_condition": str(eval_condition),
                        "n_components": int(comp_idx),
                        "projected_variance": var,
                        "projected_variance_cumulative": var_cum,
                        "projected_variance_total": (
                            float(total_projected_var)
                            if np.isfinite(total_projected_var)
                            else np.nan
                        ),
                        # Legacy column name now stores per-PC explained-variance fraction.
                        "explained_variance_fraction": frac,
                        "explained_variance_per_pc_fraction": frac,
                        "explained_variance_cumulative_fraction": frac_cum,
                        "explained_variance_measure": (
                            "projection_variance_fraction_within_retained_pcs"
                        ),
                        "explained_variance_total_reference": (
                            "sum_projection_variance_retained_pcs"
                        ),
                        "n_units_common": int(len(common_unit_keys)),
                        "n_time_bins_window": int(n_bins_window),
                    }
                )

    unit_rows: list[dict] = []
    for unit_key in common_unit_keys:
        meta = unit_meta.get(unit_key, {})
        unit_rows.append(
            {
                "region": str(region),
                "unit_key": str(unit_key),
                "date": meta.get("date"),
                "unit_uuid": meta.get("unit_uuid"),
                "spike_channel": meta.get("spike_channel"),
                "recorded_agent": meta.get("recorded_agent"),
                "recorded_monkey": meta.get("recorded_monkey"),
                "area": meta.get("area"),
            }
        )

    payload = {
        "region": str(region),
        "unit_keys": np.asarray(common_unit_keys, dtype=object),
        "condition_unit_counts": condition_unit_counts,
        "n_units_common": int(len(common_unit_keys)),
        "bin_centers_s_window": np.asarray(bin_centers_s_window, dtype=float),
        "conditions": tuple(settings.conditions),
        "condition_matrices_units_by_time": condition_matrices,
        "per_condition_fits": per_condition_fits,
        "concatenated_fit": concat_fit,
        # Primary projected output orientation is PCs x time.
        "concatenated_condition_scores": concatenated_condition_scores_pc_by_time,
        "concatenated_condition_scores_pc_by_time": concatenated_condition_scores_pc_by_time,
        # Legacy alias retained for compatibility.
        "concatenated_condition_scores_time_by_pc": {
            condition: np.asarray(scores, dtype=float).T
            for condition, scores in concatenated_condition_scores_pc_by_time.items()
        },
        # Concatenated projection directly from the concatenated fit:
        # cols are ordered by concat_fit["sample_conditions"].
        "concatenated_projection_pc_by_time": np.asarray(
            concat_fit.get("scores_fit", np.asarray([], dtype=float)),
            dtype=float,
        ),
        "concatenated_projection_time_by_pc": np.asarray(
            concat_fit.get("scores_fit", np.asarray([], dtype=float)),
            dtype=float,
        ).T,
        "concatenated_projection_sample_conditions": np.asarray(
            concat_fit.get("sample_conditions", np.asarray([], dtype=object)),
            dtype=object,
        ),
        "concatenated_projection_sample_bin_centers_s": np.asarray(
            concat_fit.get("sample_bin_centers_s", np.asarray([], dtype=float)),
            dtype=float,
        ),
        "concatenated_projection_condition_slices": dict(sample_condition_slices),
        "pairwise_geometry_timecourses": pd.DataFrame(geometry_time_rows),
        "pairwise_geometry_summary": pd.DataFrame(geometry_summary_rows),
    }
    if settings.verbose_logging:
        print(
            "[analysis] region PCA payload ready: "
            f"region={region}, n_units_common={int(len(common_unit_keys))}, "
            f"n_window_time_bins={n_bins_window}, "
            f"timecourse_rows={len(timecourse_rows)}, explained_variance_rows={len(explained_rows)}, "
            f"geometry_rows={len(geometry_time_rows)}"
        )
        print("[analysis] ===== region PCA end =====")
    return {
        "region": str(region),
        "skipped_reason": None,
        "condition_unit_counts": condition_unit_counts,
        "summary_rows": fit_summary_rows,
        "timecourse_rows": timecourse_rows,
        "explained_rows": explained_rows,
        "geometry_time_rows": geometry_time_rows,
        "geometry_summary_rows": geometry_summary_rows,
        "unit_rows": unit_rows,
        "payload": payload,
    }


def run_fixation_population_pca_analysis(
    settings: FixationPopulationPCASettings,
    *,
    dates: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
) -> dict:
    """Run fixation population PCA and save analysis artifacts."""
    if settings.verbose_logging:
        print(
            "[analysis] fixation population PCA request: "
            f"analysis_variant={settings.analysis_variant}, "
            f"fixation_types={list(settings.conditions)}, "
            f"date_filter={list(dates) if dates is not None else 'all'}, "
            f"region_filter={list(regions) if regions is not None else 'all'}"
        )
    input_df, bin_centers_s, input_source = _load_input_psth_table(settings, dates=dates)
    if settings.verbose_logging:
        print(
            "[analysis] fixation population PCA input loaded: "
            f"source={input_source}, rows={len(input_df)}, "
            f"n_bins_full={int(np.asarray(bin_centers_s).size)}"
        )
    if input_df.empty:
        print("[analysis] no usable fixation PSTH rows found for population PCA")
        return {
            "meta": {
                "analysis_variant": str(settings.analysis_variant),
                "input_source": str(input_source),
                "n_regions_input": 0,
                "n_regions_analyzed": 0,
                "n_units_common_total": 0,
                "conditions": list(settings.conditions),
            },
            "fit_summary": pd.DataFrame(),
            "concatenated_timecourses": pd.DataFrame(),
            "cross_condition_explained_variance": pd.DataFrame(),
            "pairwise_geometry_timecourses": pd.DataFrame(),
            "pairwise_geometry_summary": pd.DataFrame(),
            "pairwise_geometry_within_region_stats": pd.DataFrame(),
            "pairwise_geometry_cross_region_stats": pd.DataFrame(),
            "unit_inventory": pd.DataFrame(),
            "regions": {},
        }

    bin_mask = _window_mask_from_centers(
        bin_centers_s,
        window_start_ms=settings.window_start_ms,
        window_stop_ms=settings.window_stop_ms,
    )
    bin_centers_s_window = np.asarray(bin_centers_s, dtype=float).reshape(-1)[bin_mask]
    if settings.verbose_logging:
        print(
            "[analysis] fixation population PCA window: "
            f"window_ms=[{float(min(settings.window_start_ms, settings.window_stop_ms))}, "
            f"{float(max(settings.window_start_ms, settings.window_stop_ms))}], "
            f"n_bins_window={int(np.asarray(bin_centers_s_window).size)}"
        )

    records_windowed: list[dict] = []
    for row in input_df.to_dict(orient="records"):
        psth = np.asarray(row.get("psth_mean"), dtype=float).reshape(-1)
        if psth.size != int(np.asarray(bin_centers_s).size):
            raise ValueError(
                "PSTH length does not match resolved bin centers; "
                f"unit_key={row.get('unit_key')}, condition={row.get('condition')}"
            )
        windowed = psth[bin_mask]
        rec = dict(row)
        rec["psth_window"] = windowed
        records_windowed.append(rec)

    windowed_df = pd.DataFrame(records_windowed)
    if settings.verbose_logging and not windowed_df.empty:
        condition_counts = windowed_df["condition"].astype(str).value_counts().to_dict()
        region_counts = windowed_df["region"].astype(str).value_counts().to_dict()
        print(
            "[analysis] fixation population PCA windowed rows: "
            f"rows={len(windowed_df)}, condition_counts={condition_counts}, "
            f"region_row_counts={region_counts}"
        )
    if windowed_df.empty:
        print("[analysis] no rows remain after applying population PCA time window")
        return {
            "meta": {
                "analysis_variant": str(settings.analysis_variant),
                "n_regions_input": 0,
                "n_regions_analyzed": 0,
                "n_units_common_total": 0,
                "conditions": list(settings.conditions),
            },
            "fit_summary": pd.DataFrame(),
            "concatenated_timecourses": pd.DataFrame(),
            "cross_condition_explained_variance": pd.DataFrame(),
            "pairwise_geometry_timecourses": pd.DataFrame(),
            "pairwise_geometry_summary": pd.DataFrame(),
            "pairwise_geometry_within_region_stats": pd.DataFrame(),
            "pairwise_geometry_cross_region_stats": pd.DataFrame(),
            "unit_inventory": pd.DataFrame(),
            "regions": {},
        }

    if regions is not None:
        allowed = {str(region) for region in regions}
        windowed_df = windowed_df.loc[windowed_df["region"].astype(str).isin(allowed)].copy()
    if windowed_df.empty:
        print("[analysis] no rows remain after region filtering for population PCA")
        return {
            "meta": {
                "analysis_variant": str(settings.analysis_variant),
                "n_regions_input": 0,
                "n_regions_analyzed": 0,
                "n_units_common_total": 0,
                "conditions": list(settings.conditions),
            },
            "fit_summary": pd.DataFrame(),
            "concatenated_timecourses": pd.DataFrame(),
            "cross_condition_explained_variance": pd.DataFrame(),
            "pairwise_geometry_timecourses": pd.DataFrame(),
            "pairwise_geometry_summary": pd.DataFrame(),
            "pairwise_geometry_within_region_stats": pd.DataFrame(),
            "pairwise_geometry_cross_region_stats": pd.DataFrame(),
            "unit_inventory": pd.DataFrame(),
            "regions": {},
        }

    region_names = sorted(windowed_df["region"].astype(str).unique().tolist())
    if settings.verbose_logging:
        print(
            "[analysis] fixation population PCA regions selected: "
            f"{region_names}"
        )
    tasks = [
        (
            region,
            windowed_df.loc[windowed_df["region"].astype(str) == region].copy(),
            np.asarray(bin_centers_s_window, dtype=float),
            settings,
        )
        for region in region_names
    ]
    region_results = run_tasks(
        _build_region_analysis,
        tasks,
        desc="Fixation population PCA",
        unit="region",
        use_parallel=settings.use_parallel,
        max_procs=settings.max_procs,
    )

    summary_rows: list[dict] = []
    timecourse_rows: list[dict] = []
    explained_rows: list[dict] = []
    geometry_time_rows: list[dict] = []
    geometry_summary_rows: list[dict] = []
    unit_rows: list[dict] = []
    region_payloads: dict[str, dict] = {}
    skipped_regions: dict[str, dict] = {}

    for result in region_results:
        region = str(result.get("region"))
        skipped_reason = result.get("skipped_reason")
        if skipped_reason:
            skipped_regions[region] = {
                "reason": str(skipped_reason),
                "condition_unit_counts": result.get("condition_unit_counts", {}),
            }
            continue

        summary_rows.extend(result.get("summary_rows", []))
        timecourse_rows.extend(result.get("timecourse_rows", []))
        explained_rows.extend(result.get("explained_rows", []))
        geometry_time_rows.extend(result.get("geometry_time_rows", []))
        geometry_summary_rows.extend(result.get("geometry_summary_rows", []))
        unit_rows.extend(result.get("unit_rows", []))
        payload = result.get("payload")
        if payload is not None:
            region_payloads[region] = payload

    summary_df = pd.DataFrame(summary_rows)
    timecourse_df = pd.DataFrame(timecourse_rows)
    explained_df = pd.DataFrame(explained_rows)
    geometry_time_df = pd.DataFrame(geometry_time_rows)
    geometry_summary_df = pd.DataFrame(geometry_summary_rows)
    geometry_within_df, geometry_cross_df = _build_pairwise_geometry_stat_tables(
        geometry_time_df,
        settings=settings,
    )
    unit_df = pd.DataFrame(unit_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["region", "fit_scope", "fit_condition"],
        ).reset_index(drop=True)
    if not timecourse_df.empty:
        timecourse_df = timecourse_df.sort_values(
            ["region", "eval_condition", "pc_index", "bin_index"],
        ).reset_index(drop=True)
    if not explained_df.empty:
        explained_df = explained_df.sort_values(
            ["region", "fit_condition", "eval_condition", "n_components"],
        ).reset_index(drop=True)
    if not geometry_time_df.empty:
        geometry_time_df = geometry_time_df.sort_values(
            ["metric_name", "region", "condition_pair", "bin_index"],
        ).reset_index(drop=True)
    if not geometry_summary_df.empty:
        geometry_summary_df = geometry_summary_df.sort_values(
            ["metric_name", "region", "condition_pair"],
        ).reset_index(drop=True)
    if not unit_df.empty:
        unit_df = unit_df.sort_values(["region", "date", "unit_uuid"]).reset_index(drop=True)
    geometry_n_pcs_effective_max = 0
    if not geometry_time_df.empty and "n_pcs_used" in geometry_time_df.columns:
        n_pcs_used = pd.to_numeric(geometry_time_df["n_pcs_used"], errors="coerce").to_numpy(dtype=float)
        n_pcs_used = n_pcs_used[np.isfinite(n_pcs_used)]
        if n_pcs_used.size > 0:
            geometry_n_pcs_effective_max = int(np.max(n_pcs_used))

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = out_root / _ensure_filename(settings.summary_filename, ".csv")
    timecourse_csv = out_root / _ensure_filename(settings.timecourse_filename, ".csv")
    explained_csv = out_root / _ensure_filename(settings.explained_variance_filename, ".csv")
    geometry_time_csv = out_root / _ensure_filename(settings.pairwise_geometry_timecourse_filename, ".csv")
    geometry_summary_csv = out_root / _ensure_filename(settings.pairwise_geometry_summary_filename, ".csv")
    geometry_within_csv = out_root / _ensure_filename(
        settings.pairwise_geometry_within_region_stats_filename,
        ".csv",
    )
    geometry_cross_csv = out_root / _ensure_filename(
        settings.pairwise_geometry_cross_region_stats_filename,
        ".csv",
    )
    unit_csv = out_root / _ensure_filename(settings.unit_inventory_filename, ".csv")
    result_pkl = out_root / _ensure_filename(settings.output_pickle_filename, ".pkl")

    summary_df.to_csv(summary_csv, index=False)
    timecourse_df.to_csv(timecourse_csv, index=False)
    explained_df.to_csv(explained_csv, index=False)
    geometry_time_df.to_csv(geometry_time_csv, index=False)
    geometry_summary_df.to_csv(geometry_summary_csv, index=False)
    geometry_within_df.to_csv(geometry_within_csv, index=False)
    geometry_cross_df.to_csv(geometry_cross_csv, index=False)
    unit_df.to_csv(unit_csv, index=False)
    if settings.verbose_logging:
        print(
            "[analysis] fixation population PCA projected output table: "
            f"path={timecourse_csv}, rows={len(timecourse_df)}, "
            f"columns={list(timecourse_df.columns)}"
        )
        for region_name, payload in region_payloads.items():
            print(f"\n[analysis] --- projected outputs: region={region_name} ---")
            condition_scores = payload.get("concatenated_condition_scores", {})
            for condition_name in settings.conditions:
                shape = np.asarray(
                    condition_scores.get(condition_name, np.asarray([], dtype=float))
                ).shape
                print(
                    "[analysis] fixation population PCA projected matrix: "
                    f"region={region_name}, fixation_type={condition_name}, "
                    f"shape_pc_by_time={shape}"
                )
            concat_projection = np.asarray(
                payload.get("concatenated_projection_pc_by_time", np.asarray([], dtype=float)),
                dtype=float,
            )
            concat_conditions = np.asarray(
                payload.get("concatenated_projection_sample_conditions", np.asarray([], dtype=object)),
                dtype=object,
            )
            concat_condition_counts = {
                str(condition): int(np.sum(concat_conditions.astype(str) == str(condition)))
                for condition in settings.conditions
            }
            concat_condition_slices = payload.get("concatenated_projection_condition_slices", {})
            print(
                "[analysis] fixation population PCA concatenated projected matrix: "
                f"region={region_name}, shape_pc_by_time={concat_projection.shape}, "
                f"condition_counts={concat_condition_counts}, "
                f"condition_slices={concat_condition_slices}"
            )
            per_cond = payload.get("per_condition_fits", {})
            for condition_name in settings.conditions:
                model = per_cond.get(condition_name, {})
                components_shape = np.asarray(model.get("components", np.asarray([], dtype=float))).shape
                print(
                    "[analysis] fixation population PCA fit PCs saved: "
                    f"region={region_name}, fit_condition={condition_name}, "
                    f"components_shape_pc_by_unit={components_shape}"
                )
            concat_model = payload.get("concatenated_fit", {})
            concat_components_shape = np.asarray(
                concat_model.get("components", np.asarray([], dtype=float))
            ).shape
            print(
                "[analysis] fixation population PCA fit PCs saved: "
                f"region={region_name}, fit_condition=concatenated, "
                f"components_shape_pc_by_unit={concat_components_shape}"
            )

    result_obj = {
        "meta": {
            "analysis_variant": str(settings.analysis_variant),
            "input_source": str(input_source),
            "input_subdir": settings.input_subdir,
            "input_filename": _ensure_filename(settings.input_filename, ".pkl"),
            "object_input_subdir": settings.object_input_subdir,
            "object_input_filename": (
                _ensure_filename(settings.object_input_filename, ".pkl")
                if settings.object_input_filename is not None
                else None
            ),
            "trial_input_modality": settings.trial_input_modality,
            "trial_input_filename": _ensure_filename(settings.trial_input_filename, ".pkl"),
            "prefer_trial_input": bool(settings.prefer_trial_input),
            "allow_trial_fallback": bool(settings.allow_trial_fallback),
            "smooth_before_average": bool(settings.smooth_before_average),
            "smoothing_sigma_ms": float(settings.smoothing_sigma_ms),
            "output_subdir": settings.output_subdir,
            "window_start_ms": float(min(settings.window_start_ms, settings.window_stop_ms)),
            "window_stop_ms": float(max(settings.window_start_ms, settings.window_stop_ms)),
            "n_bins_full": int(np.asarray(bin_centers_s).size),
            "n_bins_window": int(np.asarray(bin_centers_s_window).size),
            "conditions": list(settings.conditions),
            "require_all_conditions": bool(settings.require_all_conditions),
            "min_units_per_region": int(settings.min_units_per_region),
            "max_components": (
                None if settings.max_components is None else int(settings.max_components)
            ),
            "geometry_n_pcs_requested": _resolve_geometry_n_pcs_requested(settings),
            "geometry_n_pcs_effective_max": int(geometry_n_pcs_effective_max),
            "geometry_n_pcs": (
                int(geometry_n_pcs_effective_max)
                if int(geometry_n_pcs_effective_max) > 0
                else _resolve_geometry_n_pcs_requested(settings)
            ),
            "geometry_angle_unit": _resolve_angle_unit(settings),
            "geometry_alpha": float(settings.geometry_alpha),
            "geometry_pvalue_correction": normalize_pvalue_correction(
                settings.geometry_pvalue_correction
            ),
            "n_regions_input": int(len(region_names)),
            "n_regions_analyzed": int(len(region_payloads)),
            "n_units_common_total": int(unit_df["unit_key"].nunique()) if not unit_df.empty else 0,
            "skipped_regions": skipped_regions,
        },
        "bin_centers_s_full": np.asarray(bin_centers_s, dtype=float),
        "bin_centers_s_window": np.asarray(bin_centers_s_window, dtype=float),
        "fit_summary": summary_df,
        "concatenated_timecourses": timecourse_df,
        "cross_condition_explained_variance": explained_df,
        "pairwise_geometry_timecourses": geometry_time_df,
        "pairwise_geometry_summary": geometry_summary_df,
        "pairwise_geometry_within_region_stats": geometry_within_df,
        "pairwise_geometry_cross_region_stats": geometry_cross_df,
        "unit_inventory": unit_df,
        "regions": region_payloads,
    }
    save_pickle_path(result_obj, result_pkl)
    return result_obj
