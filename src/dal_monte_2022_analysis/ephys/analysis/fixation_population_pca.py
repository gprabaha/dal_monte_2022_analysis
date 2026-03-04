"""Population PCA analysis for fixation-conditioned average PSTHs."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_date_paths
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
    scan_processed_paths_for_filename,
)
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_POPULATION_PCA_CONDITIONS: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)


@dataclass
class FixationPopulationPCASettings:
    """Configuration for population PCA from fixation PSTH averages."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations.pkl"
    prefer_trial_input: bool = True
    allow_trial_fallback: bool = True
    input_subdir: str = "ephys/psth/fixation_psth_averages"
    input_filename: str = "fixations.pkl"
    output_subdir: str = "ephys/psth/fixation_population_pca"
    summary_filename: str = "pca_fit_summary.csv"
    timecourse_filename: str = "concatenated_pc_timecourses.csv"
    explained_variance_filename: str = "cross_condition_explained_variance.csv"
    unit_inventory_filename: str = "region_unit_inventory.csv"
    output_pickle_filename: str = "results.pkl"
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    conditions: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_POPULATION_PCA_CONDITIONS),
    )
    window_start_ms: float = -500.0
    window_stop_ms: float = 500.0
    max_components: Optional[int] = None
    min_units_per_region: int = 3
    require_all_conditions: bool = True
    require_face_interactive_state: bool = True
    smooth_before_average: bool = True
    smoothing_sigma_ms: float = 20.0
    use_parallel: bool = True
    max_procs: int = 16
    bin_size_ms_fallback: float = 10.0
    window_pre_s_fallback: float = 1.0
    window_post_s_fallback: float = 1.0


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


def _resolve_condition_from_row_fields(
    *,
    fixation_category: object,
    interactive_state: object,
    is_interactive: object,
    settings: FixationPopulationPCASettings,
) -> Optional[str]:
    category_token = _norm_token(fixation_category)
    if not category_token or category_token == "nan":
        return None

    object_tokens = {
        _norm_token(settings.object_label),
        "object",
        "objects",
    }
    if category_token in object_tokens:
        return "object"

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

    face_token = _norm_token(settings.face_label)
    if category_token != face_token:
        return None

    has_interactive_state = interactive_state is not None and not pd.isna(interactive_state)
    has_is_interactive = is_interactive is not None and not pd.isna(is_interactive)
    if not has_interactive_state and not has_is_interactive:
        if settings.require_face_interactive_state:
            raise ValueError(
                "Face rows are missing interactive-state labels. "
                "Build averages with split_by_interactive_state=true "
                "or provide is_interactive/interactive_state columns."
            )
        return "face_non_interactive"

    if has_is_interactive:
        interactive = _as_bool(is_interactive, settings.interactive_label)
    else:
        interactive = _as_bool(interactive_state, settings.interactive_label)
    return "face_interactive" if interactive else "face_non_interactive"


def _resolve_condition_for_average_row(
    row: pd.Series,
    settings: FixationPopulationPCASettings,
) -> Optional[str]:
    return _resolve_condition_from_row_fields(
        fixation_category=row.get("fixation_category"),
        interactive_state=row.get("interactive_state"),
        is_interactive=row.get("is_interactive"),
        settings=settings,
    )


def _resolve_condition_for_trial_row(
    row,
    settings: FixationPopulationPCASettings,
) -> Optional[str]:
    return _resolve_condition_from_row_fields(
        fixation_category=getattr(row, "fixation_category", None),
        interactive_state=getattr(row, "interactive_state", None),
        is_interactive=getattr(row, "is_interactive", None),
        settings=settings,
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
) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = load_config(settings.cfg_path)
    subdir = str(settings.input_subdir if input_subdir is None else input_subdir).strip()
    if not subdir:
        return pd.DataFrame(), np.asarray([], dtype=float)
    rows = scan_analysis_date_paths(
        cfg,
        subdir,
        filename=_ensure_filename(settings.input_filename, ".pkl"),
        dates=dates,
    )
    if not rows:
        return pd.DataFrame(), np.asarray([], dtype=float)

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

        for _, avg_row in avg_df.iterrows():
            condition = _resolve_condition_for_average_row(avg_row, settings)
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

            date = _as_optional_str(avg_row.get("date")) or str(row["date"])
            unit_uuid = _as_optional_str(avg_row.get("unit_uuid"))
            if unit_uuid is None:
                continue
            unit_key = f"{date}|{unit_uuid}"
            region = _as_optional_str(avg_row.get("region")) or "unknown"

            records.append(
                {
                    "date": date,
                    "unit_uuid": unit_uuid,
                    "unit_key": unit_key,
                    "region": region,
                    "spike_channel": _as_optional_str(avg_row.get("spike_channel")),
                    "recorded_agent": _as_optional_str(avg_row.get("recorded_agent")),
                    "recorded_monkey": _as_optional_str(avg_row.get("recorded_monkey")),
                    "area": _as_optional_str(avg_row.get("area")),
                    "condition": condition,
                    "n_trials": _normalize_n_trials(avg_row.get("n_trials", 1.0)),
                    "psth_mean": psth_mean,
                }
            )

    return _aggregate_psth_records(
        records,
        settings=settings,
        n_bins_ref=n_bins_ref,
        bin_centers_ref=bin_centers_ref,
    )


def _load_trial_averaged_psth_table(
    settings: FixationPopulationPCASettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = load_config(settings.cfg_path)
    rows = scan_processed_paths_for_filename(
        cfg,
        settings.trial_input_modality,
        filename=_ensure_filename(settings.trial_input_filename, ".pkl"),
        dates=dates,
        sessions=sessions,
        agents=(None,),
    )
    if not rows:
        return pd.DataFrame(), np.asarray([], dtype=float)

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

            date = _as_optional_str(getattr(trial_row, "date", None)) or str(row["date"])
            unit_uuid = _as_optional_str(getattr(trial_row, "unit_uuid", None))
            if unit_uuid is None:
                continue
            unit_key = f"{date}|{unit_uuid}"
            region = _as_optional_str(getattr(trial_row, "region", None)) or "unknown"

            records.append(
                {
                    "date": date,
                    "unit_uuid": unit_uuid,
                    "unit_key": unit_key,
                    "region": region,
                    "spike_channel": _as_optional_str(getattr(trial_row, "spike_channel", None)),
                    "recorded_agent": _as_optional_str(getattr(trial_row, "recorded_agent", None)),
                    "recorded_monkey": _as_optional_str(getattr(trial_row, "recorded_monkey", None)),
                    "area": _as_optional_str(getattr(trial_row, "area", None)),
                    "condition": condition,
                    "n_trials": 1.0,
                    "psth_mean": psth_counts,
                }
            )

    return _aggregate_psth_records(
        records,
        settings=settings,
        n_bins_ref=n_bins_ref,
        bin_centers_ref=bin_centers_ref,
    )


def _load_input_psth_table(
    settings: FixationPopulationPCASettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Load PCA input rows with trial-first fallback for robust accumulation."""
    average_candidates: list[str] = []
    primary_avg_subdir = str(settings.input_subdir).strip()
    if primary_avg_subdir:
        average_candidates.append(primary_avg_subdir)
    index_avg_subdir = "ephys/psth/fixation_psth_index_averages"
    if index_avg_subdir not in average_candidates:
        average_candidates.append(index_avg_subdir)

    if settings.prefer_trial_input:
        trial_df, trial_centers = _load_trial_averaged_psth_table(
            settings,
            dates=dates,
            sessions=sessions,
        )
        if not trial_df.empty:
            return trial_df, trial_centers, "trial"
        print("[analysis] no usable trial PSTH rows found for population PCA; trying average inputs")

        for subdir in average_candidates:
            try:
                avg_df, avg_centers = _load_average_psth_table(
                    settings,
                    dates=dates,
                    input_subdir=subdir,
                )
            except ValueError as exc:
                print(f"[analysis] skipping average input subdir '{subdir}': {exc}")
                continue
            if not avg_df.empty:
                return avg_df, avg_centers, f"average:{subdir}"
        return pd.DataFrame(), np.asarray([], dtype=float), "none"

    for subdir in average_candidates:
        try:
            avg_df, avg_centers = _load_average_psth_table(
                settings,
                dates=dates,
                input_subdir=subdir,
            )
        except ValueError as exc:
            print(f"[analysis] skipping average input subdir '{subdir}': {exc}")
            continue
        if not avg_df.empty:
            return avg_df, avg_centers, f"average:{subdir}"

    if settings.allow_trial_fallback:
        trial_df, trial_centers = _load_trial_averaged_psth_table(
            settings,
            dates=dates,
            sessions=sessions,
        )
        if not trial_df.empty:
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
    scores_fit = X_centered @ components.T
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
        "scores_fit": scores_fit,
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
    return (X - mean) @ components.T


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


def _explained_variance_curve_for_eval(
    matrix_units_by_time_eval: np.ndarray,
    model: dict,
) -> np.ndarray:
    matrix = np.asarray(matrix_units_by_time_eval, dtype=float)
    X = matrix.T  # samples=time bins, features=units
    n_components = int(model["n_components"])
    if n_components <= 0:
        return np.asarray([], dtype=float)

    X_centered_eval = X - np.mean(X, axis=0, keepdims=True)
    total_var = float(np.sum(X_centered_eval**2))
    values = np.full((n_components,), np.nan, dtype=float)
    if total_var <= 0.0:
        return values

    for k in range(1, n_components + 1):
        X_hat = _reconstruct_samples_with_model(X, model, n_components=k)
        residual = X - X_hat
        sse = float(np.sum(residual**2))
        values[k - 1] = 1.0 - (sse / total_var)
    return values


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
    return {
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
        "n_units_face_interactive": int(condition_unit_counts.get("face_interactive", 0)),
        "n_units_face_non_interactive": int(condition_unit_counts.get("face_non_interactive", 0)),
        "n_units_object": int(condition_unit_counts.get("object", 0)),
    }


def _build_region_analysis(args) -> dict:
    region, region_df, bin_centers_s_window, settings = args

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
    if not condition_maps:
        return {
            "region": str(region),
            "skipped_reason": "no_condition_maps",
            "condition_unit_counts": condition_unit_counts,
            "summary_rows": [],
            "timecourse_rows": [],
            "explained_rows": [],
            "unit_rows": [],
            "payload": None,
        }

    # Cross-condition projections require a shared neuron basis across all
    # conditions, so all downstream fits are computed on the intersection.
    unit_sets = [set(unit_map.keys()) for unit_map in condition_maps.values()]
    common_unit_keys = sorted(set.intersection(*unit_sets)) if unit_sets else []

    if int(len(common_unit_keys)) < int(settings.min_units_per_region):
        return {
            "region": str(region),
            "skipped_reason": "insufficient_units",
            "condition_unit_counts": condition_unit_counts,
            "summary_rows": [],
            "timecourse_rows": [],
            "explained_rows": [],
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
    concatenated_condition_scores: dict[str, np.ndarray] = {}
    for condition in settings.conditions:
        scores = _project_units_by_time_with_model(condition_matrices[condition], concat_fit)
        concatenated_condition_scores[condition] = scores
        n_components = int(scores.shape[1])
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
                        "pc_score": float(scores[bin_idx, pc_idx]),
                        "n_units_common": int(len(common_unit_keys)),
                    }
                )

    explained_rows: list[dict] = []
    for fit_condition in settings.conditions:
        fit_model = per_condition_fits[fit_condition]
        for eval_condition in settings.conditions:
            curve = _explained_variance_curve_for_eval(
                condition_matrices[eval_condition],
                fit_model,
            )
            for comp_idx, frac in enumerate(curve, start=1):
                explained_rows.append(
                    {
                        "region": str(region),
                        "fit_condition": str(fit_condition),
                        "eval_condition": str(eval_condition),
                        "n_components": int(comp_idx),
                        "explained_variance_fraction": float(frac) if np.isfinite(frac) else np.nan,
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
        "concatenated_condition_scores": concatenated_condition_scores,
    }
    return {
        "region": str(region),
        "skipped_reason": None,
        "condition_unit_counts": condition_unit_counts,
        "summary_rows": fit_summary_rows,
        "timecourse_rows": timecourse_rows,
        "explained_rows": explained_rows,
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
    input_df, bin_centers_s, input_source = _load_input_psth_table(settings, dates=dates)
    if input_df.empty:
        print("[analysis] no usable fixation PSTH rows found for population PCA")
        return {
            "meta": {
                "input_source": str(input_source),
                "n_regions_input": 0,
                "n_regions_analyzed": 0,
                "n_units_common_total": 0,
                "conditions": list(settings.conditions),
            },
            "fit_summary": pd.DataFrame(),
            "concatenated_timecourses": pd.DataFrame(),
            "cross_condition_explained_variance": pd.DataFrame(),
            "unit_inventory": pd.DataFrame(),
            "regions": {},
        }

    bin_mask = _window_mask_from_centers(
        bin_centers_s,
        window_start_ms=settings.window_start_ms,
        window_stop_ms=settings.window_stop_ms,
    )
    bin_centers_s_window = np.asarray(bin_centers_s, dtype=float).reshape(-1)[bin_mask]

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
    if windowed_df.empty:
        print("[analysis] no rows remain after applying population PCA time window")
        return {
            "meta": {
                "n_regions_input": 0,
                "n_regions_analyzed": 0,
                "n_units_common_total": 0,
                "conditions": list(settings.conditions),
            },
            "fit_summary": pd.DataFrame(),
            "concatenated_timecourses": pd.DataFrame(),
            "cross_condition_explained_variance": pd.DataFrame(),
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
                "n_regions_input": 0,
                "n_regions_analyzed": 0,
                "n_units_common_total": 0,
                "conditions": list(settings.conditions),
            },
            "fit_summary": pd.DataFrame(),
            "concatenated_timecourses": pd.DataFrame(),
            "cross_condition_explained_variance": pd.DataFrame(),
            "unit_inventory": pd.DataFrame(),
            "regions": {},
        }

    region_names = sorted(windowed_df["region"].astype(str).unique().tolist())
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
        unit_rows.extend(result.get("unit_rows", []))
        payload = result.get("payload")
        if payload is not None:
            region_payloads[region] = payload

    summary_df = pd.DataFrame(summary_rows)
    timecourse_df = pd.DataFrame(timecourse_rows)
    explained_df = pd.DataFrame(explained_rows)
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
    if not unit_df.empty:
        unit_df = unit_df.sort_values(["region", "date", "unit_uuid"]).reset_index(drop=True)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = out_root / _ensure_filename(settings.summary_filename, ".csv")
    timecourse_csv = out_root / _ensure_filename(settings.timecourse_filename, ".csv")
    explained_csv = out_root / _ensure_filename(settings.explained_variance_filename, ".csv")
    unit_csv = out_root / _ensure_filename(settings.unit_inventory_filename, ".csv")
    result_pkl = out_root / _ensure_filename(settings.output_pickle_filename, ".pkl")

    summary_df.to_csv(summary_csv, index=False)
    timecourse_df.to_csv(timecourse_csv, index=False)
    explained_df.to_csv(explained_csv, index=False)
    unit_df.to_csv(unit_csv, index=False)

    result_obj = {
        "meta": {
            "input_source": str(input_source),
            "input_subdir": settings.input_subdir,
            "input_filename": _ensure_filename(settings.input_filename, ".pkl"),
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
        "unit_inventory": unit_df,
        "regions": region_payloads,
    }
    save_pickle_path(result_obj, result_pkl)
    return result_obj
