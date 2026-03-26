"""Factorial ROI-vs-period analysis from fixation trial-level PSTH responses."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import (
    fisher_exact,
    t,
)

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
    ensure_filename as _ensure_filename,
    extract_trials_df_and_meta as _extract_trials_df_and_meta,
    resolve_bin_centers_from_meta as _resolve_bin_centers_from_meta,
)
from dal_monte_2022_analysis.core.stats import (
    adjust_pvalues,
    apply_adjusted_pvalues,
    normalize_pvalue_correction,
    safe_welch_ttest,
)
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
    scan_processed_paths_for_filename,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


DEFAULT_FACTORIAL_WINDOWS_MS: dict[str, tuple[float, float]] = {
    "pre_fix": (-500.0, 0.0),
    "peri_fix": (-250.0, 250.0),
    "post_fix": (0.0, 500.0),
    "full_fix": (-500.0, 500.0),
}
DEFAULT_SIGNIFICANCE_WINDOWS: tuple[str, ...] = ("pre_fix", "peri_fix", "post_fix")
TERM_TO_AXIS: tuple[tuple[str, str], ...] = (
    ("roi_main", "face_object"),
    ("period_main", "interactive_state"),
    ("interaction", "cross_interaction"),
)
AXIS_ORDER: tuple[str, ...] = tuple(axis_name for _, axis_name in TERM_TO_AXIS)
CELL_MEAN_AXIS_SOURCE = "cell_means"
CELL_MEAN_UNIT_RANGE_NORM_AXIS_SOURCE = "cell_means_unit_range_norm"
CELL_MEAN_MAGNITUDE_SOURCES: tuple[str, ...] = (
    CELL_MEAN_AXIS_SOURCE,
    CELL_MEAN_UNIT_RANGE_NORM_AXIS_SOURCE,
)
UNIT_WINDOW_MEAN_FR_COLUMNS: tuple[str, ...] = (
    "mean_fr_face_interactive_hz",
    "mean_fr_face_non_interactive_hz",
    "mean_fr_object_interactive_hz",
    "mean_fr_object_non_interactive_hz",
)
_ALLOWED_UNIT_SIGNIFICANCE_MODE = {"raw", "within_unit_corrected"}
_ALLOWED_PARALLELIZATION_SCOPE = {"date", "unit"}
_ALLOWED_AXIS_COMPARISON_MODE = {
    "split_by_window",
    "averaged_across_windows",
    "max_abs_across_windows",
}
_COLLAPSED_WINDOW_NAME_BY_MODE = {
    "averaged_across_windows": "avg_pre_peri_post",
    "max_abs_across_windows": "max_abs_across_windows",
}


@dataclass
class FixationROIVsPeriodFactorialSettings:
    """Configuration for ROI-vs-period factorial fixation analysis."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations.pkl"
    output_subdir: str = "ephys/psth/fixation_roi_vs_period_factorial"
    unit_term_filename: str = "unit_glm_terms.csv"
    unit_axis_filename: str = "unit_axis_values.csv"
    unit_axis_collapsed_filename: str = "unit_axis_collapsed_magnitude.csv"
    unit_window_summary_filename: str = "unit_window_condition_means.csv"
    region_fraction_filename: str = "region_significant_fractions.csv"
    region_fraction_pairwise_filename: str = "region_significant_fraction_pairwise.csv"
    region_fraction_within_region_filename: str = "region_significant_fraction_within_region.csv"
    region_axis_summary_filename: str = "region_axis_summary.csv"
    region_axis_pairwise_filename: str = "region_axis_pairwise.csv"
    region_axis_within_region_filename: str = "region_axis_within_region.csv"
    region_axis_friedman_filename: str = "region_axis_friedman.csv"
    output_pickle_filename: str = "results.pkl"
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    windows_ms: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_FACTORIAL_WINDOWS_MS),
    )
    significance_windows: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_SIGNIFICANCE_WINDOWS),
    )
    smooth_before_window_average: bool = True
    smoothing_sigma_ms: float = 20.0
    min_trials_per_cell: int = 2
    min_units_per_region: int = 5
    alpha: float = 0.05
    pvalue_correction: str = "fdr_bh"
    unit_significance_mode: str = "within_unit_corrected"
    axis_comparison_mode: str = "max_abs_across_windows"
    parallelization_scope: str = "date"
    use_parallel: bool = True
    max_procs: int = 16
    test_single: bool = False
    bin_size_ms_fallback: float = 10.0
    window_pre_s_fallback: float = 1.0
    window_post_s_fallback: float = 1.0


def _fallback_bin_centers(settings: FixationROIVsPeriodFactorialSettings) -> np.ndarray:
    bin_size_s = float(settings.bin_size_ms_fallback) / 1000.0
    pre = float(settings.window_pre_s_fallback)
    post = float(settings.window_post_s_fallback)
    edges = np.arange(-pre, post + bin_size_s * 0.5, bin_size_s, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _normalize_windows(
    windows_raw: dict[str, tuple[float, float]] | list | tuple,
) -> dict[str, tuple[float, float]]:
    if isinstance(windows_raw, dict):
        items = windows_raw.items()
    elif isinstance(windows_raw, (list, tuple)):
        items = []
        for i, entry in enumerate(windows_raw):
            if isinstance(entry, dict):
                name = entry.get("name", f"window_{i}")
                start = entry.get("start_ms")
                stop = entry.get("stop_ms")
                items.append((name, (start, stop)))
            else:
                items.append((f"window_{i}", entry))
    else:
        raise ValueError("windows_ms must be a dict or list-like object.")

    out: dict[str, tuple[float, float]] = {}
    for name, bounds in items:
        if bounds is None or len(bounds) != 2:
            raise ValueError(f"Window '{name}' must define [start_ms, stop_ms].")
        start_ms = float(bounds[0])
        stop_ms = float(bounds[1])
        if stop_ms <= start_ms:
            raise ValueError(f"Window '{name}' has invalid bounds: {bounds}.")
        out[str(name)] = (start_ms, stop_ms)
    if not out:
        raise ValueError("At least one analysis window must be defined.")
    return out


def _normalize_significance_windows(
    windows_raw: Sequence[str] | None,
    *,
    available_windows: Sequence[str],
) -> tuple[str, ...]:
    available = [str(name).strip() for name in available_windows if str(name).strip()]
    if not available:
        raise ValueError("No available windows were provided for significance filtering.")
    requested = (
        [str(name).strip() for name in windows_raw if str(name).strip()]
        if windows_raw is not None
        else list(DEFAULT_SIGNIFICANCE_WINDOWS)
    )
    if not requested:
        raise ValueError("significance_windows resolved to empty.")
    out: list[str] = []
    for name in requested:
        if name in available and name not in out:
            out.append(name)
    if not out:
        raise ValueError(
            "No configured significance windows matched available windows. "
            f"requested={requested}, available={available}"
        )
    return tuple(out)


def _resolve_unit_significance_mode(mode: str) -> str:
    token = str(mode).strip().lower()
    if token not in _ALLOWED_UNIT_SIGNIFICANCE_MODE:
        raise ValueError(
            f"Unsupported unit_significance_mode '{mode}'. "
            f"Expected one of: {sorted(_ALLOWED_UNIT_SIGNIFICANCE_MODE)}"
        )
    return token


def _resolve_parallelization_scope(scope: str) -> str:
    token = str(scope).strip().lower()
    if token not in _ALLOWED_PARALLELIZATION_SCOPE:
        raise ValueError(
            f"Unsupported parallelization_scope '{scope}'. "
            f"Expected one of: {sorted(_ALLOWED_PARALLELIZATION_SCOPE)}"
        )
    return token


def _resolve_axis_comparison_mode(mode: str) -> str:
    token = str(mode).strip().lower()
    aliases = {
        "split": "split_by_window",
        "window_split": "split_by_window",
        "split_by_window": "split_by_window",
        "max": "max_abs_across_windows",
        "max_abs": "max_abs_across_windows",
        "max_abs_across_windows": "max_abs_across_windows",
        "max_across_windows": "max_abs_across_windows",
        "averaged": "averaged_across_windows",
        "average": "averaged_across_windows",
        "averaged_across_windows": "averaged_across_windows",
    }
    resolved = aliases.get(token, token)
    if resolved not in _ALLOWED_AXIS_COMPARISON_MODE:
        raise ValueError(
            f"Unsupported axis_comparison_mode '{mode}'. "
            f"Expected one of: {sorted(_ALLOWED_AXIS_COMPARISON_MODE)}"
        )
    return resolved


def _collapsed_window_name_for_mode(mode: str) -> str:
    return str(_COLLAPSED_WINDOW_NAME_BY_MODE.get(str(mode), str(mode)))


def _load_trial_table(
    settings: FixationROIVsPeriodFactorialSettings,
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
        return pd.DataFrame(), _fallback_bin_centers(settings)

    dfs: list[pd.DataFrame] = []
    bin_centers_ref = None
    n_empty_trials = 0
    n_missing_psth_counts = 0
    for row in rows:
        obj = load_pickle_path(row["path"])
        trial_df, meta = _extract_trials_df_and_meta(obj)
        if trial_df.empty:
            n_empty_trials += 1
            continue
        if "psth_counts" not in trial_df.columns:
            n_missing_psth_counts += 1
            continue

        centers = _resolve_bin_centers_from_meta(meta)
        if centers is not None:
            if bin_centers_ref is None:
                bin_centers_ref = centers
            elif centers.shape != bin_centers_ref.shape or not np.allclose(centers, bin_centers_ref):
                raise ValueError(f"Mismatched PSTH bin centers across trial files; path={row['path']}")

        df = trial_df.copy()
        if "date" not in df.columns:
            df["date"] = str(row["date"])
        if "session" not in df.columns:
            df["session"] = str(row["session"])
        dfs.append(df)

    if not dfs:
        print(
            "[analysis] trial PSTH files were found but usable trial rows were not. "
            f"n_files={len(rows)}, empty_trials={n_empty_trials}, "
            f"missing_psth_counts={n_missing_psth_counts}"
        )
        return pd.DataFrame(), _fallback_bin_centers(settings)

    out_df = pd.concat(dfs, axis=0, ignore_index=True)
    if bin_centers_ref is None:
        bin_centers_ref = _fallback_bin_centers(settings)
    return out_df, np.asarray(bin_centers_ref, dtype=float)


def _resolve_trial_condition(
    row,
    settings: FixationROIVsPeriodFactorialSettings,
) -> Optional[tuple[str, str, float, float]]:
    category = str(getattr(row, "fixation_category", "")).strip()
    if category not in {settings.face_label, settings.object_label}:
        return None

    interactive = False
    if hasattr(row, "is_interactive"):
        interactive = _as_bool(getattr(row, "is_interactive"), settings.interactive_label)
    elif hasattr(row, "interactive_state"):
        interactive = _as_bool(getattr(row, "interactive_state"), settings.interactive_label)

    roi_label = "face" if category == settings.face_label else "object"
    period_label = "interactive" if interactive else "non_interactive"
    roi_code = 0.5 if roi_label == "face" else -0.5
    period_code = 0.5 if period_label == "interactive" else -0.5
    return roi_label, period_label, float(roi_code), float(period_code)


def _resolve_smoothing_sigma_bins(
    settings: FixationROIVsPeriodFactorialSettings,
    *,
    bin_size_s: float,
) -> Optional[float]:
    if not settings.smooth_before_window_average:
        return None
    if not np.isfinite(float(bin_size_s)) or float(bin_size_s) <= 0:
        raise ValueError("Unable to resolve bin size for factorial smoothing.")
    sigma_ms = float(settings.smoothing_sigma_ms)
    if sigma_ms <= 0:
        raise ValueError("smoothing_sigma_ms must be > 0 when smoothing is enabled.")
    sigma_bins = sigma_ms / (float(bin_size_s) * 1000.0)
    if not np.isfinite(sigma_bins) or sigma_bins <= 0:
        raise ValueError("Resolved smoothing sigma in bins must be > 0.")
    return float(sigma_bins)


def _prepare_window_masks(
    *,
    bin_centers_s: np.ndarray,
    windows_ms: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for win_name, (start_ms, stop_ms) in windows_ms.items():
        start_s = float(start_ms) / 1000.0
        stop_s = float(stop_ms) / 1000.0
        mask = (bin_centers_s >= start_s) & (bin_centers_s < stop_s)
        if not np.any(mask):
            raise ValueError(f"Window '{win_name}' has no bins covered by current PSTH bin centers.")
        masks[win_name] = mask
    return masks


def _fit_factorial_glm(
    y: np.ndarray,
    roi_code: np.ndarray,
    period_code: np.ndarray,
) -> dict:
    arr_y = np.asarray(y, dtype=float).reshape(-1)
    arr_roi = np.asarray(roi_code, dtype=float).reshape(-1)
    arr_period = np.asarray(period_code, dtype=float).reshape(-1)
    finite = np.isfinite(arr_y) & np.isfinite(arr_roi) & np.isfinite(arr_period)
    arr_y = arr_y[finite]
    arr_roi = arr_roi[finite]
    arr_period = arr_period[finite]
    n = int(arr_y.size)
    if n <= 4:
        return {
            "valid": False,
            "n_trials": n,
            "dof_resid": np.nan,
            "r2": np.nan,
            "coef": np.full(4, np.nan, dtype=float),
            "se": np.full(4, np.nan, dtype=float),
            "t": np.full(4, np.nan, dtype=float),
            "p": np.full(4, np.nan, dtype=float),
        }

    interaction = arr_roi * arr_period
    X = np.column_stack(
        [
            np.ones(n, dtype=float),
            arr_roi,
            arr_period,
            interaction,
        ]
    )
    rank = int(np.linalg.matrix_rank(X))
    dof_resid = int(n - rank)
    if rank < 4 or dof_resid <= 0:
        beta = np.linalg.pinv(X) @ arr_y
        return {
            "valid": False,
            "n_trials": n,
            "dof_resid": float(dof_resid),
            "r2": np.nan,
            "coef": np.asarray(beta, dtype=float).reshape(-1),
            "se": np.full(4, np.nan, dtype=float),
            "t": np.full(4, np.nan, dtype=float),
            "p": np.full(4, np.nan, dtype=float),
        }

    beta = np.linalg.lstsq(X, arr_y, rcond=None)[0]
    y_hat = X @ beta
    resid = arr_y - y_hat
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((arr_y - np.mean(arr_y)) ** 2))
    r2 = float(1.0 - (rss / tss)) if np.isfinite(tss) and tss > 0.0 else np.nan

    xtx_inv = np.linalg.pinv(X.T @ X)
    sigma2 = rss / float(dof_resid) if dof_resid > 0 else np.nan
    cov = xtx_inv * float(sigma2) if np.isfinite(sigma2) else np.full((4, 4), np.nan, dtype=float)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    t_stat = np.full(4, np.nan, dtype=float)
    p_val = np.full(4, np.nan, dtype=float)
    for i in range(4):
        if np.isfinite(se[i]) and float(se[i]) > 0.0 and np.isfinite(beta[i]):
            t_stat[i] = float(beta[i]) / float(se[i])
            p_val[i] = float(2.0 * t.sf(abs(t_stat[i]), df=float(dof_resid)))

    return {
        "valid": True,
        "n_trials": n,
        "dof_resid": float(dof_resid),
        "r2": r2,
        "coef": np.asarray(beta, dtype=float).reshape(-1),
        "se": np.asarray(se, dtype=float).reshape(-1),
        "t": np.asarray(t_stat, dtype=float).reshape(-1),
        "p": np.asarray(p_val, dtype=float).reshape(-1),
    }


def _compute_axis_from_condition_means(
    mean_face_interactive: float,
    mean_face_non_interactive: float,
    mean_object_interactive: float,
    mean_object_non_interactive: float,
) -> dict[str, float]:
    vals = np.asarray(
        [
            mean_face_interactive,
            mean_face_non_interactive,
            mean_object_interactive,
            mean_object_non_interactive,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(vals)):
        return {
            "face_object": np.nan,
            "interactive_state": np.nan,
            "cross_interaction": np.nan,
        }
    fi, fn, oi, on = vals.tolist()
    return {
        "face_object": ((fi + fn) - (oi + on)) / 2.0,
        "interactive_state": ((fi + oi) - (fn + on)) / 2.0,
        "cross_interaction": ((fi - oi) - (fn - on)) / 2.0,
    }


def _build_unit_axis_normalization_table(
    unit_window_df: pd.DataFrame,
    *,
    significance_windows: Sequence[str],
) -> pd.DataFrame:
    if unit_window_df.empty:
        return pd.DataFrame()
    required = {"unit_key", "window_name", *UNIT_WINDOW_MEAN_FR_COLUMNS}
    if not required.issubset(unit_window_df.columns):
        return pd.DataFrame()

    df = unit_window_df.copy()
    df["window_name"] = df["window_name"].astype(str)
    norm_windows = {str(name).strip() for name in significance_windows if str(name).strip()}
    if norm_windows:
        df = df.loc[df["window_name"].astype(str).isin(norm_windows)].copy()
    if df.empty:
        return pd.DataFrame()

    window_rank = {str(name): idx for idx, name in enumerate([str(name) for name in significance_windows if str(name).strip()])}
    meta_cols = [
        "unit_key",
        "date",
        "unit_uuid",
        "region",
        "spike_channel",
        "recorded_agent",
        "recorded_monkey",
        "area",
        "n_sessions",
    ]
    meta_cols = [col for col in meta_cols if col in df.columns]
    rows: list[dict] = []
    for key_vals, grp in df.groupby(meta_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = {col: val for col, val in zip(meta_cols, key_vals)}
        mean_mat = grp.loc[:, list(UNIT_WINDOW_MEAN_FR_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        vals = mean_mat[np.isfinite(mean_mat)]
        if vals.size == 0:
            continue
        min_fr = float(np.min(vals))
        max_fr = float(np.max(vals))
        dynamic_range = float(max_fr - min_fr)
        used_windows = sorted(
            {str(name) for name in grp["window_name"].astype(str).tolist() if str(name).strip()},
            key=lambda name: (int(window_rank.get(str(name), len(window_rank))), str(name)),
        )
        row.update(
            {
                "unit_axis_reference_windows": "|".join(used_windows),
                "unit_fr_min_hz": min_fr,
                "unit_fr_max_hz": max_fr,
                "unit_fr_dynamic_range_hz": dynamic_range,
                "unit_axis_normalization_scale_hz": dynamic_range,
                "unit_axis_normalization_valid": bool(np.isfinite(dynamic_range) and dynamic_range > 0.0),
                "normalized_axis_source": CELL_MEAN_UNIT_RANGE_NORM_AXIS_SOURCE,
                "normalized_axis_method": "unit_dynamic_range_over_significance_windows",
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [col for col in ("date", "region", "unit_uuid") if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _normalize_axis_values_by_scale(
    values: pd.Series | np.ndarray,
    scales: pd.Series | np.ndarray,
) -> np.ndarray:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float).reshape(-1)
    scl = pd.to_numeric(pd.Series(scales), errors="coerce").to_numpy(dtype=float).reshape(-1)
    out = np.full(vals.shape, np.nan, dtype=float)
    valid = np.isfinite(vals) & np.isfinite(scl) & (scl > 0.0)
    out[valid] = vals[valid] / scl[valid]
    near_zero = np.isfinite(vals) & (np.abs(vals) <= 1e-12)
    zero_scale = np.isfinite(scl) & (scl <= 0.0)
    out[near_zero & zero_scale] = 0.0
    out[near_zero & ~np.isfinite(scl)] = 0.0
    return out


def _attach_normalized_axis_values_to_window_summary(
    unit_window_df: pd.DataFrame,
    unit_norm_df: pd.DataFrame,
) -> pd.DataFrame:
    if unit_window_df.empty or unit_norm_df.empty or "unit_key" not in unit_window_df.columns:
        return unit_window_df
    norm_cols = [
        "unit_key",
        "unit_axis_reference_windows",
        "unit_fr_min_hz",
        "unit_fr_max_hz",
        "unit_fr_dynamic_range_hz",
        "unit_axis_normalization_scale_hz",
        "unit_axis_normalization_valid",
        "normalized_axis_source",
        "normalized_axis_method",
    ]
    norm_cols = [col for col in norm_cols if col in unit_norm_df.columns]
    if "unit_key" not in norm_cols:
        norm_cols = ["unit_key", *norm_cols]
    df = unit_window_df.copy()
    df = df.merge(unit_norm_df.loc[:, norm_cols].drop_duplicates(subset=["unit_key"]), on="unit_key", how="left")
    axis_col_map = {
        "axis_face_object_from_means": "axis_face_object_from_means_unit_range_norm",
        "axis_interactive_state_from_means": "axis_interactive_state_from_means_unit_range_norm",
        "axis_cross_interaction_from_means": "axis_cross_interaction_from_means_unit_range_norm",
    }
    for raw_col, norm_col in axis_col_map.items():
        if raw_col not in df.columns:
            continue
        df[norm_col] = _normalize_axis_values_by_scale(
            df[raw_col],
            df["unit_axis_normalization_scale_hz"],
        )
    return df


def _append_normalized_cell_mean_axis_rows(
    unit_axis_df: pd.DataFrame,
    unit_norm_df: pd.DataFrame,
) -> pd.DataFrame:
    if unit_axis_df.empty:
        return unit_axis_df
    df = unit_axis_df.copy()
    if unit_norm_df.empty or "unit_key" not in df.columns or "axis_source" not in df.columns:
        return df

    norm_cols = [
        "unit_key",
        "unit_axis_reference_windows",
        "unit_fr_min_hz",
        "unit_fr_max_hz",
        "unit_fr_dynamic_range_hz",
        "unit_axis_normalization_scale_hz",
        "unit_axis_normalization_valid",
        "normalized_axis_source",
        "normalized_axis_method",
    ]
    norm_cols = [col for col in norm_cols if col in unit_norm_df.columns]
    norm_lookup = unit_norm_df.loc[:, norm_cols].drop_duplicates(subset=["unit_key"])
    df = df.merge(norm_lookup, on="unit_key", how="left")
    if "axis_value_units" not in df.columns:
        df["axis_value_units"] = np.where(
            df["axis_source"].astype(str) == str(CELL_MEAN_AXIS_SOURCE),
            "hz_difference",
            np.where(df["axis_source"].astype(str) == "glm_coef", "glm_coefficient", ""),
        )

    raw_cell_df = df.loc[df["axis_source"].astype(str) == str(CELL_MEAN_AXIS_SOURCE)].copy()
    if raw_cell_df.empty:
        return df
    raw_cell_df["value_signed"] = _normalize_axis_values_by_scale(
        raw_cell_df["value_signed"],
        raw_cell_df["unit_axis_normalization_scale_hz"],
    )
    raw_cell_df["value_abs"] = np.abs(raw_cell_df["value_signed"].to_numpy(dtype=float))
    raw_cell_df["axis_source"] = str(CELL_MEAN_UNIT_RANGE_NORM_AXIS_SOURCE)
    raw_cell_df["axis_value_units"] = "unit_dynamic_range_fraction"
    raw_cell_df = raw_cell_df.loc[raw_cell_df["value_signed"].notna()].copy()
    if raw_cell_df.empty:
        return df
    out = pd.concat([df, raw_cell_df], ignore_index=True, sort=False)
    sort_cols = [
        col
        for col in ("date", "region", "unit_uuid", "window_name", "axis_source", "axis_name")
        if col in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _unit_worker(args):
    unit_key, df_unit, bin_centers_s, settings = args

    windows_ms = _normalize_windows(settings.windows_ms)
    window_masks = _prepare_window_masks(bin_centers_s=bin_centers_s, windows_ms=windows_ms)
    bin_size_s = float(np.mean(np.diff(bin_centers_s))) if bin_centers_s.size > 1 else np.nan
    if not np.isfinite(bin_size_s) or float(bin_size_s) <= 0.0:
        raise ValueError("Unable to infer positive bin size for ROI-vs-period factorial analysis.")
    sigma_bins = _resolve_smoothing_sigma_bins(settings, bin_size_s=bin_size_s)

    row0 = df_unit.iloc[0]
    date = str(row0.get("date"))
    unit_uuid = str(row0.get("unit_uuid"))
    region = _as_optional_str(row0.get("region"))
    spike_channel = _as_optional_str(row0.get("spike_channel"))
    recorded_agent = _as_optional_str(row0.get("recorded_agent"))
    recorded_monkey = _as_optional_str(row0.get("recorded_monkey"))
    area = _as_optional_str(row0.get("area"))
    n_sessions = int(df_unit["session"].nunique()) if "session" in df_unit.columns else 0

    condition_keys = (
        "face_interactive",
        "face_non_interactive",
        "object_interactive",
        "object_non_interactive",
    )
    per_window_data: dict[str, dict] = {
        win_name: {
            "y": [],
            "roi_code": [],
            "period_code": [],
            "condition_values": {key: [] for key in condition_keys},
        }
        for win_name in windows_ms.keys()
    }

    n_bins = int(bin_centers_s.size)
    for row in df_unit.itertuples(index=False):
        cond = _resolve_trial_condition(row, settings)
        if cond is None:
            continue
        roi_label, period_label, roi_code, period_code = cond
        condition_key = f"{roi_label}_{period_label}"
        if condition_key not in condition_keys:
            continue

        counts = np.asarray(getattr(row, "psth_counts"), dtype=float).reshape(-1)
        if counts.size != n_bins:
            continue
        if sigma_bins is not None:
            counts = gaussian_filter1d(counts, sigma=sigma_bins, mode="nearest")
        rates_hz = counts / float(bin_size_s)

        for win_name, mask in window_masks.items():
            response = float(np.mean(rates_hz[mask]))
            data = per_window_data[win_name]
            data["y"].append(response)
            data["roi_code"].append(float(roi_code))
            data["period_code"].append(float(period_code))
            data["condition_values"][condition_key].append(response)

    unit_term_rows: list[dict] = []
    unit_axis_rows: list[dict] = []
    unit_window_rows: list[dict] = []
    for win_name, (start_ms, stop_ms) in windows_ms.items():
        data = per_window_data[win_name]
        values_map = {
            key: np.asarray(data["condition_values"][key], dtype=float).reshape(-1)
            for key in condition_keys
        }
        n_face_int = int(values_map["face_interactive"].size)
        n_face_non = int(values_map["face_non_interactive"].size)
        n_obj_int = int(values_map["object_interactive"].size)
        n_obj_non = int(values_map["object_non_interactive"].size)

        mean_face_int = float(np.mean(values_map["face_interactive"])) if n_face_int > 0 else np.nan
        mean_face_non = float(np.mean(values_map["face_non_interactive"])) if n_face_non > 0 else np.nan
        mean_obj_int = float(np.mean(values_map["object_interactive"])) if n_obj_int > 0 else np.nan
        mean_obj_non = float(np.mean(values_map["object_non_interactive"])) if n_obj_non > 0 else np.nan

        axis_from_means = _compute_axis_from_condition_means(
            mean_face_int,
            mean_face_non,
            mean_obj_int,
            mean_obj_non,
        )
        n_trials_total = int(n_face_int + n_face_non + n_obj_int + n_obj_non)
        meets_min_trials_per_cell = (
            n_face_int >= int(settings.min_trials_per_cell)
            and n_face_non >= int(settings.min_trials_per_cell)
            and n_obj_int >= int(settings.min_trials_per_cell)
            and n_obj_non >= int(settings.min_trials_per_cell)
        )

        glm_fit = _fit_factorial_glm(
            np.asarray(data["y"], dtype=float),
            np.asarray(data["roi_code"], dtype=float),
            np.asarray(data["period_code"], dtype=float),
        )
        glm_testable = bool(meets_min_trials_per_cell and bool(glm_fit.get("valid", False)))
        coef = np.asarray(glm_fit.get("coef", np.full(4, np.nan, dtype=float)), dtype=float).reshape(-1)
        se = np.asarray(glm_fit.get("se", np.full(4, np.nan, dtype=float)), dtype=float).reshape(-1)
        t_stat = np.asarray(glm_fit.get("t", np.full(4, np.nan, dtype=float)), dtype=float).reshape(-1)
        p_val = np.asarray(glm_fit.get("p", np.full(4, np.nan, dtype=float)), dtype=float).reshape(-1)
        if coef.size < 4:
            coef = np.full(4, np.nan, dtype=float)
            se = np.full(4, np.nan, dtype=float)
            t_stat = np.full(4, np.nan, dtype=float)
            p_val = np.full(4, np.nan, dtype=float)

        term_index = {
            "roi_main": 1,
            "period_main": 2,
            "interaction": 3,
        }
        for term_name, axis_name in TERM_TO_AXIS:
            idx = term_index[term_name]
            coeff = float(coef[idx]) if glm_testable else np.nan
            se_term = float(se[idx]) if glm_testable else np.nan
            t_term = float(t_stat[idx]) if glm_testable else np.nan
            p_term = float(p_val[idx]) if glm_testable else np.nan
            sig_raw = bool(np.isfinite(p_term) and p_term < float(settings.alpha))
            unit_term_rows.append(
                {
                    "unit_key": unit_key,
                    "date": date,
                    "unit_uuid": unit_uuid,
                    "region": region,
                    "spike_channel": spike_channel,
                    "recorded_agent": recorded_agent,
                    "recorded_monkey": recorded_monkey,
                    "area": area,
                    "n_sessions": n_sessions,
                    "window_name": win_name,
                    "window_start_ms": float(start_ms),
                    "window_stop_ms": float(stop_ms),
                    "term": term_name,
                    "axis_name": axis_name,
                    "coefficient": coeff,
                    "coefficient_abs": abs(coeff) if np.isfinite(coeff) else np.nan,
                    "standard_error": se_term,
                    "t_statistic": t_term,
                    "p_value": p_term,
                    "significant_raw": bool(sig_raw),
                    "alpha": float(settings.alpha),
                    "glm_testable": bool(glm_testable),
                    "n_trials_total": n_trials_total,
                    "n_trials_face_interactive": n_face_int,
                    "n_trials_face_non_interactive": n_face_non,
                    "n_trials_object_interactive": n_obj_int,
                    "n_trials_object_non_interactive": n_obj_non,
                    "meets_min_trials_per_cell": bool(meets_min_trials_per_cell),
                    "glm_r2": float(glm_fit.get("r2", np.nan)),
                    "glm_dof_resid": float(glm_fit.get("dof_resid", np.nan)),
                }
            )
            unit_axis_rows.append(
                {
                    "unit_key": unit_key,
                    "date": date,
                    "unit_uuid": unit_uuid,
                    "region": region,
                    "spike_channel": spike_channel,
                    "recorded_agent": recorded_agent,
                    "recorded_monkey": recorded_monkey,
                    "area": area,
                    "n_sessions": n_sessions,
                    "window_name": win_name,
                    "window_start_ms": float(start_ms),
                    "window_stop_ms": float(stop_ms),
                    "axis_name": axis_name,
                    "axis_source": "glm_coef",
                    "axis_value_units": "glm_coefficient",
                    "value_signed": coeff,
                    "value_abs": abs(coeff) if np.isfinite(coeff) else np.nan,
                    "p_value": p_term,
                    "significant_raw": bool(sig_raw),
                    "glm_testable": bool(glm_testable),
                    "n_trials_total": n_trials_total,
                }
            )
            mean_axis_val = float(axis_from_means.get(axis_name, np.nan))
            unit_axis_rows.append(
                {
                    "unit_key": unit_key,
                    "date": date,
                    "unit_uuid": unit_uuid,
                    "region": region,
                    "spike_channel": spike_channel,
                    "recorded_agent": recorded_agent,
                    "recorded_monkey": recorded_monkey,
                    "area": area,
                    "n_sessions": n_sessions,
                    "window_name": win_name,
                    "window_start_ms": float(start_ms),
                    "window_stop_ms": float(stop_ms),
                    "axis_name": axis_name,
                    "axis_source": str(CELL_MEAN_AXIS_SOURCE),
                    "axis_value_units": "hz_difference",
                    "value_signed": mean_axis_val,
                    "value_abs": abs(mean_axis_val) if np.isfinite(mean_axis_val) else np.nan,
                    "p_value": np.nan,
                    "significant_raw": np.nan,
                    "glm_testable": bool(glm_testable),
                    "n_trials_total": n_trials_total,
                }
            )

        unit_window_rows.append(
            {
                "unit_key": unit_key,
                "date": date,
                "unit_uuid": unit_uuid,
                "region": region,
                "spike_channel": spike_channel,
                "recorded_agent": recorded_agent,
                "recorded_monkey": recorded_monkey,
                "area": area,
                "n_sessions": n_sessions,
                "window_name": win_name,
                "window_start_ms": float(start_ms),
                "window_stop_ms": float(stop_ms),
                "n_trials_face_interactive": n_face_int,
                "n_trials_face_non_interactive": n_face_non,
                "n_trials_object_interactive": n_obj_int,
                "n_trials_object_non_interactive": n_obj_non,
                "n_trials_total": n_trials_total,
                "mean_fr_face_interactive_hz": mean_face_int,
                "mean_fr_face_non_interactive_hz": mean_face_non,
                "mean_fr_object_interactive_hz": mean_obj_int,
                "mean_fr_object_non_interactive_hz": mean_obj_non,
                "axis_face_object_from_means": float(axis_from_means["face_object"]),
                "axis_interactive_state_from_means": float(axis_from_means["interactive_state"]),
                "axis_cross_interaction_from_means": float(axis_from_means["cross_interaction"]),
                "meets_min_trials_per_cell": bool(meets_min_trials_per_cell),
                "glm_testable": bool(glm_testable),
                "glm_r2": float(glm_fit.get("r2", np.nan)),
                "glm_dof_resid": float(glm_fit.get("dof_resid", np.nan)),
            }
        )
    return unit_term_rows, unit_axis_rows, unit_window_rows


def _date_worker(args):
    _, df_date, bin_centers_s, settings = args
    term_rows_all: list[dict] = []
    axis_rows_all: list[dict] = []
    unit_window_rows_all: list[dict] = []
    grouped = df_date.groupby(["date", "unit_uuid"], sort=True, dropna=False)
    for (date, unit_uuid), df_unit in grouped:
        unit_key = f"{date}|{unit_uuid}"
        term_rows, axis_rows, unit_window_rows = _unit_worker(
            (unit_key, df_unit.copy(), bin_centers_s, settings)
        )
        term_rows_all.extend(term_rows)
        axis_rows_all.extend(axis_rows)
        unit_window_rows_all.extend(unit_window_rows)
    return term_rows_all, axis_rows_all, unit_window_rows_all


def _build_region_fraction_tables(
    unit_axis_significance_df: pd.DataFrame,
    *,
    settings: FixationROIVsPeriodFactorialSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if unit_axis_significance_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"region", "axis_name", "unit_key", "is_significant_axis"}
    if not required.issubset(unit_axis_significance_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    df = unit_axis_significance_df.copy()
    df["region"] = df["region"].fillna("unknown").astype(str)
    df["is_significant_axis"] = df["is_significant_axis"].map(bool)

    frac_rows: list[dict] = []
    for (region, axis_name), grp in df.groupby(["region", "axis_name"], dropna=False):
        n_total = int(grp["unit_key"].astype(str).nunique())
        n_sig = int(grp.loc[grp["is_significant_axis"], "unit_key"].astype(str).nunique())
        frac_rows.append(
            {
                "region": str(region),
                "axis_name": str(axis_name),
                "selective_units": n_sig,
                "total_units": n_total,
                "fraction_selective": (float(n_sig) / float(n_total)) if n_total > 0 else np.nan,
            }
        )
    fraction_df = pd.DataFrame(frac_rows).sort_values(
        ["axis_name", "region"]
    ).reset_index(drop=True)

    pair_rows: list[dict] = []
    for axis_name, grp in fraction_df.groupby("axis_name", dropna=False):
        by_region = {
            str(row.region): (int(row.selective_units), int(row.total_units))
            for row in grp.itertuples(index=False)
        }
        eligible_regions = sorted(
            [
                region_name
                for region_name, (_, n_total) in by_region.items()
                if n_total >= int(settings.min_units_per_region)
            ]
        )
        for region_a, region_b in combinations(eligible_regions, 2):
            n_sig_a, n_total_a = by_region[region_a]
            n_sig_b, n_total_b = by_region[region_b]
            table = np.asarray(
                [
                    [n_sig_a, max(n_total_a - n_sig_a, 0)],
                    [n_sig_b, max(n_total_b - n_sig_b, 0)],
                ],
                dtype=int,
            )
            odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
            frac_a = (float(n_sig_a) / float(n_total_a)) if n_total_a > 0 else np.nan
            frac_b = (float(n_sig_b) / float(n_total_b)) if n_total_b > 0 else np.nan
            pair_rows.append(
                {
                    "axis_name": str(axis_name),
                    "region_a": region_a,
                    "region_b": region_b,
                    "n_sig_a": int(n_sig_a),
                    "n_total_a": int(n_total_a),
                    "fraction_a": frac_a,
                    "n_sig_b": int(n_sig_b),
                    "n_total_b": int(n_total_b),
                    "fraction_b": frac_b,
                    "fraction_diff_a_minus_b": (
                        float(frac_a - frac_b) if np.isfinite(frac_a) and np.isfinite(frac_b) else np.nan
                    ),
                    "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )
    pairwise_df = pd.DataFrame(pair_rows)
    if not pairwise_df.empty:
        pairwise_df = apply_adjusted_pvalues(
            pairwise_df,
            p_col="p_value",
            out_col="p_value_adjusted",
            method=settings.pvalue_correction,
            group_cols=("axis_name",),
        )
        pairwise_df["significant_adjusted"] = (
            pairwise_df["p_value_adjusted"].to_numpy(dtype=float) < float(settings.alpha)
        )
        pairwise_df = pairwise_df.sort_values(
            ["axis_name", "region_a", "region_b"]
        ).reset_index(drop=True)

    return fraction_df, pairwise_df


def _build_unit_axis_significance_table(
    unit_term_df: pd.DataFrame,
    *,
    settings: FixationROIVsPeriodFactorialSettings,
) -> pd.DataFrame:
    if unit_term_df.empty:
        return pd.DataFrame()
    sig_mode = _resolve_unit_significance_mode(settings.unit_significance_mode)
    sig_col = "significant_raw" if sig_mode == "raw" else "significant_within_unit"
    required = {"unit_key", "axis_name", "window_name", "glm_testable", "counts_toward_significance", sig_col}
    if not required.issubset(unit_term_df.columns):
        return pd.DataFrame()

    df = unit_term_df.copy()
    df["region"] = df["region"].fillna("unknown").astype(str)
    df = df.loc[df["counts_toward_significance"]].copy()
    df = df.loc[df["glm_testable"].map(bool)].copy()
    if df.empty:
        return pd.DataFrame()
    df[sig_col] = df[sig_col].map(bool)

    meta_cols = [
        "unit_key",
        "date",
        "unit_uuid",
        "region",
        "spike_channel",
        "recorded_agent",
        "recorded_monkey",
        "area",
        "axis_name",
    ]
    meta_cols = [col for col in meta_cols if col in df.columns]
    rows: list[dict] = []
    for key_vals, grp in df.groupby(meta_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = {col: val for col, val in zip(meta_cols, key_vals)}
        sig_mask = grp[sig_col].to_numpy(dtype=bool).reshape(-1)
        n_sig = int(np.sum(sig_mask))
        n_testable = int(len(grp))
        sig_windows = (
            grp.loc[sig_mask, "window_name"].astype(str).dropna().tolist()
            if "window_name" in grp.columns
            else []
        )
        row.update(
            {
                "significance_mode": sig_mode,
                "significance_column": sig_col,
                "n_testable_windows": n_testable,
                "n_significant_windows": n_sig,
                "is_significant_axis": bool(n_sig > 0),
                "significant_windows": "|".join(sorted(set(sig_windows))),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [col for col in ("date", "region", "unit_uuid", "axis_name") if col in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _build_unit_axis_collapsed_magnitude_table(
    unit_axis_df: pd.DataFrame,
    unit_axis_significance_df: pd.DataFrame,
    *,
    settings: FixationROIVsPeriodFactorialSettings,
) -> pd.DataFrame:
    if unit_axis_df.empty:
        return pd.DataFrame()
    required_axis = {"unit_key", "axis_name", "axis_source", "window_name", "value_signed", "counts_toward_significance"}
    if not required_axis.issubset(unit_axis_df.columns):
        return pd.DataFrame()

    mode = _resolve_axis_comparison_mode(settings.axis_comparison_mode)
    collapsed_window_name = _collapsed_window_name_for_mode(mode)
    df = unit_axis_df.copy()
    df = df.loc[df["counts_toward_significance"]].copy()
    if df.empty:
        return pd.DataFrame()
    df["region"] = df["region"].fillna("unknown").astype(str)
    df["window_name"] = df["window_name"].astype(str)
    df["value_signed"] = pd.to_numeric(df["value_signed"], errors="coerce")
    if "value_abs" in df.columns:
        df["value_abs"] = pd.to_numeric(df["value_abs"], errors="coerce")
    else:
        df["value_abs"] = np.abs(df["value_signed"].to_numpy(dtype=float))

    sig_cols = {
        "unit_key",
        "axis_name",
        "significance_mode",
        "significance_column",
        "n_testable_windows",
        "n_significant_windows",
        "is_significant_axis",
        "significant_windows",
    }
    if not unit_axis_significance_df.empty and sig_cols.issubset(unit_axis_significance_df.columns):
        sig_df = unit_axis_significance_df.loc[:, sorted(sig_cols)].copy()
        sig_df["unit_key"] = sig_df["unit_key"].astype(str)
        sig_df["axis_name"] = sig_df["axis_name"].astype(str)
        df = df.merge(sig_df, on=["unit_key", "axis_name"], how="left")
    else:
        df["significance_mode"] = _resolve_unit_significance_mode(settings.unit_significance_mode)
        df["significance_column"] = (
            "significant_raw"
            if str(settings.unit_significance_mode).strip().lower() == "raw"
            else "significant_within_unit"
        )
        df["n_testable_windows"] = 0
        df["n_significant_windows"] = 0
        df["is_significant_axis"] = False
        df["significant_windows"] = ""

    window_rank: dict[str, int] = {}
    ordered_windows: list[str] = []
    ordered_windows.extend(str(name) for name in settings.significance_windows if str(name).strip())
    ordered_windows.extend(
        str(name)
        for name in settings.windows_ms.keys()
        if str(name).strip() and str(name) not in set(ordered_windows)
    )
    for idx, name in enumerate(ordered_windows):
        window_rank[str(name)] = idx

    rows: list[dict] = []
    group_cols = [
        "unit_key",
        "date",
        "unit_uuid",
        "region",
        "spike_channel",
        "recorded_agent",
        "recorded_monkey",
        "area",
        "n_sessions",
        "axis_name",
        "axis_source",
        "axis_value_units",
        "unit_axis_reference_windows",
        "unit_fr_min_hz",
        "unit_fr_max_hz",
        "unit_fr_dynamic_range_hz",
        "unit_axis_normalization_scale_hz",
        "unit_axis_normalization_valid",
        "normalized_axis_source",
        "normalized_axis_method",
    ]
    group_cols = [col for col in group_cols if col in df.columns]
    for key_vals, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = {col: val for col, val in zip(group_cols, key_vals)}
        grp_valid = grp.loc[grp["value_abs"].notna()].copy()
        if grp_valid.empty:
            continue
        grp_valid["window_rank"] = grp_valid["window_name"].map(
            lambda name: int(window_rank.get(str(name), len(window_rank)))
        )
        used_windows = sorted(
            {str(name) for name in grp_valid["window_name"].astype(str).tolist() if str(name).strip()},
            key=lambda name: (int(window_rank.get(str(name), len(window_rank))), str(name)),
        )
        is_sig_axis_raw = grp.iloc[0].get("is_significant_axis", False)
        n_sig_windows_raw = grp.iloc[0].get("n_significant_windows", 0)
        n_testable_windows_raw = grp.iloc[0].get("n_testable_windows", 0)
        sig_windows = {
            token.strip()
            for token in str(grp.iloc[0].get("significant_windows", "") or "").split("|")
            if str(token).strip()
        }
        row.update(
            {
                "axis_comparison_mode": str(mode),
                "window_name": collapsed_window_name,
                "window_start_ms": np.nan,
                "window_stop_ms": np.nan,
                "n_windows_used": int(grp_valid.shape[0]),
                "windows_used": "|".join(used_windows),
                "is_significant_axis": bool(is_sig_axis_raw) if pd.notna(is_sig_axis_raw) else False,
                "n_significant_windows": (
                    int(n_sig_windows_raw) if pd.notna(n_sig_windows_raw) else 0
                ),
                "n_testable_windows": (
                    int(n_testable_windows_raw) if pd.notna(n_testable_windows_raw) else 0
                ),
                "significant_windows": "|".join(sorted(sig_windows)),
                "significance_mode": str(grp.iloc[0].get("significance_mode", "")),
                "significance_column": str(grp.iloc[0].get("significance_column", "")),
            }
        )
        if mode == "max_abs_across_windows":
            grp_valid = grp_valid.sort_values(
                ["value_abs", "window_rank", "window_name"],
                ascending=[False, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            selected = grp_valid.iloc[0]
            selected_window_name = str(selected.get("window_name", ""))
            selected_glm_testable_raw = selected.get("glm_testable", False)
            selected_trials_total_raw = selected.get("n_trials_total", 0)
            row.update(
                {
                    "collapsed_value_method": "max_abs_over_windows",
                    "value_signed": float(selected["value_signed"]),
                    "value_abs": float(selected["value_abs"]),
                    "selected_window_name": selected_window_name,
                    "selected_window_start_ms": float(selected.get("window_start_ms", np.nan)),
                    "selected_window_stop_ms": float(selected.get("window_stop_ms", np.nan)),
                    "selected_window_p_value": float(selected.get("p_value", np.nan)),
                    "selected_window_glm_testable": (
                        bool(selected_glm_testable_raw) if pd.notna(selected_glm_testable_raw) else False
                    ),
                    "selected_window_n_trials_total": (
                        int(selected_trials_total_raw) if pd.notna(selected_trials_total_raw) else 0
                    ),
                    "selected_window_is_significant": bool(selected_window_name in sig_windows),
                }
            )
        else:
            signed_vals = grp_valid["value_signed"].to_numpy(dtype=float).reshape(-1)
            abs_vals = grp_valid["value_abs"].to_numpy(dtype=float).reshape(-1)
            row.update(
                {
                    "collapsed_value_method": "mean_over_windows",
                    "value_signed": float(np.mean(signed_vals)),
                    "value_abs": float(np.mean(abs_vals)),
                    "selected_window_name": np.nan,
                    "selected_window_start_ms": np.nan,
                    "selected_window_stop_ms": np.nan,
                    "selected_window_p_value": np.nan,
                    "selected_window_glm_testable": np.nan,
                    "selected_window_n_trials_total": np.nan,
                    "selected_window_is_significant": np.nan,
                }
            )
        row.update(
            {
                "value_abs_std_over_windows": (
                    float(np.std(grp_valid["value_abs"].to_numpy(dtype=float), ddof=1))
                    if grp_valid.shape[0] > 1
                    else np.nan
                ),
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [col for col in ("date", "region", "unit_uuid", "axis_source", "axis_name") if col in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _build_axis_magnitude_input_table(
    unit_axis_df: pd.DataFrame,
    *,
    settings: FixationROIVsPeriodFactorialSettings,
    unit_axis_collapsed_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    mode = _resolve_axis_comparison_mode(settings.axis_comparison_mode)
    if mode != "split_by_window" and unit_axis_collapsed_df is not None and not unit_axis_collapsed_df.empty:
        collapsed_df = unit_axis_collapsed_df.copy()
        required_collapsed = {
            "region",
            "window_name",
            "axis_name",
            "axis_source",
            "unit_key",
        }
        if required_collapsed.issubset(collapsed_df.columns):
            collapsed_df["region"] = collapsed_df["region"].fillna("unknown").astype(str)
            collapsed_df = collapsed_df.loc[
                collapsed_df["axis_source"].astype(str).isin(set(CELL_MEAN_MAGNITUDE_SOURCES))
            ].copy()
            if "axis_comparison_mode" in collapsed_df.columns:
                collapsed_df = collapsed_df.loc[
                    collapsed_df["axis_comparison_mode"].astype(str) == str(mode)
                ].copy()
            else:
                collapsed_df["axis_comparison_mode"] = mode
            if collapsed_df.empty:
                return pd.DataFrame()
            if "value_abs" in collapsed_df.columns:
                collapsed_df["value_abs_norm"] = pd.to_numeric(collapsed_df["value_abs"], errors="coerce")
            else:
                collapsed_df["value_abs_norm"] = np.abs(
                    pd.to_numeric(collapsed_df["value_signed"], errors="coerce")
                )
            collapsed_df = collapsed_df.loc[collapsed_df["value_abs_norm"].notna()].copy()
            if collapsed_df.empty:
                return pd.DataFrame()
            sort_cols = [
                col
                for col in ("date", "region", "unit_uuid", "window_name", "axis_source", "axis_name")
                if col in collapsed_df.columns
            ]
            if sort_cols:
                collapsed_df = collapsed_df.sort_values(sort_cols).reset_index(drop=True)
            return collapsed_df

    if unit_axis_df.empty:
        return pd.DataFrame()
    required = {
        "region",
        "window_name",
        "axis_name",
        "axis_source",
        "unit_key",
        "value_signed",
        "counts_toward_significance",
    }
    if not required.issubset(unit_axis_df.columns):
        return pd.DataFrame()

    df = unit_axis_df.copy()
    df["region"] = df["region"].fillna("unknown").astype(str)
    df = df.loc[df["counts_toward_significance"]].copy()
    df = df.loc[df["axis_source"].astype(str).isin(set(CELL_MEAN_MAGNITUDE_SOURCES))].copy()
    if df.empty:
        return pd.DataFrame()
    df["value_abs_norm"] = np.abs(df["value_signed"].to_numpy(dtype=float))

    if mode == "split_by_window":
        df["window_name"] = df["window_name"].astype(str)
        df["axis_comparison_mode"] = mode
        sort_cols = [
            col
            for col in ("date", "region", "unit_uuid", "window_name", "axis_source", "axis_name")
            if col in df.columns
        ]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        return df

    group_cols = [
        "unit_key",
        "date",
        "unit_uuid",
        "region",
        "spike_channel",
        "recorded_agent",
        "recorded_monkey",
        "area",
        "n_sessions",
        "axis_name",
        "axis_source",
    ]
    group_cols = [col for col in group_cols if col in df.columns]
    df["window_name"] = df["window_name"].astype(str)
    window_rank = {
        str(name): idx
        for idx, name in enumerate(
            [str(name) for name in settings.significance_windows if str(name).strip()]
        )
    }
    rows: list[dict] = []
    for key_vals, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = {col: val for col, val in zip(group_cols, key_vals)}
        vals = grp["value_abs_norm"].to_numpy(dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        windows_used = (
            grp["window_name"].astype(str).dropna().tolist()
            if "window_name" in grp.columns
            else []
        )
        if mode == "max_abs_across_windows":
            grp = grp.sort_values(
                by=["value_abs_norm", "window_name"],
                ascending=[False, True],
                kind="mergesort",
            ).copy()
            grp["window_rank"] = grp["window_name"].map(
                lambda name: int(window_rank.get(str(name), len(window_rank)))
            )
            grp = grp.sort_values(
                by=["value_abs_norm", "window_rank", "window_name"],
                ascending=[False, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            selected = grp.iloc[0]
            row.update(
                {
                    "window_name": _collapsed_window_name_for_mode(mode),
                    "axis_comparison_mode": mode,
                    "n_windows_used": int(vals.size),
                    "windows_used": "|".join(sorted(set(windows_used))),
                    "value_abs_norm": float(selected["value_abs_norm"]),
                    "selected_window_name": str(selected.get("window_name", "")),
                }
            )
        else:
            row.update(
                {
                    "window_name": _collapsed_window_name_for_mode(mode),
                    "axis_comparison_mode": mode,
                    "n_windows_averaged": int(vals.size),
                    "windows_used": "|".join(sorted(set(windows_used))),
                    "value_abs_norm": float(np.mean(vals)),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [
        col
        for col in ("date", "region", "unit_uuid", "window_name", "axis_source", "axis_name")
        if col in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _build_region_axis_tables(
    unit_axis_magnitude_df: pd.DataFrame,
    *,
    settings: FixationROIVsPeriodFactorialSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if unit_axis_magnitude_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required = {
        "region",
        "window_name",
        "axis_name",
        "axis_source",
        "unit_key",
        "value_abs_norm",
        "axis_comparison_mode",
    }
    if not required.issubset(unit_axis_magnitude_df.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = unit_axis_magnitude_df.copy()
    df["region"] = df["region"].fillna("unknown").astype(str)

    summary_rows: list[dict] = []
    for (mode, axis_source, region, window_name, axis_name), grp in df.groupby(
        ["axis_comparison_mode", "axis_source", "region", "window_name", "axis_name"],
        dropna=False,
    ):
        vals = grp["value_abs_norm"].to_numpy(dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        summary_rows.append(
            {
                "axis_comparison_mode": str(mode),
                "axis_source": str(axis_source),
                "region": str(region),
                "window_name": str(window_name),
                "axis_name": str(axis_name),
                "n_units": int(vals.size),
                "mean_abs": float(np.mean(vals)),
                "median_abs": float(np.median(vals)),
                "std_abs": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["axis_comparison_mode", "window_name", "axis_source", "region", "axis_name"]
        ).reset_index(drop=True)

    pairwise_rows: list[dict] = []
    for (mode, axis_source, window_name, axis_name), grp in df.groupby(
        ["axis_comparison_mode", "axis_source", "window_name", "axis_name"],
        dropna=False,
    ):
        by_region_vals = {}
        for region, g_region in grp.groupby("region", dropna=False):
            vals = g_region["value_abs_norm"].to_numpy(dtype=float).reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size >= int(settings.min_units_per_region):
                by_region_vals[str(region)] = vals
        regions = sorted(by_region_vals.keys())
        for region_a, region_b in combinations(regions, 2):
            arr_a = by_region_vals[region_a]
            arr_b = by_region_vals[region_b]
            stat, p_value = safe_welch_ttest(arr_a, arr_b)
            pairwise_rows.append(
                {
                    "axis_comparison_mode": str(mode),
                    "axis_source": str(axis_source),
                    "window_name": str(window_name),
                    "axis_name": str(axis_name),
                    "test_name": "welch_ttest",
                    "region_a": region_a,
                    "region_b": region_b,
                    "n_units_a": int(arr_a.size),
                    "n_units_b": int(arr_b.size),
                    "mean_a": float(np.mean(arr_a)),
                    "mean_b": float(np.mean(arr_b)),
                    "median_a": float(np.median(arr_a)),
                    "median_b": float(np.median(arr_b)),
                    "delta_mean_a_minus_b": float(np.mean(arr_a) - np.mean(arr_b)),
                    "delta_median_a_minus_b": float(np.median(arr_a) - np.median(arr_b)),
                    "statistic": stat,
                    "p_value": p_value,
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df = apply_adjusted_pvalues(
            pairwise_df,
            p_col="p_value",
            out_col="p_value_adjusted",
            method=settings.pvalue_correction,
            group_cols=("axis_comparison_mode", "axis_source", "window_name", "axis_name"),
        )
        pairwise_df["significant_adjusted"] = (
            pairwise_df["p_value_adjusted"].to_numpy(dtype=float) < float(settings.alpha)
        )
        pairwise_df = pairwise_df.sort_values(
            ["axis_comparison_mode", "window_name", "axis_source", "axis_name", "region_a", "region_b"]
        ).reset_index(drop=True)

    within_rows: list[dict] = []
    for (mode, axis_source, region, window_name), grp in df.groupby(
        ["axis_comparison_mode", "axis_source", "region", "window_name"], dropna=False
    ):
        pivot = grp.pivot_table(
            index="unit_key",
            columns="axis_name",
            values="value_abs_norm",
            aggfunc="mean",
        ).reindex(columns=list(AXIS_ORDER))
        for axis_a, axis_b in combinations(AXIS_ORDER, 2):
            if axis_a not in pivot.columns or axis_b not in pivot.columns:
                continue
            pair_mat = pivot.loc[:, [axis_a, axis_b]].dropna().copy()
            if pair_mat.shape[0] < 2:
                continue
            arr_a = pair_mat[axis_a].to_numpy(dtype=float)
            arr_b = pair_mat[axis_b].to_numpy(dtype=float)
            stat, p_value = safe_welch_ttest(arr_a, arr_b)
            within_rows.append(
                {
                    "axis_comparison_mode": str(mode),
                    "axis_source": str(axis_source),
                    "region": str(region),
                    "window_name": str(window_name),
                    "axis_a": str(axis_a),
                    "axis_b": str(axis_b),
                    "test_name": "welch_ttest",
                    "n_units_paired": int(pair_mat.shape[0]),
                    "mean_a": float(np.mean(arr_a)),
                    "mean_b": float(np.mean(arr_b)),
                    "median_a": float(np.median(arr_a)),
                    "median_b": float(np.median(arr_b)),
                    "delta_mean_a_minus_b": float(np.mean(arr_a) - np.mean(arr_b)),
                    "delta_median_a_minus_b": float(np.median(arr_a) - np.median(arr_b)),
                    "statistic": stat,
                    "p_value": p_value,
                }
            )
    within_df = pd.DataFrame(within_rows)
    if not within_df.empty:
        within_df = apply_adjusted_pvalues(
            within_df,
            p_col="p_value",
            out_col="p_value_adjusted",
            method=settings.pvalue_correction,
            group_cols=("axis_comparison_mode", "axis_source", "region", "window_name"),
        )
        within_df["significant_adjusted"] = (
            within_df["p_value_adjusted"].to_numpy(dtype=float) < float(settings.alpha)
        )
        within_df = within_df.sort_values(
            ["axis_comparison_mode", "window_name", "axis_source", "region", "axis_a", "axis_b"]
        ).reset_index(drop=True)

    return summary_df, pairwise_df, within_df


def run_fixation_roi_vs_period_factorial_analysis(
    settings: FixationROIVsPeriodFactorialSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    unit_uuids: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    windows: Optional[Sequence[str]] = None,
) -> dict:
    """Run ROI-vs-period factorial analysis from trial-level fixation PSTHs."""
    windows_ms = _normalize_windows(settings.windows_ms)
    if windows is not None:
        allowed_windows = {str(name) for name in windows if str(name).strip()}
        windows_ms = {
            name: bounds
            for name, bounds in windows_ms.items()
            if str(name) in allowed_windows
        }
        if not windows_ms:
            print("[analysis] no analysis windows remain after --window filtering")
            return {
                "unit_terms": pd.DataFrame(),
                "unit_axis_values": pd.DataFrame(),
                "unit_axis_collapsed": pd.DataFrame(),
                "unit_axis_significance": pd.DataFrame(),
                "unit_window_summary": pd.DataFrame(),
                "region_significant_fractions": pd.DataFrame(),
                "region_significant_fraction_pairwise": pd.DataFrame(),
                "region_significant_fraction_within_region": pd.DataFrame(),
                "region_axis_summary": pd.DataFrame(),
                "region_axis_pairwise": pd.DataFrame(),
                "region_axis_within_region": pd.DataFrame(),
                "region_axis_friedman": pd.DataFrame(),
            }
    significance_windows = _normalize_significance_windows(
        settings.significance_windows,
        available_windows=tuple(windows_ms.keys()),
    )
    settings.windows_ms = windows_ms
    settings.significance_windows = significance_windows
    settings.unit_significance_mode = _resolve_unit_significance_mode(settings.unit_significance_mode)
    settings.axis_comparison_mode = _resolve_axis_comparison_mode(settings.axis_comparison_mode)
    settings.parallelization_scope = _resolve_parallelization_scope(settings.parallelization_scope)

    trial_df, bin_centers_s = _load_trial_table(settings, dates=dates, sessions=sessions)
    if trial_df.empty or "unit_uuid" not in trial_df.columns:
        print("[analysis] no trial PSTH rows found for ROI-vs-period factorial analysis")
        return {
            "unit_terms": pd.DataFrame(),
            "unit_axis_values": pd.DataFrame(),
            "unit_axis_collapsed": pd.DataFrame(),
            "unit_axis_significance": pd.DataFrame(),
            "unit_window_summary": pd.DataFrame(),
            "region_significant_fractions": pd.DataFrame(),
            "region_significant_fraction_pairwise": pd.DataFrame(),
            "region_significant_fraction_within_region": pd.DataFrame(),
            "region_axis_summary": pd.DataFrame(),
            "region_axis_pairwise": pd.DataFrame(),
            "region_axis_within_region": pd.DataFrame(),
            "region_axis_friedman": pd.DataFrame(),
        }

    if unit_uuids is not None:
        allowed_units = {str(unit) for unit in unit_uuids}
        trial_df = trial_df.loc[trial_df["unit_uuid"].astype(str).isin(allowed_units)].copy()
    if regions is not None and "region" in trial_df.columns:
        allowed_regions = {str(region) for region in regions}
        trial_df = trial_df.loc[trial_df["region"].astype(str).isin(allowed_regions)].copy()
    if trial_df.empty:
        print("[analysis] no matching rows remain after unit/region filtering")
        return {
            "unit_terms": pd.DataFrame(),
            "unit_axis_values": pd.DataFrame(),
            "unit_axis_collapsed": pd.DataFrame(),
            "unit_axis_significance": pd.DataFrame(),
            "unit_window_summary": pd.DataFrame(),
            "region_significant_fractions": pd.DataFrame(),
            "region_significant_fraction_pairwise": pd.DataFrame(),
            "region_significant_fraction_within_region": pd.DataFrame(),
            "region_axis_summary": pd.DataFrame(),
            "region_axis_pairwise": pd.DataFrame(),
            "region_axis_within_region": pd.DataFrame(),
            "region_axis_friedman": pd.DataFrame(),
        }

    if "date" not in trial_df.columns:
        trial_df["date"] = "unknown"

    term_rows_all: list[dict] = []
    axis_rows_all: list[dict] = []
    unit_window_rows_all: list[dict] = []
    if str(settings.parallelization_scope) == "date":
        date_tasks = []
        grouped_dates = trial_df.groupby(["date"], sort=True, dropna=False)
        for date_tuple, df_date in grouped_dates:
            date_value = str(date_tuple[0]) if isinstance(date_tuple, tuple) else str(date_tuple)
            date_tasks.append((date_value, df_date.copy(), bin_centers_s, settings))
        if settings.test_single and date_tasks:
            date_tasks = [random.choice(date_tasks)]
        results = run_tasks(
            _date_worker,
            date_tasks,
            desc="ROI-vs-period factorial (date-sharded)",
            unit="date",
            use_parallel=settings.use_parallel,
            max_procs=settings.max_procs,
        )
    else:
        unit_tasks = []
        grouped = trial_df.groupby(["date", "unit_uuid"], sort=True, dropna=False)
        for (date, unit_uuid), df_unit in grouped:
            unit_key = f"{date}|{unit_uuid}"
            unit_tasks.append((unit_key, df_unit.copy(), bin_centers_s, settings))
        if settings.test_single and unit_tasks:
            unit_tasks = [random.choice(unit_tasks)]
        results = run_tasks(
            _unit_worker,
            unit_tasks,
            desc="ROI-vs-period factorial (unit-sharded)",
            unit="unit",
            use_parallel=settings.use_parallel,
            max_procs=settings.max_procs,
        )
    for term_rows, axis_rows, unit_window_rows in results:
        term_rows_all.extend(term_rows)
        axis_rows_all.extend(axis_rows)
        unit_window_rows_all.extend(unit_window_rows)

    unit_term_df = pd.DataFrame(term_rows_all)
    unit_axis_df = pd.DataFrame(axis_rows_all)
    unit_window_df = pd.DataFrame(unit_window_rows_all)

    if not unit_term_df.empty:
        unit_term_df["counts_toward_significance"] = unit_term_df["window_name"].astype(str).isin(
            set(str(name) for name in significance_windows)
        )
        # Optional per-window correction across terms (kept for auditability).
        unit_term_df = apply_adjusted_pvalues(
            unit_term_df,
            p_col="p_value",
            out_col="p_value_within_unit_window_adjusted",
            method=settings.pvalue_correction,
            group_cols=("unit_key", "window_name"),
        )
        # Unit-level selectivity call uses correction across significance windows
        # for each term independently (roi_main / period_main / interaction).
        unit_term_df["p_value_within_unit_term_adjusted"] = np.nan
        for _, idx in unit_term_df.groupby(["unit_key", "term"], dropna=False).groups.items():
            idx_list = list(idx)
            counts_mask = (
                unit_term_df.loc[idx_list, "counts_toward_significance"]
                .to_numpy(dtype=bool)
                .reshape(-1)
            )
            if not np.any(counts_mask):
                continue
            pvals = unit_term_df.loc[idx_list, "p_value"].to_numpy(dtype=float).reshape(-1)
            adj_selected = adjust_pvalues(
                pvals[counts_mask],
                settings.pvalue_correction,
            )
            adj_full = np.full(pvals.shape, np.nan, dtype=float)
            adj_full[counts_mask] = adj_selected
            unit_term_df.loc[idx_list, "p_value_within_unit_term_adjusted"] = adj_full
        unit_term_df["significant_within_unit"] = (
            unit_term_df["p_value_within_unit_term_adjusted"].to_numpy(dtype=float) < float(settings.alpha)
        )
        unit_term_df["significant_within_unit_window"] = (
            unit_term_df["p_value_within_unit_window_adjusted"].to_numpy(dtype=float) < float(settings.alpha)
        )
        unit_term_df = unit_term_df.sort_values(
            ["date", "region", "unit_uuid", "window_name", "term"]
        ).reset_index(drop=True)

    if not unit_axis_df.empty:
        unit_norm_df = _build_unit_axis_normalization_table(
            unit_window_df,
            significance_windows=significance_windows,
        )
        unit_axis_df = _append_normalized_cell_mean_axis_rows(
            unit_axis_df,
            unit_norm_df,
        )
        unit_axis_df["counts_toward_significance"] = unit_axis_df["window_name"].astype(str).isin(
            set(str(name) for name in significance_windows)
        )
        unit_axis_df["p_value_within_unit_axis_adjusted"] = np.nan
        for _, idx in unit_axis_df.groupby(["axis_source", "unit_key", "axis_name"], dropna=False).groups.items():
            idx_list = list(idx)
            counts_mask = (
                unit_axis_df.loc[idx_list, "counts_toward_significance"]
                .to_numpy(dtype=bool)
                .reshape(-1)
            )
            if not np.any(counts_mask):
                continue
            pvals = unit_axis_df.loc[idx_list, "p_value"].to_numpy(dtype=float).reshape(-1)
            adj_selected = adjust_pvalues(
                pvals[counts_mask],
                settings.pvalue_correction,
            )
            adj_full = np.full(pvals.shape, np.nan, dtype=float)
            adj_full[counts_mask] = adj_selected
            unit_axis_df.loc[idx_list, "p_value_within_unit_axis_adjusted"] = adj_full
        unit_axis_df["significant_within_unit"] = (
            unit_axis_df["p_value_within_unit_axis_adjusted"].to_numpy(dtype=float) < float(settings.alpha)
        )
        unit_axis_df = unit_axis_df.sort_values(
            ["date", "region", "unit_uuid", "window_name", "axis_source", "axis_name"]
        ).reset_index(drop=True)
    else:
        unit_norm_df = pd.DataFrame()

    if not unit_window_df.empty:
        unit_window_df = _attach_normalized_axis_values_to_window_summary(
            unit_window_df,
            unit_norm_df,
        )
        unit_window_df = unit_window_df.sort_values(
            ["date", "region", "unit_uuid", "window_name"]
        ).reset_index(drop=True)

    unit_axis_significance_df = _build_unit_axis_significance_table(
        unit_term_df,
        settings=settings,
    )
    unit_axis_collapsed_df = _build_unit_axis_collapsed_magnitude_table(
        unit_axis_df,
        unit_axis_significance_df,
        settings=settings,
    )
    unit_axis_magnitude_df = _build_axis_magnitude_input_table(
        unit_axis_df,
        settings=settings,
        unit_axis_collapsed_df=unit_axis_collapsed_df,
    )

    (
        region_fraction_df,
        region_fraction_pairwise_df,
    ) = _build_region_fraction_tables(unit_axis_significance_df, settings=settings)
    (
        region_axis_summary_df,
        region_axis_pairwise_df,
        region_axis_within_df,
    ) = _build_region_axis_tables(unit_axis_magnitude_df, settings=settings)
    region_fraction_within_df = pd.DataFrame()
    region_axis_friedman_df = pd.DataFrame()

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    unit_term_csv = out_root / _ensure_filename(settings.unit_term_filename, ".csv")
    unit_axis_csv = out_root / _ensure_filename(settings.unit_axis_filename, ".csv")
    unit_axis_collapsed_csv = out_root / _ensure_filename(settings.unit_axis_collapsed_filename, ".csv")
    unit_window_csv = out_root / _ensure_filename(settings.unit_window_summary_filename, ".csv")
    region_fraction_csv = out_root / _ensure_filename(settings.region_fraction_filename, ".csv")
    region_fraction_pairwise_csv = out_root / _ensure_filename(settings.region_fraction_pairwise_filename, ".csv")
    region_fraction_within_csv = out_root / _ensure_filename(
        settings.region_fraction_within_region_filename,
        ".csv",
    )
    region_axis_summary_csv = out_root / _ensure_filename(settings.region_axis_summary_filename, ".csv")
    region_axis_pairwise_csv = out_root / _ensure_filename(settings.region_axis_pairwise_filename, ".csv")
    region_axis_within_csv = out_root / _ensure_filename(settings.region_axis_within_region_filename, ".csv")
    region_axis_friedman_csv = out_root / _ensure_filename(settings.region_axis_friedman_filename, ".csv")
    result_pkl = out_root / _ensure_filename(settings.output_pickle_filename, ".pkl")

    unit_term_df.to_csv(unit_term_csv, index=False)
    unit_axis_df.to_csv(unit_axis_csv, index=False)
    unit_axis_collapsed_df.to_csv(unit_axis_collapsed_csv, index=False)
    unit_window_df.to_csv(unit_window_csv, index=False)
    region_fraction_df.to_csv(region_fraction_csv, index=False)
    region_fraction_pairwise_df.to_csv(region_fraction_pairwise_csv, index=False)
    region_fraction_within_df.to_csv(region_fraction_within_csv, index=False)
    region_axis_summary_df.to_csv(region_axis_summary_csv, index=False)
    region_axis_pairwise_df.to_csv(region_axis_pairwise_csv, index=False)
    region_axis_within_df.to_csv(region_axis_within_csv, index=False)
    region_axis_friedman_df.to_csv(region_axis_friedman_csv, index=False)

    result_obj = {
        "meta": {
            "windows_ms": windows_ms,
            "significance_windows": [str(name) for name in significance_windows],
            "trial_input_modality": settings.trial_input_modality,
            "trial_input_filename": _ensure_filename(settings.trial_input_filename, ".pkl"),
            "smooth_before_window_average": bool(settings.smooth_before_window_average),
            "smoothing_sigma_ms": float(settings.smoothing_sigma_ms),
            "min_trials_per_cell": int(settings.min_trials_per_cell),
            "min_units_per_region": int(settings.min_units_per_region),
            "alpha": float(settings.alpha),
            "pvalue_correction": normalize_pvalue_correction(settings.pvalue_correction),
            "unit_significance_mode": str(settings.unit_significance_mode),
            "axis_comparison_mode": str(settings.axis_comparison_mode),
            "axis_magnitude_source": str(CELL_MEAN_AXIS_SOURCE),
            "axis_magnitude_sources": list(CELL_MEAN_MAGNITUDE_SOURCES),
            "normalized_axis_source": str(CELL_MEAN_UNIT_RANGE_NORM_AXIS_SOURCE),
            "normalized_axis_method": "unit_dynamic_range_over_significance_windows",
            "parallelization_scope": str(settings.parallelization_scope),
            "n_units": int(unit_term_df["unit_key"].nunique()) if not unit_term_df.empty else 0,
            "n_regions": int(unit_term_df["region"].astype(str).nunique()) if not unit_term_df.empty else 0,
            "n_axis_magnitude_rows": int(len(unit_axis_magnitude_df)),
            "n_significant_unit_axis_rows": (
                int(unit_axis_significance_df.loc[unit_axis_significance_df["is_significant_axis"]].shape[0])
                if not unit_axis_significance_df.empty and "is_significant_axis" in unit_axis_significance_df.columns
                else 0
            ),
        },
        "unit_terms": unit_term_df,
        "unit_axis_values": unit_axis_df,
        "unit_axis_collapsed": unit_axis_collapsed_df,
        "unit_axis_significance": unit_axis_significance_df,
        "unit_window_summary": unit_window_df,
        "region_significant_fractions": region_fraction_df,
        "region_significant_fraction_pairwise": region_fraction_pairwise_df,
        "region_significant_fraction_within_region": region_fraction_within_df,
        "region_axis_summary": region_axis_summary_df,
        "region_axis_pairwise": region_axis_pairwise_df,
        "region_axis_within_region": region_axis_within_df,
        "region_axis_friedman": region_axis_friedman_df,
    }
    save_pickle_path(result_obj, result_pkl)
    return result_obj


def _print_table_block(
    title: str,
    table_df: pd.DataFrame | None,
    *,
    columns: tuple[str, ...] | None = None,
    empty_message: str = "(none)",
) -> None:
    print(f"[analysis] {title}")
    print(f"[analysis] {'-' * len(title)}")
    if table_df is None or table_df.empty:
        print(f"[analysis] {empty_message}")
        print("[analysis]")
        return
    df = table_df.copy()
    if columns is not None:
        keep = [col for col in columns if col in df.columns]
        if keep:
            df = df.loc[:, keep]
    text = df.to_string(index=False)
    for line in text.splitlines():
        print(f"[analysis] {line}")
    print("[analysis]")


def print_fixation_roi_vs_period_factorial_summary(result: dict) -> None:
    """Print human-readable summary tables for ROI-vs-period factorial outputs."""
    meta = result.get("meta") if isinstance(result, dict) else None
    unit_term_df = result.get("unit_terms")
    unit_axis_significance_df = result.get("unit_axis_significance")
    unit_axis_collapsed_df = result.get("unit_axis_collapsed")
    region_fraction_df = result.get("region_significant_fractions")
    region_fraction_pairwise_df = result.get("region_significant_fraction_pairwise")
    region_axis_summary_df = result.get("region_axis_summary")
    region_axis_pairwise_df = result.get("region_axis_pairwise")
    region_axis_within_df = result.get("region_axis_within_region")
    window_order = (
        [str(win) for win in meta.get("significance_windows", []) if str(win).strip()]
        if isinstance(meta, dict)
        else []
    )
    axis_comparison_mode = (
        str(meta.get("axis_comparison_mode", "max_abs_across_windows"))
        if isinstance(meta, dict)
        else "max_abs_across_windows"
    )
    axis_magnitude_source = (
        str(meta.get("axis_magnitude_source", "cell_means"))
        if isinstance(meta, dict)
        else "cell_means"
    )

    if unit_term_df is None or unit_term_df.empty:
        print("[analysis] no unit-term factorial rows were produced")
        return

    n_units = int(unit_term_df["unit_key"].astype(str).nunique())
    n_term_rows = int(len(unit_term_df))
    n_regions = int(unit_term_df["region"].fillna("unknown").astype(str).nunique())
    print(
        "[analysis] roi-vs-period factorial unit output: "
        f"units={n_units}, term_rows={n_term_rows}, regions={n_regions}"
    )
    print(f"[analysis] axis comparison mode: {axis_comparison_mode}")
    print(f"[analysis] axis magnitude source: {axis_magnitude_source}")
    if unit_axis_significance_df is not None and not unit_axis_significance_df.empty:
        n_sig_axis = int(unit_axis_significance_df["is_significant_axis"].sum())
        print(
            "[analysis] collapsed unit-axis significance rows: "
            f"{len(unit_axis_significance_df)} (significant={n_sig_axis})"
        )
    if unit_axis_collapsed_df is not None and not unit_axis_collapsed_df.empty:
        print(f"[analysis] collapsed unit-axis magnitude rows: {len(unit_axis_collapsed_df)}")
    if region_fraction_df is not None and not region_fraction_df.empty:
        print(f"[analysis] region significant-fraction rows: {len(region_fraction_df)}")
    if region_fraction_pairwise_df is not None and not region_fraction_pairwise_df.empty:
        n_sig_pair = int(region_fraction_pairwise_df["significant_adjusted"].sum())
        print(
            "[analysis] region fraction pairwise tests: "
            f"{len(region_fraction_pairwise_df)} (significant adjusted={n_sig_pair})"
        )
    if region_axis_pairwise_df is not None and not region_axis_pairwise_df.empty:
        n_sig_axis_pair = int(region_axis_pairwise_df["significant_adjusted"].sum())
        print(
            "[analysis] region axis pairwise tests: "
            f"{len(region_axis_pairwise_df)} (significant adjusted={n_sig_axis_pair})"
        )

    regions = (
        sorted(region_fraction_df["region"].astype(str).unique().tolist())
        if region_fraction_df is not None and not region_fraction_df.empty and "region" in region_fraction_df.columns
        else []
    )
    if not regions and "region" in unit_term_df.columns:
        regions = sorted(unit_term_df["region"].fillna("unknown").astype(str).unique().tolist())

    for region in regions:
        print("[analysis]")
        print(f"[analysis] === Region Summary: {region} ===")

        frac_region = (
            region_fraction_df.loc[region_fraction_df["region"].astype(str) == str(region)].copy()
            if region_fraction_df is not None and not region_fraction_df.empty and "region" in region_fraction_df.columns
            else pd.DataFrame()
        )
        if not frac_region.empty:
            frac_region = frac_region.sort_values(["axis_name"]).reset_index(drop=True)
        _print_table_block(
            "Selective Fraction By Axis (Collapsed Across pre/peri/post)",
            frac_region,
            columns=(
                "axis_name",
                "selective_units",
                "total_units",
                "fraction_selective",
            ),
            empty_message="no selective-fraction rows",
        )

        axis_summary_region = (
            region_axis_summary_df.loc[region_axis_summary_df["region"].astype(str) == str(region)].copy()
            if (
                region_axis_summary_df is not None
                and not region_axis_summary_df.empty
                and "region" in region_axis_summary_df.columns
            )
            else pd.DataFrame()
        )
        axis_within_region = (
            region_axis_within_df.loc[region_axis_within_df["region"].astype(str) == str(region)].copy()
            if (
                region_axis_within_df is not None
                and not region_axis_within_df.empty
                and "region" in region_axis_within_df.columns
            )
            else pd.DataFrame()
        )
        candidate_windows: set[str] = set()
        if not axis_summary_region.empty and "window_name" in axis_summary_region.columns:
            candidate_windows.update(axis_summary_region["window_name"].astype(str).unique().tolist())
        if not axis_within_region.empty and "window_name" in axis_within_region.columns:
            candidate_windows.update(axis_within_region["window_name"].astype(str).unique().tolist())
        ordered_windows = [win for win in window_order if win in candidate_windows]
        ordered_windows.extend(sorted(candidate_windows - set(ordered_windows)))
        if not ordered_windows and window_order:
            ordered_windows = list(window_order)

        for window_name in ordered_windows:
            axis_summary_window = (
                axis_summary_region.loc[axis_summary_region["window_name"].astype(str) == str(window_name)].copy()
                if not axis_summary_region.empty and "window_name" in axis_summary_region.columns
                else pd.DataFrame()
            )
            if not axis_summary_window.empty:
                axis_summary_window = axis_summary_window.sort_values(
                    ["axis_source", "axis_name"]
                ).reset_index(drop=True)
            _print_table_block(
                f"Axis Magnitude Summary ({window_name}; All Units)",
                axis_summary_window,
                columns=(
                    "axis_source",
                    "axis_name",
                    "n_units",
                    "mean_abs",
                    "median_abs",
                    "std_abs",
                ),
                empty_message="no axis-summary rows",
            )

            axis_within_window = (
                axis_within_region.loc[axis_within_region["window_name"].astype(str) == str(window_name)].copy()
                if not axis_within_region.empty and "window_name" in axis_within_region.columns
                else pd.DataFrame()
            )
            if not axis_within_window.empty:
                axis_within_window = axis_within_window.sort_values(
                    ["axis_source", "axis_a", "axis_b"]
                ).reset_index(drop=True)
            _print_table_block(
                f"Within-Region Axis Magnitude Comparisons ({window_name}; Welch, Adjusted)",
                axis_within_window,
                columns=(
                    "axis_source",
                    "axis_a",
                    "axis_b",
                    "test_name",
                    "n_units_paired",
                    "mean_a",
                    "mean_b",
                    "delta_mean_a_minus_b",
                    "p_value",
                    "p_value_adjusted",
                    "significant_adjusted",
                ),
                empty_message="no within-region axis comparisons",
            )

    _print_table_block(
        "Cross-Region Fraction Differences (All Region Pairs, Adjusted)",
        region_fraction_pairwise_df,
        columns=(
            "axis_name",
            "region_a",
            "region_b",
            "fraction_a",
            "fraction_b",
            "fraction_diff_a_minus_b",
            "p_value",
            "p_value_adjusted",
            "significant_adjusted",
        ),
        empty_message="no cross-region fraction comparison rows",
    )

    if (
        region_axis_pairwise_df is not None
        and not region_axis_pairwise_df.empty
        and "window_name" in region_axis_pairwise_df.columns
    ):
        pair_windows = set(region_axis_pairwise_df["window_name"].astype(str).unique().tolist())
        ordered_pair_windows = [win for win in window_order if win in pair_windows]
        ordered_pair_windows.extend(sorted(pair_windows - set(ordered_pair_windows)))
        for window_name in ordered_pair_windows:
            axis_pair_window = region_axis_pairwise_df.loc[
                region_axis_pairwise_df["window_name"].astype(str) == str(window_name)
            ].copy()
            if not axis_pair_window.empty:
                axis_pair_window = axis_pair_window.sort_values(
                    ["axis_source", "axis_name", "region_a", "region_b"]
                ).reset_index(drop=True)
            _print_table_block(
                f"Cross-Region Axis Magnitude Differences ({window_name}; Welch, Adjusted)",
                axis_pair_window,
                columns=(
                    "axis_source",
                    "axis_name",
                    "test_name",
                    "region_a",
                    "region_b",
                    "mean_a",
                    "mean_b",
                    "delta_mean_a_minus_b",
                    "p_value",
                    "p_value_adjusted",
                    "significant_adjusted",
                ),
                empty_message="no cross-region axis comparison rows",
            )
    else:
        _print_table_block(
            "Cross-Region Axis Magnitude Differences (Welch, Adjusted)",
            region_axis_pairwise_df,
            columns=(
                "axis_source",
                "axis_name",
                "test_name",
                "region_a",
                "region_b",
                "mean_a",
                "mean_b",
                "delta_mean_a_minus_b",
                "p_value",
                "p_value_adjusted",
                "significant_adjusted",
            ),
            empty_message="no cross-region axis comparison rows",
        )
