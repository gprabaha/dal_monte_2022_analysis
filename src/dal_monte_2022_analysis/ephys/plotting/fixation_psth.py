"""Plot multiscale per-unit fixation PSTH rasters and mean firing-rate traces."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    extract_trials_df_and_meta as _extract_trials_df_and_meta_shared,
    resolve_bin_centers_from_meta as _resolve_bin_centers_from_meta_shared,
)
from dal_monte_2022_analysis.core.stats import (
    mannwhitneyu_pvalues_per_column,
    welch_ttest,
)
from dal_monte_2022_analysis.ephys.plotting.common import (
    counts_to_spike_times as _counts_to_spike_times_shared,
    darken_color as _darken_color_shared,
    ensure_ext as _ensure_ext_shared,
    fallback_bin_centers as _fallback_bin_centers_shared,
    iter_trial_rows as _iter_trial_rows_shared,
    resolve_figsize_and_dpi as _resolve_figsize_and_dpi_shared,
    safe_optional_str as _safe_optional_str_shared,
    safe_region_folder as _safe_region_folder_shared,
    safe_unit_filename as _safe_unit_filename_shared,
    sample_rows as _sample_rows_shared,
    stable_seed as _stable_seed_shared,
    row_counts as _row_counts_shared,
    row_spike_train_counts as _row_spike_train_counts_shared,
)
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_date_paths
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure
from dal_monte_2022_analysis.runtime.execution.parallel import get_n_processes
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


DEFAULT_CONDITION_COLORS = {
    "face_interactive": "#d62728",
    "face_non_interactive": "#1f77b4",
    "object": "#2ca02c",
}
_PLOT_CONDITION_ORDER = (
    ("face_interactive", "Interactive Face"),
    ("face_non_interactive", "Non-Interactive Face"),
    ("object", "Object"),
)
_AVERAGE_TRACE_CACHE: dict[tuple[str, str, str, str], Optional[tuple[pd.DataFrame, np.ndarray, float]]] = {}


@dataclass
class FixationPSTHUnitPlotSettings:
    """Configuration for per-unit fixation PSTH plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations.pkl"
    raster_trial_input_modality: Optional[str] = None
    raster_trial_input_filename: Optional[str] = None
    use_precomputed_average_traces: bool = True
    average_trace_input_subdir: str = "ephys/psth/fixation_psth_averages"
    average_trace_input_filename: str = "fixations.pkl"
    average_trace_object_input_subdir: Optional[str] = None
    average_trace_object_input_filename: Optional[str] = None
    allow_trial_trace_fallback: bool = True
    segregate_selective_units: bool = False
    selectivity_input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    selectivity_unit_summary_filename: str = "unit_selectivity.csv"
    selective_unit_subfolder: str = "selective"
    output_subdir: str = "ephys/psth/fixation_psth_unit_plots_multiscale_5s"
    output_extension: str = "png"
    example_units_subfolder: Optional[str] = None
    figure_size: Optional[Sequence[float]] = None
    output_dpi: Optional[int] = 220
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    use_parallel: bool = True
    parallelize_units: bool = True
    unit_parallel_min_units: int = 2
    max_procs: int = 16
    test_single: bool = False
    max_trials_per_condition: Optional[int] = 300
    random_seed: int = 42
    condition_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONDITION_COLORS),
    )
    smooth_before_average: bool = True
    smoothing_sigma_ms: float = 20.0
    raster_jitter_within_bin: bool = True
    raster_linelength: float = 0.95
    raster_linewidth: float = 1.0
    raster_alpha: float = 1.0
    raster_darkening_factor: float = 0.65
    raster_show_condition_background: bool = False
    panel_raster_height_ratio: float = 1.2
    panel_rate_height_ratio: float = 2.0
    show_significance_ticks: bool = False
    significance_alpha: float = 0.05
    significance_test: str = "welch_ttest"
    significance_min_trials_per_condition: int = 2
    significance_tick_height_ratio: float = 0.03
    significance_tick_row_gap_ratio: float = 0.08
    display_half_windows_s: Sequence[float] = field(default_factory=lambda: (5.0, 3.0, 1.0))
    show_analysis_window_overlays: bool = True
    analysis_window_overlays_s: Sequence[tuple[float, float]] = field(
        default_factory=lambda: ((-0.5, 0.0), (-0.25, 0.25), (0.0, 0.5)),
    )
    analysis_window_overlay_colors: Sequence[str] = field(
        default_factory=lambda: ("#bdbdbd", "#8f8f8f", "#636363"),
    )
    analysis_window_overlay_linestyle: str = ":"
    analysis_window_overlay_linewidth: float = 0.8
    bin_size_ms_fallback: float = 10.0
    window_pre_s: float = 1.0
    window_post_s: float = 1.0


def _ensure_ext(ext: str) -> str:
    return _ensure_ext_shared(ext, fallback="png")


def _iter_trial_rows(
    cfg: dict,
    settings: FixationPSTHUnitPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    modality: Optional[str] = None,
    filename: Optional[str] = None,
) -> list[dict]:
    return _iter_trial_rows_shared(
        cfg,
        modality=str(modality or settings.trial_input_modality),
        filename=str(filename or settings.trial_input_filename),
        dates=dates,
        sessions=sessions,
    )


def _fallback_bin_centers(settings: FixationPSTHUnitPlotSettings) -> np.ndarray:
    return _fallback_bin_centers_shared(
        bin_size_ms_fallback=settings.bin_size_ms_fallback,
        window_pre_s=settings.window_pre_s,
        window_post_s=settings.window_post_s,
    )


def _resolve_spike_train_bin_centers_from_meta(meta: dict) -> Optional[np.ndarray]:
    centers = meta.get("spike_train_bin_centers_s_rel")
    if centers is not None:
        arr = np.asarray(centers, dtype=float).reshape(-1)
        if arr.size > 0:
            return arr
    edges = meta.get("spike_train_bin_edges_s_rel")
    if edges is not None:
        arr = np.asarray(edges, dtype=float).reshape(-1)
        if arr.size > 1:
            return 0.5 * (arr[:-1] + arr[1:])
    return None


def _resolve_bin_duration_from_centers(bin_centers: np.ndarray, fallback_bin_size_ms: float) -> float:
    centers = np.asarray(bin_centers, dtype=float).reshape(-1)
    if centers.size > 1:
        inferred = float(np.mean(np.diff(centers)))
        if np.isfinite(inferred) and inferred > 0.0:
            return inferred
    return float(fallback_bin_size_ms) / 1000.0


def _load_trials_from_paths_for_date(
    paths: Sequence[Path],
    *,
    date: str,
    settings: FixationPSTHUnitPlotSettings,
    require_psth_counts: bool,
) -> tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    all_rows: list[pd.DataFrame] = []
    bin_centers_ref: Optional[np.ndarray] = None
    raster_bin_centers_ref: Optional[np.ndarray] = None

    for path in paths:
        obj = load_pickle_path(path)
        trials_df, meta = _extract_trials_df_and_meta_shared(obj)
        if trials_df.empty:
            continue
        if require_psth_counts and "psth_counts" not in trials_df.columns:
            continue

        local_centers = _resolve_bin_centers_from_meta_shared(meta)
        if local_centers is not None:
            if bin_centers_ref is None:
                bin_centers_ref = local_centers
            elif (
                local_centers.shape != bin_centers_ref.shape
                or not np.allclose(local_centers, bin_centers_ref)
            ):
                print(f"[plot] skipping {path} due to mismatched PSTH bin centers")
                continue
        local_raster_centers = _resolve_spike_train_bin_centers_from_meta(meta)
        if local_raster_centers is not None:
            if raster_bin_centers_ref is None:
                raster_bin_centers_ref = local_raster_centers
            elif (
                local_raster_centers.shape != raster_bin_centers_ref.shape
                or not np.allclose(local_raster_centers, raster_bin_centers_ref)
            ):
                print(f"[plot] skipping {path} due to mismatched spike-train bin centers")
                continue

        df = trials_df.copy()
        if "date" not in df.columns:
            df["date"] = date
        if "session" not in df.columns:
            session_part = next(
                (part for part in path.parts if part.startswith("session=")),
                "session=unknown",
            )
            df["session"] = session_part.split("=", 1)[1]
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame(), None, None

    out_df = pd.concat(all_rows, axis=0, ignore_index=True)
    return out_df, bin_centers_ref, raster_bin_centers_ref


def _resolve_raster_trial_input(
    settings: FixationPSTHUnitPlotSettings,
) -> Optional[tuple[str, str]]:
    filename = _safe_optional_str_shared(settings.raster_trial_input_filename)
    if filename is None:
        return None
    modality = _safe_optional_str_shared(settings.raster_trial_input_modality)
    if modality is None:
        modality = str(settings.trial_input_modality)
    return str(modality), str(filename)


def _resolve_trial_merge_keys(primary_df: pd.DataFrame, raster_df: pd.DataFrame) -> list[str]:
    candidate_keys = (
        "date",
        "session",
        "unit_uuid",
        "fixation_agent",
        "fixation_start_idx",
        "fixation_stop_idx",
        "fixation_category",
        "interactive_state",
    )
    keys = [col for col in candidate_keys if col in primary_df.columns and col in raster_df.columns]
    if not keys:
        return []
    if primary_df.duplicated(subset=keys).any():
        return []
    if raster_df.duplicated(subset=keys).any():
        return []
    return keys


def _merge_raster_trial_rows(
    primary_df: pd.DataFrame,
    raster_df: pd.DataFrame,
) -> pd.DataFrame:
    if primary_df.empty or raster_df.empty or "spike_train_counts" not in raster_df.columns:
        return primary_df

    merge_keys = _resolve_trial_merge_keys(primary_df, raster_df)
    if not merge_keys:
        print("[plot] unable to merge separate raster trial file; no unique fixation-trial key columns found")
        return primary_df

    raster_cols = merge_keys + ["spike_train_counts"]
    raster_merge_df = raster_df.loc[:, raster_cols].copy()

    base_df = primary_df.copy()
    if "spike_train_counts" in base_df.columns:
        base_df = base_df.drop(columns=["spike_train_counts"])

    return base_df.merge(
        raster_merge_df,
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )


def _load_trials_for_date(
    primary_paths: Sequence[Path],
    *,
    raster_paths: Optional[Sequence[Path]],
    date: str,
    settings: FixationPSTHUnitPlotSettings,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    primary_df, trace_bin_centers_ref, primary_raster_centers_ref = _load_trials_from_paths_for_date(
        primary_paths,
        date=date,
        settings=settings,
        require_psth_counts=True,
    )
    if primary_df.empty:
        fallback = _fallback_bin_centers(settings)
        return pd.DataFrame(), fallback, fallback

    trace_bin_centers = (
        np.asarray(trace_bin_centers_ref, dtype=float)
        if trace_bin_centers_ref is not None
        else _fallback_bin_centers(settings)
    )
    raster_bin_centers = (
        np.asarray(primary_raster_centers_ref, dtype=float)
        if primary_raster_centers_ref is not None
        else np.asarray(trace_bin_centers, dtype=float)
    )

    if raster_paths:
        raster_df, _, raster_centers_ref = _load_trials_from_paths_for_date(
            raster_paths,
            date=date,
            settings=settings,
            require_psth_counts=False,
        )
        if not raster_df.empty and "spike_train_counts" in raster_df.columns:
            primary_df = _merge_raster_trial_rows(primary_df, raster_df)
            if raster_centers_ref is not None:
                raster_bin_centers = np.asarray(raster_centers_ref, dtype=float)

    return primary_df, trace_bin_centers, raster_bin_centers


def _resolve_figsize_and_dpi(settings: FixationPSTHUnitPlotSettings) -> tuple[list[float], Optional[int]]:
    if settings.figure_size is not None:
        figure_size = [float(val) for val in settings.figure_size]
        if len(figure_size) != 2:
            raise ValueError("figure_size must contain exactly two values: width and height in inches.")
        _, dpi = _resolve_figsize_and_dpi_shared(
            plotting_cfg_path=settings.plotting_cfg_path,
            output_dpi=settings.output_dpi,
            default_figsize=figure_size,
        )
        return figure_size, dpi
    return _resolve_figsize_and_dpi_shared(
        plotting_cfg_path=settings.plotting_cfg_path,
        output_dpi=settings.output_dpi,
        default_figsize=[15.0, 6.8],
    )


def _condition_masks(df: pd.DataFrame, interactive_label: str, face_label: str, object_label: str):
    category_series = df.get("fixation_category", pd.Series(index=df.index, dtype=str)).astype(str)
    face_mask = category_series == str(face_label)
    object_mask = category_series == str(object_label)

    if "is_interactive" in df.columns:
        interactive_mask = df["is_interactive"].map(
            lambda val: _as_bool(val, interactive_label),
        )
    elif "interactive_state" in df.columns:
        interactive_mask = df["interactive_state"].map(
            lambda val: _as_bool(val, interactive_label),
        )
    else:
        interactive_mask = pd.Series(False, index=df.index)

    interactive_mask = interactive_mask.fillna(False).astype(bool)
    return {
        "face_interactive": face_mask & interactive_mask,
        "face_non_interactive": face_mask & (~interactive_mask),
        "object": object_mask,
    }


def _ensure_pkl_filename(filename: str) -> str:
    token = str(filename).strip()
    if not token:
        token = "fixations.pkl"
    if not token.endswith(".pkl"):
        token = f"{token}.pkl"
    return token


def _extract_average_df_and_meta(obj) -> tuple[pd.DataFrame, dict]:
    if isinstance(obj, dict):
        avg_df = obj.get("averages")
        meta = obj.get("meta", {})
        if isinstance(avg_df, pd.DataFrame):
            return avg_df, meta if isinstance(meta, dict) else {}
    if isinstance(obj, pd.DataFrame):
        return obj, {}
    raise ValueError(f"Unsupported average PSTH object type for plotting: {type(obj)}")


def _extract_average_partition_df_and_meta(
    obj,
    *,
    require_split_partition: bool,
) -> tuple[pd.DataFrame, dict]:
    if isinstance(obj, dict):
        if require_split_partition:
            avg_df = obj.get("averages_split_by_interactive_state")
            meta = obj.get("meta", {})
            if isinstance(avg_df, pd.DataFrame):
                meta_dict = meta if isinstance(meta, dict) else {}
                split_meta = meta_dict.get("split_meta", {})
                if not isinstance(split_meta, dict):
                    split_meta = {}
                return avg_df, split_meta or meta_dict
        else:
            avg_df = obj.get("averages_unsplit_by_interactive_state")
            meta = obj.get("meta", {})
            if isinstance(avg_df, pd.DataFrame):
                meta_dict = meta if isinstance(meta, dict) else {}
                unsplit_meta = meta_dict.get("unsplit_meta", {})
                if not isinstance(unsplit_meta, dict):
                    unsplit_meta = {}
                return avg_df, unsplit_meta or meta_dict
    return _extract_average_df_and_meta(obj)


def _resolve_average_bin_duration_s(
    meta: dict,
    *,
    bin_centers: np.ndarray,
    settings: FixationPSTHUnitPlotSettings,
) -> float:
    duration_s = meta.get("target_bin_size_s")
    if duration_s is None:
        duration_s = meta.get("bin_size_s")
    try:
        if duration_s is not None:
            duration_s = float(duration_s)
    except Exception:
        duration_s = None
    if duration_s is not None and np.isfinite(duration_s) and duration_s > 0:
        return float(duration_s)
    if bin_centers.size > 1:
        inferred = float(np.mean(np.diff(bin_centers)))
        if np.isfinite(inferred) and inferred > 0:
            return inferred
    return float(settings.bin_size_ms_fallback) / 1000.0


def _resolve_average_trace_input_location(
    settings: FixationPSTHUnitPlotSettings,
    *,
    for_object: bool,
) -> tuple[str, str]:
    subdir = str(settings.average_trace_input_subdir).strip()
    filename = str(settings.average_trace_input_filename).strip()
    if for_object:
        if settings.average_trace_object_input_subdir is not None:
            subdir = str(settings.average_trace_object_input_subdir).strip()
        if settings.average_trace_object_input_filename is not None:
            filename = str(settings.average_trace_object_input_filename).strip()
    return subdir, _ensure_pkl_filename(filename)


def _load_average_trace_bundle_for_date(
    settings: FixationPSTHUnitPlotSettings,
    *,
    date: str,
    for_object: bool,
) -> Optional[tuple[pd.DataFrame, np.ndarray, float]]:
    if not bool(settings.use_precomputed_average_traces):
        return None

    subdir, filename = _resolve_average_trace_input_location(settings, for_object=for_object)
    if not subdir:
        return None

    cache_key = (str(settings.cfg_path), subdir, filename, str(date))
    if cache_key in _AVERAGE_TRACE_CACHE:
        return _AVERAGE_TRACE_CACHE[cache_key]

    cfg = load_config(settings.cfg_path)
    rows = scan_analysis_date_paths(
        cfg,
        subdir,
        filename=filename,
        dates=[date],
    )
    if not rows:
        _AVERAGE_TRACE_CACHE[cache_key] = None
        return None

    dfs: list[pd.DataFrame] = []
    bin_centers_ref: Optional[np.ndarray] = None
    bin_duration_ref: Optional[float] = None
    for row in rows:
        obj = load_pickle_path(row["path"])
        avg_df, meta = _extract_average_partition_df_and_meta(
            obj,
            require_split_partition=not bool(for_object),
        )
        if avg_df.empty or "psth_mean" not in avg_df.columns:
            continue

        centers = _resolve_bin_centers_from_meta_shared(meta)
        if centers is not None:
            if bin_centers_ref is None:
                bin_centers_ref = np.asarray(centers, dtype=float)
            elif centers.shape != bin_centers_ref.shape or not np.allclose(centers, bin_centers_ref):
                raise ValueError(
                    f"Mismatched average PSTH bin centers across files for plotting; path={row['path']}"
                )
            duration_s = _resolve_average_bin_duration_s(meta, bin_centers=bin_centers_ref, settings=settings)
            if bin_duration_ref is None:
                bin_duration_ref = float(duration_s)
            elif not np.isclose(float(duration_s), float(bin_duration_ref)):
                raise ValueError(
                    f"Mismatched average PSTH bin durations across files for plotting; path={row['path']}"
                )

        df = avg_df.copy()
        if "date" not in df.columns:
            df["date"] = str(row.get("date", date))
        dfs.append(df)

    if not dfs:
        _AVERAGE_TRACE_CACHE[cache_key] = None
        return None

    out_df = pd.concat(dfs, axis=0, ignore_index=True)
    if bin_centers_ref is None:
        first = out_df.iloc[0]
        first_mean = np.asarray(first.get("psth_mean"), dtype=float).reshape(-1)
        fallback = _fallback_bin_centers(settings)
        if fallback.size == first_mean.size:
            bin_centers_ref = fallback
        else:
            bin_centers_ref = np.arange(first_mean.size, dtype=float)
    if bin_duration_ref is None:
        bin_duration_ref = _resolve_average_bin_duration_s(
            {},
            bin_centers=np.asarray(bin_centers_ref, dtype=float),
            settings=settings,
        )
    bundle = (out_df, np.asarray(bin_centers_ref, dtype=float), float(bin_duration_ref))
    _AVERAGE_TRACE_CACHE[cache_key] = bundle
    return bundle


def _resolve_condition_for_average_row(
    row,
    settings: FixationPSTHUnitPlotSettings,
) -> Optional[str]:
    category = str(getattr(row, "fixation_category", "")).strip()
    if category == str(settings.object_label):
        return "object"
    if category != str(settings.face_label):
        return None

    if hasattr(row, "is_interactive"):
        interactive = _as_bool(getattr(row, "is_interactive"), settings.interactive_label)
        return "face_interactive" if interactive else "face_non_interactive"
    if hasattr(row, "interactive_state"):
        raw_value = getattr(row, "interactive_state")
        token = str(raw_value).strip()
        if not token or token.lower() in {"nan", "none", "null"}:
            return None
        interactive = _as_bool(raw_value, settings.interactive_label)
        return "face_interactive" if interactive else "face_non_interactive"
    return None


def _resolve_n_trials_for_average_row(row) -> int:
    if not hasattr(row, "n_trials"):
        return 1
    try:
        n_trials = int(float(getattr(row, "n_trials")))
    except Exception:
        return 1
    return n_trials if n_trials > 0 else 1


def _combine_average_records(
    records: list[dict[str, np.ndarray | int]],
) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    if not records:
        return None

    sum_vec: Optional[np.ndarray] = None
    sumsq_vec: Optional[np.ndarray] = None
    n_total = 0
    for record in records:
        n_trials = int(record["n_trials"])
        if n_trials <= 0:
            continue
        mean_vec = np.asarray(record["mean_counts"], dtype=float).reshape(-1)
        sem_vec = np.asarray(record["sem_counts"], dtype=float).reshape(-1)
        if mean_vec.size == 0 or sem_vec.size == 0 or mean_vec.shape != sem_vec.shape:
            return None
        if np.any(~np.isfinite(mean_vec)) or np.any(~np.isfinite(sem_vec)):
            return None

        if sum_vec is None:
            sum_vec = np.zeros_like(mean_vec, dtype=float)
            sumsq_vec = np.zeros_like(mean_vec, dtype=float)
        elif sum_vec.shape != mean_vec.shape:
            return None

        sample_var = np.square(sem_vec) * float(n_trials)
        sum_vec += mean_vec * float(n_trials)
        sumsq_vec += sample_var * float(max(0, n_trials - 1)) + float(n_trials) * np.square(mean_vec)
        n_total += n_trials

    if sum_vec is None or sumsq_vec is None or n_total <= 0:
        return None

    mean_vec = sum_vec / float(n_total)
    if n_total > 1:
        numer = sumsq_vec - (np.square(sum_vec) / float(n_total))
        numer = np.maximum(numer, 0.0)
        sample_var = numer / float(n_total - 1)
        sem_vec = np.sqrt(sample_var / float(n_total))
    else:
        sem_vec = np.zeros_like(mean_vec, dtype=float)
    return mean_vec, sem_vec, int(n_total)


def _build_precomputed_trace_overrides(
    average_df: pd.DataFrame,
    *,
    unit_uuid: str,
    bin_centers: np.ndarray,
    bin_size_s: float,
    settings: FixationPSTHUnitPlotSettings,
) -> dict[str, dict]:
    if average_df.empty or "unit_uuid" not in average_df.columns:
        return {}

    unit_token = str(unit_uuid).strip()
    subset = average_df.loc[
        average_df["unit_uuid"].astype(str).map(lambda value: value.strip()) == unit_token
    ].copy()
    if subset.empty:
        return {}

    grouped: dict[str, list[dict[str, np.ndarray | int]]] = {
        cond_key: []
        for cond_key, _ in _PLOT_CONDITION_ORDER
    }

    n_bins = int(bin_centers.size)
    for row in subset.itertuples(index=False):
        cond = _resolve_condition_for_average_row(row, settings)
        if cond is None:
            continue
        mean_counts = np.asarray(getattr(row, "psth_mean"), dtype=float).reshape(-1)
        if mean_counts.size != n_bins:
            continue
        if np.any(~np.isfinite(mean_counts)):
            continue
        if not hasattr(row, "psth_sem"):
            continue
        sem_counts = np.asarray(getattr(row, "psth_sem"), dtype=float).reshape(-1)
        if sem_counts.size != n_bins:
            continue
        if np.any(~np.isfinite(sem_counts)):
            continue
        grouped[cond].append(
            {
                "mean_counts": mean_counts,
                "sem_counts": sem_counts,
                "n_trials": _resolve_n_trials_for_average_row(row),
            }
        )

    out: dict[str, dict] = {}
    for cond, records in grouped.items():
        combined = _combine_average_records(records)
        if combined is None:
            continue
        mean_counts, sem_counts, n_trials = combined
        out[cond] = {
            "mean_hz": mean_counts / float(bin_size_s),
            "sem_hz": sem_counts / float(bin_size_s),
            "trace_bin_centers": np.asarray(bin_centers, dtype=float),
            "trace_n_trials": int(n_trials),
        }
    return out


def _build_unit_condition_payloads(
    df_unit: pd.DataFrame,
    *,
    unit_key: str,
    trace_bin_centers: np.ndarray,
    trace_bin_size_s: float,
    raster_bin_centers: np.ndarray,
    raster_bin_size_s: float,
    settings: FixationPSTHUnitPlotSettings,
) -> list[dict]:
    masks = _condition_masks(
        df_unit,
        interactive_label=settings.interactive_label,
        face_label=settings.face_label,
        object_label=settings.object_label,
    )
    n_bins = int(trace_bin_centers.size)
    raster_n_bins = int(raster_bin_centers.size)

    payloads: list[dict] = []
    sigma_bins = _resolve_plot_sigma_bins(settings, trace_bin_size_s)

    for cond_key, cond_label in _PLOT_CONDITION_ORDER:
        cond_df = df_unit.loc[masks[cond_key]].copy()
        seed = _stable_seed_shared(settings.random_seed, unit_key, cond_key)
        cond_df = _sample_rows_shared(cond_df, settings.max_trials_per_condition, seed)

        count_rows: list[np.ndarray] = []
        spike_rows: list[np.ndarray] = []
        for trial_i, row in enumerate(cond_df.itertuples(index=False)):
            counts = _row_counts_shared(row, n_bins)
            if counts is None:
                continue
            count_rows.append(counts)
            spike_counts = _row_spike_train_counts_shared(row, raster_n_bins)
            if spike_counts is not None:
                spike_bin_centers = raster_bin_centers
                spike_bin_size_s = raster_bin_size_s
            else:
                spike_counts = counts
                spike_bin_centers = trace_bin_centers
                spike_bin_size_s = trace_bin_size_s
            trial_rng = np.random.default_rng(
                _stable_seed_shared(settings.random_seed, unit_key, cond_key, str(trial_i)),
            )
            spike_rows.append(
                _counts_to_spike_times_shared(
                    spike_counts,
                    spike_bin_centers,
                    spike_bin_size_s,
                    jitter_within_bin=settings.raster_jitter_within_bin,
                    rng=trial_rng,
                )
            )

        if not count_rows:
            payloads.append(
                {
                    "key": cond_key,
                    "label": cond_label,
                    "color": settings.condition_colors.get(cond_key, "#444444"),
                    "n_trials": 0,
                    "spike_rows": [],
                    "mean_hz": np.zeros(n_bins, dtype=float),
                    "sem_hz": np.zeros(n_bins, dtype=float),
                    "trace_bin_centers": np.asarray(trace_bin_centers, dtype=float),
                    "trace_n_trials": 0,
                },
            )
            continue

        mat = np.vstack(count_rows)
        rates_hz = mat / float(trace_bin_size_s)
        if settings.smooth_before_average:
            rates_hz = gaussian_filter1d(rates_hz, sigma=sigma_bins, axis=1, mode="nearest")
        mean_hz = np.mean(rates_hz, axis=0)
        if rates_hz.shape[0] > 1:
            sem_hz = np.std(rates_hz, axis=0, ddof=1) / np.sqrt(float(rates_hz.shape[0]))
        else:
            sem_hz = np.zeros(n_bins, dtype=float)

        payloads.append(
            {
                "key": cond_key,
                "label": cond_label,
                "color": settings.condition_colors.get(cond_key, "#444444"),
                "n_trials": int(rates_hz.shape[0]),
                "spike_rows": spike_rows,
                "mean_hz": mean_hz,
                "sem_hz": sem_hz,
                "trace_bin_centers": np.asarray(trace_bin_centers, dtype=float),
                "trace_n_trials": int(rates_hz.shape[0]),
            },
        )

    if bool(settings.use_precomputed_average_traces):
        if "|" in str(unit_key):
            date_token, unit_token = str(unit_key).split("|", 1)
        else:
            date_token, unit_token = "", str(unit_key)
        date_token = str(date_token).strip()
        unit_token = str(unit_token).strip()
        resolved_overrides: dict[str, dict] = {}
        if date_token and unit_token:
            split_bundle = _load_average_trace_bundle_for_date(
                settings,
                date=date_token,
                for_object=False,
            )
            object_bundle = _load_average_trace_bundle_for_date(
                settings,
                date=date_token,
                for_object=True,
            )

            split_overrides: dict[str, dict] = {}
            if split_bundle is not None:
                split_df, split_centers, split_bin_size_s = split_bundle
                split_overrides = _build_precomputed_trace_overrides(
                    split_df,
                    unit_uuid=unit_token,
                    bin_centers=split_centers,
                    bin_size_s=split_bin_size_s,
                    settings=settings,
                )

            object_overrides: dict[str, dict] = {}
            if object_bundle is not None:
                object_df, object_centers, object_bin_size_s = object_bundle
                object_overrides = _build_precomputed_trace_overrides(
                    object_df,
                    unit_uuid=unit_token,
                    bin_centers=object_centers,
                    bin_size_s=object_bin_size_s,
                    settings=settings,
                )

            for payload in payloads:
                cond_key = str(payload["key"])
                if cond_key == "object":
                    override = object_overrides.get(cond_key) or split_overrides.get(cond_key)
                else:
                    override = split_overrides.get(cond_key)
                if override is None:
                    continue
                resolved_overrides[cond_key] = override
                payload["mean_hz"] = np.asarray(override["mean_hz"], dtype=float)
                payload["sem_hz"] = np.asarray(override["sem_hz"], dtype=float)
                payload["trace_bin_centers"] = np.asarray(
                    override["trace_bin_centers"],
                    dtype=float,
                )
                payload["trace_n_trials"] = int(override["trace_n_trials"])

        if not bool(settings.allow_trial_trace_fallback):
            for payload in payloads:
                if str(payload["key"]) in resolved_overrides:
                    continue
                mean_hz = np.asarray(payload["mean_hz"], dtype=float).reshape(-1)
                payload["mean_hz"] = np.full(mean_hz.shape, np.nan, dtype=float)
                payload["sem_hz"] = np.full(mean_hz.shape, np.nan, dtype=float)

    return payloads


def _resolve_plot_sigma_bins(settings: FixationPSTHUnitPlotSettings, bin_size_s: float) -> Optional[float]:
    if not settings.smooth_before_average:
        return None
    if float(settings.smoothing_sigma_ms) <= 0:
        raise ValueError("plot smoothing_sigma_ms must be > 0 when smoothing is enabled.")
    sigma_bins = float(settings.smoothing_sigma_ms) / (float(bin_size_s) * 1000.0)
    if sigma_bins <= 0:
        raise ValueError("resolved smoothing sigma in bins must be > 0.")
    return sigma_bins


def _collect_condition_rate_mats(
    df_unit: pd.DataFrame,
    *,
    bin_size_s: float,
    n_bins: int,
    settings: FixationPSTHUnitPlotSettings,
) -> dict[str, np.ndarray]:
    masks = _condition_masks(
        df_unit,
        interactive_label=settings.interactive_label,
        face_label=settings.face_label,
        object_label=settings.object_label,
    )
    sigma_bins = _resolve_plot_sigma_bins(settings, bin_size_s)
    out: dict[str, np.ndarray] = {}
    for cond in ("face_interactive", "face_non_interactive", "object"):
        cond_df = df_unit.loc[masks[cond]].copy()
        rows: list[np.ndarray] = []
        for row in cond_df.itertuples(index=False):
            counts = _row_counts_shared(row, n_bins)
            if counts is None:
                continue
            rows.append(counts / float(bin_size_s))
        if not rows:
            out[cond] = np.zeros((0, n_bins), dtype=float)
            continue
        mat = np.vstack(rows)
        if settings.smooth_before_average:
            mat = gaussian_filter1d(mat, sigma=sigma_bins, axis=1, mode="nearest")
        out[cond] = mat
    return out


def _pair_significance_masks(
    df_unit: pd.DataFrame,
    *,
    bin_centers: np.ndarray,
    bin_size_s: float,
    settings: FixationPSTHUnitPlotSettings,
) -> list[dict]:
    n_bins = int(bin_centers.size)
    mats = _collect_condition_rate_mats(
        df_unit,
        bin_size_s=bin_size_s,
        n_bins=n_bins,
        settings=settings,
    )
    pair_defs = [
        ("face_interactive", "face_non_interactive", "Int vs Non-Int Face", "#5E3C99"),
        ("face_interactive", "object", "Int Face vs Object", "#1B7837"),
        ("face_non_interactive", "object", "Non-Int Face vs Object", "#B35806"),
    ]
    results: list[dict] = []
    for cond_a, cond_b, label, color in pair_defs:
        mat_a = mats.get(cond_a, np.zeros((0, n_bins), dtype=float))
        mat_b = mats.get(cond_b, np.zeros((0, n_bins), dtype=float))
        if (
            mat_a.shape[0] < int(settings.significance_min_trials_per_condition)
            or mat_b.shape[0] < int(settings.significance_min_trials_per_condition)
        ):
            mask = np.zeros(n_bins, dtype=bool)
            results.append({"label": label, "color": color, "mask": mask, "n_a": int(mat_a.shape[0]), "n_b": int(mat_b.shape[0])})
            continue

        if str(settings.significance_test).lower() == "welch_ttest":
            _, p_vals = welch_ttest(mat_a, mat_b, axis=0)
            p_vals = np.asarray(p_vals, dtype=float).reshape(-1)
        elif str(settings.significance_test).lower() == "mannwhitney":
            p_vals = mannwhitneyu_pvalues_per_column(mat_a, mat_b)
        else:
            raise ValueError(
                f"Unsupported significance_test '{settings.significance_test}'. "
                "Use 'welch_ttest' or 'mannwhitney'."
            )
        mask = np.isfinite(p_vals) & (p_vals < float(settings.significance_alpha))
        results.append({"label": label, "color": color, "mask": mask, "n_a": int(mat_a.shape[0]), "n_b": int(mat_b.shape[0])})
    return results


def _safe_unit_filename(unit_uuid: str) -> str:
    return _safe_unit_filename_shared(unit_uuid)


def _safe_region_folder(region: Optional[str]) -> str:
    return _safe_region_folder_shared(region)


def _darken_color(hex_color: str, factor: float) -> str:
    return _darken_color_shared(hex_color, factor)


def _ensure_csv_filename(filename: str) -> str:
    token = str(filename).strip()
    if not token:
        token = "unit_selectivity.csv"
    if not token.endswith(".csv"):
        token = f"{token}.csv"
    return token


def _normalize_selectivity_date_str(val) -> str:
    if val is None:
        return ""
    token = str(val).strip()
    if not token:
        return ""
    if token.endswith(".0"):
        token = token[:-2]
    if token.isdigit():
        return token.zfill(8)
    try:
        intval = int(float(token))
        return str(intval).zfill(8)
    except Exception:
        return token


def _coerce_selective_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except Exception:
        pass
    token = str(val).strip().lower()
    return token in {"1", "true", "t", "yes", "y"}


def _load_selective_unit_lookup(settings: FixationPSTHUnitPlotSettings) -> dict[str, bool]:
    if not bool(settings.segregate_selective_units):
        return {}

    cfg = load_config(settings.cfg_path)
    unit_summary_path = (
        build_analysis_output_dir(cfg, settings.selectivity_input_subdir)
        / _ensure_csv_filename(settings.selectivity_unit_summary_filename)
    )
    if not unit_summary_path.exists():
        print(
            "[plot] selective-unit summary not found; plotting all units as non-selective. "
            f"path={unit_summary_path}"
        )
        return {}

    df = pd.read_csv(unit_summary_path)
    if df.empty:
        return {}
    if "unit_key" not in df.columns:
        if {"date", "unit_uuid"}.issubset(df.columns):
            df = df.copy()
            df["date"] = df["date"].map(_normalize_selectivity_date_str)
            df["unit_uuid"] = df["unit_uuid"].astype(str).map(str.strip)
            df["unit_key"] = df["date"].astype(str) + "|" + df["unit_uuid"].astype(str)
        else:
            print(
                "[plot] selective-unit summary is missing unit identifiers; "
                "plotting all units as non-selective."
            )
            return {}
    if "is_selective_unit" not in df.columns:
        print(
            "[plot] selective-unit summary is missing 'is_selective_unit'; "
            "plotting all units as non-selective."
        )
        return {}

    lookup: dict[str, bool] = {}
    for row in df.itertuples(index=False):
        unit_key = str(getattr(row, "unit_key", "")).strip()
        if not unit_key:
            continue
        is_selective = _coerce_selective_bool(getattr(row, "is_selective_unit", False))
        lookup[unit_key] = bool(lookup.get(unit_key, False) or is_selective)
    return lookup


def _format_half_window_label(half_window_s: float) -> str:
    if np.isclose(float(half_window_s), round(float(half_window_s))):
        return f"+/-{int(round(float(half_window_s)))} s"
    return f"+/-{float(half_window_s):g} s"


def _resolve_display_windows(
    settings: FixationPSTHUnitPlotSettings,
    *,
    left_bound_s: float,
    right_bound_s: float,
) -> list[tuple[float, float, str]]:
    windows: list[tuple[float, float, str]] = []
    seen: set[float] = set()
    for raw in settings.display_half_windows_s:
        try:
            half_window_s = float(raw)
        except Exception:
            continue
        if not np.isfinite(half_window_s) or half_window_s <= 0.0:
            continue
        key = round(half_window_s, 9)
        if key in seen:
            continue
        seen.add(key)
        x_min = max(float(left_bound_s), -half_window_s)
        x_max = min(float(right_bound_s), half_window_s)
        if x_max <= x_min:
            continue
        windows.append((x_min, x_max, _format_half_window_label(half_window_s)))
    if windows:
        return windows
    return [(float(left_bound_s), float(right_bound_s), "Full Window")]


def _window_spike_rows(
    spike_rows: Sequence[np.ndarray],
    *,
    x_min: float,
    x_max: float,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for row in spike_rows:
        spikes = np.asarray(row, dtype=float).reshape(-1)
        if spikes.size == 0:
            out.append(spikes)
            continue
        mask = np.isfinite(spikes) & (spikes >= float(x_min)) & (spikes <= float(x_max))
        out.append(spikes[mask])
    return out


def _window_trace_arrays(
    trace_bin_centers: np.ndarray,
    mean_hz: np.ndarray,
    sem_hz: np.ndarray,
    *,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(trace_bin_centers, dtype=float).reshape(-1)
    mean_vals = np.asarray(mean_hz, dtype=float).reshape(-1)
    sem_vals = np.asarray(sem_hz, dtype=float).reshape(-1)
    if centers.size != mean_vals.size or centers.size != sem_vals.size:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    mask = (
        np.isfinite(centers)
        & np.isfinite(mean_vals)
        & np.isfinite(sem_vals)
        & (centers >= float(x_min))
        & (centers <= float(x_max))
    )
    return centers[mask], mean_vals[mask], sem_vals[mask]


def _add_analysis_window_overlays(
    ax,
    settings: FixationPSTHUnitPlotSettings,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    if not bool(settings.show_analysis_window_overlays):
        return
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return

    colors = [str(color).strip() for color in settings.analysis_window_overlay_colors if str(color).strip()]
    default_color = colors[-1] if colors else "#808080"
    for idx, bounds in enumerate(settings.analysis_window_overlays_s):
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        try:
            start_s = float(bounds[0])
            stop_s = float(bounds[1])
        except Exception:
            continue
        if not np.isfinite(start_s) or not np.isfinite(stop_s):
            continue
        left = max(min(start_s, stop_s), float(x_min))
        right = min(max(start_s, stop_s), float(x_max))
        if right <= left:
            continue
        color = colors[idx] if idx < len(colors) else default_color
        ax.add_patch(
            Rectangle(
                (left, float(y_min)),
                right - left,
                float(y_max) - float(y_min),
                fill=False,
                edgecolor=color,
                linestyle=str(settings.analysis_window_overlay_linestyle),
                linewidth=float(settings.analysis_window_overlay_linewidth),
                zorder=0.5,
            )
        )


def _plot_single_unit(
    *,
    df_unit: pd.DataFrame,
    date: str,
    unit_uuid: str,
    trace_bin_centers: np.ndarray,
    raster_bin_centers: np.ndarray,
    settings: FixationPSTHUnitPlotSettings,
    out_dir: Path,
    figsize: list[float],
    dpi: Optional[int],
) -> Optional[Path]:
    if trace_bin_centers.size < 2:
        return None
    trace_bin_size_s = _resolve_bin_duration_from_centers(
        trace_bin_centers,
        settings.bin_size_ms_fallback,
    )
    raster_bin_size_s = _resolve_bin_duration_from_centers(
        raster_bin_centers,
        settings.bin_size_ms_fallback,
    )
    if trace_bin_size_s <= 0 or raster_bin_size_s <= 0:
        return None

    unit_key = f"{date}|{unit_uuid}"
    payloads = _build_unit_condition_payloads(
        df_unit,
        unit_key=unit_key,
        trace_bin_centers=trace_bin_centers,
        trace_bin_size_s=trace_bin_size_s,
        raster_bin_centers=raster_bin_centers,
        raster_bin_size_s=raster_bin_size_s,
        settings=settings,
    )
    if not any(payload["n_trials"] > 0 for payload in payloads):
        return None

    trace_finite = np.asarray(trace_bin_centers, dtype=float)
    trace_finite = trace_finite[np.isfinite(trace_finite)]
    raster_finite = np.asarray(raster_bin_centers, dtype=float)
    raster_finite = raster_finite[np.isfinite(raster_finite)]
    if trace_finite.size == 0 or raster_finite.size == 0:
        return None

    left_bound_s = max(float(np.min(trace_finite)), float(np.min(raster_finite)))
    right_bound_s = min(float(np.max(trace_finite)), float(np.max(raster_finite)))
    if right_bound_s <= left_bound_s:
        return None
    panel_windows = _resolve_display_windows(
        settings,
        left_bound_s=left_bound_s,
        right_bound_s=right_bound_s,
    )
    n_panels = len(panel_windows)
    if n_panels <= 0:
        return None

    fig, axes = plt.subplots(
        2,
        n_panels,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharey="row",
        gridspec_kw={
            "height_ratios": [
                float(settings.panel_raster_height_ratio),
                float(settings.panel_rate_height_ratio),
            ],
            "hspace": 0.08,
            "wspace": 0.14,
        },
    )
    ax_rasters = axes[0, :]
    ax_rates = axes[1, :]

    condition_rows: list[tuple[dict, np.ndarray]] = []
    y_cursor = 1.0
    y_ticks: list[float] = []
    y_labels: list[str] = []
    for payload in payloads:
        n_trials = int(payload["n_trials"])
        if n_trials <= 0:
            continue
        line_offsets = np.arange(y_cursor, y_cursor + n_trials, dtype=float)
        condition_rows.append((payload, line_offsets))
        mid = 0.5 * (line_offsets[0] + line_offsets[-1])
        y_ticks.append(float(mid))
        y_labels.append(f"{payload['label']} (n={n_trials})")
        y_cursor += n_trials

    raster_ylim = (float(y_cursor) - 0.5, 0.5) if y_ticks else (1.0, 0.0)
    for panel_idx, (x_min, x_max, label) in enumerate(panel_windows):
        ax_raster = ax_rasters[panel_idx]
        ax_rate = ax_rates[panel_idx]

        for payload, line_offsets in condition_rows:
            n_trials = int(payload["n_trials"])
            raster_collections = ax_raster.eventplot(
                _window_spike_rows(
                    payload["spike_rows"],
                    x_min=float(x_min),
                    x_max=float(x_max),
                ),
                lineoffsets=line_offsets,
                linelengths=float(settings.raster_linelength),
                linewidths=float(settings.raster_linewidth),
                colors=[_darken_color(payload["color"], settings.raster_darkening_factor)] * n_trials,
                alpha=float(settings.raster_alpha),
                zorder=3,
            )
            if not isinstance(raster_collections, (list, tuple)):
                raster_collections = [raster_collections]
            for collection in raster_collections:
                collection.set_clip_on(True)
                collection.set_clip_path(ax_raster.patch)
            if settings.raster_show_condition_background:
                ax_raster.axhspan(
                    float(line_offsets[0]) - 0.5,
                    float(line_offsets[-1]) + 0.5,
                    color=payload["color"],
                    alpha=0.07,
                    zorder=0,
                )
            ax_raster.axhline(float(line_offsets[-1]) + 0.5, color="#cccccc", linewidth=0.7)

        ax_raster.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax_raster.set_xlim(float(x_min), float(x_max))
        ax_raster.set_title(label)
        if y_ticks:
            ax_raster.set_ylim(*raster_ylim)
            if panel_idx == 0:
                ax_raster.set_ylabel("Trials")
                ax_raster.set_yticks(y_ticks)
                ax_raster.set_yticklabels(y_labels)
            else:
                ax_raster.set_yticks(y_ticks)
                ax_raster.tick_params(axis="y", labelleft=False)
        else:
            ax_raster.set_yticks([])
        ax_raster.tick_params(axis="x", labelbottom=False)

        for payload in payloads:
            if int(payload["n_trials"]) <= 0:
                continue
            payload_trace_bin_centers = np.asarray(
                payload.get("trace_bin_centers", trace_bin_centers),
                dtype=float,
            ).reshape(-1)
            mean_hz = np.asarray(payload["mean_hz"], dtype=float)
            sem_hz = np.asarray(payload["sem_hz"], dtype=float)
            if payload_trace_bin_centers.size != mean_hz.size or sem_hz.size != mean_hz.size:
                continue
            panel_trace_bin_centers, panel_mean_hz, panel_sem_hz = _window_trace_arrays(
                payload_trace_bin_centers,
                mean_hz,
                sem_hz,
                x_min=float(x_min),
                x_max=float(x_max),
            )
            if panel_trace_bin_centers.size == 0:
                continue
            line = ax_rate.plot(
                panel_trace_bin_centers,
                panel_mean_hz,
                color=payload["color"],
                label=payload["label"],
            )[0]
            line.set_clip_on(True)
            line.set_clip_path(ax_rate.patch)
            band = ax_rate.fill_between(
                panel_trace_bin_centers,
                panel_mean_hz - panel_sem_hz,
                panel_mean_hz + panel_sem_hz,
                color=payload["color"],
                alpha=0.22,
                linewidth=0.0,
            )
            band.set_clip_on(True)
            band.set_clip_path(ax_rate.patch)
        ax_rate.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=3)
        ax_rate.grid(True, alpha=0.16, linewidth=0.45)
        ax_rate.set_xlim(float(x_min), float(x_max))
        ax_rate.set_xlabel("Time From Fixation Start (s)")
        if panel_idx == 0:
            ax_rate.set_ylabel("Firing Rate (Hz)")
            ax_rate.legend(loc="upper right", frameon=False)
        else:
            ax_rate.tick_params(axis="y", labelleft=False)

    rate_y_lims = [ax.get_ylim() for ax in ax_rates]
    base_y_min = min(float(bounds[0]) for bounds in rate_y_lims)
    base_y_max = max(float(bounds[1]) for bounds in rate_y_lims)
    if not np.isfinite(base_y_min) or not np.isfinite(base_y_max) or np.isclose(base_y_min, base_y_max):
        base_y_min, base_y_max = (-0.5, 0.5)
    for ax in ax_rates:
        ax.set_ylim(base_y_min, base_y_max)

    for ax, (x_min, x_max, _) in zip(ax_rates, panel_windows):
        _add_analysis_window_overlays(
            ax,
            settings,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=base_y_min,
            y_max=base_y_max,
        )

    if settings.show_significance_ticks:
        pair_masks = _pair_significance_masks(
            df_unit,
            bin_centers=trace_bin_centers,
            bin_size_s=trace_bin_size_s,
            settings=settings,
        )
        span = max(1e-6, float(base_y_max - base_y_min))
        row_gap = float(settings.significance_tick_row_gap_ratio) * span
        tick_h = float(settings.significance_tick_height_ratio) * span
        n_rows = len(pair_masks)
        sig_y_min = base_y_min - (row_gap * (n_rows + 1.4))
        for panel_idx, (ax_rate, (x_min, x_max, _)) in enumerate(zip(ax_rates, panel_windows)):
            ax_rate.set_ylim(sig_y_min, base_y_max)
            for idx, pair in enumerate(pair_masks):
                y0 = base_y_min - row_gap * float(n_rows - idx)
                sig_x = trace_bin_centers[np.asarray(pair["mask"], dtype=bool)]
                sig_x = sig_x[(sig_x >= float(x_min)) & (sig_x <= float(x_max))]
                if sig_x.size > 0:
                    ax_rate.vlines(
                        sig_x,
                        y0,
                        y0 + tick_h,
                        color=pair["color"],
                        linewidth=0.8,
                        alpha=0.95,
                    )
                if panel_idx == (n_panels - 1):
                    y_frac = (y0 + 0.5 * tick_h - sig_y_min) / max(1e-6, base_y_max - sig_y_min)
                    ax_rate.text(
                        1.01,
                        float(y_frac),
                        f"{pair['label']} (p<{settings.significance_alpha:g})",
                        transform=ax_rate.transAxes,
                        ha="left",
                        va="center",
                        fontsize=8,
                        color=pair["color"],
                    )
            if panel_idx == (n_panels - 1):
                ax_rate.text(
                    0.0,
                    -0.22,
                    "Significance ticks: per-bin category-pair FR difference",
                    transform=ax_rate.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#333333",
                )

    row0 = df_unit.iloc[0]
    region = _safe_optional_str_shared(row0.get("region")) if isinstance(row0, pd.Series) else None
    channel = _safe_optional_str_shared(row0.get("spike_channel")) if isinstance(row0, pd.Series) else None
    title_bits = [f"Date {date}", f"Unit {unit_uuid}"]
    if region:
        title_bits.append(f"Region {region}")
    if channel:
        title_bits.append(f"Channel {channel}")
    fig.suptitle(" | ".join(title_bits), y=0.995)

    ext = _ensure_ext(settings.output_extension)
    out_path = out_dir / f"date={date}__unit={_safe_unit_filename(unit_uuid)}.{ext}"
    fig.patch.set_facecolor("white")
    for ax in list(ax_rasters) + list(ax_rates):
        ax.set_facecolor("white")
    save_figure(
        fig,
        out_path,
        ext=ext,
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)
    return out_path


def _plot_single_unit_worker(args) -> Optional[Path]:
    df_unit, date, unit_uuid, trace_bin_centers, raster_bin_centers, settings, out_dir, figsize, dpi = args
    return _plot_single_unit(
        df_unit=df_unit,
        date=date,
        unit_uuid=unit_uuid,
        trace_bin_centers=trace_bin_centers,
        raster_bin_centers=raster_bin_centers,
        settings=settings,
        out_dir=out_dir,
        figsize=figsize,
        dpi=dpi,
    )


def _build_unit_plot_tasks_for_date(args):
    settings, date, primary_paths, raster_paths, unit_filter, unit_key_filter, selective_lookup = args
    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    df, trace_bin_centers, raster_bin_centers = _load_trials_for_date(
        primary_paths,
        raster_paths=raster_paths,
        date=date,
        settings=settings,
    )
    if df.empty or "unit_uuid" not in df.columns:
        return []

    unit_ids = sorted({str(val) for val in df["unit_uuid"].dropna().astype(str).tolist()})
    if unit_filter is not None:
        unit_ids = [unit for unit in unit_ids if unit in unit_filter]
    if settings.test_single and unit_ids:
        unit_ids = [random.choice(unit_ids)]
    if not unit_ids:
        return []

    figsize, dpi = _resolve_figsize_and_dpi(settings)
    unit_tasks = []
    for unit_uuid in unit_ids:
        unit_key = f"{date}|{unit_uuid}"
        if unit_key_filter is not None and unit_key not in unit_key_filter:
            continue
        df_unit = df.loc[df["unit_uuid"].astype(str) == unit_uuid].copy()
        if df_unit.empty:
            continue
        region_series = (
            df_unit["region"].dropna().astype(str).map(lambda text: text.strip())
            if "region" in df_unit.columns
            else pd.Series(dtype=str)
        )
        region = None
        if not region_series.empty:
            region = next((val for val in region_series if val), None)
        unit_out_dir = out_root / _safe_region_folder(region)
        if bool(settings.segregate_selective_units) and bool(selective_lookup.get(unit_key, False)):
            unit_out_dir = unit_out_dir / str(settings.selective_unit_subfolder)
        if settings.example_units_subfolder:
            unit_out_dir = unit_out_dir / str(settings.example_units_subfolder)
        unit_tasks.append(
            (
                df_unit,
                date,
                unit_uuid,
                trace_bin_centers,
                raster_bin_centers,
                settings,
                unit_out_dir,
                figsize,
                dpi,
            )
        )
    return unit_tasks


def plot_fixation_psth_units(
    settings: FixationPSTHUnitPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    unit_uuids: Optional[Sequence[str]] = None,
    unit_keys: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Generate one multiscale raster + average firing-rate PSTH figure per unit/date."""
    cfg = load_config(settings.cfg_path)
    primary_trial_rows = _iter_trial_rows(cfg, settings, dates=dates, sessions=sessions)
    if not primary_trial_rows:
        print("[plot] no fixation PSTH trial files found")
        return []

    grouped_primary: dict[str, list[Path]] = {}
    for row in primary_trial_rows:
        grouped_primary.setdefault(str(row["date"]), []).append(Path(row["path"]))

    grouped_raster: dict[str, list[Path]] = {}
    raster_input = _resolve_raster_trial_input(settings)
    if raster_input is not None:
        raster_modality, raster_filename = raster_input
        if (
            str(raster_modality) != str(settings.trial_input_modality)
            or str(raster_filename) != str(settings.trial_input_filename)
        ):
            raster_trial_rows = _iter_trial_rows(
                cfg,
                settings,
                dates=dates,
                sessions=sessions,
                modality=raster_modality,
                filename=raster_filename,
            )
            for row in raster_trial_rows:
                grouped_raster.setdefault(str(row["date"]), []).append(Path(row["path"]))

    tasks = sorted(grouped_primary.items(), key=lambda item: item[0])

    unit_filter = None if unit_uuids is None else {str(unit) for unit in unit_uuids}
    unit_key_filter = None if unit_keys is None else {str(key) for key in unit_keys}
    selective_lookup = _load_selective_unit_lookup(settings)
    out_paths: list[Path] = []

    all_unit_tasks = []
    for date, primary_paths in tasks:
        all_unit_tasks.extend(
            _build_unit_plot_tasks_for_date(
                (
                    settings,
                    date,
                    primary_paths,
                    grouped_raster.get(date),
                    unit_filter,
                    unit_key_filter,
                    selective_lookup,
                )
            )
        )
    if settings.test_single and all_unit_tasks:
        all_unit_tasks = [random.choice(all_unit_tasks)]
    if not all_unit_tasks:
        return []

    use_global_unit_parallel = (
        settings.use_parallel
        and settings.parallelize_units
        and len(all_unit_tasks) >= int(settings.unit_parallel_min_units)
    )
    if use_global_unit_parallel:
        n_proc = get_n_processes()
        with Pool(processes=n_proc) as pool:
            for out_path in tqdm(
                pool.imap_unordered(_plot_single_unit_worker, all_unit_tasks),
                total=len(all_unit_tasks),
                desc=f"Plotting unit PSTHs ({n_proc} workers)",
                unit="unit",
            ):
                if out_path is not None:
                    out_paths.append(out_path)
        return out_paths

    for task in tqdm(all_unit_tasks, desc="Plotting unit PSTHs", unit="unit"):
        out_path = _plot_single_unit_worker(task)
        if out_path is not None:
            out_paths.append(out_path)
    return out_paths
