"""Plot per-unit fixation PSTH rasters and mean firing-rate traces."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    extract_trials_df_and_meta as _extract_trials_df_and_meta_shared,
    resolve_bin_centers_from_meta as _resolve_bin_centers_from_meta_shared,
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
)
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_date_paths
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure
from dal_monte_2022_analysis.runtime.execution.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


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
    use_precomputed_average_traces: bool = True
    average_trace_input_subdir: str = "ephys/psth/fixation_psth_averages"
    average_trace_input_filename: str = "fixations.pkl"
    allow_trial_trace_fallback: bool = True
    output_subdir: str = "ephys/psth/fixation_psth_unit_plots"
    output_extension: str = "pdf"
    example_units_subfolder: Optional[str] = None
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
    bin_size_ms_fallback: float = 10.0
    window_pre_s: float = 1.0
    window_post_s: float = 1.0


def _ensure_ext(ext: str) -> str:
    return _ensure_ext_shared(ext, fallback="pdf")


def _iter_trial_rows(
    cfg: dict,
    settings: FixationPSTHUnitPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    return _iter_trial_rows_shared(
        cfg,
        modality=settings.trial_input_modality,
        filename=settings.trial_input_filename,
        dates=dates,
        sessions=sessions,
    )


def _fallback_bin_centers(settings: FixationPSTHUnitPlotSettings) -> np.ndarray:
    return _fallback_bin_centers_shared(
        bin_size_ms_fallback=settings.bin_size_ms_fallback,
        window_pre_s=settings.window_pre_s,
        window_post_s=settings.window_post_s,
    )


def _load_trials_for_date(
    paths: Sequence[Path],
    *,
    date: str,
    settings: FixationPSTHUnitPlotSettings,
) -> tuple[pd.DataFrame, np.ndarray]:
    all_rows: list[pd.DataFrame] = []
    bin_centers_ref: Optional[np.ndarray] = None

    for path in paths:
        obj = load_pickle_path(path)
        trials_df, meta = _extract_trials_df_and_meta_shared(obj)
        if trials_df.empty or "psth_counts" not in trials_df.columns:
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
        return pd.DataFrame(), _fallback_bin_centers(settings)

    out_df = pd.concat(all_rows, axis=0, ignore_index=True)
    if bin_centers_ref is None:
        bin_centers_ref = _fallback_bin_centers(settings)
    return out_df, bin_centers_ref


def _resolve_figsize_and_dpi(settings: FixationPSTHUnitPlotSettings) -> tuple[list[float], Optional[int]]:
    return _resolve_figsize_and_dpi_shared(
        plotting_cfg_path=settings.plotting_cfg_path,
        output_dpi=settings.output_dpi,
        default_figsize=[11.0, 8.0],
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


def _load_average_trace_bundle_for_date(
    settings: FixationPSTHUnitPlotSettings,
    *,
    date: str,
) -> Optional[tuple[pd.DataFrame, np.ndarray, float]]:
    if not bool(settings.use_precomputed_average_traces):
        return None

    subdir = str(settings.average_trace_input_subdir).strip()
    if not subdir:
        return None

    filename = _ensure_pkl_filename(settings.average_trace_input_filename)
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
        avg_df, meta = _extract_average_df_and_meta(obj)
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
    bin_centers: np.ndarray,
    bin_size_s: float,
    settings: FixationPSTHUnitPlotSettings,
) -> list[dict]:
    masks = _condition_masks(
        df_unit,
        interactive_label=settings.interactive_label,
        face_label=settings.face_label,
        object_label=settings.object_label,
    )
    n_bins = int(bin_centers.size)

    payloads: list[dict] = []
    sigma_bins = _resolve_plot_sigma_bins(settings, bin_size_s)

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
            trial_rng = np.random.default_rng(
                _stable_seed_shared(settings.random_seed, unit_key, cond_key, str(trial_i)),
            )
            spike_rows.append(
                _counts_to_spike_times_shared(
                    counts,
                    bin_centers,
                    bin_size_s,
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
                    "trace_bin_centers": np.asarray(bin_centers, dtype=float),
                    "trace_n_trials": 0,
                },
            )
            continue

        mat = np.vstack(count_rows)
        rates_hz = mat / float(bin_size_s)
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
                "trace_bin_centers": np.asarray(bin_centers, dtype=float),
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
        if date_token and unit_token:
            bundle = _load_average_trace_bundle_for_date(settings, date=date_token)
            if bundle is not None:
                avg_df, avg_centers, avg_bin_size_s = bundle
                overrides = _build_precomputed_trace_overrides(
                    avg_df,
                    unit_uuid=unit_token,
                    bin_centers=avg_centers,
                    bin_size_s=avg_bin_size_s,
                    settings=settings,
                )
                if overrides:
                    for payload in payloads:
                        override = overrides.get(str(payload["key"]))
                        if override is None:
                            continue
                        payload["mean_hz"] = np.asarray(override["mean_hz"], dtype=float)
                        payload["sem_hz"] = np.asarray(override["sem_hz"], dtype=float)
                        payload["trace_bin_centers"] = np.asarray(
                            override["trace_bin_centers"],
                            dtype=float,
                        )
                        payload["trace_n_trials"] = int(override["trace_n_trials"])
                if not bool(settings.allow_trial_trace_fallback):
                    for payload in payloads:
                        if str(payload["key"]) in overrides:
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
            _, p_vals = ttest_ind(mat_a, mat_b, axis=0, equal_var=False, nan_policy="omit")
            p_vals = np.asarray(p_vals, dtype=float).reshape(-1)
        elif str(settings.significance_test).lower() == "mannwhitney":
            p_vals = np.full(n_bins, np.nan, dtype=float)
            for idx in range(n_bins):
                try:
                    _, p = mannwhitneyu(mat_a[:, idx], mat_b[:, idx], alternative="two-sided")
                    p_vals[idx] = float(p)
                except Exception:
                    p_vals[idx] = np.nan
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


def _plot_single_unit(
    *,
    df_unit: pd.DataFrame,
    date: str,
    unit_uuid: str,
    bin_centers: np.ndarray,
    settings: FixationPSTHUnitPlotSettings,
    out_dir: Path,
    figsize: list[float],
    dpi: Optional[int],
) -> Optional[Path]:
    if bin_centers.size < 2:
        return None
    bin_size_s = float(np.mean(np.diff(bin_centers)))
    if bin_size_s <= 0:
        return None

    unit_key = f"{date}|{unit_uuid}"
    payloads = _build_unit_condition_payloads(
        df_unit,
        unit_key=unit_key,
        bin_centers=bin_centers,
        bin_size_s=bin_size_s,
        settings=settings,
    )
    if not any(payload["n_trials"] > 0 for payload in payloads):
        return None

    fig, (ax_raster, ax_rate) = plt.subplots(
        2,
        1,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                float(settings.panel_raster_height_ratio),
                float(settings.panel_rate_height_ratio),
            ],
            "hspace": 0.08,
        },
    )

    y_cursor = 1
    y_ticks: list[float] = []
    y_labels: list[str] = []
    for payload in payloads:
        n_trials = int(payload["n_trials"])
        if n_trials <= 0:
            continue
        line_offsets = np.arange(y_cursor, y_cursor + n_trials, dtype=float)
        ax_raster.eventplot(
            payload["spike_rows"],
            lineoffsets=line_offsets,
            linelengths=float(settings.raster_linelength),
            linewidths=float(settings.raster_linewidth),
            colors=[_darken_color(payload["color"], settings.raster_darkening_factor)] * n_trials,
            alpha=float(settings.raster_alpha),
            zorder=3,
        )
        if settings.raster_show_condition_background:
            ax_raster.axhspan(
                float(line_offsets[0]) - 0.5,
                float(line_offsets[-1]) + 0.5,
                color=payload["color"],
                alpha=0.07,
                zorder=0,
            )
        mid = 0.5 * (line_offsets[0] + line_offsets[-1])
        y_ticks.append(float(mid))
        y_labels.append(f"{payload['label']} (n={n_trials})")
        y_cursor += n_trials
        ax_raster.axhline(float(y_cursor) - 0.5, color="#cccccc", linewidth=0.7)

    ax_raster.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax_raster.set_ylabel("Trials")
    if y_ticks:
        ax_raster.set_yticks(y_ticks)
        ax_raster.set_yticklabels(y_labels)
        ax_raster.set_ylim(float(y_cursor) - 0.5, 0.5)
    else:
        ax_raster.set_yticks([])

    for payload in payloads:
        if int(payload["n_trials"]) <= 0:
            continue
        trace_bin_centers = np.asarray(
            payload.get("trace_bin_centers", bin_centers),
            dtype=float,
        ).reshape(-1)
        mean_hz = np.asarray(payload["mean_hz"], dtype=float)
        sem_hz = np.asarray(payload["sem_hz"], dtype=float)
        if trace_bin_centers.size != mean_hz.size or sem_hz.size != mean_hz.size:
            continue
        ax_rate.plot(trace_bin_centers, mean_hz, color=payload["color"], label=payload["label"])
        ax_rate.fill_between(
            trace_bin_centers,
            mean_hz - sem_hz,
            mean_hz + sem_hz,
            color=payload["color"],
            alpha=0.22,
            linewidth=0.0,
        )

    ax_rate.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax_rate.set_xlabel("Time From Fixation Start (s)")
    ax_rate.set_ylabel("Firing Rate (Hz)")
    ax_rate.set_xlim(float(bin_centers[0]), float(bin_centers[-1]))
    ax_rate.legend(loc="upper right", frameon=False)

    if settings.show_significance_ticks:
        pair_masks = _pair_significance_masks(
            df_unit,
            bin_centers=bin_centers,
            bin_size_s=bin_size_s,
            settings=settings,
        )
        y_min, y_max = ax_rate.get_ylim()
        span = max(1e-6, float(y_max - y_min))
        row_gap = float(settings.significance_tick_row_gap_ratio) * span
        tick_h = float(settings.significance_tick_height_ratio) * span
        n_rows = len(pair_masks)
        new_y_min = y_min - (row_gap * (n_rows + 1.4))
        ax_rate.set_ylim(new_y_min, y_max)

        for idx, pair in enumerate(pair_masks):
            y0 = y_min - row_gap * float(n_rows - idx)
            sig_x = bin_centers[np.asarray(pair["mask"], dtype=bool)]
            if sig_x.size > 0:
                ax_rate.vlines(sig_x, y0, y0 + tick_h, color=pair["color"], linewidth=0.8, alpha=0.95)

            y_frac = (y0 + 0.5 * tick_h - new_y_min) / max(1e-6, y_max - new_y_min)
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
    fig.suptitle(" | ".join(title_bits), y=0.99)

    ext = _ensure_ext(settings.output_extension)
    out_path = out_dir / f"date={date}__unit={_safe_unit_filename(unit_uuid)}.{ext}"
    fig.patch.set_facecolor("white")
    ax_raster.set_facecolor("white")
    ax_rate.set_facecolor("white")
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
    df_unit, date, unit_uuid, bin_centers, settings, out_dir, figsize, dpi = args
    return _plot_single_unit(
        df_unit=df_unit,
        date=date,
        unit_uuid=unit_uuid,
        bin_centers=bin_centers,
        settings=settings,
        out_dir=out_dir,
        figsize=figsize,
        dpi=dpi,
    )


def _build_unit_plot_tasks_for_date(args):
    settings, date, paths, unit_filter, unit_key_filter = args
    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    df, bin_centers = _load_trials_for_date(paths, date=date, settings=settings)
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
        if unit_key_filter is not None and f"{date}|{unit_uuid}" not in unit_key_filter:
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
        if settings.example_units_subfolder:
            unit_out_dir = unit_out_dir / str(settings.example_units_subfolder)
        unit_tasks.append(
            (
                df_unit,
                date,
                unit_uuid,
                bin_centers,
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
    """Generate one raster + average firing-rate PSTH figure per unit/date."""
    cfg = load_config(settings.cfg_path)
    trial_rows = _iter_trial_rows(cfg, settings, dates=dates, sessions=sessions)
    if not trial_rows:
        print("[plot] no fixation PSTH trial files found")
        return []

    grouped: dict[str, list[Path]] = {}
    for row in trial_rows:
        grouped.setdefault(str(row["date"]), []).append(Path(row["path"]))

    tasks = sorted(grouped.items(), key=lambda item: item[0])

    unit_filter = None if unit_uuids is None else {str(unit) for unit in unit_uuids}
    unit_key_filter = None if unit_keys is None else {str(key) for key in unit_keys}
    out_paths: list[Path] = []

    all_unit_tasks = []
    for date, paths in tasks:
        all_unit_tasks.extend(
            _build_unit_plot_tasks_for_date((settings, date, paths, unit_filter, unit_key_filter))
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
        n_proc = get_n_processes(max_procs=settings.max_procs)
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
