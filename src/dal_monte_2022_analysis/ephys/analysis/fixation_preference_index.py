"""Compute per-bin fixation preference indices for unit condition pairs."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
    ensure_filename as _ensure_filename,
    extract_trials_df_and_meta as _extract_trials_df_and_meta,
    resolve_bin_centers_from_meta as _resolve_bin_centers_from_meta,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_selectivity import DEFAULT_CONDITION_PAIRS
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
    scan_processed_paths_for_filename,
)
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


_SUPPORTED_CONDITIONS = ("face_interactive", "face_non_interactive", "object")
_TRUE_TOKENS = {"1", "true", "t", "yes", "y"}

DEFAULT_INDEX_NAME_BY_PAIR: dict[tuple[str, str], str] = {
    ("face_interactive", "face_non_interactive"): "interactive_face_preference_index",
    ("face_interactive", "object"): "interactive_face_vs_object_index",
    ("face_non_interactive", "object"): "non_interactive_face_vs_object_index",
}


@dataclass
class FixationPSTHPreferenceIndexSettings:
    """Configuration for fixation-pair per-bin preference-index analysis."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations.pkl"
    selectivity_input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    pair_summary_filename: str = "pair_selectivity.csv"
    unit_summary_filename: str = "unit_selectivity.csv"
    output_subdir: str = "ephys/psth/fixation_psth_preference_index"
    timeseries_filename: str = "preference_index_timeseries.csv"
    output_pickle_filename: str = "results.pkl"
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    condition_pairs: tuple[tuple[str, str], ...] = field(
        default_factory=lambda: tuple(DEFAULT_CONDITION_PAIRS),
    )
    pair_index_name_overrides: dict[str, str] = field(default_factory=dict)
    denominator_epsilon: float = 0.0
    use_parallel: bool = True
    max_procs: int = 16
    test_single: bool = False
    bin_size_ms_fallback: float = 10.0
    window_pre_s_fallback: float = 1.0
    window_post_s_fallback: float = 1.0


def _coerce_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return False
        return float(value) != 0.0
    token = str(value).strip().lower()
    if not token or token == "nan":
        return False
    return token in _TRUE_TOKENS


def _fallback_bin_centers(settings: FixationPSTHPreferenceIndexSettings) -> np.ndarray:
    bin_size_s = float(settings.bin_size_ms_fallback) / 1000.0
    pre = float(settings.window_pre_s_fallback)
    post = float(settings.window_post_s_fallback)
    edges = np.arange(-pre, post + bin_size_s * 0.5, bin_size_s, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _validate_condition_pairs(settings: FixationPSTHPreferenceIndexSettings) -> None:
    for cond_a, cond_b in settings.condition_pairs:
        if cond_a not in _SUPPORTED_CONDITIONS or cond_b not in _SUPPORTED_CONDITIONS:
            raise ValueError(
                f"Unsupported condition pair ({cond_a}, {cond_b}). "
                f"Supported conditions: {', '.join(_SUPPORTED_CONDITIONS)}."
            )
        if cond_a == cond_b:
            raise ValueError(f"Condition pair ({cond_a}, {cond_b}) must compare two different conditions.")


def _pair_label(cond_a: str, cond_b: str) -> str:
    return f"{cond_a}__vs__{cond_b}"


def _resolve_pair_index_name(
    cond_a: str,
    cond_b: str,
    *,
    overrides: dict[str, str],
) -> str:
    pair_key = _pair_label(cond_a, cond_b)
    override = overrides.get(pair_key)
    if override is not None:
        text = str(override).strip()
        if text:
            return text
    return DEFAULT_INDEX_NAME_BY_PAIR.get((cond_a, cond_b), f"{pair_key}__preference_index")


def _load_trial_table(
    settings: FixationPSTHPreferenceIndexSettings,
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


def _resolve_condition_for_row(
    row,
    settings: FixationPSTHPreferenceIndexSettings,
) -> Optional[str]:
    category = str(getattr(row, "fixation_category", "")).strip()
    if category == settings.object_label:
        return "object"
    if category != settings.face_label:
        return None

    interactive = False
    if hasattr(row, "is_interactive"):
        interactive = _as_bool(getattr(row, "is_interactive"), settings.interactive_label)
    elif hasattr(row, "interactive_state"):
        interactive = _as_bool(getattr(row, "interactive_state"), settings.interactive_label)
    return "face_interactive" if interactive else "face_non_interactive"


def _build_unit_key(df: pd.DataFrame) -> pd.Series:
    date = df["date"].astype(str).map(lambda value: value.strip())
    unit = df["unit_uuid"].astype(str).map(lambda value: value.strip())
    return date + "|" + unit


def _read_selectivity_significance(
    settings: FixationPSTHPreferenceIndexSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(settings.cfg_path)
    in_root = build_analysis_output_dir(cfg, settings.selectivity_input_subdir)
    pair_path = in_root / _ensure_filename(settings.pair_summary_filename, ".csv")
    unit_path = in_root / _ensure_filename(settings.unit_summary_filename, ".csv")
    if not pair_path.exists():
        raise FileNotFoundError(f"Pair selectivity CSV not found: {pair_path}")
    if not unit_path.exists():
        raise FileNotFoundError(f"Unit selectivity CSV not found: {unit_path}")

    pair_df = pd.read_csv(pair_path)
    unit_df = pd.read_csv(unit_path)

    if "pair_label" not in pair_df.columns:
        raise ValueError("pair_selectivity.csv missing required column 'pair_label'.")
    if "is_selective_pair" not in pair_df.columns:
        raise ValueError("pair_selectivity.csv missing required column 'is_selective_pair'.")
    if "is_selective_unit" not in unit_df.columns:
        raise ValueError("unit_selectivity.csv missing required column 'is_selective_unit'.")

    if "unit_key" not in pair_df.columns:
        required = {"date", "unit_uuid"}
        if not required.issubset(pair_df.columns):
            raise ValueError("pair_selectivity.csv requires 'unit_key' or ('date', 'unit_uuid').")
        pair_df = pair_df.copy()
        pair_df["unit_key"] = _build_unit_key(pair_df)
    else:
        pair_df = pair_df.copy()
        pair_df["unit_key"] = pair_df["unit_key"].astype(str).map(lambda value: value.strip())

    if "unit_key" not in unit_df.columns:
        required = {"date", "unit_uuid"}
        if not required.issubset(unit_df.columns):
            raise ValueError("unit_selectivity.csv requires 'unit_key' or ('date', 'unit_uuid').")
        unit_df = unit_df.copy()
        unit_df["unit_key"] = _build_unit_key(unit_df)
    else:
        unit_df = unit_df.copy()
        unit_df["unit_key"] = unit_df["unit_key"].astype(str).map(lambda value: value.strip())

    pair_df["pair_label"] = pair_df["pair_label"].astype(str).map(lambda value: value.strip())
    pair_df["is_selective_pair"] = pair_df["is_selective_pair"].map(_coerce_bool)

    pair_keep = ["unit_key", "pair_label", "is_selective_pair"]
    for column in ("n_significant_windows", "n_tested_windows", "min_p_value"):
        if column in pair_df.columns:
            pair_keep.append(column)
    pair_sig_df = pair_df[pair_keep].copy()
    pair_sig_df = (
        pair_sig_df
        .groupby(["unit_key", "pair_label"], dropna=False, as_index=False)
        .agg(
            {
                "is_selective_pair": "max",
                **{
                    column: "max"
                    for column in pair_keep
                    if column not in {"unit_key", "pair_label", "is_selective_pair"}
                },
            }
        )
    )

    unit_df["is_selective_unit"] = unit_df["is_selective_unit"].map(_coerce_bool)
    if "selective_pairs" not in unit_df.columns:
        unit_df["selective_pairs"] = ""
    else:
        unit_df["selective_pairs"] = unit_df["selective_pairs"].fillna("").astype(str)

    unit_keep = ["unit_key", "is_selective_unit", "selective_pairs"]
    if "n_selective_pairs" in unit_df.columns:
        unit_keep.append("n_selective_pairs")
    unit_sig_df = unit_df[unit_keep].copy()
    unit_sig_df = (
        unit_sig_df
        .groupby("unit_key", dropna=False, as_index=False)
        .agg(
            {
                "is_selective_unit": "max",
                "selective_pairs": "first",
                **{
                    column: "max"
                    for column in unit_keep
                    if column not in {"unit_key", "is_selective_unit", "selective_pairs"}
                },
            }
        )
    )

    return pair_sig_df, unit_sig_df


def _unit_worker(args) -> pd.DataFrame:
    unit_key, df_unit, bin_centers_s, settings = args
    n_bins = int(bin_centers_s.size)
    if n_bins <= 0:
        return pd.DataFrame()

    if n_bins > 1:
        bin_size_s = float(np.mean(np.diff(bin_centers_s)))
    else:
        bin_size_s = float(settings.bin_size_ms_fallback) / 1000.0
    if not np.isfinite(bin_size_s) or bin_size_s <= 0:
        raise ValueError("Unable to infer positive bin size for preference-index analysis.")

    row0 = df_unit.iloc[0]
    date = str(row0.get("date"))
    unit_uuid = str(row0.get("unit_uuid"))
    region = _as_optional_str(row0.get("region"))
    spike_channel = _as_optional_str(row0.get("spike_channel"))
    recorded_agent = _as_optional_str(row0.get("recorded_agent"))
    recorded_monkey = _as_optional_str(row0.get("recorded_monkey"))
    area = _as_optional_str(row0.get("area"))
    n_sessions = int(df_unit["session"].nunique()) if "session" in df_unit.columns else 0

    condition_rows: dict[str, list[np.ndarray]] = {cond: [] for cond in _SUPPORTED_CONDITIONS}
    for row in df_unit.itertuples(index=False):
        condition = _resolve_condition_for_row(row, settings)
        if condition is None:
            continue
        counts = np.asarray(getattr(row, "psth_counts"), dtype=float).reshape(-1)
        if counts.size != n_bins:
            continue
        condition_rows[condition].append(counts / float(bin_size_s))

    condition_mean_hz: dict[str, np.ndarray] = {}
    condition_n_trials: dict[str, int] = {}
    for condition, rows in condition_rows.items():
        if rows:
            mat = np.vstack(rows)
            condition_mean_hz[condition] = np.mean(mat, axis=0)
            condition_n_trials[condition] = int(mat.shape[0])
        else:
            condition_mean_hz[condition] = np.full(n_bins, np.nan, dtype=float)
            condition_n_trials[condition] = 0

    rows_out: list[pd.DataFrame] = []
    eps = max(0.0, float(settings.denominator_epsilon))
    for cond_a, cond_b in settings.condition_pairs:
        mean_a = condition_mean_hz.get(cond_a, np.full(n_bins, np.nan, dtype=float))
        mean_b = condition_mean_hz.get(cond_b, np.full(n_bins, np.nan, dtype=float))
        numerator = mean_a - mean_b
        denominator = mean_a + mean_b
        valid = np.isfinite(denominator) & (np.abs(denominator) > eps)
        pref = np.full(n_bins, np.nan, dtype=float)
        pref[valid] = numerator[valid] / denominator[valid]

        rows_out.append(
            pd.DataFrame(
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
                    "pair_label": _pair_label(cond_a, cond_b),
                    "index_name": _resolve_pair_index_name(
                        cond_a,
                        cond_b,
                        overrides=settings.pair_index_name_overrides,
                    ),
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n_trials_a": int(condition_n_trials.get(cond_a, 0)),
                    "n_trials_b": int(condition_n_trials.get(cond_b, 0)),
                    "bin_index": np.arange(n_bins, dtype=int),
                    "bin_center_s": np.asarray(bin_centers_s, dtype=float),
                    "mean_fr_a_hz": np.asarray(mean_a, dtype=float),
                    "mean_fr_b_hz": np.asarray(mean_b, dtype=float),
                    "difference_fr_hz": np.asarray(numerator, dtype=float),
                    "sum_fr_hz": np.asarray(denominator, dtype=float),
                    "preference_index": np.asarray(pref, dtype=float),
                    "index_valid": np.asarray(valid, dtype=bool),
                }
            )
        )

    if not rows_out:
        return pd.DataFrame()
    return pd.concat(rows_out, axis=0, ignore_index=True)


def run_fixation_preference_index_analysis(
    settings: FixationPSTHPreferenceIndexSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    unit_uuids: Optional[Sequence[str]] = None,
) -> dict:
    """Run per-bin fixation preference-index analysis for each configured pair."""
    _validate_condition_pairs(settings)
    trial_df, bin_centers_s = _load_trial_table(settings, dates=dates, sessions=sessions)
    if trial_df.empty or "unit_uuid" not in trial_df.columns:
        print("[analysis] no trial PSTH rows found for fixation preference-index analysis")
        return {
            "timeseries": pd.DataFrame(),
            "pair_significance": pd.DataFrame(),
            "unit_significance": pd.DataFrame(),
        }

    if unit_uuids is not None:
        allowed_units = {str(unit).strip() for unit in unit_uuids}
        trial_df = trial_df.loc[trial_df["unit_uuid"].astype(str).map(lambda value: value.strip()).isin(allowed_units)]
    if trial_df.empty:
        print("[analysis] no matching units found after unit filter")
        return {
            "timeseries": pd.DataFrame(),
            "pair_significance": pd.DataFrame(),
            "unit_significance": pd.DataFrame(),
        }

    if "date" not in trial_df.columns:
        trial_df["date"] = "unknown"

    grouped = trial_df.groupby(["date", "unit_uuid"], sort=True, dropna=False)
    unit_tasks = []
    for (date, unit_uuid), df_unit in grouped:
        unit_key = f"{date}|{unit_uuid}"
        unit_tasks.append((unit_key, df_unit.copy(), np.asarray(bin_centers_s, dtype=float), settings))

    if settings.test_single and unit_tasks:
        unit_tasks = [random.choice(unit_tasks)]

    unit_results = run_tasks(
        _unit_worker,
        unit_tasks,
        desc="Fixation preference index",
        unit="unit",
        use_parallel=settings.use_parallel,
        max_procs=settings.max_procs,
    )
    if unit_results:
        timeseries_df = pd.concat(unit_results, axis=0, ignore_index=True)
    else:
        timeseries_df = pd.DataFrame()

    pair_sig_df, unit_sig_df = _read_selectivity_significance(settings)
    if not timeseries_df.empty:
        timeseries_df = timeseries_df.merge(pair_sig_df, on=["unit_key", "pair_label"], how="left")
        timeseries_df = timeseries_df.merge(unit_sig_df, on=["unit_key"], how="left")
        if "is_selective_pair" not in timeseries_df.columns:
            timeseries_df["is_selective_pair"] = False
        if "is_selective_unit" not in timeseries_df.columns:
            timeseries_df["is_selective_unit"] = False
        timeseries_df["is_selective_pair"] = timeseries_df["is_selective_pair"].map(_coerce_bool)
        timeseries_df["is_selective_unit"] = timeseries_df["is_selective_unit"].map(_coerce_bool)
        if "selective_pairs" not in timeseries_df.columns:
            timeseries_df["selective_pairs"] = ""
        timeseries_df["selective_pairs"] = timeseries_df["selective_pairs"].fillna("").astype(str)
        timeseries_df["is_selective_any_pair"] = timeseries_df["is_selective_unit"].astype(bool)
        timeseries_df = (
            timeseries_df
            .sort_values(["date", "region", "unit_uuid", "pair_label", "bin_index"])
            .reset_index(drop=True)
        )

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / _ensure_filename(settings.timeseries_filename, ".csv")
    out_pkl = out_root / _ensure_filename(settings.output_pickle_filename, ".pkl")

    timeseries_df.to_csv(out_csv, index=False)
    result_obj = {
        "meta": {
            "condition_pairs": [list(pair) for pair in settings.condition_pairs],
            "pair_index_name_overrides": dict(settings.pair_index_name_overrides),
            "default_index_name_by_pair": {
                _pair_label(cond_a, cond_b): index_name
                for (cond_a, cond_b), index_name in DEFAULT_INDEX_NAME_BY_PAIR.items()
            },
            "trial_input_modality": settings.trial_input_modality,
            "trial_input_filename": _ensure_filename(settings.trial_input_filename, ".pkl"),
            "selectivity_input_subdir": settings.selectivity_input_subdir,
            "pair_summary_filename": _ensure_filename(settings.pair_summary_filename, ".csv"),
            "unit_summary_filename": _ensure_filename(settings.unit_summary_filename, ".csv"),
            "denominator_epsilon": float(settings.denominator_epsilon),
            "n_rows": int(len(timeseries_df)),
            "n_units": int(timeseries_df["unit_key"].nunique()) if not timeseries_df.empty else 0,
        },
        "timeseries": timeseries_df,
        "pair_significance": pair_sig_df,
        "unit_significance": unit_sig_df,
    }
    save_pickle_path(result_obj, out_pkl)
    return result_obj
