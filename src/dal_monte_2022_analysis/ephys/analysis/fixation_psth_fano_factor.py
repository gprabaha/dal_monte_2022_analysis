"""Compute trial-level fixation PSTH Fano-factor summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
    extract_trials_df_and_meta as _extract_trials_df_and_meta,
    resolve_bin_centers_from_meta as _resolve_bin_centers_from_meta,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
    scan_processed_paths_for_filename,
)
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_CONDITIONS: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
_ALLOWED_UNIT_SUBSETS: frozenset[str] = frozenset({"all_units", "any_selective_unit"})
_UNIT_ARRAY_COLUMNS: tuple[str, ...] = (
    "date",
    "unit_uuid",
    "unit_key",
    "region",
    "spike_channel",
    "recorded_agent",
    "recorded_monkey",
    "area",
    "condition",
    "is_selective_unit",
    "n_selective_pairs",
    "n_trials",
    "n_valid_bins",
    "mean_count",
    "variance_count",
    "fano_factor",
)
_UNIT_TIMESERIES_COLUMNS: tuple[str, ...] = (
    "date",
    "unit_uuid",
    "unit_key",
    "region",
    "spike_channel",
    "recorded_agent",
    "recorded_monkey",
    "area",
    "condition",
    "is_selective_unit",
    "n_selective_pairs",
    "n_trials",
    "bin_index",
    "bin_center_s_rel",
    "mean_count",
    "variance_count",
    "fano_factor",
)
_REGION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "region",
    "condition",
    "bin_index",
    "bin_center_s_rel",
    "mean_fano_factor",
    "sem_fano_factor",
    "n_units",
)


@dataclass
class FixationPSTHFanoFactorSettings:
    """Configuration for fixation PSTH Fano-factor analysis."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations_psth_10ms.pkl"
    output_subdir: str = "ephys/psth/fixation_psth_fano_factor"
    unit_timeseries_filename: str = "unit_fano_factor_timeseries.csv"
    region_summary_filename: str = "region_fano_factor_summary.csv"
    output_pickle_filename: Optional[str] = "results.pkl"
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    selectivity_input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    selectivity_unit_summary_filename: str = "unit_selectivity.csv"
    region_summary_unit_subset: str = "any_selective_unit"
    conditions: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_CONDITIONS))
    min_trials_per_condition: int = 2
    variance_ddof: int = 1
    mean_epsilon: float = 1e-12
    bin_size_ms_fallback: float = 10.0
    window_pre_s_fallback: float = 1.0
    window_post_s_fallback: float = 1.0
    verbose_logging: bool = True


def _empty_unit_array_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_UNIT_ARRAY_COLUMNS))


def _empty_unit_timeseries_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_UNIT_TIMESERIES_COLUMNS))


def _empty_region_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_REGION_SUMMARY_COLUMNS))


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


def _fallback_bin_centers(settings: FixationPSTHFanoFactorSettings, n_bins: int) -> np.ndarray:
    if int(n_bins) <= 0:
        return np.asarray([], dtype=float)
    bin_size_s = float(settings.bin_size_ms_fallback) / 1000.0
    start_center_s = -float(settings.window_pre_s_fallback) + 0.5 * bin_size_s
    return start_center_s + np.arange(int(n_bins), dtype=float) * bin_size_s


def _normalize_unit_subset_mode(value: object) -> str:
    token = _norm_token(value)
    aliases = {
        "all": "all_units",
        "all_unit": "all_units",
        "all_units": "all_units",
        "allunits": "all_units",
        "selective": "any_selective_unit",
        "selective_units": "any_selective_unit",
        "significant": "any_selective_unit",
        "sig": "any_selective_unit",
        "sig_units": "any_selective_unit",
        "any_selective": "any_selective_unit",
        "any_selective_unit": "any_selective_unit",
        "anyselectiveunit": "any_selective_unit",
    }
    resolved = aliases.get(token, token)
    if resolved not in _ALLOWED_UNIT_SUBSETS:
        raise ValueError(
            f"Unsupported region-summary unit subset '{value}'. "
            f"Expected one of: {sorted(_ALLOWED_UNIT_SUBSETS)}"
        )
    return resolved


def _load_selectivity_annotations(
    settings: FixationPSTHFanoFactorSettings,
) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.selectivity_input_subdir)
        / ensure_filename(settings.selectivity_unit_summary_filename, ".csv")
    )
    subset_mode = _normalize_unit_subset_mode(settings.region_summary_unit_subset)
    if not in_path.exists():
        if subset_mode == "any_selective_unit":
            raise FileNotFoundError(
                "Fixation selectivity unit-summary CSV is required for "
                f"region_summary_unit_subset='{subset_mode}', but was not found: {in_path}"
            )
        return pd.DataFrame(columns=["unit_key", "is_selective_unit", "n_selective_pairs"])

    df = pd.read_csv(in_path)
    if df.empty:
        if subset_mode == "any_selective_unit":
            raise ValueError(
                "Fixation selectivity unit-summary CSV is empty but selective-unit "
                f"region summaries were requested: {in_path}"
            )
        return pd.DataFrame(columns=["unit_key", "is_selective_unit", "n_selective_pairs"])

    required = {"unit_key", "is_selective_unit"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Fixation selectivity unit-summary CSV is missing required columns: "
            + ", ".join(missing)
        )

    out = df.copy()
    out["unit_key"] = out["unit_key"].astype(str).map(str.strip)
    out["is_selective_unit"] = out["is_selective_unit"].map(
        lambda value: bool(_as_bool(value, settings.interactive_label))
    )
    if "n_selective_pairs" in out.columns:
        out["n_selective_pairs"] = pd.to_numeric(
            out["n_selective_pairs"],
            errors="coerce",
        ).fillna(0).astype(int)
    else:
        out["n_selective_pairs"] = 0

    out = (
        out.groupby("unit_key", as_index=False, dropna=False)
        .agg(
            {
                "is_selective_unit": "max",
                "n_selective_pairs": "max",
            }
        )
        .reset_index(drop=True)
    )
    return out.loc[:, ["unit_key", "is_selective_unit", "n_selective_pairs"]]


def _resolve_condition_from_trial_row(
    row,
    settings: FixationPSTHFanoFactorSettings,
) -> Optional[str]:
    category_token = _norm_token(getattr(row, "fixation_category", None))
    if not category_token or category_token == "nan":
        return None

    if category_token in {_norm_token(settings.object_label), "object", "objects"}:
        return "object"
    if category_token in {"out_of_roi", "outofroi"}:
        return None
    if category_token not in {
        _norm_token(settings.face_label),
        "face",
        "face_interactive",
        "face_non_interactive",
        "face_noninteractive",
        "int_face",
        "nonint_face",
    }:
        return None

    if category_token in {"face_interactive", "int_face"}:
        return "face_interactive"
    if category_token in {"face_non_interactive", "face_noninteractive", "nonint_face"}:
        return "face_non_interactive"

    interactive_state = getattr(row, "interactive_state", None)
    is_interactive = getattr(row, "is_interactive", None)
    has_interactive_state = interactive_state is not None and not pd.isna(interactive_state)
    has_is_interactive = is_interactive is not None and not pd.isna(is_interactive)
    if not has_interactive_state and not has_is_interactive:
        raise ValueError(
            "Face fixation trial rows are missing interactive-state labels. "
            "This analysis requires interactive and non-interactive face trials."
        )

    if has_is_interactive:
        interactive = _as_bool(is_interactive, settings.interactive_label)
    else:
        interactive = _as_bool(interactive_state, settings.interactive_label)
    return "face_interactive" if interactive else "face_non_interactive"


def _accumulator_template(n_bins: int) -> dict[str, object]:
    return {
        "region": None,
        "spike_channel": None,
        "recorded_agent": None,
        "recorded_monkey": None,
        "area": None,
        "sum_counts": np.zeros(int(n_bins), dtype=float),
        "sumsq_counts": np.zeros(int(n_bins), dtype=float),
        "n_trials": 0,
    }


def _merge_bucket_meta(bucket: dict[str, object], *, region: Optional[str], spike_channel: Optional[str], recorded_agent: Optional[str], recorded_monkey: Optional[str], area: Optional[str]) -> None:
    current_region = _as_optional_str(bucket.get("region"))
    if current_region is None:
        bucket["region"] = region
    elif region is not None and region != current_region:
        raise ValueError(
            f"Encountered inconsistent regions for one unit-condition bucket: "
            f"existing={current_region}, new={region}."
        )
    for key, value in (
        ("spike_channel", spike_channel),
        ("recorded_agent", recorded_agent),
        ("recorded_monkey", recorded_monkey),
        ("area", area),
    ):
        if bucket.get(key) is None and value is not None:
            bucket[key] = value


def _load_unit_array_df(
    settings: FixationPSTHFanoFactorSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    cfg = load_config(settings.cfg_path)
    rows = scan_processed_paths_for_filename(
        cfg,
        settings.trial_input_modality,
        filename=ensure_filename(settings.trial_input_filename, ".pkl"),
        dates=dates,
        sessions=sessions,
        agents=[None],
    )
    if settings.verbose_logging:
        print(
            "[analysis] fixation PSTH Fano trial-input scan: "
            f"modality={settings.trial_input_modality}, "
            f"filename={settings.trial_input_filename}, "
            f"date_filter={list(dates) if dates is not None else 'all'}, "
            f"session_filter={list(sessions) if sessions is not None else 'all'}, "
            f"matched_files={len(rows)}"
        )
    if not rows:
        return _empty_unit_array_df(), np.asarray([], dtype=float), {"trial_files": 0, "unit_conditions": 0, "skipped_small_n": 0}

    accumulators: dict[tuple[str, str, str], dict[str, object]] = {}
    bin_centers_ref: Optional[np.ndarray] = None
    n_bins_ref: Optional[int] = None

    for row in rows:
        obj = load_pickle_path(row["path"])
        trial_df, meta = _extract_trials_df_and_meta(obj)
        if trial_df.empty or "psth_counts" not in trial_df.columns:
            continue

        local_centers = _resolve_bin_centers_from_meta(meta)
        if local_centers is not None:
            if bin_centers_ref is None:
                bin_centers_ref = np.asarray(local_centers, dtype=float).reshape(-1)
            elif (
                local_centers.shape != bin_centers_ref.shape
                or not np.allclose(local_centers, bin_centers_ref)
            ):
                raise ValueError(
                    f"Encountered mismatched 10 ms PSTH bin centers across trial files; path={row['path']}"
                )

        for trial_row in trial_df.itertuples(index=False):
            condition = _resolve_condition_from_trial_row(trial_row, settings)
            if condition is None or condition not in set(settings.conditions):
                continue

            counts = np.asarray(getattr(trial_row, "psth_counts"), dtype=float).reshape(-1)
            if counts.size == 0:
                continue
            if n_bins_ref is None:
                n_bins_ref = int(counts.size)
            elif int(counts.size) != int(n_bins_ref):
                raise ValueError(
                    f"Encountered inconsistent PSTH count lengths across trial files; path={row['path']}"
                )

            date = (
                _normalize_date_token(getattr(trial_row, "date", None))
                or _normalize_date_token(row.get("date"))
                or str(row.get("date", ""))
            )
            unit_uuid = _as_optional_str(getattr(trial_row, "unit_uuid", None))
            if unit_uuid is None:
                continue

            key = (str(date), str(unit_uuid), str(condition))
            bucket = accumulators.setdefault(key, _accumulator_template(int(counts.size)))
            _merge_bucket_meta(
                bucket,
                region=_as_optional_str(getattr(trial_row, "region", None)) or "unknown",
                spike_channel=_as_optional_str(getattr(trial_row, "spike_channel", None)),
                recorded_agent=_as_optional_str(getattr(trial_row, "recorded_agent", None)),
                recorded_monkey=_as_optional_str(getattr(trial_row, "recorded_monkey", None)),
                area=_as_optional_str(getattr(trial_row, "area", None)),
            )
            bucket["sum_counts"] = np.asarray(bucket["sum_counts"], dtype=float) + counts
            bucket["sumsq_counts"] = np.asarray(bucket["sumsq_counts"], dtype=float) + np.square(counts)
            bucket["n_trials"] = int(bucket["n_trials"]) + 1

    if not accumulators:
        return _empty_unit_array_df(), np.asarray([], dtype=float), {"trial_files": len(rows), "unit_conditions": 0, "skipped_small_n": 0}

    if bin_centers_ref is None:
        if n_bins_ref is None:
            return _empty_unit_array_df(), np.asarray([], dtype=float), {"trial_files": len(rows), "unit_conditions": 0, "skipped_small_n": 0}
        bin_centers_ref = _fallback_bin_centers(settings, int(n_bins_ref))

    ddof = max(int(settings.variance_ddof), 0)
    min_trials = max(int(settings.min_trials_per_condition), ddof + 1)
    mean_eps = max(float(settings.mean_epsilon), 0.0)

    out_rows: list[dict] = []
    skipped_small_n = 0
    for (date, unit_uuid, condition), bucket in sorted(accumulators.items()):
        n_trials = int(bucket["n_trials"])
        if n_trials < min_trials:
            skipped_small_n += 1
            continue

        sum_counts = np.asarray(bucket["sum_counts"], dtype=float).reshape(-1)
        sumsq_counts = np.asarray(bucket["sumsq_counts"], dtype=float).reshape(-1)
        mean_count = sum_counts / float(n_trials)
        if n_trials <= ddof:
            variance_count = np.full(mean_count.shape, np.nan, dtype=float)
        else:
            numer = sumsq_counts - (np.square(sum_counts) / float(n_trials))
            numer = np.maximum(numer, 0.0)
            variance_count = numer / float(n_trials - ddof)

        fano_factor = np.full(mean_count.shape, np.nan, dtype=float)
        valid = (
            np.isfinite(mean_count)
            & np.isfinite(variance_count)
            & (mean_count > mean_eps)
        )
        fano_factor[valid] = variance_count[valid] / mean_count[valid]

        out_rows.append(
            {
                "date": str(date),
                "unit_uuid": str(unit_uuid),
                "unit_key": f"{date}|{unit_uuid}",
                "region": _as_optional_str(bucket.get("region")) or "unknown",
                "spike_channel": _as_optional_str(bucket.get("spike_channel")),
                "recorded_agent": _as_optional_str(bucket.get("recorded_agent")),
                "recorded_monkey": _as_optional_str(bucket.get("recorded_monkey")),
                "area": _as_optional_str(bucket.get("area")),
                "condition": str(condition),
                "n_trials": n_trials,
                "n_valid_bins": int(np.isfinite(fano_factor).sum()),
                "mean_count": mean_count,
                "variance_count": variance_count,
                "fano_factor": fano_factor,
            }
        )

    out_df = pd.DataFrame(out_rows)
    selectivity_df = _load_selectivity_annotations(settings)
    if out_df.empty:
        out_df = _empty_unit_array_df()
    else:
        if not selectivity_df.empty:
            out_df = out_df.merge(selectivity_df, on="unit_key", how="left")
        if "is_selective_unit" not in out_df.columns:
            out_df["is_selective_unit"] = False
        out_df["is_selective_unit"] = out_df["is_selective_unit"].fillna(False).map(bool)
        if "n_selective_pairs" not in out_df.columns:
            out_df["n_selective_pairs"] = 0
        out_df["n_selective_pairs"] = pd.to_numeric(
            out_df["n_selective_pairs"],
            errors="coerce",
        ).fillna(0).astype(int)
        out_df = out_df.loc[:, list(_UNIT_ARRAY_COLUMNS)]
    if not out_df.empty:
        out_df = out_df.sort_values(["region", "unit_key", "condition"]).reset_index(drop=True)

    return out_df, np.asarray(bin_centers_ref, dtype=float).reshape(-1), {
        "trial_files": len(rows),
        "unit_conditions": int(len(out_df)),
        "skipped_small_n": int(skipped_small_n),
        "n_selective_units": (
            int(out_df.loc[out_df["is_selective_unit"], "unit_key"].astype(str).nunique())
            if not out_df.empty
            else 0
        ),
    }


def _expand_unit_timeseries_df(
    unit_array_df: pd.DataFrame,
    *,
    bin_centers_s: np.ndarray,
) -> pd.DataFrame:
    if unit_array_df.empty:
        return _empty_unit_timeseries_df()

    centers = np.asarray(bin_centers_s, dtype=float).reshape(-1)
    n_bins = int(centers.size)
    out_frames: list[pd.DataFrame] = []

    for row in unit_array_df.itertuples(index=False):
        mean_count = np.asarray(getattr(row, "mean_count"), dtype=float).reshape(-1)
        variance_count = np.asarray(getattr(row, "variance_count"), dtype=float).reshape(-1)
        fano_factor = np.asarray(getattr(row, "fano_factor"), dtype=float).reshape(-1)
        if mean_count.size != n_bins or variance_count.size != n_bins or fano_factor.size != n_bins:
            raise ValueError("Encountered inconsistent Fano-factor vector lengths while expanding unit timeseries.")

        out_frames.append(
            pd.DataFrame(
                {
                    "date": str(getattr(row, "date")),
                    "unit_uuid": str(getattr(row, "unit_uuid")),
                    "unit_key": str(getattr(row, "unit_key")),
                    "region": str(getattr(row, "region")),
                    "spike_channel": getattr(row, "spike_channel"),
                    "recorded_agent": getattr(row, "recorded_agent"),
                    "recorded_monkey": getattr(row, "recorded_monkey"),
                    "area": getattr(row, "area"),
                    "condition": str(getattr(row, "condition")),
                    "is_selective_unit": bool(getattr(row, "is_selective_unit")),
                    "n_selective_pairs": int(getattr(row, "n_selective_pairs")),
                    "n_trials": int(getattr(row, "n_trials")),
                    "bin_index": np.arange(n_bins, dtype=int),
                    "bin_center_s_rel": centers,
                    "mean_count": mean_count,
                    "variance_count": variance_count,
                    "fano_factor": fano_factor,
                }
            )
        )

    if not out_frames:
        return _empty_unit_timeseries_df()

    out_df = pd.concat(out_frames, axis=0, ignore_index=True)
    return out_df.loc[:, list(_UNIT_TIMESERIES_COLUMNS)]


def _summarize_region_timeseries(
    unit_array_df: pd.DataFrame,
    *,
    bin_centers_s: np.ndarray,
    settings: FixationPSTHFanoFactorSettings,
) -> pd.DataFrame:
    if unit_array_df.empty:
        return _empty_region_summary_df()

    subset_mode = _normalize_unit_subset_mode(settings.region_summary_unit_subset)
    summary_input_df = unit_array_df
    if subset_mode == "any_selective_unit":
        summary_input_df = unit_array_df.loc[unit_array_df["is_selective_unit"].map(bool)].copy()
    if summary_input_df.empty:
        return _empty_region_summary_df()

    centers = np.asarray(bin_centers_s, dtype=float).reshape(-1)
    n_bins = int(centers.size)
    rows: list[dict] = []

    grouped = summary_input_df.groupby(["region", "condition"], dropna=False, sort=True)
    for (region, condition), group_df in grouped:
        stack = np.vstack(
            [
                np.asarray(value, dtype=float).reshape(-1)
                for value in group_df["fano_factor"].tolist()
            ]
        )
        if stack.shape[1] != n_bins:
            raise ValueError("Encountered inconsistent Fano-factor lengths while summarizing regions.")

        mean_vec = np.nanmean(stack, axis=0)
        sem_vec = np.full(n_bins, np.nan, dtype=float)
        n_units_vec = np.zeros(n_bins, dtype=int)
        for idx in range(n_bins):
            vals = stack[:, idx]
            finite = vals[np.isfinite(vals)]
            n_units_vec[idx] = int(finite.size)
            if finite.size == 0:
                continue
            if finite.size == 1:
                sem_vec[idx] = 0.0
            else:
                sem_vec[idx] = float(np.std(finite, ddof=1) / np.sqrt(float(finite.size)))

        rows.extend(
            {
                "region": str(region),
                "condition": str(condition),
                "bin_index": int(idx),
                "bin_center_s_rel": float(centers[idx]),
                "mean_fano_factor": float(mean_vec[idx]) if np.isfinite(mean_vec[idx]) else np.nan,
                "sem_fano_factor": float(sem_vec[idx]) if np.isfinite(sem_vec[idx]) else np.nan,
                "n_units": int(n_units_vec[idx]),
            }
            for idx in range(n_bins)
        )

    out_df = pd.DataFrame(rows, columns=list(_REGION_SUMMARY_COLUMNS))
    if out_df.empty:
        return _empty_region_summary_df()
    return out_df.sort_values(["region", "condition", "bin_index"]).reset_index(drop=True)


def run_fixation_psth_fano_factor_analysis(
    settings: FixationPSTHFanoFactorSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> dict:
    """Compute per-unit and per-region fixation PSTH Fano-factor summaries."""

    unit_array_df, bin_centers_s, counts = _load_unit_array_df(
        settings,
        dates=dates,
        sessions=sessions,
    )
    unit_timeseries_df = _expand_unit_timeseries_df(unit_array_df, bin_centers_s=bin_centers_s)
    region_summary_df = _summarize_region_timeseries(
        unit_array_df,
        bin_centers_s=bin_centers_s,
        settings=settings,
    )

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    unit_timeseries_path = out_root / ensure_filename(settings.unit_timeseries_filename, ".csv")
    region_summary_path = out_root / ensure_filename(settings.region_summary_filename, ".csv")

    unit_timeseries_df.to_csv(unit_timeseries_path, index=False)
    region_summary_df.to_csv(region_summary_path, index=False)

    result = {
        "unit_summary": unit_array_df,
        "region_summary": region_summary_df,
        "bin_centers_s_rel": np.asarray(bin_centers_s, dtype=float).reshape(-1),
        "meta": {
            "conditions": [str(condition) for condition in settings.conditions],
            "trial_input_modality": str(settings.trial_input_modality),
            "trial_input_filename": ensure_filename(settings.trial_input_filename, ".pkl"),
            "min_trials_per_condition": max(int(settings.min_trials_per_condition), int(settings.variance_ddof) + 1),
            "variance_ddof": int(settings.variance_ddof),
            "mean_epsilon": float(settings.mean_epsilon),
            "region_summary_unit_subset": _normalize_unit_subset_mode(settings.region_summary_unit_subset),
            "n_trial_files": int(counts["trial_files"]),
            "n_unit_conditions": int(counts["unit_conditions"]),
            "n_units": int(unit_array_df["unit_key"].astype(str).nunique()) if not unit_array_df.empty else 0,
            "n_selective_units": int(counts["n_selective_units"]),
            "n_regions": int(unit_array_df["region"].astype(str).nunique()) if not unit_array_df.empty else 0,
            "n_bins": int(np.asarray(bin_centers_s).size),
            "skipped_small_n_unit_conditions": int(counts["skipped_small_n"]),
        },
        "unit_timeseries_path": str(unit_timeseries_path),
        "region_summary_path": str(region_summary_path),
        "pickle_path": None,
    }

    if settings.output_pickle_filename is not None and str(settings.output_pickle_filename).strip():
        pickle_path = out_root / ensure_filename(str(settings.output_pickle_filename), ".pkl")
        save_pickle_path(result, pickle_path)
        result["pickle_path"] = str(pickle_path)

    if settings.verbose_logging:
        condition_counts = (
            unit_array_df["condition"].astype(str).value_counts().sort_index().to_dict()
            if not unit_array_df.empty
            else {}
        )
        print(
            "[analysis] fixation PSTH Fano summary: "
            f"units={result['meta']['n_units']}, "
            f"regions={result['meta']['n_regions']}, "
            f"unit_conditions={result['meta']['n_unit_conditions']}, "
            f"condition_counts={condition_counts}, "
            f"skipped_small_n={result['meta']['skipped_small_n_unit_conditions']}"
        )
        print(f"[analysis] wrote unit timeseries: {unit_timeseries_path}")
        print(f"[analysis] wrote region summary: {region_summary_path}")
        if result["pickle_path"] is not None:
            print(f"[analysis] wrote results pickle: {result['pickle_path']}")

    return result
