"""Compute fixation-level neural PSTH cross-correlations within and across regions."""

from __future__ import annotations

import pickle
import random
import re
from dataclasses import dataclass, field, replace
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


WITHIN_ANALYSIS_KIND = "within_region"
CROSS_ANALYSIS_KIND = "cross_region"
_ALLOWED_SIGNAL_TRANSFORMS = {"none", "demean", "zscore"}
_REGION_TOKEN_RE = re.compile(r"[^a-z0-9]+")
DEFAULT_FIXATION_ROI_GROUPS: dict[str, tuple[str, ...]] = {
    "face": ("face", "mouth", "eyes_nf"),
    "object": ("right_nonsocial_object", "left_nonsocial_object"),
    "out_of_roi": ("out_of_roi",),
}


@dataclass
class FixationNeuralCrossCorrelationSettings:
    """Configuration for fixation-level neural PSTH cross-correlation analysis."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations.pkl"
    within_output_subdir: str = "ephys/psth/fixation_neural_crosscorr/within_region"
    cross_output_subdir: str = "ephys/psth/fixation_neural_crosscorr/cross_region"
    within_output_filename: str = "fixations.pkl"
    cross_output_filename: str = "fixations.pkl"
    anchor_region: str = "BLA"
    partner_regions: Optional[Sequence[str]] = ("ACCg", "dmPFC", "OFC")
    include_regions: Optional[Sequence[str]] = None
    roi_groups: dict[str, Sequence[str]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_FIXATION_ROI_GROUPS.items()},
    )
    signal_transform: str = "zscore"
    max_lag: Optional[int] = None
    use_parallel: bool = True
    max_procs: int = 32
    parallelize_across_sessions: bool = True
    pair_chunk_size: int = 64
    test_single: bool = False


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _ensure_pkl_filename(name: str) -> str:
    token = str(name).strip()
    if not token:
        raise ValueError("Output filename cannot be empty.")
    return token if token.endswith(".pkl") else f"{token}.pkl"


def _as_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    token = str(value).strip()
    return token or None


def _as_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return float(value) != 0.0
    token = str(value).strip().lower()
    return token in {"1", "true", "t", "yes", "y", "interactive"}


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


def _normalize_roi_groups(groups: Optional[dict[str, Sequence[str]]]) -> dict[str, list[str]]:
    if not groups:
        return {k: [str(v) for v in vals] for k, vals in DEFAULT_FIXATION_ROI_GROUPS.items()}
    out: dict[str, list[str]] = {}
    for group_name, labels in groups.items():
        if labels is None:
            continue
        if isinstance(labels, (str, bytes)):
            label_list = [str(labels)]
        else:
            label_list = [str(label) for label in labels]
        out[str(group_name)] = [label.lower() for label in label_list if label]
    for name, labels in DEFAULT_FIXATION_ROI_GROUPS.items():
        out.setdefault(name, [str(v).lower() for v in labels])
    return out


def _canonical_fixation_category(value: Optional[str]) -> Optional[str]:
    token = _as_optional_str(value)
    if token is None:
        return None
    normalized = token.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "face": "face",
        "object": "object",
        "out_of_roi": "out_of_roi",
        "outofroi": "out_of_roi",
        "outside_roi": "out_of_roi",
    }
    return aliases.get(normalized, None)


def _infer_fixation_category_from_locations(
    locations: tuple[str, ...],
    roi_groups: dict[str, list[str]],
) -> Optional[str]:
    labels = [str(loc).lower() for loc in locations]
    for group in ("face", "object", "out_of_roi"):
        keywords = roi_groups.get(group, [])
        if not keywords:
            continue
        for label in labels:
            if any(keyword in label for keyword in keywords):
                return group
    return None


def _iter_trial_files(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    root = Path(cfg["processed_data_root"])
    filename = _ensure_pkl_filename(settings.trial_input_filename)
    pattern = root / "date=*" / "session=*" / settings.trial_input_modality / filename

    date_filter = None if dates is None else {str(v) for v in dates}
    session_filter = None if sessions is None else {str(v) for v in sessions}

    rows: list[dict] = []
    for path in root.glob(str(pattern.relative_to(root))):
        parts = path.parts
        try:
            date_part = next(part for part in parts if part.startswith("date="))
            session_part = next(part for part in parts if part.startswith("session="))
        except StopIteration:
            continue

        date = date_part.split("=", 1)[1]
        session = session_part.split("=", 1)[1]
        if date_filter is not None and date not in date_filter:
            continue
        if session_filter is not None and session not in session_filter:
            continue

        rows.append({"date": date, "session": session, "path": path})

    rows.sort(key=lambda row: (row["date"], row["session"]))
    return rows


def _extract_trials_df_and_meta(obj) -> tuple[pd.DataFrame, dict]:
    if isinstance(obj, dict) and "trials" in obj:
        trials = obj["trials"]
        meta = obj.get("meta", {})
        return (trials if isinstance(trials, pd.DataFrame) else pd.DataFrame(), meta or {})
    if isinstance(obj, pd.DataFrame):
        return obj, {}
    return pd.DataFrame(), {}


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
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(x_vec.size)
    m = int(y_vec.size)
    if n == 0 or m == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    full_len = n + m - 1
    nfft = 1 << (full_len - 1).bit_length()
    corr_circular = np.fft.irfft(
        np.fft.rfft(x_vec, nfft) * np.conj(np.fft.rfft(y_vec, nfft)),
        nfft,
    )

    if m == 1:
        corr_full = corr_circular[:n]
    else:
        corr_full = np.concatenate([corr_circular[-(m - 1):], corr_circular[:n]])
    lags = np.arange(-(m - 1), n, dtype=np.int64)

    if max_lag is not None:
        keep = np.abs(lags) <= int(max(0, int(max_lag)))
        lags = lags[keep]
        corr_full = corr_full[keep]

    return lags, corr_full


def _summarize_cross_correlation(lags: np.ndarray, corr: np.ndarray) -> dict:
    if corr.size == 0:
        return {
            "n_lags": 0,
            "zero_lag_correlation": None,
            "peak_lag": None,
            "peak_correlation": None,
        }

    zero_lag = None
    zero_idx = np.where(lags == 0)[0]
    if zero_idx.size > 0:
        zero_lag = float(corr[int(zero_idx[0])])

    peak_idx = int(np.argmax(corr))
    return {
        "n_lags": int(corr.size),
        "zero_lag_correlation": zero_lag,
        "peak_lag": int(lags[peak_idx]),
        "peak_correlation": float(corr[peak_idx]),
    }


_GLOBAL_FIXATION_META: list[dict] = []
_GLOBAL_SIGNAL_ENTRIES: list[dict] = []
_GLOBAL_SIGNAL_TRANSFORM: str = "none"
_GLOBAL_MAX_LAG: Optional[int] = None


def _init_pair_worker(
    fixation_meta: list[dict],
    signal_entries: list[dict],
    signal_transform: str,
    max_lag: Optional[int],
) -> None:
    global _GLOBAL_FIXATION_META, _GLOBAL_SIGNAL_ENTRIES
    global _GLOBAL_SIGNAL_TRANSFORM, _GLOBAL_MAX_LAG
    _GLOBAL_FIXATION_META = fixation_meta
    _GLOBAL_SIGNAL_ENTRIES = signal_entries
    _GLOBAL_SIGNAL_TRANSFORM = signal_transform
    _GLOBAL_MAX_LAG = max_lag


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
    if reference_lags.shape != lags.shape or not np.array_equal(reference_lags, lags):
        raise RuntimeError(
            "Encountered inconsistent lag vectors across fixation-neuron pairs. "
            "Use consistent PSTH windows/binning and a fixed max_lag."
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


def _build_session_output_path(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    analysis_kind: str,
    date: str,
    session: str,
) -> Path:
    if analysis_kind == WITHIN_ANALYSIS_KIND:
        subdir = settings.within_output_subdir
        filename = settings.within_output_filename
    elif analysis_kind == CROSS_ANALYSIS_KIND:
        subdir = settings.cross_output_subdir
        filename = settings.cross_output_filename
    else:
        raise ValueError(f"Unsupported analysis_kind='{analysis_kind}'.")

    output_root = build_analysis_output_dir(cfg, subdir)
    return output_root / f"date={date}" / f"session={session}" / _ensure_pkl_filename(filename)


def build_fixation_neural_cross_correlations_for_session(
    settings: FixationNeuralCrossCorrelationSettings,
    session_row: dict,
    *,
    analysis_kind: str,
    show_progress: bool = True,
) -> Optional[dict]:
    """Compute fixation-level neural cross-correlations for one session file."""
    signal_transform = _validate_signal_transform(settings.signal_transform)
    max_lag = None if settings.max_lag is None else int(max(0, int(settings.max_lag)))

    obj = _load_pickle(Path(session_row["path"]))
    trial_df, trial_meta = _extract_trials_df_and_meta(obj)
    if trial_df.empty or "psth_counts" not in trial_df.columns:
        return None

    include_region_keys = _normalize_region_keys(settings.include_regions)
    roi_groups = _normalize_roi_groups(settings.roi_groups)
    fixation_groups = _collect_fixation_groups(
        trial_df,
        default_date=str(session_row["date"]),
        default_session=str(session_row["session"]),
        include_region_keys=include_region_keys,
        roi_groups=roi_groups,
    )
    if not fixation_groups:
        return None

    anchor_region_key = _canonical_region_name(settings.anchor_region)
    partner_region_keys = _normalize_region_keys(settings.partner_regions)
    if partner_region_keys is not None and anchor_region_key is not None:
        partner_region_keys = {key for key in partner_region_keys if key != anchor_region_key}

    fixation_meta, signal_entries, pair_tasks, n_fixations_with_pairs = _build_pair_tasks(
        fixation_groups,
        analysis_kind=analysis_kind,
        anchor_region_key=anchor_region_key,
        partner_region_keys=partner_region_keys,
    )

    if settings.test_single and pair_tasks:
        pair_tasks = [random.choice(pair_tasks)]

    if not pair_tasks:
        return None

    lag_axis: Optional[np.ndarray] = None
    rows: list[dict] = []

    use_parallel = bool(settings.use_parallel and len(pair_tasks) > 1)
    if use_parallel:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        chunk_size = max(1, int(settings.pair_chunk_size))
        with Pool(
            processes=n_proc,
            initializer=_init_pair_worker,
            initargs=(fixation_meta, signal_entries, signal_transform, max_lag),
        ) as pool:
            iterator = pool.imap_unordered(
                _compute_pair_xcorr_worker,
                pair_tasks,
                chunksize=chunk_size,
            )
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(pair_tasks),
                    desc=f"{analysis_kind} xcorr {session_row['date']}-{session_row['session']} ({n_proc} workers)",
                    unit="pair",
                )
            for result in iterator:
                if result is None:
                    continue
                lags = np.asarray(result.pop("lags"), dtype=np.int64)
                if lag_axis is None:
                    lag_axis = lags
                else:
                    _assert_lag_axis_match(lag_axis, lags)
                rows.append(result)
    else:
        _init_pair_worker(fixation_meta, signal_entries, signal_transform, max_lag)
        iterator = pair_tasks
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"{analysis_kind} xcorr {session_row['date']}-{session_row['session']}",
                unit="pair",
            )
        for task in iterator:
            result = _compute_pair_xcorr_worker(task)
            if result is None:
                continue
            lags = np.asarray(result.pop("lags"), dtype=np.int64)
            if lag_axis is None:
                lag_axis = lags
            else:
                _assert_lag_axis_match(lag_axis, lags)
            rows.append(result)

    if not rows or lag_axis is None:
        return None

    result_df = _sort_result_dataframe(pd.DataFrame(rows))

    meta = {
        "analysis_kind": analysis_kind,
        "date": str(session_row["date"]),
        "session": str(session_row["session"]),
        "source_modality": settings.trial_input_modality,
        "source_filename": _ensure_pkl_filename(settings.trial_input_filename),
        "signal_transform": signal_transform,
        "max_lag": max_lag,
        "anchor_region": _as_optional_str(settings.anchor_region),
        "partner_regions": (
            None if settings.partner_regions is None else [str(v) for v in settings.partner_regions]
        ),
        "include_regions": (
            None if settings.include_regions is None else [str(v) for v in settings.include_regions]
        ),
        "n_fixations_total": int(len(fixation_groups)),
        "n_fixations_with_pairs": int(n_fixations_with_pairs),
        "n_pairs_requested": int(len(pair_tasks)),
        "n_pairs_computed": int(len(result_df)),
        "lags": lag_axis,
    }

    for key in ("bin_size_ms", "window_pre_s", "window_post_s", "bin_edges_s_rel", "bin_centers_s_rel"):
        if key in trial_meta:
            meta[key] = trial_meta[key]

    return {
        "meta": meta,
        "cross_correlations": result_df,
    }


def process_and_save_fixation_neural_cross_correlations_for_session(
    settings: FixationNeuralCrossCorrelationSettings,
    session_row: dict,
    *,
    analysis_kind: str,
    show_progress: bool = True,
) -> Optional[dict]:
    """Build and persist fixation-level neural cross-correlation output for one session."""
    data = build_fixation_neural_cross_correlations_for_session(
        settings,
        session_row,
        analysis_kind=analysis_kind,
        show_progress=show_progress,
    )
    if data is None:
        return None

    cfg = load_config(settings.cfg_path)
    out_path = _build_session_output_path(
        cfg,
        settings,
        analysis_kind=analysis_kind,
        date=str(session_row["date"]),
        session=str(session_row["session"]),
    )
    _save_pickle(data, out_path)
    return data


def _process_and_save_session_worker(
    args: tuple[FixationNeuralCrossCorrelationSettings, dict, str],
) -> int:
    settings, session_row, analysis_kind = args
    local_settings = replace(
        settings,
        use_parallel=False,
        test_single=False,
    )
    data = process_and_save_fixation_neural_cross_correlations_for_session(
        local_settings,
        session_row,
        analysis_kind=analysis_kind,
        show_progress=False,
    )
    return 1 if data is not None else 0


def _run_fixation_neural_cross_correlation_analysis(
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    analysis_kind: str,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> dict:
    if use_parallel is not None:
        settings.use_parallel = bool(use_parallel)
    if test_single is not None:
        settings.test_single = bool(test_single)

    cfg = load_config(settings.cfg_path)
    session_rows = _iter_trial_files(cfg, settings, dates=dates, sessions=sessions)
    if not session_rows:
        print("No fixation PSTH trial files found for neural cross-correlation analysis.")
        return {"n_sessions_total": 0, "n_sessions_written": 0}

    if settings.test_single and session_rows:
        session_rows = [random.choice(session_rows)]

    n_written = 0
    run_session_pool = bool(
        settings.use_parallel
        and settings.parallelize_across_sessions
        and len(session_rows) > 1
    )
    if run_session_pool:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        worker_tasks = [(settings, row, analysis_kind) for row in session_rows]
        with Pool(processes=n_proc) as pool:
            iterator = pool.imap_unordered(_process_and_save_session_worker, worker_tasks, chunksize=1)
            for wrote in tqdm(
                iterator,
                total=len(worker_tasks),
                desc=f"{analysis_kind} sessions ({n_proc} workers)",
                unit="session",
            ):
                n_written += int(wrote)
    else:
        local_settings = replace(settings, use_parallel=False)
        for session_row in tqdm(
            session_rows,
            desc=f"{analysis_kind} sessions",
            unit="session",
        ):
            data = process_and_save_fixation_neural_cross_correlations_for_session(
                local_settings,
                session_row,
                analysis_kind=analysis_kind,
                show_progress=True,
            )
            if data is not None:
                n_written += 1

    return {
        "n_sessions_total": int(len(session_rows)),
        "n_sessions_written": int(n_written),
    }


def run_within_region_fixation_neural_cross_correlation(
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> dict:
    """Run within-region fixation-level neural PSTH cross-correlation analysis."""
    return _run_fixation_neural_cross_correlation_analysis(
        settings,
        analysis_kind=WITHIN_ANALYSIS_KIND,
        dates=dates,
        sessions=sessions,
        use_parallel=use_parallel,
        test_single=test_single,
    )


def run_cross_region_fixation_neural_cross_correlation(
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> dict:
    """Run cross-region fixation-level neural PSTH cross-correlation analysis."""
    return _run_fixation_neural_cross_correlation_analysis(
        settings,
        analysis_kind=CROSS_ANALYSIS_KIND,
        dates=dates,
        sessions=sessions,
        use_parallel=use_parallel,
        test_single=test_single,
    )
