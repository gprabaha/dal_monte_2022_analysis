"""Build fixation-triggered PSTH features from unit-level ephys data."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.data.load import load_ephys_units
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    build_processed_data_path,
    build_processed_out_dir,
    scan_processed_data_paths,
)


DEFAULT_FIXATION_ROI_GROUPS: Dict[str, Sequence[str]] = {
    "face": ("face", "mouth", "eyes_nf"),
    "object": ("right_nonsocial_object", "left_nonsocial_object"),
    "out_of_roi": ("out_of_roi",),
}


@dataclass
class FixationPSTHSettings:
    """Configuration for session-level trial PSTH extraction."""

    cfg_path: str
    ephys_cfg_path: str = "configs/ephys_data.yaml"
    fixations_modality: str = "fixations"
    timeline_modality: str = "neural_timeline"
    interactive_modality: str = "interactive_periods"
    output_modality: str = "psth"
    trial_output_filename: str = "fixations.pkl"
    roi_groups: Dict[str, Sequence[str]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_FIXATION_ROI_GROUPS.items()}
    )
    agent_roi_groups: Optional[Dict[str, Dict[str, Sequence[str]]]] = None
    categories: Optional[Sequence[str]] = ("face", "object", "out_of_roi")
    include_interactive_state: bool = True
    interactive_high_label: str = "interactive"
    bin_size_ms: float = 10.0
    window_pre_s: float = 1.0
    window_post_s: float = 1.0
    use_parallel: bool = True
    max_procs: int = 32
    test_single: bool = False
    agents: Optional[Sequence[str]] = None


@dataclass
class FixationPSTHAverageSettings:
    """Configuration for date-level averaged PSTH outputs."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations.pkl"
    output_subdir: str = "ephys/psth/fixation_psth_averages"
    output_filename: str = "fixations.pkl"
    split_by_interactive_state: bool = False
    restrict_interactive_state: Optional[str] = None
    group_by_session: bool = False
    smooth_before_average: bool = True
    smoothing_sigma_ms: float = 30.0
    use_parallel: bool = True
    max_procs: int = 16
    test_single: bool = False
    categories: Optional[Sequence[str]] = ("face", "object", "out_of_roi")


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _normalize_roi_groups(groups: Dict[str, Sequence[str]]) -> Dict[str, list[str]]:
    normalized: Dict[str, list[str]] = {}
    for group_name, labels in (groups or {}).items():
        if labels is None:
            continue
        if isinstance(labels, (str, bytes)):
            label_list = [labels]
        else:
            label_list = list(labels)
        normalized[str(group_name)] = [str(label).lower() for label in label_list]
    return normalized


def _resolve_roi_groups(settings: FixationPSTHSettings, agent: str) -> Dict[str, list[str]]:
    if settings.agent_roi_groups and agent in settings.agent_roi_groups:
        return _normalize_roi_groups(settings.agent_roi_groups[agent])
    return _normalize_roi_groups(settings.roi_groups)


def _coerce_location(loc) -> list[str]:
    if loc is None:
        return []
    if isinstance(loc, (list, tuple, set, np.ndarray)):
        return [str(val) for val in loc if val is not None]
    try:
        if pd.isna(loc):
            return []
    except Exception:
        pass
    return [str(loc)]


def _locations_match(locations: Iterable[str], keywords: Sequence[str]) -> bool:
    for loc in locations:
        loc_lower = str(loc).lower()
        for keyword in keywords:
            if keyword in loc_lower:
                return True
    return False


def _ensure_pkl_filename(filename: str) -> str:
    name = str(filename).strip()
    if not name:
        raise ValueError("Output filename cannot be empty.")
    return name if name.endswith(".pkl") else f"{name}.pkl"


def _build_trial_output_path(cfg: dict, row: dict, settings: FixationPSTHSettings) -> Path:
    out_dir = build_processed_out_dir(cfg, row, settings.output_modality)
    return out_dir / _ensure_pkl_filename(settings.trial_output_filename)


def _build_average_output_path(cfg: dict, date: str, settings: FixationPSTHAverageSettings) -> Path:
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    return out_root / f"date={date}" / _ensure_pkl_filename(settings.output_filename)


def _build_bin_edges(settings: FixationPSTHSettings) -> np.ndarray:
    bin_size_s = float(settings.bin_size_ms) / 1000.0
    if bin_size_s <= 0:
        raise ValueError("bin_size_ms must be > 0.")
    if settings.window_pre_s <= 0 or settings.window_post_s <= 0:
        raise ValueError("window_pre_s and window_post_s must be > 0.")
    pre = float(settings.window_pre_s)
    post = float(settings.window_post_s)
    return np.arange(-pre, post + bin_size_s * 0.5, bin_size_s, dtype=float)


def _iter_interactive_periods(df: Optional[pd.DataFrame]) -> list[tuple[int, int, str]]:
    if df is None or df.empty:
        return []
    required = {"start", "stop", "state"}
    if not required.issubset(df.columns):
        return []
    periods: list[tuple[int, int, str]] = []
    for row in df.itertuples(index=False):
        try:
            start = int(getattr(row, "start"))
            stop = int(getattr(row, "stop"))
        except Exception:
            continue
        if stop < start:
            continue
        state = str(getattr(row, "state"))
        periods.append((start, stop, state))
    periods.sort(key=lambda tup: tup[0])
    return periods


def _lookup_interactive_state(periods: list[tuple[int, int, str]], idx: int) -> Optional[str]:
    for start, stop, state in periods:
        if start <= idx <= stop:
            return state
        if start > idx:
            break
    return None


def _categorize_fixation(
    locations: list[str],
    roi_groups: Dict[str, list[str]],
    allowed_categories: Optional[set[str]],
) -> Optional[str]:
    labels = [loc.lower() for loc in locations]
    for group in ("face", "object", "out_of_roi"):
        keywords = roi_groups.get(group)
        if not keywords:
            continue
        if _locations_match(labels, keywords):
            if allowed_categories is not None and group not in allowed_categories:
                return None
            return group
    return None


def _build_fixation_tasks(
    settings: FixationPSTHSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    cfg = load_dataset_config(settings.cfg_path)
    rows = scan_processed_data_paths(
        cfg,
        settings.fixations_modality,
        dates=dates,
        sessions=sessions,
        agents=settings.agents,
    )
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if row.get("agent") is None:
            continue
        key = (row["date"], row["session"])
        grouped.setdefault(
            key,
            {"date": row["date"], "session": row["session"], "agent_paths": {}},
        )
        grouped[key]["agent_paths"][row["agent"]] = row["path"]

    tasks = list(grouped.values())
    tasks.sort(key=lambda row: (row["date"], row["session"]))
    if settings.test_single and tasks:
        return [random.choice(tasks)]
    return tasks


def _build_session_events(
    settings: FixationPSTHSettings,
    row: dict,
) -> tuple[list[dict], np.ndarray]:
    cfg = load_dataset_config(settings.cfg_path)
    timeline_path = build_processed_data_path(cfg, row, settings.timeline_modality, None)
    if not timeline_path.exists():
        return [], np.array([], dtype=float)

    timeline_obj = _load_pickle(timeline_path)
    timeline_t = np.asarray(getattr(timeline_obj, "t", []), dtype=float)
    if timeline_t.size == 0:
        return [], timeline_t

    interactive_periods = []
    if settings.include_interactive_state:
        interactive_path = build_processed_data_path(cfg, row, settings.interactive_modality, None)
        if interactive_path.exists():
            interactive_df = _load_pickle(interactive_path)
            if isinstance(interactive_df, pd.DataFrame):
                interactive_periods = _iter_interactive_periods(interactive_df)

    allowed_categories = None
    if settings.categories is not None:
        allowed_categories = {str(name) for name in settings.categories}

    events: list[dict] = []
    for agent, fix_path in sorted(row["agent_paths"].items()):
        fix_df = _load_pickle(fix_path)
        if not isinstance(fix_df, pd.DataFrame) or fix_df.empty:
            continue

        roi_groups = _resolve_roi_groups(settings, agent)
        for fix_row in fix_df.itertuples(index=False):
            try:
                start_idx = int(getattr(fix_row, "start"))
                stop_idx = int(getattr(fix_row, "stop"))
            except Exception:
                continue

            if start_idx < 0 or start_idx >= timeline_t.shape[0]:
                continue
            if stop_idx < start_idx:
                continue

            start_time_s = float(timeline_t[start_idx])
            if not np.isfinite(start_time_s):
                continue

            location_raw = _coerce_location(getattr(fix_row, "location", None))
            category = _categorize_fixation(location_raw, roi_groups, allowed_categories)
            if category is None:
                continue

            interactive_state = _lookup_interactive_state(interactive_periods, start_idx)
            events.append(
                {
                    "date": row["date"],
                    "session": row["session"],
                    "fixation_agent": str(agent),
                    "fixation_monkey_name": _as_optional_str(getattr(fix_row, "monkey_name", None)),
                    "fixation_category": category,
                    "fixation_location": tuple(location_raw),
                    "fixation_start_idx": start_idx,
                    "fixation_stop_idx": stop_idx,
                    "fixation_start_time_s": start_time_s,
                    "interactive_state": interactive_state,
                    "is_interactive": bool(
                        interactive_state is not None
                        and str(interactive_state).lower() == settings.interactive_high_label.lower()
                    ),
                }
            )
    return events, timeline_t


_GLOBAL_TRIAL_EVENTS: list[dict] = []
_GLOBAL_BIN_EDGES: np.ndarray = np.array([], dtype=float)


def _init_trial_worker(events: list[dict], bin_edges: np.ndarray) -> None:
    global _GLOBAL_TRIAL_EVENTS, _GLOBAL_BIN_EDGES
    _GLOBAL_TRIAL_EVENTS = events
    _GLOBAL_BIN_EDGES = bin_edges


def _compute_unit_trial_rows(unit_payload: dict) -> list[dict]:
    rows: list[dict] = []
    spike_ts = np.asarray(unit_payload["spike_ts"], dtype=float)

    for event in _GLOBAL_TRIAL_EVENTS:
        rel = spike_ts - float(event["fixation_start_time_s"])
        counts, _ = np.histogram(rel, bins=_GLOBAL_BIN_EDGES)
        rows.append(
            {
                **event,
                "unit_uuid": unit_payload["unit_uuid"],
                "region": unit_payload["region"],
                "spike_channel": unit_payload["spike_channel"],
                "session_name": unit_payload["session_name"],
                "recorded_agent": unit_payload["recorded_agent"],
                "recorded_monkey": unit_payload["recorded_monkey"],
                "area": unit_payload["area"],
                "psth_counts": counts.astype(np.int32),
            }
        )
    return rows


def _units_to_payloads(units) -> list[dict]:
    payloads: list[dict] = []
    for unit in units:
        ctx = unit.context
        payloads.append(
            {
                "unit_uuid": str(ctx.unit_uuid),
                "region": _as_optional_str(ctx.region),
                "spike_channel": _as_optional_str(ctx.spike_channel),
                "session_name": str(ctx.session_name),
                "recorded_agent": _as_optional_str(ctx.recorded_agent),
                "recorded_monkey": _as_optional_str(ctx.recorded_monkey),
                "area": _as_optional_str(ctx.area),
                "spike_ts": np.asarray(unit.spike_ts, dtype=float),
            }
        )
    return payloads


def build_fixation_psth_trials_for_session(
    settings: FixationPSTHSettings,
    row: dict,
    units_for_date: Sequence[object],
) -> Optional[dict]:
    """Build session-level fixation PSTH trials for all units on a date."""
    events, _ = _build_session_events(settings, row)
    if not events:
        return None
    if not units_for_date:
        return None

    bin_edges = _build_bin_edges(settings)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    payloads = _units_to_payloads(units_for_date)
    if not payloads:
        return None

    all_rows: list[dict] = []
    if settings.use_parallel and len(payloads) > 1:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        with Pool(
            processes=n_proc,
            initializer=_init_trial_worker,
            initargs=(events, bin_edges),
        ) as pool:
            for unit_rows in pool.imap_unordered(_compute_unit_trial_rows, payloads):
                if unit_rows:
                    all_rows.extend(unit_rows)
    else:
        _init_trial_worker(events, bin_edges)
        for payload in payloads:
            unit_rows = _compute_unit_trial_rows(payload)
            if unit_rows:
                all_rows.extend(unit_rows)

    if not all_rows:
        return None

    trial_df = pd.DataFrame(all_rows)
    return {
        "meta": {
            "date": row["date"],
            "session": row["session"],
            "event_source": "fixations",
            "event_anchor": "fixation_start",
            "bin_size_ms": float(settings.bin_size_ms),
            "window_pre_s": float(settings.window_pre_s),
            "window_post_s": float(settings.window_post_s),
            "bin_edges_s_rel": bin_edges,
            "bin_centers_s_rel": bin_centers,
            "output_modality": settings.output_modality,
            "trial_output_filename": _ensure_pkl_filename(settings.trial_output_filename),
        },
        "trials": trial_df,
    }


def process_fixation_psth_trials_for_session(
    settings: FixationPSTHSettings,
    row: dict,
    units_for_date: Sequence[object],
) -> Optional[dict]:
    """Build and persist session-level fixation PSTH trial data."""
    data = build_fixation_psth_trials_for_session(settings, row, units_for_date)
    if data is None:
        return None
    cfg = load_dataset_config(settings.cfg_path)
    out_path = _build_trial_output_path(cfg, row, settings)
    _save_pickle(data, out_path)
    return data


def run_fixation_psth_trial_build(
    settings: FixationPSTHSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> None:
    """Run session-level fixation PSTH extraction."""
    if use_parallel is not None:
        settings.use_parallel = bool(use_parallel)
    if test_single is not None:
        settings.test_single = bool(test_single)

    tasks = _build_fixation_tasks(settings, dates=dates, sessions=sessions)
    if not tasks:
        print("No fixation PSTH tasks found.")
        return

    all_units = load_ephys_units(cfg_path=settings.cfg_path, ephys_cfg_path=settings.ephys_cfg_path)
    units_by_date: dict[str, list[object]] = {}
    for unit in all_units:
        units_by_date.setdefault(unit.context.date, []).append(unit)

    for row in tqdm(tasks, desc="Building fixation PSTH trials", unit="session"):
        date = str(row["date"])
        units = units_by_date.get(date, [])
        process_fixation_psth_trials_for_session(settings, row, units)


def _iter_trial_files(
    cfg: dict,
    settings: FixationPSTHAverageSettings,
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


def _extract_trials_df(obj) -> tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    if isinstance(obj, dict) and "trials" in obj:
        trial_df = obj["trials"]
        meta = obj.get("meta", {})
        bin_edges = meta.get("bin_edges_s_rel")
        bin_centers = meta.get("bin_centers_s_rel")
        return trial_df, bin_edges, bin_centers
    if isinstance(obj, pd.DataFrame):
        return obj, None, None
    raise ValueError(f"Unsupported trial PSTH object type: {type(obj)}")


def _category_filter(settings: FixationPSTHAverageSettings) -> Optional[set[str]]:
    if settings.categories is None:
        return None
    return {str(val) for val in settings.categories}


def _resolve_smoothing_sigma_bins(
    settings: FixationPSTHAverageSettings,
    bin_edges_ref: Optional[np.ndarray],
) -> Optional[float]:
    if not settings.smooth_before_average:
        return None
    if float(settings.smoothing_sigma_ms) <= 0:
        raise ValueError("smoothing_sigma_ms must be > 0 when smooth_before_average is enabled.")
    if bin_edges_ref is not None and np.asarray(bin_edges_ref).size > 1:
        bin_size_ms = float(np.mean(np.diff(np.asarray(bin_edges_ref, dtype=float)))) * 1000.0
    else:
        bin_size_ms = 10.0
    if bin_size_ms <= 0:
        raise ValueError("Encountered non-positive PSTH bin size while averaging.")
    return float(settings.smoothing_sigma_ms) / bin_size_ms


def _aggregate_group_counts(
    key: tuple,
    counts_list: Sequence[np.ndarray],
    *,
    smooth_before_average: bool,
    sigma_bins: Optional[float],
):
    acc = None
    n_trials = 0
    for raw in counts_list:
        counts = np.asarray(raw, dtype=float).reshape(-1)
        if counts.size == 0:
            continue
        if smooth_before_average:
            counts = gaussian_filter1d(counts, sigma=sigma_bins, mode="nearest")
        if acc is None:
            acc = np.zeros_like(counts, dtype=float)
        elif acc.shape != counts.shape:
            raise ValueError("Encountered inconsistent PSTH bin counts while aggregating averages.")
        acc += counts
        n_trials += 1
    return key, acc, n_trials


def build_fixation_psth_averages_for_date(
    settings: FixationPSTHAverageSettings,
    date: str,
    trial_paths: Sequence[Path],
) -> Optional[dict]:
    """Build one date-level averaged PSTH object from trial files."""
    category_allow = _category_filter(settings)

    grouped_counts: dict[tuple, list[np.ndarray]] = {}
    bin_edges_ref = None
    bin_centers_ref = None

    for path in trial_paths:
        obj = _load_pickle(path)
        trial_df, bin_edges, bin_centers = _extract_trials_df(obj)
        if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
            continue
        if "psth_counts" not in trial_df.columns:
            continue

        if bin_edges is not None:
            arr = np.asarray(bin_edges, dtype=float)
            if bin_edges_ref is None:
                bin_edges_ref = arr
            elif arr.shape != bin_edges_ref.shape or not np.allclose(arr, bin_edges_ref):
                raise ValueError(
                    f"Found mismatched bin edges across trial files for date {date}; path={path}"
                )
        if bin_centers is not None:
            arr = np.asarray(bin_centers, dtype=float)
            if bin_centers_ref is None:
                bin_centers_ref = arr
            elif arr.shape != bin_centers_ref.shape or not np.allclose(arr, bin_centers_ref):
                raise ValueError(
                    f"Found mismatched bin centers across trial files for date {date}; path={path}"
                )

        for row in trial_df.itertuples(index=False):
            fixation_category = str(getattr(row, "fixation_category", ""))
            if category_allow is not None and fixation_category not in category_allow:
                continue

            interactive_state = _as_optional_str(getattr(row, "interactive_state", None))
            if settings.restrict_interactive_state is not None:
                if interactive_state != settings.restrict_interactive_state:
                    continue

            counts = np.asarray(getattr(row, "psth_counts"), dtype=float)
            if counts.size == 0:
                continue

            key_values = [
                date,
                str(getattr(row, "unit_uuid")),
                _as_optional_str(getattr(row, "region", None)),
                _as_optional_str(getattr(row, "spike_channel", None)),
                _as_optional_str(getattr(row, "recorded_agent", None)),
                _as_optional_str(getattr(row, "recorded_monkey", None)),
                _as_optional_str(getattr(row, "area", None)),
                fixation_category,
            ]
            if settings.group_by_session:
                key_values.append(str(getattr(row, "session")))
            if settings.split_by_interactive_state:
                key_values.append(interactive_state)

            key = tuple(key_values)
            grouped_counts.setdefault(key, []).append(counts)

    if not grouped_counts:
        return None

    sigma_bins = _resolve_smoothing_sigma_bins(settings, bin_edges_ref)
    grouped: dict[tuple, dict] = {}
    for key, counts_list in grouped_counts.items():
        _, acc, n_trials = _aggregate_group_counts(
            key,
            counts_list,
            smooth_before_average=settings.smooth_before_average,
            sigma_bins=sigma_bins,
        )
        if n_trials <= 0:
            continue
        grouped[key] = {"sum": acc, "n": int(n_trials)}

    if not grouped:
        return None

    records: list[dict] = []
    for key, val in grouped.items():
        idx = 0
        record = {
            "date": key[idx],
            "unit_uuid": key[idx + 1],
            "region": key[idx + 2],
            "spike_channel": key[idx + 3],
            "recorded_agent": key[idx + 4],
            "recorded_monkey": key[idx + 5],
            "area": key[idx + 6],
            "fixation_category": key[idx + 7],
        }
        idx = 8
        if settings.group_by_session:
            record["session"] = key[idx]
            idx += 1
        if settings.split_by_interactive_state:
            record["interactive_state"] = key[idx]
            idx += 1

        n_trials = int(val["n"])
        record["n_trials"] = n_trials
        record["psth_mean"] = val["sum"] / max(1, n_trials)
        records.append(record)

    averages_df = pd.DataFrame(records)
    return {
        "meta": {
            "date": date,
            "source_modality": settings.trial_input_modality,
            "source_filename": _ensure_pkl_filename(settings.trial_input_filename),
            "smooth_before_average": bool(settings.smooth_before_average),
            "smoothing_sigma_ms": float(settings.smoothing_sigma_ms),
            "smoothing_sigma_bins": sigma_bins,
            "split_by_interactive_state": bool(settings.split_by_interactive_state),
            "restrict_interactive_state": settings.restrict_interactive_state,
            "group_by_session": bool(settings.group_by_session),
            "bin_edges_s_rel": bin_edges_ref,
            "bin_centers_s_rel": bin_centers_ref,
        },
        "averages": averages_df,
    }


def process_fixation_psth_averages_for_date(
    settings: FixationPSTHAverageSettings,
    date: str,
    trial_paths: Sequence[Path],
) -> Optional[dict]:
    """Build and persist one date-level PSTH average object."""
    data = build_fixation_psth_averages_for_date(settings, date, trial_paths)
    if data is None:
        return None
    cfg = load_dataset_config(settings.cfg_path)
    out_path = _build_average_output_path(cfg, date, settings)
    _save_pickle(data, out_path)
    return data


def _average_for_date_worker(args) -> int:
    settings, date, paths = args
    data = process_fixation_psth_averages_for_date(settings, date, paths)
    return 1 if data is not None else 0


def run_fixation_psth_average_build(
    settings: FixationPSTHAverageSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> None:
    """Run date-level PSTH averaging from session trial files."""
    cfg = load_dataset_config(settings.cfg_path)
    trial_rows = _iter_trial_files(cfg, settings, dates=dates, sessions=sessions)
    if not trial_rows:
        print("No fixation PSTH trial files found.")
        return

    grouped: dict[str, list[Path]] = {}
    for row in trial_rows:
        grouped.setdefault(row["date"], []).append(row["path"])

    tasks = sorted(grouped.items(), key=lambda item: item[0])
    if settings.test_single and tasks:
        tasks = [random.choice(tasks)]

    if settings.use_parallel and len(tasks) > 1:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        worker_tasks = [(settings, date, paths) for date, paths in tasks]
        with Pool(processes=n_proc) as pool:
            for _ in tqdm(
                pool.imap_unordered(_average_for_date_worker, worker_tasks),
                total=len(worker_tasks),
                desc=f"Building fixation PSTH averages ({n_proc} workers)",
                unit="date",
            ):
                pass
        return

    for date, paths in tqdm(tasks, desc="Building fixation PSTH averages", unit="date"):
        process_fixation_psth_averages_for_date(settings, date, paths)
