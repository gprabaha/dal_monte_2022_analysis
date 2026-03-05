"""Build fixation-triggered PSTH features from unit-level ephys data."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.data.loaders.ephys import load_ephys_units
from dal_monte_2022_analysis.ephys.features.common import (
    as_optional_str as _as_optional_str,
    build_symmetric_bin_edges,
    ensure_pkl_filename as _ensure_pkl_filename,
    units_to_payloads as _units_to_payloads,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    build_processed_pickle_path,
    load_pickle_path,
    save_pickle_path,
    scan_processed_paths,
    scan_processed_paths_for_filename,
)
from dal_monte_2022_analysis.runtime.execution.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    build_processed_out_dir,
)
from dal_monte_2022_analysis.core.behav.roi_groups import (
    DEFAULT_FIXATION_ROI_GROUPS as DEFAULT_SHARED_FIXATION_ROI_GROUPS,
    categorize_locations,
    coerce_location_labels,
    resolve_agent_roi_groups,
)


DEFAULT_FIXATION_ROI_GROUPS: Dict[str, Sequence[str]] = DEFAULT_SHARED_FIXATION_ROI_GROUPS


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
    store_split_and_unsplit_together: bool = True
    restrict_interactive_state: Optional[str] = None
    group_by_session: bool = False
    smooth_before_average: bool = True
    smoothing_sigma_ms: float = 30.0
    convert_to_firing_rate_before_average: bool = True
    target_bin_size_ms: Optional[float] = None
    target_bin_step_ms: Optional[float] = None
    use_parallel: bool = True
    max_procs: int = 16
    test_single: bool = False
    categories: Optional[Sequence[str]] = ("face", "object", "out_of_roi")


def _resolve_roi_groups(settings: FixationPSTHSettings, agent: str) -> Dict[str, list[str]]:
    return resolve_agent_roi_groups(
        agent=agent,
        roi_groups=settings.roi_groups,
        agent_roi_groups=settings.agent_roi_groups,
        include_defaults=False,
    )


def _build_trial_output_path(cfg: dict, row: dict, settings: FixationPSTHSettings) -> Path:
    out_dir = build_processed_out_dir(cfg, row, settings.output_modality)
    return out_dir / _ensure_pkl_filename(settings.trial_output_filename)


def _build_average_output_path(cfg: dict, date: str, settings: FixationPSTHAverageSettings) -> Path:
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    return out_root / f"date={date}" / _ensure_pkl_filename(settings.output_filename)


def _build_bin_edges(settings: FixationPSTHSettings) -> np.ndarray:
    return build_symmetric_bin_edges(
        bin_size_ms=settings.bin_size_ms,
        window_pre_s=settings.window_pre_s,
        window_post_s=settings.window_post_s,
    )


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
    return categorize_locations(
        locations,
        roi_groups,
        ordered_groups=("face", "object", "out_of_roi"),
        allowed_categories=allowed_categories,
    )


def _build_fixation_tasks(
    settings: FixationPSTHSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    cfg = load_config(settings.cfg_path)
    rows = scan_processed_paths(
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
    cfg = load_config(settings.cfg_path)
    timeline_path = build_processed_pickle_path(cfg, row, settings.timeline_modality, None)
    if not timeline_path.exists():
        return [], np.array([], dtype=float)

    timeline_obj = load_pickle_path(timeline_path)
    timeline_t = np.asarray(getattr(timeline_obj, "t", []), dtype=float)
    if timeline_t.size == 0:
        return [], timeline_t

    interactive_periods = []
    if settings.include_interactive_state:
        interactive_path = build_processed_pickle_path(cfg, row, settings.interactive_modality, None)
        if interactive_path.exists():
            interactive_df = load_pickle_path(interactive_path)
            if isinstance(interactive_df, pd.DataFrame):
                interactive_periods = _iter_interactive_periods(interactive_df)

    allowed_categories = None
    if settings.categories is not None:
        allowed_categories = {str(name) for name in settings.categories}

    events: list[dict] = []
    for agent, fix_path in sorted(row["agent_paths"].items()):
        fix_df = load_pickle_path(fix_path)
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
            if stop_idx < 0 or stop_idx >= timeline_t.shape[0]:
                continue
            if stop_idx < start_idx:
                continue

            start_time_s = float(timeline_t[start_idx])
            stop_time_s = float(timeline_t[stop_idx])
            if not np.isfinite(start_time_s):
                continue
            if not np.isfinite(stop_time_s):
                continue

            location_raw = coerce_location_labels(getattr(fix_row, "location", None))
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
                    "fixation_stop_time_s": stop_time_s,
                    "fixation_duration_s": max(0.0, stop_time_s - start_time_s),
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


_GLOBAL_FIXATION_EVENTS_BY_DATE: dict[str, list[dict]] = {}
_GLOBAL_FIXATION_BIN_EDGES: np.ndarray = np.array([], dtype=float)


def _init_fixation_unit_worker(events_by_date: dict[str, list[dict]], bin_edges: np.ndarray) -> None:
    global _GLOBAL_FIXATION_EVENTS_BY_DATE, _GLOBAL_FIXATION_BIN_EDGES
    _GLOBAL_FIXATION_EVENTS_BY_DATE = events_by_date
    _GLOBAL_FIXATION_BIN_EDGES = bin_edges


def _compute_unit_rows_across_fixation_sessions(
    unit_payload: dict,
) -> tuple[str, dict[tuple[str, str], list[dict]]]:
    unit_date = str(unit_payload.get("unit_date", ""))
    session_payloads = _GLOBAL_FIXATION_EVENTS_BY_DATE.get(unit_date, [])
    rows_by_session: dict[tuple[str, str], list[dict]] = {}
    spike_ts = np.asarray(unit_payload["spike_ts"], dtype=float)

    for session_payload in session_payloads:
        session_date = str(session_payload["date"])
        session_name = str(session_payload["session"])
        events = session_payload["events"]
        session_rows: list[dict] = []
        for event in events:
            rel = spike_ts - float(event["fixation_start_time_s"])
            counts, _ = np.histogram(rel, bins=_GLOBAL_FIXATION_BIN_EDGES)
            session_rows.append(
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
        if session_rows:
            rows_by_session[(session_date, session_name)] = session_rows

    return unit_date, rows_by_session


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
    cfg = load_config(settings.cfg_path)
    out_path = _build_trial_output_path(cfg, row, settings)
    save_pickle_path(data, out_path)
    return data


def _build_fixation_trial_payload(
    settings: FixationPSTHSettings,
    *,
    date: str,
    session: str,
    rows: list[dict],
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
) -> dict:
    trial_df = pd.DataFrame(rows)
    return {
        "meta": {
            "date": date,
            "session": session,
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


def _flush_fixation_session_rows(
    cfg: dict,
    settings: FixationPSTHSettings,
    rows_for_session: dict[tuple[str, str], list[dict]],
    *,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
) -> None:
    for (date, session), rows in rows_for_session.items():
        if not rows:
            continue
        row = {"date": str(date), "session": str(session)}
        data = _build_fixation_trial_payload(
            settings,
            date=str(date),
            session=str(session),
            rows=rows,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
        )
        out_path = _build_trial_output_path(cfg, row, settings)
        save_pickle_path(data, out_path)


def run_fixation_psth_trial_build(
    settings: FixationPSTHSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> None:
    """Run fixation PSTH extraction with global unit-level parallelization."""
    if use_parallel is not None:
        settings.use_parallel = bool(use_parallel)
    if test_single is not None:
        settings.test_single = bool(test_single)

    session_rows = _build_fixation_tasks(settings, dates=dates, sessions=sessions)
    if not session_rows:
        print("No fixation PSTH tasks found.")
        return

    rows_by_date: dict[str, list[dict]] = {}
    for row in session_rows:
        rows_by_date.setdefault(str(row["date"]), []).append(row)
    for date in rows_by_date:
        rows_by_date[date].sort(key=lambda row: str(row["session"]))

    session_events_by_date: dict[str, list[dict]] = {}
    for date, date_rows in rows_by_date.items():
        payloads_for_date: list[dict] = []
        for row in date_rows:
            events, _ = _build_session_events(settings, row)
            if not events:
                continue
            payloads_for_date.append(
                {
                    "date": str(row["date"]),
                    "session": str(row["session"]),
                    "events": events,
                }
            )
        if payloads_for_date:
            session_events_by_date[date] = payloads_for_date

    if not session_events_by_date:
        print("No fixation session events found.")
        return

    cfg = load_config(settings.cfg_path)
    all_units = load_ephys_units(
        cfg_path=settings.cfg_path,
        ephys_cfg_path=settings.ephys_cfg_path,
        dates=sorted(session_events_by_date.keys()),
    )
    payloads = _units_to_payloads(all_units)
    payloads = [payload for payload in payloads if payload["unit_date"] in session_events_by_date]
    if not payloads:
        print("No matching ephys units found for fixation PSTH tasks.")
        return

    bin_edges = _build_bin_edges(settings)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    buffered_rows_by_date: dict[str, dict[tuple[str, str], list[dict]]] = {}
    for date, session_payloads in session_events_by_date.items():
        buffered_rows_by_date[date] = {
            (date, str(payload["session"])): [] for payload in session_payloads
        }

    remaining_units_by_date: dict[str, int] = {}
    for payload in payloads:
        date = str(payload["unit_date"])
        remaining_units_by_date[date] = remaining_units_by_date.get(date, 0) + 1

    def _accumulate_and_flush(unit_date: str, rows_by_session: dict[tuple[str, str], list[dict]]) -> None:
        for key, rows in rows_by_session.items():
            date_key = str(key[0])
            date_bucket = buffered_rows_by_date.setdefault(date_key, {})
            date_bucket.setdefault((str(key[0]), str(key[1])), []).extend(rows)

        remaining = remaining_units_by_date.get(unit_date, 0) - 1
        remaining_units_by_date[unit_date] = remaining
        if remaining <= 0:
            rows_for_date = buffered_rows_by_date.pop(unit_date, {})
            _flush_fixation_session_rows(
                cfg,
                settings,
                rows_for_date,
                bin_edges=bin_edges,
                bin_centers=bin_centers,
            )

    if settings.use_parallel and len(payloads) > 1:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        with Pool(
            processes=n_proc,
            initializer=_init_fixation_unit_worker,
            initargs=(session_events_by_date, bin_edges),
        ) as pool:
            iterator = pool.imap_unordered(_compute_unit_rows_across_fixation_sessions, payloads)
            for unit_date, rows_by_session in tqdm(
                iterator,
                total=len(payloads),
                desc="Building fixation PSTH trials",
                unit="unit",
            ):
                _accumulate_and_flush(str(unit_date), rows_by_session)
    else:
        _init_fixation_unit_worker(session_events_by_date, bin_edges)
        for payload in tqdm(payloads, desc="Building fixation PSTH trials", unit="unit"):
            unit_date, rows_by_session = _compute_unit_rows_across_fixation_sessions(payload)
            _accumulate_and_flush(str(unit_date), rows_by_session)

    for _, rows_for_date in buffered_rows_by_date.items():
        _flush_fixation_session_rows(
            cfg,
            settings,
            rows_for_date,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
        )


def _iter_trial_files(
    cfg: dict,
    settings: FixationPSTHAverageSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    return scan_processed_paths_for_filename(
        cfg,
        settings.trial_input_modality,
        filename=_ensure_pkl_filename(settings.trial_input_filename),
        dates=dates,
        sessions=sessions,
        agents=[None],
    )


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


def _resolve_average_output_centers(
    *,
    bin_edges_ref: Optional[np.ndarray],
    bin_centers_ref: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if bin_centers_ref is not None:
        arr = np.asarray(bin_centers_ref, dtype=float).reshape(-1)
        if arr.size > 0:
            return arr
    if bin_edges_ref is not None:
        edges = np.asarray(bin_edges_ref, dtype=float).reshape(-1)
        if edges.size > 1:
            return 0.5 * (edges[:-1] + edges[1:])
    return None


def _build_resample_counts_matrix(
    source_centers: np.ndarray,
    *,
    target_bin_size_ms: Optional[float],
    target_bin_step_ms: Optional[float],
) -> tuple[Optional[np.ndarray], np.ndarray, Optional[float], Optional[float]]:
    centers = np.asarray(source_centers, dtype=float).reshape(-1)
    if centers.size == 0:
        raise ValueError("Cannot resample PSTH averages with empty source bin centers.")
    if centers.size == 1:
        return None, centers, None, None

    diffs = np.diff(centers)
    if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
        raise ValueError("Source PSTH bin centers must be strictly increasing for resampling.")
    source_bin_size_s = float(np.mean(diffs))
    if not np.allclose(diffs, source_bin_size_s, atol=max(1e-9, abs(source_bin_size_s) * 1e-6)):
        raise ValueError("Source PSTH bin centers must be approximately uniform for resampling.")
    if not np.isfinite(source_bin_size_s) or source_bin_size_s <= 0:
        raise ValueError("Encountered non-positive source PSTH bin size while resampling averages.")

    if target_bin_size_ms is None:
        return None, centers, None, None

    target_bin_size_s = float(target_bin_size_ms) / 1000.0
    if not np.isfinite(target_bin_size_s) or target_bin_size_s <= 0:
        raise ValueError("target_bin_size_ms must be > 0 when resampling PSTH averages.")
    if target_bin_step_ms is None:
        target_bin_step_s = target_bin_size_s
    else:
        target_bin_step_s = float(target_bin_step_ms) / 1000.0
    if not np.isfinite(target_bin_step_s) or target_bin_step_s <= 0:
        raise ValueError("target_bin_step_ms must be > 0 when resampling PSTH averages.")

    source_edges = np.empty(centers.size + 1, dtype=float)
    source_edges[:-1] = centers - 0.5 * source_bin_size_s
    source_edges[-1] = centers[-1] + 0.5 * source_bin_size_s

    start_center = float(source_edges[0]) + 0.5 * target_bin_size_s
    end_center = float(source_edges[-1]) - 0.5 * target_bin_size_s
    if end_center < start_center:
        raise ValueError(
            "target_bin_size_ms is larger than the available PSTH window for average resampling."
        )

    target_centers = np.arange(
        start_center,
        end_center + 0.5 * target_bin_step_s,
        target_bin_step_s,
        dtype=float,
    )
    if target_centers.size == 0:
        raise ValueError("No target bin centers were generated for average PSTH resampling.")

    target_left = target_centers[:, None] - 0.5 * target_bin_size_s
    target_right = target_centers[:, None] + 0.5 * target_bin_size_s
    source_left = source_edges[:-1][None, :]
    source_right = source_edges[1:][None, :]
    overlap = np.minimum(target_right, source_right) - np.maximum(target_left, source_left)
    overlap = np.clip(overlap, 0.0, None)
    source_widths = (source_edges[1:] - source_edges[:-1])[None, :]
    weights = overlap / source_widths
    return weights, target_centers, target_bin_size_s, target_bin_step_s


def _resolve_uniform_bin_size_s(
    source_centers: np.ndarray,
    *,
    label: str,
) -> float:
    centers = np.asarray(source_centers, dtype=float).reshape(-1)
    if centers.size < 2:
        raise ValueError(f"Cannot resolve {label} bin size from fewer than 2 centers.")
    diffs = np.diff(centers)
    if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
        raise ValueError(f"{label} bin centers must be strictly increasing.")
    bin_size_s = float(np.mean(diffs))
    if not np.allclose(diffs, bin_size_s, atol=max(1e-9, abs(bin_size_s) * 1e-6)):
        raise ValueError(f"{label} bin centers must be approximately uniform.")
    if not np.isfinite(bin_size_s) or bin_size_s <= 0:
        raise ValueError(f"Encountered non-positive {label} bin size.")
    return bin_size_s


def _aggregate_group_counts(
    key: tuple,
    counts_list: Sequence[np.ndarray],
    *,
    smooth_before_average: bool,
    sigma_bins: Optional[float],
    resample_weights: Optional[np.ndarray],
    convert_to_firing_rate: bool,
    firing_rate_bin_size_s: Optional[float],
):
    if convert_to_firing_rate:
        if firing_rate_bin_size_s is None:
            raise ValueError("Missing firing-rate bin size for average conversion.")
        if not np.isfinite(float(firing_rate_bin_size_s)) or float(firing_rate_bin_size_s) <= 0:
            raise ValueError("firing_rate_bin_size_s must be > 0 for average conversion.")

    sum_acc = None
    sumsq_acc = None
    n_trials = 0
    for raw in counts_list:
        counts = np.asarray(raw, dtype=float).reshape(-1)
        if counts.size == 0:
            continue
        if smooth_before_average:
            counts = gaussian_filter1d(counts, sigma=sigma_bins, mode="nearest")
        if resample_weights is not None:
            if resample_weights.shape[1] != counts.size:
                raise ValueError("Encountered inconsistent PSTH bin counts while resampling averages.")
            counts = np.matmul(resample_weights, counts)
        if convert_to_firing_rate:
            counts = counts / float(firing_rate_bin_size_s)
        if sum_acc is None:
            sum_acc = np.zeros_like(counts, dtype=float)
            sumsq_acc = np.zeros_like(counts, dtype=float)
        elif sum_acc.shape != counts.shape:
            raise ValueError("Encountered inconsistent PSTH bin counts while aggregating averages.")
        sum_acc += counts
        sumsq_acc += np.square(counts)
        n_trials += 1
    return key, sum_acc, sumsq_acc, n_trials


def build_fixation_psth_averages_for_date(
    settings: FixationPSTHAverageSettings,
    date: str,
    trial_paths: Sequence[Path],
) -> Optional[dict]:
    """Build one date-level averaged PSTH object from trial files."""
    category_allow = _category_filter(settings)

    grouped_counts: dict[tuple, list[np.ndarray]] = {}
    grouped_labels: dict[tuple, dict[str, set[str]]] = {}
    bin_edges_ref = None
    bin_centers_ref = None

    for path in trial_paths:
        obj = load_pickle_path(path)
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
            label_state = grouped_labels.setdefault(
                key,
                {
                    "fixation_location_labels": set(),
                    "source_fixation_agents": set(),
                    "source_fixation_monkeys": set(),
                    "source_sessions": set(),
                    "source_interactive_states": set(),
                },
            )
            for loc in coerce_location_labels(getattr(row, "fixation_location", None)):
                if str(loc).strip():
                    label_state["fixation_location_labels"].add(str(loc))
            fixation_agent = _as_optional_str(getattr(row, "fixation_agent", None))
            fixation_monkey = _as_optional_str(getattr(row, "fixation_monkey_name", None))
            session_name = _as_optional_str(getattr(row, "session", None))
            if fixation_agent is not None:
                label_state["source_fixation_agents"].add(fixation_agent)
            if fixation_monkey is not None:
                label_state["source_fixation_monkeys"].add(fixation_monkey)
            if session_name is not None:
                label_state["source_sessions"].add(session_name)
            if interactive_state is not None:
                label_state["source_interactive_states"].add(interactive_state)

    if not grouped_counts:
        return None

    source_centers = _resolve_average_output_centers(
        bin_edges_ref=bin_edges_ref,
        bin_centers_ref=bin_centers_ref,
    )
    if source_centers is None:
        raise ValueError(f"Unable to resolve source bin centers for fixation PSTH averages (date={date}).")
    source_bin_size_s = _resolve_uniform_bin_size_s(source_centers, label="source PSTH")

    resample_weights, out_bin_centers, target_bin_size_s, target_bin_step_s = _build_resample_counts_matrix(
        source_centers,
        target_bin_size_ms=settings.target_bin_size_ms,
        target_bin_step_ms=settings.target_bin_step_ms,
    )
    output_bin_size_s = target_bin_size_s if target_bin_size_s is not None else source_bin_size_s

    sigma_bins = _resolve_smoothing_sigma_bins(settings, bin_edges_ref)
    grouped: dict[tuple, dict] = {}
    for key, counts_list in grouped_counts.items():
        _, sum_acc, sumsq_acc, n_trials = _aggregate_group_counts(
            key,
            counts_list,
            smooth_before_average=settings.smooth_before_average,
            sigma_bins=sigma_bins,
            resample_weights=resample_weights,
            convert_to_firing_rate=bool(settings.convert_to_firing_rate_before_average),
            firing_rate_bin_size_s=output_bin_size_s,
        )
        if n_trials <= 0:
            continue
        grouped[key] = {"sum": sum_acc, "sumsq": sumsq_acc, "n": int(n_trials)}

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
        else:
            # Keep schema stable across split and unsplit average outputs.
            record["interactive_state"] = None

        interactive_state_value = _as_optional_str(record.get("interactive_state"))
        if interactive_state_value is None:
            record["is_interactive"] = None
        else:
            record["is_interactive"] = bool(
                interactive_state_value.lower() == "interactive"
            )

        n_trials = int(val["n"])
        sum_vec = np.asarray(val["sum"], dtype=float).reshape(-1)
        sumsq_vec = np.asarray(val["sumsq"], dtype=float).reshape(-1)
        if sum_vec.shape != sumsq_vec.shape:
            raise ValueError("Encountered inconsistent PSTH vector lengths while finalizing averages.")

        record["n_trials"] = n_trials
        mean_vec = sum_vec / max(1, n_trials)
        if n_trials > 1:
            numer = sumsq_vec - (np.square(sum_vec) / float(n_trials))
            numer = np.maximum(numer, 0.0)
            sample_var = numer / float(n_trials - 1)
            sem_vec = np.sqrt(sample_var / float(n_trials))
        else:
            sem_vec = np.zeros_like(mean_vec, dtype=float)

        record["psth_mean"] = mean_vec
        record["psth_sem"] = sem_vec
        label_state = grouped_labels.get(key, {})
        record["fixation_location_labels"] = tuple(
            sorted(label_state.get("fixation_location_labels", set()))
        )
        record["source_fixation_agents"] = tuple(
            sorted(label_state.get("source_fixation_agents", set()))
        )
        record["source_fixation_monkeys"] = tuple(
            sorted(label_state.get("source_fixation_monkeys", set()))
        )
        record["source_sessions"] = tuple(
            sorted(label_state.get("source_sessions", set()))
        )
        record["source_interactive_states"] = tuple(
            sorted(label_state.get("source_interactive_states", set()))
        )
        records.append(record)

    averages_df = pd.DataFrame(records)
    if resample_weights is None:
        out_bin_edges = bin_edges_ref
    else:
        out_bin_edges = None

    return {
        "meta": {
            "date": date,
            "source_modality": settings.trial_input_modality,
            "source_filename": _ensure_pkl_filename(settings.trial_input_filename),
            "smooth_before_average": bool(settings.smooth_before_average),
            "smoothing_sigma_ms": float(settings.smoothing_sigma_ms),
            "smoothing_sigma_bins": sigma_bins,
            "convert_to_firing_rate_before_average": bool(
                settings.convert_to_firing_rate_before_average
            ),
            "psth_value_kind": (
                "firing_rate_hz" if settings.convert_to_firing_rate_before_average else "counts"
            ),
            "output_bin_size_s": output_bin_size_s,
            "output_bin_size_ms": output_bin_size_s * 1000.0,
            "split_by_interactive_state": bool(settings.split_by_interactive_state),
            "restrict_interactive_state": settings.restrict_interactive_state,
            "group_by_session": bool(settings.group_by_session),
            "bin_edges_s_rel": out_bin_edges,
            "bin_centers_s_rel": out_bin_centers,
            "source_bin_edges_s_rel": bin_edges_ref,
            "source_bin_centers_s_rel": source_centers,
            "target_bin_size_ms": (
                float(settings.target_bin_size_ms)
                if settings.target_bin_size_ms is not None
                else None
            ),
            "target_bin_step_ms": (
                float(settings.target_bin_step_ms)
                if settings.target_bin_step_ms is not None
                else None
            ),
            "target_bin_size_s": target_bin_size_s,
            "target_bin_step_s": target_bin_step_s,
        },
        "averages": averages_df,
    }


def build_fixation_psth_averages_bundle_for_date(
    settings: FixationPSTHAverageSettings,
    date: str,
    trial_paths: Sequence[Path],
) -> Optional[dict]:
    split_settings = replace(
        settings,
        split_by_interactive_state=True,
        store_split_and_unsplit_together=False,
    )
    unsplit_settings = replace(
        settings,
        split_by_interactive_state=False,
        store_split_and_unsplit_together=False,
    )

    split_data = build_fixation_psth_averages_for_date(split_settings, date, trial_paths)
    unsplit_data = build_fixation_psth_averages_for_date(unsplit_settings, date, trial_paths)
    if split_data is None and unsplit_data is None:
        return None

    split_meta = split_data.get("meta", {}) if isinstance(split_data, dict) else {}
    unsplit_meta = unsplit_data.get("meta", {}) if isinstance(unsplit_data, dict) else {}
    split_df = (
        split_data.get("averages")
        if isinstance(split_data, dict)
        else pd.DataFrame()
    )
    unsplit_df = (
        unsplit_data.get("averages")
        if isinstance(unsplit_data, dict)
        else pd.DataFrame()
    )

    return {
        "meta": {
            "date": date,
            "source_modality": settings.trial_input_modality,
            "source_filename": _ensure_pkl_filename(settings.trial_input_filename),
            "store_split_and_unsplit_together": True,
            "split_by_interactive_state_default": bool(settings.split_by_interactive_state),
            "restrict_interactive_state": settings.restrict_interactive_state,
            "group_by_session": bool(settings.group_by_session),
            "smooth_before_average": bool(settings.smooth_before_average),
            "smoothing_sigma_ms": float(settings.smoothing_sigma_ms),
            "convert_to_firing_rate_before_average": bool(
                settings.convert_to_firing_rate_before_average
            ),
            "psth_value_kind": (
                "firing_rate_hz" if settings.convert_to_firing_rate_before_average else "counts"
            ),
            "split_filename_key": "averages_split_by_interactive_state",
            "unsplit_filename_key": "averages_unsplit_by_interactive_state",
            "split_meta": split_meta,
            "unsplit_meta": unsplit_meta,
        },
        "averages_split_by_interactive_state": split_df,
        "averages_unsplit_by_interactive_state": unsplit_df,
    }


def process_fixation_psth_averages_for_date(
    settings: FixationPSTHAverageSettings,
    date: str,
    trial_paths: Sequence[Path],
) -> Optional[dict]:
    """Build and persist one date-level PSTH average object."""
    if settings.store_split_and_unsplit_together:
        data = build_fixation_psth_averages_bundle_for_date(settings, date, trial_paths)
    else:
        data = build_fixation_psth_averages_for_date(settings, date, trial_paths)
    if data is None:
        return None
    cfg = load_config(settings.cfg_path)
    out_path = _build_average_output_path(cfg, date, settings)
    save_pickle_path(data, out_path)
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
    cfg = load_config(settings.cfg_path)
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
