"""Build period-centered PSTH trial features from unit-level ephys data."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.data.load import load_ephys_units
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import (
    build_processed_data_path,
    build_processed_out_dir,
    scan_processed_data_paths,
)


@dataclass
class PeriodPSTHSettings:
    """Configuration for session-level period-centered PSTH extraction."""

    cfg_path: str
    ephys_cfg_path: str = "configs/ephys_data.yaml"
    timeline_modality: str = "neural_timeline"
    periods_modality: str = "interactive_periods"
    output_modality: str = "psth"
    trial_output_filename: str = "interactive_periods.pkl"
    state_column: str = "state"
    start_column: str = "start"
    stop_column: str = "stop"
    include_states: Optional[Sequence[str]] = ("interactive", "non_interactive")
    interactive_label: str = "interactive"
    bin_size_ms: float = 100.0
    window_pre_s: float = 14.0
    window_post_s: float = 14.0
    use_parallel: bool = True
    max_procs: int = 16
    test_single: bool = False
    restrict_units_to_date: bool = True


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


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


def _ensure_pkl_filename(filename: str) -> str:
    name = str(filename).strip()
    if not name:
        raise ValueError("Output filename cannot be empty.")
    return name if name.endswith(".pkl") else f"{name}.pkl"


def _build_trial_output_path(cfg: dict, row: dict, settings: PeriodPSTHSettings) -> Path:
    out_dir = build_processed_out_dir(cfg, row, settings.output_modality)
    return out_dir / _ensure_pkl_filename(settings.trial_output_filename)


def _build_bin_edges(settings: PeriodPSTHSettings) -> np.ndarray:
    bin_size_s = float(settings.bin_size_ms) / 1000.0
    if bin_size_s <= 0:
        raise ValueError("bin_size_ms must be > 0.")
    if settings.window_pre_s <= 0 or settings.window_post_s <= 0:
        raise ValueError("window_pre_s and window_post_s must be > 0.")
    pre = float(settings.window_pre_s)
    post = float(settings.window_post_s)
    return np.arange(-pre, post + bin_size_s * 0.5, bin_size_s, dtype=float)


def _as_timeline_array(obj) -> np.ndarray:
    if hasattr(obj, "t"):
        return np.asarray(getattr(obj, "t"), dtype=float).reshape(-1)
    if isinstance(obj, dict) and "t" in obj:
        return np.asarray(obj["t"], dtype=float).reshape(-1)
    if isinstance(obj, pd.DataFrame) and "t" in obj.columns:
        return obj["t"].to_numpy(dtype=float).reshape(-1)
    return np.array([], dtype=float)


def _build_period_tasks(
    settings: PeriodPSTHSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    cfg = load_dataset_config(settings.cfg_path)
    rows = scan_processed_data_paths(
        cfg,
        settings.periods_modality,
        dates=dates,
        sessions=sessions,
        agents=[None],
    )
    tasks: list[dict] = []
    for row in rows:
        tasks.append(
            {
                "date": row["date"],
                "session": row["session"],
                "path": row["path"],
            }
        )
    tasks.sort(key=lambda row: (row["date"], row["session"]))
    if settings.test_single and tasks:
        return [random.choice(tasks)]
    return tasks


def _build_session_period_events(
    settings: PeriodPSTHSettings,
    row: dict,
) -> tuple[list[dict], np.ndarray]:
    cfg = load_dataset_config(settings.cfg_path)
    timeline_path = build_processed_data_path(cfg, row, settings.timeline_modality, None)
    if not timeline_path.exists():
        return [], np.array([], dtype=float)

    timeline_obj = _load_pickle(timeline_path)
    timeline_t = _as_timeline_array(timeline_obj)
    if timeline_t.size == 0:
        return [], timeline_t

    periods_obj = _load_pickle(row["path"])
    if not isinstance(periods_obj, pd.DataFrame) or periods_obj.empty:
        return [], timeline_t

    required = {settings.start_column, settings.stop_column, settings.state_column}
    if not required.issubset(periods_obj.columns):
        return [], timeline_t

    allowed_states = None
    if settings.include_states is not None:
        allowed_states = {str(state).strip().lower() for state in settings.include_states}

    n_t = int(timeline_t.shape[0])
    events: list[dict] = []
    for period_idx, period_row in enumerate(periods_obj.itertuples(index=False)):
        try:
            raw_start = int(getattr(period_row, settings.start_column))
            raw_stop = int(getattr(period_row, settings.stop_column))
        except Exception:
            continue
        if raw_stop < raw_start:
            continue

        start_idx = max(0, raw_start)
        stop_idx = min(raw_stop, n_t - 1)
        if stop_idx < start_idx:
            continue

        center_idx = int(round((start_idx + stop_idx) / 2.0))
        if center_idx < 0 or center_idx >= n_t:
            continue

        state = _as_optional_str(getattr(period_row, settings.state_column))
        if state is None:
            continue
        state_l = state.lower()
        if allowed_states is not None and state_l not in allowed_states:
            continue

        center_time_s = float(timeline_t[center_idx])
        start_time_s = float(timeline_t[start_idx])
        stop_time_s = float(timeline_t[stop_idx])
        if not (np.isfinite(center_time_s) and np.isfinite(start_time_s) and np.isfinite(stop_time_s)):
            continue

        events.append(
            {
                "date": row["date"],
                "session": row["session"],
                "period_index": int(period_idx),
                "period_state": state,
                "is_interactive": bool(state_l == settings.interactive_label.lower()),
                "period_start_idx": int(start_idx),
                "period_stop_idx": int(stop_idx),
                "period_center_idx": int(center_idx),
                "period_start_time_s": start_time_s,
                "period_stop_time_s": stop_time_s,
                "period_center_time_s": center_time_s,
                "period_duration_samples": int(stop_idx - start_idx + 1),
                "period_duration_s": float(stop_time_s - start_time_s),
            }
        )
    return events, timeline_t


_GLOBAL_PERIOD_EVENTS: list[dict] = []
_GLOBAL_PERIOD_BIN_EDGES: np.ndarray = np.array([], dtype=float)


def _init_period_trial_worker(events: list[dict], bin_edges: np.ndarray) -> None:
    global _GLOBAL_PERIOD_EVENTS, _GLOBAL_PERIOD_BIN_EDGES
    _GLOBAL_PERIOD_EVENTS = events
    _GLOBAL_PERIOD_BIN_EDGES = bin_edges


def _compute_unit_period_trial_rows(unit_payload: dict) -> list[dict]:
    rows: list[dict] = []
    spike_ts = np.asarray(unit_payload["spike_ts"], dtype=float)
    for event in _GLOBAL_PERIOD_EVENTS:
        rel = spike_ts - float(event["period_center_time_s"])
        counts, _ = np.histogram(rel, bins=_GLOBAL_PERIOD_BIN_EDGES)
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


_GLOBAL_EVENTS_BY_DATE: dict[str, list[dict]] = {}
_GLOBAL_ALL_SESSION_EVENTS: list[dict] = []
_GLOBAL_DATE_BIN_EDGES: np.ndarray = np.array([], dtype=float)
_GLOBAL_RESTRICT_UNITS_TO_DATE: bool = True


def _init_unit_first_worker(
    events_by_date: dict[str, list[dict]],
    all_session_events: list[dict],
    bin_edges: np.ndarray,
    restrict_units_to_date: bool,
) -> None:
    global _GLOBAL_EVENTS_BY_DATE, _GLOBAL_ALL_SESSION_EVENTS
    global _GLOBAL_DATE_BIN_EDGES, _GLOBAL_RESTRICT_UNITS_TO_DATE
    _GLOBAL_EVENTS_BY_DATE = events_by_date
    _GLOBAL_ALL_SESSION_EVENTS = all_session_events
    _GLOBAL_DATE_BIN_EDGES = bin_edges
    _GLOBAL_RESTRICT_UNITS_TO_DATE = bool(restrict_units_to_date)


def _compute_unit_rows_across_sessions(unit_payload: dict) -> tuple[str, dict[tuple[str, str], list[dict]]]:
    unit_date = str(unit_payload.get("unit_date", ""))
    spike_ts = np.asarray(unit_payload["spike_ts"], dtype=float)
    rows_by_session: dict[tuple[str, str], list[dict]] = {}
    if _GLOBAL_RESTRICT_UNITS_TO_DATE:
        session_payloads = _GLOBAL_EVENTS_BY_DATE.get(unit_date, [])
    else:
        session_payloads = _GLOBAL_ALL_SESSION_EVENTS
    for session_payload in session_payloads:
        session_date = str(session_payload["date"])
        session_name = str(session_payload["session"])
        events = session_payload["events"]
        session_rows: list[dict] = []
        for event in events:
            rel = spike_ts - float(event["period_center_time_s"])
            counts, _ = np.histogram(rel, bins=_GLOBAL_DATE_BIN_EDGES)
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


def _units_to_payloads(units) -> list[dict]:
    payloads: list[dict] = []
    for unit in units:
        ctx = unit.context
        payloads.append(
            {
                "unit_uuid": str(ctx.unit_uuid),
                "unit_date": str(ctx.date),
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


def build_period_psth_trials_for_session(
    settings: PeriodPSTHSettings,
    row: dict,
    units_for_session: Sequence[object],
) -> Optional[dict]:
    """Build session-level period-centered PSTH trials for selected units."""
    events, _ = _build_session_period_events(settings, row)
    if not events:
        return None
    if not units_for_session:
        return None

    bin_edges = _build_bin_edges(settings)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    payloads = _units_to_payloads(units_for_session)
    if not payloads:
        return None

    all_rows: list[dict] = []
    if settings.use_parallel and len(payloads) > 1:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        with Pool(
            processes=n_proc,
            initializer=_init_period_trial_worker,
            initargs=(events, bin_edges),
        ) as pool:
            for unit_rows in pool.imap_unordered(_compute_unit_period_trial_rows, payloads):
                if unit_rows:
                    all_rows.extend(unit_rows)
    else:
        _init_period_trial_worker(events, bin_edges)
        for payload in payloads:
            unit_rows = _compute_unit_period_trial_rows(payload)
            if unit_rows:
                all_rows.extend(unit_rows)

    if not all_rows:
        return None

    trial_df = pd.DataFrame(all_rows)
    return {
        "meta": {
            "date": row["date"],
            "session": row["session"],
            "event_source": settings.periods_modality,
            "event_anchor": "period_center",
            "state_column": settings.state_column,
            "start_column": settings.start_column,
            "stop_column": settings.stop_column,
            "include_states": (
                None if settings.include_states is None else [str(v) for v in settings.include_states]
            ),
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


def process_period_psth_trials_for_session(
    settings: PeriodPSTHSettings,
    row: dict,
    units_for_session: Sequence[object],
) -> Optional[dict]:
    """Build and persist session-level period-centered PSTH trial data."""
    data = build_period_psth_trials_for_session(settings, row, units_for_session)
    if data is None:
        return None
    cfg = load_dataset_config(settings.cfg_path)
    out_path = _build_trial_output_path(cfg, row, settings)
    _save_pickle(data, out_path)
    return data


def _build_trial_payload(
    settings: PeriodPSTHSettings,
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
            "event_source": settings.periods_modality,
            "event_anchor": "period_center",
            "state_column": settings.state_column,
            "start_column": settings.start_column,
            "stop_column": settings.stop_column,
            "include_states": (
                None if settings.include_states is None else [str(v) for v in settings.include_states]
            ),
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


def _flush_session_rows(
    cfg: dict,
    settings: PeriodPSTHSettings,
    rows_for_session: dict[tuple[str, str], list[dict]],
    *,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
) -> None:
    for (date, session), rows in rows_for_session.items():
        if not rows:
            continue
        row = {"date": str(date), "session": str(session)}
        data = _build_trial_payload(
            settings,
            date=str(date),
            session=str(session),
            rows=rows,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
        )
        out_path = _build_trial_output_path(cfg, row, settings)
        _save_pickle(data, out_path)


def run_period_psth_trial_build(
    settings: PeriodPSTHSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> None:
    """Run interactive/non-interactive period PSTH extraction.

    Execution model:
    - Build session events once per session.
    - Run one global worker pool across all unit payloads.
    - Aggregate rows back into per-session outputs.
    """
    if use_parallel is not None:
        settings.use_parallel = bool(use_parallel)
    if test_single is not None:
        settings.test_single = bool(test_single)

    session_rows = _build_period_tasks(settings, dates=dates, sessions=sessions)
    if not session_rows:
        print("No period PSTH tasks found.")
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
            events, _ = _build_session_period_events(settings, row)
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
        print("No period PSTH session events found.")
        return

    all_session_events: list[dict] = []
    for date in sorted(session_events_by_date):
        all_session_events.extend(session_events_by_date[date])

    cfg = load_dataset_config(settings.cfg_path)
    date_filter = sorted(session_events_by_date.keys()) if settings.restrict_units_to_date else None
    all_units = load_ephys_units(
        cfg_path=settings.cfg_path,
        ephys_cfg_path=settings.ephys_cfg_path,
        dates=date_filter,
    )
    payloads = _units_to_payloads(all_units)
    if settings.restrict_units_to_date:
        payloads = [payload for payload in payloads if payload["unit_date"] in session_events_by_date]
    if not payloads:
        print("No matching ephys units found for period PSTH tasks.")
        return

    bin_edges = _build_bin_edges(settings)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    buffered_rows_by_date: dict[str, dict[tuple[str, str], list[dict]]] = {}
    for date, session_payloads in session_events_by_date.items():
        buffered_rows_by_date[date] = {
            (date, str(payload["session"])): [] for payload in session_payloads
        }

    remaining_units_by_date: dict[str, int] = {}
    if settings.restrict_units_to_date:
        for payload in payloads:
            date = str(payload["unit_date"])
            remaining_units_by_date[date] = remaining_units_by_date.get(date, 0) + 1

    def _accumulate_and_flush(unit_date: str, rows_by_session: dict[tuple[str, str], list[dict]]) -> None:
        for key, rows in rows_by_session.items():
            date_key = str(key[0])
            date_bucket = buffered_rows_by_date.setdefault(date_key, {})
            date_bucket.setdefault((str(key[0]), str(key[1])), []).extend(rows)

        if not settings.restrict_units_to_date:
            return

        remaining = remaining_units_by_date.get(unit_date, 0) - 1
        remaining_units_by_date[unit_date] = remaining
        if remaining <= 0:
            rows_for_date = buffered_rows_by_date.pop(unit_date, {})
            _flush_session_rows(
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
            initializer=_init_unit_first_worker,
            initargs=(
                session_events_by_date,
                all_session_events,
                bin_edges,
                settings.restrict_units_to_date,
            ),
        ) as pool:
            iterator = pool.imap_unordered(_compute_unit_rows_across_sessions, payloads)
            for unit_date, rows_by_session in tqdm(
                iterator,
                total=len(payloads),
                desc="Building period PSTH trials",
                unit="unit",
            ):
                _accumulate_and_flush(str(unit_date), rows_by_session)
    else:
        _init_unit_first_worker(
            session_events_by_date,
            all_session_events,
            bin_edges,
            settings.restrict_units_to_date,
        )
        for payload in tqdm(payloads, desc="Building period PSTH trials", unit="unit"):
            unit_date, rows_by_session = _compute_unit_rows_across_sessions(payload)
            _accumulate_and_flush(str(unit_date), rows_by_session)

    for _, rows_for_date in buffered_rows_by_date.items():
        _flush_session_rows(
            cfg,
            settings,
            rows_for_date,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
        )
