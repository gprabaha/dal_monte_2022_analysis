"""Compute face-fixation gap distributions for m1 and cross-monkey events."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.behav.feature_primitives import (
    extract_monkey_name,
)
from dal_monte_2022_analysis.core.behav.roi_groups import (
    DEFAULT_FIXATION_ROI_GROUPS,
    coerce_location_labels,
    locations_match,
    resolve_agent_roi_groups,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.processed_data import (
    index_agent_paths,
    index_shared_paths,
    load_pickle_path,
)


_FIXATION_REQUIRED_COLUMNS = {"start", "stop", "location"}
_PERIOD_REQUIRED_COLUMNS = {"start", "stop", "state"}
_M1_OUTPUT_COLUMNS = [
    "date",
    "session",
    "agent",
    "monkey_name",
    "period_index",
    "pair_index",
    "period_state",
    "period_start",
    "period_stop",
    "prev_fixation_index",
    "prev_fixation_start",
    "prev_fixation_stop",
    "next_fixation_index",
    "next_fixation_start",
    "next_fixation_stop",
    "pair_start_to_start_samples",
    "pair_start_to_start_ms",
    "pair_within_max_start_to_start_gap",
    "gap_metric",
    "gap_samples",
    "gap_ms",
    "sample_rate_hz",
]
_M1_M2_OUTPUT_COLUMNS = [
    "date",
    "session",
    "period_index",
    "pair_index",
    "period_state",
    "period_start",
    "period_stop",
    "prev_agent",
    "next_agent",
    "transition",
    "prev_monkey_name",
    "next_monkey_name",
    "prev_fixation_index",
    "prev_fixation_start",
    "prev_fixation_stop",
    "next_fixation_index",
    "next_fixation_start",
    "next_fixation_stop",
    "pair_start_to_start_samples",
    "pair_start_to_start_ms",
    "pair_within_max_start_to_start_gap",
    "gap_metric",
    "gap_samples",
    "gap_ms",
    "sample_rate_hz",
]
_SUMMARY_OUTPUT_COLUMNS = [
    "scope",
    "group_type",
    "group_label",
    "max_pair_gap_ms",
    "n_candidate_pairs",
    "n_kept_pairs",
    "n_discarded_pairs",
    "kept_fraction",
    "discarded_fraction",
]


@dataclass
class FaceFixationGapDistributionSettings:
    """Configuration for face-fixation gap distribution analysis."""

    cfg_path: str
    fixations_modality: str = "fixations"
    interactive_modality: str = "interactive_periods"
    fixation_label: str = "face"
    output_subdir: str = "face_fixation_gap_distributions"
    m1_output_filename: str = "within_session_m1_face_fixation_gap_distribution.csv"
    m1_m2_output_filename: str = (
        "within_session_interactive_m1_m2_face_fixation_gap_distribution.csv"
    )
    filter_summary_filename: str = "face_fixation_gap_distribution_filter_summary.csv"
    interactive_state_label: str = "interactive"
    non_interactive_state_label: str = "non_interactive"
    max_pair_gap_ms: Optional[float] = 5000.0
    sample_rate_hz: float = 1000.0
    agent_m1: str = "m1"
    agent_m2: str = "m2"
    roi_groups: dict[str, Sequence[str]] = field(
        default_factory=lambda: {
            key: tuple(value) for key, value in DEFAULT_FIXATION_ROI_GROUPS.items()
        }
    )
    agent_roi_groups: Optional[dict[str, dict[str, Sequence[str]]]] = None
    test_single: bool = False


def _gap_ms(gap_samples: int, sample_rate_hz: float) -> float:
    """Convert a sample-index difference to milliseconds."""
    return (float(gap_samples) * 1000.0) / float(sample_rate_hz)


def _resolve_face_keywords(
    settings: FaceFixationGapDistributionSettings,
    agent: str,
) -> tuple[str, ...]:
    """Resolve the face ROI keywords for one agent."""
    fixation_label = str(settings.fixation_label).strip().lower()
    roi_groups = resolve_agent_roi_groups(
        agent=agent,
        roi_groups=settings.roi_groups,
        agent_roi_groups=settings.agent_roi_groups,
        include_defaults=True,
    )
    keywords = tuple(roi_groups.get(fixation_label, []))
    if not keywords:
        raise RuntimeError(
            f"Missing ROI keywords for fixation label '{fixation_label}' "
            f"and agent '{agent}'."
        )
    return keywords


def _coerce_fixation_table(obj) -> pd.DataFrame:
    """Coerce a fixation pickle payload to a valid fixation table."""
    if not isinstance(obj, pd.DataFrame) or obj.empty:
        return pd.DataFrame(columns=sorted(_FIXATION_REQUIRED_COLUMNS))
    if not _FIXATION_REQUIRED_COLUMNS.issubset(obj.columns):
        return pd.DataFrame(columns=sorted(_FIXATION_REQUIRED_COLUMNS))
    return obj.copy()


def _coerce_period_table(obj) -> pd.DataFrame:
    """Coerce an interactive-period pickle payload to a valid period table."""
    if not isinstance(obj, pd.DataFrame) or obj.empty:
        return pd.DataFrame(columns=["start", "stop", "state"])
    if not _PERIOD_REQUIRED_COLUMNS.issubset(obj.columns):
        return pd.DataFrame(columns=["start", "stop", "state"])

    frame = obj.copy()
    starts = pd.to_numeric(frame["start"], errors="coerce")
    stops = pd.to_numeric(frame["stop"], errors="coerce")
    valid = starts.notna() & stops.notna() & (stops >= starts)
    if not valid.any():
        return pd.DataFrame(columns=["start", "stop", "state"])

    out = frame.loc[valid, ["start", "stop", "state"]].copy()
    out["start"] = starts.loc[valid].astype(int)
    out["stop"] = stops.loc[valid].astype(int)
    out["state"] = out["state"].astype(str)
    return out.sort_values(["start", "stop"]).reset_index(drop=True)


def _extract_face_fixation_events(
    fix_df: pd.DataFrame,
    *,
    agent: str,
    keywords: Sequence[str],
) -> list[dict]:
    """Extract sorted face-fixation events from a fixation table."""
    if fix_df.empty:
        return []

    monkey_name = extract_monkey_name(fix_df)
    events: list[dict] = []
    for _, row in fix_df.iterrows():
        start = pd.to_numeric(row.get("start"), errors="coerce")
        stop = pd.to_numeric(row.get("stop"), errors="coerce")
        if pd.isna(start) or pd.isna(stop):
            continue

        start_i = int(start)
        stop_i = int(stop)
        if stop_i < start_i:
            continue

        locations = coerce_location_labels(row.get("location"), lowercase=True)
        if not locations_match(locations, keywords):
            continue

        events.append(
            {
                "agent": str(agent),
                "monkey_name": monkey_name,
                "start": start_i,
                "stop": stop_i,
            }
        )

    events.sort(key=lambda event: (event["start"], event["stop"]))
    for idx, event in enumerate(events):
        event["fixation_index"] = int(idx)
    return events


def _events_starting_within_period(
    events: Sequence[dict],
    *,
    start: int,
    stop: int,
) -> list[dict]:
    """Return events whose fixation starts fall within one period."""
    return [
        event
        for event in events
        if int(start) <= int(event["start"]) <= int(stop)
    ]


def _gap_rows_from_pair(
    *,
    base_row: dict,
    prev_event: dict,
    next_event: dict,
    pair_index: int,
    max_pair_gap_ms: Optional[float],
    sample_rate_hz: float,
) -> list[dict]:
    """Build tidy gap rows for one consecutive fixation pair."""
    start_to_start_samples = int(next_event["start"]) - int(prev_event["start"])
    start_to_start_ms = _gap_ms(start_to_start_samples, sample_rate_hz)
    within_limit = True
    if max_pair_gap_ms is not None:
        within_limit = bool(start_to_start_ms <= float(max_pair_gap_ms))

    gap_specs = [
        ("start_to_start", start_to_start_samples),
        ("stop_to_start", int(next_event["start"]) - int(prev_event["stop"])),
    ]

    rows: list[dict] = []
    for gap_metric, gap_samples in gap_specs:
        rows.append(
            {
                **base_row,
                "pair_index": int(pair_index),
                "prev_fixation_index": int(prev_event["fixation_index"]),
                "prev_fixation_start": int(prev_event["start"]),
                "prev_fixation_stop": int(prev_event["stop"]),
                "next_fixation_index": int(next_event["fixation_index"]),
                "next_fixation_start": int(next_event["start"]),
                "next_fixation_stop": int(next_event["stop"]),
                "pair_start_to_start_samples": int(start_to_start_samples),
                "pair_start_to_start_ms": float(start_to_start_ms),
                "pair_within_max_start_to_start_gap": bool(within_limit),
                "gap_metric": gap_metric,
                "gap_samples": int(gap_samples),
                "gap_ms": _gap_ms(int(gap_samples), sample_rate_hz),
                "sample_rate_hz": float(sample_rate_hz),
            }
        )
    return rows


def build_m1_face_fixation_gap_distribution_table(
    settings: FaceFixationGapDistributionSettings,
) -> pd.DataFrame:
    """Build the tidy gap table for m1 face fixations by interactive state."""
    if float(settings.sample_rate_hz) <= 0:
        raise ValueError("sample_rate_hz must be > 0.")

    cfg = load_config(settings.cfg_path)
    m1_paths, _ = index_agent_paths(
        cfg,
        settings.fixations_modality,
        agent_a=settings.agent_m1,
        agent_b=settings.agent_m2,
    )
    interactive_paths = index_shared_paths(cfg, settings.interactive_modality)

    if not m1_paths:
        raise RuntimeError(
            f"Missing processed fixation files for agent '{settings.agent_m1}' "
            f"under modality '{settings.fixations_modality}'."
        )

    shared_keys = sorted(set(m1_paths).intersection(interactive_paths))
    if settings.test_single and shared_keys:
        shared_keys = [shared_keys[0]]

    face_keywords = _resolve_face_keywords(settings, settings.agent_m1)
    rows: list[dict] = []

    for date, session in shared_keys:
        fix_df = _coerce_fixation_table(load_pickle_path(m1_paths[(date, session)]))
        periods_df = _coerce_period_table(load_pickle_path(interactive_paths[(date, session)]))
        if fix_df.empty or periods_df.empty:
            continue

        events = _extract_face_fixation_events(
            fix_df,
            agent=settings.agent_m1,
            keywords=face_keywords,
        )
        if len(events) < 2:
            continue

        for period_index, period_row in periods_df.iterrows():
            period_start = int(period_row["start"])
            period_stop = int(period_row["stop"])
            period_state = str(period_row["state"])
            period_events = _events_starting_within_period(
                events,
                start=period_start,
                stop=period_stop,
            )
            if len(period_events) < 2:
                continue

            base_row = {
                "date": str(date),
                "session": str(session),
                "agent": settings.agent_m1,
                "monkey_name": period_events[0].get("monkey_name"),
                "period_index": int(period_index),
                "period_state": period_state,
                "period_start": period_start,
                "period_stop": period_stop,
            }
            for pair_index, (prev_event, next_event) in enumerate(
                zip(period_events, period_events[1:])
            ):
                rows.extend(
                    _gap_rows_from_pair(
                        base_row=base_row,
                        prev_event=prev_event,
                        next_event=next_event,
                        pair_index=pair_index,
                        max_pair_gap_ms=settings.max_pair_gap_ms,
                        sample_rate_hz=float(settings.sample_rate_hz),
                    )
                )

    if not rows:
        return pd.DataFrame(columns=_M1_OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=_M1_OUTPUT_COLUMNS)


def build_interactive_m1_m2_face_fixation_gap_distribution_table(
    settings: FaceFixationGapDistributionSettings,
) -> pd.DataFrame:
    """Build the tidy gap table for interactive cross-monkey face-fixation pairs."""
    if float(settings.sample_rate_hz) <= 0:
        raise ValueError("sample_rate_hz must be > 0.")

    cfg = load_config(settings.cfg_path)
    m1_paths, m2_paths = index_agent_paths(
        cfg,
        settings.fixations_modality,
        agent_a=settings.agent_m1,
        agent_b=settings.agent_m2,
    )
    interactive_paths = index_shared_paths(cfg, settings.interactive_modality)

    if not m1_paths or not m2_paths:
        raise RuntimeError(
            "Missing processed fixation files for m1 or m2 under modality "
            f"'{settings.fixations_modality}'."
        )

    shared_keys = sorted(set(m1_paths).intersection(m2_paths).intersection(interactive_paths))
    if settings.test_single and shared_keys:
        shared_keys = [shared_keys[0]]

    face_keywords_m1 = _resolve_face_keywords(settings, settings.agent_m1)
    face_keywords_m2 = _resolve_face_keywords(settings, settings.agent_m2)
    agent_order = {settings.agent_m1: 0, settings.agent_m2: 1}
    rows: list[dict] = []

    for date, session in shared_keys:
        m1_fix_df = _coerce_fixation_table(load_pickle_path(m1_paths[(date, session)]))
        m2_fix_df = _coerce_fixation_table(load_pickle_path(m2_paths[(date, session)]))
        periods_df = _coerce_period_table(load_pickle_path(interactive_paths[(date, session)]))
        if m1_fix_df.empty or m2_fix_df.empty or periods_df.empty:
            continue

        periods_df = periods_df[
            periods_df["state"].astype(str) == str(settings.interactive_state_label)
        ].reset_index(drop=True)
        if periods_df.empty:
            continue

        m1_events = _extract_face_fixation_events(
            m1_fix_df,
            agent=settings.agent_m1,
            keywords=face_keywords_m1,
        )
        m2_events = _extract_face_fixation_events(
            m2_fix_df,
            agent=settings.agent_m2,
            keywords=face_keywords_m2,
        )
        if not m1_events or not m2_events:
            continue

        for period_index, period_row in periods_df.iterrows():
            period_start = int(period_row["start"])
            period_stop = int(period_row["stop"])
            period_state = str(period_row["state"])

            period_events = _events_starting_within_period(
                m1_events,
                start=period_start,
                stop=period_stop,
            ) + _events_starting_within_period(
                m2_events,
                start=period_start,
                stop=period_stop,
            )
            if len(period_events) < 2:
                continue

            period_events.sort(
                key=lambda event: (
                    int(event["start"]),
                    int(event["stop"]),
                    agent_order.get(str(event["agent"]), 99),
                )
            )

            pair_index = 0
            for prev_event, next_event in zip(period_events, period_events[1:]):
                if str(prev_event["agent"]) == str(next_event["agent"]):
                    continue

                base_row = {
                    "date": str(date),
                    "session": str(session),
                    "period_index": int(period_index),
                    "period_state": period_state,
                    "period_start": period_start,
                    "period_stop": period_stop,
                    "prev_agent": str(prev_event["agent"]),
                    "next_agent": str(next_event["agent"]),
                    "transition": f"{prev_event['agent']}_to_{next_event['agent']}",
                    "prev_monkey_name": prev_event.get("monkey_name"),
                    "next_monkey_name": next_event.get("monkey_name"),
                }
                rows.extend(
                    _gap_rows_from_pair(
                        base_row=base_row,
                        prev_event=prev_event,
                        next_event=next_event,
                        pair_index=pair_index,
                        max_pair_gap_ms=settings.max_pair_gap_ms,
                        sample_rate_hz=float(settings.sample_rate_hz),
                    )
                )
                pair_index += 1

    if not rows:
        return pd.DataFrame(columns=_M1_M2_OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=_M1_M2_OUTPUT_COLUMNS)


def _filter_gap_rows_to_max_pair_gap(
    df: pd.DataFrame,
    *,
    max_pair_gap_ms: Optional[float],
) -> pd.DataFrame:
    """Restrict a gap table to pairs within the requested start-to-start cutoff."""
    if df.empty or max_pair_gap_ms is None:
        return df.copy()
    if "pair_within_max_start_to_start_gap" not in df.columns:
        return df.copy()
    keep_mask = df["pair_within_max_start_to_start_gap"].fillna(False).astype(bool)
    return df.loc[keep_mask].reset_index(drop=True)


def _summary_row(
    pair_rows: pd.DataFrame,
    *,
    scope: str,
    group_type: str,
    group_label: str,
    max_pair_gap_ms: Optional[float],
) -> dict:
    """Build one pair-retention summary row from start-to-start rows."""
    n_candidate = int(len(pair_rows))
    if pair_rows.empty:
        n_kept = 0
    else:
        kept_mask = pair_rows["pair_within_max_start_to_start_gap"].fillna(False).astype(bool)
        n_kept = int(kept_mask.sum())
    n_discarded = int(n_candidate - n_kept)
    kept_fraction = float(n_kept / n_candidate) if n_candidate > 0 else 0.0
    discarded_fraction = float(n_discarded / n_candidate) if n_candidate > 0 else 0.0
    return {
        "scope": scope,
        "group_type": group_type,
        "group_label": group_label,
        "max_pair_gap_ms": max_pair_gap_ms,
        "n_candidate_pairs": n_candidate,
        "n_kept_pairs": n_kept,
        "n_discarded_pairs": n_discarded,
        "kept_fraction": kept_fraction,
        "discarded_fraction": discarded_fraction,
    }


def _build_scope_filter_summary(
    df: pd.DataFrame,
    *,
    scope: str,
    max_pair_gap_ms: Optional[float],
) -> list[dict]:
    """Summarize kept vs discarded consecutive pairs for one scope."""
    if df.empty:
        return []
    if "gap_metric" not in df.columns:
        return []

    pair_rows = df[df["gap_metric"] == "start_to_start"].copy()
    if pair_rows.empty:
        return []

    rows = [
        _summary_row(
            pair_rows,
            scope=scope,
            group_type="overall",
            group_label="all",
            max_pair_gap_ms=max_pair_gap_ms,
        )
    ]
    if "period_state" in pair_rows.columns:
        for label, subset in pair_rows.groupby("period_state", dropna=False):
            rows.append(
                _summary_row(
                    subset,
                    scope=scope,
                    group_type="period_state",
                    group_label=str(label),
                    max_pair_gap_ms=max_pair_gap_ms,
                )
            )
    if "transition" in pair_rows.columns:
        for label, subset in pair_rows.groupby("transition", dropna=False):
            rows.append(
                _summary_row(
                    subset,
                    scope=scope,
                    group_type="transition",
                    group_label=str(label),
                    max_pair_gap_ms=max_pair_gap_ms,
                )
            )
    return rows


def build_face_fixation_gap_filter_summary_table(
    m1_df: pd.DataFrame,
    m1_m2_df: pd.DataFrame,
    *,
    max_pair_gap_ms: Optional[float],
) -> pd.DataFrame:
    """Summarize the retained/discarded consecutive-pair fractions."""
    rows = _build_scope_filter_summary(
        m1_df,
        scope="m1_face",
        max_pair_gap_ms=max_pair_gap_ms,
    )
    rows.extend(
        _build_scope_filter_summary(
            m1_m2_df,
            scope="interactive_m1_m2_face",
            max_pair_gap_ms=max_pair_gap_ms,
        )
    )
    if not rows:
        return pd.DataFrame(columns=_SUMMARY_OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=_SUMMARY_OUTPUT_COLUMNS)


def run_face_fixation_gap_distribution_analysis(
    settings: FaceFixationGapDistributionSettings,
) -> tuple[Path, Path, Path]:
    """Build and persist the m1 and interactive m1-m2 face-fixation gap tables."""
    cfg = load_config(settings.cfg_path)
    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m1_all_df = build_m1_face_fixation_gap_distribution_table(settings)
    m1_m2_all_df = build_interactive_m1_m2_face_fixation_gap_distribution_table(settings)
    summary_df = build_face_fixation_gap_filter_summary_table(
        m1_all_df,
        m1_m2_all_df,
        max_pair_gap_ms=settings.max_pair_gap_ms,
    )
    m1_df = _filter_gap_rows_to_max_pair_gap(
        m1_all_df,
        max_pair_gap_ms=settings.max_pair_gap_ms,
    )
    m1_m2_df = _filter_gap_rows_to_max_pair_gap(
        m1_m2_all_df,
        max_pair_gap_ms=settings.max_pair_gap_ms,
    )

    m1_path = out_dir / settings.m1_output_filename
    m1_m2_path = out_dir / settings.m1_m2_output_filename
    summary_path = out_dir / settings.filter_summary_filename
    m1_df.to_csv(m1_path, index=False)
    m1_m2_df.to_csv(m1_m2_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return m1_path, m1_m2_path, summary_path
