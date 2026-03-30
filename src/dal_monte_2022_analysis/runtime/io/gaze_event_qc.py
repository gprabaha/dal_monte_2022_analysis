"""Runtime IO helpers for gaze-event QC session discovery and artifact loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.contracts.gaze_events import validate_gaze_event_frame
from dal_monte_2022_analysis.data.records.behavioral import PositionData, ROIRectsData
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_processed_pickle,
    scan_processed_paths,
)

DEFAULT_GAZE_EVENT_AGENTS: tuple[str, ...] = ("m1", "m2")
DEFAULT_GAZE_EVENT_MODALITIES: tuple[str, ...] = (
    "gaze_position",
    "fixations",
    "saccades",
    "roi_vertices",
)


@dataclass(frozen=True)
class SessionKey:
    """Logical identifier for one date/session pair."""

    date: str
    session: str


@dataclass
class AgentGazeEventArtifacts:
    """Behavioral artifacts required for fixation/saccade QC plotting."""

    position: PositionData
    fixations: pd.DataFrame
    saccades: pd.DataFrame
    rois: ROIRectsData


@dataclass
class SessionGazeEventArtifacts:
    """Paired-agent artifacts for one behavioral session."""

    key: SessionKey
    agents: dict[str, AgentGazeEventArtifacts]


def _resolve_cfg(cfg_or_path: dict | str | Path) -> dict:
    if isinstance(cfg_or_path, (str, Path)):
        return load_config(str(cfg_or_path))
    return cfg_or_path


def normalize_gaze_event_agents(agents: Sequence[str]) -> tuple[str, ...]:
    """Normalize requested gaze-event agent labels."""
    out = tuple(str(agent) for agent in agents)
    if not out:
        raise ValueError("Expected at least one agent.")
    return out


def _session_agent_presence(
    cfg: dict,
    *,
    modalities: Sequence[str],
    agents: Sequence[str],
) -> pd.DataFrame:
    requested_agents = set(normalize_gaze_event_agents(agents))
    merged: Optional[pd.DataFrame] = None
    for modality in modalities:
        modality_df = pd.DataFrame(
            scan_processed_paths(cfg, modality, agents=list(requested_agents))
        )
        if modality_df.empty:
            return pd.DataFrame(columns=["date", "session"])
        modality_df = modality_df[["date", "session", "agent"]].copy()
        modality_df["agent"] = modality_df["agent"].astype(str)
        modality_df = modality_df.drop_duplicates()
        merged = modality_df if merged is None else merged.merge(
            modality_df,
            on=["date", "session", "agent"],
            how="inner",
        )

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["date", "session"])

    counts = (
        merged.groupby(["date", "session"], as_index=False)["agent"]
        .nunique()
        .rename(columns={"agent": "n_agents"})
    )
    paired = counts[counts["n_agents"] == len(requested_agents)][["date", "session"]]
    paired = paired.sort_values(["date", "session"]).reset_index(drop=True)
    return paired


def find_paired_gaze_event_sessions(
    cfg_or_path: dict | str | Path,
    *,
    agents: Sequence[str] = DEFAULT_GAZE_EVENT_AGENTS,
    required_modalities: Sequence[str] = DEFAULT_GAZE_EVENT_MODALITIES,
) -> pd.DataFrame:
    """Return date/session pairs that exist for all requested agents/modalities."""
    cfg = _resolve_cfg(cfg_or_path)
    return _session_agent_presence(
        cfg,
        modalities=required_modalities,
        agents=agents,
    )


def sample_random_paired_gaze_event_sessions(
    cfg_or_path: dict | str | Path,
    *,
    n_sessions: int = 5,
    random_state: Optional[int] = None,
    agents: Sequence[str] = DEFAULT_GAZE_EVENT_AGENTS,
    required_modalities: Sequence[str] = DEFAULT_GAZE_EVENT_MODALITIES,
) -> pd.DataFrame:
    """Sample random date/session pairs that contain all requested gaze artifacts."""
    available = find_paired_gaze_event_sessions(
        cfg_or_path,
        agents=agents,
        required_modalities=required_modalities,
    )
    if available.empty or n_sessions <= 0:
        return available.head(0).copy()

    sample_n = min(int(n_sessions), len(available))
    sampled = available.sample(n=sample_n, random_state=random_state)
    return sampled.sort_values(["date", "session"]).reset_index(drop=True)


def load_gaze_event_session_artifacts(
    cfg_or_path: dict | str | Path,
    *,
    date: str,
    session: str,
    agents: Sequence[str] = DEFAULT_GAZE_EVENT_AGENTS,
) -> SessionGazeEventArtifacts:
    """Load gaze positions, fixations, saccades, and ROI boxes for one session."""
    cfg = _resolve_cfg(cfg_or_path)
    row = {"date": str(date), "session": str(session)}
    agent_payloads: dict[str, AgentGazeEventArtifacts] = {}
    for agent in normalize_gaze_event_agents(agents):
        position = load_processed_pickle(cfg, row, "gaze_position", agent)
        fixations = validate_gaze_event_frame(
            load_processed_pickle(cfg, row, "fixations", agent)
        )
        saccades = validate_gaze_event_frame(
            load_processed_pickle(cfg, row, "saccades", agent)
        )
        rois = load_processed_pickle(cfg, row, "roi_vertices", agent)
        agent_payloads[agent] = AgentGazeEventArtifacts(
            position=position,
            fixations=fixations,
            saccades=saccades,
            rois=rois,
        )

    return SessionGazeEventArtifacts(
        key=SessionKey(date=str(date), session=str(session)),
        agents=agent_payloads,
    )


__all__ = [
    "AgentGazeEventArtifacts",
    "DEFAULT_GAZE_EVENT_AGENTS",
    "DEFAULT_GAZE_EVENT_MODALITIES",
    "SessionGazeEventArtifacts",
    "SessionKey",
    "find_paired_gaze_event_sessions",
    "load_gaze_event_session_artifacts",
    "normalize_gaze_event_agents",
    "sample_random_paired_gaze_event_sessions",
]
