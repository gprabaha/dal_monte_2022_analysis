"""Processed-data IO adapters.

This module centralizes path resolution plus pickle read/write for processed
artifacts so workflow modules can stay focused on domain orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from dal_monte_2022_analysis.utils.io import load_pickle, save_pickle
from dal_monte_2022_analysis.utils.paths import (
    build_processed_data_path,
    scan_processed_data_paths,
)


def build_processed_pickle_path(
    cfg: dict,
    row: dict,
    modality: str,
    agent: Optional[str],
):
    """Build the canonical processed pickle path for one row/modality/agent."""
    return build_processed_data_path(cfg, row, modality, agent)


def load_processed_pickle(
    cfg: dict,
    row: dict,
    modality: str,
    agent: Optional[str],
) -> Any:
    """Load one processed artifact pickle by logical coordinates."""
    path = build_processed_pickle_path(cfg, row, modality, agent)
    return load_pickle(path)


def load_pickle_path(path: str | Path) -> Any:
    """Load one pickle object from an explicit path."""
    return load_pickle(path)


def save_processed_pickle(
    obj: Any,
    cfg: dict,
    row: dict,
    modality: str,
    agent: Optional[str],
) -> None:
    """Save one processed artifact pickle by logical coordinates."""
    path = build_processed_pickle_path(cfg, row, modality, agent)
    save_pickle(obj, path)


def save_pickle_path(obj: Any, path: str | Path) -> None:
    """Save one pickle object to an explicit path."""
    save_pickle(obj, path)


def scan_processed_paths(
    cfg: dict,
    modality: str,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> list[dict]:
    """List available processed artifact paths for a modality."""
    return scan_processed_data_paths(
        cfg,
        modality,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )


def scan_processed_paths_for_filename(
    cfg: dict,
    modality: str,
    *,
    filename: str,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> list[dict]:
    """List processed artifact paths matching one filename under a modality."""
    target_name = Path(str(filename)).name
    rows = scan_processed_paths(
        cfg,
        modality,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )
    filtered = [row for row in rows if Path(row["path"]).name == target_name]
    filtered.sort(key=lambda row: (row["date"], row["session"], str(row.get("agent") or "")))
    return filtered


def index_agent_paths(
    cfg: dict,
    modality: str,
    *,
    agent_a: str = "m1",
    agent_b: str = "m2",
) -> tuple[dict, dict]:
    """Index two agent-specific modality paths by (date, session)."""
    rows = scan_processed_paths(cfg, modality)
    if not rows:
        raise RuntimeError(f"No processed files found for modality '{modality}'")

    a_paths: dict[tuple[str, str], object] = {}
    b_paths: dict[tuple[str, str], object] = {}
    for row in rows:
        agent = row.get("agent")
        key = (row["date"], row["session"])
        if agent == agent_a:
            a_paths[key] = row["path"]
        elif agent == agent_b:
            b_paths[key] = row["path"]
    return a_paths, b_paths


def index_shared_paths(cfg: dict, modality: str) -> dict:
    """Index shared (agent-less) modality paths by (date, session)."""
    rows = scan_processed_paths(cfg, modality)
    if not rows:
        raise RuntimeError(f"No processed files found for modality '{modality}'")

    shared_paths: dict[tuple[str, str], object] = {}
    for row in rows:
        if row.get("agent") is None:
            shared_paths[(row["date"], row["session"])] = row["path"]
    return shared_paths
