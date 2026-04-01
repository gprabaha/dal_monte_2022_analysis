"""Processed-data IO adapters.

This module is the canonical source of truth for processed-data path
construction, scanning, and pickle IO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from dal_monte_2022_analysis.utils.io import load_pickle, save_pickle


def _processed_layout_pattern(cfg: dict) -> str:
    layout_cfg = cfg.get("processed_data_layout", {}) or {}
    return str(layout_cfg.get("pattern", "date={date}/session={session}/{modality}"))


def build_processed_out_dir(cfg: dict, row: dict, modality: str) -> Path:
    """Build the canonical processed-data output directory for one row/modality."""
    layout = _processed_layout_pattern(cfg)
    rel_path = layout.format(
        date=row["date"],
        session=row["session"],
        modality=modality,
    )
    return Path(cfg["processed_data_root"]) / rel_path


def build_processed_data_path(
    cfg: dict,
    row: dict,
    modality: str,
    agent: Optional[str],
) -> Path:
    """Build the canonical processed pickle path for one row/modality/agent."""
    out_dir = build_processed_out_dir(cfg, row, modality)
    suffix = f"agent={agent}" if agent else "shared"
    return out_dir / f"{suffix}.pkl"


def build_processed_output_path(
    cfg: dict,
    row: dict,
    modality: str,
    agent: Optional[str],
    *,
    output_suffix: str,
) -> Path:
    """Build the canonical processed pickle path for a suffixed output modality."""
    output_modality = f"{modality}{output_suffix}" if output_suffix else modality
    return build_processed_data_path(cfg, row, output_modality, agent)


def _normalize_filter(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def scan_processed_data_paths(
    cfg: dict,
    modality: str,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> list[dict]:
    """Scan processed-data folders and return available paths for a modality."""
    root = Path(cfg["processed_data_root"])
    layout = _processed_layout_pattern(cfg)
    rel_pattern = layout.format(date="*", session="*", modality=modality)
    pattern = Path(rel_pattern) / "*.pkl"

    dates_filter = _normalize_filter(dates)
    sessions_filter = _normalize_filter(sessions)
    agents_filter = None
    if agents is not None:
        agents_filter = set()
        for agent in agents:
            if agent is None or str(agent).strip().lower() == "shared":
                agents_filter.add(None)
            else:
                agents_filter.add(str(agent).strip())

    rows: list[dict] = []
    for pkl_path in root.glob(str(pattern)):
        parts = pkl_path.parts
        date_part = next((part for part in parts if part.startswith("date=")), None)
        session_part = next((part for part in parts if part.startswith("session=")), None)
        if date_part is None or session_part is None:
            continue

        date = date_part.split("=", 1)[1]
        session = session_part.split("=", 1)[1]
        if dates_filter is not None and date not in dates_filter:
            continue
        if sessions_filter is not None and session not in sessions_filter:
            continue

        agent = None
        stem = pkl_path.stem
        if stem.startswith("agent="):
            agent = stem.split("=", 1)[1]
        elif stem != "shared":
            continue

        if agents_filter is not None and agent not in agents_filter:
            continue

        rows.append(
            {
                "date": date,
                "session": session,
                "agent": agent,
                "path": pkl_path,
            }
        )

    rows.sort(key=lambda row: (row["date"], row["session"], str(row.get("agent") or "")))
    return rows


def list_processed_modalities(cfg: dict) -> set[str]:
    """List modality directory names found under the processed-data tree."""
    root = Path(cfg["processed_data_root"])
    if not root.exists():
        return set()

    layout = _processed_layout_pattern(cfg)
    rel_pattern = layout.format(date="*", session="*", modality="*")
    modalities: set[str] = set()
    for modality_dir in root.glob(rel_pattern):
        if modality_dir.is_dir():
            modalities.add(modality_dir.name)
    return modalities


def build_processed_pickle_path(
    cfg: dict,
    row: dict,
    modality: str,
    agent: Optional[str],
) -> Path:
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


def build_processed_variant_pickle_path(
    cfg: dict,
    row: dict,
    modality: str,
    agent: Optional[str],
    *,
    output_suffix: str,
) -> Path:
    """Build the canonical processed pickle path for a suffixed output modality."""
    return build_processed_output_path(
        cfg,
        row,
        modality,
        agent,
        output_suffix=output_suffix,
    )


def save_processed_variant_pickle(
    obj: Any,
    cfg: dict,
    row: dict,
    modality: str,
    agent: Optional[str],
    *,
    output_suffix: str,
) -> None:
    """Save one processed artifact pickle under a suffixed output modality."""
    path = build_processed_variant_pickle_path(
        cfg,
        row,
        modality,
        agent,
        output_suffix=output_suffix,
    )
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
    root = Path(cfg["processed_data_root"])
    layout = _processed_layout_pattern(cfg)
    rel_pattern = layout.format(date="*", session="*", modality=modality)
    pattern = Path(rel_pattern) / target_name

    dates_filter = _normalize_filter(dates)
    sessions_filter = _normalize_filter(sessions)
    agents_filter = None
    if agents is not None:
        agents_filter = set()
        for agent in agents:
            if agent is None or str(agent).strip().lower() == "shared":
                agents_filter.add(None)
            else:
                agents_filter.add(str(agent).strip())

    rows: list[dict] = []
    for pkl_path in root.glob(str(pattern)):
        parts = pkl_path.parts
        date_part = next((part for part in parts if part.startswith("date=")), None)
        session_part = next((part for part in parts if part.startswith("session=")), None)
        if date_part is None or session_part is None:
            continue

        date = date_part.split("=", 1)[1]
        session = session_part.split("=", 1)[1]
        if dates_filter is not None and date not in dates_filter:
            continue
        if sessions_filter is not None and session not in sessions_filter:
            continue

        stem = pkl_path.stem
        if stem.startswith("agent="):
            agent = stem.split("=", 1)[1]
        else:
            agent = None

        if agents_filter is not None and agent not in agents_filter:
            continue

        rows.append(
            {
                "date": date,
                "session": session,
                "agent": agent,
                "path": pkl_path,
            }
        )

    rows.sort(key=lambda row: (row["date"], row["session"], str(row.get("agent") or "")))
    return rows


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
