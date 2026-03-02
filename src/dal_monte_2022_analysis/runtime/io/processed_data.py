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
