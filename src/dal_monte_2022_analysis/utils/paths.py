"""Compatibility wrappers for centralized path and filename helpers."""

from __future__ import annotations

from typing import Optional, Sequence

from dal_monte_2022_analysis.core.behav.analysis_filenames import (
    build_fix_cross_correlation_output_filename,
    normalize_fix_cross_correlation_time_scope,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import (
    build_analysis_output_dir,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    build_processed_data_path as _build_processed_data_path,
    build_processed_out_dir as _build_processed_out_dir,
    build_processed_output_path as _build_processed_output_path,
    list_processed_modalities,
    scan_processed_data_paths as _scan_processed_data_paths,
)


def build_processed_out_dir_wrapper(cfg, index_row, modality):
    """Deprecated compatibility alias for processed output directory builders."""
    return _build_processed_out_dir(cfg, index_row, modality)


def build_processed_data_path_wrapper(cfg, index_row, modality, agent):
    """Deprecated compatibility alias for processed pickle path builders."""
    return _build_processed_data_path(cfg, index_row, modality, agent)


def build_processed_output_path_wrapper(cfg, index_row, modality, agent, *, output_suffix):
    """Deprecated compatibility alias for processed output path builders."""
    return _build_processed_output_path(
        cfg,
        index_row,
        modality,
        agent,
        output_suffix=output_suffix,
    )


def scan_processed_data_paths_wrapper(
    cfg: dict,
    modality: str,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> list[dict]:
    """Deprecated compatibility alias for processed path scanning."""
    return _scan_processed_data_paths(
        cfg,
        modality,
        dates=dates,
        sessions=sessions,
        agents=agents,
    )


# Preserve the historical public names so old imports keep working.
build_processed_out_dir = build_processed_out_dir_wrapper
build_processed_data_path = build_processed_data_path_wrapper
build_processed_output_path = build_processed_output_path_wrapper
scan_processed_data_paths = scan_processed_data_paths_wrapper
