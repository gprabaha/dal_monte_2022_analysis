"""Data-table and session-stream transform utilities."""

from .annotate import (
    annotate_ephys_dates_with_pair_context,
    annotate_with_pair_context,
    load_pair_context_table,
    load_pair_context_table_from_cfg,
)

_CLEANING_EXPORTS = {
    "prune_timeline",
    "interpolate_nans",
    "interpolate_position",
    "interpolate_pupil",
    "prune_and_interpolate_session",
}

__all__ = [
    "load_pair_context_table",
    "load_pair_context_table_from_cfg",
    "annotate_with_pair_context",
    "annotate_ephys_dates_with_pair_context",
    "prune_timeline",
    "interpolate_nans",
    "interpolate_position",
    "interpolate_pupil",
    "prune_and_interpolate_session",
]


def __getattr__(name: str):
    if name in _CLEANING_EXPORTS:
        from . import cleaning

        return getattr(cleaning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
