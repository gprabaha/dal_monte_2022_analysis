"""Ephys-domain core logic."""

from .analysis_primitives import (
    as_bool,
    as_optional_str,
    ensure_filename,
    extract_trials_df_and_meta,
    resolve_bin_centers_from_meta,
)

__all__ = [
    "as_bool",
    "as_optional_str",
    "ensure_filename",
    "extract_trials_df_and_meta",
    "resolve_bin_centers_from_meta",
]
