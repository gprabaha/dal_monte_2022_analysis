"""Compatibility layer for behavioral MAT extraction helpers.

Canonical implementations live in:
`dal_monte_2022_analysis.core.behav.mat_extraction`.
"""

from dal_monte_2022_analysis.core.behav.mat_extraction import (  # noqa: F401
    extract_aligned_struct,
    extract_neural_timeline,
    extract_position,
    extract_pupil,
    extract_roi_rects,
    unwrap_mat_struct,
)

__all__ = [
    "extract_aligned_struct",
    "extract_neural_timeline",
    "extract_position",
    "extract_pupil",
    "extract_roi_rects",
    "unwrap_mat_struct",
]
