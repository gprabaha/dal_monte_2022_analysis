"""Compatibility shim for fixation detection helpers.

Canonical import path:
`dal_monte_2022_analysis.core.behav.fixation_detection`.
"""

from dal_monte_2022_analysis.core.behav.fixation_detection import (  # noqa: F401
    detect_fixations_and_saccades,
)

__all__ = ["detect_fixations_and_saccades"]
