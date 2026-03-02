"""Compatibility shim for ROI-group helpers.

Canonical import path:
`dal_monte_2022_analysis.core.behav.roi_groups`.
"""

from dal_monte_2022_analysis.core.behav.roi_groups import (  # noqa: F401
    DEFAULT_FIXATION_CATEGORY_ORDER,
    DEFAULT_FIXATION_ROI_GROUPS,
    canonical_fixation_category,
    categorize_locations,
    coerce_location_labels,
    keywords_for_fixation_label,
    locations_match,
    normalize_roi_groups,
    resolve_agent_roi_groups,
)

__all__ = [
    "DEFAULT_FIXATION_CATEGORY_ORDER",
    "DEFAULT_FIXATION_ROI_GROUPS",
    "canonical_fixation_category",
    "categorize_locations",
    "coerce_location_labels",
    "keywords_for_fixation_label",
    "locations_match",
    "normalize_roi_groups",
    "resolve_agent_roi_groups",
]
