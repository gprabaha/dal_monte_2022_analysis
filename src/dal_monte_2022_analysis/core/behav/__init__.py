"""Behavioral-domain core logic."""

from .fixation_detection import (
    FixationDetectionConfig,
    coerce_fixation_detection_config,
    detect_fixations_and_saccades,
)
from .roi_groups import (
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
    "FixationDetectionConfig",
    "coerce_fixation_detection_config",
    "categorize_locations",
    "coerce_location_labels",
    "detect_fixations_and_saccades",
    "keywords_for_fixation_label",
    "locations_match",
    "normalize_roi_groups",
    "resolve_agent_roi_groups",
]
