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
from .roi_geometry import (
    DEFAULT_ROI_ASSIGNMENT_EXPANSION_FRACTION,
    coerce_roi_expansion_fraction,
    expand_roi_rect_bounds,
    iter_roi_rect_bounds,
    normalize_roi_rect_bounds,
)

__all__ = [
    "DEFAULT_FIXATION_CATEGORY_ORDER",
    "DEFAULT_FIXATION_ROI_GROUPS",
    "DEFAULT_ROI_ASSIGNMENT_EXPANSION_FRACTION",
    "canonical_fixation_category",
    "FixationDetectionConfig",
    "coerce_fixation_detection_config",
    "coerce_roi_expansion_fraction",
    "categorize_locations",
    "coerce_location_labels",
    "detect_fixations_and_saccades",
    "expand_roi_rect_bounds",
    "iter_roi_rect_bounds",
    "keywords_for_fixation_label",
    "locations_match",
    "normalize_roi_groups",
    "normalize_roi_rect_bounds",
    "resolve_agent_roi_groups",
]
