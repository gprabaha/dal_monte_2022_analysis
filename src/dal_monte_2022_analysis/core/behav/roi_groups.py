"""Behavioral ROI-group domain helpers."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

DEFAULT_FIXATION_ROI_GROUPS: dict[str, tuple[str, ...]] = {
    "face": ("face", "mouth", "eyes_nf"),
    "object": ("right_nonsocial_object", "left_nonsocial_object"),
    "out_of_roi": ("out_of_roi",),
}

DEFAULT_FIXATION_CATEGORY_ORDER: tuple[str, ...] = ("face", "object", "out_of_roi")
_FIXATION_CATEGORY_ALIASES: dict[str, str] = {
    "face": "face",
    "object": "object",
    "out_of_roi": "out_of_roi",
    "outofroi": "out_of_roi",
    "outside_roi": "out_of_roi",
}


def _is_na(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except Exception:
        return False


def canonical_fixation_category(value: object) -> Optional[str]:
    """Map a category label to a canonical fixation category token."""
    if _is_na(value):
        return None
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _FIXATION_CATEGORY_ALIASES.get(token)


def normalize_roi_groups(
    groups: Optional[dict[str, Sequence[str]]],
    *,
    include_defaults: bool = True,
    default_groups: Optional[dict[str, Sequence[str]]] = None,
) -> dict[str, list[str]]:
    """Normalize ROI-group mappings to lowercase string lists."""
    defaults = default_groups or DEFAULT_FIXATION_ROI_GROUPS
    out: dict[str, list[str]] = {}

    if groups:
        for group_name, labels in groups.items():
            if labels is None:
                continue
            if isinstance(labels, (str, bytes)):
                label_list = [labels]
            else:
                label_list = list(labels)
            out[str(group_name)] = [str(label).lower() for label in label_list]

    if include_defaults:
        for name, labels in defaults.items():
            out.setdefault(str(name), [str(label).lower() for label in labels])

    return out


def resolve_agent_roi_groups(
    *,
    agent: str,
    roi_groups: Optional[dict[str, Sequence[str]]],
    agent_roi_groups: Optional[dict[str, dict[str, Sequence[str]]]] = None,
    include_defaults: bool = True,
    default_groups: Optional[dict[str, Sequence[str]]] = None,
) -> dict[str, list[str]]:
    """Resolve ROI groups for one agent, honoring per-agent overrides."""
    if agent_roi_groups and agent in agent_roi_groups:
        return normalize_roi_groups(
            agent_roi_groups[agent],
            include_defaults=include_defaults,
            default_groups=default_groups,
        )
    return normalize_roi_groups(
        roi_groups,
        include_defaults=include_defaults,
        default_groups=default_groups,
    )


def coerce_location_labels(loc: object, *, lowercase: bool = False) -> list[str]:
    """Normalize a location field into a list of strings."""
    if _is_na(loc):
        return []
    if isinstance(loc, (list, tuple, set, np.ndarray)):
        values = [str(val) for val in loc if val is not None]
        return [val.lower() for val in values] if lowercase else values
    token = str(loc)
    return [token.lower() if lowercase else token]


def locations_match(locations: Iterable[str], keywords: Sequence[str]) -> bool:
    """Return True when any location label contains any keyword."""
    for loc in locations:
        loc_lower = str(loc).lower()
        for keyword in keywords:
            if str(keyword).lower() in loc_lower:
                return True
    return False


def categorize_locations(
    locations: Sequence[str],
    roi_groups: dict[str, list[str]],
    *,
    ordered_groups: Sequence[str] = DEFAULT_FIXATION_CATEGORY_ORDER,
    allowed_categories: Optional[set[str]] = None,
) -> Optional[str]:
    """Map fixation locations to the first matching ROI category."""
    labels = [str(loc).lower() for loc in locations]
    for group in ordered_groups:
        keywords = roi_groups.get(str(group), [])
        if not keywords:
            continue
        if locations_match(labels, keywords):
            if allowed_categories is not None and group not in allowed_categories:
                return None
            return str(group)
    return None


def keywords_for_fixation_label(
    fixation_label: str,
    *,
    roi_groups: Optional[dict[str, Sequence[str]]] = None,
) -> tuple[str, ...]:
    """Return ROI keywords for a fixation label from normalized ROI groups."""
    groups = normalize_roi_groups(
        roi_groups,
        include_defaults=True,
        default_groups=DEFAULT_FIXATION_ROI_GROUPS,
    )
    category = canonical_fixation_category(fixation_label)
    if category is None:
        return tuple()
    return tuple(groups.get(category, []))
