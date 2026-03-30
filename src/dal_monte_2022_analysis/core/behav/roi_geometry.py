"""Shared ROI rectangle geometry helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from dal_monte_2022_analysis.data.records.behavioral import ROIRectsData

DEFAULT_ROI_ASSIGNMENT_EXPANSION_FRACTION = 0.2


def coerce_roi_expansion_fraction(
    value: object | None,
    *,
    default: float = DEFAULT_ROI_ASSIGNMENT_EXPANSION_FRACTION,
) -> float:
    """Coerce and validate an ROI rectangle expansion fraction."""
    if value is None:
        value = default
    fraction = float(value)
    if fraction < 0:
        raise ValueError(f"ROI expansion fraction must be non-negative, got {fraction}.")
    return fraction


def normalize_roi_rect_bounds(rect: object) -> tuple[float, float, float, float] | None:
    """Normalize a raw ROI rect into sorted x/y min/max bounds."""
    coords = np.asarray(rect, dtype=float).reshape(-1)
    if coords.size < 4 or not np.all(np.isfinite(coords[:4])):
        return None
    x1, y1, x2, y2 = coords[:4]
    return (
        float(min(x1, x2)),
        float(max(x1, x2)),
        float(min(y1, y2)),
        float(max(y1, y2)),
    )


def expand_roi_rect_bounds(
    bounds: tuple[float, float, float, float],
    *,
    expansion_fraction: float = 0.0,
) -> tuple[float, float, float, float]:
    """Expand normalized ROI bounds around their center."""
    fraction = coerce_roi_expansion_fraction(expansion_fraction, default=0.0)
    x_min, x_max, y_min, y_max = bounds
    x_pad = (x_max - x_min) * fraction / 2.0
    y_pad = (y_max - y_min) * fraction / 2.0
    return (
        x_min - x_pad,
        x_max + x_pad,
        y_min - y_pad,
        y_max + y_pad,
    )


def iter_roi_rect_bounds(
    rois: ROIRectsData,
    *,
    expansion_fraction: float = 0.0,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return named ROI bounds with optional width/height expansion."""
    items: list[tuple[str, tuple[float, float, float, float]]] = []
    for name, rect in rois.rois.items():
        bounds = normalize_roi_rect_bounds(rect)
        if bounds is None:
            continue
        items.append(
            (
                str(name),
                expand_roi_rect_bounds(bounds, expansion_fraction=expansion_fraction),
            )
        )
    return items


__all__ = [
    "DEFAULT_ROI_ASSIGNMENT_EXPANSION_FRACTION",
    "coerce_roi_expansion_fraction",
    "expand_roi_rect_bounds",
    "iter_roi_rect_bounds",
    "normalize_roi_rect_bounds",
]
