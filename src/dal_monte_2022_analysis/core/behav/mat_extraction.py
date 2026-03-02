"""Pure extraction helpers for behavioral MATLAB source structs."""

from __future__ import annotations

from typing import Optional

import numpy as np

from dal_monte_2022_analysis.data.records.behavioral import (
    NeuralTimelineData,
    PositionData,
    PupilSizeData,
    RecordingContext,
    ROIRectsData,
)


def unwrap_mat_struct(value):
    """Unwrap MATLAB 1x1 struct arrays into their scalar element."""
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


def extract_aligned_struct(mat_data: dict):
    """Return aligned MATLAB struct field if present, otherwise None."""
    for key in ("var", "aligned_position_file"):
        if key in mat_data:
            return mat_data[key]
    return None


def extract_position(mat_data, context: RecordingContext) -> Optional[PositionData]:
    """Extract aligned gaze position for one agent if present."""
    aligned = extract_aligned_struct(mat_data)
    if aligned is None:
        return None

    aligned = unwrap_mat_struct(aligned)
    if not hasattr(aligned, context.agent):
        return None

    data = getattr(aligned, context.agent)
    if data is None or data.size == 0:
        return None

    return PositionData(context=context, x=data[0, :], y=data[1, :])


def extract_neural_timeline(mat_data, context: RecordingContext) -> Optional[NeuralTimelineData]:
    """Extract shared timeline for a session if present."""
    for key in ("time_file", "aligned_position_file", "var"):
        if key not in mat_data:
            continue

        candidate = unwrap_mat_struct(mat_data[key])
        if hasattr(candidate, "t"):
            return NeuralTimelineData(context=context, t=candidate.t.flatten())
    return None


def extract_pupil(mat_data, context: RecordingContext) -> Optional[PupilSizeData]:
    """Extract aligned pupil size for one agent if present."""
    aligned = extract_aligned_struct(mat_data)
    if aligned is None:
        return None

    aligned = unwrap_mat_struct(aligned)
    if not hasattr(aligned, context.agent):
        return None

    data = getattr(aligned, context.agent)
    if data is None or data.size == 0:
        return None

    return PupilSizeData(context=context, d=data.flatten())


def extract_roi_rects(mat_data, context: RecordingContext) -> Optional[ROIRectsData]:
    """Extract per-agent ROI rectangles from MATLAB structure."""
    if "roi_rects" not in mat_data:
        return None

    roi_data = unwrap_mat_struct(mat_data["roi_rects"])
    if not hasattr(roi_data, context.agent):
        return None

    agent_data = unwrap_mat_struct(getattr(roi_data, context.agent))
    fieldnames = getattr(agent_data, "_fieldnames", None)
    if fieldnames is None:
        return None

    rois = {}
    for roi_name in fieldnames:
        if context.agent.lower() == "m2" and "object" in roi_name.lower():
            continue

        roi_val = getattr(agent_data, roi_name)
        roi_val = unwrap_mat_struct(roi_val).squeeze()
        if roi_val.size != 4:
            continue
        rois[roi_name] = roi_val

    if not rois:
        return None
    return ROIRectsData(context=context, rois=rois)


__all__ = [
    "extract_aligned_struct",
    "extract_neural_timeline",
    "extract_position",
    "extract_pupil",
    "extract_roi_rects",
    "unwrap_mat_struct",
]
