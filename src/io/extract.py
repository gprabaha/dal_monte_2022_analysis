import numpy as np
from typing import Optional
from src.data.gaze_data import (
    RecordingContext,
    PositionData,
    PupilSizeData,
    NeuralTimelineData,
    ROIRectsData,
)


def extract_aligned_struct(mat_data: dict):
    for key in ["var", "aligned_position_file"]:
        if key in mat_data:
            return mat_data[key]
    return None


def extract_position(mat_data, context: RecordingContext) -> Optional[PositionData]:
    aligned = extract_aligned_struct(mat_data)
    if aligned is None or context.agent not in aligned.dtype.names:
        return None

    data = aligned[context.agent]
    if data is None or data.size == 0:
        return None

    return PositionData(
        context=context,
        x=data[0, :],
        y=data[1, :],
    )


def extract_pupil(mat_data, context: RecordingContext) -> Optional[PupilSizeData]:
    aligned = extract_aligned_struct(mat_data)
    if aligned is None or context.agent not in aligned.dtype.names:
        return None

    data = aligned[context.agent]
    if data is None or data.size == 0:
        return None

    return PupilSizeData(context=context, d=data.flatten())


def extract_neural_timeline(mat_data, context: RecordingContext) -> Optional[NeuralTimelineData]:
    for key in ["time_file", "aligned_position_file", "var"]:
        if key in mat_data and hasattr(mat_data[key], "t"):
            return NeuralTimelineData(
                context=context,
                t=mat_data[key].t.flatten(),
            )
    return None


def extract_roi_rects(mat_data, context: RecordingContext) -> Optional[ROIRectsData]:
    if "roi_rects" not in mat_data:
        return None

    roi_data = mat_data["roi_rects"]
    if context.agent not in roi_data.dtype.names:
        return None

    agent_data = roi_data[context.agent]
    rois = {}

    for roi_name in agent_data.dtype.names:
        if context.agent == "M2" and "object" in roi_name.lower():
            continue
        rois[roi_name] = agent_data[roi_name][0][0][0]

    return ROIRectsData(context=context, rois=rois)
