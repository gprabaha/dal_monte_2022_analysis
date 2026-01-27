import numpy as np
import pdb
from typing import Optional
from src.data.gaze_data import (
    RecordingContext,
    PositionData,
    PupilSizeData,
    NeuralTimelineData,
    ROIRectsData,
)


def _unwrap_mat_struct(x):
    """If MATLAB exported a 1x1 struct array, unwrap it."""
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.item()
    return x

def extract_aligned_struct(mat_data: dict):
    for key in ["var", "aligned_position_file"]:
        if key in mat_data:
            return mat_data[key]
    return None


def extract_position(mat_data, context: RecordingContext) -> Optional[PositionData]:
    aligned = extract_aligned_struct(mat_data)
    if aligned is None:
        return None

    aligned = _unwrap_mat_struct(aligned)
    
    if not hasattr(aligned, context.agent):
        return None
    
    data = getattr(aligned, context.agent)
    if data is None or data.size == 0:
        return None
    
    return PositionData(
        context=context,
        x=data[0, :],
        y=data[1, :],
    )



def extract_pupil(mat_data, context: RecordingContext) -> Optional[PupilSizeData]:
    aligned = extract_aligned_struct(mat_data)
    if aligned is None:
        return None

    aligned = _unwrap_mat_struct(aligned)

    if not hasattr(aligned, context.agent):
        return None

    data = getattr(aligned, context.agent)
    if data is None or data.size == 0:
        return None

    return PupilSizeData(
        context=context,
        d=data.flatten(),
    )



def extract_neural_timeline(mat_data, context: RecordingContext) -> Optional[NeuralTimelineData]:
    for key in ["time_file", "aligned_position_file", "var"]:
        if key not in mat_data:
            continue

        candidate = _unwrap_mat_struct(mat_data[key])

        if hasattr(candidate, "t"):
            return NeuralTimelineData(
                context=context,
                t=candidate.t.flatten(),
            )

    return None



def extract_roi_rects(mat_data, context: RecordingContext) -> Optional[ROIRectsData]:
    if "roi_rects" not in mat_data:
        return None

    roi_data = _unwrap_mat_struct(mat_data["roi_rects"])

    if not hasattr(roi_data, context.agent):
        return None

    agent_data = _unwrap_mat_struct(getattr(roi_data, context.agent))

    rois = {}

    for roi_name, roi_val in agent_data.__dict__.items():
        if context.agent == "M2" and "object" in roi_name.lower():
            continue

        roi_val = _unwrap_mat_struct(roi_val)

        # MATLAB nesting: often [[[x1 x2 y1 y2]]]
        try:
            rois[roi_name] = roi_val[0][0][0]
        except Exception:
            rois[roi_name] = roi_val

    return ROIRectsData(
        context=context,
        rois=rois,
    )

