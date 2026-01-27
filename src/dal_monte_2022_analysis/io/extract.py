import numpy as np
import pdb
from typing import Optional
from dal_monte_2022_analysis.data.gaze_data import (
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



def extract_roi_rects(mat_data, context: RecordingContext) -> Optional[ROIRectsData]:
    if "roi_rects" not in mat_data:
        return None

    roi_data = _unwrap_mat_struct(mat_data["roi_rects"])

    if not hasattr(roi_data, context.agent):
        return None

    agent_data = _unwrap_mat_struct(getattr(roi_data, context.agent))

    # MATLAB struct metadata, not an ROI
    fieldnames = getattr(agent_data, "_fieldnames", None)
    if fieldnames is None:
        return None

    rois = {}

    for roi_name in fieldnames:
        if context.agent.lower() == "m2" and "object" in roi_name.lower():
            continue

        roi_val = getattr(agent_data, roi_name)
        roi_val = _unwrap_mat_struct(roi_val)

        # Expected format: [x1, y1, x2, y2]
        roi_val = roi_val.squeeze()

        if roi_val.size != 4:
            continue
        
        rois[roi_name] = roi_val

    if not rois:
        return None

    return ROIRectsData(
        context=context,
        rois=rois,
    )
