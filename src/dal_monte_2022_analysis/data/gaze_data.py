
"""Backward-compatibility shim for legacy pickles referencing gaze_data.

Older processed pickles may store class references under
`dal_monte_2022_analysis.data.gaze_data`. The canonical module is now
`dal_monte_2022_analysis.data.behavioral_data`.
"""

from dal_monte_2022_analysis.data.behavioral_data import (
    BehaviorRunContext,
    FixationBinaryVectorsData,
    FixationDensityVectorsData,
    JointFixationDensityData,
    NeuralTimelineData,
    PositionData,
    PupilSizeData,
    RecordingContext,
    ROIRectsData,
)

# Legacy alias names that may appear in historic pickles.
ROIsData = ROIRectsData
RoiRectsData = ROIRectsData
RoiData = ROIRectsData

__all__ = [
    "BehaviorRunContext",
    "RecordingContext",
    "PositionData",
    "PupilSizeData",
    "NeuralTimelineData",
    "ROIRectsData",
    "ROIsData",
    "RoiRectsData",
    "RoiData",
    "FixationBinaryVectorsData",
    "FixationDensityVectorsData",
    "JointFixationDensityData",
]
