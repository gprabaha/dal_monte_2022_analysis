"""Backward-compatible shim for behavioral record dataclasses."""

from dal_monte_2022_analysis.data.behavioral_records import (
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

__all__ = [
    "BehaviorRunContext",
    "RecordingContext",
    "PositionData",
    "PupilSizeData",
    "NeuralTimelineData",
    "ROIRectsData",
    "FixationBinaryVectorsData",
    "FixationDensityVectorsData",
    "JointFixationDensityData",
]
