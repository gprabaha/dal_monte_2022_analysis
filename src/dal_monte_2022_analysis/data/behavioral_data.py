"""Compatibility shim for behavioral record dataclasses.

Canonical import path:
`dal_monte_2022_analysis.data.records.behavioral`.
"""

from dal_monte_2022_analysis.data.records.behavioral import (  # noqa: F401
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
