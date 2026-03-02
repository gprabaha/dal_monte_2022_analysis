"""Dataclasses that define behavioral data records."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class BehaviorRunContext:
    """Immutable metadata identifying one behavioral run/agent stream."""

    date: str
    session: str
    agent: Optional[str] = None
    monkey_name: Optional[str] = None

    @property
    def day(self) -> str:
        return self.date

    @property
    def run(self) -> str:
        return self.session


RecordingContext = BehaviorRunContext


@dataclass
class PositionData:
    context: BehaviorRunContext
    x: np.ndarray
    y: np.ndarray


@dataclass
class PupilSizeData:
    context: BehaviorRunContext
    d: np.ndarray


@dataclass
class NeuralTimelineData:
    context: BehaviorRunContext
    t: np.ndarray


@dataclass
class ROIRectsData:
    context: BehaviorRunContext
    rois: Dict[str, np.ndarray]


@dataclass
class FixationBinaryVectorsData:
    context: BehaviorRunContext
    vectors: Dict[str, np.ndarray]


@dataclass
class FixationDensityVectorsData:
    context: BehaviorRunContext
    vectors: Dict[str, np.ndarray]


@dataclass
class JointFixationDensityData:
    context: BehaviorRunContext
    density: np.ndarray
