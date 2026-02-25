"""Dataclasses that define the in-memory representation of extracted data."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass(frozen=True)
class BehaviorRunContext:
    """
    Immutable metadata identifying one behavioral run/agent stream.

    Naming:
    - `date` is the recording day token (MMDDYYYY).
    - `session` is the within-day run identifier.
    - `agent` is behavioral stream identity ("m1"/"m2"), if applicable.
    """
    date: str
    session: str
    agent: Optional[str] = None  # "m1", "m2", or None if agent-less
    monkey_name: Optional[str] = None

    @property
    def day(self) -> str:
        """Canonical alias for date."""
        return self.date

    @property
    def run(self) -> str:
        """Canonical alias for session."""
        return self.session


# Backward-compatible alias used across existing modules.
RecordingContext = BehaviorRunContext


@dataclass
class PositionData:
    """
    Gaze position time series for a single agent in a single session.
    """
    context: BehaviorRunContext
    x: np.ndarray
    y: np.ndarray


@dataclass
class PupilSizeData:
    """Pupil diameter time series for a single agent in a single session."""
    context: BehaviorRunContext
    d: np.ndarray


@dataclass
class NeuralTimelineData:
    """Neural timeline timestamps for a session (shared across agents)."""
    context: BehaviorRunContext
    t: np.ndarray


@dataclass
class ROIRectsData:
    """Per-agent ROI rectangles keyed by ROI name."""
    context: BehaviorRunContext
    rois: Dict[str, np.ndarray]  # roi_name -> [x1, y1, x2, y2]


@dataclass
class FixationBinaryVectorsData:
    """Binary fixation vectors aligned to a neural timeline."""
    context: BehaviorRunContext
    vectors: Dict[str, np.ndarray]  # label -> 1D binary vector


@dataclass
class FixationDensityVectorsData:
    """Smoothed fixation density vectors aligned to a neural timeline."""
    context: BehaviorRunContext
    vectors: Dict[str, np.ndarray]  # label -> 1D density vector


@dataclass
class JointFixationDensityData:
    """Joint fixation density vector (shared across agents)."""
    context: BehaviorRunContext
    density: np.ndarray
