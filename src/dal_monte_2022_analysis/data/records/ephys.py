"""Dataclasses that define ephys data records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class EphysUnitContext:
    """Immutable metadata identifying one recorded neural unit.

    Notes:
    - Ephys data is date-level (no within-date behavioral session concept).
    - `session_name` is a legacy source-column token retained for compatibility.
    - `legacy_session_token` stores any suffix parsed from legacy `session_name`
      formats like `MMDDYYYY_<token>`.
    """

    date: str
    session_name: str
    unit_uuid: str
    legacy_session_token: Optional[str] = None
    region: Optional[str] = None
    spike_channel: Optional[str] = None
    recorded_agent: str = "m1"
    recorded_monkey: Optional[str] = None
    area: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def day(self) -> str:
        return self.date

    @property
    def run(self) -> None:
        """Behavior-style alias; always None for ephys date-level data."""
        return None

    @property
    def session(self) -> None:
        """Compatibility alias; always None for ephys date-level data."""
        return None

    @property
    def channel(self) -> Optional[str]:
        return self.spike_channel


@dataclass
class UnitSpikeData:
    context: EphysUnitContext
    spike_ts: np.ndarray

    @property
    def spike_times(self) -> np.ndarray:
        return self.spike_ts


@dataclass(frozen=True)
class WidebandChannelContext:
    date: str
    channel: str
    region: Optional[str] = None
    monkey: Optional[str] = None
    session_name: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class WidebandChannelData:
    context: WidebandChannelContext
    samples: np.ndarray
    sample_rate_hz: Optional[float] = None
