"""Dataclasses that define the in-memory representation of ephys data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class EphysUnitContext:
    """Immutable metadata identifying one recorded neural unit.

    Notes:
    - `session_name` is preserved from source files and often encodes date.
    - `date` is the normalized recording day token (MMDDYYYY).
    - `session` is kept as an optional compatibility field; current ephys unit
      recordings are expected to be date-level with no run/session split.
    """

    date: str
    session_name: str
    unit_uuid: str
    session: Optional[str] = None
    region: Optional[str] = None
    spike_channel: Optional[str] = None
    recorded_agent: str = "m1"
    recorded_monkey: Optional[str] = None
    area: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def day(self) -> str:
        """Canonical alias for date."""
        return self.date

    @property
    def run(self) -> Optional[str]:
        """Canonical alias for session."""
        return self.session

    @property
    def channel(self) -> Optional[str]:
        """Backward-compatible alias for spike_channel."""
        return self.spike_channel


@dataclass
class UnitSpikeData:
    """Spike timestamps for a single neural unit."""

    context: EphysUnitContext
    spike_ts: np.ndarray

    @property
    def spike_times(self) -> np.ndarray:
        """Canonical alias for spike_ts."""
        return self.spike_ts


@dataclass(frozen=True)
class WidebandChannelContext:
    """Metadata for one wideband channel recording (future modality)."""

    date: str
    channel: str
    region: Optional[str] = None
    monkey: Optional[str] = None
    session_name: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class WidebandChannelData:
    """Wideband samples for one channel (future modality)."""

    context: WidebandChannelContext
    samples: np.ndarray
    sample_rate_hz: Optional[float] = None
