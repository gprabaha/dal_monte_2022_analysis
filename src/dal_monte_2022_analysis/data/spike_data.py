"""Dataclasses that define the in-memory representation of neural spike data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class NeuralUnitContext:
    """Immutable metadata identifying one recorded neural unit.

    Naming:
    - `date` is the recording day token (MMDDYYYY), parsed from source
      `session_name` values.
    - `session` is an optional within-day run token when source names include
      one (e.g., MMDDYYYY_<run>).
    """

    date: str
    session_name: str
    unit_uuid: str
    session: Optional[str] = None
    recorded_agent: str = "m1"
    recorded_monkey: Optional[str] = None
    region: Optional[str] = None
    area: Optional[str] = None
    channel: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def day(self) -> str:
        """Canonical alias for date."""
        return self.date

    @property
    def run(self) -> Optional[str]:
        """Canonical alias for session."""
        return self.session


@dataclass
class SpikeTrainData:
    """Spike timestamps for a single unit in the global neural timeline."""

    context: NeuralUnitContext
    spike_ts: np.ndarray


# Backward-compatible alias.
SpikeUnitContext = NeuralUnitContext
