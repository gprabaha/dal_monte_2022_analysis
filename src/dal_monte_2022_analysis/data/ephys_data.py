"""Backward-compatible shim for ephys record dataclasses."""

from dal_monte_2022_analysis.data.ephys_records import (
    EphysUnitContext,
    UnitSpikeData,
    WidebandChannelContext,
    WidebandChannelData,
)

__all__ = [
    "EphysUnitContext",
    "UnitSpikeData",
    "WidebandChannelContext",
    "WidebandChannelData",
]
