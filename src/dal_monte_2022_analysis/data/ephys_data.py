"""Compatibility shim for ephys record dataclasses.

Canonical import path:
`dal_monte_2022_analysis.data.records.ephys`.
"""

from dal_monte_2022_analysis.data.records.ephys import (  # noqa: F401
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
