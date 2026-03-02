"""Backward-compatibility shim for legacy pickles referencing spike_data.

Older ephys pickles may store class references under
`dal_monte_2022_analysis.data.spike_data`. The canonical module is now
`dal_monte_2022_analysis.data.ephys_records`.
"""

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
