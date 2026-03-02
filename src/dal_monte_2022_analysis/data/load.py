"""Compatibility shim for data loaders.

Canonical import path:
`dal_monte_2022_analysis.data.loaders`.
"""

from dal_monte_2022_analysis.data.loaders.behavioral import (  # noqa: F401
    BehavioralDataItem,
    group_behavioral_items,
    index_behavioral_data,
    load_behavioral_data_dataframe,
    load_behavioral_data_modality,
    load_behavioral_data_objects,
)
from dal_monte_2022_analysis.data.loaders.ephys import (  # noqa: F401
    load_ephys_unit_dataframe,
    load_ephys_units,
)

__all__ = [
    "BehavioralDataItem",
    "index_behavioral_data",
    "load_behavioral_data_objects",
    "load_behavioral_data_dataframe",
    "load_behavioral_data_modality",
    "group_behavioral_items",
    "load_ephys_unit_dataframe",
    "load_ephys_units",
]
