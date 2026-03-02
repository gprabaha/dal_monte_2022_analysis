"""Shared IO helpers."""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from typing import Any


_LEGACY_PICKLE_CLASS_MAP: dict[tuple[str, str], tuple[str, str]] = {
    # Behavioral legacy module remaps.
    ("dal_monte_2022_analysis.data.gaze_data", "BehaviorRunContext"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "BehaviorRunContext",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "RecordingContext"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "RecordingContext",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "PositionData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "PositionData",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "PupilSizeData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "PupilSizeData",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "NeuralTimelineData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "NeuralTimelineData",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "ROIRectsData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "ROIRectsData",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "ROIsData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "ROIRectsData",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "RoiRectsData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "ROIRectsData",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "RoiData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "ROIRectsData",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "FixationBinaryVectorsData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "FixationBinaryVectorsData",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "FixationDensityVectorsData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "FixationDensityVectorsData",
    ),
    ("dal_monte_2022_analysis.data.gaze_data", "JointFixationDensityData"): (
        "dal_monte_2022_analysis.data.records.behavioral",
        "JointFixationDensityData",
    ),
    # Ephys legacy module remaps.
    ("dal_monte_2022_analysis.data.spike_data", "EphysUnitContext"): (
        "dal_monte_2022_analysis.data.records.ephys",
        "EphysUnitContext",
    ),
    ("dal_monte_2022_analysis.data.spike_data", "UnitSpikeData"): (
        "dal_monte_2022_analysis.data.records.ephys",
        "UnitSpikeData",
    ),
    ("dal_monte_2022_analysis.data.spike_data", "WidebandChannelContext"): (
        "dal_monte_2022_analysis.data.records.ephys",
        "WidebandChannelContext",
    ),
    ("dal_monte_2022_analysis.data.spike_data", "WidebandChannelData"): (
        "dal_monte_2022_analysis.data.records.ephys",
        "WidebandChannelData",
    ),
}


class _LegacyAwareUnpickler(pickle.Unpickler):
    """Unpickler that remaps legacy module/class paths to canonical locations."""

    def find_class(self, module: str, name: str):
        target = _LEGACY_PICKLE_CLASS_MAP.get((module, name))
        if target is not None:
            target_module, target_name = target
            mod = importlib.import_module(target_module)
            return getattr(mod, target_name)

        if module == "dal_monte_2022_analysis.data.gaze_data":
            mod = importlib.import_module("dal_monte_2022_analysis.data.records.behavioral")
            return getattr(mod, name)
        if module == "dal_monte_2022_analysis.data.spike_data":
            mod = importlib.import_module("dal_monte_2022_analysis.data.records.ephys")
            return getattr(mod, name)
        return super().find_class(module, name)


def load_pickle(path: str | Path) -> Any:
    """Load a pickled object from disk."""
    with Path(path).open("rb") as f:
        return _LegacyAwareUnpickler(f).load()


def save_pickle(obj: Any, path: str | Path) -> None:
    """Write an object to a pickle file, creating parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(obj, f)
