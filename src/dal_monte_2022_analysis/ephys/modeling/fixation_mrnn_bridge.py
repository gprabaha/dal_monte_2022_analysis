"""Bridge combined fixation PSTH exports into legacy mRNN training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.io.analysis_index import (
    build_analysis_output_dir,
)

DEFAULT_INPUT_SUBDIR = "ephys/psth/fixation_psth_averages"
DEFAULT_DATAFRAME_FILENAME = (
    "fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl"
)
DEFAULT_TIMELINE_FILENAME = (
    "fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl"
)
MRNN_CONDITION_COLUMN_ORDER = (
    "high_interactivity_face",
    "low_interactivity_face",
    "object",
)
MRNN_CONDITION_LABELS = {
    "high_interactivity_face": "interactive face",
    "low_interactivity_face": "noninteractive face",
    "object": "object",
}
MRNN_REGION_ORDER = ("ofc", "bla", "dmpfc", "accg")
ANALYSIS_TO_MRNN_REGION = {
    "ofc": "ofc",
    "bla": "bla",
    "dmpfc": "dmpfc",
    "accg": "accg",
}

_UNIT_KEY_COLUMNS = (
    "date",
    "unit_uuid",
    "region",
    "spike_channel",
    "recorded_agent",
)
_CONDITION_SPECS = {
    "high_interactivity_face": {
        "average_partition": "split",
        "fixation_category": "face",
        "interactive_state": "interactive",
    },
    "low_interactivity_face": {
        "average_partition": "split",
        "fixation_category": "face",
        "interactive_state": "non_interactive",
    },
    "object": {
        "average_partition": "unsplit",
        "fixation_category": "object",
        "interactive_state": None,
    },
}


@dataclass(frozen=True)
class CombinedFixationPSTHLoadResult:
    """Loaded combined fixation PSTH dataframe and its shared timeline."""

    dataframe: pd.DataFrame
    timeline_s_rel: np.ndarray
    dataframe_path: Path
    timeline_path: Path


def resolve_combined_fixation_psth_paths(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    input_subdir: str = DEFAULT_INPUT_SUBDIR,
    dataframe_filename: str = DEFAULT_DATAFRAME_FILENAME,
    timeline_filename: str = DEFAULT_TIMELINE_FILENAME,
) -> tuple[Path, Path]:
    """Resolve the saved combined fixation PSTH dataframe and timeline paths."""
    cfg = load_config(cfg_path)
    root = build_analysis_output_dir(cfg, input_subdir)
    return (
        root / Path(str(dataframe_filename)).name,
        root / Path(str(timeline_filename)).name,
    )


def load_combined_fixation_psth(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    input_subdir: str = DEFAULT_INPUT_SUBDIR,
    dataframe_filename: str = DEFAULT_DATAFRAME_FILENAME,
    timeline_filename: str = DEFAULT_TIMELINE_FILENAME,
) -> CombinedFixationPSTHLoadResult:
    """Load the combined fixation PSTH dataframe and relative timeline array."""
    dataframe_path, timeline_path = resolve_combined_fixation_psth_paths(
        cfg_path,
        input_subdir=input_subdir,
        dataframe_filename=dataframe_filename,
        timeline_filename=timeline_filename,
    )
    if not dataframe_path.exists():
        raise FileNotFoundError(
            f"Combined fixation PSTH dataframe not found: {dataframe_path}"
        )
    if not timeline_path.exists():
        raise FileNotFoundError(
            f"Combined fixation PSTH timeline not found: {timeline_path}"
        )

    dataframe = pd.read_pickle(dataframe_path)
    with timeline_path.open("rb") as f:
        timeline_s_rel = np.asarray(pickle.load(f), dtype=float).reshape(-1)

    return CombinedFixationPSTHLoadResult(
        dataframe=dataframe,
        timeline_s_rel=timeline_s_rel,
        dataframe_path=dataframe_path,
        timeline_path=timeline_path,
    )


def _validate_required_columns(dataframe: pd.DataFrame) -> None:
    required = set(_UNIT_KEY_COLUMNS) | {
        "average_partition",
        "fixation_category",
        "interactive_state",
        "n_trials",
        "psth_mean",
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise KeyError(
            "Combined fixation PSTH dataframe is missing required columns: "
            + ", ".join(missing)
        )


def _subset_condition_frame(
    dataframe: pd.DataFrame,
    *,
    condition_name: str,
) -> pd.DataFrame:
    spec = _CONDITION_SPECS[condition_name]
    mask = (
        dataframe["average_partition"].astype(str) == spec["average_partition"]
    ) & (dataframe["fixation_category"].astype(str) == spec["fixation_category"])
    if spec["interactive_state"] is None:
        mask &= dataframe["interactive_state"].isna()
    else:
        mask &= dataframe["interactive_state"].astype(str) == spec["interactive_state"]

    columns = list(_UNIT_KEY_COLUMNS) + ["n_trials", "psth_mean"]
    frame = dataframe.loc[mask, columns].copy()
    if frame.empty:
        raise ValueError(
            f"No rows were found for modeled condition '{condition_name}'."
        )
    if frame.duplicated(list(_UNIT_KEY_COLUMNS)).any():
        raise ValueError(
            f"Found duplicate unit rows while extracting '{condition_name}'."
        )

    return frame.rename(
        columns={
            "unit_uuid": "uuid",
            "region": "source_region",
            "n_trials": f"{condition_name}_n_trials",
            "psth_mean": condition_name,
        }
    )


def _validate_trace_lengths(
    dataframe: pd.DataFrame,
    *,
    condition_columns: tuple[str, ...],
) -> None:
    expected_lengths: dict[str, int] = {}
    for condition_name in condition_columns:
        lengths = {
            int(np.asarray(values, dtype=float).reshape(-1).shape[0])
            for values in dataframe[condition_name].tolist()
        }
        if len(lengths) != 1:
            raise ValueError(
                f"Condition '{condition_name}' has inconsistent PSTH lengths: "
                f"{sorted(lengths)}"
            )
        expected_lengths[condition_name] = next(iter(lengths))

    if len(set(expected_lengths.values())) != 1:
        raise ValueError(
            "Modeled conditions do not share the same PSTH length: "
            f"{expected_lengths}"
        )


def build_mrnn_training_dataframe(
    combined_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Reshape combined PSTHs into one-row-per-unit legacy mRNN training format."""
    _validate_required_columns(combined_dataframe)

    condition_frames = [
        _subset_condition_frame(combined_dataframe, condition_name=condition_name)
        for condition_name in MRNN_CONDITION_COLUMN_ORDER
    ]

    merge_keys = ["date", "uuid", "source_region", "spike_channel", "recorded_agent"]
    training_df = condition_frames[0]
    for frame in condition_frames[1:]:
        training_df = training_df.merge(
            frame,
            on=merge_keys,
            how="inner",
            validate="one_to_one",
        )

    if training_df.empty:
        raise ValueError("No units remained after merging the modeled conditions.")

    training_df["region"] = training_df["source_region"].map(ANALYSIS_TO_MRNN_REGION)
    if training_df["region"].isna().any():
        missing = sorted(training_df.loc[training_df["region"].isna(), "source_region"].unique())
        raise ValueError(
            "Encountered source regions with no legacy mRNN mapping: "
            + ", ".join(str(value) for value in missing)
        )

    _validate_trace_lengths(
        training_df,
        condition_columns=MRNN_CONDITION_COLUMN_ORDER,
    )

    region_type = pd.CategoricalDtype(categories=MRNN_REGION_ORDER, ordered=True)
    training_df["region"] = training_df["region"].astype(region_type)
    if training_df["region"].isna().any():
        raise ValueError("Some modeled units were assigned an invalid mRNN region label.")

    ordered_columns = [
        "date",
        "uuid",
        "region",
        "source_region",
        "spike_channel",
        "recorded_agent",
        *MRNN_CONDITION_COLUMN_ORDER,
        *(f"{name}_n_trials" for name in MRNN_CONDITION_COLUMN_ORDER),
    ]
    training_df = training_df.loc[:, ordered_columns].sort_values(
        ["region", "date", "uuid"]
    )
    training_df = training_df.reset_index(drop=True)
    training_df["region"] = training_df["region"].astype(str)
    return training_df


__all__ = [
    "ANALYSIS_TO_MRNN_REGION",
    "CombinedFixationPSTHLoadResult",
    "DEFAULT_DATAFRAME_FILENAME",
    "DEFAULT_INPUT_SUBDIR",
    "DEFAULT_TIMELINE_FILENAME",
    "MRNN_CONDITION_COLUMN_ORDER",
    "MRNN_REGION_ORDER",
    "build_mrnn_training_dataframe",
    "load_combined_fixation_psth",
    "resolve_combined_fixation_psth_paths",
]
