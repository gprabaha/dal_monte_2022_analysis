"""Compatibility wrappers for behavioral raw/processed dataset indexing."""

import pandas as pd

from dal_monte_2022_analysis.data.loaders.behavioral import (
    index_behavioral_processed_data_from_cfg,
    index_behavioral_source_data_from_cfg,
)
from dal_monte_2022_analysis.data.transforms.annotate import (
    load_pair_context_table_from_cfg,
)

def _load_ephys_days(cfg: dict) -> pd.DataFrame:
    """Load session-level monkey metadata for behavioral indexing."""
    return load_pair_context_table_from_cfg(cfg)[["date", "monkey_name_m1", "monkey_name_m2"]]


def index_dataset(cfg: dict, modality: str) -> pd.DataFrame:
    """Index raw behavioral source files for one configured modality."""
    return index_behavioral_source_data_from_cfg(cfg, modality)


def index_processed_dataset(cfg: dict, modality: str) -> pd.DataFrame:
    """Index processed pickles for a modality into a DataFrame."""
    return index_behavioral_processed_data_from_cfg(cfg, modality)
