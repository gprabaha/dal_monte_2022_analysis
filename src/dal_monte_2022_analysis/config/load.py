"""Configuration loading helpers."""

import yaml
from pathlib import Path


def load_dataset_config(path: str) -> dict:
    """Load the dataset config and normalize path entries."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["raw_data_root"] = Path(cfg["raw_data_root"])
    cfg["processed_data_root"] = Path(cfg["processed_data_root"])

    return cfg
