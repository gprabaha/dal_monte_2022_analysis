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


def _resolve_paths(cfg: dict, keys, base_dir: Path) -> dict:
    for key in keys:
        if key in cfg:
            cfg[key] = (base_dir / cfg[key]).resolve() if not Path(cfg[key]).is_absolute() else Path(cfg[key])
    return cfg


def load_gaze_event_config(path: str) -> dict:
    """Load fixation/saccade detection config and normalize paths."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def load_hpc_config(path: str) -> dict:
    """Load HPC config and normalize path entries."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base_dir = Path(path).resolve().parent
    cfg = _resolve_paths(
        cfg,
        keys=["job_file_path", "sbatch_script_path", "log_dir"],
        base_dir=base_dir,
    )
    return cfg
