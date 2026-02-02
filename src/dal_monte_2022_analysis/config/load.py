"""Configuration loading helpers."""

import yaml
from pathlib import Path


def load_dataset_config(path: str) -> dict:
    """Load the dataset config and normalize path entries.

    Args:
        path: Path to the dataset YAML config file.

    Returns:
        Parsed config dictionary with Path objects for data roots.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["raw_data_root"] = Path(cfg["raw_data_root"])
    cfg["processed_data_root"] = Path(cfg["processed_data_root"])

    return cfg


def _resolve_paths(cfg: dict, keys, base_dir: Path, *, alt_base_dir: Path | None = None) -> dict:
    """Resolve selected keys in a config dict relative to provided base dirs.

    Args:
        cfg: Config dictionary to update in place.
        keys: Iterable of keys to resolve to absolute paths.
        base_dir: Base directory for resolving relative paths.
        alt_base_dir: Alternate base directory to prefer when provided.

    Returns:
        The updated config dictionary.
    """
    for key in keys:
        if key not in cfg:
            continue
        path = Path(cfg[key])
        if path.is_absolute():
            cfg[key] = path
            continue
        if alt_base_dir is not None:
            alt_candidate = (alt_base_dir / path).resolve()
            cfg[key] = alt_candidate
        else:
            cfg[key] = (base_dir / path).resolve()
    return cfg


def load_gaze_event_config(path: str) -> dict:
    """Load fixation/saccade detection config (no path normalization).

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dictionary (empty if file is empty).
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def load_hpc_config(path: str) -> dict:
    """Load HPC config and normalize relevant path entries.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config with resolved paths for job files and scripts.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base_dir = Path(path).resolve().parent
    repo_root = base_dir.parent
    cfg = _resolve_paths(
        cfg,
        keys=["job_file_path", "sbatch_script_path", "log_dir", "worker_script_path"],
        base_dir=base_dir,
        alt_base_dir=repo_root,
    )
    return cfg


def load_fixation_binary_vector_config(path: str) -> dict:
    """Load fixation binary vector config (no path normalization).

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dictionary (empty if file is empty).
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}
