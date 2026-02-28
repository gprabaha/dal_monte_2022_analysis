"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DATASET_REQUIRED_KEYS = {"raw_data_root", "processed_data_root"}
_HPC_KEYS = {"job_file_path", "sbatch_script_path", "log_dir", "worker_script_path"}


def _resolve_paths(cfg: dict, keys, base_dir: Path, *, alt_base_dir: Path | None = None) -> dict:
    """Resolve selected keys in-place to Path values."""
    for key in keys:
        if key not in cfg:
            continue
        path = Path(cfg[key])
        if path.is_absolute():
            cfg[key] = path
            continue
        if alt_base_dir is not None:
            cfg[key] = (alt_base_dir / path).resolve()
        else:
            cfg[key] = (base_dir / path).resolve()
    return cfg


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _infer_config_type(path: Path, cfg: dict[str, Any]) -> str:
    stem = path.stem.lower()
    cfg_keys = set(cfg.keys())

    if stem == "dataset" or _DATASET_REQUIRED_KEYS.issubset(cfg_keys):
        return "dataset"
    if stem == "ephys_data" or "ephys_data_path" in cfg:
        return "ephys_data"
    if stem.startswith("hpc_") or bool(_HPC_KEYS & cfg_keys):
        return "hpc"
    return "generic"


def _normalize_dataset_paths(cfg: dict[str, Any], _: Path) -> dict[str, Any]:
    for key in ("raw_data_root", "processed_data_root", "analysis_output_root"):
        if key in cfg:
            cfg[key] = Path(cfg[key])
    return cfg


def _normalize_ephys_paths(cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    base_dir = cfg_path.resolve().parent
    repo_root = base_dir.parent
    return _resolve_paths(
        cfg,
        keys=["ephys_data_path"],
        base_dir=base_dir,
        alt_base_dir=repo_root,
    )


def _normalize_hpc_paths(cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    base_dir = cfg_path.resolve().parent
    repo_root = base_dir.parent
    return _resolve_paths(
        cfg,
        keys=["job_file_path", "sbatch_script_path", "log_dir", "worker_script_path"],
        base_dir=base_dir,
        alt_base_dir=repo_root,
    )


_NORMALIZERS = {
    "generic": lambda cfg, _cfg_path: cfg,
    "dataset": _normalize_dataset_paths,
    "ephys_data": _normalize_ephys_paths,
    "hpc": _normalize_hpc_paths,
}


def load_config(path: str | Path, *, config_type: str | None = None) -> dict[str, Any]:
    """Load YAML config and apply optional config-type-specific normalization."""
    cfg_path = Path(path)
    cfg = _load_yaml(cfg_path)

    resolved_type = config_type.lower() if config_type is not None else _infer_config_type(cfg_path, cfg)
    if resolved_type not in _NORMALIZERS:
        raise ValueError(
            f"Unknown config_type={config_type!r}. Expected one of: {', '.join(sorted(_NORMALIZERS))}."
        )
    return _NORMALIZERS[resolved_type](cfg, cfg_path)


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for dataset config loading."""
    return load_config(path, config_type="dataset")


def load_ephys_data_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for ephys data config loading."""
    return load_config(path, config_type="ephys_data")


def load_gaze_event_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_hpc_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for HPC config loading."""
    return load_config(path, config_type="hpc")


def load_fixation_binary_vector_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_fixation_density_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_joint_fixation_density_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_interactive_periods_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_pupil_smoothing_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_face_fixation_probability_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_out_of_roi_fixation_probability_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_face_fix_cross_correlation_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_out_of_roi_fix_cross_correlation_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_plotting_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_face_fixation_hsmm_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_ephys_fixation_psth_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_ephys_period_psth_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_ephys_fixation_neural_cross_correlation_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")
