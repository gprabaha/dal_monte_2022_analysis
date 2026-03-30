"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DATASET_REQUIRED_KEYS = {"raw_data_root", "processed_data_root"}
_PROJECT_REQUIRED_KEYS = {"dataset_cfg_path", "ephys_data_cfg_path"}
_HPC_KEYS = {"job_file_path", "sbatch_script_path", "log_dir", "worker_script_path"}
DEFAULT_PROJECT_CONFIG_PATH = Path("configs/project.yaml")
DEFAULT_DATASET_CONFIG_PATH = Path("configs/dataset.yaml")
DEFAULT_EPHYS_CONFIG_PATH = Path("configs/ephys_data.yaml")
_MODULE_REPO_ROOT = Path(__file__).resolve().parents[3]


def get_repo_root(anchor: str | Path | None = None) -> Path:
    """Infer the repository root for a config or file path."""
    if anchor is None:
        return _MODULE_REPO_ROOT

    anchor_path = Path(anchor).expanduser()
    if not anchor_path.is_absolute():
        anchor_path = (_MODULE_REPO_ROOT / anchor_path).resolve()
    else:
        anchor_path = anchor_path.resolve()

    search_start = anchor_path if anchor_path.is_dir() else anchor_path.parent
    for candidate in (search_start, *search_start.parents):
        if candidate.name == "configs":
            return candidate.parent
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return search_start


def resolve_repo_path(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve a path relative to the repo root when it is not absolute."""
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    roots_to_try: list[Path] = []
    if repo_root is not None:
        roots_to_try.append(Path(repo_root).expanduser().resolve())
    roots_to_try.append(_MODULE_REPO_ROOT)

    for root in roots_to_try:
        candidate = (root / raw_path).resolve()
        if candidate.exists():
            return candidate

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    base_root = roots_to_try[0] if roots_to_try else _MODULE_REPO_ROOT
    return (base_root / raw_path).resolve()


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

    if stem == "project" or _PROJECT_REQUIRED_KEYS.issubset(cfg_keys):
        return "project"
    if stem == "dataset" or _DATASET_REQUIRED_KEYS.issubset(cfg_keys):
        return "dataset"
    if stem == "ephys_data" or "ephys_data_path" in cfg:
        return "ephys_data"
    if stem.startswith("hpc_") or bool(_HPC_KEYS & cfg_keys):
        return "hpc"
    return "generic"


def _normalize_dataset_paths(cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    base_dir = get_repo_root(cfg_path)
    return _resolve_paths(
        cfg,
        keys=["raw_data_root", "processed_data_root", "analysis_output_root"],
        base_dir=base_dir,
    )


def _normalize_ephys_paths(cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    base_dir = get_repo_root(cfg_path)
    return _resolve_paths(
        cfg,
        keys=["ephys_data_path"],
        base_dir=base_dir,
    )


def _normalize_hpc_paths(cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    base_dir = get_repo_root(cfg_path)
    return _resolve_paths(
        cfg,
        keys=["job_file_path", "sbatch_script_path", "log_dir", "worker_script_path"],
        base_dir=base_dir,
    )


def _normalize_project_paths(cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    base_dir = get_repo_root(cfg_path)
    cfg = _resolve_paths(
        cfg,
        keys=["dataset_cfg_path", "ephys_data_cfg_path", "plotting_cfg_path"],
        base_dir=base_dir,
    )
    return _resolve_paths(
        cfg,
        keys=["raw_data_root", "processed_data_root", "analysis_output_root"],
        base_dir=base_dir,
    )


_NORMALIZERS = {
    "generic": lambda cfg, _cfg_path: cfg,
    "project": _normalize_project_paths,
    "dataset": _normalize_dataset_paths,
    "ephys_data": _normalize_ephys_paths,
    "hpc": _normalize_hpc_paths,
}


def load_config(path: str | Path, *, config_type: str | None = None) -> dict[str, Any]:
    """Load YAML config and apply optional config-type-specific normalization."""
    cfg_path = resolve_repo_path(path)
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


def load_project_config(path: str | Path = DEFAULT_PROJECT_CONFIG_PATH) -> dict[str, Any]:
    """Load project-level config and normalize referenced config paths."""
    return load_config(path, config_type="project")


def resolve_dataset_cfg_path(path: str | Path = DEFAULT_PROJECT_CONFIG_PATH) -> Path:
    """Resolve a dataset config path from dataset/project config input."""
    cfg_path = resolve_repo_path(path)
    if cfg_path.exists():
        cfg = _load_yaml(cfg_path)
        cfg_type = _infer_config_type(cfg_path, cfg)
        if cfg_type == "project":
            project_cfg = load_project_config(cfg_path)
            dataset_cfg = project_cfg.get("dataset_cfg_path")
            if dataset_cfg is None:
                raise KeyError(f"Project config missing required key 'dataset_cfg_path': {cfg_path}")
            return resolve_repo_path(dataset_cfg, repo_root=get_repo_root(cfg_path))
        return cfg_path.resolve()

    if cfg_path == resolve_repo_path(DEFAULT_PROJECT_CONFIG_PATH):
        fallback = resolve_repo_path(DEFAULT_DATASET_CONFIG_PATH)
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Config path does not exist: {cfg_path}")


def resolve_ephys_cfg_path(
    path: str | Path = DEFAULT_EPHYS_CONFIG_PATH,
    *,
    project_cfg_path: str | Path | None = None,
) -> Path:
    """Resolve an ephys-data config path from ephys/project config input."""
    cfg_path = resolve_repo_path(path)
    if cfg_path.exists():
        cfg = _load_yaml(cfg_path)
        cfg_type = _infer_config_type(cfg_path, cfg)
        if cfg_type == "project":
            project_cfg = load_project_config(cfg_path)
            ephys_cfg = project_cfg.get("ephys_data_cfg_path")
            if ephys_cfg is None:
                raise KeyError(f"Project config missing required key 'ephys_data_cfg_path': {cfg_path}")
            return resolve_repo_path(ephys_cfg, repo_root=get_repo_root(cfg_path))
        return cfg_path.resolve()

    if project_cfg_path is not None:
        project_path = resolve_repo_path(project_cfg_path)
        if project_path.exists():
            project_cfg = load_project_config(project_path)
            ephys_cfg = project_cfg.get("ephys_data_cfg_path")
            if ephys_cfg is not None:
                return resolve_repo_path(ephys_cfg, repo_root=get_repo_root(project_path))

    if cfg_path == resolve_repo_path(DEFAULT_EPHYS_CONFIG_PATH):
        fallback = resolve_repo_path(DEFAULT_EPHYS_CONFIG_PATH)
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Config path does not exist: {cfg_path}")


def load_ephys_data_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for ephys data config loading."""
    return load_config(path, config_type="ephys_data")


def load_gaze_event_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_hpc_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for HPC config loading."""
    return load_config(path, config_type="hpc")


def load_face_fixation_hsmm_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")


def load_ephys_fixation_psth_config(path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper for generic config loading."""
    return load_config(path, config_type="generic")
