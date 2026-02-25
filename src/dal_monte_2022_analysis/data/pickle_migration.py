"""Utilities to migrate legacy pickle module paths in stored data files."""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_ephys_data_config,
)

# Import compatibility shims so legacy module paths can be resolved during unpickle.
import dal_monte_2022_analysis.data.gaze_data  # noqa: F401
import dal_monte_2022_analysis.data.spike_data  # noqa: F401


LEGACY_MODULE_TOKENS = (
    b"dal_monte_2022_analysis.data.gaze_data",
    b"dal_monte_2022_analysis.data.spike_data",
)


@dataclass(frozen=True)
class PickleMigrationRecord:
    """One-file result for a legacy pickle migration pass."""

    path: Path
    had_legacy_reference: bool
    migrated: bool
    skipped_reason: Optional[str] = None
    error: Optional[str] = None
    backup_path: Optional[Path] = None


@dataclass
class PickleMigrationSummary:
    """Summary for a migration run."""

    total_files_seen: int = 0
    total_with_legacy_reference: int = 0
    total_migrated: int = 0
    total_failed: int = 0
    records: list[PickleMigrationRecord] = field(default_factory=list)


def _iter_pickle_paths(roots: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".pkl":
            paths.add(root.resolve())
            continue
        if root.is_dir():
            for path in root.rglob("*.pkl"):
                paths.add(path.resolve())
    return sorted(paths)


def _resolve_ephys_table_path(
    dataset_cfg: dict,
    ephys_cfg: dict,
) -> Path:
    raw_path = ephys_cfg.get("ephys_data_path")
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else (dataset_cfg["processed_data_root"] / path)
    filename = str(ephys_cfg.get("ephys_data_filename", "ephys_unit_data.pkl")).strip()
    return dataset_cfg["processed_data_root"] / filename


def _contains_any_token(path: Path, tokens: Sequence[bytes], chunk_size: int = 1024 * 1024) -> bool:
    tail = b""
    max_token_len = max(len(token) for token in tokens) if tokens else 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            haystack = tail + chunk
            if any(token in haystack for token in tokens):
                return True
            if max_token_len > 1:
                tail = haystack[-(max_token_len - 1) :]
            else:
                tail = b""
    return False


def _build_backup_path(path: Path, suffix: str) -> Path:
    candidate = path.with_name(f"{path.name}{suffix}")
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = path.with_name(f"{path.name}{suffix}.{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def migrate_legacy_pickle_modules(
    *,
    cfg_path: str = "configs/dataset.yaml",
    ephys_cfg_path: str = "configs/ephys_data.yaml",
    roots: Optional[Sequence[str]] = None,
    include_ephys_table: bool = True,
    dry_run: bool = False,
    force_all: bool = False,
    create_backup: bool = True,
    backup_suffix: str = ".pre_module_migration.bak",
) -> PickleMigrationSummary:
    """Rewrite pickles so they no longer depend on legacy module paths."""
    dataset_cfg = load_dataset_config(cfg_path)
    ephys_cfg = load_ephys_data_config(ephys_cfg_path)

    scan_roots: list[Path] = []
    if roots:
        scan_roots.extend(Path(root).expanduser().resolve() for root in roots)
    else:
        scan_roots.append(Path(dataset_cfg["processed_data_root"]).resolve())

    if include_ephys_table:
        ephys_table_path = _resolve_ephys_table_path(dataset_cfg, ephys_cfg).resolve()
        if ephys_table_path.exists():
            scan_roots.append(ephys_table_path)

    paths = _iter_pickle_paths(scan_roots)
    summary = PickleMigrationSummary(total_files_seen=len(paths))

    for path in paths:
        had_legacy_reference = _contains_any_token(path, LEGACY_MODULE_TOKENS)
        if had_legacy_reference:
            summary.total_with_legacy_reference += 1

        if not had_legacy_reference and not force_all:
            summary.records.append(
                PickleMigrationRecord(
                    path=path,
                    had_legacy_reference=False,
                    migrated=False,
                    skipped_reason="no_legacy_reference",
                )
            )
            continue

        if dry_run:
            summary.records.append(
                PickleMigrationRecord(
                    path=path,
                    had_legacy_reference=had_legacy_reference,
                    migrated=True,
                    skipped_reason="dry_run",
                )
            )
            summary.total_migrated += 1
            continue

        backup_path = None
        try:
            if create_backup:
                backup_path = _build_backup_path(path, backup_suffix)
                shutil.copy2(path, backup_path)

            with open(path, "rb") as f:
                obj = pickle.load(f)
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

            summary.records.append(
                PickleMigrationRecord(
                    path=path,
                    had_legacy_reference=had_legacy_reference,
                    migrated=True,
                    backup_path=backup_path,
                )
            )
            summary.total_migrated += 1
        except Exception as exc:
            summary.records.append(
                PickleMigrationRecord(
                    path=path,
                    had_legacy_reference=had_legacy_reference,
                    migrated=False,
                    error=str(exc),
                    backup_path=backup_path,
                )
            )
            summary.total_failed += 1

    return summary
