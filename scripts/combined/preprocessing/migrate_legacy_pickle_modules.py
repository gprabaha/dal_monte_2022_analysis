"""Migrate old pickle files that reference legacy data module paths."""

from __future__ import annotations

import argparse

from dal_monte_2022_analysis.data.migrations.pickle_modules import (
    migrate_legacy_pickle_modules,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite pickle files so legacy module paths "
            "(data.gaze_data / data.spike_data) are replaced by current modules."
        )
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-cfg", default="configs/ephys_data.yaml")
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Optional root file/dir to scan (can be passed multiple times).",
    )
    parser.add_argument(
        "--no-ephys-table",
        action="store_true",
        help="Do not include the configured ephys table path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be migrated without rewriting files.",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Rewrite all discovered pickles, not only those with legacy references.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create backup files before rewriting.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".pre_module_migration.bak",
        help="Backup suffix used when rewriting files.",
    )
    args = parser.parse_args()

    summary = migrate_legacy_pickle_modules(
        cfg_path=args.dataset_cfg,
        ephys_cfg_path=args.ephys_cfg,
        roots=args.root,
        include_ephys_table=not args.no_ephys_table,
        dry_run=args.dry_run,
        force_all=args.force_all,
        create_backup=not args.no_backup,
        backup_suffix=args.backup_suffix,
    )

    print("Legacy pickle module migration summary:")
    print(f"  total_files_seen={summary.total_files_seen}")
    print(f"  total_with_legacy_reference={summary.total_with_legacy_reference}")
    print(f"  total_migrated={summary.total_migrated}")
    print(f"  total_failed={summary.total_failed}")

    failures = [rec for rec in summary.records if rec.error]
    if failures:
        print("\nFailures:")
        for rec in failures[:20]:
            print(f"  - path={rec.path}")
            print(f"    error={rec.error}")


if __name__ == "__main__":
    main()
