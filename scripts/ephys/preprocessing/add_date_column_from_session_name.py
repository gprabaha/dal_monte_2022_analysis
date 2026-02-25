"""Add a `date` column to unit-level ephys spike pickle from `session_name`."""

from __future__ import annotations

import argparse

from dal_monte_2022_analysis.ephys.preprocessing.spike_data import (
    add_date_column_to_ephys_pickle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add/update date column in ephys spike table so date matches session_name values."
        )
    )
    parser.add_argument(
        "--dataset-cfg",
        default="configs/dataset.yaml",
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--ephys-cfg",
        default="configs/ephys_data.yaml",
        help="Path to ephys YAML config.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional explicit path to input spike table pickle.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output pickle path. Defaults to in-place update.",
    )
    parser.add_argument(
        "--session-col",
        default="session_name",
        help="Source column name for session/day token.",
    )
    parser.add_argument(
        "--date-col",
        default="date",
        help="Target date column name.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing date values if they differ from session_name.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup when writing in-place.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix for in-place updates (default: .bak).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize changes without writing output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = add_date_column_to_ephys_pickle(
        cfg_path=args.dataset_cfg,
        ephys_cfg_path=args.ephys_cfg,
        input_path=args.input_path,
        output_path=args.output_path,
        session_col=args.session_col,
        date_col=args.date_col,
        overwrite_existing=args.overwrite_existing,
        create_backup=not args.no_backup,
        backup_suffix=args.backup_suffix,
        dry_run=args.dry_run,
    )
    print("Ephys date-column preprocessing complete:")
    print(f"  source_path={summary.source_path}")
    print(f"  output_path={summary.output_path}")
    print(f"  n_rows={summary.n_rows}")
    print(f"  date_column_created={summary.date_column_created}")
    print(f"  n_overwritten={summary.n_overwritten}")
    print(f"  backup_path={summary.backup_path}")
    print(f"  dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
