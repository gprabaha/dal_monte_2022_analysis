"""Verify fixation location labels in random fixation files."""

from __future__ import annotations

import argparse
import pickle
import random
from typing import Iterable

import pandas as pd

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.utils.paths import scan_processed_data_paths


def _coerce_location(loc) -> list[str]:
    """Normalize a location entry to a list of strings."""
    if loc is None:
        return []
    if isinstance(loc, (list, tuple, set)):
        return [str(val) for val in loc if val is not None]
    try:
        if pd.isna(loc):
            return []
    except Exception:
        pass
    return [str(loc)]


def _summarize_locations(locations: Iterable) -> tuple[dict[str, int], int]:
    """Count how often each ROI label appears across fixation rows."""
    counts: dict[str, int] = {}
    total = 0
    for loc in locations:
        total += 1
        labels = set(_coerce_location(loc))
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    return counts, total


def _print_summary(row: dict, counts: dict[str, int], total: int) -> None:
    """Print a human-readable summary for one fixation file."""
    path = row["path"]
    header = f"File: {path}"
    print("=" * len(header))
    print(header)
    print(f"date={row['date']} session={row['session']} agent={row['agent']}")
    print(f"Total fixations: {total}")

    if total == 0:
        print("No fixation rows found.")
        return

    if not counts:
        print("No location labels found.")
        return

    unique_labels = sorted(counts)
    print("Unique locations:", ", ".join(unique_labels))
    print("Location proportions (per fixation row):")
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        proportion = count / total
        print(f"  {label:24s} {count:6d}  {proportion:6.2%}")


def main() -> None:
    """Sample fixation files and report location label frequencies."""
    parser = argparse.ArgumentParser(
        description="Verify fixation location labels in random fixation files.",
    )
    parser.add_argument(
        "--dataset-cfg",
        default="configs/dataset.yaml",
        help="Path to the dataset YAML config file.",
    )
    parser.add_argument("--n-files", type=int, default=5, help="Number of random files to sample.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")

    args = parser.parse_args()

    cfg = load_dataset_config(args.dataset_cfg)
    rows = scan_processed_data_paths(cfg, "fixations")
    if not rows:
        print(f"No fixation files found under {cfg['processed_data_root']}")
        return

    rng = random.Random(args.seed)
    sample_size = min(args.n_files, len(rows))
    sampled = rng.sample(rows, k=sample_size)

    print(f"Sampling {sample_size} fixation files from {cfg['processed_data_root']}")
    for row in sampled:
        path = row["path"]
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, pd.DataFrame):
            print(f"Skipping non-DataFrame object in {path}")
            continue
        if "location" not in obj.columns:
            print(f"Skipping {path}: missing location column")
            continue

        counts, total = _summarize_locations(obj["location"])
        _print_summary(row, counts, total)


if __name__ == "__main__":
    main()
