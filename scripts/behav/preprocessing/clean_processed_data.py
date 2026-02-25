"""CLI wrapper for pruning timelines and interpolating processed data."""

import argparse

from dal_monte_2022_analysis.behav.preprocessing.clean_dataset import clean_dataset


def main():
    """Parse CLI arguments and run pruning/interpolation on processed data."""
    parser = argparse.ArgumentParser(
        description="Prune timelines and interpolate gaze/pupil data."
    )
    parser.add_argument(
        "--cfg",
        default="configs/dataset.yaml",
        help="Path to dataset config YAML.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_cleaned",
        help="Suffix appended to modality names for outputs.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Window size for sliding interpolation.",
    )
    parser.add_argument(
        "--max-nans",
        type=int,
        default=3,
        help="Max NaNs allowed in a window for interpolation.",
    )
    args = parser.parse_args()

    clean_dataset(
        cfg_path=args.cfg,
        output_suffix=args.output_suffix,
        window_size=args.window_size,
        max_nans=args.max_nans,
    )


if __name__ == "__main__":
    main()
