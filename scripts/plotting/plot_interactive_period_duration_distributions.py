"""Plot interactive/non-interactive period duration distributions by monkey pair."""

import argparse

from dal_monte_2022_analysis.plotting.interactive_period_durations import (
    InteractivePeriodDurationDistributionPlotSettings,
    plot_interactive_period_duration_distributions,
)


def main():
    """Parse CLI args and generate per-pair interactive-period duration plots."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot interactive and non-interactive period duration distributions "
            "for each monkey pair."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--interactive-periods-cfg", default="configs/interactive_periods.yaml")
    parser.add_argument("--analysis-subdir", default="interactive_periods")
    parser.add_argument("--output-subdir", default="duration_distributions")
    parser.add_argument(
        "--output-filename-prefix",
        default="interactive_period_duration_distribution",
    )
    parser.add_argument(
        "--max-samples-per-state",
        type=int,
        default=0,
        help="Optional cap for displayed samples per state (0 means no subsampling).",
    )
    args = parser.parse_args()

    settings = InteractivePeriodDurationDistributionPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        interactive_periods_cfg_path=args.interactive_periods_cfg,
        analysis_subdir=args.analysis_subdir,
        output_subdir=args.output_subdir,
        output_filename_prefix=args.output_filename_prefix,
        max_samples_per_state=int(args.max_samples_per_state),
    )

    out_paths = plot_interactive_period_duration_distributions(settings)
    print(
        "[plot] wrote "
        f"{len(out_paths)} interactive-period duration distribution plot(s) to: "
        f"{out_paths[0].parent}"
    )


if __name__ == "__main__":
    main()
