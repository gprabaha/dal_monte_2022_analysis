"""Plot interactive/non-interactive period duration distributions by monkey pair."""

import argparse

from dal_monte_2022_analysis.plotting.interactive_period_durations import (
    InteractivePeriodDurationDistributionPlotSettings,
    plot_interactive_period_duration_distributions,
)


def main():
    """Parse CLI args and generate multi-panel interactive-period duration histograms."""
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
        "--output-filename",
        default="interactive_period_duration_distributions_histogram.pdf",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=60,
        help="Number of shared bins used across all histograms.",
    )
    args = parser.parse_args()

    settings = InteractivePeriodDurationDistributionPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        interactive_periods_cfg_path=args.interactive_periods_cfg,
        analysis_subdir=args.analysis_subdir,
        output_subdir=args.output_subdir,
        output_filename=args.output_filename,
        histogram_bins=int(args.histogram_bins),
    )

    out_path = plot_interactive_period_duration_distributions(settings)
    print(f"[plot] wrote interactive-period duration histogram grid: {out_path}")


if __name__ == "__main__":
    main()
