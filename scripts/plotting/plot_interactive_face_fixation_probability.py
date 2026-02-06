"""Plot interactive-period face fixation probability violin comparisons."""

import argparse

from dal_monte_2022_analysis.plotting.fixation_probability import (
    InteractiveFixationProbabilityPlotSettings,
    plot_interactive_fixation_probability_violin,
)


def main():
    """Parse CLI args and run the plot."""
    parser = argparse.ArgumentParser(
        description="Plot interactive-period face fixation probability violins.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument(
        "--analysis-subdir",
        default="fixation_probability",
    )
    parser.add_argument(
        "--interactive-periods-filename",
        default="within_session_interactive_period_face_fixation_probability.csv",
    )
    parser.add_argument(
        "--interactive-concat-filename",
        default="within_session_interactive_concat_face_fixation_probability.csv",
    )
    parser.add_argument(
        "--output-filename",
        default="interactive_face_fixation_probability_violin.pdf",
    )

    args = parser.parse_args()

    settings = InteractiveFixationProbabilityPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        analysis_subdir=args.analysis_subdir,
        interactive_periods_filename=args.interactive_periods_filename,
        interactive_concat_filename=args.interactive_concat_filename,
        output_filename=args.output_filename,
    )

    plot_interactive_fixation_probability_violin(settings)


if __name__ == "__main__":
    main()
