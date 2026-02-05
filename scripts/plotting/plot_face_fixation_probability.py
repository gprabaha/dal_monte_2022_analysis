"""Plot face fixation probability violin comparisons."""

import argparse

from dal_monte_2022_analysis.plotting.face_fixation_probability import (
    FaceFixationProbabilityPlotSettings,
    plot_face_fixation_probability_violin,
)


def main():
    """Parse CLI args and run the plot."""
    parser = argparse.ArgumentParser(
        description="Plot face fixation probability violins.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument(
        "--analysis-subdir",
        default="face_fixation_probability",
    )
    parser.add_argument(
        "--within-filename",
        default="within_session_face_fixation_probability.csv",
    )
    parser.add_argument(
        "--cross-filename",
        default="cross_session_face_fixation_probability.csv",
    )
    parser.add_argument(
        "--output-filename",
        default="face_fixation_probability_violin.pdf",
    )

    args = parser.parse_args()

    settings = FaceFixationProbabilityPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        analysis_subdir=args.analysis_subdir,
        within_filename=args.within_filename,
        cross_filename=args.cross_filename,
        output_filename=args.output_filename,
    )

    plot_face_fixation_probability_violin(settings)


if __name__ == "__main__":
    main()
