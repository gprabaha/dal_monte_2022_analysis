"""Plot pupil-vs-fixation-density correlation violins."""

import argparse

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.behav.plotting.pupil_fixation_density_correlation import (
    PupilFixationDensityCorrelationPlotSettings,
    plot_pupil_fixation_density_correlation_violin,
)


def main() -> None:
    """Parse CLI args and render the pupil-density correlation violin plot."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot m1/m2 pupil correlation violins against m1/m2/joint "
            "face fixation density (shared y-axis)."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument(
        "--pupil-fixation-density-correlation-cfg",
        default="configs/pupil_fixation_density_correlation.yaml",
    )
    args = parser.parse_args()

    corr_cfg = load_config(args.pupil_fixation_density_correlation_cfg)
    settings = PupilFixationDensityCorrelationPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        analysis_subdir=corr_cfg.get("output_subdir", "pupil_fixation_density_correlation"),
        correlations_filename=corr_cfg.get(
            "output_filename",
            "within_session_pupil_vs_face_fixation_density_correlation.csv",
        ),
        output_filename=corr_cfg.get(
            "plot_output_filename",
            "pupil_fixation_density_correlation_violin.pdf",
        ),
    )

    out_path = plot_pupil_fixation_density_correlation_violin(settings)
    print(f"[plot] wrote pupil-density correlation violin: {out_path}")


if __name__ == "__main__":
    main()

