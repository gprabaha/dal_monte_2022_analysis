"""Plot the saved face-fixation gap distributions."""

import argparse

from dal_monte_2022_analysis.behav.plotting.face_fixation_gap_distributions import (
    FaceFixationGapDistributionPlotSettings,
    plot_face_fixation_gap_distribution_figures,
)
from dal_monte_2022_analysis.config.load import load_config


def main():
    """Parse CLI args and write the two gap-distribution figures."""
    parser = argparse.ArgumentParser(
        description="Plot face-fixation gap distributions.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument(
        "--face-fixation-gap-cfg",
        default="configs/face_fixation_gap_distribution.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.face_fixation_gap_cfg)
    settings = FaceFixationGapDistributionPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        analysis_subdir=cfg.get("output_subdir", "face_fixation_gap_distributions"),
        m1_input_filename=cfg.get(
            "m1_output_filename",
            "within_session_m1_face_fixation_gap_distribution.csv",
        ),
        m1_m2_input_filename=cfg.get(
            "m1_m2_output_filename",
            "within_session_interactive_m1_m2_face_fixation_gap_distribution.csv",
        ),
        m1_output_filename=cfg.get(
            "m1_plot_filename",
            "m1_face_fixation_gap_distributions.pdf",
        ),
        m1_m2_output_filename=cfg.get(
            "m1_m2_plot_filename",
            "interactive_m1_m2_face_fixation_gap_distributions.pdf",
        ),
        interactive_state_label=cfg.get("interactive_state_label", "interactive"),
        non_interactive_state_label=cfg.get(
            "non_interactive_state_label",
            "non_interactive",
        ),
        histogram_bins=int(cfg.get("histogram_bins", 60)),
    )

    m1_out, m1_m2_out = plot_face_fixation_gap_distribution_figures(settings)
    print(f"[plot] wrote m1 face-fixation gap distribution figure: {m1_out}")
    print(f"[plot] wrote interactive m1-m2 face-fixation gap distribution figure: {m1_m2_out}")


if __name__ == "__main__":
    main()
