"""Plot m1-m2 face cross-correlation traces against cross-session/shuffle controls."""

import argparse

from dal_monte_2022_analysis.config.load import load_face_fix_cross_correlation_config
from dal_monte_2022_analysis.plotting.fix_cross_correlation_m1_m2 import (
    M1M2CrossCorrComparisonPlotSettings,
    plot_observed_vs_cross_session_m1_m2,
    plot_observed_vs_shuffle_m1_m2,
)


def main():
    """Parse CLI args and render both m1-m2 comparison figures."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot face m1-m2 observed within-session cross-correlation traces against "
            "cross-session and shuffled controls (whole/interactive/non-interactive)."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument(
        "--face-fix-cross-correlation-cfg",
        default="configs/face_fix_cross_correlation.yaml",
    )
    args = parser.parse_args()

    cfg = load_face_fix_cross_correlation_config(args.face_fix_cross_correlation_cfg)
    crosscorr_subdir = cfg.get("crosscorr_output_subdir", cfg.get("output_subdir", "crosscorr_outputs"))

    settings = M1M2CrossCorrComparisonPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        analysis_subdir=crosscorr_subdir,
        fixation_label=cfg.get("fixation_label", "face"),
        scopes=tuple(cfg.get("m1_m2_plot_scopes", ("whole", "interactive", "non_interactive"))),
        significance_alpha=float(cfg.get("m1_m2_plot_significance_alpha", 0.05)),
        output_subdir=cfg.get("m1_m2_plot_output_subdir", "plots/m1-m2"),
        observed_vs_cross_filename=cfg.get(
            "m1_m2_observed_vs_cross_filename",
            "observed_vs_cross_session_face_m1_m2_crosscorr.pdf",
        ),
        observed_vs_shuffle_filename=cfg.get(
            "m1_m2_observed_vs_shuffle_filename",
            "observed_vs_shuffle_face_m1_m2_crosscorr.pdf",
        ),
    )

    out_cross = plot_observed_vs_cross_session_m1_m2(settings)
    print(f"[plot] wrote observed-vs-cross-session m1-m2 figure: {out_cross}")

    out_shuffle = plot_observed_vs_shuffle_m1_m2(settings)
    print(f"[plot] wrote observed-vs-shuffle m1-m2 figure: {out_shuffle}")


if __name__ == "__main__":
    main()
