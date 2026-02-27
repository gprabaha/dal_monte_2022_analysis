"""Build per-session pupil vs fixation-density correlation tables."""

import argparse

from dal_monte_2022_analysis.behav.analysis.pupil_fixation_density_correlation import (
    PupilFixationDensityCorrelationSettings,
    run_pupil_fixation_density_correlation_analysis,
)
from dal_monte_2022_analysis.config.load import load_config


def main() -> None:
    """Parse CLI args and run pupil-fixation density correlation analysis."""
    parser = argparse.ArgumentParser(
        description="Build per-session pupil vs fixation-density correlation tables.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--pupil-fixation-density-correlation-cfg",
        default="configs/pupil_fixation_density_correlation.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--pupil-modality", default=None)
    parser.add_argument(
        "--correlation-method",
        default=None,
        choices=["pearson", "spearman"],
    )
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.pupil_fixation_density_correlation_cfg)
    settings = PupilFixationDensityCorrelationSettings(
        cfg_path=args.dataset_cfg,
        pupil_modality=cfg.get("pupil_modality", "smoothed_pupil_size"),
        fixation_density_modality=cfg.get(
            "fixation_density_modality",
            "fixation_density_vectors",
        ),
        joint_fixation_density_modality=cfg.get(
            "joint_fixation_density_modality",
            "joint_face_fixation_density",
        ),
        face_label=cfg.get("face_label", "face"),
        correlation_method=cfg.get("correlation_method", "pearson"),
        output_subdir=cfg.get(
            "output_subdir",
            "pupil_fixation_density_correlation",
        ),
        output_filename=cfg.get(
            "output_filename",
            "within_session_pupil_vs_face_fixation_density_correlation.csv",
        ),
        use_parallel=cfg.get("use_parallel", False),
        parallel_max_procs=cfg.get("parallel_max_procs", 32),
        test_single=cfg.get("test_single", False),
    )

    if args.use_parallel:
        settings.use_parallel = True
    if args.test_single:
        settings.test_single = True
    if args.pupil_modality:
        settings.pupil_modality = args.pupil_modality
    if args.correlation_method:
        settings.correlation_method = args.correlation_method

    run_pupil_fixation_density_correlation_analysis(
        settings,
        date=args.date,
        session=args.session,
    )


if __name__ == "__main__":
    main()
