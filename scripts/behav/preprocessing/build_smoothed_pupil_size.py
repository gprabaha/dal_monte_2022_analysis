"""Build fixation-guided smoothed pupil time series."""

import argparse

from dal_monte_2022_analysis.config.load import load_pupil_smoothing_config
from dal_monte_2022_analysis.behav.preprocessing.pupil_smoothing import (
    PupilSmoothingSettings,
    run_pupil_smoothing,
)


def main() -> None:
    """Parse CLI args and run pupil smoothing."""
    parser = argparse.ArgumentParser(
        description=(
            "Smooth pupil traces using fixation-derived noise estimates, then "
            "interpolate non-fixation bins from fixation-constrained anchors."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--pupil-smoothing-cfg", default="configs/pupil_smoothing.yaml")
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    cfg = load_pupil_smoothing_config(args.pupil_smoothing_cfg)
    settings = PupilSmoothingSettings(
        cfg_path=args.dataset_cfg,
        input_pupil_modality=cfg.get("input_pupil_modality", "pupil_size"),
        fixations_modality=cfg.get("fixations_modality", "fixations"),
        output_modality=cfg.get("output_modality", "smoothed_pupil_size"),
        base_sigma_samples=float(cfg.get("base_sigma_samples", 6.0)),
        min_sigma_samples=float(cfg.get("min_sigma_samples", 2.0)),
        max_sigma_samples=float(cfg.get("max_sigma_samples", 40.0)),
        adaptive_noise_gain=float(cfg.get("adaptive_noise_gain", 2.0)),
        flatten_within_fixation=bool(cfg.get("flatten_within_fixation", True)),
        use_parallel=bool(cfg.get("use_parallel", True)),
        test_single=bool(cfg.get("test_single", False)),
        agents=cfg.get("agents"),
    )

    if args.use_parallel:
        settings.use_parallel = True
    if args.test_single:
        settings.test_single = True

    run_pupil_smoothing(
        settings,
        use_parallel=settings.use_parallel,
        test_single=settings.test_single,
    )


if __name__ == "__main__":
    main()
