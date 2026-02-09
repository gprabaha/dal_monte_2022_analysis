"""Build face fixation cross-correlation tables."""

import argparse

from dal_monte_2022_analysis.analysis.fix_cross_correlation import (
    FixCrossCorrelationSettings,
    run_fix_cross_correlation_analysis,
)
from dal_monte_2022_analysis.config.load import load_face_fix_cross_correlation_config


def main():
    """Parse CLI args and run face fixation cross-correlation analysis."""
    parser = argparse.ArgumentParser(
        description="Build face fixation cross-correlation tables.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--face-fix-cross-correlation-cfg",
        default="configs/face_fix_cross_correlation.yaml",
    )
    parser.add_argument("--no-cross", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--max-cross-pairs", type=int, default=None)
    parser.add_argument("--exclude-same-date", action="store_true")
    parser.add_argument("--include-same-session", action="store_true")
    parser.add_argument("--parallelize-across-crosscorr-pairs", action="store_true")
    parser.add_argument("--max-lag", type=int, default=None)

    args = parser.parse_args()

    cfg = load_face_fix_cross_correlation_config(args.face_fix_cross_correlation_cfg)
    fixation_label = cfg.get("fixation_label", cfg.get("face_label", "face"))
    settings = FixCrossCorrelationSettings(
        cfg_path=args.dataset_cfg,
        input_modality=cfg.get("input_modality", "fixation_binary_vectors"),
        fixation_label=fixation_label,
        output_subdir=cfg.get("output_subdir", "fix_cross_correlation"),
        within_filename=cfg.get(
            "within_filename",
            "within_session_face_fix_cross_correlation.pkl",
        ),
        cross_filename=cfg.get(
            "cross_filename",
            "cross_session_face_fix_cross_correlation.pkl",
        ),
        lags_filename=cfg.get("lags_filename"),
        max_lag=cfg.get("max_lag", 60000),
        cross_pairs_max=cfg.get("cross_pairs_max"),
        cross_pairs_seed=cfg.get("cross_pairs_seed", 13),
        cross_exclude_same_session=cfg.get("cross_exclude_same_session", True),
        cross_exclude_same_date=cfg.get("cross_exclude_same_date", False),
        parallelize_across_crosscorr_pairs=cfg.get(
            "parallelize_across_crosscorr_pairs",
            False,
        ),
        test_single=cfg.get("test_single", False),
    )

    if args.test_single:
        settings.test_single = True
    if args.max_cross_pairs is not None:
        settings.cross_pairs_max = args.max_cross_pairs
    if args.exclude_same_date:
        settings.cross_exclude_same_date = True
    if args.include_same_session:
        settings.cross_exclude_same_session = False
    if args.parallelize_across_crosscorr_pairs:
        settings.parallelize_across_crosscorr_pairs = True
    if args.max_lag is not None:
        settings.max_lag = max(0, args.max_lag)

    run_fix_cross_correlation_analysis(
        settings,
        compute_cross=not args.no_cross,
    )


if __name__ == "__main__":
    main()
