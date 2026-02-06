"""Build face fixation probability tables."""

import argparse

from dal_monte_2022_analysis.analysis.face_fixation_probability import (
    FaceFixationProbabilitySettings,
    run_face_fixation_probability_analysis,
    run_interactive_face_fixation_probability_analysis,
)
from dal_monte_2022_analysis.config.load import load_face_fixation_probability_config


def main():
    """Parse CLI args and run face fixation probability analysis."""
    parser = argparse.ArgumentParser(
        description="Build face fixation probability tables.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--face-fixation-probability-cfg",
        default="configs/face_fixation_probability.yaml",
    )
    parser.add_argument("--no-cross", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--max-cross-pairs", type=int, default=None)
    parser.add_argument("--exclude-same-date", action="store_true")
    parser.add_argument("--include-same-session", action="store_true")
    parser.add_argument("--precision", type=int, default=None)

    args = parser.parse_args()

    cfg = load_face_fixation_probability_config(args.face_fixation_probability_cfg)
    settings = FaceFixationProbabilitySettings(
        cfg_path=args.dataset_cfg,
        input_modality=cfg.get("input_modality", "fixation_binary_vectors"),
        face_label=cfg.get("face_label", "face"),
        output_subdir=cfg.get("output_subdir", "face_fixation_probability"),
        within_filename=cfg.get(
            "within_filename",
            "within_session_face_fixation_probability.csv",
        ),
        cross_filename=cfg.get(
            "cross_filename",
            "cross_session_face_fixation_probability.csv",
        ),
        violin_filename=cfg.get(
            "violin_filename",
            "face_fixation_probability_violin.csv",
        ),
        interactive_modality=cfg.get("interactive_modality", "interactive_periods"),
        interactive_state_label=cfg.get("interactive_state_label", "interactive"),
        interactive_periods_filename=cfg.get(
            "interactive_periods_filename",
            "within_session_interactive_period_face_fixation_probability.csv",
        ),
        interactive_concat_filename=cfg.get(
            "interactive_concat_filename",
            "within_session_interactive_concat_face_fixation_probability.csv",
        ),
        decimal_precision=cfg.get("decimal_precision", 50),
        cross_pairs_max=cfg.get("cross_pairs_max"),
        cross_pairs_seed=cfg.get("cross_pairs_seed", 13),
        cross_exclude_same_session=cfg.get("cross_exclude_same_session", True),
        cross_exclude_same_date=cfg.get("cross_exclude_same_date", False),
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
    if args.precision is not None:
        settings.decimal_precision = args.precision

    # run_face_fixation_probability_analysis(
    #     settings,
    #     compute_cross=not args.no_cross,
    # )
    run_interactive_face_fixation_probability_analysis(settings)


if __name__ == "__main__":
    main()
