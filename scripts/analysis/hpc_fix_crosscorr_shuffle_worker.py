"""HPC worker script for one within-session shuffled cross-correlation pair."""

import argparse

from dal_monte_2022_analysis.analysis.fix_cross_correlation import (
    FixCrossCorrelationSettings,
    process_and_save_within_session_shuffle_pair,
)
from dal_monte_2022_analysis.config.load import load_face_fix_cross_correlation_config


def main():
    """Parse CLI args and process one within-session shuffle pair."""
    parser = argparse.ArgumentParser(
        description="Within-session shuffled cross-correlation worker.",
    )
    parser.add_argument("--dataset-cfg", required=True)
    parser.add_argument("--fix-crosscorr-cfg", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    cfg = load_face_fix_cross_correlation_config(args.fix_crosscorr_cfg)
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
        parallelize_across_crosscorr_pairs=False,
        shuffle_output_filename=cfg.get(
            "shuffle_output_filename",
            "within_session_face_fix_cross_correlation_shuffle.pkl",
        ),
        shuffle_pairs_subdir=cfg.get("shuffle_pairs_subdir", "within_session_shuffle_pair_results"),
        shuffle_n_shuffles=cfg.get("shuffle_n_shuffles", 1000),
        shuffle_stringent=cfg.get("shuffle_stringent", True),
        shuffle_seed=cfg.get("shuffle_seed", 13),
        shuffle_parallelize_within_pair=cfg.get("shuffle_parallelize_within_pair", True),
        test_single=False,
    )

    process_and_save_within_session_shuffle_pair(
        settings=settings,
        date=args.date,
        session=args.session,
    )


if __name__ == "__main__":
    main()

