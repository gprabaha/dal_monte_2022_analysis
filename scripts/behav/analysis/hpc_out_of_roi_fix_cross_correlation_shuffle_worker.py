"""HPC worker script for one within-session shuffled out-of-ROI pair."""

import argparse

from dal_monte_2022_analysis.behav.analysis.fix_cross_correlation import (
    FixCrossCorrelationSettings,
    process_and_save_within_session_shuffle_pair,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.utils.paths import normalize_fix_cross_correlation_time_scope


def main():
    """Parse CLI args and process one within-session shuffle pair."""
    parser = argparse.ArgumentParser(
        description="Within-session shuffled out-of-ROI cross-correlation worker.",
    )
    parser.add_argument("--dataset-cfg", required=True)
    parser.add_argument(
        "--fix-cross-correlation-cfg",
        "--fix-crosscorr-cfg",
        dest="fix_cross_correlation_cfg",
        required=True,
    )
    parser.add_argument("--date", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--time-scope",
        default=None,
        choices=["whole", "interactive", "non_interactive"],
    )
    args = parser.parse_args()

    cfg = load_config(args.fix_cross_correlation_cfg)
    fixation_label = cfg.get("fixation_label", cfg.get("face_label", "out_of_roi"))
    settings = FixCrossCorrelationSettings(
        cfg_path=args.dataset_cfg,
        input_modality=cfg.get("input_modality", "fixation_binary_vectors"),
        fixation_label=fixation_label,
        output_subdir=cfg.get(
            "cross_correlation_output_subdir",
            cfg.get(
                "crosscorr_output_subdir",
                cfg.get("output_subdir", "cross_correlation_outputs"),
            ),
        ),
        within_filename=cfg.get("within_filename"),
        cross_filename=cfg.get("cross_filename"),
        lags_filename=cfg.get("lags_filename"),
        max_lag=cfg.get("max_lag", 60000),
        time_scope=normalize_fix_cross_correlation_time_scope(cfg.get("time_scope", "whole")),
        interactive_modality=cfg.get("interactive_modality", "interactive_periods"),
        interactive_state_label=cfg.get("interactive_state_label", "interactive"),
        cross_pairs_max=cfg.get("cross_pairs_max"),
        cross_pairs_seed=cfg.get("cross_pairs_seed", 13),
        cross_exclude_same_session=cfg.get("cross_exclude_same_session", True),
        cross_exclude_same_date=cfg.get("cross_exclude_same_date", False),
        parallelize_across_cross_correlation_pairs=False,
        shuffle_output_filename=cfg.get("shuffle_output_filename"),
        shuffle_pairs_subdir=cfg.get(
            "shuffle_pairs_subdir",
            "within_session_out_of_roi_shuffle_pair_results",
        ),
        shuffle_n_shuffles=cfg.get("shuffle_n_shuffles", 1000),
        shuffle_stringent=cfg.get("shuffle_stringent", True),
        shuffle_seed=cfg.get("shuffle_seed", 13),
        shuffle_parallelize_within_pair=cfg.get("shuffle_parallelize_within_pair", True),
        shuffle_log_every=cfg.get("shuffle_log_every", 100),
        test_single=False,
    )
    if args.time_scope is not None:
        settings.time_scope = normalize_fix_cross_correlation_time_scope(args.time_scope)

    process_and_save_within_session_shuffle_pair(
        settings=settings,
        date=args.date,
        session=args.session,
    )


if __name__ == "__main__":
    main()
