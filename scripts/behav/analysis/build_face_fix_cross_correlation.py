"""Build fixation cross-correlation tables (observed and shuffled modes)."""

import argparse

from dal_monte_2022_analysis.behav.analysis.fix_cross_correlation_cli import (
    apply_fix_cross_correlation_cli_overrides,
    build_fix_cross_correlation_settings_from_config,
    run_fix_cross_correlation_mode,
)


def main():
    """Parse CLI args and run requested cross-correlation mode."""
    parser = argparse.ArgumentParser(
        description="Build fixation cross-correlation tables.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--face-fix-cross-correlation-cfg",
        default="configs/face_fix_cross_correlation.yaml",
    )
    parser.add_argument(
        "--mode",
        default="observed",
        choices=[
            "observed",
            "shuffle_submit_hpc",
            "shuffle_worker",
            "shuffle_collate",
            "shuffle_local",
        ],
    )
    parser.add_argument(
        "--hpc-cfg",
        default="configs/hpc_face_fix_cross_correlation_shuffle.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--no-cross", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--max-cross-pairs", type=int, default=None)
    parser.add_argument("--exclude-same-date", action="store_true")
    parser.add_argument("--include-same-session", action="store_true")
    parser.add_argument(
        "--parallelize-across-cross-correlation-pairs",
        "--parallelize-across-crosscorr-pairs",
        dest="parallelize_across_cross_correlation_pairs",
        action="store_true",
    )
    parser.add_argument("--max-lag", type=int, default=None)
    parser.add_argument(
        "--time-scope",
        type=str,
        default=None,
        choices=["whole", "interactive", "non_interactive"],
    )
    parser.add_argument("--shuffle-n-shuffles", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--shuffle-log-every", type=int, default=None)
    parser.add_argument("--shuffle-non-stringent", action="store_true")
    parser.add_argument("--shuffle-no-within-pair-parallel", action="store_true")

    args = parser.parse_args()
    settings = build_fix_cross_correlation_settings_from_config(
        dataset_cfg_path=args.dataset_cfg,
        fix_cross_correlation_cfg_path=args.face_fix_cross_correlation_cfg,
        default_fixation_label="face",
        default_shuffle_pairs_subdir="within_session_shuffle_pair_results",
    )
    settings = apply_fix_cross_correlation_cli_overrides(
        settings,
        test_single=args.test_single,
        max_cross_pairs=args.max_cross_pairs,
        exclude_same_date=args.exclude_same_date,
        include_same_session=args.include_same_session,
        parallelize_across_cross_correlation_pairs=args.parallelize_across_cross_correlation_pairs,
        max_lag=args.max_lag,
        time_scope=args.time_scope,
        shuffle_n_shuffles=args.shuffle_n_shuffles,
        shuffle_seed=args.shuffle_seed,
        shuffle_log_every=args.shuffle_log_every,
        shuffle_non_stringent=args.shuffle_non_stringent,
        shuffle_no_within_pair_parallel=args.shuffle_no_within_pair_parallel,
    )
    run_fix_cross_correlation_mode(
        settings,
        mode=args.mode,
        compute_cross=not args.no_cross,
        date=args.date,
        session=args.session,
        hpc_cfg_path=args.hpc_cfg,
        dataset_cfg_path=args.dataset_cfg,
        fix_cross_correlation_cfg_path=args.face_fix_cross_correlation_cfg,
    )


if __name__ == "__main__":
    main()
