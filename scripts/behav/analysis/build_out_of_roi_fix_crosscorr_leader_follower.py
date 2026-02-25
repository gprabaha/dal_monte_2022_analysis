"""Build leader-follower summaries from out-of-ROI fixation cross-correlation outputs."""

import argparse

from dal_monte_2022_analysis.behav.analysis.fix_crosscorr_leader_follower import (
    FixCrossCorrLeaderFollowerSettings,
    run_fix_crosscorr_leader_follower_analysis,
)
from dal_monte_2022_analysis.config.load import (
    load_out_of_roi_fix_cross_correlation_config,
)
from dal_monte_2022_analysis.utils.paths import normalize_fix_crosscorr_time_scope


def _build_settings(args) -> FixCrossCorrLeaderFollowerSettings:
    """Construct leader-follower settings from config + CLI overrides."""
    cfg = load_out_of_roi_fix_cross_correlation_config(
        args.out_of_roi_fix_cross_correlation_cfg
    )
    fixation_label = cfg.get("fixation_label", cfg.get("face_label", "out_of_roi"))
    crosscorr_subdir = cfg.get(
        "crosscorr_output_subdir",
        cfg.get("output_subdir", "fix_cross_correlation"),
    )
    leader_follower_subdir = cfg.get(
        "leader_follower_output_subdir",
        f"{crosscorr_subdir}/leader_follower",
    )
    settings = FixCrossCorrLeaderFollowerSettings(
        cfg_path=args.dataset_cfg,
        fixation_label=fixation_label,
        output_subdir=leader_follower_subdir,
        crosscorr_input_subdir=crosscorr_subdir,
        within_filename=cfg.get("within_filename"),
        lags_filename=cfg.get("lags_filename"),
        time_scope=normalize_fix_crosscorr_time_scope(
            cfg.get("leader_follower_time_scope", "whole")
        ),
        session_output_filename=cfg.get(
            "leader_follower_session_filename",
            "within_session_out_of_roi_fix_crosscorr_leader_follower.pkl",
        ),
        date_summary_filename=cfg.get(
            "leader_follower_date_summary_filename",
            "date_summary_out_of_roi_fix_crosscorr_leader_follower.pkl",
        ),
        pair_summary_filename=cfg.get(
            "leader_follower_pair_summary_filename",
            cfg.get(
                "leader_follower_total_summary_filename",
                "pair_summary_out_of_roi_fix_crosscorr_leader_follower.pkl",
            ),
        ),
        global_summary_filename=cfg.get(
            "leader_follower_global_summary_filename",
            "global_summary_out_of_roi_fix_crosscorr_leader_follower.pkl",
        ),
        fixations_modality=cfg.get("leader_follower_fixations_modality", "fixations"),
        pupil_modality=cfg.get("leader_follower_pupil_modality", "pupil_size"),
        pupil_session_output_filename=cfg.get(
            "leader_follower_pupil_session_filename",
            "within_session_out_of_roi_fix_crosscorr_leader_follower_pupil_during_fixation.csv",
        ),
        pupil_date_summary_filename=cfg.get(
            "leader_follower_pupil_date_summary_filename",
            "date_summary_out_of_roi_fix_crosscorr_leader_follower_pupil_during_fixation.csv",
        ),
        pupil_pair_summary_filename=cfg.get(
            "leader_follower_pupil_pair_summary_filename",
            "pair_summary_out_of_roi_fix_crosscorr_leader_follower_pupil_during_fixation.csv",
        ),
        pupil_global_summary_filename=cfg.get(
            "leader_follower_pupil_global_summary_filename",
            "global_summary_out_of_roi_fix_crosscorr_leader_follower_pupil_during_fixation.csv",
        ),
        monkey_role_pupil_session_output_filename=cfg.get(
            "leader_follower_monkey_role_pupil_session_filename",
            "within_session_out_of_roi_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv",
        ),
        monkey_role_pupil_session_raw_filename=cfg.get(
            "leader_follower_monkey_role_pupil_session_raw_filename",
            "within_session_out_of_roi_fix_crosscorr_leader_follower_pupil_by_monkey_role_raw.pkl",
        ),
        monkey_role_pupil_summary_filename=cfg.get(
            "leader_follower_monkey_role_pupil_summary_filename",
            "summary_out_of_roi_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv",
        ),
        monkey_role_fixation_count_session_output_filename=cfg.get(
            "leader_follower_monkey_role_fixation_count_session_filename",
            "within_session_out_of_roi_fix_crosscorr_leader_follower_fixation_count_by_monkey_role.csv",
        ),
        monkey_role_fixation_count_summary_filename=cfg.get(
            "leader_follower_monkey_role_fixation_count_summary_filename",
            "summary_out_of_roi_fix_crosscorr_leader_follower_fixation_count_by_monkey_role.csv",
        ),
        monkey_role_fixation_duration_session_output_filename=cfg.get(
            "leader_follower_monkey_role_fixation_duration_session_filename",
            cfg.get(
                "leader_follower_monkey_role_fixation_count_session_filename",
                "within_session_out_of_roi_fix_crosscorr_leader_follower_fixation_duration_by_monkey_role.csv",
            ),
        ),
        monkey_role_fixation_duration_summary_filename=cfg.get(
            "leader_follower_monkey_role_fixation_duration_summary_filename",
            cfg.get(
                "leader_follower_monkey_role_fixation_count_summary_filename",
                "summary_out_of_roi_fix_crosscorr_leader_follower_fixation_duration_by_monkey_role.csv",
            ),
        ),
        property_use_all_fixations=bool(
            cfg.get("leader_follower_property_use_all_fixations", False)
        ),
        use_only_interactive_states=bool(
            cfg.get("leader_follower_use_only_interactive_states", False)
        ),
        interactive_modality=cfg.get(
            "leader_follower_interactive_modality",
            "interactive_periods",
        ),
        interactive_state_label=cfg.get(
            "leader_follower_interactive_state_label",
            "interactive",
        ),
        pupil_roi_keywords=cfg.get("leader_follower_pupil_roi_keywords"),
        pupil_test_alpha=float(cfg.get("leader_follower_pupil_test_alpha", 0.05)),
        pupil_parallelize_sessions=bool(
            cfg.get("leader_follower_pupil_parallelize_sessions", True)
        ),
        pupil_parallel_max_procs=int(
            cfg.get("leader_follower_pupil_parallel_max_procs", 16)
        ),
        tie_epsilon=float(cfg.get("leader_follower_tie_epsilon", 0.0)),
    )

    if args.tie_epsilon is not None:
        settings.tie_epsilon = max(0.0, float(args.tie_epsilon))
    return settings


def main():
    """Parse CLI args and run out-of-ROI leader-follower summaries."""
    parser = argparse.ArgumentParser(
        description="Build leader-follower summaries from out-of-ROI fixation cross-correlation outputs.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--out-of-roi-fix-cross-correlation-cfg",
        default="configs/out_of_roi_fix_cross_correlation.yaml",
    )
    parser.add_argument(
        "--tie-epsilon",
        type=float,
        default=None,
        help="Absolute lead-score threshold for calling ties (default from config or 0.0).",
    )
    args = parser.parse_args()

    settings = _build_settings(args)
    run_fix_crosscorr_leader_follower_analysis(settings)


if __name__ == "__main__":
    main()
