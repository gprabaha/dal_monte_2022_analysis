"""Build leader-follower summaries from face fixation cross-correlation outputs."""

import argparse

from dal_monte_2022_analysis.analysis.fix_crosscorr_leader_follower import (
    FixCrossCorrLeaderFollowerSettings,
    run_fix_crosscorr_leader_follower_analysis,
)
from dal_monte_2022_analysis.config.load import load_face_fix_cross_correlation_config


def _build_settings(args) -> FixCrossCorrLeaderFollowerSettings:
    """Construct leader-follower settings from config + CLI overrides."""
    cfg = load_face_fix_cross_correlation_config(args.face_fix_cross_correlation_cfg)
    fixation_label = cfg.get("fixation_label", cfg.get("face_label", "face"))
    settings = FixCrossCorrLeaderFollowerSettings(
        cfg_path=args.dataset_cfg,
        fixation_label=fixation_label,
        output_subdir=cfg.get("output_subdir", "fix_cross_correlation"),
        within_filename=cfg.get(
            "within_filename",
            "within_session_face_fix_cross_correlation.pkl",
        ),
        lags_filename=cfg.get("lags_filename"),
        session_output_filename=cfg.get(
            "leader_follower_session_filename",
            "within_session_face_fix_crosscorr_leader_follower.csv",
        ),
        date_summary_filename=cfg.get(
            "leader_follower_date_summary_filename",
            "date_summary_face_fix_crosscorr_leader_follower.csv",
        ),
        pair_summary_filename=cfg.get(
            "leader_follower_pair_summary_filename",
            cfg.get(
                "leader_follower_total_summary_filename",
                "pair_summary_face_fix_crosscorr_leader_follower.csv",
            ),
        ),
        global_summary_filename=cfg.get(
            "leader_follower_global_summary_filename",
            "global_summary_face_fix_crosscorr_leader_follower.csv",
        ),
        tie_epsilon=float(cfg.get("leader_follower_tie_epsilon", 0.0)),
    )

    if args.tie_epsilon is not None:
        settings.tie_epsilon = max(0.0, float(args.tie_epsilon))
    return settings


def main():
    """Parse CLI args and run face leader-follower summaries."""
    parser = argparse.ArgumentParser(
        description="Build leader-follower summaries from face fixation cross-correlation outputs.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--face-fix-cross-correlation-cfg",
        default="configs/face_fix_cross_correlation.yaml",
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
