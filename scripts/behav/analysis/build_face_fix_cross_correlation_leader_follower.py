"""Build leader-follower summaries from face fixation cross-correlation outputs."""

import argparse

from dal_monte_2022_analysis.behav.analysis.fix_cross_correlation_leader_follower import (
    run_fix_cross_correlation_leader_follower_analysis,
)
from dal_monte_2022_analysis.behav.analysis.fix_cross_correlation_leader_follower_cli import (
    apply_leader_follower_cli_overrides,
    build_leader_follower_settings_from_config,
)


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

    settings = build_leader_follower_settings_from_config(
        dataset_cfg_path=args.dataset_cfg,
        fix_cross_correlation_cfg_path=args.face_fix_cross_correlation_cfg,
        default_fixation_label="face",
        default_tag="face",
    )
    settings = apply_leader_follower_cli_overrides(
        settings,
        tie_epsilon=args.tie_epsilon,
    )
    run_fix_cross_correlation_leader_follower_analysis(settings)


if __name__ == "__main__":
    main()
