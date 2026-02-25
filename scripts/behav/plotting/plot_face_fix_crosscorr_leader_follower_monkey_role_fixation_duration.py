"""Plot monkey-level leader-vs-follower fixation-duration violins for face analysis."""

import argparse

from dal_monte_2022_analysis.config.load import load_face_fix_cross_correlation_config
from dal_monte_2022_analysis.behav.plotting.fix_crosscorr_leader_follower import (
    LeaderFollowerMonkeyRoleFixationDurationPlotSettings,
    plot_leader_follower_monkey_role_fixation_duration_violin,
)


def main():
    """Parse CLI args and plot monkey-role fixation-duration violins."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot monkey-level leader-vs-follower fixation-duration violins "
            "from face leader-follower analysis outputs."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument(
        "--face-fix-cross-correlation-cfg",
        default="configs/face_fix_cross_correlation.yaml",
    )
    args = parser.parse_args()

    cfg = load_face_fix_cross_correlation_config(args.face_fix_cross_correlation_cfg)
    make_plot = cfg.get(
        "leader_follower_monkey_role_fixation_duration_make_violin_plot",
        cfg.get("leader_follower_monkey_role_fixation_count_make_violin_plot", True),
    )
    if not bool(make_plot):
        print("[plot] skipping monkey-role fixation-duration violin (disabled in config).")
        return

    crosscorr_subdir = cfg.get("crosscorr_output_subdir", cfg.get("output_subdir", "fix_cross_correlation"))
    leader_follower_subdir = cfg.get(
        "leader_follower_output_subdir",
        f"{crosscorr_subdir}/leader_follower",
    )
    settings = LeaderFollowerMonkeyRoleFixationDurationPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        analysis_subdir=leader_follower_subdir,
        monkey_role_session_filename=cfg.get(
            "leader_follower_monkey_role_fixation_duration_session_filename",
            cfg.get(
                "leader_follower_monkey_role_fixation_count_session_filename",
                "within_session_face_fix_crosscorr_leader_follower_fixation_duration_by_monkey_role.csv",
            ),
        ),
        monkey_role_summary_filename=cfg.get(
            "leader_follower_monkey_role_fixation_duration_summary_filename",
            cfg.get(
                "leader_follower_monkey_role_fixation_count_summary_filename",
                "summary_face_fix_crosscorr_leader_follower_fixation_duration_by_monkey_role.csv",
            ),
        ),
        output_filename=cfg.get(
            "leader_follower_monkey_role_fixation_duration_violin_filename",
            cfg.get(
                "leader_follower_monkey_role_fixation_count_violin_filename",
                "summary_face_fix_crosscorr_leader_follower_fixation_duration_by_monkey_role_violin.pdf",
            ),
        ),
        max_samples_per_role=int(
            cfg.get(
                "leader_follower_monkey_role_fixation_duration_plot_max_samples_per_role",
                cfg.get(
                    "leader_follower_monkey_role_fixation_count_plot_max_samples_per_role",
                    20000,
                ),
            )
        ),
    )

    out_path = plot_leader_follower_monkey_role_fixation_duration_violin(settings)
    print(f"[plot] wrote leader-follower monkey-role fixation-duration violin: {out_path}")


if __name__ == "__main__":
    main()
