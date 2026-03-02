"""Plot monkey-level leader-vs-follower pupil violins for face fixation analysis."""

import argparse

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.behav.plotting.fix_cross_correlation_leader_follower_monkey_role import (
    LeaderFollowerMonkeyRolePupilPlotSettings,
    plot_leader_follower_monkey_role_pupil_violin,
)


def main():
    """Parse CLI args and plot monkey-role pupil violins."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot monkey-level leader-vs-follower pupil violins "
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

    cfg = load_config(args.face_fix_cross_correlation_cfg)
    if not bool(cfg.get("leader_follower_monkey_role_pupil_make_violin_plot", True)):
        print("[plot] skipping monkey-role pupil violin (disabled in config).")
        return

    cross_correlation_subdir = cfg.get(
        "cross_correlation_output_subdir",
        cfg.get("crosscorr_output_subdir", cfg.get("output_subdir", "fix_cross_correlation")),
    )
    leader_follower_subdir = cfg.get(
        "leader_follower_output_subdir",
        f"{cross_correlation_subdir}/leader_follower",
    )
    settings = LeaderFollowerMonkeyRolePupilPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        analysis_subdir=leader_follower_subdir,
        monkey_role_session_filename=cfg.get(
            "leader_follower_monkey_role_pupil_session_filename",
            "within_session_face_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv",
        ),
        monkey_role_summary_filename=cfg.get(
            "leader_follower_monkey_role_pupil_summary_filename",
            "summary_face_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv",
        ),
        output_filename=cfg.get(
            "leader_follower_monkey_role_pupil_violin_filename",
            "summary_face_fix_crosscorr_leader_follower_pupil_by_monkey_role_violin.pdf",
        ),
        max_samples_per_role=int(
            cfg.get("leader_follower_monkey_role_pupil_plot_max_samples_per_role", 20000)
        ),
    )

    out_path = plot_leader_follower_monkey_role_pupil_violin(settings)
    print(f"[plot] wrote leader-follower monkey-role pupil violin: {out_path}")


if __name__ == "__main__":
    main()
