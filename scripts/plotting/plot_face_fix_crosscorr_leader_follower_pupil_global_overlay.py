"""Plot pooled leader-vs-follower pupil violin with per-monkey mean overlays."""

import argparse

from dal_monte_2022_analysis.config.load import load_face_fix_cross_correlation_config
from dal_monte_2022_analysis.plotting.fix_crosscorr_leader_follower import (
    LeaderFollowerPupilGlobalOverlayPlotSettings,
    plot_leader_follower_pupil_global_overlay_violin,
)


def main():
    """Parse CLI args and plot pooled pupil violin with monkey overlays."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot pooled leader-vs-follower pupil distributions across all sessions "
            "and overlay each monkey's leader/follower mean trend."
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
    if not bool(cfg.get("leader_follower_pupil_global_overlay_make_plot", True)):
        print("[plot] skipping leader-vs-follower pooled pupil overlay plot (disabled in config).")
        return

    crosscorr_subdir = cfg.get("crosscorr_output_subdir", cfg.get("output_subdir", "fix_cross_correlation"))
    leader_follower_subdir = cfg.get(
        "leader_follower_output_subdir",
        f"{crosscorr_subdir}/leader_follower",
    )
    settings = LeaderFollowerPupilGlobalOverlayPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        analysis_subdir=leader_follower_subdir,
        monkey_role_session_filename=cfg.get(
            "leader_follower_monkey_role_pupil_session_filename",
            "within_session_face_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv",
        ),
        monkey_role_session_raw_filename=cfg.get(
            "leader_follower_monkey_role_pupil_session_raw_filename",
            "within_session_face_fix_crosscorr_leader_follower_pupil_by_monkey_role_raw.pkl",
        ),
        monkey_role_summary_filename=cfg.get(
            "leader_follower_monkey_role_pupil_summary_filename",
            "summary_face_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv",
        ),
        output_filename=cfg.get(
            "leader_follower_pupil_global_overlay_filename",
            (
                "global_face_fix_crosscorr_leader_follower_pupil_"
                "leader_vs_follower_with_monkey_overlay.pdf"
            ),
        ),
        max_samples_per_role=int(
            cfg.get("leader_follower_pupil_global_overlay_plot_max_samples_per_role", 50000)
        ),
        show_monkey_legend=bool(cfg.get("leader_follower_pupil_global_overlay_show_legend", False)),
        alpha=float(cfg.get("leader_follower_pupil_test_alpha", 0.05)),
    )

    out_path = plot_leader_follower_pupil_global_overlay_violin(settings)
    print(f"[plot] wrote leader-vs-follower pooled pupil overlay plot: {out_path}")


if __name__ == "__main__":
    main()
