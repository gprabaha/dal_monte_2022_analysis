"""Plot leader-aligned face cross-correlation comparisons across scopes."""

import argparse

from dal_monte_2022_analysis.config.load import load_face_fix_cross_correlation_config
from dal_monte_2022_analysis.plotting.fix_cross_correlation_leader_follower import (
    LeaderFollowerCrossCorrComparisonPlotSettings,
    plot_leader_follower_crosscorr_comparisons,
)


def main():
    """Parse CLI args and render leader-aligned cross-correlation comparison figures."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot leader-aligned face cross-correlation traces (whole/interactive/"
            "non-interactive) comparing observed vs cross-session and observed vs shuffled "
            "controls, for session/day/pair leader bases."
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
    crosscorr_subdir = cfg.get("crosscorr_output_subdir", cfg.get("output_subdir", "crosscorr_outputs"))
    leader_follower_subdir = cfg.get(
        "leader_follower_output_subdir",
        f"{crosscorr_subdir}/leader_follower",
    )

    settings = LeaderFollowerCrossCorrComparisonPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        crosscorr_analysis_subdir=crosscorr_subdir,
        leader_follower_subdir=leader_follower_subdir,
        fixation_label=cfg.get("fixation_label", "face"),
        scopes=tuple(cfg.get("lf_plot_scopes", ("whole", "interactive", "non_interactive"))),
        leader_bases=tuple(cfg.get("lf_plot_leader_bases", ("session", "date", "pair"))),
        leader_reference_scope=cfg.get("leader_follower_time_scope", "whole"),
        leader_session_filename=cfg.get(
            "leader_follower_session_filename",
            "within_session_face_fix_crosscorr_leader_follower.pkl",
        ),
        leader_date_filename=cfg.get(
            "leader_follower_date_summary_filename",
            "date_summary_face_fix_crosscorr_leader_follower.pkl",
        ),
        leader_pair_filename=cfg.get(
            "leader_follower_pair_summary_filename",
            "pair_summary_face_fix_crosscorr_leader_follower.pkl",
        ),
        significance_alpha=float(cfg.get("lf_plot_significance_alpha", 0.05)),
        lag_sampling_rate_hz=float(cfg.get("lf_plot_lag_sampling_rate_hz", 1000.0)),
        max_plot_points=int(cfg.get("lf_plot_max_points", 4000)),
        max_sig_markers=int(cfg.get("lf_plot_max_sig_markers", 1000)),
        rasterize_bands=bool(cfg.get("lf_plot_rasterize_bands", True)),
        rasterize_sig_markers=bool(cfg.get("lf_plot_rasterize_sig_markers", True)),
        ttest_parallel=bool(cfg.get("lf_plot_ttest_parallel", True)),
        ttest_parallel_workers=cfg.get("lf_plot_ttest_parallel_workers", None),
        ttest_parallel_min_lags=int(cfg.get("lf_plot_ttest_parallel_min_lags", 4000)),
        ttest_parallel_chunk_size=int(cfg.get("lf_plot_ttest_parallel_chunk_size", 4096)),
        output_subdir=cfg.get("lf_plot_output_subdir", "plots/leader_follower"),
        observed_vs_cross_filename_template=cfg.get(
            "lf_observed_vs_cross_filename_template",
            "observed_vs_cross_session_face_leader_follower_basis={basis}.pdf",
        ),
        observed_vs_shuffle_filename_template=cfg.get(
            "lf_observed_vs_shuffle_filename_template",
            "observed_vs_shuffle_face_leader_follower_basis={basis}.pdf",
        ),
    )

    out_paths = plot_leader_follower_crosscorr_comparisons(settings)
    for path in out_paths:
        print(f"[plot] wrote leader-follower crosscorr figure: {path}")


if __name__ == "__main__":
    main()
