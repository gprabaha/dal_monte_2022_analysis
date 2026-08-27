"""Build face-fixation gap distribution tables."""

import argparse

from dal_monte_2022_analysis.behav.analysis.face_fixation_gap_distributions import (
    FaceFixationGapDistributionSettings,
    run_face_fixation_gap_distribution_analysis,
)
from dal_monte_2022_analysis.config.load import load_config


def main():
    """Parse CLI args and build the gap-distribution tables."""
    parser = argparse.ArgumentParser(
        description="Build face-fixation gap distribution tables.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--face-fixation-gap-cfg",
        default="configs/face_fixation_gap_distribution.yaml",
    )
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.face_fixation_gap_cfg)
    settings = FaceFixationGapDistributionSettings(
        cfg_path=args.dataset_cfg,
        fixations_modality=cfg.get("fixations_modality", "fixations"),
        interactive_modality=cfg.get("interactive_modality", "interactive_periods"),
        fixation_label=cfg.get("fixation_label", "face"),
        output_subdir=cfg.get("output_subdir", "face_fixation_gap_distributions"),
        m1_output_filename=cfg.get(
            "m1_output_filename",
            "within_session_m1_face_fixation_gap_distribution.csv",
        ),
        m1_m2_output_filename=cfg.get(
            "m1_m2_output_filename",
            "within_session_interactive_m1_m2_face_fixation_gap_distribution.csv",
        ),
        filter_summary_filename=cfg.get(
            "filter_summary_filename",
            "face_fixation_gap_distribution_filter_summary.csv",
        ),
        interactive_state_label=cfg.get("interactive_state_label", "interactive"),
        non_interactive_state_label=cfg.get(
            "non_interactive_state_label",
            "non_interactive",
        ),
        max_pair_gap_ms=cfg.get("max_pair_gap_ms", 5000.0),
        sample_rate_hz=float(cfg.get("sample_rate_hz", 1000.0)),
        agent_m1=cfg.get("agent_m1", "m1"),
        agent_m2=cfg.get("agent_m2", "m2"),
        test_single=bool(cfg.get("test_single", False)),
    )
    if cfg.get("roi_groups") is not None:
        settings.roi_groups = cfg.get("roi_groups")
    if cfg.get("agent_roi_groups") is not None:
        settings.agent_roi_groups = cfg.get("agent_roi_groups")
    if args.test_single:
        settings.test_single = True

    m1_path, m1_m2_path, summary_path = run_face_fixation_gap_distribution_analysis(settings)
    print(f"[analysis] wrote m1 face-fixation gap distribution: {m1_path}")
    print(f"[analysis] wrote interactive m1-m2 face-fixation gap distribution: {m1_m2_path}")
    print(f"[analysis] wrote gap-filter summary: {summary_path}")


if __name__ == "__main__":
    main()
