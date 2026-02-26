"""Build fixation binary vectors from processed fixation events."""

import argparse

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.behav.features.fixation_binary_vectors import (
    DEFAULT_ROI_GROUPS,
    FixationBinaryVectorSettings,
    process_fixation_binary_vectors_for_row,
    run_fixation_binary_vector_build,
)


def main():
    """Parse CLI args and run fixation binary vector creation."""
    parser = argparse.ArgumentParser(description="Build fixation binary vectors.")
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--fixation-vector-cfg", default="configs/fixation_binary_vectors.yaml")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")

    args = parser.parse_args()

    vectors_cfg = load_config(args.fixation_vector_cfg)

    settings = FixationBinaryVectorSettings(
        cfg_path=args.dataset_cfg,
        fixations_modality=vectors_cfg.get("fixations_modality", "fixations"),
        timeline_modality=vectors_cfg.get("timeline_modality", "neural_timeline"),
        output_modality=vectors_cfg.get("output_modality", "fixation_binary_vectors"),
        roi_groups=vectors_cfg.get("roi_groups", DEFAULT_ROI_GROUPS),
        agent_roi_groups=vectors_cfg.get("agent_roi_groups"),
        use_parallel=vectors_cfg.get("use_parallel", False),
        test_single=vectors_cfg.get("test_single", False),
        agents=vectors_cfg.get("agents"),
    )

    if args.use_parallel:
        settings.use_parallel = True
    if args.test_single:
        settings.test_single = True

    if args.date and args.session and args.agent:
        row = {"date": args.date, "session": args.session}
        process_fixation_binary_vectors_for_row(settings, row, args.agent)
        return

    run_fixation_binary_vector_build(
        settings,
        use_parallel=settings.use_parallel,
        test_single=settings.test_single,
    )


if __name__ == "__main__":
    main()
