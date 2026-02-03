"""Build fixation density vectors from fixation binary vectors."""

import argparse

from dal_monte_2022_analysis.config.load import load_fixation_density_config
from dal_monte_2022_analysis.features.fixation_density import (
    DEFAULT_ROI_GROUPS,
    FixationDensitySettings,
    process_fixation_density_for_row,
    run_fixation_density_build,
)


def main():
    """Parse CLI args and run fixation density creation."""
    parser = argparse.ArgumentParser(description="Build fixation density vectors.")
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--fixation-density-cfg", default="configs/fixation_density.yaml")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")

    args = parser.parse_args()

    density_cfg = load_fixation_density_config(args.fixation_density_cfg)

    settings = FixationDensitySettings(
        cfg_path=args.dataset_cfg,
        fixations_modality=density_cfg.get("fixations_modality", "fixations"),
        fixation_vectors_modality=density_cfg.get(
            "fixation_vectors_modality", "fixation_binary_vectors"
        ),
        output_modality=density_cfg.get("output_modality", "fixation_density_vectors"),
        roi_groups=density_cfg.get("roi_groups", DEFAULT_ROI_GROUPS),
        agent_roi_groups=density_cfg.get("agent_roi_groups"),
        binwidth_method=density_cfg.get("binwidth_method", "mean"),
        sigma_method=density_cfg.get("sigma_method", "binwidth"),
        kernel_width_factor=density_cfg.get("kernel_width_factor", 6.0),
        min_kernel_width=density_cfg.get("min_kernel_width", 3),
        sigma_floor=density_cfg.get("sigma_floor", 1.0),
        truncate_sigmas=density_cfg.get("truncate_sigmas", 3.0),
        inter_fixation_fallback=density_cfg.get(
            "inter_fixation_fallback", "fixation_duration"
        ),
        normalize=density_cfg.get("normalize", True),
        use_parallel=density_cfg.get("use_parallel", False),
        test_single=density_cfg.get("test_single", False),
        agents=density_cfg.get("agents"),
    )

    if args.use_parallel:
        settings.use_parallel = True
    if args.test_single:
        settings.test_single = True

    if args.date and args.session and args.agent:
        row = {"date": args.date, "session": args.session}
        process_fixation_density_for_row(settings, row, args.agent)
        return

    run_fixation_density_build(
        settings,
        use_parallel=settings.use_parallel,
        test_single=settings.test_single,
    )


if __name__ == "__main__":
    main()
