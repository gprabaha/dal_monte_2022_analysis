"""Build interactive periods from joint face fixation density."""

import argparse

from dal_monte_2022_analysis.behav.features.interactive_periods import (
    InteractivePeriodsSettings,
    process_interactive_periods_for_row,
    run_interactive_periods_build,
)
from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_interactive_periods_config,
)
from dal_monte_2022_analysis.utils.paths import build_processed_data_path


def main():
    """Parse CLI args and run interactive period creation."""
    parser = argparse.ArgumentParser(
        description="Build interactive periods from joint face fixation density.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--interactive-periods-cfg",
        default="configs/interactive_periods.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")

    args = parser.parse_args()

    cfg = load_interactive_periods_config(args.interactive_periods_cfg)
    settings = InteractivePeriodsSettings(
        cfg_path=args.dataset_cfg,
        input_modality=cfg.get("input_modality", "joint_face_fixation_density"),
        output_modality=cfg.get("output_modality", "interactive_periods"),
        threshold_factor=cfg.get("threshold_factor", 0.34),
        include_low=cfg.get("include_low", True),
        high_label=cfg.get("high_label", "interactive"),
        low_label=cfg.get("low_label", "non_interactive"),
        use_parallel=cfg.get("use_parallel", False),
        test_single=cfg.get("test_single", False),
    )

    if args.use_parallel:
        settings.use_parallel = True
    if args.test_single:
        settings.test_single = True

    if args.date and args.session:
        row = {"date": args.date, "session": args.session}
        dataset_cfg = load_dataset_config(args.dataset_cfg)
        density_path = build_processed_data_path(
            dataset_cfg,
            row,
            settings.input_modality,
            None,
        )
        if not density_path.exists():
            raise FileNotFoundError(f"Missing joint density input: {density_path}")
        process_interactive_periods_for_row(settings, row, density_path=density_path)
        return

    run_interactive_periods_build(
        settings,
        use_parallel=settings.use_parallel,
        test_single=settings.test_single,
    )


if __name__ == "__main__":
    main()
