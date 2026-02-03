"""Build joint face fixation density vectors from per-agent densities."""

import argparse

from dal_monte_2022_analysis.analysis.joint_fixation_density import (
    JointFixationDensitySettings,
    process_joint_face_density_for_row,
    run_joint_face_density_build,
)
from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_joint_fixation_density_config,
)
from dal_monte_2022_analysis.utils.paths import build_processed_data_path


def main():
    """Parse CLI args and run joint face fixation density creation."""
    parser = argparse.ArgumentParser(
        description="Build joint face fixation density vectors.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--joint-density-cfg",
        default="configs/joint_face_fixation_density.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")

    args = parser.parse_args()

    joint_cfg = load_joint_fixation_density_config(args.joint_density_cfg)
    settings = JointFixationDensitySettings(
        cfg_path=args.dataset_cfg,
        input_modality=joint_cfg.get("input_modality", "fixation_density_vectors"),
        output_modality=joint_cfg.get("output_modality", "joint_face_fixation_density"),
        face_label=joint_cfg.get("face_label", "face"),
        normalize=joint_cfg.get("normalize", True),
        use_parallel=joint_cfg.get("use_parallel", False),
        test_single=joint_cfg.get("test_single", False),
    )

    if args.use_parallel:
        settings.use_parallel = True
    if args.test_single:
        settings.test_single = True

    if args.date and args.session:
        row = {"date": args.date, "session": args.session}
        cfg = load_dataset_config(args.dataset_cfg)
        m1_path = build_processed_data_path(cfg, row, settings.input_modality, "m1")
        m2_path = build_processed_data_path(cfg, row, settings.input_modality, "m2")
        if not m1_path.exists() or not m2_path.exists():
            raise FileNotFoundError(
                f"Missing fixation density inputs: {m1_path} or {m2_path}"
            )
        process_joint_face_density_for_row(settings, row, m1_path=m1_path, m2_path=m2_path)
        return

    run_joint_face_density_build(
        settings,
        use_parallel=settings.use_parallel,
        test_single=settings.test_single,
    )


if __name__ == "__main__":
    main()
