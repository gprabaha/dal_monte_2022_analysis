"""Fit Poisson-HSMMs for joint face-fixation observations."""

import argparse

from dal_monte_2022_analysis.config.load import load_face_fixation_hsmm_config
from dal_monte_2022_analysis.modeling.face_fixation_hsmm import (
    FaceFixationHSMMSettings,
    run_face_fixation_hsmm_analysis,
)


def main():
    """Parse CLI args and run face-fixation HSMM fitting."""
    parser = argparse.ArgumentParser(
        description="Fit 2-state Poisson-HSMMs to joint face-fixation vectors.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--face-fixation-hsmm-cfg",
        default="configs/face_fixation_hsmm.yaml",
    )
    parser.add_argument(
        "--grouping",
        choices=["session", "day", "pair", "global"],
        default=None,
    )
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--max-duration", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-self-transitions", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    cfg = load_face_fixation_hsmm_config(args.face_fixation_hsmm_cfg)
    settings = FaceFixationHSMMSettings(
        cfg_path=args.dataset_cfg,
        input_modality=cfg.get("input_modality", "fixation_binary_vectors"),
        fixation_label=cfg.get("fixation_label", "face"),
        output_subdir=cfg.get("output_subdir", "face_fixation_hsmm"),
        grouping=cfg.get("grouping", "session"),
        n_hidden_states=cfg.get("n_hidden_states", 2),
        max_duration=cfg.get("max_duration", 300),
        n_iter=cfg.get("n_iter", 50),
        tol=cfg.get("tol", 1e-3),
        n_init=cfg.get("n_init", 3),
        seed=cfg.get("seed", 13),
        allow_self_transitions=cfg.get("allow_self_transitions", False),
        transition_pseudocount=cfg.get("transition_pseudocount", 1.0),
        emission_pseudocount=cfg.get("emission_pseudocount", 1.0),
        group_summary_filename=cfg.get(
            "group_summary_filename",
            "face_fixation_hsmm_group_summary.csv",
        ),
        session_summary_filename=cfg.get(
            "session_summary_filename",
            "face_fixation_hsmm_session_summary.csv",
        ),
        segments_filename=cfg.get(
            "segments_filename",
            "face_fixation_hsmm_segments.csv",
        ),
        fits_filename=cfg.get("fits_filename", "face_fixation_hsmm_fits.pkl"),
        dates=cfg.get("dates"),
        sessions=cfg.get("sessions"),
        test_single=cfg.get("test_single", False),
    )

    if args.grouping is not None:
        settings.grouping = args.grouping
    if args.date:
        settings.dates = args.date
    if args.session:
        settings.sessions = args.session
    if args.max_duration is not None:
        settings.max_duration = args.max_duration
    if args.n_iter is not None:
        settings.n_iter = args.n_iter
    if args.n_init is not None:
        settings.n_init = args.n_init
    if args.seed is not None:
        settings.seed = args.seed
    if args.allow_self_transitions:
        settings.allow_self_transitions = True
    if args.test_single:
        settings.test_single = True

    run_face_fixation_hsmm_analysis(settings)


if __name__ == "__main__":
    main()
