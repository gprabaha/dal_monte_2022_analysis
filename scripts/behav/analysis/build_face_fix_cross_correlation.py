"""Build fixation cross-correlation tables (observed and shuffled modes)."""

import argparse
from pathlib import Path

from dal_monte_2022_analysis.behav.analysis.fix_cross_correlation import (
    FixCrossCorrelationSettings,
    build_within_session_pair_tasks,
    collate_within_session_shuffle_results,
    process_and_save_within_session_shuffle_pair,
    run_fix_cross_correlation_analysis,
)
from dal_monte_2022_analysis.config.load import (
    load_face_fix_cross_correlation_config,
    load_hpc_config,
)
from dal_monte_2022_analysis.utils.hpc_utils import (
    generate_fix_crosscorr_shuffle_job_file,
    submit_dsq_array_job,
    track_job_completion,
)
from dal_monte_2022_analysis.utils.paths import normalize_fix_crosscorr_time_scope


def _build_settings(args) -> FixCrossCorrelationSettings:
    """Construct analysis settings from config + CLI overrides."""
    cfg = load_face_fix_cross_correlation_config(args.face_fix_cross_correlation_cfg)
    fixation_label = cfg.get("fixation_label", cfg.get("face_label", "face"))
    settings = FixCrossCorrelationSettings(
        cfg_path=args.dataset_cfg,
        input_modality=cfg.get("input_modality", "fixation_binary_vectors"),
        fixation_label=fixation_label,
        output_subdir=cfg.get(
            "crosscorr_output_subdir",
            cfg.get("output_subdir", "crosscorr_outputs"),
        ),
        within_filename=cfg.get("within_filename"),
        cross_filename=cfg.get("cross_filename"),
        lags_filename=cfg.get("lags_filename"),
        max_lag=cfg.get("max_lag", 60000),
        time_scope=normalize_fix_crosscorr_time_scope(cfg.get("time_scope", "whole")),
        interactive_modality=cfg.get("interactive_modality", "interactive_periods"),
        interactive_state_label=cfg.get("interactive_state_label", "interactive"),
        cross_pairs_max=cfg.get("cross_pairs_max"),
        cross_pairs_seed=cfg.get("cross_pairs_seed", 13),
        cross_exclude_same_session=cfg.get("cross_exclude_same_session", True),
        cross_exclude_same_date=cfg.get("cross_exclude_same_date", False),
        parallelize_across_crosscorr_pairs=cfg.get(
            "parallelize_across_crosscorr_pairs",
            False,
        ),
        shuffle_output_filename=cfg.get("shuffle_output_filename"),
        shuffle_pairs_subdir=cfg.get("shuffle_pairs_subdir", "within_session_shuffle_pair_results"),
        shuffle_n_shuffles=cfg.get("shuffle_n_shuffles", 1000),
        shuffle_stringent=cfg.get("shuffle_stringent", True),
        shuffle_seed=cfg.get("shuffle_seed", 13),
        shuffle_parallelize_within_pair=cfg.get("shuffle_parallelize_within_pair", True),
        shuffle_log_every=cfg.get("shuffle_log_every", 100),
        test_single=cfg.get("test_single", False),
    )

    if args.test_single:
        settings.test_single = True
    if args.max_cross_pairs is not None:
        settings.cross_pairs_max = args.max_cross_pairs
    if args.exclude_same_date:
        settings.cross_exclude_same_date = True
    if args.include_same_session:
        settings.cross_exclude_same_session = False
    if args.parallelize_across_crosscorr_pairs:
        settings.parallelize_across_crosscorr_pairs = True
    if args.max_lag is not None:
        settings.max_lag = max(0, args.max_lag)
    if args.time_scope is not None:
        settings.time_scope = normalize_fix_crosscorr_time_scope(args.time_scope)
    if args.shuffle_n_shuffles is not None:
        settings.shuffle_n_shuffles = max(0, args.shuffle_n_shuffles)
    if args.shuffle_seed is not None:
        settings.shuffle_seed = args.shuffle_seed
    if args.shuffle_log_every is not None:
        settings.shuffle_log_every = max(1, args.shuffle_log_every)
    if args.shuffle_non_stringent:
        settings.shuffle_stringent = False
    if args.shuffle_no_within_pair_parallel:
        settings.shuffle_parallelize_within_pair = False

    return settings


def _run_shuffle_submit_hpc(settings: FixCrossCorrelationSettings, args) -> None:
    """Submit one-array-task-per-within-pair shuffle jobs, then collate."""
    hpc_cfg = load_hpc_config(args.hpc_cfg)
    tasks = build_within_session_pair_tasks(settings)
    if not tasks:
        print("No within-session pairs found for shuffled cross-correlation.")
        return

    worker_script = Path(hpc_cfg["worker_script_path"]).resolve()
    dataset_cfg_path = str(Path(args.dataset_cfg).resolve())
    fix_cfg_path = str(Path(args.face_fix_cross_correlation_cfg).resolve())
    generate_fix_crosscorr_shuffle_job_file(
        tasks=tasks,
        job_file_path=hpc_cfg["job_file_path"],
        worker_script=worker_script,
        env_name=hpc_cfg["env_name"],
        dataset_cfg_path=dataset_cfg_path,
        fix_crosscorr_cfg_path=fix_cfg_path,
        time_scope=settings.time_scope,
    )

    job_id = submit_dsq_array_job(
        job_file_path=hpc_cfg["job_file_path"],
        sbatch_script_path=hpc_cfg["sbatch_script_path"],
        log_dir=hpc_cfg["log_dir"],
        job_name=hpc_cfg["job_name"],
        partition=hpc_cfg["partition"],
        cpus_per_task=hpc_cfg["cpus_per_task"],
        mem_per_cpu=hpc_cfg["mem_per_cpu"],
        time_limit=hpc_cfg["time_limit"],
    )
    track_job_completion(job_id)
    collate_within_session_shuffle_results(settings)


def main():
    """Parse CLI args and run requested cross-correlation mode."""
    parser = argparse.ArgumentParser(
        description="Build fixation cross-correlation tables.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--face-fix-cross-correlation-cfg",
        default="configs/face_fix_cross_correlation.yaml",
    )
    parser.add_argument(
        "--mode",
        default="observed",
        choices=[
            "observed",
            "shuffle_submit_hpc",
            "shuffle_worker",
            "shuffle_collate",
            "shuffle_local",
        ],
    )
    parser.add_argument(
        "--hpc-cfg",
        default="configs/hpc_face_fix_cross_correlation_shuffle.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--no-cross", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--max-cross-pairs", type=int, default=None)
    parser.add_argument("--exclude-same-date", action="store_true")
    parser.add_argument("--include-same-session", action="store_true")
    parser.add_argument("--parallelize-across-crosscorr-pairs", action="store_true")
    parser.add_argument("--max-lag", type=int, default=None)
    parser.add_argument(
        "--time-scope",
        type=str,
        default=None,
        choices=["whole", "interactive", "non_interactive"],
    )
    parser.add_argument("--shuffle-n-shuffles", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--shuffle-log-every", type=int, default=None)
    parser.add_argument("--shuffle-non-stringent", action="store_true")
    parser.add_argument("--shuffle-no-within-pair-parallel", action="store_true")

    args = parser.parse_args()
    settings = _build_settings(args)

    if args.mode == "observed":
        run_fix_cross_correlation_analysis(
            settings,
            compute_cross=not args.no_cross,
        )
        return

    if args.mode == "shuffle_submit_hpc":
        _run_shuffle_submit_hpc(settings, args)
        return

    if args.mode == "shuffle_worker":
        if args.date is None or args.session is None:
            raise RuntimeError("--mode shuffle_worker requires --date and --session.")
        process_and_save_within_session_shuffle_pair(
            settings=settings,
            date=args.date,
            session=args.session,
        )
        return

    if args.mode == "shuffle_collate":
        collate_within_session_shuffle_results(settings)
        return

    if args.mode == "shuffle_local":
        tasks = build_within_session_pair_tasks(settings)
        if not tasks:
            print("No within-session pairs found for shuffled cross-correlation.")
            return
        for date, session in tasks:
            process_and_save_within_session_shuffle_pair(
                settings=settings,
                date=date,
                session=session,
            )
        collate_within_session_shuffle_results(settings)
        return


if __name__ == "__main__":
    main()
