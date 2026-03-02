"""Shared CLI helpers for fixation cross-correlation entrypoint scripts."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from dal_monte_2022_analysis.behav.analysis.fix_cross_correlation import (
    FixCrossCorrelationSettings,
    build_within_session_pair_tasks,
    collate_within_session_shuffle_results,
    process_and_save_within_session_shuffle_pair,
    run_fix_cross_correlation_analysis,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.hpc.jobs import (
    generate_fix_cross_correlation_shuffle_job_file,
    submit_dsq_array_job,
    track_job_completion,
)
from dal_monte_2022_analysis.utils.paths import normalize_fix_cross_correlation_time_scope


def build_fix_cross_correlation_settings_from_config(
    *,
    dataset_cfg_path: str,
    fix_cross_correlation_cfg_path: Optional[str] = None,
    fix_crosscorr_cfg_path: Optional[str] = None,
    default_fixation_label: str,
    default_shuffle_pairs_subdir: str,
) -> FixCrossCorrelationSettings:
    """Build cross-correlation settings from dataset + task config paths."""
    if fix_cross_correlation_cfg_path is None and fix_crosscorr_cfg_path is not None:
        warnings.warn(
            (
                "fix_crosscorr_cfg_path is deprecated; "
                "use fix_cross_correlation_cfg_path instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
    cfg_path = fix_cross_correlation_cfg_path or fix_crosscorr_cfg_path
    if cfg_path is None:
        raise ValueError("Expected one of fix_cross_correlation_cfg_path or fix_crosscorr_cfg_path.")
    cfg = load_config(cfg_path)
    fixation_label = cfg.get("fixation_label", cfg.get("face_label", default_fixation_label))
    return FixCrossCorrelationSettings(
        cfg_path=dataset_cfg_path,
        input_modality=cfg.get("input_modality", "fixation_binary_vectors"),
        fixation_label=fixation_label,
        output_subdir=cfg.get(
            "cross_correlation_output_subdir",
            cfg.get(
                "crosscorr_output_subdir",
                cfg.get("output_subdir", "cross_correlation_outputs"),
            ),
        ),
        within_filename=cfg.get("within_filename"),
        cross_filename=cfg.get("cross_filename"),
        lags_filename=cfg.get("lags_filename"),
        max_lag=cfg.get("max_lag", 60000),
        time_scope=normalize_fix_cross_correlation_time_scope(cfg.get("time_scope", "whole")),
        interactive_modality=cfg.get("interactive_modality", "interactive_periods"),
        interactive_state_label=cfg.get("interactive_state_label", "interactive"),
        cross_pairs_max=cfg.get("cross_pairs_max"),
        cross_pairs_seed=cfg.get("cross_pairs_seed", 13),
        cross_exclude_same_session=cfg.get("cross_exclude_same_session", True),
        cross_exclude_same_date=cfg.get("cross_exclude_same_date", False),
        parallelize_across_cross_correlation_pairs=cfg.get(
            "parallelize_across_cross_correlation_pairs",
            cfg.get("parallelize_across_crosscorr_pairs", False),
        ),
        shuffle_output_filename=cfg.get("shuffle_output_filename"),
        shuffle_pairs_subdir=cfg.get("shuffle_pairs_subdir", default_shuffle_pairs_subdir),
        shuffle_n_shuffles=cfg.get("shuffle_n_shuffles", 1000),
        shuffle_stringent=cfg.get("shuffle_stringent", True),
        shuffle_seed=cfg.get("shuffle_seed", 13),
        shuffle_parallelize_within_pair=cfg.get("shuffle_parallelize_within_pair", True),
        shuffle_log_every=cfg.get("shuffle_log_every", 100),
        test_single=cfg.get("test_single", False),
    )


def apply_fix_cross_correlation_cli_overrides(
    settings: FixCrossCorrelationSettings,
    *,
    test_single: bool = False,
    max_cross_pairs: Optional[int] = None,
    exclude_same_date: bool = False,
    include_same_session: bool = False,
    parallelize_across_cross_correlation_pairs: bool = False,
    parallelize_across_crosscorr_pairs: bool = False,
    max_lag: Optional[int] = None,
    time_scope: Optional[str] = None,
    shuffle_n_shuffles: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    shuffle_log_every: Optional[int] = None,
    shuffle_non_stringent: bool = False,
    shuffle_no_within_pair_parallel: bool = False,
) -> FixCrossCorrelationSettings:
    """Apply CLI overrides onto settings in-place and return settings."""
    if test_single:
        settings.test_single = True
    if max_cross_pairs is not None:
        settings.cross_pairs_max = max_cross_pairs
    if exclude_same_date:
        settings.cross_exclude_same_date = True
    if include_same_session:
        settings.cross_exclude_same_session = False
    if parallelize_across_crosscorr_pairs:
        warnings.warn(
            (
                "parallelize_across_crosscorr_pairs is deprecated; "
                "use parallelize_across_cross_correlation_pairs instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
    if parallelize_across_cross_correlation_pairs or parallelize_across_crosscorr_pairs:
        settings.parallelize_across_cross_correlation_pairs = True
    if max_lag is not None:
        settings.max_lag = max(0, int(max_lag))
    if time_scope is not None:
        settings.time_scope = normalize_fix_cross_correlation_time_scope(time_scope)
    if shuffle_n_shuffles is not None:
        settings.shuffle_n_shuffles = max(0, int(shuffle_n_shuffles))
    if shuffle_seed is not None:
        settings.shuffle_seed = int(shuffle_seed)
    if shuffle_log_every is not None:
        settings.shuffle_log_every = max(1, int(shuffle_log_every))
    if shuffle_non_stringent:
        settings.shuffle_stringent = False
    if shuffle_no_within_pair_parallel:
        settings.shuffle_parallelize_within_pair = False
    return settings


def run_fix_cross_correlation_shuffle_submit_hpc(
    settings: FixCrossCorrelationSettings,
    *,
    hpc_cfg_path: str,
    dataset_cfg_path: str,
    fix_cross_correlation_cfg_path: Optional[str] = None,
    fix_crosscorr_cfg_path: Optional[str] = None,
) -> None:
    """Submit one-array-task-per-within-pair shuffle jobs, then collate."""
    hpc_cfg = load_config(hpc_cfg_path)
    tasks = build_within_session_pair_tasks(settings)
    if not tasks:
        print("No within-session pairs found for shuffled cross-correlation.")
        return

    if fix_cross_correlation_cfg_path is None and fix_crosscorr_cfg_path is not None:
        warnings.warn(
            (
                "fix_crosscorr_cfg_path is deprecated; "
                "use fix_cross_correlation_cfg_path instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
    fix_cfg_path = fix_cross_correlation_cfg_path or fix_crosscorr_cfg_path
    if fix_cfg_path is None:
        raise ValueError("Expected one of fix_cross_correlation_cfg_path or fix_crosscorr_cfg_path.")

    worker_script = Path(hpc_cfg["worker_script_path"]).resolve()
    dataset_cfg_resolved = str(Path(dataset_cfg_path).resolve())
    fix_cfg_resolved = str(Path(fix_cfg_path).resolve())
    generate_fix_cross_correlation_shuffle_job_file(
        tasks=tasks,
        job_file_path=hpc_cfg["job_file_path"],
        worker_script=worker_script,
        env_name=hpc_cfg["env_name"],
        dataset_cfg_path=dataset_cfg_resolved,
        fix_cross_correlation_cfg_path=fix_cfg_resolved,
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


def run_fix_cross_correlation_mode(
    settings: FixCrossCorrelationSettings,
    *,
    mode: str,
    compute_cross: bool,
    date: Optional[str] = None,
    session: Optional[str] = None,
    hpc_cfg_path: Optional[str] = None,
    dataset_cfg_path: Optional[str] = None,
    fix_cross_correlation_cfg_path: Optional[str] = None,
    fix_crosscorr_cfg_path: Optional[str] = None,
) -> None:
    """Run one cross-correlation workflow mode based on CLI mode flags."""
    if mode == "observed":
        run_fix_cross_correlation_analysis(settings, compute_cross=compute_cross)
        return

    if mode == "shuffle_submit_hpc":
        if fix_cross_correlation_cfg_path is None and fix_crosscorr_cfg_path is not None:
            warnings.warn(
                (
                    "fix_crosscorr_cfg_path is deprecated; "
                    "use fix_cross_correlation_cfg_path instead."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        fix_cfg_path = fix_cross_correlation_cfg_path or fix_crosscorr_cfg_path
        if hpc_cfg_path is None or dataset_cfg_path is None or fix_cfg_path is None:
            raise RuntimeError("shuffle_submit_hpc mode requires hpc/dataset/fix-cross-correlation config paths.")
        run_fix_cross_correlation_shuffle_submit_hpc(
            settings,
            hpc_cfg_path=hpc_cfg_path,
            dataset_cfg_path=dataset_cfg_path,
            fix_cross_correlation_cfg_path=fix_cfg_path,
        )
        return

    if mode == "shuffle_worker":
        if date is None or session is None:
            raise RuntimeError("--mode shuffle_worker requires --date and --session.")
        process_and_save_within_session_shuffle_pair(settings=settings, date=date, session=session)
        return

    if mode == "shuffle_collate":
        collate_within_session_shuffle_results(settings)
        return

    if mode == "shuffle_local":
        tasks = build_within_session_pair_tasks(settings)
        if not tasks:
            print("No within-session pairs found for shuffled cross-correlation.")
            return
        for task_date, task_session in tasks:
            process_and_save_within_session_shuffle_pair(
                settings=settings,
                date=task_date,
                session=task_session,
            )
        collate_within_session_shuffle_results(settings)
        return

    raise RuntimeError(f"Unsupported mode: {mode}")
