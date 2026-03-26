"""Shared CLI helpers for leader/follower cross-correlation entrypoints."""

from __future__ import annotations

import warnings

from dal_monte_2022_analysis.behav.analysis.fix_cross_correlation_leader_follower import (
    FixCrossCorrLeaderFollowerSettings,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.behav.analysis_filenames import (
    normalize_fix_cross_correlation_time_scope,
)


def _default_leader_follower_filenames(tag: str) -> dict[str, str]:
    token = str(tag).strip()
    return {
        "session_output_filename": f"within_session_{token}_fix_crosscorr_leader_follower.pkl",
        "date_summary_filename": f"date_summary_{token}_fix_crosscorr_leader_follower.pkl",
        "pair_summary_filename": f"pair_summary_{token}_fix_crosscorr_leader_follower.pkl",
        "global_summary_filename": f"global_summary_{token}_fix_crosscorr_leader_follower.pkl",
        "pupil_session_output_filename": (
            f"within_session_{token}_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
        ),
        "pupil_date_summary_filename": (
            f"date_summary_{token}_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
        ),
        "pupil_pair_summary_filename": (
            f"pair_summary_{token}_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
        ),
        "pupil_global_summary_filename": (
            f"global_summary_{token}_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
        ),
        "monkey_role_pupil_session_output_filename": (
            f"within_session_{token}_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv"
        ),
        "monkey_role_pupil_session_raw_filename": (
            f"within_session_{token}_fix_crosscorr_leader_follower_pupil_by_monkey_role_raw.pkl"
        ),
        "monkey_role_pupil_summary_filename": (
            f"summary_{token}_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv"
        ),
        "monkey_role_fixation_count_session_output_filename": (
            f"within_session_{token}_fix_crosscorr_leader_follower_fixation_count_by_monkey_role.csv"
        ),
        "monkey_role_fixation_count_summary_filename": (
            f"summary_{token}_fix_crosscorr_leader_follower_fixation_count_by_monkey_role.csv"
        ),
        "monkey_role_fixation_duration_session_output_filename": (
            f"within_session_{token}_fix_crosscorr_leader_follower_fixation_duration_by_monkey_role.csv"
        ),
        "monkey_role_fixation_duration_summary_filename": (
            f"summary_{token}_fix_crosscorr_leader_follower_fixation_duration_by_monkey_role.csv"
        ),
    }


def build_leader_follower_settings_from_config(
    *,
    dataset_cfg_path: str,
    fix_cross_correlation_cfg_path: str | None = None,
    fix_crosscorr_cfg_path: str | None = None,
    default_fixation_label: str,
    default_tag: str,
) -> FixCrossCorrLeaderFollowerSettings:
    """Build leader/follower settings from dataset + task config paths."""
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
    defaults = _default_leader_follower_filenames(default_tag)
    fixation_label = cfg.get("fixation_label", cfg.get("face_label", default_fixation_label))
    cross_correlation_subdir = cfg.get(
        "cross_correlation_output_subdir",
        cfg.get(
            "crosscorr_output_subdir",
            cfg.get("output_subdir", "fix_cross_correlation"),
        ),
    )
    leader_follower_subdir = cfg.get(
        "leader_follower_output_subdir",
        f"{cross_correlation_subdir}/leader_follower",
    )
    return FixCrossCorrLeaderFollowerSettings(
        cfg_path=dataset_cfg_path,
        fixation_label=fixation_label,
        output_subdir=leader_follower_subdir,
        cross_correlation_input_subdir=cross_correlation_subdir,
        within_filename=cfg.get("within_filename"),
        lags_filename=cfg.get("lags_filename"),
        time_scope=normalize_fix_cross_correlation_time_scope(cfg.get("leader_follower_time_scope", "whole")),
        session_output_filename=cfg.get(
            "leader_follower_session_filename",
            defaults["session_output_filename"],
        ),
        date_summary_filename=cfg.get(
            "leader_follower_date_summary_filename",
            defaults["date_summary_filename"],
        ),
        pair_summary_filename=cfg.get(
            "leader_follower_pair_summary_filename",
            cfg.get("leader_follower_total_summary_filename", defaults["pair_summary_filename"]),
        ),
        global_summary_filename=cfg.get(
            "leader_follower_global_summary_filename",
            defaults["global_summary_filename"],
        ),
        fixations_modality=cfg.get("leader_follower_fixations_modality", "fixations"),
        pupil_modality=cfg.get("leader_follower_pupil_modality", "pupil_size"),
        pupil_session_output_filename=cfg.get(
            "leader_follower_pupil_session_filename",
            defaults["pupil_session_output_filename"],
        ),
        pupil_date_summary_filename=cfg.get(
            "leader_follower_pupil_date_summary_filename",
            defaults["pupil_date_summary_filename"],
        ),
        pupil_pair_summary_filename=cfg.get(
            "leader_follower_pupil_pair_summary_filename",
            defaults["pupil_pair_summary_filename"],
        ),
        pupil_global_summary_filename=cfg.get(
            "leader_follower_pupil_global_summary_filename",
            defaults["pupil_global_summary_filename"],
        ),
        monkey_role_pupil_session_output_filename=cfg.get(
            "leader_follower_monkey_role_pupil_session_filename",
            defaults["monkey_role_pupil_session_output_filename"],
        ),
        monkey_role_pupil_session_raw_filename=cfg.get(
            "leader_follower_monkey_role_pupil_session_raw_filename",
            defaults["monkey_role_pupil_session_raw_filename"],
        ),
        monkey_role_pupil_summary_filename=cfg.get(
            "leader_follower_monkey_role_pupil_summary_filename",
            defaults["monkey_role_pupil_summary_filename"],
        ),
        monkey_role_fixation_count_session_output_filename=cfg.get(
            "leader_follower_monkey_role_fixation_count_session_filename",
            defaults["monkey_role_fixation_count_session_output_filename"],
        ),
        monkey_role_fixation_count_summary_filename=cfg.get(
            "leader_follower_monkey_role_fixation_count_summary_filename",
            defaults["monkey_role_fixation_count_summary_filename"],
        ),
        monkey_role_fixation_duration_session_output_filename=cfg.get(
            "leader_follower_monkey_role_fixation_duration_session_filename",
            cfg.get(
                "leader_follower_monkey_role_fixation_count_session_filename",
                defaults["monkey_role_fixation_duration_session_output_filename"],
            ),
        ),
        monkey_role_fixation_duration_summary_filename=cfg.get(
            "leader_follower_monkey_role_fixation_duration_summary_filename",
            cfg.get(
                "leader_follower_monkey_role_fixation_count_summary_filename",
                defaults["monkey_role_fixation_duration_summary_filename"],
            ),
        ),
        property_use_all_fixations=bool(cfg.get("leader_follower_property_use_all_fixations", False)),
        use_only_interactive_states=bool(cfg.get("leader_follower_use_only_interactive_states", False)),
        interactive_modality=cfg.get("leader_follower_interactive_modality", "interactive_periods"),
        interactive_state_label=cfg.get("leader_follower_interactive_state_label", "interactive"),
        pupil_roi_keywords=cfg.get("leader_follower_pupil_roi_keywords"),
        pupil_test_alpha=float(cfg.get("leader_follower_pupil_test_alpha", 0.05)),
        pupil_parallelize_sessions=bool(cfg.get("leader_follower_pupil_parallelize_sessions", True)),
        pupil_parallel_max_procs=int(cfg.get("leader_follower_pupil_parallel_max_procs", 16)),
        tie_epsilon=float(cfg.get("leader_follower_tie_epsilon", 0.0)),
    )


def apply_leader_follower_cli_overrides(
    settings: FixCrossCorrLeaderFollowerSettings,
    *,
    tie_epsilon: float | None = None,
) -> FixCrossCorrLeaderFollowerSettings:
    """Apply CLI-specific leader/follower overrides."""
    if tie_epsilon is not None:
        settings.tie_epsilon = max(0.0, float(tie_epsilon))
    return settings
