"""Derive leader-follower summaries from within-session fixation cross-correlations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from dal_monte_2022_analysis.behav.analysis.fix_cross_correlation_leader_follower_helpers import (
    _build_session_leader_output,
    _determine_session_leader_follower,
    _load_lags,
    _summarize_session_leader_by_date,
    _summarize_session_leader_by_pair,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.behav.analysis_filenames import (
    normalize_fix_cross_correlation_time_scope,
    resolve_fix_cross_correlation_filename,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.utils.filenames import (
    resolve_filename_override,
)


@dataclass
class FixCrossCorrLeaderFollowerSettings:
    """Configuration for leader-follower summaries from cross-correlation outputs."""

    cfg_path: str
    fixation_label: str = "face"
    output_subdir: str = "fix_cross_correlation"
    cross_correlation_input_subdir: Optional[str] = None
    within_filename: Optional[str] = None
    lags_filename: Optional[str] = None
    time_scope: str = "whole"
    session_output_filename: str = "within_session_face_fix_crosscorr_leader_follower.pkl"
    date_summary_filename: str = "date_summary_face_fix_crosscorr_leader_follower.pkl"
    pair_summary_filename: str = "pair_summary_face_fix_crosscorr_leader_follower.pkl"
    # Backward compatibility for older callers/configs.
    total_summary_filename: Optional[str] = None
    global_summary_filename: str = "global_summary_face_fix_crosscorr_leader_follower.pkl"
    fixations_modality: str = "fixations"
    pupil_modality: str = "pupil_size"
    pupil_session_output_filename: str = (
        "within_session_face_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
    )
    pupil_date_summary_filename: str = (
        "date_summary_face_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
    )
    pupil_pair_summary_filename: str = (
        "pair_summary_face_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
    )
    pupil_global_summary_filename: str = (
        "global_summary_face_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
    )
    monkey_role_pupil_session_output_filename: str = (
        "within_session_face_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv"
    )
    monkey_role_pupil_session_raw_filename: str = (
        "within_session_face_fix_crosscorr_leader_follower_pupil_by_monkey_role_raw.pkl"
    )
    monkey_role_pupil_summary_filename: str = (
        "summary_face_fix_crosscorr_leader_follower_pupil_by_monkey_role.csv"
    )
    monkey_role_fixation_count_session_output_filename: str = (
        "within_session_face_fix_crosscorr_leader_follower_fixation_count_by_monkey_role.csv"
    )
    monkey_role_fixation_count_summary_filename: str = (
        "summary_face_fix_crosscorr_leader_follower_fixation_count_by_monkey_role.csv"
    )
    monkey_role_fixation_duration_session_output_filename: str = (
        "within_session_face_fix_crosscorr_leader_follower_fixation_duration_by_monkey_role.csv"
    )
    monkey_role_fixation_duration_summary_filename: str = (
        "summary_face_fix_crosscorr_leader_follower_fixation_duration_by_monkey_role.csv"
    )
    property_use_all_fixations: bool = False
    use_only_interactive_states: bool = False
    interactive_modality: str = "interactive_periods"
    interactive_state_label: Optional[str] = "interactive"
    pupil_roi_keywords: Optional[list[str]] = None
    pupil_test_alpha: float = 0.05
    pupil_parallelize_sessions: bool = True
    pupil_parallel_max_procs: int = 16
    tie_epsilon: float = 0.0

    @property
    def crosscorr_input_subdir(self) -> Optional[str]:
        """Backward-compatible alias for legacy setting name."""
        return self.cross_correlation_input_subdir

    @crosscorr_input_subdir.setter
    def crosscorr_input_subdir(self, value: Optional[str]) -> None:
        self.cross_correlation_input_subdir = value


def run_fix_cross_correlation_leader_follower_analysis(
    settings: FixCrossCorrLeaderFollowerSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build simplified leader/follower outputs (session, date, pair)."""
    cfg = load_config(settings.cfg_path)
    scope = normalize_fix_cross_correlation_time_scope(settings.time_scope)
    input_subdir = settings.cross_correlation_input_subdir or settings.output_subdir
    input_dir = build_analysis_output_dir(cfg, input_subdir)
    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    within_path = input_dir / resolve_fix_cross_correlation_filename(
        fixation_label=settings.fixation_label,
        output_kind="within",
        time_scope=settings.time_scope,
        override=settings.within_filename,
    )
    lags_path = input_dir / resolve_fix_cross_correlation_filename(
        fixation_label=settings.fixation_label,
        output_kind="lags",
        time_scope=settings.time_scope,
        override=settings.lags_filename,
    )

    if not within_path.exists():
        raise RuntimeError(f"Missing within-session cross-correlation file: {within_path}")
    if not lags_path.exists():
        raise RuntimeError(f"Missing lag-axis file: {lags_path}")

    within_df = pd.read_pickle(within_path)
    lags = _load_lags(lags_path)
    session_df_full = _determine_session_leader_follower(
        within_df=within_df,
        lags=lags,
        fixation_label=settings.fixation_label,
        tie_epsilon=settings.tie_epsilon,
    )
    session_df_full["time_scope"] = scope

    session_df = _build_session_leader_output(session_df_full)
    date_summary_df = _summarize_session_leader_by_date(
        session_df_full,
        tie_epsilon=settings.tie_epsilon,
    )
    pair_summary_df = _summarize_session_leader_by_pair(
        session_df_full,
        tie_epsilon=settings.tie_epsilon,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    session_out = out_dir / settings.session_output_filename
    date_out = out_dir / settings.date_summary_filename
    pair_out = out_dir / resolve_filename_override(
        settings.pair_summary_filename,
        resolve_filename_override(
            settings.total_summary_filename,
            f"pair_summary_{settings.fixation_label}_fix_crosscorr_leader_follower.csv",
        ),
    )
    session_df.to_pickle(session_out)
    date_summary_df.to_pickle(date_out)
    pair_summary_df.to_pickle(pair_out)

    print(f"[leader-follower] wrote session-level leader calls: {session_out}")
    print(f"[leader-follower] wrote date-level leader calls: {date_out}")
    print(f"[leader-follower] wrote pair-level leader calls: {pair_out}")
    print(
        "[leader-follower] rows: "
        f"session={len(session_df)} date={len(date_summary_df)} pair={len(pair_summary_df)}"
    )

    return session_df, date_summary_df, pair_summary_df
