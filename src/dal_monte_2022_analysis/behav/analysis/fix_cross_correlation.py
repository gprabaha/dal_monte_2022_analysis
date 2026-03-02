"""Compute fixation binary-vector cross-correlations within and across sessions."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from dal_monte_2022_analysis.behav.analysis.fix_cross_correlation_helpers import (
    _build_cross_session_control_rows,
    _build_within_session_rows,
    _print_output_sanity_summary,
    build_within_session_pair_tasks,
    collate_within_session_shuffle_results,
    process_and_save_within_session_shuffle_pair,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.behav.analysis_filenames import (
    resolve_fix_cross_correlation_filename,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    index_agent_paths as _index_agent_paths,
    index_shared_paths as _index_shared_paths,
)
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    normalize_fix_cross_correlation_time_scope,
)


@dataclass
class FixCrossCorrelationSettings:
    """Configuration for fixation cross-correlation analysis."""

    cfg_path: str
    input_modality: str = "fixation_binary_vectors"
    fixation_label: str = "face"
    output_subdir: str = "cross_correlation_outputs"
    within_filename: Optional[str] = None
    cross_filename: Optional[str] = None
    lags_filename: Optional[str] = None
    max_lag: Optional[int] = 60000
    time_scope: str = "whole"
    interactive_modality: str = "interactive_periods"
    interactive_state_label: str = "interactive"
    cross_pairs_max: Optional[int] = None
    cross_pairs_seed: int = 13
    cross_exclude_same_session: bool = True
    cross_exclude_same_date: bool = False
    parallelize_across_cross_correlation_pairs: bool = False
    shuffle_output_filename: Optional[str] = None
    shuffle_pairs_subdir: str = "within_session_shuffle_pair_results"
    shuffle_n_shuffles: int = 1000
    shuffle_stringent: bool = True
    shuffle_seed: int = 13
    shuffle_parallelize_within_pair: bool = True
    shuffle_log_every: int = 100
    test_single: bool = False

    @property
    def parallelize_across_crosscorr_pairs(self) -> bool:
        """Backward-compatible alias for legacy setting name."""
        return bool(self.parallelize_across_cross_correlation_pairs)

    @parallelize_across_crosscorr_pairs.setter
    def parallelize_across_crosscorr_pairs(self, value: bool) -> None:
        self.parallelize_across_cross_correlation_pairs = bool(value)


def run_fix_cross_correlation_analysis(
    settings: FixCrossCorrelationSettings,
    *,
    compute_cross: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run fixation cross-correlation analysis and persist outputs.

    Output files:
    - within filename (resolved by scope/label): per-session within-session vectors.
    - cross filename (resolved by scope/label): per-session cross-session controls.
    - lags filename (resolved by scope/label): shared lag axis.
    """
    cfg = load_config(settings.cfg_path)
    m1_paths, m2_paths = _index_agent_paths(cfg, settings.input_modality)
    scope = normalize_fix_cross_correlation_time_scope(settings.time_scope)

    if not m1_paths or not m2_paths:
        raise RuntimeError(
            "Missing fixation binary vectors for m1 or m2. "
            f"Found m1={len(m1_paths)} m2={len(m2_paths)}."
        )

    interactive_paths = None
    if scope != "whole":
        interactive_paths = _index_shared_paths(cfg, settings.interactive_modality)
        if not interactive_paths:
            raise RuntimeError(
                "No interactive-period files found for non-whole cross-correlation scope "
                f"'{scope}' in modality '{settings.interactive_modality}'."
            )

    within_rows, metadata_by_key, lag_axis = _build_within_session_rows(
        settings,
        m1_paths,
        m2_paths,
        interactive_paths=interactive_paths,
    )
    within_df = pd.DataFrame.from_records(within_rows)

    cross_df = None
    if compute_cross:
        within_keys = sorted(metadata_by_key)
        cross_rows, lag_axis = _build_cross_session_control_rows(
            settings=settings,
            m1_paths=m1_paths,
            m2_paths=m2_paths,
            within_keys=within_keys,
            metadata_by_key=metadata_by_key,
            lag_axis=lag_axis,
            interactive_paths=interactive_paths,
        )
        cross_df = pd.DataFrame.from_records(cross_rows)

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    within_path = out_dir / resolve_fix_cross_correlation_filename(
        fixation_label=settings.fixation_label,
        output_kind="within",
        time_scope=settings.time_scope,
        override=settings.within_filename,
    )
    within_df.to_pickle(within_path)

    if cross_df is not None:
        cross_path = out_dir / resolve_fix_cross_correlation_filename(
            fixation_label=settings.fixation_label,
            output_kind="cross",
            time_scope=settings.time_scope,
            override=settings.cross_filename,
        )
        cross_df.to_pickle(cross_path)

    if lag_axis is not None:
        lags_path = out_dir / resolve_fix_cross_correlation_filename(
            fixation_label=settings.fixation_label,
            output_kind="lags",
            time_scope=settings.time_scope,
            override=settings.lags_filename,
        )
        with open(lags_path, "wb") as f:
            pickle.dump(lag_axis, f)

    _print_output_sanity_summary(within_df, cross_df, lag_axis)

    return within_df, cross_df
