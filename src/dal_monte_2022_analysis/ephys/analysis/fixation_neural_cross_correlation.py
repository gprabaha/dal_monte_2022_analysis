"""Compute fixation-level neural PSTH cross-correlations within and across regions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_helpers import (
    CROSS_ANALYSIS_KIND,
    WITHIN_ANALYSIS_KIND,
    _PLOT_ALLOWED_ANALYSIS_KINDS,
    _extract_xcorr_dataframes_and_meta,
    _run_fixation_neural_cross_correlation_analysis,
    FixationNeuralCrossCorrelationPlotAggregationSettings,
    build_cross_region_fixation_neural_cross_correlation_plot_payload,
    build_fixation_neural_cross_correlation_plot_payload,
    build_fixation_neural_cross_correlations_for_session,
    build_within_region_fixation_neural_cross_correlation_plot_payload,
    process_and_save_fixation_neural_cross_correlations_for_session,
)
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    ensure_filename as _ensure_filename,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_paths
from dal_monte_2022_analysis.core.behav.roi_groups import (
    DEFAULT_FIXATION_ROI_GROUPS as DEFAULT_SHARED_FIXATION_ROI_GROUPS,
)


DEFAULT_FIXATION_ROI_GROUPS: dict[str, tuple[str, ...]] = DEFAULT_SHARED_FIXATION_ROI_GROUPS


@dataclass
class FixationNeuralCrossCorrelationSettings:
    """Configuration for fixation-level neural PSTH cross-correlation analysis."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations.pkl"
    within_output_subdir: str = "ephys/psth/fixation_neural_cross_correlation/within_region"
    cross_output_subdir: str = "ephys/psth/fixation_neural_cross_correlation/cross_region"
    within_output_filename: str = "fixations.pkl"
    cross_output_filename: str = "fixations.pkl"
    within_pair_average_output_filename: str = "pair_averages.pkl"
    cross_pair_average_output_filename: str = "pair_averages.pkl"
    anchor_region: str = "BLA"
    partner_regions: Optional[Sequence[str]] = ("ACCg", "dmPFC", "OFC")
    include_regions: Optional[Sequence[str]] = None
    roi_groups: dict[str, Sequence[str]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_FIXATION_ROI_GROUPS.items()},
    )
    signal_transform: str = "zscore"
    xcorr_normalization: str = "energy"
    max_lag: Optional[int] = None
    use_parallel: bool = True
    max_procs: int = 32
    parallelize_across_sessions: bool = True
    pair_chunk_size: int = 64
    test_single: bool = False


def coerce_nonempty_str_list(values) -> Optional[list[str]]:
    """Normalize repeated CLI string options to a non-empty list or None."""
    if not values:
        return None
    out = [str(value).strip() for value in values if str(value).strip()]
    return out or None


def build_fixation_neural_cross_correlation_settings_from_config(
    *,
    dataset_cfg_path: str,
    ephys_fixation_neural_cross_correlation_cfg_path: str | None = None,
    ephys_fixation_neural_crosscorr_cfg_path: str | None = None,
) -> FixationNeuralCrossCorrelationSettings:
    """Build analysis settings from dataset + task config paths."""
    cfg_path = (
        ephys_fixation_neural_cross_correlation_cfg_path
        or ephys_fixation_neural_crosscorr_cfg_path
    )
    if cfg_path is None:
        raise ValueError(
            "Expected one of ephys_fixation_neural_cross_correlation_cfg_path "
            "or ephys_fixation_neural_crosscorr_cfg_path.",
        )
    cfg = load_config(cfg_path)
    return FixationNeuralCrossCorrelationSettings(
        cfg_path=dataset_cfg_path,
        trial_input_modality=cfg.get("trial_input_modality", "psth"),
        trial_input_filename=cfg.get("trial_input_filename", "fixations.pkl"),
        within_output_subdir=cfg.get(
            "within_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/within_region",
        ),
        cross_output_subdir=cfg.get(
            "cross_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/cross_region",
        ),
        within_output_filename=cfg.get("within_output_filename", "fixations.pkl"),
        cross_output_filename=cfg.get("cross_output_filename", "fixations.pkl"),
        within_pair_average_output_filename=cfg.get("within_pair_average_output_filename", "pair_averages.pkl"),
        cross_pair_average_output_filename=cfg.get("cross_pair_average_output_filename", "pair_averages.pkl"),
        anchor_region=cfg.get("anchor_region", "BLA"),
        partner_regions=cfg.get("partner_regions", ("ACCg", "dmPFC", "OFC")),
        include_regions=cfg.get("include_regions"),
        roi_groups=cfg.get("roi_groups"),
        signal_transform=cfg.get("signal_transform", "zscore"),
        xcorr_normalization=cfg.get("xcorr_normalization", "energy"),
        max_lag=cfg.get("max_lag"),
        use_parallel=cfg.get("use_parallel", True),
        max_procs=cfg.get("max_procs", 32),
        parallelize_across_sessions=cfg.get("parallelize_across_sessions", True),
        pair_chunk_size=cfg.get("pair_chunk_size", 64),
        test_single=cfg.get("test_single", False),
    )


def apply_fixation_neural_cross_correlation_cli_overrides(
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    anchor_region: Optional[str] = None,
    partner_regions: Optional[Sequence[str]] = None,
    include_regions: Optional[Sequence[str]] = None,
    no_parallel: bool = False,
    test_single: bool = False,
    max_lag: Optional[int] = None,
    signal_transform: Optional[str] = None,
    xcorr_normalization: Optional[str] = None,
) -> FixationNeuralCrossCorrelationSettings:
    """Apply CLI-specific overrides to an existing settings object."""
    if anchor_region is not None:
        settings.anchor_region = str(anchor_region)
    if partner_regions is not None:
        settings.partner_regions = list(partner_regions)
    if include_regions is not None:
        settings.include_regions = list(include_regions)
    if no_parallel:
        settings.use_parallel = False
    if test_single:
        settings.test_single = True
    if max_lag is not None:
        settings.max_lag = max(0, int(max_lag))
    if signal_transform is not None:
        settings.signal_transform = str(signal_transform)
    if xcorr_normalization is not None:
        settings.xcorr_normalization = str(xcorr_normalization)
    return settings


def iter_fixation_neural_cross_correlation_output_paths(
    *,
    dataset_cfg_path: str,
    output_subdir: str,
    output_filename: str,
    date: Optional[str] = None,
    session: Optional[str] = None,
) -> list[Path]:
    """List analysis output files for one cross-correlation analysis kind."""
    cfg = load_config(dataset_cfg_path)
    rows = scan_analysis_paths(
        cfg,
        output_subdir,
        filename=_ensure_filename(output_filename),
        dates=[str(date)] if date is not None else None,
        sessions=[str(session)] if session is not None else None,
    )
    return [Path(row["path"]) for row in rows]


def print_fixation_neural_cross_correlation_example(
    path: str | Path,
    *,
    analysis_kind: str,
    max_lags: int = 12,
) -> None:
    """Print a compact preview from one saved neural cross-correlation output."""
    if analysis_kind not in _PLOT_ALLOWED_ANALYSIS_KINDS:
        raise ValueError(
            "Unsupported analysis_kind="
            f"{analysis_kind!r}. Expected one of: {', '.join(_PLOT_ALLOWED_ANALYSIS_KINDS)}."
        )

    obj = load_pickle_path(path)
    df, pair_avg_df, meta = _extract_xcorr_dataframes_and_meta(obj)
    n_pair_avg_rows = len(pair_avg_df) if not pair_avg_df.empty else 0
    path = Path(path)
    label = "within-region" if analysis_kind == WITHIN_ANALYSIS_KIND else "cross-region"

    if df.empty:
        print(f"[example] Output exists but is empty: {path}")
        return

    row = df.iloc[0]
    corr = np.asarray(row.get("cross_correlation"), dtype=float).reshape(-1)
    preview = corr[: max(1, int(max_lags))]

    print(f"\nExample {label} neural xcorr output:")
    print(f"  file: {path}")
    print(f"  n_rows: {len(df)}")
    print(f"  n_pair_average_rows: {n_pair_avg_rows}")
    if meta:
        base_meta = (
            f"signal_transform={meta.get('signal_transform')}, "
            f"xcorr_normalization={meta.get('xcorr_normalization')}, "
            f"max_lag={meta.get('max_lag')}, "
            f"n_pair_averages={meta.get('n_pair_averages')}"
        )
        if analysis_kind == WITHIN_ANALYSIS_KIND:
            print(
                "  meta: "
                f"{base_meta}, "
                f"n_fixations_with_pairs={meta.get('n_fixations_with_pairs')}"
            )
        else:
            print(
                "  meta: "
                f"anchor_region={meta.get('anchor_region')}, "
                f"partner_regions={meta.get('partner_regions')}, "
                f"{base_meta}"
            )
    print(
        "  sample_row: "
        f"date={row.get('date')}, session={row.get('session')}, "
        f"fixation_id={row.get('fixation_id')}, "
        f"region_1={row.get('region_1')}, unit_1={row.get('unit_uuid_1')}, "
        f"region_2={row.get('region_2')}, unit_2={row.get('unit_uuid_2')}, "
        f"interactive_state={row.get('interactive_state')}, "
        f"fixation_location={row.get('fixation_location')}"
    )
    print(
        "  sample_corr_summary: "
        f"n_lags={row.get('n_lags')}, "
        f"zero_lag={row.get('zero_lag_correlation')}, "
        f"peak_lag={row.get('peak_lag')}, "
        f"peak_corr={row.get('peak_correlation')}"
    )
    print(f"  sample_cross_correlation_first_{len(preview)}: {preview.tolist()}")


def run_within_region_fixation_neural_cross_correlation(
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> dict:
    """Run within-region fixation-level neural PSTH cross-correlation analysis."""
    return _run_fixation_neural_cross_correlation_analysis(
        settings,
        analysis_kind=WITHIN_ANALYSIS_KIND,
        dates=dates,
        sessions=sessions,
        use_parallel=use_parallel,
        test_single=test_single,
    )


def run_cross_region_fixation_neural_cross_correlation(
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> dict:
    """Run cross-region fixation-level neural PSTH cross-correlation analysis."""
    return _run_fixation_neural_cross_correlation_analysis(
        settings,
        analysis_kind=CROSS_ANALYSIS_KIND,
        dates=dates,
        sessions=sessions,
        use_parallel=use_parallel,
        test_single=test_single,
    )
