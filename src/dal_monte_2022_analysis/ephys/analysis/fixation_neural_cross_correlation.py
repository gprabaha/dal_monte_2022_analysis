"""Compute fixation-level neural PSTH cross-correlations within and across regions."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, replace
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_helpers import (
    _assert_lag_axis_match,
    _build_session_pair_average_dataframe,
    _canonical_region_name,
    _collect_fixation_groups,
    _extract_xcorr_dataframes_and_meta,
    _normalize_region_keys,
    _normalize_roi_groups,
    _sort_result_dataframe,
    _validate_signal_transform,
    _validate_xcorr_normalization,
    build_cross_region_fixation_neural_cross_correlation_plot_payload,
    build_fixation_neural_cross_correlation_plot_payload,
    build_within_region_fixation_neural_cross_correlation_plot_payload,
)
from dal_monte_2022_analysis.core.signal.cross_correlation import (
    fft_cross_correlation,
    normalize_cross_correlation_energy,
    summarize_cross_correlation,
)
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_optional_str as _as_optional_str,
    ensure_filename as _ensure_filename,
    extract_trials_df_and_meta as _extract_trials_df_and_meta,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    scan_processed_paths_for_filename,
    save_pickle_path,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_paths
from dal_monte_2022_analysis.runtime.execution.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir
from dal_monte_2022_analysis.core.behav.roi_groups import (
    DEFAULT_FIXATION_ROI_GROUPS as DEFAULT_SHARED_FIXATION_ROI_GROUPS,
)


WITHIN_ANALYSIS_KIND = "within_region"
CROSS_ANALYSIS_KIND = "cross_region"
_PLOT_ALLOWED_ANALYSIS_KINDS = (WITHIN_ANALYSIS_KIND, CROSS_ANALYSIS_KIND)
_ALLOWED_SIGNAL_TRANSFORMS = {"none", "demean", "zscore"}
_ALLOWED_XCORR_NORMALIZATIONS = {"none", "energy"}
_PLOT_CONDITION_ORDER = ("face_interactive", "face_non_interactive", "object")
_REGION_TOKEN_RE = re.compile(r"[^a-z0-9]+")
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


@dataclass
class FixationNeuralCrossCorrelationPlotAggregationSettings:
    """Configuration for analysis-side aggregation used by plotting."""

    cfg_path: str
    within_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/within_region"
    cross_input_subdir: str = "ephys/psth/fixation_neural_cross_correlation/cross_region"
    within_input_filename: str = "fixations.pkl"
    cross_input_filename: str = "fixations.pkl"
    within_pair_average_input_filename: str = "pair_averages.pkl"
    cross_pair_average_input_filename: str = "pair_averages.pkl"
    face_label: str = "face"
    object_label: str = "object"
    interactive_label: str = "interactive"
    condition_order: Sequence[str] = field(default_factory=lambda: tuple(_PLOT_CONDITION_ORDER))


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

def _build_session_output_path(
    cfg: dict,
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    analysis_kind: str,
    output_kind: str = "xcorr",
    date: str,
    session: str,
) -> Path:
    if output_kind not in {"xcorr", "pair_averages"}:
        raise ValueError("output_kind must be one of: xcorr, pair_averages.")

    if analysis_kind == WITHIN_ANALYSIS_KIND:
        subdir = settings.within_output_subdir
        if output_kind == "xcorr":
            filename = settings.within_output_filename
        else:
            filename = settings.within_pair_average_output_filename
    elif analysis_kind == CROSS_ANALYSIS_KIND:
        subdir = settings.cross_output_subdir
        if output_kind == "xcorr":
            filename = settings.cross_output_filename
        else:
            filename = settings.cross_pair_average_output_filename
    else:
        raise ValueError(f"Unsupported analysis_kind='{analysis_kind}'.")

    output_root = build_analysis_output_dir(cfg, subdir)
    return output_root / f"date={date}" / f"session={session}" / _ensure_filename(filename, ".pkl")


def build_fixation_neural_cross_correlations_for_session(
    settings: FixationNeuralCrossCorrelationSettings,
    session_row: dict,
    *,
    analysis_kind: str,
    show_progress: bool = True,
) -> Optional[dict]:
    """Compute fixation-level neural cross-correlations for one session file."""
    signal_transform = _validate_signal_transform(settings.signal_transform)
    xcorr_normalization = _validate_xcorr_normalization(settings.xcorr_normalization)
    max_lag = None if settings.max_lag is None else int(max(0, int(settings.max_lag)))

    obj = load_pickle_path(Path(session_row["path"]))
    trial_df, trial_meta = _extract_trials_df_and_meta(obj)
    if trial_df.empty or "psth_counts" not in trial_df.columns:
        return None

    include_region_keys = _normalize_region_keys(settings.include_regions)
    roi_groups = _normalize_roi_groups(settings.roi_groups)
    fixation_groups = _collect_fixation_groups(
        trial_df,
        default_date=str(session_row["date"]),
        default_session=str(session_row["session"]),
        include_region_keys=include_region_keys,
        roi_groups=roi_groups,
    )
    if not fixation_groups:
        return None

    anchor_region_key = _canonical_region_name(settings.anchor_region)
    partner_region_keys = _normalize_region_keys(settings.partner_regions)
    if partner_region_keys is not None and anchor_region_key is not None:
        partner_region_keys = {key for key in partner_region_keys if key != anchor_region_key}

    fixation_meta, signal_entries, pair_tasks, n_fixations_with_pairs = _build_pair_tasks(
        fixation_groups,
        analysis_kind=analysis_kind,
        anchor_region_key=anchor_region_key,
        partner_region_keys=partner_region_keys,
    )

    if settings.test_single and pair_tasks:
        pair_tasks = [random.choice(pair_tasks)]

    if not pair_tasks:
        return None

    lag_axis: Optional[np.ndarray] = None
    rows: list[dict] = []

    use_parallel = bool(settings.use_parallel and len(pair_tasks) > 1)
    if use_parallel:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        chunk_size = max(1, int(settings.pair_chunk_size))
        with Pool(
            processes=n_proc,
            initializer=_init_pair_worker,
            initargs=(fixation_meta, signal_entries, signal_transform, max_lag, xcorr_normalization),
        ) as pool:
            iterator = pool.imap_unordered(
                _compute_pair_xcorr_worker,
                pair_tasks,
                chunksize=chunk_size,
            )
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(pair_tasks),
                    desc=f"{analysis_kind} xcorr {session_row['date']}-{session_row['session']} ({n_proc} workers)",
                    unit="pair",
                )
            for result in iterator:
                if result is None:
                    continue
                lags = np.asarray(result.pop("lags"), dtype=np.int64)
                if lag_axis is None:
                    lag_axis = lags
                else:
                    _assert_lag_axis_match(lag_axis, lags)
                rows.append(result)
    else:
        _init_pair_worker(
            fixation_meta,
            signal_entries,
            signal_transform,
            max_lag,
            xcorr_normalization,
        )
        iterator = pair_tasks
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"{analysis_kind} xcorr {session_row['date']}-{session_row['session']}",
                unit="pair",
            )
        for task in iterator:
            result = _compute_pair_xcorr_worker(task)
            if result is None:
                continue
            lags = np.asarray(result.pop("lags"), dtype=np.int64)
            if lag_axis is None:
                lag_axis = lags
            else:
                _assert_lag_axis_match(lag_axis, lags)
            rows.append(result)

    if not rows or lag_axis is None:
        return None

    result_df = _sort_result_dataframe(pd.DataFrame(rows))
    pair_averages_df = _build_session_pair_average_dataframe(
        result_df,
        analysis_kind=analysis_kind,
        lags=lag_axis,
        face_label="face",
        object_label="object",
        interactive_label="interactive",
    )

    meta = {
        "analysis_kind": analysis_kind,
        "date": str(session_row["date"]),
        "session": str(session_row["session"]),
        "source_modality": settings.trial_input_modality,
        "source_filename": _ensure_filename(settings.trial_input_filename, ".pkl"),
        "signal_transform": signal_transform,
        "xcorr_normalization": xcorr_normalization,
        "max_lag": max_lag,
        "anchor_region": _as_optional_str(settings.anchor_region),
        "partner_regions": (
            None if settings.partner_regions is None else [str(v) for v in settings.partner_regions]
        ),
        "include_regions": (
            None if settings.include_regions is None else [str(v) for v in settings.include_regions]
        ),
        "n_fixations_total": int(len(fixation_groups)),
        "n_fixations_with_pairs": int(n_fixations_with_pairs),
        "n_pairs_requested": int(len(pair_tasks)),
        "n_pairs_computed": int(len(result_df)),
        "n_pair_averages": int(len(pair_averages_df)),
        "lags": lag_axis,
    }

    for key in ("bin_size_ms", "window_pre_s", "window_post_s", "bin_edges_s_rel", "bin_centers_s_rel"):
        if key in trial_meta:
            meta[key] = trial_meta[key]

    return {
        "meta": meta,
        "cross_correlations": result_df,
        "pair_averages": pair_averages_df,
    }


def process_and_save_fixation_neural_cross_correlations_for_session(
    settings: FixationNeuralCrossCorrelationSettings,
    session_row: dict,
    *,
    analysis_kind: str,
    show_progress: bool = True,
) -> Optional[dict]:
    """Build and persist fixation-level neural cross-correlation output for one session."""
    data = build_fixation_neural_cross_correlations_for_session(
        settings,
        session_row,
        analysis_kind=analysis_kind,
        show_progress=show_progress,
    )
    if data is None:
        return None

    cfg = load_config(settings.cfg_path)
    xcorr_out_path = _build_session_output_path(
        cfg,
        settings,
        analysis_kind=analysis_kind,
        output_kind="xcorr",
        date=str(session_row["date"]),
        session=str(session_row["session"]),
    )
    pair_avg_out_path = _build_session_output_path(
        cfg,
        settings,
        analysis_kind=analysis_kind,
        output_kind="pair_averages",
        date=str(session_row["date"]),
        session=str(session_row["session"]),
    )

    if xcorr_out_path == pair_avg_out_path:
        save_pickle_path(data, xcorr_out_path)
        return data

    save_pickle_path(
        {
            "meta": data.get("meta", {}),
            "cross_correlations": data.get("cross_correlations", pd.DataFrame()),
        },
        xcorr_out_path,
    )
    save_pickle_path(
        {
            "meta": data.get("meta", {}),
            "pair_averages": data.get("pair_averages", pd.DataFrame()),
        },
        pair_avg_out_path,
    )
    return data


def _process_and_save_session_worker(
    args: tuple[FixationNeuralCrossCorrelationSettings, dict, str],
) -> int:
    settings, session_row, analysis_kind = args
    local_settings = replace(
        settings,
        use_parallel=False,
        test_single=False,
    )
    data = process_and_save_fixation_neural_cross_correlations_for_session(
        local_settings,
        session_row,
        analysis_kind=analysis_kind,
        show_progress=False,
    )
    return 1 if data is not None else 0


def _run_fixation_neural_cross_correlation_analysis(
    settings: FixationNeuralCrossCorrelationSettings,
    *,
    analysis_kind: str,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    use_parallel: Optional[bool] = None,
    test_single: Optional[bool] = None,
) -> dict:
    if use_parallel is not None:
        settings.use_parallel = bool(use_parallel)
    if test_single is not None:
        settings.test_single = bool(test_single)

    cfg = load_config(settings.cfg_path)
    session_rows = scan_processed_paths_for_filename(
        cfg,
        settings.trial_input_modality,
        filename=_ensure_filename(settings.trial_input_filename, ".pkl"),
        dates=dates,
        sessions=sessions,
        agents=(None,),
    )
    if not session_rows:
        print("No fixation PSTH trial files found for neural cross-correlation analysis.")
        return {"n_sessions_total": 0, "n_sessions_written": 0}

    if settings.test_single and session_rows:
        session_rows = [random.choice(session_rows)]

    n_written = 0
    run_session_pool = bool(
        settings.use_parallel
        and settings.parallelize_across_sessions
        and len(session_rows) > 1
    )
    if run_session_pool:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        worker_tasks = [(settings, row, analysis_kind) for row in session_rows]
        with Pool(processes=n_proc) as pool:
            iterator = pool.imap_unordered(_process_and_save_session_worker, worker_tasks, chunksize=1)
            for wrote in tqdm(
                iterator,
                total=len(worker_tasks),
                desc=f"{analysis_kind} sessions ({n_proc} workers)",
                unit="session",
            ):
                n_written += int(wrote)
    else:
        local_settings = replace(settings, use_parallel=False)
        for session_row in tqdm(
            session_rows,
            desc=f"{analysis_kind} sessions",
            unit="session",
        ):
            data = process_and_save_fixation_neural_cross_correlations_for_session(
                local_settings,
                session_row,
                analysis_kind=analysis_kind,
                show_progress=True,
            )
            if data is not None:
                n_written += 1

    return {
        "n_sessions_total": int(len(session_rows)),
        "n_sessions_written": int(n_written),
    }


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
