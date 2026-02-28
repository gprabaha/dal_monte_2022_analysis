"""Build within-region fixation-level neural PSTH cross-correlations."""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation import (
    FixationNeuralCrossCorrelationSettings,
    run_within_region_fixation_neural_cross_correlation,
)


def _as_str_list(values):
    if not values:
        return None
    out = [str(value).strip() for value in values if str(value).strip()]
    return out or None


def _iter_output_paths(
    dataset_cfg_path: str,
    output_subdir: str,
    output_filename: str,
    *,
    date: Optional[str] = None,
    session: Optional[str] = None,
) -> list[Path]:
    cfg = load_config(dataset_cfg_path)
    root = Path(cfg["analysis_output_root"]) / output_subdir
    date_glob = f"date={date}" if date else "date=*"
    session_glob = f"session={session}" if session else "session=*"
    filename = output_filename if output_filename.endswith(".pkl") else f"{output_filename}.pkl"
    pattern = root / date_glob / session_glob / filename
    return sorted(root.glob(str(pattern.relative_to(root))))


def _print_example(path: Path, *, max_lags: int = 12) -> None:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "cross_correlations" in obj:
        meta = obj.get("meta", {})
        df = obj["cross_correlations"]
    elif isinstance(obj, pd.DataFrame):
        meta = {}
        df = obj
    else:
        print(f"[example] Unsupported output object type: {type(obj)}")
        return

    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"[example] Output exists but is empty: {path}")
        return

    row = df.iloc[0]
    corr = np.asarray(row.get("cross_correlation"), dtype=float).reshape(-1)
    preview = corr[: max(1, int(max_lags))]

    print("\nExample within-region neural xcorr output:")
    print(f"  file: {path}")
    print(f"  n_rows: {len(df)}")
    if meta:
        print(
            "  meta: "
            f"signal_transform={meta.get('signal_transform')}, "
            f"max_lag={meta.get('max_lag')}, "
            f"n_fixations_with_pairs={meta.get('n_fixations_with_pairs')}"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build within-region fixation-level neural PSTH cross-correlations.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--ephys-fixation-neural-crosscorr-cfg",
        default="configs/ephys_fixation_neural_cross_correlation.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--max-lag", type=int, default=None)
    parser.add_argument(
        "--signal-transform",
        choices=["none", "demean", "zscore"],
        default=None,
    )
    parser.add_argument("--include-region", action="append", default=None)
    parser.add_argument("--no-show-example", action="store_true")
    parser.add_argument("--example-max-lags", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_neural_crosscorr_cfg)
    settings = FixationNeuralCrossCorrelationSettings(
        cfg_path=args.dataset_cfg,
        trial_input_modality=cfg.get("trial_input_modality", "psth"),
        trial_input_filename=cfg.get("trial_input_filename", "fixations.pkl"),
        within_output_subdir=cfg.get(
            "within_output_subdir",
            "ephys/psth/fixation_neural_crosscorr/within_region",
        ),
        cross_output_subdir=cfg.get(
            "cross_output_subdir",
            "ephys/psth/fixation_neural_crosscorr/cross_region",
        ),
        within_output_filename=cfg.get("within_output_filename", "fixations.pkl"),
        cross_output_filename=cfg.get("cross_output_filename", "fixations.pkl"),
        anchor_region=cfg.get("anchor_region", "BLA"),
        partner_regions=cfg.get("partner_regions", ("ACCg", "dmPFC", "OFC")),
        include_regions=cfg.get("include_regions"),
        roi_groups=cfg.get("roi_groups"),
        signal_transform=cfg.get("signal_transform", "zscore"),
        max_lag=cfg.get("max_lag"),
        use_parallel=cfg.get("use_parallel", True),
        max_procs=cfg.get("max_procs", 32),
        parallelize_across_sessions=cfg.get("parallelize_across_sessions", True),
        pair_chunk_size=cfg.get("pair_chunk_size", 64),
        test_single=cfg.get("test_single", False),
    )

    cli_include_regions = _as_str_list(args.include_region)
    if cli_include_regions is not None:
        settings.include_regions = cli_include_regions
    if args.no_parallel:
        settings.use_parallel = False
    if args.test_single:
        settings.test_single = True
    if args.max_lag is not None:
        settings.max_lag = max(0, int(args.max_lag))
    if args.signal_transform is not None:
        settings.signal_transform = args.signal_transform

    summary = run_within_region_fixation_neural_cross_correlation(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
    )
    print(
        "[analysis] within-region fixation neural xcorr: "
        f"wrote {summary.get('n_sessions_written', 0)}/{summary.get('n_sessions_total', 0)} session files"
    )

    if not args.no_show_example:
        paths = _iter_output_paths(
            args.dataset_cfg,
            settings.within_output_subdir,
            settings.within_output_filename,
            date=args.date,
            session=args.session,
        )
        if not paths:
            print("\n[example] No within-region output files found to preview.")
            return
        _print_example(paths[0], max_lags=args.example_max_lags)


if __name__ == "__main__":
    main()
