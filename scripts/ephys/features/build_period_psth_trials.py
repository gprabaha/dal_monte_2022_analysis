"""Build session-level interactive/non-interactive period PSTH trials."""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.config.load import load_ephys_period_psth_config
from dal_monte_2022_analysis.ephys.features.period_psth import (
    PeriodPSTHSettings,
    run_period_psth_trial_build,
)


def _iter_trial_output_paths(
    dataset_cfg_path: str,
    output_modality: str,
    output_filename: str,
    *,
    date: Optional[str] = None,
    session: Optional[str] = None,
) -> list[Path]:
    cfg = load_dataset_config(dataset_cfg_path)
    root = Path(cfg["processed_data_root"])
    date_glob = f"date={date}" if date else "date=*"
    session_glob = f"session={session}" if session else "session=*"
    filename = output_filename if output_filename.endswith(".pkl") else f"{output_filename}.pkl"
    pattern = root / date_glob / session_glob / output_modality / filename
    return sorted(root.glob(str(pattern.relative_to(root))))


def _print_trial_example(path: Path, *, max_bins: int = 12) -> None:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "trials" in obj:
        meta = obj.get("meta", {})
        df = obj["trials"]
    elif isinstance(obj, pd.DataFrame):
        meta = {}
        df = obj
    else:
        print(f"[example] Unsupported trial output object type: {type(obj)}")
        return

    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"[example] Trial output exists but is empty: {path}")
        return

    row = df.iloc[0]
    counts = np.asarray(row.get("psth_counts"), dtype=float).reshape(-1)
    preview = counts[: max(1, int(max_bins))]

    print("\nExample period PSTH trial output:")
    print(f"  file: {path}")
    print(f"  n_rows: {len(df)}")
    if meta:
        print(
            "  meta: "
            f"bin_size_ms={meta.get('bin_size_ms')}, "
            f"window_pre_s={meta.get('window_pre_s')}, "
            f"window_post_s={meta.get('window_post_s')}, "
            f"include_states={meta.get('include_states')}"
        )
    if "period_state" in df.columns:
        state_counts = df["period_state"].value_counts().to_dict()
        print(f"  period_state_counts: {state_counts}")
    print(
        "  sample_row: "
        f"date={row.get('date')}, session={row.get('session')}, "
        f"unit_uuid={row.get('unit_uuid')}, state={row.get('period_state')}, "
        f"period_start={row.get('period_start_idx')}, period_stop={row.get('period_stop_idx')}, "
        f"period_center={row.get('period_center_idx')}"
    )
    print(f"  sample_psth_counts_first_{len(preview)}bins: {preview.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build interactive/non-interactive period-centered PSTH trial features."
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-cfg", default="configs/ephys_data.yaml")
    parser.add_argument("--ephys-period-psth-cfg", default="configs/ephys_period_psth.yaml")
    parser.add_argument(
        "--ephys-fixation-psth-cfg",
        dest="ephys_period_psth_cfg",
        default="configs/ephys_period_psth.yaml",
        help="Deprecated alias for --ephys-period-psth-cfg.",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--no-show-example", action="store_true")
    parser.add_argument("--example-max-bins", type=int, default=12)
    args = parser.parse_args()

    cfg = load_ephys_period_psth_config(args.ephys_period_psth_cfg)
    settings = PeriodPSTHSettings(
        cfg_path=args.dataset_cfg,
        ephys_cfg_path=args.ephys_cfg,
        timeline_modality=cfg.get("timeline_modality", "neural_timeline"),
        periods_modality=cfg.get("periods_modality", cfg.get("interactive_modality", "interactive_periods")),
        output_modality=cfg.get("output_modality", cfg.get("period_trial_output_modality", "psth")),
        trial_output_filename=cfg.get(
            "trial_output_filename",
            cfg.get("period_trial_output_filename", "interactive_periods.pkl"),
        ),
        state_column=cfg.get("state_column", cfg.get("period_state_column", "state")),
        start_column=cfg.get("start_column", cfg.get("period_start_column", "start")),
        stop_column=cfg.get("stop_column", cfg.get("period_stop_column", "stop")),
        include_states=cfg.get("states", cfg.get("period_states", ("interactive", "non_interactive"))),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        bin_size_ms=cfg.get("bin_size_ms", cfg.get("period_bin_size_ms", 100.0)),
        window_pre_s=cfg.get("window_pre_s", cfg.get("period_window_pre_s", 14.0)),
        window_post_s=cfg.get("window_post_s", cfg.get("period_window_post_s", 14.0)),
        use_parallel=cfg.get("use_parallel", cfg.get("period_use_parallel", True)),
        max_procs=cfg.get("max_procs", cfg.get("period_max_procs", 16)),
        test_single=cfg.get("test_single", cfg.get("period_test_single", False)),
        restrict_units_to_date=cfg.get("restrict_units_to_date", True),
    )

    if args.use_parallel:
        settings.use_parallel = True
    if args.test_single:
        settings.test_single = True

    if args.date and args.session:
        run_period_psth_trial_build(
            settings,
            dates=[args.date],
            sessions=[args.session],
            use_parallel=settings.use_parallel,
            test_single=False,
        )
    else:
        run_period_psth_trial_build(
            settings,
            dates=[args.date] if args.date else None,
            sessions=[args.session] if args.session else None,
            use_parallel=settings.use_parallel,
            test_single=settings.test_single,
        )

    if not args.no_show_example:
        paths = _iter_trial_output_paths(
            args.dataset_cfg,
            settings.output_modality,
            settings.trial_output_filename,
            date=args.date,
            session=args.session,
        )
        if not paths:
            print("\n[example] No period PSTH trial output files found to preview.")
            return
        _print_trial_example(paths[0], max_bins=args.example_max_bins)


if __name__ == "__main__":
    main()
