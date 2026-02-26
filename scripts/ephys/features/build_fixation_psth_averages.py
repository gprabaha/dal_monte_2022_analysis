"""Build date-level averaged fixation PSTH features from trial PSTH outputs."""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.features.fixation_psth import (
    FixationPSTHAverageSettings,
    run_fixation_psth_average_build,
)


def _iter_average_output_paths(
    dataset_cfg_path: str,
    output_subdir: str,
    output_filename: str,
    *,
    date: Optional[str] = None,
) -> list[Path]:
    cfg = load_config(dataset_cfg_path)
    root = Path(cfg["analysis_output_root"]) / output_subdir
    date_glob = f"date={date}" if date else "date=*"
    filename = output_filename if output_filename.endswith(".pkl") else f"{output_filename}.pkl"
    pattern = root / date_glob / filename
    return sorted(root.glob(str(pattern.relative_to(root))))


def _print_average_example(path: Path, *, max_bins: int = 12) -> None:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "averages" in obj:
        meta = obj.get("meta", {})
        df = obj["averages"]
    elif isinstance(obj, pd.DataFrame):
        meta = {}
        df = obj
    else:
        print(f"[example] Unsupported average output object type: {type(obj)}")
        return

    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"[example] Average output exists but is empty: {path}")
        return

    row = df.iloc[0]
    psth_mean = np.asarray(row.get("psth_mean"), dtype=float).reshape(-1)
    preview = psth_mean[: max(1, int(max_bins))]

    print("\nExample fixation PSTH average output:")
    print(f"  file: {path}")
    print(f"  n_rows: {len(df)}")
    if meta:
        print(
            "  meta: "
            f"smooth_before_average={meta.get('smooth_before_average')}, "
            f"smoothing_sigma_ms={meta.get('smoothing_sigma_ms')}, "
            f"split_by_interactive_state={meta.get('split_by_interactive_state')}"
        )
    if "fixation_category" in df.columns:
        cat_counts = df["fixation_category"].value_counts().to_dict()
        print(f"  category_counts: {cat_counts}")
    print(
        "  sample_row: "
        f"date={row.get('date')}, unit_uuid={row.get('unit_uuid')}, "
        f"category={row.get('fixation_category')}, n_trials={row.get('n_trials')}, "
        f"interactive_state={row.get('interactive_state') if 'interactive_state' in df.columns else None}"
    )
    print(f"  sample_psth_mean_first_{len(preview)}bins: {preview.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build date-level averaged fixation PSTH features.")
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--split-by-interactive-state", action="store_true")
    parser.add_argument("--no-show-example", action="store_true")
    parser.add_argument("--example-max-bins", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    settings = FixationPSTHAverageSettings(
        cfg_path=args.dataset_cfg,
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        output_subdir=cfg.get("average_output_subdir", "ephys/psth/fixation_psth_averages"),
        output_filename=cfg.get("average_output_filename", "fixations.pkl"),
        split_by_interactive_state=cfg.get("split_by_interactive_state", False),
        restrict_interactive_state=cfg.get("restrict_interactive_state"),
        group_by_session=cfg.get("group_by_session", False),
        smooth_before_average=cfg.get("smooth_before_average", True),
        smoothing_sigma_ms=cfg.get("smoothing_sigma_ms", 20.0),
        use_parallel=cfg.get("average_use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        categories=cfg.get("categories", ("face", "object", "out_of_roi")),
    )

    if args.use_parallel:
        settings.use_parallel = True
    if args.test_single:
        settings.test_single = True
    if args.no_smooth:
        settings.smooth_before_average = False
    if args.split_by_interactive_state:
        settings.split_by_interactive_state = True

    run_fixation_psth_average_build(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
    )

    if not args.no_show_example:
        paths = _iter_average_output_paths(
            args.dataset_cfg,
            settings.output_subdir,
            settings.output_filename,
            date=args.date,
        )
        if not paths:
            print("\n[example] No average PSTH output files found to preview.")
            return
        _print_average_example(paths[0], max_bins=args.example_max_bins)


if __name__ == "__main__":
    main()
