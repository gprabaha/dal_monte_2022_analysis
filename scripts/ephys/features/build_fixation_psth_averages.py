"""Build split and unsplit date-level fixation PSTH averages from trial PSTH outputs."""

import argparse
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.features.fixation_psth import (
    FixationPSTHAverageSettings,
    run_fixation_psth_average_build,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import (
    scan_analysis_date_paths,
)


def _iter_average_output_paths(
    dataset_cfg_path: str,
    output_subdir: str,
    output_filename: str,
    *,
    date: Optional[str] = None,
) -> list[Path]:
    cfg = load_config(dataset_cfg_path)
    rows = scan_analysis_date_paths(
        cfg,
        output_subdir,
        filename=output_filename,
        dates=[date] if date is not None else None,
    )
    return [row["path"] for row in rows]


def _ensure_pkl_filename(filename: str) -> str:
    token = str(filename).strip()
    if not token:
        token = "fixations.pkl"
    if not token.endswith(".pkl"):
        token = f"{token}.pkl"
    return token


def _derive_unsplit_filename(split_filename: str) -> str:
    token = _ensure_pkl_filename(split_filename)
    stem = token[:-4]
    if stem.endswith("_split_by_interactive_state"):
        return f"{stem[: -len('_split_by_interactive_state')]}_unsplit_by_interactive_state.pkl"
    return f"{stem}_unsplit_by_interactive_state.pkl"


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
    if "psth_sem" in df.columns:
        psth_sem = np.asarray(row.get("psth_sem"), dtype=float).reshape(-1)
    else:
        psth_sem = np.asarray([], dtype=float)
    preview = psth_mean[: max(1, int(max_bins))]
    sem_preview = psth_sem[: max(1, int(max_bins))]

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
    if psth_sem.size > 0:
        print(f"  sample_psth_sem_first_{len(sem_preview)}bins: {sem_preview.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build date-level averaged fixation PSTH features.")
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument(
        "--split-by-interactive-state",
        action="store_true",
        help="Legacy alias for --split-only.",
    )
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--unsplit-only", action="store_true")
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--output-subdir-unsplit", default=None)
    parser.add_argument("--output-filename-unsplit", default=None)
    parser.add_argument("--target-bin-size-ms", type=float, default=None)
    parser.add_argument("--target-bin-step-ms", type=float, default=None)
    parser.add_argument("--no-show-example", action="store_true")
    parser.add_argument("--example-max-bins", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    split_output_subdir = cfg.get(
        "average_output_subdir_split",
        cfg.get("average_output_subdir", "ephys/psth/fixation_psth_averages"),
    )
    split_output_filename = _ensure_pkl_filename(
        cfg.get(
            "average_output_filename_split",
            cfg.get("average_output_filename", "fixations.pkl"),
        ),
    )
    unsplit_output_subdir = cfg.get("average_output_subdir_unsplit", split_output_subdir)
    unsplit_output_filename = _ensure_pkl_filename(
        cfg.get(
            "average_output_filename_unsplit",
            _derive_unsplit_filename(split_output_filename),
        ),
    )

    settings_common = FixationPSTHAverageSettings(
        cfg_path=args.dataset_cfg,
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        output_subdir=split_output_subdir,
        output_filename=split_output_filename,
        split_by_interactive_state=True,
        restrict_interactive_state=cfg.get("restrict_interactive_state"),
        group_by_session=cfg.get("group_by_session", False),
        smooth_before_average=cfg.get("smooth_before_average", True),
        smoothing_sigma_ms=cfg.get("smoothing_sigma_ms", 20.0),
        target_bin_size_ms=cfg.get("average_target_bin_size_ms"),
        target_bin_step_ms=cfg.get("average_target_bin_step_ms"),
        use_parallel=cfg.get("average_use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        categories=cfg.get("categories", ("face", "object", "out_of_roi")),
    )

    if args.split_only and args.unsplit_only:
        raise ValueError("Cannot use --split-only and --unsplit-only together.")

    if args.use_parallel:
        settings_common.use_parallel = True
    if args.test_single:
        settings_common.test_single = True
    if args.no_smooth:
        settings_common.smooth_before_average = False
    if args.target_bin_size_ms is not None:
        settings_common.target_bin_size_ms = float(args.target_bin_size_ms)
    if args.target_bin_step_ms is not None:
        settings_common.target_bin_step_ms = float(args.target_bin_step_ms)

    run_split_output = not bool(args.unsplit_only)
    run_unsplit_output = not bool(args.split_only or args.split_by_interactive_state)

    if args.output_subdir:
        split_output_subdir = str(args.output_subdir)
        unsplit_output_subdir = str(args.output_subdir)
    if args.output_subdir_unsplit:
        unsplit_output_subdir = str(args.output_subdir_unsplit)
    if args.output_filename:
        split_output_filename = _ensure_pkl_filename(args.output_filename)
    if args.output_filename_unsplit:
        unsplit_output_filename = _ensure_pkl_filename(args.output_filename_unsplit)

    settings_by_label: list[tuple[str, FixationPSTHAverageSettings]] = []
    if run_split_output:
        settings_by_label.append(
            (
                "split",
                replace(
                    settings_common,
                    output_subdir=split_output_subdir,
                    output_filename=split_output_filename,
                    split_by_interactive_state=True,
                ),
            ),
        )
    if run_unsplit_output:
        settings_by_label.append(
            (
                "unsplit",
                replace(
                    settings_common,
                    output_subdir=unsplit_output_subdir,
                    output_filename=unsplit_output_filename,
                    split_by_interactive_state=False,
                ),
            ),
        )
    if not settings_by_label:
        raise ValueError("No average output mode selected.")

    for _, mode_settings in settings_by_label:
        run_fixation_psth_average_build(
            mode_settings,
            dates=[args.date] if args.date else None,
            sessions=[args.session] if args.session else None,
        )

    if not args.no_show_example:
        any_examples = False
        for label, mode_settings in settings_by_label:
            paths = _iter_average_output_paths(
                args.dataset_cfg,
                mode_settings.output_subdir,
                mode_settings.output_filename,
                date=args.date,
            )
            if not paths:
                print(f"\n[example] No {label} average PSTH output files found to preview.")
                continue
            print(f"\n[example] Previewing {label} average output")
            _print_average_example(paths[0], max_bins=args.example_max_bins)
            any_examples = True
        if not any_examples:
            print("\n[example] No average PSTH output files found to preview.")


if __name__ == "__main__":
    main()
