"""Build session-level fixation PSTH trials for ephys units."""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.features.fixation_psth import (
    DEFAULT_FIXATION_ROI_GROUPS,
    FixationPSTHSettings,
    run_fixation_psth_trial_build,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    scan_processed_paths_for_filename,
)


def _iter_trial_output_paths(
    dataset_cfg_path: str,
    output_modality: str,
    output_filename: str,
    *,
    date: Optional[str] = None,
    session: Optional[str] = None,
) -> list[Path]:
    cfg = load_config(dataset_cfg_path)
    rows = scan_processed_paths_for_filename(
        cfg,
        output_modality,
        filename=output_filename,
        dates=[date] if date is not None else None,
        sessions=[session] if session is not None else None,
        agents=[None],
    )
    return [row["path"] for row in rows]


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

    print("\nExample fixation PSTH trial output:")
    print(f"  file: {path}")
    print(f"  n_rows: {len(df)}")
    if meta:
        print(
            "  meta: "
            f"bin_size_ms={meta.get('bin_size_ms')}, "
            f"window_pre_s={meta.get('window_pre_s')}, "
            f"window_post_s={meta.get('window_post_s')}"
        )
    if "fixation_category" in df.columns:
        cat_counts = df["fixation_category"].value_counts().to_dict()
        print(f"  category_counts: {cat_counts}")
    print(
        "  sample_row: "
        f"date={row.get('date')}, session={row.get('session')}, "
        f"unit_uuid={row.get('unit_uuid')}, category={row.get('fixation_category')}, "
        f"fix_start={row.get('fixation_start_idx')}, fix_stop={row.get('fixation_stop_idx')}, "
        f"interactive_state={row.get('interactive_state')}"
    )
    print(f"  sample_psth_counts_first_{len(preview)}bins: {preview.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixation-triggered PSTH trial features.")
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-cfg", default="configs/ephys_data.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--no-show-example", action="store_true")
    parser.add_argument("--example-max-bins", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    settings = FixationPSTHSettings(
        cfg_path=args.dataset_cfg,
        ephys_cfg_path=args.ephys_cfg,
        fixations_modality=cfg.get("fixations_modality", "fixations"),
        timeline_modality=cfg.get("timeline_modality", "neural_timeline"),
        interactive_modality=cfg.get("interactive_modality", "interactive_periods"),
        output_modality=cfg.get("trial_output_modality", "psth"),
        trial_output_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        roi_groups=cfg.get("roi_groups", DEFAULT_FIXATION_ROI_GROUPS),
        agent_roi_groups=cfg.get("agent_roi_groups"),
        categories=cfg.get("categories", ("face", "object", "out_of_roi")),
        include_interactive_state=cfg.get("include_interactive_state", True),
        interactive_high_label=cfg.get("interactive_high_label", "interactive"),
        bin_size_ms=cfg.get("bin_size_ms", 10.0),
        window_pre_s=cfg.get("window_pre_s", 1.0),
        window_post_s=cfg.get("window_post_s", 1.0),
        use_parallel=cfg.get("use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        agents=cfg.get("agents"),
    )

    if args.use_parallel:
        settings.use_parallel = True
    if args.test_single:
        settings.test_single = True

    if args.date and args.session:
        run_fixation_psth_trial_build(
            settings,
            dates=[args.date],
            sessions=[args.session],
            use_parallel=settings.use_parallel,
            test_single=False,
        )
    else:
        run_fixation_psth_trial_build(
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
            print("\n[example] No trial PSTH output files found to preview.")
            return
        _print_trial_example(paths[0], max_bins=args.example_max_bins)


if __name__ == "__main__":
    main()
