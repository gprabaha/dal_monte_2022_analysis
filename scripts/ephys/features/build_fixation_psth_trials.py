"""Build fixation-aligned trial outputs from unit spike times.

This legacy entrypoint preserves the combined trial file that stores both:
- `psth_counts` at the configured PSTH bin size
- `spike_train_counts` at the configured spike-train bin size
- `smoothed_spike_train_counts` using Gaussian smoothing on the 1 ms train

Newer explicit entrypoints split those signals into separate files.
"""

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


TRIAL_BUILD_PROFILES = {
    "legacy_combined": {
        "description": "Build fixation-triggered trial outputs (legacy combined PSTH + spike-train file).",
        "trial_output_filename": None,
        "bin_size_ms": None,
        "bin_step_ms": None,
        "spike_train_bin_size_ms": None,
        "store_psth_counts": True,
        "store_spike_train_counts": True,
    },
    "psth_10ms": {
        "description": "Build 10 ms non-overlapping fixation PSTH trial spike counts.",
        "trial_output_filename": "fixations_psth_10ms.pkl",
        "bin_size_ms": 10.0,
        "bin_step_ms": 10.0,
        "spike_train_bin_size_ms": 1.0,
        "store_psth_counts": True,
        "store_spike_train_counts": False,
    },
    "psth_50ms_step_25ms": {
        "description": "Build 50 ms fixation PSTH trial spike counts with 25 ms stride.",
        "trial_output_filename": "fixations_psth_50ms_step_25ms.pkl",
        "bin_size_ms": 50.0,
        "bin_step_ms": 25.0,
        "spike_train_bin_size_ms": 1.0,
        "store_psth_counts": True,
        "store_spike_train_counts": False,
    },
    "spike_train_1ms": {
        "description": "Build 1 ms fixation-aligned trial spike-train counts plus a smoothed copy.",
        "trial_output_filename": "fixations_spike_train_1ms.pkl",
        "bin_size_ms": 10.0,
        "bin_step_ms": 10.0,
        "spike_train_bin_size_ms": 1.0,
        "store_psth_counts": False,
        "store_spike_train_counts": True,
    },
}


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
    print("\nExample fixation trial output:")
    print(f"  file: {path}")
    print(f"  n_rows: {len(df)}")
    if meta:
        print(
            "  meta: "
            f"bin_size_ms={meta.get('bin_size_ms')}, "
            f"bin_step_ms={meta.get('bin_step_ms')}, "
            f"spike_train_bin_size_ms={meta.get('spike_train_bin_size_ms')}, "
            f"spike_train_smoothing_sigma_bins={meta.get('spike_train_smoothing_sigma_bins')}, "
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
    if "psth_counts" in row:
        counts = np.asarray(row.get("psth_counts"), dtype=float).reshape(-1)
        preview = counts[: max(1, int(max_bins))]
        print(f"  sample_psth_counts_first_{len(preview)}bins: {preview.tolist()}")
    if "spike_train_counts" in row:
        spike_counts = np.asarray(row.get("spike_train_counts"), dtype=float).reshape(-1)
        spike_preview = spike_counts[: max(1, int(max_bins))]
        print(f"  sample_spike_train_counts_first_{len(spike_preview)}bins: {spike_preview.tolist()}")
    if "smoothed_spike_train_counts" in row:
        smoothed = np.asarray(row.get("smoothed_spike_train_counts"), dtype=float).reshape(-1)
        smoothed_preview = smoothed[: max(1, int(max_bins))]
        print(
            f"  sample_smoothed_spike_train_counts_first_{len(smoothed_preview)}bins: "
            f"{smoothed_preview.tolist()}"
        )


def _build_settings(
    *,
    dataset_cfg: str,
    ephys_cfg: str,
    fixation_cfg_path: str,
    profile_name: str,
) -> FixationPSTHSettings:
    cfg = load_config(fixation_cfg_path)
    profile = TRIAL_BUILD_PROFILES[profile_name]
    trial_output_filename = profile["trial_output_filename"]
    if trial_output_filename is None:
        trial_output_filename = cfg.get("trial_output_filename", "fixations.pkl")

    bin_size_ms = profile["bin_size_ms"]
    if bin_size_ms is None:
        bin_size_ms = cfg.get("bin_size_ms", 10.0)

    bin_step_ms = profile["bin_step_ms"]
    if bin_step_ms is None and profile_name == "legacy_combined":
        bin_step_ms = cfg.get("bin_step_ms")

    spike_train_bin_size_ms = profile["spike_train_bin_size_ms"]
    if spike_train_bin_size_ms is None:
        spike_train_bin_size_ms = cfg.get("spike_train_bin_size_ms", 1.0)

    spike_train_smoothing_sigma_bins = cfg.get("spike_train_smoothing_sigma_bins", 2.0)

    return FixationPSTHSettings(
        cfg_path=dataset_cfg,
        ephys_cfg_path=ephys_cfg,
        fixations_modality=cfg.get("fixations_modality", "fixations"),
        timeline_modality=cfg.get("timeline_modality", "neural_timeline"),
        interactive_modality=cfg.get("interactive_modality", "interactive_periods"),
        output_modality=cfg.get("trial_output_modality", "psth"),
        trial_output_filename=trial_output_filename,
        roi_groups=cfg.get("roi_groups", DEFAULT_FIXATION_ROI_GROUPS),
        agent_roi_groups=cfg.get("agent_roi_groups"),
        categories=cfg.get("categories", ("face", "object")),
        include_interactive_state=cfg.get("include_interactive_state", True),
        interactive_high_label=cfg.get("interactive_high_label", "interactive"),
        bin_size_ms=bin_size_ms,
        bin_step_ms=bin_step_ms,
        spike_train_bin_size_ms=spike_train_bin_size_ms,
        spike_train_smoothing_sigma_bins=spike_train_smoothing_sigma_bins,
        store_psth_counts=bool(profile["store_psth_counts"]),
        store_spike_train_counts=bool(profile["store_spike_train_counts"]),
        window_pre_s=cfg.get("window_pre_s", 1.0),
        window_post_s=cfg.get("window_post_s", 1.0),
        use_parallel=cfg.get("use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        agents=cfg.get("agents"),
    )


def main(*, profile_name: str = "legacy_combined") -> None:
    profile = TRIAL_BUILD_PROFILES[profile_name]
    parser = argparse.ArgumentParser(description=str(profile["description"]))
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

    settings = _build_settings(
        dataset_cfg=args.dataset_cfg,
        ephys_cfg=args.ephys_cfg,
        fixation_cfg_path=args.ephys_fixation_psth_cfg,
        profile_name=profile_name,
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
            print("\n[example] No fixation trial output files found to preview.")
            return
        _print_trial_example(paths[0], max_bins=args.example_max_bins)


if __name__ == "__main__":
    main()
