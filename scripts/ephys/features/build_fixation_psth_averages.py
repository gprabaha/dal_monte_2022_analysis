"""Build date-level fixation PSTH averages from trial PSTH outputs.

This legacy entrypoint preserves the original average builder configuration.
Newer explicit entrypoints split the 10 ms and 50 ms / 25 ms average outputs
into separate files while reusing the same averaging implementation.
"""

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


AVERAGE_BUILD_PROFILES = {
    "legacy_combined": {
        "description": "Build date-level averaged fixation PSTH features (legacy average builder).",
        "config_prefix": "average",
        "trial_input_filename": None,
        "combined_output_filename": None,
        "categories_key": "categories",
        "restrict_interactive_state_key": "restrict_interactive_state",
        "group_by_session_key": "group_by_session",
        "smooth_before_average_key": "smooth_before_average",
        "smoothing_sigma_ms_key": "smoothing_sigma_ms",
        "convert_to_firing_rate_key": "average_convert_to_firing_rate_before_average",
        "store_together_key": "average_store_split_and_unsplit_together",
        "use_cfg_target_bins": True,
    },
    "average_10ms": {
        "description": "Build date-level averaged fixation PSTHs from explicit 10 ms trial spike counts.",
        "config_prefix": "average",
        "trial_input_filename": "fixations_psth_10ms.pkl",
        "combined_output_filename": "fixations_psth_10ms.pkl",
        "categories_key": "categories",
        "restrict_interactive_state_key": "restrict_interactive_state",
        "group_by_session_key": "group_by_session",
        "smooth_before_average_key": "smooth_before_average",
        "smoothing_sigma_ms_key": "smoothing_sigma_ms",
        "convert_to_firing_rate_key": "average_convert_to_firing_rate_before_average",
        "store_together_key": "average_store_split_and_unsplit_together",
        "use_cfg_target_bins": False,
    },
    "average_50ms_step_25ms": {
        "description": "Build date-level averaged fixation PSTHs from explicit 50 ms / 25 ms trial spike counts.",
        "config_prefix": "selective_index_average",
        "trial_input_filename": "fixations_psth_50ms_step_25ms.pkl",
        "combined_output_filename": "fixations_psth_50ms_step_25ms.pkl",
        "categories_key": "selective_index_average_categories",
        "restrict_interactive_state_key": "selective_index_average_restrict_interactive_state",
        "group_by_session_key": "selective_index_average_group_by_session",
        "smooth_before_average_key": "selective_index_average_smooth_before_average",
        "smoothing_sigma_ms_key": "selective_index_average_smoothing_sigma_ms",
        "convert_to_firing_rate_key": "selective_index_average_convert_to_firing_rate_before_average",
        "store_together_key": "selective_index_average_store_split_and_unsplit_together",
        "use_cfg_target_bins": False,
    },
}


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


def _derive_split_filename(base_filename: str) -> str:
    token = _ensure_pkl_filename(base_filename)
    stem = token[:-4]
    return f"{stem}_split_by_interactive_state.pkl"


def _default_output_subdir(prefix: str) -> str:
    if prefix == "selective_index_average":
        return "ephys/psth/fixation_preference_index_input_averages"
    return "ephys/psth/fixation_psth_averages"


def _resolve_subdirs(cfg: dict, prefix: str) -> tuple[str, str, str]:
    default_subdir = _default_output_subdir(prefix)
    combined_output_subdir = cfg.get(
        f"{prefix}_output_subdir_combined",
        cfg.get(f"{prefix}_output_subdir", default_subdir),
    )
    split_output_subdir = cfg.get(
        f"{prefix}_output_subdir_split",
        cfg.get(f"{prefix}_output_subdir", default_subdir),
    )
    unsplit_output_subdir = cfg.get(f"{prefix}_output_subdir_unsplit", split_output_subdir)
    return (
        str(combined_output_subdir),
        str(split_output_subdir),
        str(unsplit_output_subdir),
    )


def _resolve_legacy_output_filenames(cfg: dict, prefix: str) -> tuple[str, str, str]:
    combined_output_filename_raw = cfg.get(f"{prefix}_output_filename_combined")
    if combined_output_filename_raw is None:
        legacy_output_filename = _ensure_pkl_filename(
            cfg.get(f"{prefix}_output_filename", "fixations.pkl")
        )
        if (
            legacy_output_filename.endswith("_split_by_interactive_state.pkl")
            or legacy_output_filename.endswith("_unsplit_by_interactive_state.pkl")
        ):
            combined_output_filename_raw = "fixations.pkl"
        else:
            combined_output_filename_raw = legacy_output_filename
    combined_output_filename = _ensure_pkl_filename(combined_output_filename_raw)

    split_output_filename = _ensure_pkl_filename(
        cfg.get(
            f"{prefix}_output_filename_split",
            cfg.get(f"{prefix}_output_filename", "fixations.pkl"),
        ),
    )
    unsplit_output_filename = _ensure_pkl_filename(
        cfg.get(
            f"{prefix}_output_filename_unsplit",
            _derive_unsplit_filename(split_output_filename),
        ),
    )
    return combined_output_filename, split_output_filename, unsplit_output_filename


def _resolve_profile_output_filenames(profile_name: str, cfg: dict, prefix: str) -> tuple[str, str, str]:
    profile = AVERAGE_BUILD_PROFILES[profile_name]
    if profile_name == "legacy_combined":
        return _resolve_legacy_output_filenames(cfg, prefix)

    combined_output_filename = _ensure_pkl_filename(str(profile["combined_output_filename"]))
    split_output_filename = _derive_split_filename(combined_output_filename)
    unsplit_output_filename = _derive_unsplit_filename(split_output_filename)
    return combined_output_filename, split_output_filename, unsplit_output_filename


def _print_average_example(path: Path, *, max_bins: int = 12) -> None:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "averages_split_by_interactive_state" in obj:
        top_meta = obj.get("meta", {})
        split_df = obj.get("averages_split_by_interactive_state")
        unsplit_df = obj.get("averages_unsplit_by_interactive_state")
        print("\nExample fixation PSTH average output (combined split+unsplit file):")
        print(f"  file: {path}")
        print(
            "  meta: "
            f"smooth_before_average={top_meta.get('smooth_before_average')}, "
            f"smoothing_sigma_ms={top_meta.get('smoothing_sigma_ms')}, "
            f"convert_to_firing_rate_before_average={top_meta.get('convert_to_firing_rate_before_average')}, "
            f"psth_value_kind={top_meta.get('psth_value_kind')}"
        )
        if isinstance(split_df, pd.DataFrame):
            print(f"  split_rows: {len(split_df)}")
        if isinstance(unsplit_df, pd.DataFrame):
            print(f"  unsplit_rows: {len(unsplit_df)}")

        def _preview_partition(df: pd.DataFrame, label: str) -> None:
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f"  [{label}] empty")
                return
            row = df.iloc[0]
            psth_mean = np.asarray(row.get("psth_mean"), dtype=float).reshape(-1)
            psth_sem = np.asarray(row.get("psth_sem"), dtype=float).reshape(-1)
            preview = psth_mean[: max(1, int(max_bins))]
            sem_preview = psth_sem[: max(1, int(max_bins))]
            print(
                f"  [{label}] sample_row: "
                f"date={row.get('date')}, unit_uuid={row.get('unit_uuid')}, "
                f"category={row.get('fixation_category')}, n_trials={row.get('n_trials')}, "
                f"interactive_state={row.get('interactive_state') if 'interactive_state' in df.columns else None}"
            )
            print(f"  [{label}] sample_psth_mean_first_{len(preview)}bins: {preview.tolist()}")
            print(f"  [{label}] sample_psth_sem_first_{len(sem_preview)}bins: {sem_preview.tolist()}")

        _preview_partition(split_df, "split")
        _preview_partition(unsplit_df, "unsplit")
        return

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
    psth_sem = np.asarray(row.get("psth_sem"), dtype=float).reshape(-1)
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
            f"split_by_interactive_state={meta.get('split_by_interactive_state')}, "
            f"convert_to_firing_rate_before_average={meta.get('convert_to_firing_rate_before_average')}, "
            f"psth_value_kind={meta.get('psth_value_kind')}"
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
    print(f"  sample_psth_sem_first_{len(sem_preview)}bins: {sem_preview.tolist()}")


def _build_settings_common(
    *,
    dataset_cfg: str,
    fixation_cfg_path: str,
    profile_name: str,
) -> tuple[
    FixationPSTHAverageSettings,
    str,
    str,
    str,
    str,
    str,
    str,
]:
    cfg = load_config(fixation_cfg_path)
    profile = AVERAGE_BUILD_PROFILES[profile_name]
    prefix = str(profile["config_prefix"])

    combined_output_subdir, split_output_subdir, unsplit_output_subdir = _resolve_subdirs(cfg, prefix)
    combined_output_filename, split_output_filename, unsplit_output_filename = _resolve_profile_output_filenames(
        profile_name,
        cfg,
        prefix,
    )

    trial_input_filename = profile["trial_input_filename"]
    if trial_input_filename is None:
        trial_input_filename = cfg.get("trial_output_filename", "fixations.pkl")

    if bool(profile["use_cfg_target_bins"]):
        target_bin_size_ms = cfg.get(f"{prefix}_target_bin_size_ms")
        target_bin_step_ms = cfg.get(f"{prefix}_target_bin_step_ms")
    else:
        target_bin_size_ms = None
        target_bin_step_ms = None

    settings_common = FixationPSTHAverageSettings(
        cfg_path=dataset_cfg,
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=str(trial_input_filename),
        output_subdir=combined_output_subdir,
        output_filename=combined_output_filename,
        split_by_interactive_state=True,
        store_split_and_unsplit_together=cfg.get(str(profile["store_together_key"]), True),
        restrict_interactive_state=cfg.get(str(profile["restrict_interactive_state_key"])),
        group_by_session=cfg.get(str(profile["group_by_session_key"]), False),
        smooth_before_average=cfg.get(str(profile["smooth_before_average_key"]), True),
        smoothing_sigma_ms=cfg.get(str(profile["smoothing_sigma_ms_key"]), 20.0),
        convert_to_firing_rate_before_average=cfg.get(
            str(profile["convert_to_firing_rate_key"]),
            True,
        ),
        target_bin_size_ms=target_bin_size_ms,
        target_bin_step_ms=target_bin_step_ms,
        use_parallel=cfg.get("average_use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        categories=cfg.get(str(profile["categories_key"]), ("face", "object", "out_of_roi")),
    )
    return (
        settings_common,
        combined_output_subdir,
        combined_output_filename,
        split_output_subdir,
        split_output_filename,
        unsplit_output_subdir,
        unsplit_output_filename,
    )


def main(*, profile_name: str = "legacy_combined") -> None:
    profile = AVERAGE_BUILD_PROFILES[profile_name]
    parser = argparse.ArgumentParser(description=str(profile["description"]))
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
    parser.add_argument(
        "--store-separate-files",
        action="store_true",
        help=(
            "Legacy mode: write split and unsplit averages as separate files "
            "instead of one combined output file."
        ),
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

    (
        settings_common,
        combined_output_subdir,
        combined_output_filename,
        split_output_subdir,
        split_output_filename,
        unsplit_output_subdir,
        unsplit_output_filename,
    ) = _build_settings_common(
        dataset_cfg=args.dataset_cfg,
        fixation_cfg_path=args.ephys_fixation_psth_cfg,
        profile_name=profile_name,
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
    legacy_separate_mode = bool(
        args.store_separate_files
        or args.split_only
        or args.unsplit_only
        or args.split_by_interactive_state
    )

    if args.output_subdir:
        combined_output_subdir = str(args.output_subdir)
        split_output_subdir = str(args.output_subdir)
        unsplit_output_subdir = str(args.output_subdir)
    if args.output_subdir_unsplit:
        unsplit_output_subdir = str(args.output_subdir_unsplit)
    if args.output_filename:
        combined_output_filename = _ensure_pkl_filename(args.output_filename)
        split_output_filename = _ensure_pkl_filename(args.output_filename)
    if args.output_filename_unsplit:
        unsplit_output_filename = _ensure_pkl_filename(args.output_filename_unsplit)

    settings_by_label: list[tuple[str, FixationPSTHAverageSettings]] = []
    if legacy_separate_mode:
        if run_split_output:
            settings_by_label.append(
                (
                    "split",
                    replace(
                        settings_common,
                        output_subdir=split_output_subdir,
                        output_filename=split_output_filename,
                        split_by_interactive_state=True,
                        store_split_and_unsplit_together=False,
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
                        store_split_and_unsplit_together=False,
                    ),
                ),
            )
        if not settings_by_label:
            raise ValueError("No average output mode selected.")
    else:
        settings_by_label = [
            (
                "combined",
                replace(
                    settings_common,
                    output_subdir=combined_output_subdir,
                    output_filename=combined_output_filename,
                    store_split_and_unsplit_together=True,
                ),
            )
        ]

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
                continue
            print(f"\n[example] Previewing {label} output")
            _print_average_example(paths[0], max_bins=args.example_max_bins)
            any_examples = True
        if not any_examples:
            print("\n[example] No fixation PSTH average output files found to preview.")


if __name__ == "__main__":
    main()
