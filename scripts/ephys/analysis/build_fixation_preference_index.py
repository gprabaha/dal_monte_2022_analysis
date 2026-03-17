"""Build per-bin fixation preference indices from trial or average PSTH outputs."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
_DEFAULT_DATASET_CFG = _REPO_ROOT / "configs" / "dataset.yaml"
_DEFAULT_EPHYS_FIX_PSTH_CFG = _REPO_ROOT / "configs" / "ephys_fixation_psth.yaml"

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_preference_index import (
    FixationPSTHPreferenceIndexSettings,
    run_fixation_preference_index_analysis,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_date_paths


def _as_str_list(values):
    if not values:
        return None
    return [str(v) for v in values if str(v).strip()]


def _scan_trial_files(dataset_cfg_path: str, modality: str, filename: str) -> tuple[Path, int, int, int]:
    dataset_cfg = load_config(dataset_cfg_path)
    root = Path(dataset_cfg["processed_data_root"])
    base_glob = root / "date=*" / "session=*" / str(modality) / "*.pkl"
    all_rows = list(root.glob(str(base_glob.relative_to(root))))
    matched_rows = [path for path in all_rows if path.name == str(filename)]
    shared_rows = [path for path in all_rows if path.name == "shared.pkl"]
    return root, len(all_rows), len(matched_rows), len(shared_rows)


def _scan_average_files(dataset_cfg_path: str, subdir: str, filename: str) -> tuple[Path, int]:
    dataset_cfg = load_config(dataset_cfg_path)
    root = Path(dataset_cfg["analysis_output_root"]) / str(subdir)
    rows = scan_analysis_date_paths(
        dataset_cfg,
        str(subdir),
        filename=str(filename),
    )
    return root, len(rows)


def _normalize_pair_name_overrides(raw):
    out = {}
    if not isinstance(raw, dict):
        return out
    for pair_label, index_name in raw.items():
        key = str(pair_label).strip()
        value = str(index_name).strip()
        if key and value:
            out[key] = value
    return out


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def _resolve_index_window_s_from_cfg(cfg: dict) -> tuple[float, float]:
    raw = cfg.get("selective_index_time_window_ms")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        windows = cfg.get("selective_windows_ms")
        if isinstance(windows, dict):
            full_fix = windows.get("full_fix")
            if isinstance(full_fix, (list, tuple)) and len(full_fix) == 2:
                raw = full_fix
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return -0.5, 0.5
    start_s = float(raw[0]) / 1000.0
    end_s = float(raw[1]) / 1000.0
    if start_s > end_s:
        start_s, end_s = end_s, start_s
    return start_s, end_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-bin fixation preference indices (A-B)/denominator for each "
            "configured fixation-pair comparison."
        ),
    )
    parser.add_argument("--dataset-cfg", default=str(_DEFAULT_DATASET_CFG))
    parser.add_argument("--ephys-fixation-psth-cfg", default=str(_DEFAULT_EPHYS_FIX_PSTH_CFG))
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--unit-uuid", action="append", default=None)
    parser.add_argument(
        "--normalization-mode",
        default=None,
        help=(
            "Preference-index denominator mode: "
            "'unit_max_sum' (default) or 'per_bin_sum'."
        ),
    )
    parser.add_argument(
        "--use-average-input",
        action="store_true",
        help="Use date-level average PSTH input (recommended for 50 ms / 25 ms index heatmaps).",
    )
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)
    index_window_start_s, index_window_end_s = _resolve_index_window_s_from_cfg(cfg)
    use_average_input = bool(cfg.get("selective_index_use_average_input", True) or args.use_average_input)
    use_combined_index_average = bool(
        cfg.get("selective_index_average_store_split_and_unsplit_together", False)
    )
    average_split_subdir = cfg.get(
        "selective_index_average_output_subdir",
        "ephys/psth/fixation_psth_averages",
    )
    combined_average_filename = cfg.get(
        "selective_index_average_output_filename_combined",
        cfg.get(
            "selective_index_average_output_filename",
            "fixations_psth_50ms_step_25ms.pkl",
        ),
    )
    if use_combined_index_average:
        average_split_filename = combined_average_filename
        average_object_filename = combined_average_filename
    else:
        average_split_filename = cfg.get(
            "selective_index_average_output_filename_split",
            "fixations_psth_50ms_step_25ms_split_by_interactive_state.pkl",
        )
        average_object_filename = cfg.get(
            "selective_index_average_output_filename_unsplit",
            "fixations_psth_50ms_step_25ms_unsplit_by_interactive_state.pkl",
        )
    average_object_subdir_raw = cfg.get("selective_index_average_object_output_subdir")
    if average_object_subdir_raw is None and average_object_filename is None:
        average_object_subdir = None
    else:
        average_object_subdir = (
            average_object_subdir_raw
            if average_object_subdir_raw is not None
            else average_split_subdir
        )

    settings = FixationPSTHPreferenceIndexSettings(
        cfg_path=str(dataset_cfg_path),
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        average_input_subdir=(
            average_split_subdir
            if use_average_input
            else None
        ),
        average_input_filename=average_split_filename,
        average_object_input_subdir=(average_object_subdir if use_average_input else None),
        average_object_input_filename=(average_object_filename if use_average_input else None),
        selectivity_input_subdir=cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity"),
        pair_summary_filename=cfg.get("selective_pair_summary_filename", "pair_selectivity.csv"),
        unit_summary_filename=cfg.get("selective_unit_summary_filename", "unit_selectivity.csv"),
        output_subdir=cfg.get("selective_index_output_subdir", "ephys/fixation_preference_index"),
        timeseries_filename=cfg.get("selective_index_timeseries_filename", "preference_index_timeseries.csv"),
        output_pickle_filename=cfg.get("selective_index_output_pickle_filename", "results.pkl"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        pair_index_name_overrides=_normalize_pair_name_overrides(cfg.get("selective_index_pair_names")),
        normalization_mode=cfg.get("selective_index_normalization_mode", "unit_max_sum"),
        denominator_epsilon=float(cfg.get("selective_index_denominator_epsilon", 0.0)),
        index_window_start_s=index_window_start_s,
        index_window_end_s=index_window_end_s,
        use_parallel=cfg.get("selective_index_use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        bin_size_ms_fallback=cfg.get("bin_size_ms", 10.0),
        window_pre_s_fallback=cfg.get("window_pre_s", 1.0),
        window_post_s_fallback=cfg.get("window_post_s", 1.0),
    )

    if use_average_input and settings.average_input_subdir:
        avg_root, n_avg = _scan_average_files(
            str(dataset_cfg_path),
            settings.average_input_subdir,
            settings.average_input_filename,
        )
        print(
            "[analysis] split-average file scan: "
            f"root={avg_root}, subdir={settings.average_input_subdir}, "
            f"filename={settings.average_input_filename}, matched={n_avg}"
        )
        if settings.average_object_input_subdir is not None:
            object_filename = (
                settings.average_object_input_filename
                if settings.average_object_input_filename is not None
                else settings.average_input_filename
            )
            obj_root, n_obj = _scan_average_files(
                str(dataset_cfg_path),
                settings.average_object_input_subdir,
                object_filename,
            )
            print(
                "[analysis] unsplit-object average file scan: "
                f"root={obj_root}, subdir={settings.average_object_input_subdir}, "
                f"filename={object_filename}, matched={n_obj}"
            )
    else:
        root, n_all, n_match, n_shared = _scan_trial_files(
            str(dataset_cfg_path),
            settings.trial_input_modality,
            settings.trial_input_filename,
        )
        if n_match == 0 and settings.trial_input_filename != "shared.pkl" and n_shared > 0:
            print(
                "[analysis] no files found for "
                f"{settings.trial_input_modality}/{settings.trial_input_filename}; "
                "falling back to shared.pkl."
            )
            settings.trial_input_filename = "shared.pkl"
            _, n_all, n_match, n_shared = _scan_trial_files(
                str(dataset_cfg_path),
                settings.trial_input_modality,
                settings.trial_input_filename,
            )

    print(
        "[analysis] config paths: "
        f"dataset={dataset_cfg_path}, "
        f"ephys_fixation_psth={ephys_fix_psth_cfg_path}"
    )
    if not use_average_input:
        print(
            "[analysis] trial file scan: "
            f"root={root}, modality={settings.trial_input_modality}, "
            f"filename={settings.trial_input_filename}, matched={n_match}, "
            f"all_pkls_in_modality={n_all}, shared_pkls={n_shared}"
        )

    if args.no_parallel:
        settings.use_parallel = False
    if args.test_single:
        settings.test_single = True
    if args.normalization_mode is not None:
        settings.normalization_mode = str(args.normalization_mode)

    result = run_fixation_preference_index_analysis(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
        unit_uuids=_as_str_list(args.unit_uuid),
    )
    timeseries_df = result.get("timeseries")
    if timeseries_df is None or timeseries_df.empty:
        print("[analysis] no preference-index rows were produced")
        print(
            "[analysis] ensure selectivity outputs exist in selective_output_subdir "
            "and index input points to valid fixation PSTH trial or average files."
        )
        return

    n_rows = int(len(timeseries_df))
    n_units = int(timeseries_df["unit_key"].nunique())
    n_selective_units = int(timeseries_df.loc[timeseries_df["is_selective_unit"], "unit_key"].nunique())
    n_selective_pairs = int(
        timeseries_df.loc[timeseries_df["is_selective_pair"], ["unit_key", "pair_label"]]
        .drop_duplicates()
        .shape[0]
    )
    print(f"[analysis] preference-index rows: {n_rows}")
    print(f"[analysis] units with index output: {n_units}")
    print(f"[analysis] selective units in output: {n_selective_units}")
    print(f"[analysis] selective unit-pair combinations in output: {n_selective_pairs}")
    print("[analysis] wrote fixation preference-index outputs to configured analysis subdir")


if __name__ == "__main__":
    main()
