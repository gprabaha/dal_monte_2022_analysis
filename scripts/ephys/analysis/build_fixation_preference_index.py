"""Build per-bin fixation preference indices from trial PSTH outputs."""

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-bin fixation preference indices (A-B)/(A+B) for each "
            "configured fixation-pair comparison."
        ),
    )
    parser.add_argument("--dataset-cfg", default=str(_DEFAULT_DATASET_CFG))
    parser.add_argument("--ephys-fixation-psth-cfg", default=str(_DEFAULT_EPHYS_FIX_PSTH_CFG))
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--unit-uuid", action="append", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)

    settings = FixationPSTHPreferenceIndexSettings(
        cfg_path=str(dataset_cfg_path),
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        selectivity_input_subdir=cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity"),
        pair_summary_filename=cfg.get("selective_pair_summary_filename", "pair_selectivity.csv"),
        unit_summary_filename=cfg.get("selective_unit_summary_filename", "unit_selectivity.csv"),
        output_subdir=cfg.get("selective_index_output_subdir", "ephys/psth/fixation_psth_preference_index"),
        timeseries_filename=cfg.get("selective_index_timeseries_filename", "preference_index_timeseries.csv"),
        output_pickle_filename=cfg.get("selective_index_output_pickle_filename", "results.pkl"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        pair_index_name_overrides=_normalize_pair_name_overrides(cfg.get("selective_index_pair_names")),
        denominator_epsilon=float(cfg.get("selective_index_denominator_epsilon", 0.0)),
        use_parallel=cfg.get("selective_index_use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        bin_size_ms_fallback=cfg.get("bin_size_ms", 10.0),
        window_pre_s_fallback=cfg.get("window_pre_s", 1.0),
        window_post_s_fallback=cfg.get("window_post_s", 1.0),
    )

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
            "and trial_input points to fixation PSTH trial files."
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

