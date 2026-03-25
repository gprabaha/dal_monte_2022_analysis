"""Build fixation PSTH Fano-factor summaries from 10 ms trial counts."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_psth_fano_factor import (
    FixationPSTHFanoFactorSettings,
    run_fixation_psth_fano_factor_analysis,
)


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-unit fixation PSTH Fano-factor timeseries from "
            "10 ms trial spike-count PSTHs."
        ),
    )
    parser.add_argument("--dataset-cfg", default=str(_REPO_ROOT / "configs" / "dataset.yaml"))
    parser.add_argument(
        "--ephys-fixation-psth-cfg",
        default=str(_REPO_ROOT / "configs" / "ephys_fixation_psth.yaml"),
    )
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--session", action="append", default=None)
    subset_group = parser.add_mutually_exclusive_group()
    subset_group.add_argument("--all-units", action="store_true")
    subset_group.add_argument("--selective-units", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)
    region_summary_unit_subset = cfg.get(
        "fano_factor_region_summary_unit_subset",
        "any_selective_unit",
    )
    if args.all_units:
        region_summary_unit_subset = "all_units"
    elif args.selective_units:
        region_summary_unit_subset = "any_selective_unit"

    settings = FixationPSTHFanoFactorSettings(
        cfg_path=str(dataset_cfg_path),
        trial_input_modality=cfg.get("fano_factor_trial_input_modality", "psth"),
        trial_input_filename=cfg.get("fano_factor_trial_input_filename", "fixations_psth_10ms.pkl"),
        output_subdir=cfg.get("fano_factor_output_subdir", "ephys/psth/fixation_psth_fano_factor"),
        unit_timeseries_filename=cfg.get("fano_factor_unit_timeseries_filename", "unit_fano_factor_timeseries.csv"),
        region_summary_filename=cfg.get("fano_factor_region_summary_filename", "region_fano_factor_summary.csv"),
        output_pickle_filename=cfg.get("fano_factor_output_pickle_filename", "results.pkl"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        selectivity_input_subdir=cfg.get("fano_factor_selectivity_input_subdir", cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity")),
        selectivity_unit_summary_filename=cfg.get("fano_factor_selectivity_unit_summary_filename", cfg.get("selective_unit_summary_filename", "unit_selectivity.csv")),
        region_summary_unit_subset=str(region_summary_unit_subset),
        min_trials_per_condition=int(cfg.get("fano_factor_min_trials_per_condition", 2)),
        variance_ddof=int(cfg.get("fano_factor_variance_ddof", 1)),
        mean_epsilon=float(cfg.get("fano_factor_mean_epsilon", 1e-12)),
        bin_size_ms_fallback=float(cfg.get("bin_size_ms", 10.0)),
        window_pre_s_fallback=float(cfg.get("window_pre_s", 1.0)),
        window_post_s_fallback=float(cfg.get("window_post_s", 1.0)),
    )

    result = run_fixation_psth_fano_factor_analysis(
        settings,
        dates=args.date if args.date else None,
        sessions=args.session if args.session else None,
    )
    print(f"[analysis] unit timeseries: {result['unit_timeseries_path']}")
    print(f"[analysis] region summary: {result['region_summary_path']}")
    print(f"[analysis] region-summary unit subset: {result['meta']['region_summary_unit_subset']}")
    if result["pickle_path"] is not None:
        print(f"[analysis] results pickle: {result['pickle_path']}")


if __name__ == "__main__":
    main()
