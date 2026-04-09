"""Build region-level fixation-condition dominance summaries."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_condition_dominance import (
    DOMINANCE_CONDITIONS,
    DOMINANCE_UNIT_SUBSETS,
    FixationConditionDominanceSettings,
    run_fixation_condition_dominance_analysis,
)


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build region-level fixation-condition dominance summaries.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--region", action="append", default=None)
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)
    window_ms = cfg.get("condition_dominance_window_ms", [-500.0, 500.0])
    if window_ms is None or len(window_ms) != 2:
        window_ms = [-500.0, 500.0]

    settings = FixationConditionDominanceSettings(
        cfg_path=str(dataset_cfg_path),
        average_input_subdir=cfg.get(
            "condition_dominance_average_input_subdir",
            cfg.get("plot_average_input_subdir", "ephys/psth/fixation_psth_averages"),
        ),
        average_input_filename=cfg.get(
            "condition_dominance_average_input_filename",
            cfg.get("plot_average_input_filename", "fixations_psth_10ms.pkl"),
        ),
        selectivity_input_subdir=cfg.get(
            "condition_dominance_selectivity_input_subdir",
            cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity"),
        ),
        selectivity_unit_summary_filename=cfg.get(
            "condition_dominance_selectivity_unit_summary_filename",
            cfg.get("selective_unit_summary_filename", "unit_selectivity.csv"),
        ),
        output_subdir=cfg.get(
            "condition_dominance_output_subdir",
            "ephys/psth/fixation_condition_dominance",
        ),
        unit_output_filename=cfg.get(
            "condition_dominance_unit_output_filename",
            "unit_condition_dominance.csv",
        ),
        region_summary_filename=cfg.get(
            "condition_dominance_region_summary_filename",
            "region_condition_dominance_summary.csv",
        ),
        output_pickle_filename=cfg.get(
            "condition_dominance_output_pickle_filename",
            "results.pkl",
        ),
        window_start_ms=float(window_ms[0]),
        window_stop_ms=float(window_ms[1]),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        min_trials_per_condition=cfg.get("condition_dominance_min_trials_per_condition", 1),
        tie_tolerance_hz=cfg.get("condition_dominance_tie_tolerance_hz", 1e-12),
        bin_size_ms_fallback=cfg.get("bin_size_ms", 10.0),
        condition_order=tuple(cfg.get("condition_dominance_condition_order", DOMINANCE_CONDITIONS)),
        unit_subset_order=tuple(cfg.get("condition_dominance_unit_subset_order", DOMINANCE_UNIT_SUBSETS)),
        region_order=cfg.get("condition_dominance_region_order"),
    )

    result = run_fixation_condition_dominance_analysis(
        settings,
        dates=args.date,
        regions=args.region,
    )
    unit_df = result.get("unit_dominance")
    summary_df = result.get("region_summary")
    print(f"[analysis] dominance unit rows: {0 if unit_df is None else len(unit_df)}")
    print(f"[analysis] dominance region-summary rows: {0 if summary_df is None else len(summary_df)}")
    print("[analysis] wrote fixation condition dominance outputs to configured analysis subdir")


if __name__ == "__main__":
    main()
