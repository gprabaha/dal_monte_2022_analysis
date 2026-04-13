"""Build fixation peakiness summaries from average PSTHs."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_peakiness import (
    PEAKINESS_CONDITIONS,
    FixationPeakinessSettings,
    run_fixation_peakiness_analysis,
)


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixation peakiness summaries from average fixation PSTHs.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--unit-uuid", action="append", default=None)
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)

    settings = FixationPeakinessSettings(
        cfg_path=str(dataset_cfg_path),
        average_input_subdir=cfg.get(
            "peakiness_average_input_subdir",
            cfg.get("plot_average_input_subdir", "ephys/psth/fixation_psth_averages"),
        ),
        average_input_filename=cfg.get(
            "peakiness_average_input_filename",
            cfg.get("plot_average_input_filename", "fixations_psth_10ms.pkl"),
        ),
        output_subdir=cfg.get("peakiness_output_subdir", "ephys/psth/fixation_peakiness"),
        unit_output_filename=cfg.get("peakiness_unit_output_filename", "unit_peakiness.csv"),
        condition_output_filename=cfg.get(
            "peakiness_condition_output_filename",
            "unit_condition_peakiness.csv",
        ),
        region_summary_filename=cfg.get(
            "peakiness_region_summary_filename",
            "region_peakiness_summary.csv",
        ),
        output_pickle_filename=cfg.get("peakiness_output_pickle_filename", "results.pkl"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        condition_order=tuple(cfg.get("peakiness_condition_order", PEAKINESS_CONDITIONS)),
        min_trials_per_condition=int(cfg.get("peakiness_min_trials_per_condition", 1)),
        mean_rate_floor_hz=float(cfg.get("peakiness_mean_rate_floor_hz", 0.5)),
        peak_distance_ms=float(cfg.get("peakiness_peak_distance_ms", 30.0)),
        peak_prominence_floor=float(cfg.get("peakiness_peak_prominence_floor", 0.0)),
        competition_penalty_lambda=float(cfg.get("peakiness_competition_penalty_lambda", 0.5)),
        prominence_epsilon=float(cfg.get("peakiness_prominence_epsilon", 1.0e-12)),
        bin_size_ms_fallback=float(cfg.get("bin_size_ms", 10.0)),
        region_order=cfg.get("peakiness_region_order"),
    )

    result = run_fixation_peakiness_analysis(
        settings,
        dates=args.date,
        regions=args.region,
        unit_uuids=args.unit_uuid,
    )
    unit_df = result.get("unit_peakiness")
    condition_df = result.get("condition_peakiness")
    region_df = result.get("region_summary")
    print(f"[analysis] peakiness unit rows: {0 if unit_df is None else len(unit_df)}")
    print(f"[analysis] peakiness condition rows: {0 if condition_df is None else len(condition_df)}")
    print(f"[analysis] peakiness region-summary rows: {0 if region_df is None else len(region_df)}")

    queried_df = result.get("queried_units")
    if queried_df is not None and hasattr(queried_df, "empty") and not queried_df.empty:
        display_cols = [
            "date",
            "unit_uuid",
            "region",
            "peakiness_score",
            "best_condition",
            "best_peak_latency_ms",
            "best_peak_prominence",
            "best_peak_dominance",
        ]
        available_cols = [col for col in display_cols if col in queried_df.columns]
        print("[analysis] queried unit peakiness:")
        print(queried_df.loc[:, available_cols].to_string(index=False))

    print("[analysis] wrote fixation peakiness outputs to configured analysis subdir")


if __name__ == "__main__":
    main()
