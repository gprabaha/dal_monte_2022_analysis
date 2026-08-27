"""Build fixation temporal-specificity metrics from average PSTHs."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_temporal_specificity import (
    TEMPORAL_SPECIFICITY_CONDITIONS,
    FixationTemporalSpecificitySettings,
    run_fixation_temporal_specificity_analysis,
)


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def _window_ms(cfg: dict) -> tuple[float, float]:
    raw = cfg.get("temporal_specificity_analysis_window_ms")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        start, stop = float(raw[0]), float(raw[1])
        return (start, stop) if start <= stop else (stop, start)
    return (-500.0, 500.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score the temporal specificity (concentration, single-peak dominance, "
            "sustainedness, fluctuation) of average fixation PSTHs."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--window-start-ms", type=float, default=None)
    parser.add_argument("--window-stop-ms", type=float, default=None)
    parser.add_argument("--no-store-traces", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)

    window_ms = _window_ms(cfg)
    if args.window_start_ms is not None:
        window_ms = (float(args.window_start_ms), window_ms[1])
    if args.window_stop_ms is not None:
        window_ms = (window_ms[0], float(args.window_stop_ms))

    settings = FixationTemporalSpecificitySettings(
        cfg_path=str(dataset_cfg_path),
        average_input_subdir=cfg.get(
            "temporal_specificity_average_input_subdir",
            cfg.get(
                "peakiness_average_input_subdir",
                cfg.get("plot_average_input_subdir", "ephys/psth/fixation_psth_averages"),
            ),
        ),
        average_input_filename=cfg.get(
            "temporal_specificity_average_input_filename",
            cfg.get(
                "peakiness_average_input_filename",
                cfg.get("plot_average_input_filename", "fixations_psth_10ms.pkl"),
            ),
        ),
        output_subdir=cfg.get(
            "temporal_specificity_output_subdir",
            "ephys/psth/fixation_temporal_specificity",
        ),
        condition_output_filename=cfg.get(
            "temporal_specificity_condition_output_filename",
            "unit_condition_temporal_specificity.csv",
        ),
        unit_output_filename=cfg.get(
            "temporal_specificity_unit_output_filename",
            "unit_temporal_specificity.csv",
        ),
        region_summary_filename=cfg.get(
            "temporal_specificity_region_summary_filename",
            "region_temporal_specificity_summary.csv",
        ),
        trace_output_filename=cfg.get(
            "temporal_specificity_trace_output_filename",
            "unit_condition_traces.pkl",
        ),
        output_pickle_filename=cfg.get("temporal_specificity_output_pickle_filename", "results.pkl"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        condition_order=tuple(
            cfg.get("temporal_specificity_condition_order", TEMPORAL_SPECIFICITY_CONDITIONS)
        ),
        analysis_window_ms=window_ms,
        baseline_quantile=float(cfg.get("temporal_specificity_baseline_quantile", 0.10)),
        peak_distance_ms=float(
            cfg.get(
                "temporal_specificity_peak_distance_ms",
                cfg.get("peakiness_peak_distance_ms", 30.0),
            )
        ),
        prominent_peak_fraction=float(
            cfg.get("temporal_specificity_prominent_peak_fraction", 0.25)
        ),
        sustained_threshold_fraction=float(
            cfg.get("temporal_specificity_sustained_threshold_fraction", 0.25)
        ),
        mass_fraction=float(cfg.get("temporal_specificity_mass_fraction", 0.50)),
        min_peak_z=float(cfg.get("temporal_specificity_min_peak_z", 3.0)),
        min_modulation_index=float(cfg.get("temporal_specificity_min_modulation_index", 0.05)),
        min_mean_fr_hz=float(
            cfg.get(
                "temporal_specificity_min_mean_fr_hz",
                cfg.get("peakiness_mean_rate_floor_hz", 0.5),
            )
        ),
        min_trials_per_condition=int(
            cfg.get(
                "temporal_specificity_min_trials_per_condition",
                cfg.get("peakiness_min_trials_per_condition", 1),
            )
        ),
        bin_size_ms_fallback=float(cfg.get("bin_size_ms", 10.0)),
        region_order=cfg.get(
            "temporal_specificity_region_order",
            cfg.get("peakiness_region_order"),
        ),
        store_traces=not args.no_store_traces,
    )

    result = run_fixation_temporal_specificity_analysis(
        settings,
        dates=args.date,
        regions=args.region,
    )
    unit_df = result.get("unit_specificity")
    condition_df = result.get("condition_specificity")
    region_df = result.get("region_summary")
    print(f"[analysis] temporal-specificity unit rows: {0 if unit_df is None else len(unit_df)}")
    print(
        f"[analysis] temporal-specificity condition rows: "
        f"{0 if condition_df is None else len(condition_df)}"
    )
    print(
        f"[analysis] temporal-specificity region-summary rows: "
        f"{0 if region_df is None else len(region_df)}"
    )
    print(f"[analysis] analysis window (ms): {list(settings.analysis_window_ms)}")
    print("[analysis] wrote fixation temporal-specificity outputs to configured analysis subdir")


if __name__ == "__main__":
    main()
