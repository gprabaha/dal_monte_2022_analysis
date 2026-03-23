"""Build per-unit fixation PSTH variability summaries from average traces."""

import argparse
from pathlib import Path
import sys
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_psth_variability import (
    FixationPSTHVariabilitySettings,
    run_fixation_psth_variability_analysis,
)


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def _resolve_time_window_ms(cfg: dict) -> tuple[Optional[float], Optional[float]]:
    raw = cfg.get("variability_time_window_ms")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None, None
    return float(raw[0]), float(raw[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-unit variability in average fixation PSTHs for "
            "interactive face, non-interactive face, and object conditions."
        ),
    )
    parser.add_argument("--dataset-cfg", default=str(_REPO_ROOT / "configs" / "dataset.yaml"))
    parser.add_argument(
        "--ephys-fixation-psth-cfg",
        default=str(_REPO_ROOT / "configs" / "ephys_fixation_psth.yaml"),
    )
    parser.add_argument("--date", action="append", default=None)
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)
    window_start_ms, window_stop_ms = _resolve_time_window_ms(cfg)

    settings = FixationPSTHVariabilitySettings(
        cfg_path=str(dataset_cfg_path),
        input_subdir=cfg.get("variability_input_subdir", cfg.get("plot_average_input_subdir", "ephys/psth/fixation_psth_averages")),
        input_filename=cfg.get(
            "variability_input_filename",
            cfg.get("plot_average_input_filename_split", cfg.get("plot_average_input_filename", "fixations_psth_10ms.pkl")),
        ),
        object_input_subdir=cfg.get(
            "variability_object_input_subdir",
            cfg.get("plot_average_object_input_subdir", "ephys/psth/fixation_psth_averages"),
        ),
        object_input_filename=cfg.get(
            "variability_object_input_filename",
            cfg.get("plot_average_object_input_filename", cfg.get("plot_average_input_filename_unsplit", "fixations_psth_10ms.pkl")),
        ),
        output_subdir=cfg.get("variability_output_subdir", "ephys/psth/fixation_psth_variability"),
        unit_summary_filename=cfg.get("variability_unit_summary_filename", "unit_condition_variability.csv"),
        within_region_stats_filename=cfg.get("variability_within_region_stats_filename", "within_region_condition_variability_stats.csv"),
        output_pickle_filename=cfg.get("variability_output_pickle_filename"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        variability_metric_name=cfg.get("variability_metric_name", "std_over_time_bins"),
        variability_metric_label=cfg.get("variability_metric_label", "SD of Mean FR"),
        variability_metric_unit=cfg.get("variability_metric_unit", "Hz"),
        variability_window_start_ms=window_start_ms,
        variability_window_stop_ms=window_stop_ms,
        pvalue_correction=cfg.get("variability_pvalue_correction", "fdr_bh"),
        alpha=float(cfg.get("variability_alpha", 0.05)),
        min_paired_units_per_region=int(cfg.get("variability_min_paired_units_per_region", 2)),
        bin_size_ms_fallback=float(cfg.get("bin_size_ms", 10.0)),
        window_pre_s_fallback=float(cfg.get("window_pre_s", 1.0)),
        window_post_s_fallback=float(cfg.get("window_post_s", 1.0)),
    )

    result = run_fixation_psth_variability_analysis(
        settings,
        dates=args.date if args.date else None,
    )
    print(f"[analysis] unit summary: {result['unit_summary_path']}")
    print(f"[analysis] within-region stats: {result['within_region_stats_path']}")
    if result["pickle_path"] is not None:
        print(f"[analysis] results pickle: {result['pickle_path']}")


if __name__ == "__main__":
    main()
