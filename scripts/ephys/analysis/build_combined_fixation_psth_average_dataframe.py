"""Combine date-level 10 ms fixation PSTH averages into one sliced dataframe pickle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_psth_average_combiner import (
    FixationPSTHAverageCombinerSettings,
    combine_fixation_psth_average_dataframes,
)


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine date-partitioned 10 ms fixation PSTH averages into one dataframe "
            "and save the matching sliced timeline."
        ),
    )
    parser.add_argument(
        "--dataset-cfg",
        default=str(_REPO_ROOT / "configs" / "dataset.yaml"),
    )
    parser.add_argument(
        "--ephys-fixation-psth-cfg",
        default=str(_REPO_ROOT / "configs" / "ephys_fixation_psth.yaml"),
    )
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument(
        "--partition",
        action="append",
        choices=("split", "unsplit"),
        default=None,
        help="Average partitions to include. Defaults to both split and unsplit.",
    )
    parser.add_argument("--window-start-ms", type=float, default=-500.0)
    parser.add_argument("--window-stop-ms", type=float, default=500.0)
    parser.add_argument(
        "--output-dataframe-filename",
        default="fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl",
    )
    parser.add_argument(
        "--output-timeline-filename",
        default="fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl",
    )
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)

    settings = FixationPSTHAverageCombinerSettings(
        cfg_path=str(dataset_cfg_path),
        input_subdir=cfg.get("average_output_subdir", "ephys/psth/fixation_psth_averages"),
        input_filename=cfg.get("plot_average_input_filename", "fixations_psth_10ms.pkl"),
        output_subdir=cfg.get("average_output_subdir", "ephys/psth/fixation_psth_averages"),
        output_dataframe_filename=str(args.output_dataframe_filename),
        output_timeline_filename=str(args.output_timeline_filename),
        window_start_s=float(args.window_start_ms) / 1000.0,
        window_stop_s=float(args.window_stop_ms) / 1000.0,
        partitions=tuple(args.partition) if args.partition else ("split", "unsplit"),
    )

    result = combine_fixation_psth_average_dataframes(
        settings,
        dates=args.date if args.date else None,
    )
    print(f"[analysis] combined dataframe: {result['dataframe_path']}")
    print(f"[analysis] timeline: {result['timeline_path']}")
    print(
        "[analysis] combined summary: "
        f"{result['n_rows']} rows across {result['n_dates_scanned']} date file(s), "
        f"{len(result['timeline_s_rel'])} timeline bins, "
        f"partitions={','.join(result['partitions'])}"
    )


if __name__ == "__main__":
    main()
