"""Aggregate per-session pair coordination into the tables the notebook reads.

Run after every session has been built. Writes scalar summaries, condition
comparisons and the zero-lag artifact diagnostics as CSVs beside the
per-session outputs.

    python scripts/ephys/analysis/build_fixation_pair_spike_coordination_summary.py
"""

import argparse

from dal_monte_2022_analysis.ephys.analysis.fixation_pair_spike_coordination import (
    build_pair_spike_coordination_settings_from_config,
    run_summary_build,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize neural pair spike coordination.")
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--coordination-cfg",
        default="configs/ephys_fixation_pair_spike_coordination.yaml",
    )
    parser.add_argument(
        "--metric",
        default="trial_shuffle_mean_effect_pm10ms",
        help="Effect column compared across conditions (must not scale with trial count).",
    )
    args = parser.parse_args()

    settings = build_pair_spike_coordination_settings_from_config(
        dataset_cfg_path=args.dataset_cfg,
        coordination_cfg_path=args.coordination_cfg,
    )
    outputs = run_summary_build(settings, metric=args.metric)
    for name, frame in outputs.items():
        print(f"  {name:42s} rows={len(frame)}")


if __name__ == "__main__":
    main()
