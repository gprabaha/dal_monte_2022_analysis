"""Aggregate per-session pair coordination into the tables the notebook reads.

Run after every session has been built. Writes scalar summaries, condition
comparisons and the zero-lag artifact diagnostics as CSVs beside the
per-session outputs.

    python scripts/ephys/analysis/build_fixation_pair_spike_coordination_summary.py
"""

import argparse
from pathlib import Path

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.ephys.analysis.fixation_pair_spike_coordination import (
    build_group_z_traces,
    build_pair_inventory,
    build_pair_spike_coordination_settings_from_config,
    build_zero_lag_diagnostics,
    compare_conditions,
    load_pair_coordination,
    summarize_coordination,
    test_against_null,
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
    cfg = load_config(args.dataset_cfg)
    out_dir: Path = build_analysis_output_dir(cfg, settings.output_subdir) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, _ = load_pair_coordination(
        args.dataset_cfg,
        output_subdir=settings.output_subdir,
        output_filename=settings.output_filename,
    )
    print(f"pair-condition rows loaded: {len(pairs):,}")

    outputs = {
        "pair_inventory.csv": build_pair_inventory(pairs),
        "coordination_summary.csv": summarize_coordination(pairs, metric=args.metric),
        "coordination_vs_null.csv": test_against_null(pairs, metric=args.metric),
        "condition_comparisons.csv": compare_conditions(pairs, metric=args.metric),
        "condition_comparisons_selective.csv": compare_conditions(
            pairs.loc[pairs["both_selective"]], metric=args.metric
        ),
        "zero_lag_diagnostics.csv": build_zero_lag_diagnostics(pairs),
    }
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        print(f"  wrote {path.name:38s} rows={len(frame)}")

    # Group-mean lag traces, streamed rather than loaded, for the trace figures.
    for stem, kwargs in {
        "group_z_traces_by_scope": {"group_columns": ("scope", "condition")},
        "group_z_traces_by_region_pair": {
            "group_columns": ("scope", "region_pair", "condition")
        },
        "group_z_traces_selective": {
            "group_columns": ("scope", "condition"),
            "selective_only": True,
        },
    }.items():
        payload = build_group_z_traces(
            args.dataset_cfg,
            output_subdir=settings.output_subdir,
            output_filename=settings.output_filename,
            **kwargs,
        )
        path = out_dir / f"{stem}.pkl"
        import pandas as pd

        pd.to_pickle(payload, path)
        print(f"  wrote {path.name:38s} rows={len(payload['traces'])}")


if __name__ == "__main__":
    main()
