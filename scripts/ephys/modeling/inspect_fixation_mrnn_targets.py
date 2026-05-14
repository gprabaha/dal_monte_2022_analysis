"""Inspect fixation mRNN targets, PCA dimensions, and shuffle plans."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.ephys.modeling import (
    build_fixation_mrnn_targets,
    derive_internal_feature_order_by_region,
    derive_internal_region_order,
    load_fixation_mrnn_config,
    normalize_target_mode,
    settings_from_config,
    summarize_fixation_mrnn_pca,
    summarize_fixation_mrnn_shuffles,
    summarize_fixation_mrnn_targets,
)


def _print_frame(title: str, frame: pd.DataFrame, *, max_colwidth: int = 120) -> None:
    print(f"\n[{title}]")
    with pd.option_context(
        "display.max_rows",
        200,
        "display.max_columns",
        40,
        "display.width",
        240,
        "display.max_colwidth",
        max_colwidth,
    ):
        print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print fixation mRNN loaded-data snippets and shuffle diagnostics.",
    )
    parser.add_argument(
        "--mrnn-cfg",
        default=str(_REPO_ROOT / "configs" / "ephys_fixation_mrnn.yaml"),
    )
    parser.add_argument("--target-mode", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--region-seed", type=int, default=None)
    parser.add_argument("--feature-seed", type=int, default=None)
    parser.add_argument("--sample-timepoints", type=int, default=5)
    parser.add_argument("--sample-features", type=int, default=5)
    parser.add_argument("--no-region-shuffle", action="store_true")
    parser.add_argument("--no-feature-shuffle", action="store_true")
    args = parser.parse_args()

    cfg = load_fixation_mrnn_config(args.mrnn_cfg)
    settings = settings_from_config(
        cfg,
        overrides={
            "target_mode": args.target_mode,
            "seed": args.seed,
        },
    )
    target_mode = normalize_target_mode(settings.target_mode)

    targets = build_fixation_mrnn_targets(
        settings.dataset_cfg_path,
        input_subdir=settings.input_subdir,
        dataframe_filename=settings.dataframe_filename,
        timeline_filename=settings.timeline_filename,
        canonical_region_order=settings.canonical_region_order,
        normalize_targets=settings.normalize_targets,
        normalization_stabilizer=settings.normalization_stabilizer,
        pca_variance_threshold=settings.pca_variance_threshold,
    )

    print(f"[dataframe] {targets.dataframe_path}")
    print(f"[timeline] {targets.timeline_path}")
    print(f"[conditions] {targets.condition_names}")
    print(f"[input_shape] {targets.input_tensor.shape}")

    target_summary = summarize_fixation_mrnn_targets(
        targets,
        target_modes=(target_mode,),
        sample_timepoints=int(args.sample_timepoints),
        sample_features=int(args.sample_features),
    )
    _print_frame(
        "target snippets",
        target_summary[
            [
                "target_mode",
                "region",
                "condition",
                "condition_shape",
                "n_finite",
                "n_nan",
                "n_inf",
                "min",
                "max",
                "mean",
                "sample_values",
            ]
        ],
    )

    if target_mode == "region_pcs":
        _print_frame("pca dimensions", summarize_fixation_mrnn_pca(targets))

    canonical_features = targets.feature_order_for_mode(target_mode)
    region_seed = int(args.region_seed) if args.region_seed is not None else int(settings.seed)
    feature_seed = (
        int(args.feature_seed)
        if args.feature_seed is not None
        else int(settings.seed) + 1_000_003
    )
    internal_region_order = derive_internal_region_order(
        settings.canonical_region_order,
        seed=region_seed,
        shuffle=not bool(args.no_region_shuffle),
    )
    internal_feature_order = derive_internal_feature_order_by_region(
        canonical_features,
        seed=feature_seed,
        shuffle=not bool(args.no_feature_shuffle),
    )
    print(f"\n[region_order_shuffle_seed] {region_seed}")
    print(f"[feature_order_shuffle_seed] {feature_seed}")
    _print_frame(
        "shuffle summary",
        summarize_fixation_mrnn_shuffles(
            canonical_region_order=settings.canonical_region_order,
            internal_region_order=internal_region_order,
            canonical_feature_order_by_region=canonical_features,
            internal_feature_order_by_region=internal_feature_order,
            sample_features=int(args.sample_features),
        ),
    )


if __name__ == "__main__":
    main()
