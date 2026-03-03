"""Build region-comparison summaries for three-way fixation selectivity."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_three_way_region_comparison import (
    FixationThreeWayRegionComparisonSettings,
    run_fixation_three_way_region_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare three-way fixation-response composition distributions across "
            "regions for each analysis window."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--window", action="append", default=None)
    parser.add_argument("--n-permutations", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    settings = FixationThreeWayRegionComparisonSettings(
        cfg_path=args.dataset_cfg,
        input_subdir=cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity"),
        condition_summary_filename=cfg.get("selective_condition_summary_filename", "condition_window_means.csv"),
        unit_summary_filename=cfg.get("selective_unit_summary_filename", "unit_selectivity.csv"),
        output_subdir=cfg.get(
            "selective_region_comparison_output_subdir",
            "ephys/psth/fixation_psth_selectivity_region_comparison",
        ),
        pairwise_summary_filename=cfg.get(
            "selective_region_comparison_pairwise_filename",
            "pairwise_region_comparisons.csv",
        ),
        window_summary_filename=cfg.get(
            "selective_region_comparison_window_filename",
            "window_region_comparisons.csv",
        ),
        output_pickle_filename=cfg.get(
            "selective_region_comparison_output_pickle_filename",
            "results.pkl",
        ),
        min_units_per_region=cfg.get("selective_region_comparison_min_units_per_region", 5),
        min_regions_per_window=cfg.get("selective_region_comparison_min_regions_per_window", 2),
        n_permutations=cfg.get("selective_region_comparison_n_permutations", 1000),
        random_seed=cfg.get("selective_region_comparison_random_seed", 42),
        pvalue_correction=cfg.get("selective_region_comparison_pvalue_correction", "fdr_bh"),
        alpha=cfg.get("selective_region_comparison_alpha", 0.05),
        require_all_conditions_observed=cfg.get(
            "selective_region_comparison_require_all_conditions_observed",
            True,
        ),
        require_meets_min_trials=cfg.get(
            "selective_region_comparison_require_meets_min_trials",
            False,
        ),
        require_selective_units=cfg.get(
            "selective_region_comparison_require_selective_units",
            False,
        ),
        pseudo_count=cfg.get("selective_region_comparison_pseudo_count", 1e-6),
        alignment_cosine_threshold=cfg.get(
            "selective_region_comparison_alignment_cosine_threshold",
            0.95,
        ),
    )
    if args.n_permutations is not None:
        settings.n_permutations = int(args.n_permutations)

    result = run_fixation_three_way_region_comparison(
        settings,
        regions=args.region,
        windows=args.window,
    )
    pair_df = result.get("pairwise_summary")
    win_df = result.get("window_summary")

    n_pairs = 0 if pair_df is None else int(len(pair_df))
    n_windows = 0 if win_df is None else int(len(win_df))
    print(f"[analysis] window summaries: {n_windows}")
    print(f"[analysis] pairwise region comparisons: {n_pairs}")
    if win_df is not None and not win_df.empty:
        n_sig_global = int(win_df["global_significant"].sum()) if "global_significant" in win_df.columns else 0
        print(f"[analysis] globally significant windows (adjusted): {n_sig_global}/{len(win_df)}")
        if "global_dispersion_significant" in win_df.columns:
            n_sig_disp = int(win_df["global_dispersion_significant"].sum())
            print(f"[analysis] global dispersion-significant windows (adjusted): {n_sig_disp}/{len(win_df)}")
        if "global_alignment_significant" in win_df.columns:
            n_sig_align = int(win_df["global_alignment_significant"].sum())
            print(f"[analysis] global alignment-significant windows (adjusted): {n_sig_align}/{len(win_df)}")
    print("[analysis] wrote region comparison outputs to configured analysis subdir")


if __name__ == "__main__":
    main()
