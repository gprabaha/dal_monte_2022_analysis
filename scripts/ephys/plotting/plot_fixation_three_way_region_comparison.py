"""Plot region-comparison heatmaps for three-way fixation selectivity."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_three_way_region_comparison import (
    FixationThreeWayRegionComparisonPlotSettings,
    plot_fixation_three_way_region_comparison_heatmaps,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-window region-comparison heatmaps from three-way "
            "fixation composition statistics."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--window", action="append", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    settings = FixationThreeWayRegionComparisonPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get(
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
        output_subdir=cfg.get(
            "selective_region_comparison_plot_output_subdir",
            "ephys/psth/fixation_psth_selectivity_region_comparison/plots",
        ),
        output_filename=cfg.get(
            "selective_region_comparison_plot_output_filename",
            "region_comparison_heatmaps",
        ),
        output_extension=cfg.get("selective_region_comparison_plot_output_extension", "png"),
        output_dpi=cfg.get("selective_region_comparison_plot_output_dpi", 220),
        alpha=cfg.get("selective_region_comparison_alpha", 0.05),
        pvalue_floor=cfg.get("selective_region_comparison_plot_pvalue_floor", 1e-6),
        annotation_max_regions=cfg.get("selective_region_comparison_plot_annotation_max_regions", 10),
    )

    out = plot_fixation_three_way_region_comparison_heatmaps(
        settings,
        regions=args.region,
        windows=args.window,
    )
    if not out:
        print("[plot] no region-comparison heatmap figure was generated")
        return

    print(f"[plot] region comparison figure: {out.get('output_path')}")
    print(
        "[plot] windows x regions: "
        f"{len(out.get('window_order', []))} x {len(out.get('region_order', []))}"
    )


if __name__ == "__main__":
    main()

