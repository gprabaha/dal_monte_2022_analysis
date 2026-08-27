"""Plot region-level fixation-condition dominance summaries."""

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
)
from dal_monte_2022_analysis.ephys.plotting.fixation_condition_dominance import (
    DEFAULT_CONDITION_COLORS,
    DEFAULT_CONDITION_LABELS,
    DEFAULT_SUBSET_LABELS,
    FixationConditionDominancePlotSettings,
    plot_fixation_condition_dominance_by_region,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot region-level fixation-condition dominance summaries.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--unit-subset", action="append", default=None)
    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--output-extension", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    settings = FixationConditionDominancePlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get(
            "condition_dominance_plot_input_subdir",
            cfg.get("condition_dominance_output_subdir", "ephys/psth/fixation_condition_dominance"),
        ),
        region_summary_filename=cfg.get(
            "condition_dominance_plot_region_summary_filename",
            cfg.get("condition_dominance_region_summary_filename", "region_condition_dominance_summary.csv"),
        ),
        output_subdir=cfg.get(
            "condition_dominance_plot_output_subdir",
            "ephys/psth/fixation_condition_dominance/plots",
        ),
        output_filename=(
            args.output_filename
            if args.output_filename is not None
            else cfg.get("condition_dominance_plot_output_filename", "condition_dominance_by_region")
        ),
        output_extension=(
            args.output_extension
            if args.output_extension is not None
            else cfg.get("condition_dominance_plot_output_extension", "pdf")
        ),
        output_dpi=cfg.get("condition_dominance_plot_output_dpi", 220),
        region_order=cfg.get("condition_dominance_plot_region_order", cfg.get("condition_dominance_region_order")),
        condition_order=tuple(cfg.get("condition_dominance_plot_condition_order", DOMINANCE_CONDITIONS)),
        unit_subset_order=tuple(cfg.get("condition_dominance_plot_unit_subset_order", DOMINANCE_UNIT_SUBSETS)),
        condition_labels=cfg.get("condition_dominance_plot_condition_labels", DEFAULT_CONDITION_LABELS),
        condition_colors=cfg.get("condition_dominance_plot_condition_colors", DEFAULT_CONDITION_COLORS),
        subset_labels=cfg.get("condition_dominance_plot_subset_labels", DEFAULT_SUBSET_LABELS),
        figure_width_in=cfg.get("condition_dominance_plot_figure_width_in"),
        figure_height_in=cfg.get("condition_dominance_plot_figure_height_in"),
        show_suptitle=cfg.get("condition_dominance_plot_show_suptitle", False),
        left_margin=cfg.get("condition_dominance_plot_left_margin", 0.06),
        right_margin=cfg.get("condition_dominance_plot_right_margin", 0.995),
        top_margin=cfg.get("condition_dominance_plot_top_margin", 0.88),
        bottom_margin=cfg.get("condition_dominance_plot_bottom_margin", 0.18),
        panel_wspace=cfg.get("condition_dominance_plot_panel_wspace", 0.18),
        panel_hspace=cfg.get("condition_dominance_plot_panel_hspace", 0.32),
        bar_width=cfg.get("condition_dominance_plot_bar_width", 0.72),
    )

    result = plot_fixation_condition_dominance_by_region(
        settings,
        regions=args.region,
        unit_subsets=args.unit_subset,
    )
    if not result:
        print("[plot] no dominance figure was generated")
        return
    print(f"[plot] dominance figure: {result.get('output_path')}")
    print(
        "[plot] subset x region panels: "
        f"{len(result.get('unit_subsets', []))} x {len(result.get('regions', []))}"
    )


if __name__ == "__main__":
    main()
