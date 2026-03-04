"""Plot per-pair fixation preference-index heatmaps by region."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_preference_index_heatmap import (
    FixationPreferenceIndexHeatmapPlotSettings,
    plot_fixation_preference_index_heatmaps,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fixation-pair preference index heatmaps as a combined "
            "rows-by-pairs x columns-by-regions figure."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--pair-label", action="append", default=None)
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument(
        "--normalization-mode",
        default=None,
        help=(
            "Which stored preference-index normalization to plot: "
            "'unit_max_sum' (default) or 'per_bin_sum'."
        ),
    )
    parser.add_argument(
        "--unit-filter-mode",
        default=None,
        help=(
            "Unit filtering mode: "
            "'pair_selective' (only AB-selective units), "
            "'any_selective' (units selective for any pair), or "
            "'all' (no selectivity filter)."
        ),
    )
    parser.add_argument(
        "--sort-reference-pair",
        default=None,
        help=(
            "Optional pair_label to define shared unit sort order across all pair plots. "
            "Example: face_interactive__vs__face_non_interactive"
        ),
    )
    parser.add_argument(
        "--separate-pair-figures",
        action="store_true",
        help="Disable combined 3x4 output and write one figure per pair (legacy mode).",
    )
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    settings = FixationPreferenceIndexHeatmapPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get("selective_index_output_subdir", "ephys/psth/fixation_psth_preference_index"),
        timeseries_filename=cfg.get("selective_index_timeseries_filename", "preference_index_timeseries.csv"),
        output_subdir=cfg.get(
            "selective_index_plot_output_subdir",
            "ephys/psth/fixation_psth_preference_index/plots",
        ),
        output_filename=cfg.get("selective_index_plot_output_filename", "preference_index_heatmaps"),
        output_extension=cfg.get("selective_index_plot_output_extension", "pdf"),
        output_dpi=cfg.get("selective_index_plot_output_dpi", 220),
        include_only_pair_selective_units=cfg.get(
            "selective_index_plot_include_only_pair_selective_units",
            cfg.get("selective_index_plot_include_only_selective_units", True),
        ),
        unit_filter_mode=cfg.get("selective_index_plot_unit_filter_mode"),
        sort_reference_pair_label=cfg.get("selective_index_plot_sort_reference_pair"),
        selective_windows_for_significance=cfg.get(
            "selective_index_plot_selective_windows",
            cfg.get("selective_venn_selective_windows", ["pre_fix", "peri_fix", "post_fix"]),
        ),
        combine_pairs_into_single_figure=cfg.get("selective_index_plot_combine_pairs_into_single_figure", True),
        normalization_mode=cfg.get("selective_index_plot_normalization_mode", "unit_max_sum"),
        region_order=cfg.get("selective_index_plot_region_order", ["BLA", "ACCg", "dmPFC", "OFC"]),
        default_pair_order=cfg.get("selective_index_plot_pair_order"),
        figure_width_in=cfg.get("selective_index_plot_figure_width_in", 8.3),
        figure_height_in=cfg.get("selective_index_plot_figure_height_in", 4.4),
        left_margin=cfg.get("selective_index_plot_left_margin", 0.03),
        right_margin=cfg.get("selective_index_plot_right_margin", 0.995),
        top_margin=cfg.get("selective_index_plot_top_margin", 0.86),
        bottom_margin=cfg.get("selective_index_plot_bottom_margin", 0.26),
        panel_wspace=cfg.get("selective_index_plot_panel_wspace", 0.10),
        panel_hspace=cfg.get("selective_index_plot_panel_hspace", 0.24),
        show_suptitle=cfg.get("selective_index_plot_show_suptitle", False),
        colorbar_orientation=cfg.get("selective_index_plot_colorbar_orientation", "horizontal"),
        colorbar_label=cfg.get("selective_index_plot_colorbar_label", "|Preference Index|"),
        colorbar_fraction=cfg.get("selective_index_plot_colorbar_fraction", 0.025),
        colorbar_pad=cfg.get("selective_index_plot_colorbar_pad", 0.08),
        colorbar_shrink=cfg.get("selective_index_plot_colorbar_shrink", 0.72),
        colorbar_aspect=cfg.get("selective_index_plot_colorbar_aspect", 48.0),
    )
    if args.normalization_mode is not None:
        settings.normalization_mode = str(args.normalization_mode)
    if args.unit_filter_mode is not None:
        settings.unit_filter_mode = str(args.unit_filter_mode)
    if args.sort_reference_pair is not None:
        settings.sort_reference_pair_label = str(args.sort_reference_pair)
    if args.separate_pair_figures:
        settings.combine_pairs_into_single_figure = False

    out = plot_fixation_preference_index_heatmaps(
        settings,
        pair_labels=args.pair_label,
        regions=args.region,
    )
    if not out:
        print("[plot] no preference-index heatmaps were generated")
        return

    outputs = out.get("outputs", [])
    print(f"[plot] generated {len(outputs)} preference-index heatmap figure(s)")
    for row in outputs:
        if row.get("combined_pairs"):
            print(
                "[plot] combined pairs: "
                f"{', '.join(row.get('pair_labels', []))} -> {row.get('output_path')}"
            )
        else:
            print(f"[plot] {row.get('pair_label')}: {row.get('output_path')}")


if __name__ == "__main__":
    main()
