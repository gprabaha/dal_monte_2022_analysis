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
            "Plot one heatmap figure per fixation-pair preference index with "
            "one region subplot per panel."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--pair-label", action="append", default=None)
    parser.add_argument("--region", action="append", default=None)
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
        output_extension=cfg.get("selective_index_plot_output_extension", "png"),
        output_dpi=cfg.get("selective_index_plot_output_dpi", 220),
        include_only_pair_selective_units=cfg.get(
            "selective_index_plot_include_only_pair_selective_units",
            cfg.get("selective_index_plot_include_only_selective_units", True),
        ),
        region_order=cfg.get("selective_index_plot_region_order", ["BLA", "ACCg", "dmPFC", "OFC"]),
        default_pair_order=cfg.get("selective_index_plot_pair_order"),
        figure_width_in=cfg.get("selective_index_plot_figure_width_in", 8.5),
        figure_height_in=cfg.get("selective_index_plot_figure_height_in", 2.0),
        left_margin=cfg.get("selective_index_plot_left_margin", 0.04),
        right_margin=cfg.get("selective_index_plot_right_margin", 0.992),
        top_margin=cfg.get("selective_index_plot_top_margin", 0.86),
        bottom_margin=cfg.get("selective_index_plot_bottom_margin", 0.22),
        panel_wspace=cfg.get("selective_index_plot_panel_wspace", 0.18),
        show_suptitle=cfg.get("selective_index_plot_show_suptitle", False),
        colorbar_label=cfg.get("selective_index_plot_colorbar_label", "Preference Index (A-B)/(A+B)"),
        colorbar_fraction=cfg.get("selective_index_plot_colorbar_fraction", 0.02),
        colorbar_pad=cfg.get("selective_index_plot_colorbar_pad", 0.02),
    )

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
        print(f"[plot] {row.get('pair_label')}: {row.get('output_path')}")


if __name__ == "__main__":
    main()
