"""Plot region-by-window triangular populations for three-way fixation responses."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_three_way_selectivity_triangular import (
    DEFAULT_COLORS,
    FixationThreeWayTriangularPlotSettings,
    plot_fixation_three_way_selectivity_triangular,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one triangular population summary figure with regions as columns "
            "and analysis windows as rows."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--window", action="append", default=None)
    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--output-extension", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    condition_colors = cfg.get("plot_condition_colors", dict(DEFAULT_COLORS))
    if not isinstance(condition_colors, dict):
        condition_colors = dict(DEFAULT_COLORS)

    settings = FixationThreeWayTriangularPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity"),
        condition_summary_filename=cfg.get("selective_condition_summary_filename", "condition_window_means.csv"),
        output_subdir=cfg.get("selective_triangular_output_subdir", "ephys/psth/fixation_psth_selectivity_triangular"),
        output_filename=(
            args.output_filename
            if args.output_filename is not None
            else cfg.get("selective_triangular_output_filename", "population_triangular")
        ),
        output_extension=(
            args.output_extension
            if args.output_extension is not None
            else cfg.get("selective_triangular_output_extension", "png")
        ),
        output_dpi=cfg.get("selective_triangular_output_dpi", 220),
        min_units_per_panel=cfg.get("selective_triangular_min_units_per_panel", 1),
        point_size=cfg.get("selective_triangular_point_size", 18.0),
        point_alpha=cfg.get("selective_triangular_point_alpha", 0.68),
        marker_edge_width=cfg.get("selective_triangular_marker_edge_width", 0.28),
        draw_centroid=cfg.get("selective_triangular_draw_centroid", True),
        condition_colors={str(k): str(v) for k, v in condition_colors.items()},
    )

    result = plot_fixation_three_way_selectivity_triangular(
        settings,
        regions=args.region,
        windows=args.window,
    )
    if not result:
        print("[plot] no triangular figure was generated")
        return

    output_path = result.get("output_path")
    panels = result.get("panel_counts", [])
    nonempty_panels = sum(1 for row in panels if int(row.get("n_units", 0)) > 0)
    print(f"[plot] triangular figure: {output_path}")
    print(
        "[plot] panels with >=1 unit: "
        f"{nonempty_panels}/{len(panels)} "
        f"(regions={len(result.get('regions', []))}, windows={len(result.get('windows', []))})"
    )


if __name__ == "__main__":
    main()
