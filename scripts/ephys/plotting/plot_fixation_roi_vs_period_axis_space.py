"""Plot ROI-vs-period 3D axis-space regional surface summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_roi_vs_period_factorial import (
    FixationROIVsPeriodFactorialPlotSettings,
    plot_fixation_roi_vs_period_axis_space,
)


def _normalize_color_map(raw) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        k = str(key).strip()
        v = str(value).strip()
        if k and v:
            out[k] = v
    return out


def _normalize_str_tuple(raw, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return tuple(fallback)
    out = [str(item).strip() for item in raw if str(item).strip()]
    return tuple(out) if out else tuple(fallback)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ROI-vs-period 3D regional density sheets with 95% support "
            "and mean axis lines on the base plane."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--window", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    settings = FixationROIVsPeriodFactorialPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get(
            "roi_vs_period_plot_input_subdir",
            cfg.get("roi_vs_period_output_subdir", "ephys/psth/fixation_roi_vs_period_factorial"),
        ),
        input_filename=cfg.get(
            "roi_vs_period_plot_input_filename",
            cfg.get("roi_vs_period_output_pickle_filename", "results.pkl"),
        ),
        output_subdir=cfg.get(
            "roi_vs_period_plot_output_subdir",
            "ephys/psth/fixation_roi_vs_period_factorial/plots",
        ),
        output_extension=cfg.get("roi_vs_period_plot_output_extension", "pdf"),
        output_dpi=cfg.get("roi_vs_period_plot_output_dpi", 300),
        axis_magnitude_source=cfg.get("roi_vs_period_plot_axis_magnitude_source", "cell_means"),
        axis_comparison_mode=cfg.get("roi_vs_period_axis_comparison_mode", "averaged_across_windows"),
        region_order=_normalize_str_tuple(cfg.get("roi_vs_period_plot_region_order"), ("bla", "accg", "dmpfc", "ofc")),
        axis_order=_normalize_str_tuple(cfg.get("roi_vs_period_plot_axis_order"), ("face_object", "interactive_state", "cross_interaction")),
        axis_colors=_normalize_color_map(cfg.get("roi_vs_period_plot_axis_colors", {})),
        region_outline_colors=_normalize_color_map(cfg.get("roi_vs_period_plot_region_outline_colors", {})),
        axis_space_regions_letter_width_in=cfg.get("roi_vs_period_plot_axis_space_regions_letter_width_in", 8.5),
        axis_space_regions_letter_height_frac=cfg.get("roi_vs_period_plot_axis_space_regions_letter_height_frac", 0.30),
        axis_space_overlay_letter_width_in=cfg.get("roi_vs_period_plot_axis_space_overlay_letter_width_in", 8.5),
        axis_space_overlay_letter_height_frac=cfg.get("roi_vs_period_plot_axis_space_overlay_letter_height_frac", 0.30),
        axis_space_disk_alpha=cfg.get("roi_vs_period_plot_axis_space_disk_alpha", 0.35),
        axis_space_disk_layers=cfg.get("roi_vs_period_plot_axis_space_disk_layers", 14),
        axis_space_quantile_low=cfg.get("roi_vs_period_plot_axis_space_quantile_low", 0.025),
        axis_space_quantile_high=cfg.get("roi_vs_period_plot_axis_space_quantile_high", 0.975),
    )
    outputs = plot_fixation_roi_vs_period_axis_space(
        settings,
        regions=args.region,
        window=args.window,
        output_filename_regions=cfg.get(
            "roi_vs_period_plot_axis_space_regions_output_filename",
            "roi_vs_period_axis_space_regions",
        ),
    )
    if not outputs:
        print("[plot] no ROI-vs-period axis-space figures generated")
        return
    for row in outputs:
        print(f"[plot] axis-space figure ({row.get('kind')}): {row.get('output_path')}")


if __name__ == "__main__":
    main()
