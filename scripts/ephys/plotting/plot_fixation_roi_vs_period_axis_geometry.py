"""Plot ROI-vs-period axis-definition geometry diagram."""

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
    plot_fixation_roi_vs_period_axis_geometry,
)


def _normalize_axis_colors(raw) -> dict[str, str]:
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
        description="Plot ROI-vs-period axis-geometry formulas and vectors.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    settings = FixationROIVsPeriodFactorialPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get("roi_vs_period_plot_input_subdir", cfg.get("roi_vs_period_output_subdir", "ephys/psth/fixation_roi_vs_period_factorial")),
        input_filename=cfg.get("roi_vs_period_plot_input_filename", cfg.get("roi_vs_period_output_pickle_filename", "results.pkl")),
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
        axis_colors=_normalize_axis_colors(cfg.get("roi_vs_period_plot_axis_colors", {})),
    )
    out = plot_fixation_roi_vs_period_axis_geometry(
        settings,
        output_filename=cfg.get("roi_vs_period_plot_axis_geometry_output_filename", "roi_vs_period_axis_geometry"),
    )
    print(f"[plot] axis-geometry figure: {out.get('output_path')}")


if __name__ == "__main__":
    main()

