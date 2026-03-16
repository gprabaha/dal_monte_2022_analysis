"""Plot ROI-vs-period cross-region significant comparisons as node graphs."""

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
    plot_fixation_roi_vs_period_cross_region_graph,
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


def _normalize_axis_sources(raw_multi, raw_single) -> tuple[str, ...]:
    if isinstance(raw_multi, (list, tuple)):
        vals = [str(item).strip() for item in raw_multi if str(item).strip()]
    else:
        vals = [str(raw_single).strip()] if str(raw_single).strip() else ["cell_means"]
    out: list[str] = []
    for value in vals:
        if value not in out:
            out.append(value)
    return tuple(out) if out else ("cell_means",)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ROI-vs-period cross-region significant-axis graph summaries.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--window", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    axis_sources = _normalize_axis_sources(
        cfg.get("roi_vs_period_plot_axis_magnitude_sources"),
        cfg.get("roi_vs_period_plot_axis_magnitude_source", "cell_means"),
    )
    base_output = cfg.get(
        "roi_vs_period_plot_cross_region_graph_output_filename",
        "roi_vs_period_cross_region_graph",
    )
    all_outputs = []
    for axis_source in axis_sources:
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
            axis_magnitude_source=axis_source,
            axis_comparison_mode=cfg.get("roi_vs_period_axis_comparison_mode", "max_abs_across_windows"),
            region_order=_normalize_str_tuple(cfg.get("roi_vs_period_plot_region_order"), ("bla", "accg", "dmpfc", "ofc")),
            axis_order=_normalize_str_tuple(cfg.get("roi_vs_period_plot_axis_order"), ("face_object", "interactive_state", "cross_interaction")),
            axis_colors=_normalize_axis_colors(cfg.get("roi_vs_period_plot_axis_colors", {})),
        )
        suffix = f"__source={axis_source}" if len(axis_sources) > 1 or str(axis_source) != "cell_means" else ""
        outputs = plot_fixation_roi_vs_period_cross_region_graph(
            settings,
            regions=args.region,
            window=args.window,
            output_filename=f"{base_output}{suffix}",
        )
        all_outputs.extend(outputs)
    if not all_outputs:
        print("[plot] no ROI-vs-period cross-region graph figure generated")
        return
    for row in all_outputs:
        print(f"[plot] cross-region graph figure: {row.get('output_path')}")


if __name__ == "__main__":
    main()
