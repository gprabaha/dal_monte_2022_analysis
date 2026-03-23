"""Plot region-wise fixation PSTH variability violins."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_psth_variability import (
    DEFAULT_CONDITION_COLORS,
    DEFAULT_CONDITION_LABELS,
    DEFAULT_CONDITION_ORDER,
    DEFAULT_REGION_ORDER,
    FixationPSTHVariabilityPlotSettings,
    plot_fixation_psth_variability_violins,
)


def _normalize_str_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        seq = raw
    else:
        seq = [raw]
    out: list[str] = []
    for item in seq:
        token = str(item).strip()
        if token:
            out.append(token)
    return out


def _normalize_label_map(raw) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        key_token = str(key).strip()
        value_token = str(value).strip()
        if key_token and value_token:
            out[key_token] = value_token
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot region-wise violins of condition-specific fixation PSTH variability.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--output-extension", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    condition_order = _normalize_str_list(
        cfg.get("variability_plot_condition_order", list(DEFAULT_CONDITION_ORDER))
    )
    if not condition_order:
        condition_order = list(DEFAULT_CONDITION_ORDER)
    region_order = _normalize_str_list(
        cfg.get("variability_plot_region_order", list(DEFAULT_REGION_ORDER))
    )
    if not region_order:
        region_order = list(DEFAULT_REGION_ORDER)

    condition_labels = dict(DEFAULT_CONDITION_LABELS)
    condition_labels.update(_normalize_label_map(cfg.get("variability_plot_condition_labels", {})))
    condition_colors = dict(DEFAULT_CONDITION_COLORS)
    condition_colors.update(
        _normalize_label_map(
            cfg.get(
                "variability_plot_condition_colors",
                cfg.get("plot_condition_colors", DEFAULT_CONDITION_COLORS),
            )
        )
    )

    settings = FixationPSTHVariabilityPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get("variability_plot_input_subdir", cfg.get("variability_output_subdir", "ephys/psth/fixation_psth_variability")),
        unit_summary_filename=cfg.get("variability_plot_unit_summary_filename", cfg.get("variability_unit_summary_filename", "unit_condition_variability.csv")),
        within_region_stats_filename=cfg.get("variability_plot_within_region_stats_filename", cfg.get("variability_within_region_stats_filename", "within_region_condition_variability_stats.csv")),
        output_subdir=cfg.get("variability_plot_output_subdir", "ephys/psth/fixation_psth_variability/plots"),
        output_filename=cfg.get("variability_plot_output_filename", "fixation_psth_variability_violin"),
        output_extension=(
            str(args.output_extension).strip()
            if args.output_extension is not None
            else str(cfg.get("variability_plot_output_extension", "pdf"))
        ),
        output_dpi=cfg.get("variability_plot_output_dpi", 220),
        region_order=tuple(region_order),
        condition_order=tuple(condition_order),
        condition_labels=condition_labels,
        condition_colors=condition_colors,
        y_label=str(cfg.get("variability_plot_y_label", "SD of Mean FR (Hz)")),
        figure_width_in=float(cfg.get("variability_plot_figure_width_in", 8.6)),
        figure_height_in=float(cfg.get("variability_plot_figure_height_in", 3.2)),
        left_margin=float(cfg.get("variability_plot_left_margin", 0.06)),
        right_margin=float(cfg.get("variability_plot_right_margin", 0.995)),
        top_margin=float(cfg.get("variability_plot_top_margin", 0.83)),
        bottom_margin=float(cfg.get("variability_plot_bottom_margin", 0.22)),
        wspace=float(cfg.get("variability_plot_wspace", 0.28)),
        min_units_per_region=int(cfg.get("variability_plot_min_units_per_region", 1)),
    )

    out = plot_fixation_psth_variability_violins(settings, regions=args.region)
    if out is None:
        print("[plot] no fixation PSTH variability figure was generated")
        return
    print(f"[plot] wrote fixation PSTH variability figure: {out['output_path']}")


if __name__ == "__main__":
    main()
