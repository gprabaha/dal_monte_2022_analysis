"""Plot comparison-group fixation selectivity summaries across regions."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_selectivity_comparison_group import (
    DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_COLORS,
    DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_LABELS,
    DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_ORDER,
    FixationSelectivityComparisonGroupPlotSettings,
    plot_fixation_selectivity_comparison_group_summaries,
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
        description=(
            "Plot region-wise fixation selectivity summaries for a named "
            "comparison group."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--comparison-label", default=None)
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--output-extension", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    pair_order = _normalize_str_list(
        cfg.get(
            "selective_comparison_plot_pair_order",
            list(DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_ORDER),
        )
    )
    if not pair_order:
        pair_order = list(DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_ORDER)

    pair_labels = dict(DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_LABELS)
    pair_labels.update(_normalize_label_map(cfg.get("selective_comparison_plot_pair_labels", {})))

    pair_colors = dict(DEFAULT_INTERACTIVE_STATE_MATCHED_PAIR_COLORS)
    pair_colors.update(_normalize_label_map(cfg.get("selective_comparison_plot_pair_colors", {})))

    settings = FixationSelectivityComparisonGroupPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity"),
        pair_summary_filename=cfg.get("selective_pair_summary_filename", "pair_selectivity.csv"),
        output_subdir=cfg.get(
            "selective_comparison_plot_output_subdir",
            "ephys/psth/fixation_psth_selectivity_comparison_group_plots",
        ),
        comparison_label=(
            str(args.comparison_label).strip()
            if args.comparison_label is not None
            else str(cfg.get("selective_comparison_plot_comparison_label", "interactive_state_matched"))
        ),
        output_extension=(
            str(args.output_extension).strip()
            if args.output_extension is not None
            else str(cfg.get("selective_comparison_plot_output_extension", "pdf"))
        ),
        output_dpi=cfg.get("selective_comparison_plot_output_dpi", 220),
        selective_windows=cfg.get("selective_comparison_plot_selective_windows", ["pre_fix", "peri_fix", "post_fix"]),
        region_order=cfg.get("selective_comparison_plot_region_order", ["BLA", "ACCg", "dmPFC", "OFC"]),
        pair_order=pair_order,
        pair_labels=pair_labels,
        pair_colors=pair_colors,
        min_units_per_region=cfg.get("selective_comparison_plot_min_units_per_region", 1),
        fraction_bar_output_filename=cfg.get("selective_comparison_plot_fraction_bar_output_filename"),
        overlap_matrix_output_filename=cfg.get("selective_comparison_plot_overlap_output_filename"),
        fraction_bar_figure_width_in=cfg.get("selective_comparison_plot_fraction_bar_figure_width_in", 8.5),
        fraction_bar_figure_height_in=cfg.get("selective_comparison_plot_fraction_bar_figure_height_in", 2.9),
        fraction_bar_left_margin=cfg.get("selective_comparison_plot_fraction_bar_left_margin", 0.05),
        fraction_bar_right_margin=cfg.get("selective_comparison_plot_fraction_bar_right_margin", 0.995),
        fraction_bar_top_margin=cfg.get("selective_comparison_plot_fraction_bar_top_margin", 0.86),
        fraction_bar_bottom_margin=cfg.get("selective_comparison_plot_fraction_bar_bottom_margin", 0.26),
        fraction_bar_wspace=cfg.get("selective_comparison_plot_fraction_bar_wspace", 0.22),
        overlap_figure_width_in=cfg.get("selective_comparison_plot_overlap_figure_width_in", 9.2),
        overlap_figure_height_in=cfg.get("selective_comparison_plot_overlap_figure_height_in", 5.6),
        overlap_left_margin=cfg.get("selective_comparison_plot_overlap_left_margin", 0.08),
        overlap_right_margin=cfg.get("selective_comparison_plot_overlap_right_margin", 0.995),
        overlap_top_margin=cfg.get("selective_comparison_plot_overlap_top_margin", 0.90),
        overlap_bottom_margin=cfg.get("selective_comparison_plot_overlap_bottom_margin", 0.10),
        overlap_wspace=cfg.get("selective_comparison_plot_overlap_wspace", 0.18),
        overlap_hspace=cfg.get("selective_comparison_plot_overlap_hspace", 0.28),
    )

    result = plot_fixation_selectivity_comparison_group_summaries(
        settings,
        regions=args.region,
    )
    if not result:
        print("[plot] no comparison-group figures were generated")
        return

    print(f"[plot] comparison group: {result['comparison_label']}")
    print(f"[plot] fraction bars: {result['fraction_bar_output_path']}")
    print(f"[plot] overlap matrix: {result['overlap_matrix_output_path']}")
    for summary in result.get("region_summaries", []):
        print(
            "[plot] region summary: "
            f"{summary['region']} "
            f"(N={summary['total_units']}, any_selective={summary['any_selective_units']})"
        )


if __name__ == "__main__":
    main()
