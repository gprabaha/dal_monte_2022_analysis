"""Plot pooled fixation peakiness comparisons across conditions."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_peakiness_condition_comparison import (
    DEFAULT_CONDITION_COLORS,
    DEFAULT_CONDITION_LABELS,
    DEFAULT_CONDITION_ORDER,
    DEFAULT_REGION_LABELS,
    DEFAULT_REGION_ORDER,
    FixationPeakinessConditionComparisonPlotSettings,
    plot_fixation_peakiness_condition_comparison_by_region,
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
        description="Plot pooled fixation peakiness comparisons across conditions.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--output-extension", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    condition_order = _normalize_str_list(
        cfg.get("peakiness_condition_plot_condition_order", list(DEFAULT_CONDITION_ORDER))
    )
    if not condition_order:
        condition_order = list(DEFAULT_CONDITION_ORDER)

    condition_labels = dict(DEFAULT_CONDITION_LABELS)
    condition_labels.update(_normalize_label_map(cfg.get("peakiness_condition_plot_condition_labels", {})))
    condition_colors = dict(DEFAULT_CONDITION_COLORS)
    condition_colors.update(_normalize_label_map(cfg.get("peakiness_condition_plot_condition_colors", {})))
    region_order = _normalize_str_list(cfg.get("peakiness_condition_plot_region_order", list(DEFAULT_REGION_ORDER)))
    if not region_order:
        region_order = list(DEFAULT_REGION_ORDER)
    region_labels = dict(DEFAULT_REGION_LABELS)
    region_labels.update(_normalize_label_map(cfg.get("peakiness_condition_plot_region_labels", {})))

    settings = FixationPeakinessConditionComparisonPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get("peakiness_condition_plot_input_subdir", cfg.get("peakiness_plot_input_subdir", "ephys/psth/fixation_peakiness")),
        unit_peakiness_filename=cfg.get("peakiness_condition_plot_unit_output_filename", cfg.get("peakiness_plot_unit_output_filename", "unit_peakiness.csv")),
        output_subdir=cfg.get("peakiness_condition_plot_output_subdir", "ephys/psth/fixation_peakiness/plots"),
        output_extension=(
            str(args.output_extension).strip()
            if args.output_extension is not None
            else str(cfg.get("peakiness_condition_plot_output_extension", "pdf"))
        ),
        output_dpi=cfg.get("peakiness_condition_plot_output_dpi", 220),
        condition_order=tuple(condition_order),
        condition_labels=condition_labels,
        condition_colors=condition_colors,
        region_order=tuple(region_order),
        region_labels=region_labels,
        figure_width_in=cfg.get("peakiness_condition_plot_figure_width_in", 7.8),
        figure_height_in=cfg.get("peakiness_condition_plot_figure_height_in", 2.8),
        left_margin=float(cfg.get("peakiness_condition_plot_left_margin", 0.08)),
        right_margin=float(cfg.get("peakiness_condition_plot_right_margin", 0.995)),
        top_margin=float(cfg.get("peakiness_condition_plot_top_margin", 0.86)),
        bottom_margin=float(cfg.get("peakiness_condition_plot_bottom_margin", 0.22)),
        panel_wspace=float(cfg.get("peakiness_condition_plot_panel_wspace", 0.22)),
        violin_width=float(cfg.get("peakiness_condition_plot_violin_width", 0.82)),
        violin_alpha=float(cfg.get("peakiness_condition_plot_violin_alpha", 0.72)),
        violin_edgecolor=str(cfg.get("peakiness_condition_plot_violin_edgecolor", "#2f3136")),
        show_violin_points=bool(cfg.get("peakiness_condition_plot_show_violin_points", False)),
        violin_point_color=str(cfg.get("peakiness_condition_plot_violin_point_color", "#6f6f6f")),
        violin_point_alpha=float(cfg.get("peakiness_condition_plot_violin_point_alpha", 0.22)),
        violin_point_size=float(cfg.get("peakiness_condition_plot_violin_point_size", 9.0)),
        density_alpha=float(cfg.get("peakiness_condition_plot_density_alpha", 0.24)),
        density_linewidth=float(cfg.get("peakiness_condition_plot_density_linewidth", 1.6)),
        density_grid_n=int(cfg.get("peakiness_condition_plot_density_grid_n", 400)),
        score_label=str(cfg.get("peakiness_condition_plot_score_label", "Peakiness Score")),
        density_label=str(cfg.get("peakiness_condition_plot_density_label", "Density")),
        violin_by_region_output_filename=str(
            cfg.get(
                "peakiness_condition_plot_violin_by_region_output_filename",
                "fixation_peakiness_condition_comparison_by_region_violin",
            )
        ),
        density_by_region_output_filename=str(
            cfg.get(
                "peakiness_condition_plot_density_by_region_output_filename",
                "fixation_peakiness_condition_comparison_by_region_density",
            )
        ),
        stats_output_filename=str(
            cfg.get(
                "peakiness_condition_plot_stats_output_filename",
                "fixation_peakiness_condition_comparison_by_region_stats.csv",
            )
        ),
        pvalue_correction=str(cfg.get("peakiness_condition_plot_pvalue_correction", "fdr_bh")),
        alpha=float(cfg.get("peakiness_condition_plot_alpha", 0.05)),
        min_paired_units_per_region=int(cfg.get("peakiness_condition_plot_min_paired_units_per_region", 2)),
        show_suptitle=bool(cfg.get("peakiness_condition_plot_show_suptitle", False)),
    )

    result = plot_fixation_peakiness_condition_comparison_by_region(settings)
    if not result:
        print("[plot] no fixation peakiness condition-comparison figure was generated")
        return
    print(f"[plot] fixation peakiness region violin figure: {result.get('violin_output_path')}")
    print(f"[plot] fixation peakiness region density figure: {result.get('density_output_path')}")
    print(f"[plot] fixation peakiness region stats: {result.get('stats_output_path')}")
    mean_summary = result.get("mean_summary", [])
    if mean_summary:
        print("[plot] region mean peakiness summary:")
        for row in mean_summary:
            print(
                "  "
                f"{row.get('region')}: "
                f"n={int(row.get('n_units', 0))} "
                f"Int={float(row.get('face_interactive_mean', float('nan'))):.3f} "
                f"Non-Int={float(row.get('face_non_interactive_mean', float('nan'))):.3f} "
                f"Obj={float(row.get('object_mean', float('nan'))):.3f}"
            )
    stats_df = result.get("stats_df")
    if isinstance(stats_df, type(None)) or getattr(stats_df, "empty", True):
        return
    print("[plot] adjusted paired tests by region:")
    for row in stats_df.itertuples(index=False):
        print(
            "  "
            f"{getattr(row, 'region')} "
            f"{getattr(row, 'condition_a')} vs {getattr(row, 'condition_b')}: "
            f"n={int(getattr(row, 'n_units_paired'))} "
            f"mean_diff={float(getattr(row, 'mean_difference_a_minus_b')):.3f} "
            f"t={float(getattr(row, 'statistic')):.3f} "
            f"p_adj={float(getattr(row, 'p_value_adjusted')):.4g}"
        )


if __name__ == "__main__":
    main()
