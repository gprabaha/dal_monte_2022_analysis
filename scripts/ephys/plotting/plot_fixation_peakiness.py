"""Plot region-wise fixation peakiness distributions."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_peakiness import (
    DEFAULT_HIGHLIGHT_STYLE_COLORS,
    DEFAULT_HIGHLIGHT_STYLE_LABELS,
    DEFAULT_HIGHLIGHT_STYLE_MARKERS,
    DEFAULT_HIGHLIGHT_STYLE_ORDER,
    DEFAULT_REGION_LABELS,
    DEFAULT_REGION_ORDER,
    FixationPeakinessPlotSettings,
    plot_fixation_peakiness_by_region,
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
        description="Plot region-wise fixation peakiness distributions.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--output-extension", default=None)
    parser.add_argument("--no-highlights", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    region_order = _normalize_str_list(cfg.get("peakiness_plot_region_order", cfg.get("peakiness_region_order", list(DEFAULT_REGION_ORDER))))
    if not region_order:
        region_order = list(DEFAULT_REGION_ORDER)

    region_labels = dict(DEFAULT_REGION_LABELS)
    region_labels.update(_normalize_label_map(cfg.get("peakiness_plot_region_labels", {})))

    highlight_style_order = _normalize_str_list(
        cfg.get(
            "peakiness_plot_highlight_style_order",
            cfg.get("phasic_tonic_example_grid_row_styles", list(DEFAULT_HIGHLIGHT_STYLE_ORDER)),
        )
    )
    if not highlight_style_order:
        highlight_style_order = list(DEFAULT_HIGHLIGHT_STYLE_ORDER)
    highlight_style_labels = dict(DEFAULT_HIGHLIGHT_STYLE_LABELS)
    highlight_style_labels.update(
        _normalize_label_map(
            cfg.get(
                "peakiness_plot_highlight_style_labels",
                cfg.get("phasic_tonic_example_grid_row_labels", {}),
            )
        )
    )
    highlight_style_colors = dict(DEFAULT_HIGHLIGHT_STYLE_COLORS)
    highlight_style_colors.update(_normalize_label_map(cfg.get("peakiness_plot_highlight_style_colors", {})))
    highlight_style_markers = dict(DEFAULT_HIGHLIGHT_STYLE_MARKERS)
    highlight_style_markers.update(_normalize_label_map(cfg.get("peakiness_plot_highlight_style_markers", {})))

    highlight_units = cfg.get(
        "peakiness_plot_highlight_units",
        cfg.get("phasic_tonic_example_grid_units", {}),
    )

    settings = FixationPeakinessPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get("peakiness_plot_input_subdir", cfg.get("peakiness_output_subdir", "ephys/psth/fixation_peakiness")),
        unit_peakiness_filename=cfg.get("peakiness_plot_unit_output_filename", cfg.get("peakiness_unit_output_filename", "unit_peakiness.csv")),
        output_subdir=cfg.get("peakiness_plot_output_subdir", "ephys/psth/fixation_peakiness/plots"),
        output_filename=(
            str(args.output_filename).strip()
            if args.output_filename is not None
            else str(cfg.get("peakiness_plot_output_filename", "fixation_peakiness_by_region"))
        ),
        output_extension=(
            str(args.output_extension).strip()
            if args.output_extension is not None
            else str(cfg.get("peakiness_plot_output_extension", "pdf"))
        ),
        output_dpi=cfg.get("peakiness_plot_output_dpi", 220),
        region_order=tuple(region_order),
        region_labels=region_labels,
        figure_width_in=cfg.get("peakiness_plot_figure_width_in", 7.8),
        figure_height_in=cfg.get("peakiness_plot_figure_height_in", 2.6),
        left_margin=float(cfg.get("peakiness_plot_left_margin", 0.08)),
        right_margin=float(cfg.get("peakiness_plot_right_margin", 0.995)),
        top_margin=float(cfg.get("peakiness_plot_top_margin", 0.84)),
        bottom_margin=float(cfg.get("peakiness_plot_bottom_margin", 0.26)),
        panel_wspace=float(cfg.get("peakiness_plot_panel_wspace", 0.12)),
        violin_width=float(cfg.get("peakiness_plot_violin_width", 0.86)),
        show_points=bool(cfg.get("peakiness_plot_show_points", False)),
        point_color=str(cfg.get("peakiness_plot_point_color", "#7a7a7a")),
        point_alpha=float(cfg.get("peakiness_plot_point_alpha", 0.28)),
        point_size=float(cfg.get("peakiness_plot_point_size", 10.0)),
        violin_facecolor=str(cfg.get("peakiness_plot_violin_facecolor", "#d2d7df")),
        violin_edgecolor=str(cfg.get("peakiness_plot_violin_edgecolor", "#3b3f47")),
        violin_alpha=float(cfg.get("peakiness_plot_violin_alpha", 0.92)),
        show_highlight_units=bool(cfg.get("peakiness_plot_show_highlight_units", True)) and not bool(args.no_highlights),
        highlight_units=highlight_units,
        highlight_style_order=tuple(highlight_style_order),
        highlight_style_labels=highlight_style_labels,
        highlight_style_colors=highlight_style_colors,
        highlight_style_markers=highlight_style_markers,
        highlight_marker_size=float(cfg.get("peakiness_plot_highlight_marker_size", 42.0)),
        highlight_annotation_fontsize=float(cfg.get("peakiness_plot_highlight_annotation_fontsize", 6.8)),
        jitter_seed=int(cfg.get("peakiness_plot_jitter_seed", 0)),
        y_label=str(cfg.get("peakiness_plot_y_label", "Peakiness Score")),
        show_suptitle=bool(cfg.get("peakiness_plot_show_suptitle", False)),
    )

    result = plot_fixation_peakiness_by_region(
        settings,
        regions=args.region,
    )
    if not result:
        print("[plot] no fixation peakiness figure was generated")
        return
    print(f"[plot] fixation peakiness figure: {result.get('output_path')}")
    highlighted_units = result.get("highlighted_units", [])
    matched = [row for row in highlighted_units if bool(row.get("matched"))]
    missing = [row for row in highlighted_units if not bool(row.get("matched"))]
    if matched:
        print("[plot] highlighted unit peakiness:")
        for row in matched:
            print(
                "  "
                f"{row.get('region_label')} {row.get('style_label')}: "
                f"{row.get('matched_unit_uuid')} score={float(row.get('peakiness_score')):.3f} "
                f"best={row.get('best_condition')}"
            )
    if missing:
        print("[plot] highlight specs not found in peakiness CSV:")
        for row in missing:
            print(
                "  "
                f"{row.get('region_label')} {row.get('style_label')}: "
                f"{row.get('configured_unit_uuid')}"
            )


if __name__ == "__main__":
    main()
