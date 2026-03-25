"""Plot region-wise fixation PSTH Fano-factor summaries."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_psth_fano_factor import (
    DEFAULT_CONDITION_COLORS,
    DEFAULT_CONDITION_LABELS,
    DEFAULT_CONDITION_ORDER,
    DEFAULT_REGION_LABELS,
    DEFAULT_REGION_ORDER,
    FixationPSTHFanoFactorPlotSettings,
    plot_fixation_psth_fano_factor_by_region,
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


def _resolve_time_window_ms(raw):
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("fano_factor_plot_time_window_ms must be [start_ms, stop_ms] or null.")
    return float(raw[0]), float(raw[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot region-wise fixation PSTH Fano-factor traces.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--output-extension", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    region_order = _normalize_str_list(
        cfg.get("fano_factor_plot_region_order", list(DEFAULT_REGION_ORDER))
    )
    if not region_order:
        region_order = list(DEFAULT_REGION_ORDER)
    condition_order = _normalize_str_list(
        cfg.get("fano_factor_plot_condition_order", list(DEFAULT_CONDITION_ORDER))
    )
    if not condition_order:
        condition_order = list(DEFAULT_CONDITION_ORDER)

    region_labels = dict(DEFAULT_REGION_LABELS)
    region_labels.update(_normalize_label_map(cfg.get("fano_factor_plot_region_labels", {})))
    condition_labels = dict(DEFAULT_CONDITION_LABELS)
    condition_labels.update(_normalize_label_map(cfg.get("fano_factor_plot_condition_labels", {})))
    condition_colors = dict(DEFAULT_CONDITION_COLORS)
    condition_colors.update(
        _normalize_label_map(
            cfg.get(
                "fano_factor_plot_condition_colors",
                cfg.get("plot_condition_colors", DEFAULT_CONDITION_COLORS),
            )
        )
    )

    settings = FixationPSTHFanoFactorPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get("fano_factor_plot_input_subdir", cfg.get("fano_factor_output_subdir", "ephys/psth/fixation_psth_fano_factor")),
        region_summary_filename=cfg.get("fano_factor_plot_region_summary_filename", cfg.get("fano_factor_region_summary_filename", "region_fano_factor_summary.csv")),
        output_subdir=cfg.get("fano_factor_plot_output_subdir", "ephys/psth/fixation_psth_fano_factor/plots"),
        output_filename=cfg.get("fano_factor_plot_output_filename", "fixation_psth_fano_factor_by_region"),
        output_extension=(
            str(args.output_extension).strip()
            if args.output_extension is not None
            else str(cfg.get("fano_factor_plot_output_extension", "pdf"))
        ),
        output_dpi=cfg.get("fano_factor_plot_output_dpi", 220),
        region_order=tuple(region_order),
        region_labels=region_labels,
        condition_order=tuple(condition_order),
        condition_labels=condition_labels,
        condition_colors=condition_colors,
        subplot_ncols=int(cfg.get("fano_factor_plot_subplot_ncols", 2)),
        figure_width_in=float(cfg.get("fano_factor_plot_figure_width_in", 9.5)),
        figure_height_in=float(cfg.get("fano_factor_plot_figure_height_in", 6.8)),
        left_margin=float(cfg.get("fano_factor_plot_left_margin", 0.08)),
        right_margin=float(cfg.get("fano_factor_plot_right_margin", 0.99)),
        top_margin=float(cfg.get("fano_factor_plot_top_margin", 0.92)),
        bottom_margin=float(cfg.get("fano_factor_plot_bottom_margin", 0.10)),
        wspace=float(cfg.get("fano_factor_plot_wspace", 0.22)),
        hspace=float(cfg.get("fano_factor_plot_hspace", 0.26)),
        line_width=float(cfg.get("fano_factor_plot_line_width", 1.8)),
        shade_alpha=float(cfg.get("fano_factor_plot_shade_alpha", 0.18)),
        min_units_per_condition=int(cfg.get("fano_factor_plot_min_units_per_condition", 1)),
        show_zero_line=bool(cfg.get("fano_factor_plot_show_zero_line", True)),
        show_unity_line=bool(cfg.get("fano_factor_plot_show_unity_line", True)),
        time_window_ms=_resolve_time_window_ms(
            cfg.get("fano_factor_plot_time_window_ms", (-500.0, 500.0))
        ),
        x_label=str(cfg.get("fano_factor_plot_x_label", "Time from fixation onset (ms)")),
        y_label=str(cfg.get("fano_factor_plot_y_label", "Fano Factor")),
    )

    out = plot_fixation_psth_fano_factor_by_region(settings, regions=args.region)
    if out is None:
        print("[plot] no fixation PSTH Fano figure was generated")
        return
    print(f"[plot] wrote fixation PSTH Fano figure: {out['output_path']}")


if __name__ == "__main__":
    main()
