"""Plot a 3x4 selective example-unit fixation PSTH grid."""

import argparse

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_psth import (
    DEFAULT_CONDITION_COLORS,
    FixationPSTHUnitPlotSettings,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_psth_example_grid import (
    DEFAULT_EXAMPLE_GRID_REGIONS,
    DEFAULT_EXAMPLE_GRID_ROW_LABELS,
    DEFAULT_EXAMPLE_GRID_ROW_PREFERENCES,
    FixationPSTHExampleGridPlotSettings,
    normalize_example_preference,
    parse_example_grid_unit_specs,
    plot_fixation_psth_example_grid,
)


def _normalize_row_labels(raw: object) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        try:
            pref = normalize_example_preference(key)
        except ValueError:
            continue
        label = str(value).strip()
        if label:
            out[pref] = label
    return out


def _base_plot_settings(dataset_cfg_path: str, plotting_cfg_path: str, cfg: dict) -> FixationPSTHUnitPlotSettings:
    return FixationPSTHUnitPlotSettings(
        cfg_path=dataset_cfg_path,
        plotting_cfg_path=plotting_cfg_path,
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        output_subdir=cfg.get("plot_output_subdir", "ephys/psth/fixation_psth_unit_plots"),
        output_extension=cfg.get("plot_output_extension", "pdf"),
        output_dpi=cfg.get("plot_output_dpi", 220),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        use_parallel=False,
        parallelize_units=False,
        unit_parallel_min_units=cfg.get("plot_unit_parallel_min_units", 2),
        max_procs=cfg.get("max_procs", 16),
        test_single=False,
        max_trials_per_condition=cfg.get("plot_max_trials_per_condition", 300),
        random_seed=cfg.get("plot_random_seed", 42),
        condition_colors=cfg.get("plot_condition_colors", DEFAULT_CONDITION_COLORS),
        smooth_before_average=cfg.get("plot_smooth_before_average", True),
        smoothing_sigma_ms=cfg.get("plot_smoothing_sigma_ms", 20.0),
        raster_jitter_within_bin=cfg.get("plot_raster_jitter_within_bin", True),
        raster_linelength=cfg.get("plot_raster_linelength", 1.0),
        raster_linewidth=cfg.get("plot_raster_linewidth", 2.0),
        raster_alpha=cfg.get("plot_raster_alpha", 1.0),
        raster_darkening_factor=cfg.get("plot_raster_darkening_factor", 0.65),
        raster_show_condition_background=cfg.get("plot_raster_show_condition_background", False),
        panel_raster_height_ratio=cfg.get("plot_panel_raster_height_ratio", 1.2),
        panel_rate_height_ratio=cfg.get("plot_panel_rate_height_ratio", 2.0),
        bin_size_ms_fallback=cfg.get("bin_size_ms", 10.0),
        window_pre_s=cfg.get("window_pre_s", 1.0),
        window_post_s=cfg.get("window_post_s", 1.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one selective example-unit fixation PSTH grid with "
            "rows=preference and columns=region."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--row-preference", action="append", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--output-extension", default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--figure-width-in", type=float, default=None)
    parser.add_argument("--figure-height-in", type=float, default=None)
    parser.add_argument("--max-trials-per-condition", type=int, default=None)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--smoothing-sigma-ms", type=float, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    unit_settings = _base_plot_settings(args.dataset_cfg, args.plotting_cfg, cfg)
    if args.max_trials_per_condition is not None:
        unit_settings.max_trials_per_condition = int(args.max_trials_per_condition)
    if args.no_smooth:
        unit_settings.smooth_before_average = False
    if args.smoothing_sigma_ms is not None:
        unit_settings.smoothing_sigma_ms = float(args.smoothing_sigma_ms)

    regions = args.region if args.region else cfg.get(
        "selective_example_grid_regions",
        list(DEFAULT_EXAMPLE_GRID_REGIONS),
    )
    row_preferences = args.row_preference if args.row_preference else cfg.get(
        "selective_example_grid_preferences",
        list(DEFAULT_EXAMPLE_GRID_ROW_PREFERENCES),
    )
    row_labels = dict(DEFAULT_EXAMPLE_GRID_ROW_LABELS)
    row_labels.update(_normalize_row_labels(cfg.get("selective_example_grid_row_labels", {})))

    unit_specs = parse_example_grid_unit_specs(
        cfg,
        regions=regions,
        row_preferences=row_preferences,
    )
    if not unit_specs:
        print("[plot] no selective example-unit entries parsed from selective_example_grid_units")
        return

    grid_settings = FixationPSTHExampleGridPlotSettings(
        unit_plot_settings=unit_settings,
        output_subdir=cfg.get(
            "selective_example_grid_output_subdir",
            cfg.get("selective_plot_output_subdir", "ephys/psth/fixation_psth_selective_unit_plots"),
        ),
        output_filename=cfg.get("selective_example_grid_output_filename", "selective_example_grid_3x4"),
        output_extension=cfg.get("selective_example_grid_output_extension", "pdf"),
        output_dpi=cfg.get("selective_example_grid_output_dpi", cfg.get("plot_output_dpi", 220)),
        figure_width_in=cfg.get("selective_example_grid_figure_width_in", 8.5),
        figure_height_in=cfg.get("selective_example_grid_figure_height_in", 6.6),
        column_regions=regions,
        row_preferences=row_preferences,
        row_labels=row_labels,
        left_margin=cfg.get("selective_example_grid_left_margin", 0.07),
        right_margin=cfg.get("selective_example_grid_right_margin", 0.995),
        top_margin=cfg.get("selective_example_grid_top_margin", 0.91),
        bottom_margin=cfg.get("selective_example_grid_bottom_margin", 0.095),
        panel_wspace=cfg.get("selective_example_grid_panel_wspace", 0.16),
        panel_hspace=cfg.get("selective_example_grid_panel_hspace", 0.24),
        inner_hspace=cfg.get("selective_example_grid_inner_hspace", 0.06),
        show_global_legend=cfg.get("selective_example_grid_show_legend", True),
        legend_ncol=cfg.get("selective_example_grid_legend_ncol", 3),
        pdf_compression=cfg.get("selective_example_grid_pdf_compression", 0),
    )

    if args.output_subdir:
        grid_settings.output_subdir = args.output_subdir
    if args.output_filename:
        grid_settings.output_filename = args.output_filename
    if args.output_extension:
        grid_settings.output_extension = args.output_extension
    if args.dpi is not None:
        grid_settings.output_dpi = int(args.dpi)
    if args.figure_width_in is not None:
        grid_settings.figure_width_in = float(args.figure_width_in)
    if args.figure_height_in is not None:
        grid_settings.figure_height_in = float(args.figure_height_in)

    allow_missing = bool(cfg.get("selective_example_grid_allow_missing", False) or args.allow_missing)
    result = plot_fixation_psth_example_grid(
        grid_settings,
        unit_specs=unit_specs,
        sessions=args.session,
        allow_missing=allow_missing,
    )
    print(
        "[plot] wrote selective example grid: "
        f"{result.get('output_path')} "
        f"(resolved {result.get('resolved_cells')}/{result.get('expected_cells')} cells)"
    )
    missing = result.get("missing_specs", [])
    unresolved = result.get("unresolved_specs", [])
    if missing:
        print(f"[plot] missing configured cells: {missing}")
    if unresolved:
        print(f"[plot] unresolved cells: {unresolved}")


if __name__ == "__main__":
    main()
