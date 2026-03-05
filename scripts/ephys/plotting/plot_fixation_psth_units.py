"""Plot per-unit fixation PSTH rasters and average firing-rate traces."""

import argparse

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_psth import (
    DEFAULT_CONDITION_COLORS,
    FixationPSTHUnitPlotSettings,
    plot_fixation_psth_units,
)


def _as_unit_list(values):
    if not values:
        return None
    return [str(v) for v in values if str(v).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-unit fixation PSTH figures with top rasters and bottom "
            "mean firing-rate ± SEM traces."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--unit-uuid", action="append", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--output-extension", default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--max-trials-per-condition", type=int, default=None)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--smoothing-sigma-ms", type=float, default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    plot_colors = cfg.get("plot_condition_colors", DEFAULT_CONDITION_COLORS)

    settings = FixationPSTHUnitPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        use_precomputed_average_traces=cfg.get("plot_use_precomputed_average_traces", True),
        average_trace_input_subdir=cfg.get("plot_average_input_subdir", "ephys/psth/fixation_psth_averages"),
        average_trace_input_filename=cfg.get(
            "plot_average_input_filename_split",
            cfg.get("plot_average_input_filename", "fixations.pkl"),
        ),
        average_trace_object_input_subdir=cfg.get(
            "plot_average_object_input_subdir",
            cfg.get("plot_average_input_subdir", "ephys/psth/fixation_psth_averages"),
        ),
        average_trace_object_input_filename=cfg.get(
            "plot_average_object_input_filename",
            cfg.get("plot_average_input_filename_unsplit"),
        ),
        allow_trial_trace_fallback=cfg.get("plot_allow_trial_trace_fallback", True),
        output_subdir=cfg.get("plot_output_subdir", "ephys/psth/fixation_psth_unit_plots"),
        output_extension=cfg.get("plot_output_extension", "pdf"),
        output_dpi=cfg.get("plot_output_dpi", 220),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        use_parallel=cfg.get("plot_use_parallel", True),
        parallelize_units=cfg.get("plot_parallelize_units", True),
        unit_parallel_min_units=cfg.get("plot_unit_parallel_min_units", 2),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        max_trials_per_condition=cfg.get("plot_max_trials_per_condition", 300),
        random_seed=cfg.get("plot_random_seed", 42),
        condition_colors=plot_colors,
        smooth_before_average=cfg.get("plot_smooth_before_average", True),
        smoothing_sigma_ms=cfg.get("plot_smoothing_sigma_ms", 20.0),
        raster_jitter_within_bin=cfg.get("plot_raster_jitter_within_bin", True),
        raster_linelength=cfg.get("plot_raster_linelength", 0.95),
        raster_linewidth=cfg.get("plot_raster_linewidth", 1.0),
        raster_alpha=cfg.get("plot_raster_alpha", 1.0),
        raster_darkening_factor=cfg.get("plot_raster_darkening_factor", 0.65),
        raster_show_condition_background=cfg.get("plot_raster_show_condition_background", False),
        panel_raster_height_ratio=cfg.get("plot_panel_raster_height_ratio", 1.2),
        panel_rate_height_ratio=cfg.get("plot_panel_rate_height_ratio", 2.0),
        bin_size_ms_fallback=cfg.get("bin_size_ms", 10.0),
        window_pre_s=cfg.get("window_pre_s", 1.0),
        window_post_s=cfg.get("window_post_s", 1.0),
    )

    if args.output_subdir:
        settings.output_subdir = args.output_subdir
    if args.output_extension:
        settings.output_extension = args.output_extension
    if args.dpi is not None:
        settings.output_dpi = int(args.dpi)
    if args.max_trials_per_condition is not None:
        settings.max_trials_per_condition = int(args.max_trials_per_condition)
    if args.no_smooth:
        settings.smooth_before_average = False
    if args.smoothing_sigma_ms is not None:
        settings.smoothing_sigma_ms = float(args.smoothing_sigma_ms)
    if args.no_parallel:
        settings.use_parallel = False
    if args.test_single:
        settings.test_single = True

    out_paths = plot_fixation_psth_units(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
        unit_uuids=_as_unit_list(args.unit_uuid),
    )

    print(f"[plot] wrote {len(out_paths)} unit fixation PSTH figure(s)")
    if out_paths:
        print(f"[plot] first output: {out_paths[0]}")


if __name__ == "__main__":
    main()
