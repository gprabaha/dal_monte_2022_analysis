"""Plot per-unit interactive/non-interactive period PSTHs."""

import argparse

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.period_psth import (
    DEFAULT_PERIOD_CONDITION_COLORS,
    PeriodPSTHUnitPlotSettings,
    plot_period_psth_units,
)


def _as_unit_list(values):
    if not values:
        return None
    return [str(v) for v in values if str(v).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-unit period PSTH figures with top rasters and bottom "
            "mean firing-rate ± SEM traces."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-period-psth-cfg", default="configs/ephys_period_psth.yaml")
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
    parser.add_argument("--raster-max-spikes-per-bin", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_period_psth_cfg)
    plot_colors = cfg.get("plot_condition_colors", DEFAULT_PERIOD_CONDITION_COLORS)
    states = cfg.get("states", ["interactive", "non_interactive"])
    interactive_label = cfg.get("interactive_high_label", "interactive")
    non_interactive_label = cfg.get("non_interactive_label")
    if not non_interactive_label and isinstance(states, (list, tuple)) and len(states) >= 2:
        non_interactive_label = str(states[1])
    if not non_interactive_label:
        non_interactive_label = "non_interactive"

    settings = PeriodPSTHUnitPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        trial_input_modality=cfg.get("output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "interactive_periods.pkl"),
        output_subdir=cfg.get("plot_output_subdir", "ephys/psth/period_psth_unit_plots"),
        output_extension=cfg.get("plot_output_extension", "png"),
        output_dpi=cfg.get("plot_output_dpi", 220),
        interactive_label=interactive_label,
        non_interactive_label=str(non_interactive_label),
        use_parallel=cfg.get("plot_use_parallel", True),
        parallelize_units=cfg.get("plot_parallelize_units", True),
        unit_parallel_min_units=cfg.get("plot_unit_parallel_min_units", 2),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        max_trials_per_condition=cfg.get("plot_max_trials_per_condition"),
        random_seed=cfg.get("plot_random_seed", 42),
        condition_colors=plot_colors,
        smooth_before_average=cfg.get("plot_smooth_before_average", True),
        smoothing_sigma_ms=cfg.get("plot_smoothing_sigma_ms", 200.0),
        raster_jitter_within_bin=cfg.get("plot_raster_jitter_within_bin", True),
        raster_linelength=cfg.get("plot_raster_linelength", 0.95),
        raster_linewidth=cfg.get("plot_raster_linewidth", 1.0),
        raster_alpha=cfg.get("plot_raster_alpha", 1.0),
        raster_darkening_factor=cfg.get("plot_raster_darkening_factor", 0.65),
        raster_show_condition_background=cfg.get("plot_raster_show_condition_background", False),
        raster_max_spikes_per_bin=cfg.get("plot_raster_max_spikes_per_bin"),
        panel_raster_height_ratio=cfg.get("plot_panel_raster_height_ratio", 1.2),
        panel_rate_height_ratio=cfg.get("plot_panel_rate_height_ratio", 2.0),
        bin_size_ms_fallback=cfg.get("bin_size_ms", 100.0),
        window_pre_s=cfg.get("window_pre_s", 14.0),
        window_post_s=cfg.get("window_post_s", 14.0),
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
    if args.raster_max_spikes_per_bin is not None:
        settings.raster_max_spikes_per_bin = int(args.raster_max_spikes_per_bin)

    out_paths = plot_period_psth_units(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
        unit_uuids=_as_unit_list(args.unit_uuid),
    )

    print(f"[plot] wrote {len(out_paths)} unit period PSTH figure(s)")
    if out_paths:
        print(f"[plot] first output: {out_paths[0]}")


if __name__ == "__main__":
    main()
