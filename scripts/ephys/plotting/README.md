# scripts/ephys/plotting

CLI entrypoints for ephys plotting.

Current scripts:
- `plot_fixation_psth_units.py`
  Generates one figure per unit using fixation PSTH trial data.
  Output layout is grouped by region (across all dates):
  `.../region=<region>/date=<date>__unit=<unit_uuid>.pdf`
  Each figure has:
  - top: color-coded rasters for interactive face, non-interactive face, and object fixations
  - bottom: mean firing rate (Hz) with SEM shading for the same conditions
  - trial PSTHs are smoothed before mean/SEM by default (`plot_smooth_before_average`)

Config:
- `configs/ephys_fixation_psth.yaml`
  - `plot_output_subdir`
  - `plot_output_extension`
  - `plot_output_dpi`
  - `plot_use_parallel`
  - `plot_parallelize_units`
  - `plot_unit_parallel_min_units`
  - `plot_max_trials_per_condition`
  - `plot_random_seed`
  - `plot_smooth_before_average`
  - `plot_smoothing_sigma_ms`
  - `plot_raster_jitter_within_bin`
  - `plot_raster_linelength`
  - `plot_raster_linewidth`
  - `plot_raster_alpha`
  - `plot_raster_darkening_factor`
  - `plot_raster_show_condition_background`
  - `plot_panel_raster_height_ratio`
  - `plot_panel_rate_height_ratio`
  - `plot_condition_colors`
