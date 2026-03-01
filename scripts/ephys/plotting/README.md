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

- `plot_period_psth_units.py`
  Generates one figure per unit using period PSTH trial data
  (`interactive_periods.pkl`).
  Output layout is grouped by region (across all dates):
  `.../region=<region>/date=<date>__unit=<unit_uuid>.png`
  Each figure has:
  - top: rasters for interactive and non-interactive period trials
  - bottom: mean firing rate (Hz) with SEM shading for the same conditions
  - trial rates are smoothed before mean/SEM by default (`plot_smoothing_sigma_ms=200`)

  Config:
  - `configs/ephys_period_psth.yaml`
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
  - `plot_raster_max_spikes_per_bin`
  - `plot_raster_linelength`
  - `plot_raster_linewidth`
  - `plot_raster_alpha`
  - `plot_raster_darkening_factor`
  - `plot_raster_show_condition_background`
  - `plot_panel_raster_height_ratio`
  - `plot_panel_rate_height_ratio`
  - `plot_condition_colors`

- `plot_fixation_selectivity_venn.py`
  Generates one Venn diagram per region using fixation selectivity pair outputs.
  Uses area-scaled 3-set Venn fitting (`matplotlib-venn`) so set/overlap
  bubble geometry reflects relative magnitudes.
  Also prints region-level counts and percentages to terminal for:
  - each pair set
  - all seven Venn intersections

  Config:
  - `selective_venn_output_subdir`
  - `selective_venn_output_extension`
  - `selective_venn_output_dpi`
  - `selective_venn_use_parallel`
  - `selective_venn_min_units_per_region`

- `plot_selective_fixation_psth_units.py`
  Selective-unit PSTH plotting with two modes:
  - batch mode (default): generate PNG plots for all selective units
    in region folders (no extra statistical computation)
  - example mode (`--example-unit-uuid`): generate PDF + PNG for a selected
    selective unit under an `example units/` subfolder with per-bin
    significance tick marks for all three fixation-pair comparisons

  Config:
  - `selective_plot_output_subdir`
  - `selective_plot_output_extension`
  - `selective_example_subfolder`
  - `selective_example_output_extensions`
  - `selective_example_significance_alpha`
  - `selective_example_significance_test`
  - `selective_example_significance_min_trials_per_condition`

- `plot_within_region_fixation_neural_cross_correlation.py`
  Within-region neural xcorr summary plotting:
  - date-level (PNG): one figure per date for `pairs` and one for `mean`,
    each containing all regions as subplots
  - global-level (PDF): one figure for `pairs` and one for `mean`,
    each containing all regions as subplots

- `plot_cross_region_fixation_neural_cross_correlation.py`
  Cross-region neural xcorr summary plotting:
  - date-level (PNG): one figure per date for `pairs` and one for `mean`,
    each containing all ordered region pairs as subplots
  - global-level (PDF): one figure for `pairs` and one for `mean`,
    each containing all ordered region pairs as subplots

- `plot_fixation_neural_cross_correlation.py`
  Combined neural xcorr summary plotting (within + cross) with optional filter:
  - `--analysis-kind both|within|cross`
  - same date/global outputs as above
  - uses stored `pair_averages` from analysis outputs when available
    (falls back to per-fixation `cross_correlations`)

  In each subplot, all 3 fixation-condition traces are overlaid in the same axis:
  - face interactive
  - face non-interactive
  - object
  The script writes separate files for:
  - `pairs` (individual neural-pair traces)
  - `mean` (condition means across neural pairs)
  Date-level files include `date=<date>` in filename (not date subfolders).
  PNG outputs are saved with white background.

  Config:
  - `configs/ephys_fixation_neural_cross_correlation.yaml`
  - `within_pair_average_output_filename` / `cross_pair_average_output_filename`
  - `plot_output_subdir`
  - `plot_date_output_extension`
  - `plot_region_output_extension`
  - `plot_date_output_dpi`
  - `plot_region_output_dpi`
  - `plot_date_figsize`
  - `plot_region_figsize`
  - `plot_condition_order`
  - `plot_condition_labels`
  - `plot_condition_colors`
  - `plot_pair_trace_alpha`
  - `plot_pair_trace_linewidth`
  - `plot_mean_trace_linewidth`
  - `plot_subplot_ncols`
  - `plot_normalize_traces`
  - `plot_normalization_method`
  - `plot_use_parallel`
  - `plot_max_procs`
  - `plot_parallelize_date_plots`
  - `plot_parallelize_global_plots`
  - `plot_max_pair_traces_per_plot`
  - `plot_max_points_per_pdf_trace`
