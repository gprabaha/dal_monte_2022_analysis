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
  By default, the script now writes one combined 1x4 region figure
  (columns ordered as `BLA`, `ACCg`, `dmPFC`, `OFC`) sized to letter width
  x one-fifth letter height.

  Config:
  - `selective_venn_output_subdir`
  - `selective_venn_output_extension`
  - `selective_venn_output_dpi`
  - `selective_venn_selective_windows`
  - `selective_venn_use_parallel`
  - `selective_venn_min_units_per_region`
  - `selective_venn_combine_regions_into_single_figure`
  - `selective_venn_region_order`
  - `selective_venn_combined_output_filename`
  - `selective_venn_combined_figure_width_in`
  - `selective_venn_combined_figure_height_in`
  - `selective_venn_combined_left_margin`
  - `selective_venn_combined_right_margin`
  - `selective_venn_combined_top_margin`
  - `selective_venn_combined_bottom_margin`
  - `selective_venn_combined_wspace`
  - `selective_venn_combined_show_pair_key`

- `plot_fixation_three_way_selectivity_triangular.py`
  Generates one large triangular-population summary with:
  - rows = fixation PSTH windows
  - columns = regions
  - points = units positioned by normalized three-condition mean firing
    (`face_interactive`, `face_non_interactive`, `object`)
  - point alpha encodes unit selectivity from `unit_selectivity.csv`:
    - alpha = 1.0 for units selective in any fixation-pair comparison
    - alpha = 0.5 for units selective in no pair comparisons
  Points use a single fixed color (no condition-based color coding).
  Input comes from `condition_window_means.csv` produced by
  `build_fixation_selective_units.py`.
  The script can also be configured for a compact single-row layout
  (e.g., one chosen window across ordered regions) and can restrict
  plotted points to selective units only.

  Config:
  - `selective_output_subdir`
  - `selective_condition_summary_filename`
  - `selective_unit_summary_filename`
  - `selective_triangular_output_subdir`
  - `selective_triangular_output_filename`
  - `selective_triangular_output_extension`
  - `selective_triangular_output_dpi`
  - `selective_triangular_min_units_per_panel`
  - `selective_triangular_point_size`
  - `selective_triangular_point_color`
  - `selective_triangular_point_alpha_significant`
  - `selective_triangular_point_alpha_non_significant`
  - `selective_triangular_marker_edge_width`
  - `selective_triangular_draw_centroid`
  - `selective_triangular_include_only_selective_units`
  - `selective_triangular_region_order`
  - `selective_triangular_windows`
  - `selective_triangular_figure_width_in`
  - `selective_triangular_figure_height_in`
  - `selective_triangular_show_suptitle`
  - `selective_triangular_left_margin`
  - `selective_triangular_right_margin`
  - `selective_triangular_top_margin`
  - `selective_triangular_bottom_margin`
  - `selective_triangular_panel_wspace`
  - `selective_triangular_panel_hspace`

- `plot_fixation_three_way_region_comparison.py`
  Plots cross-region comparison heatmaps for each time window from
  `pairwise_region_comparisons.csv` and `window_region_comparisons.csv`.
  The figure shows:
  - centroid-distance (ILR) matrices across regions
  - `-log10(adjusted p)` matrices for pairwise region tests
  - per-window global adjusted p-values in row labels

  Config:
  - `selective_region_comparison_output_subdir`
  - `selective_region_comparison_pairwise_filename`
  - `selective_region_comparison_window_filename`
  - `selective_region_comparison_plot_output_subdir`
  - `selective_region_comparison_plot_output_filename`
  - `selective_region_comparison_plot_output_extension`
  - `selective_region_comparison_plot_output_dpi`
  - `selective_region_comparison_alpha`
  - `selective_region_comparison_plot_pvalue_floor`
  - `selective_region_comparison_plot_annotation_max_regions`

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

- `plot_selective_fixation_psth_example_grid.py`
  Builds one selective-unit example grid figure with:
  - rows = preferred response class (`face_interactive`, `face_non_interactive`, `object`)
  - columns = regions (default: `BLA`, `ACCg`, `dmPFC`, `OFC`)
  - each cell = one manually selected unit from config, rendered as:
    - top raster (all three fixation conditions)
    - bottom mean firing-rate traces with SEM shading
  Default output is an editable vector PDF sized to letter width x 3/5 letter height.

  Config:
  - `selective_example_grid_output_subdir`
  - `selective_example_grid_output_filename`
  - `selective_example_grid_output_extension`
  - `selective_example_grid_output_dpi`
  - `selective_example_grid_pdf_compression`
  - `selective_example_grid_allow_missing`
  - `selective_example_grid_figure_width_in`
  - `selective_example_grid_figure_height_in`
  - `selective_example_grid_regions`
  - `selective_example_grid_preferences`
  - `selective_example_grid_row_labels`
  - `selective_example_grid_units`

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
