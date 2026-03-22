# scripts/ephys/analysis

CLI entrypoints for ephys analysis.

Current scripts:
- `build_fixation_selective_units.py`
  Computes fixation-pair selectivity per unit from trial PSTH data.
  For each unit/day, trial PSTHs are smoothed (configurable, default matches
  mean-PSTH smoothing), converted to Hz, reduced to window means per trial,
  then tested with unpaired tests across configured windows (default includes four):
  - `pre_fix`  (-500 ms to 0 ms)
  - `peri_fix` (-250 ms to 250 ms)
  - `post_fix` (0 ms to 500 ms)
  - `full_fix` (-500 ms to 500 ms)
  Multiple named comparison families run in every analysis pass
  (configurable, defaults include `three_condition_core`,
  `interactive_state_matched`, `face_vs_object_unsplit`).
  Pair is selective if any configured significance window is significant
  (default significance windows: `pre_fix`, `peri_fix`, `post_fix`).
  `full_fix` can remain in outputs without affecting significance calls.
  Writes per-comparison CSVs suffixed with `__<comparison_label>.csv`
  and keeps unsuffixed CSVs for the configured primary comparison.
  Also writes one per-unit/per-window condition summary table
  (`condition_window_means.csv`) with mean firing rates for:
  - interactive face
  - non-interactive face
  - face (unsplit)
  - object interactive
  - object non-interactive
  - object (unsplit)
  and normalized relative components for triangular plotting.
- `build_fixation_roi_vs_period_factorial.py`
  Runs a per-unit 2x2 factorial analysis from trial fixation PSTHs:
  - factors: ROI (`face` vs `object`) and period (`interactive` vs `non_interactive`)
  - pipeline per window: smooth trial PSTH, convert to Hz, compute trial window means
  - GLM: `response ~ roi + period + roi:period`
  Exports:
  - per-unit GLM term table (`roi_main`, `period_main`, `interaction`)
  - per-unit axis-value table (`face_object`, `interactive_state`, `cross_interaction`)
    from both condition means and GLM coefficients
  - per-unit collapsed axis-significance table (collapsed across configured
    significance windows)
  - per-unit collapsed axis table with one stored row per unit/axis/source
    across the configured significance windows
  - region-level selective-fraction tables and cross-region pairwise fraction
    comparisons (multiple-comparison corrected)
  - region-level axis-magnitude summaries and Welch comparisons with two modes:
    - `split_by_window`: separate `pre/peri/post` tables
    - `max_abs_across_windows`: one reduced table using the max-abs window per
      unit-axis
    - `averaged_across_windows`: per-unit magnitudes averaged across
      `pre/peri/post`, then compared in one reduced table
    GLM fitting remains per-window in both modes.
    Axis magnitude comparisons use the `cell_means` axis source.
- `build_fixation_preference_index.py`
  Computes per-bin pairwise fixation preference indices for each unit using:
  - numerator: `A - B` where A/B are condition mean firing rates (Hz) per bin
  - denominator mode (configurable):
    - `unit_max_sum` (default): `max(A+B)` across bins for that unit/pair
    - `per_bin_sum` (fallback): `A+B` at each bin (legacy behavior)
  - bin window (configurable, default `[-500, 500] ms`): only bins in this
    window are included in index outputs
  Writes both index variants in every output row so plotting can choose mode later:
  - `preference_index_unit_max_sum`
  - `preference_index_per_bin_sum`
  Keeps `preference_index` as the active mode selected during build.
  - default comparisons:
    - `face_interactive__vs__face_non_interactive`
    - `face_interactive__vs__object`
    - `face_non_interactive__vs__object`
  Also merges selectivity labels from:
  - `pair_selectivity.csv` (`is_selective_pair`)
  - `unit_selectivity.csv` (`is_selective_unit`, `selective_pairs`)
  Input source can be either:
  - trial PSTH rows (`psth_counts`) from processed data, or
  - date-level average PSTH rows (`psth_mean`) from analysis outputs.
- `build_fixation_preference_index_wide_binned_firing_rate_averages.py`
  Builds wide-binned mean firing-rate averages used as fixation preference-index input.
  Intended defaults:
  - split by interactive state (`face_interactive` vs `face_non_interactive`)
  - target bins of 50 ms with 25 ms stride
  - resample to wide bins (if configured), smooth, convert to firing rate (Hz), then average
  - default output is one combined `fixations.pkl` containing split + unsplit tables
    (legacy separate-file mode is still available)
  This keeps existing 10 ms trial PSTHs unchanged.
- `build_fixation_three_way_region_comparison.py`
  Compares three-way fixation-response compositions across regions per
  analysis window using permutation tests for:
  - centroid separation in ILR space (pseudo-F)
  - dispersion differences (distance-to-centroid spread)
  - vertex-axis alignment concentration (line-like clustering toward centroid-to-vertex axes)
  Writes:
  - `pairwise_region_comparisons.csv`
  - `window_region_comparisons.csv`
- `build_within_region_fixation_neural_cross_correlation.py`
  Computes fixation-level neural cross-correlations for within-region
  unit pairs (nC2 per region per fixation) from fixation-aligned trial signals.
  Current defaults use `spike_train_counts` from `fixations_spike_train_1ms.pkl`,
  windowed to `[-500, 500] ms` before xcorr.
  Output pickle stores:
  - `cross_correlations` (per-fixation pair traces)
  - `pair_averages` (per-session pair averages by condition)
  Writes run-level reports beside the session outputs:
  - `session_report.csv` / `session_report_smoothed.csv`
  - `skipped_sessions.csv` / `skipped_sessions_smoothed.csv`

- `build_within_region_fixation_neural_cross_correlation_sig_xcorr_pairs.py`
  Aggregates saved within-region fixation neural xcorr outputs across sessions
  within each date, grouped by neural pair and fixation condition.
  Writes separate raw and smoothed date-level sig-pair outputs with:
  - one row per date / pair / fixation condition
  - `n_fixations` and lag-mean significance stats (`> 0` one-sided test)
  - one companion region summary table counting total pairs, condition-selective pairs,
    and pairs significant for any fixation condition
- `build_cross_region_fixation_neural_cross_correlation_sig_xcorr_pairs.py`
  Aggregates saved cross-region fixation neural xcorr outputs across sessions
  within each date, grouped by neural pair and fixation condition.
  Writes separate raw and smoothed date-level sig-pair outputs with:
  - one row per date / pair / fixation condition
  - `n_fixations` and lag-mean significance stats (`> 0` one-sided test)
  - one companion region-pair summary table counting total pairs, condition-selective pairs,
    and pairs significant for any fixation condition
- `build_fixation_population_pca.py`
  Builds region-level population PCA outputs in a configurable time window
  (default `[-500, 500] ms`):
  - primary input: date-level average PSTH pickles
    (split face file + optional unsplit-object file)
  - optional fallback input: processed trial PSTHs
    (`processed_data_root/.../psth/fixations.pkl`) aggregated to per-unit condition means
  - per-condition PCA fits (`face_interactive`, `face_non_interactive`, `object`)
  - concatenated-condition PCA fit (conditions concatenated along time)
  - concatenated-fit PC timecourses for each fixation condition
  - cross-condition explained-variance curves from per-condition fits
  Writes:
  - `pca_fit_summary.csv`
  - `concatenated_pc_timecourses.csv`
  - `cross_condition_explained_variance.csv`
  - `region_unit_inventory.csv`
  - `results.pkl`
- `build_cross_region_fixation_neural_cross_correlation.py`
  Computes fixation-level neural cross-correlations for cross-region
  unit pairs (anchor-region units x partner-region units per fixation) from
  fixation-aligned trial signals. Current defaults use `spike_train_counts`
  from `fixations_spike_train_1ms.pkl`, windowed to `[-500, 500] ms`
  before xcorr.
  Output pickle stores:
  - `cross_correlations` (per-fixation pair traces)
  - `pair_averages` (per-session pair averages by condition)
  Writes run-level reports beside the session outputs:
  - `session_report.csv` / `session_report_smoothed.csv`
  - `skipped_sessions.csv` / `skipped_sessions_smoothed.csv`

Config:
- `configs/ephys_fixation_psth.yaml`
  - `selective_output_subdir`
  - `selective_window_stats_filename`
  - `selective_pair_summary_filename`
  - `selective_unit_summary_filename`
  - `selective_condition_summary_filename`
  - `selective_output_pickle_filename`
  - `selective_smooth_before_window_average`
  - `selective_smoothing_sigma_ms`
  - `selective_primary_comparison_group`
  - `selective_comparison_groups`
  - `selective_alpha`
  - `selective_test`
  - `selective_min_trials_per_condition`
  - `selective_use_parallel`
  - `selective_windows_ms`
  - `selective_significance_windows`
  - `roi_vs_period_output_subdir`
  - `roi_vs_period_unit_term_filename`
  - `roi_vs_period_unit_axis_filename`
  - `roi_vs_period_unit_axis_collapsed_filename`
  - `roi_vs_period_unit_window_summary_filename`
  - `roi_vs_period_region_fraction_filename`
  - `roi_vs_period_region_fraction_pairwise_filename`
  - `roi_vs_period_region_fraction_within_region_filename`
  - `roi_vs_period_region_axis_summary_filename`
  - `roi_vs_period_region_axis_pairwise_filename`
  - `roi_vs_period_region_axis_within_region_filename`
  - `roi_vs_period_region_axis_friedman_filename`
  - `roi_vs_period_output_pickle_filename`
  - `roi_vs_period_windows_ms`
  - `roi_vs_period_significance_windows`
  - `roi_vs_period_smooth_before_window_average`
  - `roi_vs_period_smoothing_sigma_ms`
  - `roi_vs_period_min_trials_per_cell`
  - `roi_vs_period_min_units_per_region`
  - `roi_vs_period_alpha`
  - `roi_vs_period_pvalue_correction`
  - `roi_vs_period_unit_significance_mode`
  - `roi_vs_period_axis_comparison_mode`
  - `roi_vs_period_parallelization_scope`
  - `roi_vs_period_use_parallel`
  - `average_target_bin_size_ms`
  - `average_target_bin_step_ms`
  - `selective_index_average_output_subdir`
  - `selective_index_average_output_filename`
  - `selective_index_average_split_by_interactive_state`
  - `selective_index_average_restrict_interactive_state`
  - `selective_index_average_group_by_session`
  - `selective_index_average_smooth_before_average`
  - `selective_index_average_smoothing_sigma_ms`
  - `selective_index_average_target_bin_size_ms`
  - `selective_index_average_target_bin_step_ms`
  - `selective_index_average_categories`
  - `selective_index_output_subdir`
  - `selective_index_timeseries_filename`
  - `selective_index_output_pickle_filename`
  - `selective_index_use_parallel`
  - `selective_index_use_average_input`
  - `selective_index_normalization_mode`
  - `selective_index_time_window_ms`
  - `selective_index_denominator_epsilon`
  - `selective_index_pair_names`
  - `selective_region_comparison_output_subdir`
  - `selective_region_comparison_pairwise_filename`
  - `selective_region_comparison_window_filename`
  - `selective_region_comparison_output_pickle_filename`
  - `selective_region_comparison_min_units_per_region`
  - `selective_region_comparison_min_regions_per_window`
  - `selective_region_comparison_n_permutations`
  - `selective_region_comparison_random_seed`
  - `selective_region_comparison_pvalue_correction`
  - `selective_region_comparison_alpha`
  - `selective_region_comparison_require_all_conditions_observed`
  - `selective_region_comparison_require_meets_min_trials`
  - `selective_region_comparison_require_selective_units`
  - `selective_region_comparison_pseudo_count`
  - `selective_region_comparison_alignment_cosine_threshold`
  - `population_pca_trial_input_modality`
  - `population_pca_trial_input_filename`
  - `population_pca_prefer_trial_input`
  - `population_pca_allow_trial_fallback`
  - `population_pca_smooth_before_average`
  - `population_pca_smoothing_sigma_ms`
  - `population_pca_input_subdir`
  - `population_pca_input_filename`
  - `population_pca_input_filename_split`
  - `population_pca_object_input_subdir`
  - `population_pca_object_input_filename`
  - `population_pca_output_subdir`
  - `population_pca_summary_filename`
  - `population_pca_timecourse_filename`
  - `population_pca_explained_variance_filename`
  - `population_pca_unit_inventory_filename`
  - `population_pca_output_pickle_filename`
  - `population_pca_window_ms`
  - `population_pca_max_components`
  - `population_pca_min_units_per_region`
  - `population_pca_require_all_conditions`
  - `population_pca_require_face_interactive_state`
  - `population_pca_use_parallel`
- `configs/ephys_fixation_neural_cross_correlation.yaml`
  - `trial_input_modality`
  - `trial_input_filename`
  - `signal_input_column`
  - `signal_window_ms`
  - `within_output_subdir` / `cross_output_subdir`
  - `within_output_filename` / `cross_output_filename`
  - `within_pair_average_output_filename` / `cross_pair_average_output_filename`
  - `pair_meta_within_output_subdir` / `pair_meta_cross_output_subdir`
  - `pair_meta_within_output_filename` / `pair_meta_cross_output_filename`
  - `pair_meta_within_output_csv_filename` / `pair_meta_cross_output_csv_filename`
  - `pair_meta_alpha`
  - `pair_meta_min_fixations`
  - `anchor_region`
  - `partner_regions`
  - `include_regions`
  - `roi_groups`
  - `signal_transform`
  - `xcorr_normalization`
  - `max_lag`
  - `use_parallel`
  - `parallelize_across_sessions`
  - `max_procs`
  - `pair_chunk_size`
