# scripts/ephys/analysis

CLI entrypoints for ephys analysis.

Current scripts:
- `build_fixation_selective_units.py`
  Computes fixation-pair selectivity per unit from trial PSTH data.
  For each unit, each pair of categories is tested across configured windows
  (default now includes four):
  - `pre_fix`  (-500 ms to 0 ms)
  - `peri_fix` (-250 ms to 250 ms)
  - `post_fix` (0 ms to 500 ms)
  - `full_fix` (-500 ms to 500 ms)
  Pair is selective if any window is significant.
  Also writes one per-unit/per-window three-condition summary table
  (`condition_window_means.csv`) with mean firing rates for:
  - interactive face
  - non-interactive face
  - object
  and normalized relative components for triangular plotting.
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
  Computes fixation-level neural PSTH cross-correlations for within-region
  unit pairs (nC2 per region per fixation).
  Output pickle stores:
  - `cross_correlations` (per-fixation pair traces)
  - `pair_averages` (per-session pair averages by condition)
- `build_cross_region_fixation_neural_cross_correlation.py`
  Computes fixation-level neural PSTH cross-correlations for cross-region
  unit pairs (anchor-region units x partner-region units per fixation).
  Output pickle stores:
  - `cross_correlations` (per-fixation pair traces)
  - `pair_averages` (per-session pair averages by condition)

Config:
- `configs/ephys_fixation_psth.yaml`
  - `selective_output_subdir`
  - `selective_window_stats_filename`
  - `selective_pair_summary_filename`
  - `selective_unit_summary_filename`
  - `selective_condition_summary_filename`
  - `selective_output_pickle_filename`
  - `selective_alpha`
  - `selective_test`
  - `selective_min_trials_per_condition`
  - `selective_use_parallel`
  - `selective_windows_ms`
  - `selective_index_output_subdir`
  - `selective_index_timeseries_filename`
  - `selective_index_output_pickle_filename`
  - `selective_index_use_parallel`
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
- `configs/ephys_fixation_neural_cross_correlation.yaml`
  - `trial_input_modality`
  - `trial_input_filename`
  - `within_output_subdir` / `cross_output_subdir`
  - `within_output_filename` / `cross_output_filename`
  - `within_pair_average_output_filename` / `cross_pair_average_output_filename`
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
