# scripts/ephys/analysis

CLI entrypoints for ephys analysis.

Current scripts:
- `build_fixation_selective_units.py`
  Computes fixation-pair selectivity per unit from trial PSTH data.
  For each unit, each pair of categories is tested across three windows:
  - `pre_fix`  (-500 ms to 0 ms)
  - `peri_fix` (-250 ms to 250 ms)
  - `post_fix` (0 ms to 500 ms)
  Pair is selective if any window is significant.
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
  - `selective_output_pickle_filename`
  - `selective_alpha`
  - `selective_test`
  - `selective_min_trials_per_condition`
  - `selective_use_parallel`
  - `selective_windows_ms`
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
