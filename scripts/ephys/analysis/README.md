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
