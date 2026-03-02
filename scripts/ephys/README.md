# scripts/ephys

Ephys CLI entrypoints.

## Stage Folders

- `preprocessing/`
- `features/`
- `analysis/`
- `plotting/`
- `modeling/`
- `bash/`

## Typical Execution Order

1. Preprocessing
- `add_date_column_from_session_name.py`
- optional migration: `migrate_legacy_pickle_modules.py`

2. Features
- `build_fixation_psth_trials.py`
- `build_period_psth_trials.py`
- `build_fixation_psth_averages.py`

3. Analysis
- `build_fixation_selective_units.py`
- `build_within_region_fixation_neural_cross_correlation.py`
- `build_cross_region_fixation_neural_cross_correlation.py`

4. Plotting
- PSTH unit plots
- selectivity Venn summaries
- three-way triangular selectivity population summary
- neural cross-correlation summaries

Detailed script-level descriptions are in each subfolder README.
