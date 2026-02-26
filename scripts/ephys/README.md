# scripts/ephys

Ephys CLI entrypoints.

Subfolders:
- `preprocessing/`
- `features/`
- `analysis/`
- `plotting/`
- `modeling/`
- `bash/`

These scripts call into `src/dal_monte_2022_analysis/ephys/*`.

Current implemented step:
- `preprocessing/add_date_column_from_session_name.py`
- `features/build_fixation_psth_trials.py`
- `features/build_period_psth_trials.py`
- `features/build_fixation_psth_averages.py`
- `analysis/build_fixation_selective_units.py`
- `plotting/plot_fixation_psth_units.py`
- `plotting/plot_fixation_selectivity_venn.py`
- `plotting/plot_selective_fixation_psth_units.py`
