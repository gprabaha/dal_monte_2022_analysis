# scripts/ephys/features

CLI entrypoints for ephys feature generation.

Current scripts:
- `build_fixation_psth_trials.py`
  Builds per-fixation trial PSTH rows for all units in each date/session.
  Outputs are stored in `processed_data_root` under:
  `date=<date>/session=<session>/psth/fixations.pkl`
  (`fixations.pkl` is intentionally event-specific, not `shared.pkl`).
- `build_period_psth_trials.py`
  Builds per-period trial PSTH rows (interactive and non-interactive periods)
  centered at each period midpoint for all units in each date/session.
  Outputs are stored in `processed_data_root` under:
  `date=<date>/session=<session>/psth/interactive_periods.pkl`.
- `build_fixation_psth_averages.py`
  Builds date-level average PSTH summaries from trial PSTHs.
  Each averaged row stores `psth_mean`, `psth_sem`, and `n_trials`.
  By default, trial PSTH counts are smoothed, converted to firing rate (Hz)
  using the output bin size, and then averaged across trials.
  Outputs are stored in `analysis_output_root/ephys/psth/fixation_psth_averages/date=<date>/`
  as one file by default:
  - `fixations.pkl` containing:
    - `averages_split_by_interactive_state`
    - `averages_unsplit_by_interactive_state`
  Legacy separate-file mode is still available:
  - `fixations_split_by_interactive_state.pkl`
  - `fixations_unsplit_by_interactive_state.pkl`

Config:
- `configs/ephys_fixation_psth.yaml`
  Used by fixation PSTH scripts.
- `configs/ephys_period_psth.yaml`
  Used by period-centered PSTH trial script.
- `average_use_parallel`: parallelize averaging across dates.

Preview output:
- Both scripts print one example output row by default after completion.
- Use `--no-show-example` to disable.
