# scripts/ephys/features

CLI entrypoints for ephys feature generation.

Current scripts:
- `build_fixation_psth_trials.py`
  Legacy combined fixation-trial builder retained for backward compatibility.
  Each trial row stores both:
  - `psth_counts`: the configured PSTH counts (10 ms bins by default)
  - `spike_train_counts`: a higher-resolution spike-train vector
    (1 ms bins by default for fixation trials)
  Outputs are stored in `processed_data_root` under:
  `date=<date>/session=<session>/psth/fixations.pkl`
  (`fixations.pkl` is intentionally event-specific, not `shared.pkl`).
- `build_fixation_psth_trials_10ms.py`
  Builds per-fixation `10 ms` non-overlapping spike-count PSTHs only.
  Outputs are stored as:
  `date=<date>/session=<session>/psth/fixations_psth_10ms.pkl`
- `build_fixation_psth_trials_50ms_step_25ms.py`
  Builds per-fixation wide-binned spike-count PSTHs with `50 ms` bin width
  and `25 ms` stride for overlapping heatmap-style views.
  Outputs are stored as:
  `date=<date>/session=<session>/psth/fixations_psth_50ms_step_25ms.pkl`
- `build_fixation_spike_train_trials_1ms.py`
  Builds per-fixation `1 ms` spike-train counts only.
  Outputs are stored as:
  `date=<date>/session=<session>/psth/fixations_spike_train_1ms.pkl`
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
- The fixation and period trial-builder scripts print one example output row by default after completion.
- Use `--no-show-example` to disable.
