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
  Outputs are stored in `analysis_output_root/ephys/psth/fixation_psth_averages/date=<date>/fixations.pkl`.

Config:
- `configs/ephys_fixation_psth.yaml`
  Used by fixation PSTH scripts.
- `configs/ephys_period_psth.yaml`
  Used by period-centered PSTH trial script.
- `average_use_parallel`: parallelize averaging across dates.

Preview output:
- Both scripts print one example output row by default after completion.
- Use `--no-show-example` to disable.
