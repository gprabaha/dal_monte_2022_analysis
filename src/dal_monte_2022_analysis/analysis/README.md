# analysis

Core analysis logic for derived behavioral metrics.

Modules:
- `fixation_probability.py`
  Builds within-session and cross-session fixation probability tables, plus interactive-period summaries.
- `fix_cross_correlation.py`
  Builds fixation binary-vector cross-correlation outputs for within-session and cross-session comparisons, including shuffled controls.
- `fix_crosscorr_leader_follower.py`
  Derives per-session leader/follower calls from within-session cross-correlation traces and summarizes per-date and total counts for each `monkey_name_m1`/`monkey_name_m2` pair, including `m1_lead_fraction`.

Design notes:
- Inputs are indexed from processed modality pickles (usually `fixation_binary_vectors`).
- Outputs are written to analysis-specific directories via `build_analysis_output_dir(...)`.
