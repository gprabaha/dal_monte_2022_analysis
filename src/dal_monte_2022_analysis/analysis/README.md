# analysis

Core analysis logic for derived behavioral metrics.

Modules:
- `fixation_probability.py`
  Builds within-session and cross-session fixation probability tables, plus interactive-period summaries.
- `fix_cross_correlation.py`
  Builds fixation binary-vector cross-correlation outputs for within-session and cross-session comparisons, including shuffled controls.
- `fix_crosscorr_leader_follower.py`
  Derives per-session leader/follower calls from within-session cross-correlation traces, then runs property analyses (fixation-count deltas and pupil-size-during-fixation comparisons) at session/date/pair/global levels.

Design notes:
- Inputs are indexed from processed modality pickles (usually `fixation_binary_vectors`).
- Outputs are written to analysis-specific directories via `build_analysis_output_dir(...)`.
