# plotting

Plotting utilities for analysis output tables.

Modules:
- `fixation_probability.py`
  Loads probability tables, computes statistical comparisons, and renders violin plots.
- `fix_crosscorr_leader_follower.py`
  Loads leader-follower monkey-role pupil, fixation-count, or fixation-duration summary/session tables and renders per-monkey leader-vs-follower violin panels.
- `fix_cross_correlation_m1_m2.py`
  Loads within-session, cross-session, and shuffled cross-correlation tables and renders observed-vs-control m1-m2 traces across whole/interactive/non-interactive scopes with per-lag paired t-tests.
- `interactive_period_durations.py`
  Loads interactive-period segment tables, computes period durations in seconds, and renders shared-axis histogram grids by monkey pair and by unique m1 (state-split), plus a separate all-pairs aggregate histogram with mean/quartile overlays.
- `interactive_period_detection.py`
  Builds per-session detection figures with fixation binary timelines (broken bars), individual fixation-density traces, joint face-density thresholding, and interactive-period overlays, writing outputs under per-date subfolders.
- `pupil_smoothing.py`
  Draws raw vs smoothed pupil timecourses for random sessions in a rows-by-agent QC panel layout.
- `face_fixation_probability.py`
  Backward-compatible wrapper around the fixation probability plotting API.
- `common.py`
  Shared style helpers (`rcParams`, figure sizing, p-value formatting).

Plot settings are loaded from `configs/plotting.yaml`.
