# scripts/behav/analysis

CLI entrypoints for analysis outputs written to `analysis_output_root`.

Scripts:
- `build_face_fixation_probability.py`
  Computes within-session, cross-session, and interactive-period face fixation probabilities.
- `build_out_of_roi_fixation_probability.py`
  Computes within-session and cross-session out-of-ROI fixation probabilities.
- `build_face_fix_cross_correlation.py`
  Computes within-session and cross-session face fixation cross-correlation outputs, and supports shuffled control workflows.
- `build_out_of_roi_fix_cross_correlation.py`
  Computes within-session and cross-session out-of-ROI fixation cross-correlation outputs, and supports shuffled control workflows.
- `build_face_fix_crosscorr_leader_follower.py`
  Computes face fixation leader-follower labels from within-session cross-correlation outputs and reports date-level, pair-level, and global summaries.
- `build_out_of_roi_fix_crosscorr_leader_follower.py`
  Computes out-of-ROI fixation leader-follower labels from within-session cross-correlation outputs and reports date-level, pair-level, and global summaries.
- `build_pupil_fixation_density_correlation.py`
  Computes per-session correlations between each pupil trace (`m1`, `m2`) and each face fixation density trace (`m1`, `m2`, `joint`).
  Supports session-level parallelization via config (`use_parallel`, `parallel_max_procs`) or CLI `--use-parallel`.
- `hpc_face_fix_crosscorr_shuffle_worker.py`
  Worker script used by array jobs for shuffled within-session cross-correlation pairs.
- `hpc_out_of_roi_fix_crosscorr_shuffle_worker.py`
  Worker script used by array jobs for shuffled within-session out-of-ROI cross-correlation pairs.

Primary configs:
- `configs/face_fixation_probability.yaml`
- `configs/out_of_roi_fixation_probability.yaml`
- `configs/face_fix_cross_correlation.yaml`
- `configs/out_of_roi_fix_cross_correlation.yaml`
- `configs/pupil_fixation_density_correlation.yaml`
- `configs/hpc_face_fix_cross_correlation_shuffle.yaml` (for `shuffle_submit_hpc` mode)
- `configs/hpc_out_of_roi_fix_cross_correlation_shuffle.yaml` (for `shuffle_submit_hpc` mode)

Leader-follower scripts reuse the corresponding cross-correlation configs above and read:
- within-session cross-correlation pickle (`within_filename` or algorithmic default)
- lag axis pickle (`lags_filename` or algorithmic default)
- by default they read from `crosscorr_output_subdir` at `leader_follower_time_scope` (default `whole`)
  and write outputs to `leader_follower_output_subdir` (default `<crosscorr_output_subdir>/leader_follower`).

Cross-correlation naming/layout notes:
- Cross-correlation pickles are written to `crosscorr_output_subdir` (default: `crosscorr_outputs`).
- Leader-follower pickles are written to `leader_follower_output_subdir` (default: `<crosscorr_output_subdir>/leader_follower`).
- If `within_filename`/`cross_filename`/`shuffle_output_filename`/`lags_filename` are null,
  names are generated from fixation label + `time_scope` (whole / interactive / non_interactive).

Leader-follower outputs:
- three pickle files with leader/follower calls:
  - session-level (`leader_follower_session_filename`)
  - date-level averages across sessions (`leader_follower_date_summary_filename`)
  - monkey-pair-level averages across sessions (`leader_follower_pair_summary_filename`)

Leader-follower pupil extraction controls:
- `leader_follower_property_use_all_fixations` (default `false`): use all fixations instead of ROI-matching fixations.
- `leader_follower_use_only_interactive_states` (default `false`): only keep pupil samples from fixation bins overlapping interactive periods.
- `leader_follower_interactive_modality` (default `interactive_periods`) and `leader_follower_interactive_state_label` (default `interactive`) choose the interactive-period source/state.
