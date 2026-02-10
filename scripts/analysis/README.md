# scripts/analysis

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
- `hpc_face_fix_crosscorr_shuffle_worker.py`
  Worker script used by array jobs for shuffled within-session cross-correlation pairs.
- `hpc_out_of_roi_fix_crosscorr_shuffle_worker.py`
  Worker script used by array jobs for shuffled within-session out-of-ROI cross-correlation pairs.

Primary configs:
- `configs/face_fixation_probability.yaml`
- `configs/out_of_roi_fixation_probability.yaml`
- `configs/face_fix_cross_correlation.yaml`
- `configs/out_of_roi_fix_cross_correlation.yaml`
- `configs/hpc_face_fix_cross_correlation_shuffle.yaml` (for `shuffle_submit_hpc` mode)
- `configs/hpc_out_of_roi_fix_cross_correlation_shuffle.yaml` (for `shuffle_submit_hpc` mode)

Leader-follower scripts reuse the corresponding cross-correlation configs above and read:
- within-session cross-correlation pickle (`within_filename`)
- lag axis pickle (`lags_filename` or `<fixation_label>_crosscorrelation_lags.pkl`)

Leader-follower outputs:
- session-level CSV with per-session leader labels and lead scores.
- fixation-count property summaries at date/pair/global levels (`n_pos`, `n_neg`, `n_zero`, `mean_delta`, `delta_consistency`).
- pupil-during-fixation property summaries at session/date/pair/global levels (`n_lead`, `n_follow`, `lead_mean`, `follow_mean`, `mean_diff`, `p`, `sig`, `higher`).
- monkey-role pupil tables (`within_session_*_pupil_by_monkey_role.csv`, `summary_*_pupil_by_monkey_role.csv`) for plotting in `scripts/plotting`.
- monkey-role fixation-count tables (`within_session_*_fixation_count_by_monkey_role.csv`, `summary_*_fixation_count_by_monkey_role.csv`) for plotting in `scripts/plotting`.
- monkey-role fixation-duration tables (`within_session_*_fixation_duration_by_monkey_role.csv`, `summary_*_fixation_duration_by_monkey_role.csv`) for plotting in `scripts/plotting`.
