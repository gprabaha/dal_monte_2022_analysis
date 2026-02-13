# scripts/plotting

CLI entrypoints for plotting analysis outputs.

Scripts:
- `plot_face_fixation_probability.py`
  Violin plots for within-session and cross-session face fixation probabilities.
- `plot_interactive_face_fixation_probability.py`
  Violin plots for interactive-period fixation probability summaries.
- `plot_out_of_roi_fixation_probability.py`
  Violin plots for within-session and cross-session out-of-ROI probabilities.
- `plot_face_fix_crosscorr_leader_follower_monkey_role_pupil.py`
  Violin panels comparing each monkey's pupil size as leader vs follower using face leader-follower analysis outputs.
- `plot_face_fix_crosscorr_leader_follower_pupil_global_overlay.py`
  One pooled leader-vs-follower pupil violin across all sessions with per-monkey leader/follower mean trends overlaid as connected points (uses raw per-session pupil arrays from `within_session_*_pupil_by_monkey_role_raw.pkl`).
- `plot_face_fix_crosscorr_leader_follower_monkey_role_fixation_count.py`
  Violin panels comparing each monkey's fixation count as leader vs follower using face leader-follower analysis outputs.
- `plot_face_fix_crosscorr_leader_follower_monkey_role_fixation_duration.py`
  Violin panels comparing each monkey's fixation duration (in bins) as leader vs follower using face leader-follower analysis outputs.
- `plot_interactive_period_duration_distributions.py`
  One multi-panel histogram figure (rows: monkey pairs; columns: interactive/non-interactive) plus a separate all-pairs aggregate histogram figure.
- `plot_interactive_period_detection.py`
  One per-session figure with top fixation binary timelines (m1 face, m2 face, m1 object) and bottom density-based interactive-period detection traces; outputs are grouped in date subfolders and support PDF-size reduction flags.
- `plot_smoothed_pupil_timecourses.py`
  QC figure with random-session raw vs smoothed pupil traces (rows: sessions, columns: m1/m2).

Inputs:
- analysis CSV outputs in `analysis_output_root` (table-driven plots)
- processed `interactive_periods` pickles + `ephys_days_and_monkeys.pkl` (duration distributions)
- processed `fixation_binary_vectors`, `fixation_density_vectors`, `joint_face_fixation_density`, and `interactive_periods` pickles (interactive period detection plots)
- processed `pupil_size` and `smoothed_pupil_size` pickles (pupil smoothing QC)
- style config in `configs/plotting.yaml`

Outputs:
- PDF figures written under the relevant analysis output subdirectory.
