# scripts/analysis

CLI entrypoints for analysis outputs written to `analysis_output_root`.

Scripts:
- `build_face_fixation_probability.py`
  Computes within-session, cross-session, and interactive-period face fixation probabilities.
- `build_out_of_roi_fixation_probability.py`
  Computes within-session and cross-session out-of-ROI fixation probabilities.
- `build_face_fix_cross_correlation.py`
  Computes within-session and cross-session face fixation cross-correlation outputs, and supports shuffled control workflows.
- `hpc_fix_crosscorr_shuffle_worker.py`
  Worker script used by array jobs for shuffled within-session cross-correlation pairs.

Primary configs:
- `configs/face_fixation_probability.yaml`
- `configs/out_of_roi_fixation_probability.yaml`
- `configs/face_fix_cross_correlation.yaml`
- `configs/hpc_fix_cross_correlation_shuffle.yaml` (for `shuffle_submit_hpc` mode)
