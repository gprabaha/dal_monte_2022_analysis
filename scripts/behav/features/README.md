# scripts/behav/features

CLI entrypoints for feature-generation stages on processed data.

Scripts:
- `detect_fixations_and_saccades.py`
  Detects gaze events from gaze position data; supports local or HPC execution.
- `build_fixation_binary_vectors.py`
  Converts fixation event intervals into timeline-aligned binary vectors.
- `build_fixation_density.py`
  Smooths fixation vectors into density vectors.
- `build_joint_face_fixation_density.py`
  Combines m1/m2 face densities into a joint density trace.
- `build_interactive_periods.py`
  Converts joint density into labeled interactive/non-interactive intervals.
- `verify_fixation_detection.py`
  Samples fixation files and summarizes ROI label frequencies for QC.
- `hpc_fixation_saccade_detection_worker.py`
  Worker script used by array jobs for gaze event detection.

Typical run order:
1. `detect_fixations_and_saccades.py`
2. `build_fixation_binary_vectors.py`
3. `build_fixation_density.py`
4. `build_joint_face_fixation_density.py`
5. `build_interactive_periods.py`
