# configs

Configuration files for the full pipeline.

## Core dataset config

- `dataset.yaml`
  Defines dataset roots, agents, modality discovery patterns, and processed output layout.

With current settings, processed outputs are written under:
`../local_data/dal_monte_2022/data_files/date=<date>/session=<session>/<modality>/`

## Feature configs

- `gaze_event_detection.yaml`
- `fixation_binary_vectors.yaml`
- `fixation_density.yaml`
- `joint_face_fixation_density.yaml`
- `interactive_periods.yaml`
- `pupil_smoothing.yaml`

## Ephys data config

- `ephys_data.yaml`
  Canonical unit-level ephys loader settings (column aliases, required fields,
  file path/filename).
  Used by `scripts/ephys/preprocessing/add_date_column_from_session_name.py`
  to locate and normalize the unit-level ephys pickle.

## Analysis configs

- `face_fixation_probability.yaml`
- `out_of_roi_fixation_probability.yaml`
- `face_fix_cross_correlation.yaml`
- `out_of_roi_fix_cross_correlation.yaml`
- `face_fixation_hsmm.yaml`

Cross-correlation configs also hold optional leader-follower settings used by:
- `scripts/behav/analysis/build_face_fix_crosscorr_leader_follower.py`
- `scripts/behav/analysis/build_out_of_roi_fix_crosscorr_leader_follower.py`

## HPC configs

- `hpc_gaze_event_detection.yaml`
- `hpc_face_fix_cross_correlation_shuffle.yaml`
- `hpc_out_of_roi_fix_cross_correlation_shuffle.yaml`

## Plotting config

- `plotting.yaml`

When adding a new modality or output product, start by updating `dataset.yaml` and the relevant task-specific config in this directory.
