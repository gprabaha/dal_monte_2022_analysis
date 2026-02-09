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

## Analysis configs

- `face_fixation_probability.yaml`
- `out_of_roi_fixation_probability.yaml`
- `face_fix_cross_correlation.yaml`

## HPC configs

- `hpc_gaze_event_detection.yaml`
- `hpc_fix_cross_correlation_shuffle.yaml`

## Plotting config

- `plotting.yaml`

When adding a new modality or output product, start by updating `dataset.yaml` and the relevant task-specific config in this directory.
