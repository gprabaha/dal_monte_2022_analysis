# configs

This folder contains dataset-level configuration files. The intent is to keep all
paths and modality definitions in one place so that scripts stay thin and reusable.

`dataset.yaml` defines:
- `raw_data_root` and `processed_data_root`
- `agents` present in the dataset
- `modalities` with their folder names and file patterns
- `processed_data_layout` for how outputs are organized on disk

With the current config, processed outputs land under
`../local_data/dal_monte_2022/date=<date>/session=<session>/<modality>/`.

When adding new modalities or changing file naming conventions, update this file first.

Other configs in this folder:
- `gaze_event_detection.yaml` for fixation/saccade detection.
- `fixation_binary_vectors.yaml` for building fixation binary vectors aligned to the
  neural timeline.
- `fixation_density.yaml` for building smoothed fixation density vectors.
- `joint_face_fixation_density.yaml` for joint face fixation density analysis.
