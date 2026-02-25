# scripts/behav/preprocessing

This folder contains the end-to-end preprocessing pipeline:

- `extract_data_from_raw_mat_files.py`: read raw `.mat` files and serialize per-session
  pickles for each modality.
- `clean_processed_data.py`: prune timelines and interpolate position/pupil samples.
- `build_smoothed_pupil_size.py`: smooth pupil traces with fixation-guided noise
  estimates, then interpolate non-fixation bins.
- `verify_data_pruning.py`: sanity-check a random session after cleaning.

Typical order of use is extract → clean → detect fixations (`scripts/behav/features`) →
build smoothed pupil → verify.
Outputs are written under the `processed_data_root` (see `configs/dataset.yaml`).
