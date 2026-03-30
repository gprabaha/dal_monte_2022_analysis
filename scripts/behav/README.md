# scripts/behav

Behavioral CLI entrypoints.

## Stage Folders

- `preprocessing/`
- `features/`
- `analysis/`
- `plotting/`
- `modeling/`

## Typical Execution Order

1. Preprocessing
- `extract_data_from_raw_mat_files.py`
- `clean_processed_data.py`
- optional QC: `verify_data_pruning.py`

2. Features
- `detect_fixations_and_saccades.py`
- `build_fixation_binary_vectors.py`
- `build_fixation_density.py`
- `build_joint_face_fixation_density.py`
- `build_interactive_periods.py`
- optional QC: `verify_fixation_detection.py`

3. Analysis
- fixation probabilities
- fixation cross-correlation (including shuffled controls)
- leader/follower summaries
- pupil-density correlations

4. Plotting
- probability, cross-correlation, leader/follower, interactive-period, and QC plots

5. Modeling
- HSMM (`build_face_fixation_hsmm.py`)

Detailed script-level descriptions are in each subfolder README.

SLURM launchers live under `hpc/behav/`.
