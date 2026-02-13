# dal_monte_2022_analysis

Script-driven pipeline for extracting, cleaning, and analyzing gaze data from the Dal Monte 2022 dataset. The repository keeps CLI entrypoints in `scripts/` and reusable logic in `src/`.

## Repository map

- `configs/`: YAML configuration for dataset paths, feature settings, analysis settings, plotting, and HPC submission.
- `scripts/`: CLI wrappers for preprocessing, feature building, analysis, and plotting.
- `src/dal_monte_2022_analysis/`: package code for preprocessing, data models, features, analysis, plotting, and utilities.
- `hpc/`: generated job files, sbatch scripts, and logs for dSQ/SLURM runs.
- `notebooks/`: exploratory notebooks.
- `plots/`: exported figure outputs.

See folder-specific docs in:
- `configs/README.md`
- `scripts/README.md`
- `src/dal_monte_2022_analysis/README.md`
- `hpc/README.md`

## Data layout

`configs/dataset.yaml` defines:
- `raw_data_root`
- `processed_data_root`
- `analysis_output_root`
- `processed_data_layout.pattern`

With current defaults, processed pickles are written to:

`../local_data/dal_monte_2022/data_files/date=<date>/session=<session>/<modality>/`

Each modality directory contains either:
- `agent=<agent>.pkl` (agent-specific)
- `shared.pkl` (agentless/shared)

Analysis outputs are written under `analysis_output_root`.

## Pipeline overview

1. Preprocess raw `.mat` data
- `python scripts/preprocessing/extract_data_from_raw_mat_files.py`
- `python scripts/preprocessing/clean_processed_data.py`
- `python scripts/preprocessing/verify_data_pruning.py`

2. Build features
- `python scripts/features/detect_fixations_and_saccades.py`
- `python scripts/preprocessing/build_smoothed_pupil_size.py` (requires `fixations`)
- `python scripts/features/build_fixation_binary_vectors.py`
- `python scripts/features/build_fixation_density.py`
- `python scripts/features/build_joint_face_fixation_density.py`
- `python scripts/features/build_interactive_periods.py`
- `python scripts/features/verify_fixation_detection.py`

3. Run analyses
- `python scripts/analysis/build_face_fixation_probability.py`
- `python scripts/analysis/build_out_of_roi_fixation_probability.py`
- `python scripts/analysis/build_face_fix_cross_correlation.py`
- `python scripts/analysis/build_out_of_roi_fix_cross_correlation.py`
- `python scripts/analysis/build_face_fix_crosscorr_leader_follower.py`
- `python scripts/analysis/build_out_of_roi_fix_crosscorr_leader_follower.py`

4. Plot outputs
- `python scripts/plotting/plot_face_fixation_probability.py`
- `python scripts/plotting/plot_interactive_face_fixation_probability.py`
- `python scripts/plotting/plot_out_of_roi_fixation_probability.py`

## HPC workflows

Two HPC-enabled paths are supported:
- Gaze event detection (`configs/hpc_gaze_event_detection.yaml`)
- Shuffled fixation cross-correlation (`configs/hpc_fix_cross_correlation_shuffle.yaml`)

Related worker scripts live in:
- `scripts/features/hpc_fixation_saccade_detection_worker.py`
- `scripts/analysis/hpc_fix_crosscorr_shuffle_worker.py`

Generated artifacts are written under `hpc/`.

## Quick start

1. Extract raw data:
`python scripts/preprocessing/extract_data_from_raw_mat_files.py`
2. Clean processed outputs:
`python scripts/preprocessing/clean_processed_data.py`
3. Detect fixations/saccades:
`python scripts/features/detect_fixations_and_saccades.py`
4. Build fixation-guided smoothed pupil:
`python scripts/preprocessing/build_smoothed_pupil_size.py`
5. Build vectors and density:
`python scripts/features/build_fixation_binary_vectors.py`
`python scripts/features/build_fixation_density.py`
6. Build joint/interactive features:
`python scripts/features/build_joint_face_fixation_density.py`
`python scripts/features/build_interactive_periods.py`
7. Run analyses:
`python scripts/analysis/build_face_fixation_probability.py`
`python scripts/analysis/build_out_of_roi_fixation_probability.py`
`python scripts/analysis/build_face_fix_cross_correlation.py`
`python scripts/analysis/build_out_of_roi_fix_cross_correlation.py`
`python scripts/analysis/build_face_fix_crosscorr_leader_follower.py`
`python scripts/analysis/build_out_of_roi_fix_crosscorr_leader_follower.py`
8. Plot:
`python scripts/plotting/plot_face_fixation_probability.py`
