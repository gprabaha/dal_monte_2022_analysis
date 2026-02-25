# dal_monte_2022_analysis

Script-driven pipeline for extracting, cleaning, and analyzing gaze data from the Dal Monte 2022 dataset. The repository keeps CLI entrypoints in `scripts/` and reusable logic in `src/`.

## Repository map

- `configs/`: YAML configuration for dataset paths, feature settings, analysis settings, plotting, and HPC submission.
- `scripts/`: domain-first CLI wrappers under `scripts/behav/`, `scripts/ephys/`, and `scripts/combined/`.
- `src/dal_monte_2022_analysis/`: domain-first package code under `behav/`, `ephys/`, and `combined/`, plus shared `data/`, `config/`, and `utils/`.
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
- `python scripts/behav/preprocessing/extract_data_from_raw_mat_files.py`
- `python scripts/behav/preprocessing/clean_processed_data.py`
- `python scripts/behav/preprocessing/verify_data_pruning.py`

2. Build features
- `python scripts/behav/features/detect_fixations_and_saccades.py`
- `python scripts/behav/preprocessing/build_smoothed_pupil_size.py` (requires `fixations`)
- `python scripts/behav/features/build_fixation_binary_vectors.py`
- `python scripts/behav/features/build_fixation_density.py`
- `python scripts/behav/features/build_joint_face_fixation_density.py`
- `python scripts/behav/features/build_interactive_periods.py`
- `python scripts/behav/features/verify_fixation_detection.py`

3. Run analyses
- `python scripts/behav/analysis/build_face_fixation_probability.py`
- `python scripts/behav/analysis/build_out_of_roi_fixation_probability.py`
- `python scripts/behav/analysis/build_face_fix_cross_correlation.py`
- `python scripts/behav/analysis/build_out_of_roi_fix_cross_correlation.py`
- `python scripts/behav/analysis/build_face_fix_crosscorr_leader_follower.py`
- `python scripts/behav/analysis/build_out_of_roi_fix_crosscorr_leader_follower.py`

4. Plot outputs
- `python scripts/behav/plotting/plot_face_fixation_probability.py`
- `python scripts/behav/plotting/plot_interactive_face_fixation_probability.py`
- `python scripts/behav/plotting/plot_out_of_roi_fixation_probability.py`

## HPC workflows

Two HPC-enabled paths are supported:
- Gaze event detection (`configs/hpc_gaze_event_detection.yaml`)
- Shuffled face fixation cross-correlation (`configs/hpc_face_fix_cross_correlation_shuffle.yaml`)
- Shuffled out-of-ROI fixation cross-correlation (`configs/hpc_out_of_roi_fix_cross_correlation_shuffle.yaml`)

Related worker scripts live in:
- `scripts/behav/features/hpc_fixation_saccade_detection_worker.py`
- `scripts/behav/analysis/hpc_face_fix_crosscorr_shuffle_worker.py`
- `scripts/behav/analysis/hpc_out_of_roi_fix_crosscorr_shuffle_worker.py`

Generated artifacts are written under `hpc/`.

## Quick start

1. Extract raw data:
`python scripts/behav/preprocessing/extract_data_from_raw_mat_files.py`
2. Clean processed outputs:
`python scripts/behav/preprocessing/clean_processed_data.py`
3. Detect fixations/saccades:
`python scripts/behav/features/detect_fixations_and_saccades.py`
4. Build fixation-guided smoothed pupil:
`python scripts/behav/preprocessing/build_smoothed_pupil_size.py`
5. Build vectors and density:
`python scripts/behav/features/build_fixation_binary_vectors.py`
`python scripts/behav/features/build_fixation_density.py`
6. Build joint/interactive features:
`python scripts/behav/features/build_joint_face_fixation_density.py`
`python scripts/behav/features/build_interactive_periods.py`
7. Run analyses:
`python scripts/behav/analysis/build_face_fixation_probability.py`
`python scripts/behav/analysis/build_out_of_roi_fixation_probability.py`
`python scripts/behav/analysis/build_face_fix_cross_correlation.py`
`python scripts/behav/analysis/build_out_of_roi_fix_cross_correlation.py`
`python scripts/behav/analysis/build_face_fix_crosscorr_leader_follower.py`
`python scripts/behav/analysis/build_out_of_roi_fix_crosscorr_leader_follower.py`
8. Plot:
`python scripts/behav/plotting/plot_face_fixation_probability.py`
