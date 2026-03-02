# configs

YAML configuration files for dataset paths, task settings, plotting styles, and HPC submission.

## Config Hierarchy

Primary entrypoint:
- `project.yaml`
  - `dataset_cfg_path`
  - `ephys_data_cfg_path`
  - `plotting_cfg_path`

Core datasets:
- `dataset.yaml`
  - `raw_data_root`
  - `processed_data_root`
  - `analysis_output_root`
  - modality file-discovery patterns
  - processed layout pattern (`date={date}/session={session}/{modality}`)
- `ephys_data.yaml`
  - unit-table schema/aliases and source path or filename

Task configs:
- Behavioral preprocessing/features:
  - `gaze_event_detection.yaml`, `fixation_binary_vectors.yaml`, `fixation_density.yaml`,
    `joint_face_fixation_density.yaml`, `interactive_periods.yaml`, `pupil_smoothing.yaml`
- Behavioral analysis/modeling:
  - `face_fixation_probability.yaml`, `out_of_roi_fixation_probability.yaml`,
    `face_fix_cross_correlation.yaml`, `out_of_roi_fix_cross_correlation.yaml`,
    `pupil_fixation_density_correlation.yaml`, `face_fixation_hsmm.yaml`
- Ephys features/analysis:
  - `ephys_fixation_psth.yaml`, `ephys_period_psth.yaml`,
    `ephys_fixation_neural_cross_correlation.yaml`
- HPC:
  - `hpc_gaze_event_detection.yaml`,
    `hpc_face_fix_cross_correlation_shuffle.yaml`,
    `hpc_out_of_roi_fix_cross_correlation_shuffle.yaml`
- Plot styling:
  - `plotting.yaml`

## Operational Notes

- Scripts typically accept explicit config paths; defaults point into this directory.
- Config loading and path normalization are implemented in
  `src/dal_monte_2022_analysis/config/load.py`.
- Add new task configs here when introducing new stage modules or output products.
