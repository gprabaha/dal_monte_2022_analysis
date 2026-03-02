# Repo Design And Pipelines

This guide describes the current architecture, conventions, and execution flow of the repository.

## 1) Design Philosophy

Primary goals:
- Keep pipeline entrypoints stable (`scripts/*`) while internals remain modular.
- Separate pure logic from IO/runtime concerns.
- Keep behavioral and ephys workflows separate, while sharing reusable primitives.
- Preserve deterministic data layout and naming conventions.

Practical rules:
- Pure algorithms live in `core/`.
- Filesystem, serialization, plotting export, parallelism, and HPC integration live in `runtime/`.
- Workflow orchestration stays in `behav/` and `ephys/`.
- CLIs stay thin and call package modules in `src/`.

## 2) Source Tree Model

`src/dal_monte_2022_analysis/`
- `behav/`: behavioral workflow stages (`preprocessing`, `features`, `analysis`, `plotting`, `modeling`).
- `ephys/`: ephys workflow stages (`preprocessing`, `features`, `analysis`, `plotting`, `modeling`).
- `core/`: pure cross-workflow logic.
  - `core/behav/`: fixation detection, preprocessing primitives, feature primitives, analysis filename/primitives.
  - `core/ephys/`: shared ephys analysis primitives.
  - `core/signal/`: signal algorithms (e.g., cross-correlation kernels).
  - `core/stats/`: hypothesis test utilities.
  - `core/contracts/`: shared data contracts.
- `runtime/`: environment adapters.
  - `runtime/io/`: processed-data scanning/loading, analysis path scanning, figure export helper.
  - `runtime/execution/`: process-count and task runner helpers.
  - `runtime/hpc/`: job file generation and submission tracking.
- `data/`: dataclasses, records, loaders, transforms, migration utilities.
- `config/`: YAML config loading and path normalization.
- `utils/`: small generic helpers (`paths`, `filenames`, low-level pickle IO).

## 3) Dependency Direction

Intended direction:
- `behav/*`, `ephys/*` -> `core/*`, `runtime/*`, `data/*`, `config/*`, `utils/*`
- `runtime/*` -> `config/*`, `utils/*` (and standard libs)
- `core/*` -> pure libs only (no runtime/HPC/plotting/filesystem orchestration)
- `scripts/*` -> workflow/runtime APIs in `src/`

Avoid:
- `runtime/*` importing domain workflow modules.
- `core/*` importing `runtime/*`.

## 4) Config Model

Canonical config entrypoint:
- `configs/project.yaml`
  - points to `dataset_cfg_path`, `ephys_data_cfg_path`, `plotting_cfg_path`.

Main config files:
- `configs/dataset.yaml`
  - defines `raw_data_root`, `processed_data_root`, `analysis_output_root`, modality discovery patterns, and processed-data layout.
- `configs/ephys_data.yaml`
  - defines unit table loading schema/aliases.
- Task configs (features/analysis/plotting/HPC)
  - define task-specific settings and filenames.

Loader behavior:
- `dal_monte_2022_analysis.config.load.load_config(...)` infers/normalizes config type.
- Dataset/project configs normalize key filesystem values to `Path` objects.

## 5) Data Storage Conventions

Configured in `configs/dataset.yaml`:
- `processed_data_layout.pattern: date={date}/session={session}/{modality}`

Processed artifacts:
- Agent-specific: `.../<modality>/agent=<agent>.pkl`
- Shared: `.../<modality>/shared.pkl`

Examples:
- `.../date=01012020/session=1/fixations/agent=m1.pkl`
- `.../date=01012020/session=1/joint_face_fixation_density/shared.pkl`

Analysis artifacts:
- Written under `analysis_output_root/<task_subdir>/...`
- Tables may be CSV/PKL; plots may be PDF/PNG depending on task settings.

Plot export convention:
- Shared figure writing goes through `runtime.io.plot_output.save_figure(...)`.

## 6) Behavioral Pipeline Order

Typical execution order:

1. Preprocessing (`scripts/behav/preprocessing`)
- `extract_data_from_raw_mat_files.py`
- `clean_processed_data.py`
- optional QC: `verify_data_pruning.py`

2. Feature extraction (`scripts/behav/features`)
- `detect_fixations_and_saccades.py`
- `build_fixation_binary_vectors.py`
- `build_fixation_density.py`
- `build_joint_face_fixation_density.py`
- `build_interactive_periods.py`
- optional QC: `verify_fixation_detection.py`

3. Analysis (`scripts/behav/analysis`)
- fixation probabilities (face / out-of-ROI)
- fixation cross-correlations (face / out-of-ROI)
- leader-follower summaries
- pupil-vs-density correlations

4. Plotting (`scripts/behav/plotting`)
- probability violins
- cross-correlation summaries
- leader/follower monkey-role plots
- interactive period detection and duration distributions
- pupil smoothing QC and pupil-density correlation plots

5. Modeling (`scripts/behav/modeling`)
- HSMM modeling.

## 7) Ephys Pipeline Order

Typical execution order:

1. Preprocessing (`scripts/ephys/preprocessing`)
- `add_date_column_from_session_name.py`
- optional migration: `migrate_legacy_pickle_modules.py`

2. Feature extraction (`scripts/ephys/features`)
- `build_fixation_psth_trials.py`
- `build_period_psth_trials.py`
- `build_fixation_psth_averages.py`

3. Analysis (`scripts/ephys/analysis`)
- `build_fixation_selective_units.py`
- `build_within_region_fixation_neural_cross_correlation.py`
- `build_cross_region_fixation_neural_cross_correlation.py`

4. Plotting (`scripts/ephys/plotting`)
- per-unit fixation/period PSTH plots
- fixation selectivity Venn summaries
- neural cross-correlation summary plots (within/cross/both)

## 8) Core vs Runtime Responsibilities

`core/`
- deterministic transforms and algorithms
- domain primitives used by multiple workflows
- no plotting/HPC orchestration

`runtime/`
- path scanning/indexing helpers
- serialization adapters
- figure output/export behavior
- process-parallel helpers
- HPC job submission utilities

## 9) Legacy Compatibility Notes

- `data/gaze_data.py` and `data/spike_data.py` remain as pickle compatibility shims.
- `data/migrations/pickle_modules.py` provides pickle module-path migration helpers.
- `utils/io.py` includes legacy-aware unpickling remaps.

## 10) Conventions For New Code

When adding functionality:
1. Put pure algorithmic logic into `core/` first.
2. Put data/path/serialization orchestration into `runtime/`.
3. Keep domain-specific stage orchestration in `behav/` or `ephys/`.
4. Add or update a CLI in `scripts/` only as a thin wrapper.
5. Update relevant config file in `configs/`.
6. Document new module/task in the nearest README.

When adding new outputs:
- Keep processed outputs in `processed_data_root` with modality folders.
- Keep analysis/plot outputs in task-specific subdirs under `analysis_output_root`.
- Use deterministic filenames and avoid hidden side effects.
