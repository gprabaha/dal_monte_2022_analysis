# dal_monte_2022_analysis

Script-driven behavioral and ephys analysis pipelines for the Dal Monte 2022 dataset.

This repository uses a layered design:
- `scripts/` contains CLI entrypoints.
- `src/dal_monte_2022_analysis/` contains reusable implementation modules.
- `configs/` defines dataset paths and task parameters.

## Start Here

For a full architecture + pipeline walkthrough, read:
- `docs/repo_design_and_pipelines.md`

Then use folder-level docs as needed:
- `configs/README.md`
- `scripts/README.md`
- `src/dal_monte_2022_analysis/README.md`
- `hpc/README.md`

## Repository Layout

- `configs/`: YAML for project, dataset, feature, analysis, plotting, and HPC settings.
- `scripts/`: domain-first CLIs (`behav/`, `ephys/`) grouped by stage.
- `src/dal_monte_2022_analysis/`: package code organized into:
  - `behav/`, `ephys/`: workflow layer by stage.
  - `core/`: pure algorithms and shared primitives.
  - `runtime/`: IO, parallel execution, and HPC adapters.
  - `data/`: dataclasses, loaders, transforms, and pickle migration helpers.
  - `config/`: config loading/normalization.
  - `utils/`: small generic helpers.
- `hpc/`: generated job files / sbatch scripts / logs.
- `notebooks/`: exploratory work.

## Data Storage Conventions

Configured in `configs/dataset.yaml`:
- `raw_data_root`: source `.mat` and metadata inputs.
- `processed_data_root`: derived per-session feature artifacts.
- `analysis_output_root`: analysis tables and plots.

Processed layout:
- `processed_data_root/date=<date>/session=<session>/<modality>/agent=<agent>.pkl`
- `processed_data_root/date=<date>/session=<session>/<modality>/shared.pkl`

Analysis outputs are task-specific subdirs under `analysis_output_root`.

## Pipeline Summary

Behavioral stages:
1. Preprocessing (`scripts/behav/preprocessing/*`)
2. Feature extraction (`scripts/behav/features/*`)
3. Analysis (`scripts/behav/analysis/*`)
4. Plotting (`scripts/behav/plotting/*`)
5. Modeling (`scripts/behav/modeling/*`)

Ephys stages:
1. Preprocessing (`scripts/ephys/preprocessing/*`)
2. Feature extraction (`scripts/ephys/features/*`)
3. Analysis (`scripts/ephys/analysis/*`)
4. Plotting (`scripts/ephys/plotting/*`)

Detailed stage ordering and dependencies are documented in:
- `docs/repo_design_and_pipelines.md`
