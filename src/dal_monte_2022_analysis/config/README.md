# config

Configuration loading and normalization utilities.

## Entry Points

- `load_config(path, config_type=None)`
  - generic loader with type inference (`project`, `dataset`, `ephys_data`, `hpc`, `generic`)
- `load_project_config(...)`
- `load_dataset_config(...)`
- `resolve_dataset_cfg_path(...)`
- `resolve_ephys_cfg_path(...)`

## Path Normalization

- Project config resolves referenced config paths relative to `configs/`.
- Dataset/project roots are resolved relative to the source config file and normalized to absolute `Path` objects.
- Ephys/HPC configs normalize file paths for robust script/runtime use.

Keep this module dependency-light so all layers can import it safely.
