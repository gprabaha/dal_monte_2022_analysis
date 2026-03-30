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

- Relative config references and path-valued settings are resolved relative to
  the repo root and normalized to absolute `Path` objects.
- This keeps local runs and HPC job submission consistent even when the current
  working directory is not the repo root.

Keep this module dependency-light so all layers can import it safely.
