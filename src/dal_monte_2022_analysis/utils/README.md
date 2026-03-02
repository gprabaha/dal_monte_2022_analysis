# utils

Small, shared helpers that don’t belong to a specific domain module.
These utilities are intentionally shared across behavioral, ephys, and combined codepaths.

Currently includes:
- `paths.py` for consistent processed-data and analysis-output paths.
- `parallel.py` for worker-count selection (SLURM-aware).
- `fixation.py` as the canonical fixation/saccade API.
- `hpc.py` as the canonical HPC job API.

Backward-compatible modules remain available:
- `fixation_utils.py`
- `hpc_utils.py`

`paths.build_processed_data_path` is the main helper for constructing
`.../date=<date>/session=<session>/<modality>/agent=<agent>.pkl` or `shared.pkl` paths.
`paths.scan_processed_data_paths` scans the processed data tree for available files.
`paths.list_processed_modalities` reports modality folders found on disk.
