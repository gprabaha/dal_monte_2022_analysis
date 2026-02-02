# utils

Small, shared helpers that don’t belong to a specific domain module.

Currently includes:
- `paths.py` for consistent processed-data paths.
- `parallel.py` for worker-count selection (SLURM-aware).

`paths.build_processed_data_path` is the main helper for constructing
`.../date=<date>/session=<session>/<modality>/agent=<agent>.pkl` or `shared.pkl` paths.
`paths.scan_processed_data_paths` scans the processed data tree for available files.
`paths.list_processed_modalities` reports modality folders found on disk.
