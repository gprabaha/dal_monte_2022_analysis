# utils

Small, shared helpers that don’t belong to a specific domain module.
These utilities are intentionally shared across behavioral and ephys codepaths.

Currently includes:
- `paths.py` for consistent processed-data and analysis-output paths.
- `parallel.py` for worker-count selection (SLURM-aware).
- `filenames.py` for generic filename helpers (`ensure_filename`, override resolution).
- `fixation.py` as a compatibility shim (canonical: `core/behav/fixation_detection.py`).
- `hpc.py` as a compatibility shim (canonical: `runtime/hpc/jobs.py`).

Domain-specific filename policies should live in their domain package
(for example `core/behav/analysis_filenames.py`) rather than in `utils`.

`paths.build_processed_data_path` is the main helper for constructing
`.../date=<date>/session=<session>/<modality>/agent=<agent>.pkl` or `shared.pkl` paths.
`paths.scan_processed_data_paths` scans the processed data tree for available files.
`paths.list_processed_modalities` reports modality folders found on disk.
