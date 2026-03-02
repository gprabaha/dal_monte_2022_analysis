# utils

Small, cross-domain helpers that are generic enough to be shared by behavioral,
ephys, and runtime modules.

Current scope:
- `io.py` for plain pickle read/write helpers.
- `paths.py` for deterministic processed-data and analysis-output path handling.
- `filenames.py` for generic filename helpers (`ensure_filename`, override resolution).

Non-generic logic should live in domain/runtime packages, for example:
- `core/behav/analysis_filenames.py`
- `core/signal/cross_correlation.py`
- `runtime/execution/parallel.py`
- `runtime/hpc/jobs.py`
