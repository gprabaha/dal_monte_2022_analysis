# runtime

Environment-aware adapters used by domain workflows.

## Submodules

- `runtime/io/`
  - `processed_data.py`: processed artifact path building, scans, and pickle IO adapters
  - `analysis_index.py`: analysis tree indexing helpers
  - `plot_output.py`: shared figure-export behavior (extension normalization, output creation)
- `runtime/execution/`
  - process-count and task-runner helpers for local parallel workflows
- `runtime/hpc/`
  - job-file generation and SLURM/dSQ submission polling helpers

## Responsibility Boundary

`runtime/` is where side effects and environment details live.
Domain workflow code (`behav/`, `ephys/`) should depend on `runtime/` for IO/export/execution infrastructure instead of reimplementing these concerns.
