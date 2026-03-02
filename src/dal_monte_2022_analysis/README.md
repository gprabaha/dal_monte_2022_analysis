# dal_monte_2022_analysis package

This package holds the reusable core of the project:
- `config/` loads dataset configuration and normalizes paths.
- `core/` contains pure domain logic and invariants (no plotting, filesystem, or HPC code), plus shared contracts under `core/contracts/`.
  Current canonical core modules include:
  `core/behav/fixation_detection.py`,
  `core/behav/roi_groups.py`,
  and `core/signal/cross_correlation.py`.
- `data/` defines typed data containers, shared annotation helpers, and ephys loaders.
- `behav/` contains behavioral workflows organized by stage:
  `preprocessing/`, `features/`, `analysis/`, `plotting/`, `modeling/`.
- `ephys/` contains ephys workflows organized by stage:
  `preprocessing/`, `features/`, `analysis/`, `plotting/`, `modeling/`.
- `combined/` contains joint behavioral+ephys workflows organized by stage:
  `preprocessing/`, `features/`, `analysis/`, `plotting/`, `modeling/`.
- `runtime/` contains environment-specific adapters (execution/parallelism, processed-data IO, HPC job submission).
- behavioral feature-product loading lives in `behav/features/load.py`.
- `utils/` provides shared helpers (paths, parallelism, io) plus compatibility shims for older imports.

Architecture rules
- Keep `behav/`, `ephys/`, and `combined/` for domain-specific workflow code.
- Keep `data/`, `config/`, `core/`, and `runtime/` shared at top level (do not duplicate under each domain).
- Put pure logic in `core/`; put side-effecting adapters (HPC, CLI, cluster env) in `runtime/`.
- Keep `scripts/` mirroring the same domain-first structure used in `src/`.

Design philosophy
- Keep data objects small and serializable (pickle-friendly).
- Make each preprocessing step explicit and testable.
- Prefer straightforward, readable code over clever abstractions.
