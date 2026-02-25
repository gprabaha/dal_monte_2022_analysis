# dal_monte_2022_analysis package

This package holds the reusable core of the project:
- `config/` loads dataset configuration and normalizes paths.
- `data/` defines typed data containers, shared annotation helpers, and ephys loaders.
- `behav/` contains behavioral workflows organized by stage:
  `preprocessing/`, `features/`, `analysis/`, `plotting/`, `modeling/`.
- `ephys/` contains ephys workflows organized by stage:
  `preprocessing/`, `features/`, `analysis/`, `plotting/`, `modeling/`.
- `combined/` contains joint behavioral+ephys workflows organized by stage:
  `preprocessing/`, `features/`, `analysis/`, `plotting/`, `modeling/`.
- behavioral feature-product loading lives in `behav/features/load.py`.
- `utils/` provides shared helpers (paths, parallelism, fixation detection, and HPC job helpers).

Architecture rule
- Keep `behav/`, `ephys/`, and `combined/` for domain-specific workflow code.
- Keep `data/`, `config/`, and `utils/` shared at top level (do not duplicate under each domain).
- Keep `scripts/` mirroring the same domain-first structure used in `src/`.

Design philosophy
- Keep data objects small and serializable (pickle-friendly).
- Make each preprocessing step explicit and testable.
- Prefer straightforward, readable code over clever abstractions.
