# dal_monte_2022_analysis package

This package holds the reusable core of the project:
- `config/` loads dataset configuration and normalizes paths.
- `data/` defines typed data containers, cleaning utilities, and a loader in `data/load.py`.
- `preprocessing/` handles indexing, extraction, cleaning, and fixation-guided pupil smoothing pipelines.
- `features/` builds fixations, fixation vectors/density, joint density, and interactive periods.
- `analysis/` computes fixation probability, fixation cross-correlation, and cross-correlation leader-follower outputs.
- `modeling/` fits latent-state models (currently Poisson-HSMM for joint face fixation).
- `plotting/` renders analysis figures from saved outputs.
- `utils/` provides shared helpers (paths, parallelism, fixation detection, and HPC job helpers).

Design philosophy
- Keep data objects small and serializable (pickle-friendly).
- Make each preprocessing step explicit and testable.
- Prefer straightforward, readable code over clever abstractions.
