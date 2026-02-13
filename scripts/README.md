# scripts

Thin CLI wrappers over package logic in `src/dal_monte_2022_analysis/`.

Subdirectories:
- `scripts/preprocessing/`: raw extraction, cleaning, fixation-guided pupil smoothing, and quick verification.
- `scripts/features/`: fixation/saccade detection and downstream feature products.
- `scripts/analysis/`: fixation probability and cross-correlation analyses.
- `scripts/modeling/`: latent-state model fitting workflows.
- `scripts/plotting/`: figure generation from analysis outputs.

Design rule: keep orchestration and argument parsing here, keep data logic in `src/`.
