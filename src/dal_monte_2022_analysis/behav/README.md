# behav

Behavioral workflow implementation modules.

## Stage Layout

- `preprocessing/`: raw extraction, cleaning, and preprocessing transforms
- `features/`: fixation events, binary vectors, densities, interactive periods
- `analysis/`: fixation probabilities, cross-correlation, leader/follower, pupil-density correlation
- `plotting/`: figure generation for behavioral analyses and QC
- `modeling/`: behavioral modeling (HSMM)

## Typical Flow

1. Build cleaned per-session processed artifacts.
2. Generate feature modalities (`fixations`, `fixation_binary_vectors`, `fixation_density_vectors`, `joint_face_fixation_density`, `interactive_periods`).
3. Compute analysis tables under `analysis_output_root`.
4. Build publication/QC plots from analysis outputs (and some processed modalities where needed).

## Shared Dependencies

Behavioral modules use shared layers for reuse and consistency:
- `core/behav/*` for algorithmic primitives
- `runtime/io/*` for processed/analysis path and file operations
- `core/stats/*` for reusable statistical helpers

For script entrypoints and exact order, see:
- `scripts/behav/README.md`
- `docs/repo_design_and_pipelines.md`
