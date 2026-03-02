# Repository Target Architecture

This document defines the target structure for a stable, script-driven analysis repository that preserves existing pipelines during migration.

## Design goals

- Keep pipeline entrypoints stable while internals evolve.
- Separate pure domain logic from IO, plotting, and HPC orchestration.
- Make dependencies directional and easy to reason about.
- Minimize breakage by using compatibility shims during transition.

## Target tree

```text
src/dal_monte_2022_analysis/
  config/                      # Typed config loading and normalization
  core/                        # Pure domain logic (no filesystem / plotting / HPC)
    contracts/
    behav/
      fixation_detection.py
      roi_groups.py
    signal/
      cross_correlation.py
    ephys/
    combined/
  data/                        # Data containers, records, migration helpers
  runtime/                     # Environment-specific adapters
    execution/
      parallel.py
    io/
    hpc/
      jobs.py
  behav/                       # Workflow layer (preprocess/features/analysis/plotting/modeling)
    preprocessing/
    features/
    analysis/
    plotting/
    modeling/
  ephys/                       # Same layered shape as behav
  combined/                    # Same layered shape as behav
  utils/                       # Shared utility helpers + backward-compatible import shims
```

## Dependency rules

- `core/*` may depend on: stdlib, numpy/scipy/sklearn, shared constants/types.
- `core/*` must not depend on: `runtime/*`, plotting modules, script modules, filesystem path builders.
- `runtime/*` may depend on: stdlib, config/path helpers, subprocess/environment integrations.
- `behav/*`, `ephys/*`, `combined/*` may depend on: `core/*`, `data/*`, `config/*`, `runtime/*`, `utils/*`.
- `scripts/*` should call public workflow/runtime APIs and avoid reimplementing logic.
- `utils/*` should only host generic helpers and temporary compatibility shims.

## What belongs in `core`

- Canonical domain algorithms (example: fixation/saccade detection).
- Domain-level transformations with deterministic inputs/outputs.
- Validation/invariant checks for domain objects.
- Shared domain dataclasses and typed contracts.

## What does not belong in `core`

- SLURM/dSQ submission, polling, or environment activation.
- Figure rendering and plotting style logic.
- Dataset path scanning and file read/write orchestration.
- CLI argument parsing.

## Migration strategy

1. Move one canonical algorithm/orchestrator at a time into `core`/`runtime`.
2. Keep old import paths as thin shims that re-export the new symbols.
3. Update first-party imports to canonical paths.
4. Keep shims for at least one release cycle before removal.
5. Add tests around moved modules before deleting shims.
