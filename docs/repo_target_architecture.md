# Repository Architecture (Current Target)

This document tracks the intended stable architecture for the repository.
For the full operational guide (pipeline order, storage layout, conventions), see:
- `docs/repo_design_and_pipelines.md`

## Design Goals

- Stable script entrypoints with modular internals.
- Clear boundaries between pure logic and runtime orchestration.
- Shared infrastructure without collapsing behavioral and ephys domains.

## Target Tree

```text
src/dal_monte_2022_analysis/
  config/         # Config loading + path normalization
  core/           # Pure logic/primitives/contracts/stats/signal
  data/           # Dataclasses, loaders, transforms, migration helpers
  runtime/        # IO adapters, execution helpers, HPC adapters
  behav/          # Behavioral workflow stages
  ephys/          # Ephys workflow stages
  utils/          # Small generic helpers only
```

## Dependency Rules

- `core/*` must not depend on `runtime/*`.
- `runtime/*` should not import domain workflow modules (`behav/*`, `ephys/*`).
- `behav/*` and `ephys/*` may depend on `core/*`, `runtime/*`, `data/*`, `config/*`, `utils/*`.
- `scripts/*` should remain thin wrappers over `src/` APIs.

## Migration/Refactor Policy

- Migrate reusable logic first; keep wrappers only when needed for compatibility.
- Prefer central shared helpers for repeated concerns:
  - figure export (`runtime/io/plot_output.py`)
  - processed artifact scanning/loading (`runtime/io/processed_data.py`)
  - common stats/signal primitives (`core/stats`, `core/signal`)
- Remove wrappers once first-party imports no longer depend on them.
