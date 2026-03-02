# ephys

Ephys workflow implementation modules.

## Stage Layout

- `preprocessing/`: source-table normalization and migration utilities
- `features/`: fixation and period PSTH trial/average builders
- `analysis/`: fixation selectivity and neural cross-correlation analyses
- `plotting/`: per-unit PSTH, selectivity, and cross-correlation plotting
- `modeling/`: reserved for future ephys modeling modules

## Typical Flow

1. Normalize ephys unit table (`session_name`/`date` consistency).
2. Build PSTH-derived processed outputs from behavioral/ephys inputs.
3. Run selectivity and cross-correlation analyses.
4. Generate per-unit and summary figures.

## Shared Dependencies

Ephys modules use shared layers for consistency:
- `core/ephys/*`, `core/stats/*`, `core/signal/*`
- `runtime/io/*` for loading/indexing/export
- `runtime/execution/*` for parallel orchestration

For script entrypoints and exact order, see:
- `scripts/ephys/README.md`
- `docs/repo_design_and_pipelines.md`
