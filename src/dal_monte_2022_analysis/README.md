# dal_monte_2022_analysis package

Reusable implementation modules for behavioral and ephys pipelines.

## Package Structure

- `behav/`: behavioral workflow stages (`preprocessing`, `features`, `analysis`, `plotting`, `modeling`)
- `ephys/`: ephys workflow stages (`preprocessing`, `features`, `analysis`, `plotting`, `modeling`)
- `core/`: pure logic, contracts, shared stats/signal primitives
- `runtime/`: IO/export adapters, execution helpers, HPC integration
- `data/`: dataclasses, table loaders, transforms, migration helpers
- `config/`: YAML loading + path normalization
- `utils/`: minimal generic helpers (`paths`, `filenames`, low-level pickle IO)

## Architectural Boundaries

- Keep algorithmic/domain primitives in `core/`.
- Keep side-effecting orchestration (path scans, save/load, submission, process orchestration) in `runtime/`.
- Keep stage-specific orchestration in `behav/` and `ephys/`.
- Keep script/CLI parsing in `scripts/`.

## Key Shared Modules

- `runtime/io/processed_data.py`: processed artifact path building/scanning/loading/saving
- `runtime/io/analysis_index.py`: analysis output tree scanning
- `runtime/io/plot_output.py`: shared figure export helper
- `core/stats/hypothesis.py`: shared hypothesis tests
- `core/signal/cross_correlation.py`: cross-correlation primitives

For pipeline order and conventions, see:
- `docs/repo_design_and_pipelines.md`
