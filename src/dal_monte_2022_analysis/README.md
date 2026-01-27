# dal_monte_2022_analysis package

This package holds the reusable core of the project:
- `config/` loads dataset configuration and normalizes paths.
- `data/` defines typed data containers and cleaning utilities.
- `io/` handles indexing, extraction, and cleaning pipelines.
- `utils/` provides small helpers shared across modules.

Design philosophy
- Keep data objects small and serializable (pickle-friendly).
- Make each IO step explicit and testable.
- Prefer straightforward, readable code over clever abstractions.
