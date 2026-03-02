# src

`src/` contains the installable Python package (`dal_monte_2022_analysis`) used by all scripts.

Why `src/` layout:
- prevents accidental local-import shadowing
- supports editable installs (`pip install -e .`)

Primary package docs:
- `src/dal_monte_2022_analysis/README.md`
- `docs/repo_design_and_pipelines.md`
