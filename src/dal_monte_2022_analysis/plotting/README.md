# plotting

Plotting utilities for analysis output tables.

Modules:
- `fixation_probability.py`
  Loads probability tables, computes statistical comparisons, and renders violin plots.
- `face_fixation_probability.py`
  Backward-compatible wrapper around the fixation probability plotting API.
- `common.py`
  Shared style helpers (`rcParams`, figure sizing, p-value formatting).

Plot settings are loaded from `configs/plotting.yaml`.
