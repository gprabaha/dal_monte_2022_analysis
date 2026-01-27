# dal_monte_2022_analysis

This repository is a lightweight, script-driven pipeline for extracting and cleaning
behavioral gaze data (positions, pupil size, ROI rectangles, and timelines) from MATLAB
exports. The design favors small, composable modules over a heavy framework so that you
can pick up the code quickly after time away.

High-level flow
- `configs/` defines where raw data lives and how processed outputs are laid out.
- `scripts/` provides thin CLI wrappers that call the package functions.
- `src/` contains the reusable package code (IO, data models, cleaning).

Design principles
- Keep IO and data modeling separate.
- Store processed outputs as simple pickles keyed by session/agent.
- Prefer small, explicit scripts that are easy to re-run and debug.

Getting back up to speed
- Re-read `configs/dataset.yaml` to confirm raw/processed roots and file patterns.
- Run `python scripts/preprocessing/verify_data_pruning.py` to sanity-check a session.
- If pickles fail to load, re-run extraction with current code to refresh module paths.
- Use `clean_processed_data.py` after any new extraction or config change.

Quick start
1) Extract raw `.mat` files: `python scripts/preprocessing/extract_data_from_raw_mat_files.py`
2) Clean/interpolate processed outputs: `python scripts/preprocessing/clean_processed_data.py`
3) Sanity-check a random session: `python scripts/preprocessing/verify_data_pruning.py`
