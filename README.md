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

Processed data layout
- Processed outputs live under the `processed_data_root` in `configs/dataset.yaml`.
- With the current config, this resolves to `../local_data/dal_monte_2022`.
- The layout pattern is `date={date}/session={session}/{modality}/`, so full paths look like
  `../local_data/dal_monte_2022/date=<date>/session=<session>/<modality>/agent=<agent>.pkl`
  or `../local_data/dal_monte_2022/date=<date>/session=<session>/<modality>/shared.pkl`.
- Each modality folder contains either `agent=<agent>.pkl` (per-agent) or `shared.pkl`.
- Example modalities include `gaze_position`, `pupil_size`, `neural_timeline`, `roi_vertices`,
  `fixations`, `saccades`, and `fixation_binary_vectors`.

Loading processed outputs
```python
import pickle
from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.utils.paths import build_processed_data_path

cfg = load_dataset_config("configs/dataset.yaml")
row = {"date": "01252022", "session": "1"}

# Per-agent data (gaze_position for m1)
path = build_processed_data_path(cfg, row, "gaze_position", "m1")
gaze = pickle.load(open(path, "rb"))

# Shared data (neural_timeline)
shared_path = build_processed_data_path(cfg, row, "neural_timeline", None)
timeline = pickle.load(open(shared_path, "rb"))
```

Getting back up to speed
- Re-read `configs/dataset.yaml` to confirm raw/processed roots and file patterns.
- Run `python scripts/preprocessing/verify_data_pruning.py` to sanity-check a session.
- If pickles fail to load, re-run extraction with current code to refresh module paths.
- Use `clean_processed_data.py` after any new extraction or config change.

Quick start
1) Extract raw `.mat` files: `python scripts/preprocessing/extract_data_from_raw_mat_files.py`
2) Clean/interpolate processed outputs: `python scripts/preprocessing/clean_processed_data.py`
3) Sanity-check a random session: `python scripts/preprocessing/verify_data_pruning.py`
