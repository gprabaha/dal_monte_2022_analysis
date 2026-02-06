# dal_monte_2022_analysis

This repository is a lightweight, script-driven pipeline for extracting, cleaning, and
analyzing behavioral gaze data (positions, pupil size, ROI rectangles, timelines), along
with downstream feature detection (fixations, saccades, density vectors, interactive
periods) and analysis outputs (fixation probability tables + plots). The design favors
small, composable modules over a heavy framework so that you can pick up the code quickly
after time away.

High-level flow
- `configs/` defines where raw data lives and how processed outputs are laid out.
- `scripts/` provides thin CLI wrappers for preprocessing, feature detection, analysis, and plotting.
- `src/` contains the reusable package code (IO, data models, cleaning, features, analysis).

Design principles
- Keep IO and data modeling separate.
- Store processed outputs as simple pickles keyed by session/agent.
- Prefer small, explicit scripts that are easy to re-run and debug.

Processed data layout
- Processed outputs live under the `processed_data_root` in `configs/dataset.yaml`.
 - With the current config, this resolves to `../local_data/dal_monte_2022/data_files/`.
- The layout pattern is `date={date}/session={session}/{modality}/`, so full paths look like
  `../local_data/dal_monte_2022/data_files/date=<date>/session=<session>/<modality>/agent=<agent>.pkl`
  or `../local_data/dal_monte_2022/data_files/date=<date>/session=<session>/<modality>/shared.pkl`.
- Each modality folder contains either `agent=<agent>.pkl` (per-agent) or `shared.pkl`.
- Example modalities include `gaze_position`, `pupil_size`, `neural_timeline`, `roi_vertices`,
  `fixations`, `saccades`, `fixation_binary_vectors`, `fixation_density_vectors`,
  `joint_face_fixation_density`, and `interactive_periods`.
- Analysis outputs (CSV tables) live under `analysis_output_root` in `configs/dataset.yaml`.

Listing and loading processed outputs
```python
import pickle
from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.utils.paths import (
    build_processed_data_path,
    list_processed_modalities,
    scan_processed_data_paths,
)

cfg = load_dataset_config("configs/dataset.yaml")

# List available modalities on disk.
modalities = sorted(list_processed_modalities(cfg))
print(modalities)

# List all files for a modality (optionally filter by date/session/agent).
rows = scan_processed_data_paths(cfg, "fixation_density_vectors", agents=["m1"])
print(rows[0])

# Load the first available file for that modality.
with open(rows[0]["path"], "rb") as f:
    density = pickle.load(f)

# Load a specific row/agent if you already know the session.
row = {"date": "01252022", "session": "1"}
path = build_processed_data_path(cfg, row, "gaze_position", "m1")
gaze = pickle.load(open(path, "rb"))

# Shared data (agentless).
shared_path = build_processed_data_path(cfg, row, "neural_timeline", None)
timeline = pickle.load(open(shared_path, "rb"))
```

Feature detection (processed pickles)
- Detect fixations and saccades from gaze position: `python scripts/features/detect_fixations_and_saccades.py`
- Optional HPC run for gaze event detection: `python scripts/features/detect_fixations_and_saccades.py --run-hpc`
- Build fixation binary vectors: `python scripts/features/build_fixation_binary_vectors.py`
- Build fixation density vectors: `python scripts/features/build_fixation_density.py`
- Build joint face fixation density (m1+m2): `python scripts/features/build_joint_face_fixation_density.py`
- Derive interactive periods from joint density: `python scripts/features/build_interactive_periods.py`
- Quick QC of fixation location labels: `python scripts/features/verify_fixation_detection.py`

Feature configs live in `configs/` (for example: `gaze_event_detection.yaml`,
`hpc_gaze_event_detection.yaml`, `fixation_binary_vectors.yaml`, `fixation_density.yaml`,
`joint_face_fixation_density.yaml`, `interactive_periods.yaml`).

Analysis (CSV outputs under `analysis_output_root`)
- Face fixation probability (within-session, cross-session, and interactive): `python scripts/analysis/build_face_fixation_probability.py`
- Out-of-ROI fixation probability (within-session and cross-session): `python scripts/analysis/build_out_of_roi_fixation_probability.py`

Analysis configs live in `configs/` (for example: `face_fixation_probability.yaml`,
`out_of_roi_fixation_probability.yaml`).

Plotting
- Face fixation probability plots: `python scripts/plotting/plot_face_fixation_probability.py`
- Interactive-period face fixation plots: `python scripts/plotting/plot_interactive_face_fixation_probability.py`
- Out-of-ROI fixation plots: `python scripts/plotting/plot_out_of_roi_fixation_probability.py`

Plotting styles are configured in `configs/plotting.yaml`.

Getting back up to speed
- Re-read `configs/dataset.yaml` to confirm raw/processed roots and file patterns.
- Run `python scripts/preprocessing/verify_data_pruning.py` to sanity-check a session.
- If pickles fail to load, re-run extraction with current code to refresh module paths.
- Use `python scripts/preprocessing/clean_processed_data.py` after any new extraction or config change.

Quick start
1) Extract raw `.mat` files: `python scripts/preprocessing/extract_data_from_raw_mat_files.py`
2) Clean/interpolate processed outputs: `python scripts/preprocessing/clean_processed_data.py`
3) Detect fixations/saccades: `python scripts/features/detect_fixations_and_saccades.py`
4) Build fixation vectors + density: `python scripts/features/build_fixation_binary_vectors.py` then `python scripts/features/build_fixation_density.py`
5) Build joint density + interactive periods: `python scripts/features/build_joint_face_fixation_density.py` then `python scripts/features/build_interactive_periods.py`
6) Run fixation probability analyses: `python scripts/analysis/build_face_fixation_probability.py` and `python scripts/analysis/build_out_of_roi_fixation_probability.py`
7) Plot outputs (optional): `python scripts/plotting/plot_face_fixation_probability.py`
8) Sanity-check a random session: `python scripts/preprocessing/verify_data_pruning.py`
