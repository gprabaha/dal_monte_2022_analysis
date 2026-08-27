# Codex Instructions For This Repository

## Environment

- Work from the repository root: `/gpfs/milgram/project/chang/pg496/repositories/dal_monte_2022_analysis`.
- Use the `gaze_processing` Conda environment for analysis, tests, notebooks, and scripts:
  `conda run -n gaze_processing python ...`
- Conda may need unsandboxed execution on this system because its env directories are not writable from the default sandbox.

## Data Layout

- Canonical config entry point: `configs/project.yaml`.
- Dataset config: `configs/dataset.yaml`.
- Processed behavioral data root: `../local_data/dal_monte_2022/data_files/`.
- Analysis output root: `../local_data/dal_monte_2022/analysis_outputs/`.
- Processed per-session files use:
  `date={date}/session={session}/{modality}/agent={m1|m2}.pkl` or `shared.pkl`.
- Common behavior modalities:
  `gaze_position`, `roi_vertices`, `pupil_size`, `smoothed_pupil_size`,
  `fixations`, `saccades`, `fixation_binary_vectors`,
  `fixation_density_vectors`, `joint_face_fixation_density`,
  `interactive_periods`.

## Source Structure

- `src/dal_monte_2022_analysis/config`: config loading and path resolution.
- `src/dal_monte_2022_analysis/data`: records, loaders, migrations, and transforms.
- `src/dal_monte_2022_analysis/core`: reusable pure primitives for behavior, ephys, signal, and stats.
- `src/dal_monte_2022_analysis/behav`: behavior feature, analysis, plotting, modeling, and preprocessing modules.
- `src/dal_monte_2022_analysis/ephys`: ephys feature, analysis, plotting, modeling, and preprocessing modules.
- `scripts/behav` and `scripts/ephys`: runnable pipeline entry points.
- `notebooks`: exploratory and review notebooks. Put behavior-specific exploratory notebooks under `notebooks/behavior`.

## Expansion Strategy

- Put all functions and classes under `src/dal_monte_2022_analysis/...`. Scripts and notebooks should be thin orchestration/display layers that import from `src`; do not define reusable functions or classes in scripts or notebooks.
- Use existing loaders such as `index_behavioral_processed_data_from_cfg`, `load_behavioral_data_objects`, and `load_feature_objects` instead of hand-building paths.
- Use existing record dataclasses and core helpers for ROI geometry, period masks, fixation vectors, and cross-correlation primitives.
- Keep generated figures/tables under the configured local data `analysis_output_root`, not in the repository and not inside `src`.
- For large gaze arrays, stream session files and accumulate summaries or histograms; avoid concatenating all sessions into memory.
- Preserve the existing behavior/ephys split. Shared pure math or indexing helpers belong in `core`; domain workflows belong in `behav` or `ephys`.

## Conventions

- Dates are MMDDYYYY strings and sessions are stringified integers.
- Agents are `m1` and `m2`; shared session products use `agent=None`/`shared.pkl`.
- Interactive periods are stored as DataFrames with inclusive `start` and `stop` sample indices and `state` values `interactive` or `non_interactive`.
- Fixation binary vectors are sample-level arrays keyed by ROI labels such as `face`, `object`, and `out_of_roi`.
- Existing behavioral sample-index analyses assume 1 kHz sampling unless a task explicitly checks the neural timeline.

## Output Conventions

- Use `build_analysis_output_dir(cfg, "<analysis_name>")` for analysis outputs so files land under `../local_data/dal_monte_2022/analysis_outputs/`.
- Write summary tables as CSV with explicit, analysis-specific filenames.
- Save manuscript-facing figures as both editable PDF and high-resolution PNG.
- Use illustrator-friendly matplotlib settings where practical: embedded TrueType fonts (`pdf.fonttype=42`, `ps.fonttype=42`), transparent backgrounds, and no top/right axis spines on standard bar/box/scatter plots.
- Keep notebooks focused on loading/displaying generated outputs and optionally calling a thin script or imported `src` entry point to regenerate them.
