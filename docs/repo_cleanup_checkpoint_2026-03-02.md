# Repo Cleanup Checkpoint (2026-03-02)

Note:
- This is a historical checkpoint snapshot.
- It reflects intermediate migration state from March 2, 2026 and is not the current architecture source of truth.
- Current architecture/pipeline docs are in `docs/repo_design_and_pipelines.md` and `docs/repo_target_architecture.md`.

Scope of this checkpoint:
- Step 1: structure/naming/duplication audit for `src/` and `configs/`
- Step 2: centralize pickle IO helpers and migrate repeated local wrappers

## 1) Audit Summary

### Structure Snapshot
- Top-level source domains are clear: `behav/`, `ephys/`, `combined/`, `data/`, `config/`, `utils/`.
- `combined/` is currently mostly placeholders (`analysis/features/modeling/plotting` packages with minimal content).
- `scripts/` mirrors domain folders but has mixed naming styles.

### Consistency Findings

1. Naming inconsistency for cross-correlation modules/scripts/configs
- Mixed forms exist: `crosscorr` and `cross_correlation`.
- Example paths:
  - `src/dal_monte_2022_analysis/behav/analysis/fix_crosscorr_leader_follower.py`
  - `src/dal_monte_2022_analysis/behav/analysis/fix_cross_correlation.py`
  - `scripts/behav/analysis/build_face_fix_crosscorr_leader_follower.py`
  - `scripts/behav/analysis/build_face_fix_cross_correlation.py`

2. Config layering inconsistency
- `configs/dataset.yaml` governs roots + behavioral modality discovery.
- `configs/ephys_data.yaml` is loader-schema-like and does not share the same structural pattern.
- Current split is workable but semantically uneven for future unified modality expansion.

3. Data model consistency gap (behavior vs ephys)
- Behavioral context uses `BehaviorRunContext(date, session, agent, monkey_name)`.
- Ephys context uses `EphysUnitContext(date, session_name, unit_uuid, session, ...)`.
- Similar concepts differ by key naming (`session` vs `session_name`) and optionality assumptions.

4. Utility duplication (improved in this checkpoint)
- Before this checkpoint, local pickle wrapper helpers were duplicated broadly.
- The same pattern (`open(..., "rb")`, `pickle.load`, `mkdir`, `pickle.dump`) existed across many modules.

5. Alias/wrapper modules are present and intentional
- Example: `behav/plotting/face_fixation_probability.py` wraps generic fixation probability plotting.
- This is acceptable for backward compatibility, but should be documented as alias modules.

## 2) Step-2 Implementation (IO Dedup)

### Added shared utility
- `src/dal_monte_2022_analysis/utils/io.py`
  - `load_pickle(path)`
  - `save_pickle(obj, path)`

### Migration result
- Replaced local pickle helper implementations with shared aliases/imports across analysis/features/preprocessing/plotting modules.
- Reduced local helper definitions from many duplicated wrappers to one special-case loader function.

Current counts after migration:
- `def _load_pickle` / `def _save_pickle`: **1** remaining definition
- `_load_pickle = load_pickle` / `_save_pickle = save_pickle`: **33** alias assignments

## 3) Recommended Next Architecture Pass (Major Redesign)

When proceeding to the bigger redesign (population analysis, PC projections, etc.), implement in this order:

1. Unified data contracts
- Introduce shared base context protocol with canonical keys:
  - `date`, `session`, `subject/agent`, `modality`, optional `unit_uuid`
- Keep ephys-specific fields on derived contexts.

2. Config model normalization
- Keep one root project config for paths/runtime defaults.
- Keep task configs schema-focused and modality-specific.
- Replace ad-hoc wrapper loader functions with typed config dataclasses/validators.

3. Core compute layer split
- Move reusable numeric logic into `behav/core` and `ephys/core`.
- Keep `analysis/` as orchestration; keep `plotting/` visualization-only.

4. Naming policy enforcement
- Standardize on one token style for cross-correlation (`cross_correlation` recommended).
- Maintain temporary compatibility wrappers/aliases for existing script entrypoints.

5. Expandable neural analysis namespace
- Add future-safe packages now:
  - `ephys/features/population.py`
  - `ephys/analysis/population_projection.py`
  - `ephys/plotting/population_projection.py`
- This avoids mixing upcoming PCA/projection workflows into single-unit PSTH modules.

## 4) Notes

- This checkpoint intentionally avoided broad file moves/renames to reduce break risk.
- Behavior of existing pipelines was preserved while reducing implementation redundancy.
