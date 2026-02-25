# data

This folder defines the in-memory data model for the project. The dataclasses in
`behavioral_data.py` and `ephys_data.py` are small containers that mirror the
extracted modalities, and `cleaning.py` provides timeline pruning and
interpolation helpers for behavioral streams.

This module is intentionally shared across domains (`behav`, `ephys`, `combined`)
rather than duplicated per-domain.

Use:
- `load.py` to load behavioral data modalities (`index_behavioral_data`,
  `load_behavioral_data_objects`, `load_behavioral_data_dataframe`) and ephys
  unit data (`load_ephys_unit_dataframe`, `load_ephys_units`).
- `annotate.py` to attach shared pair labels (`m1_name`, `m2_name`,
  `pair_label`) to behavioral/ephys tables by date.

Keep these classes lightweight so they remain easy to serialize and debug.

Canonical structure
- `BehaviorRunContext` (`RecordingContext` alias) clarifies behavioral metadata
  semantics: `date`=day, `session`=within-day run.
- `FixationBinaryVectorsData` stores per-agent binary fixation vectors aligned to the
  neural timeline.
- `ephys_data.py` defines per-unit ephys dataclasses (`EphysUnitContext`,
  `UnitSpikeData`) and includes a wideband channel placeholder for future
  modality support.
- `load.py` is the data loader module (behavioral data + ephys data).
- processed behavioral/feature product loading now lives in
  `behav/features/load.py`.
- `annotate.py` is the single annotation module for shared label/context joins.
- `gaze_data.py` and `spike_data.py` are compatibility shims for loading legacy
  pickles; use `scripts/combined/preprocessing/migrate_legacy_pickle_modules.py`
  to rewrite old pickles to current module paths.
