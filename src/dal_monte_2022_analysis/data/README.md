# data

This folder defines the in-memory data model for the project. The dataclasses in
`records/` mirror extracted modalities, and `cleaning.py` provides timeline
pruning and interpolation helpers for behavioral streams.

This module is intentionally shared across domains (`behav`, `ephys`, `combined`)
rather than duplicated per-domain.

Use:
- `loaders/behavioral.py` for behavioral modalities
  (`index_behavioral_data`, `load_behavioral_data_objects`,
  `load_behavioral_data_dataframe`).
- `loaders/ephys.py` for ephys unit data
  (`load_ephys_unit_dataframe`, `load_ephys_units`).
- `annotate.py` to attach shared pair labels (`m1_name`, `m2_name`,
  `pair_label`) to behavioral/ephys tables by date.

Keep these classes lightweight so they remain easy to serialize and debug.

Canonical structure
- `BehaviorRunContext` (`RecordingContext` alias) clarifies behavioral metadata
  semantics: `date`=day, `session`=within-day run.
- `FixationBinaryVectorsData` stores per-agent binary fixation vectors aligned to the
  neural timeline.
- `records/ephys.py` defines per-unit ephys dataclasses (`EphysUnitContext`,
  `UnitSpikeData`) and includes a wideband channel placeholder for future modality support.
- `load.py` is a compatibility re-export for `loaders/*`.
- processed behavioral/feature product loading now lives in
  `behav/features/load.py`.
- `annotate.py` is the single annotation module for shared label/context joins.
- `behavioral_data.py`, `ephys_data.py`, `behavioral_records.py`,
  `ephys_records.py`, `gaze_data.py`, and `spike_data.py` are compatibility
  shims for legacy imports/pickles.
- `migrations/pickle_modules.py` is the canonical pickle-module migration entrypoint.
