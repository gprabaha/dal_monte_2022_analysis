# data

This folder defines the in-memory data model for the project. The dataclasses in
`records/` mirror extracted modalities, and `transforms/cleaning.py` provides
timeline pruning and interpolation helpers for behavioral streams.

This module is intentionally shared across domains (`behav`, `ephys`)
rather than duplicated per-domain.

Use:
- `loaders/behavioral.py` for behavioral modalities
  (`index_behavioral_data`, `load_behavioral_data_objects`,
  `load_behavioral_data_dataframe`).
- `loaders/ephys.py` for ephys unit data
  (`load_ephys_unit_dataframe`, `load_ephys_units`).
- `transforms/annotate.py` to attach shared pair labels (`m1_name`, `m2_name`,
  `pair_label`) to behavioral/ephys tables by date.

Keep these classes lightweight so they remain easy to serialize and debug.

Canonical structure
- `BehaviorRunContext` (`RecordingContext` alias) clarifies behavioral metadata
  semantics: `date`=day, `session`=within-day run.
- `FixationBinaryVectorsData` stores per-agent binary fixation vectors aligned to the
  neural timeline.
- `records/ephys.py` defines per-unit ephys dataclasses (`EphysUnitContext`,
  `UnitSpikeData`) and includes a wideband channel placeholder for future modality support.
- `loaders/` is split into behavioral and ephys loader modules.
- processed behavioral/feature product loading now lives in
  `behav/features/load.py`.
- `transforms/` contains shared table/session transforms (`annotate`, `cleaning`).
- `gaze_data.py` and `spike_data.py` remain as compatibility shims for
  legacy pickle module paths.
- `migrations/pickle_modules.py` is the canonical pickle-module migration entrypoint.
