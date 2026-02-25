# data

This folder defines the in-memory data model for the project. The dataclasses in
`gaze_data.py` are small containers that mirror the extracted modalities, and
`cleaning.py` provides timeline pruning and interpolation helpers. Use
`load.py` to:
- index available processed files without loading them,
- load native data objects with metadata, or
- materialize DataFrames for analysis.

Keep these classes lightweight so they remain easy to serialize and debug.

Recent additions
- `BehaviorRunContext` (`RecordingContext` alias) clarifies behavioral metadata
  semantics: `date`=day, `session`=within-day run.
- `FixationBinaryVectorsData` stores per-agent binary fixation vectors aligned to the
  neural timeline.
- `spike_data.py` defines per-unit spike dataclasses (`NeuralUnitContext`,
  `SpikeTrainData`) with `day`/`run` aliases and a backward-compatible
  `SpikeUnitContext` alias.
- `spike_load.py` loads `spike_data.pkl`, normalizes schema fields, parses
  `session_name -> day/run` (`date/session` compatibility columns are also
  emitted), provides dataclass conversion helpers, and includes a day-level
  merge helper to attach `m1/m2` pair context from ephys metadata.
