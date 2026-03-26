# data

Shared data model and loader/transform modules used by behavioral and ephys pipelines.

## Structure

- `records/`: canonical dataclasses for behavioral and ephys payloads
- `loaders/`: raw-source and processed-data loaders (`behavioral.py`, `ephys.py`)
- `transforms/`: shared transforms (annotation and cleaning helpers)
- `migrations/`: pickle migration helpers for legacy module-path compatibility

## Compatibility

- `gaze_data.py`, `behavioral_data.py`, and `spike_data.py` are compatibility
  shims for historic pickle module paths.
- `migrations/pickle_modules.py` can rewrite stored pickles to canonical modules.

## Usage Notes

- Prefer `data.records.*` and `data.loaders.*` as canonical imports in new code.
- Prefer `index_behavioral_source_data*` for raw behavioral session discovery and
  `runtime.io.processed_data` or `index_behavioral_processed_data_from_cfg(...)`
  for processed artifact enumeration.
- Pair-context metadata is configured in `configs/dataset.yaml` via
  `pair_context.*` so new raw datasets can reuse the same local processed layout.
- Keep dataclasses lightweight and serialization-friendly.
