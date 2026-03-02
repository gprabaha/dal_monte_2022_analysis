# data

Shared data model and loader/transform modules used by behavioral and ephys pipelines.

## Structure

- `records/`: canonical dataclasses for behavioral and ephys payloads
- `loaders/`: source table loaders (`behavioral.py`, `ephys.py`)
- `transforms/`: shared transforms (annotation and cleaning helpers)
- `migrations/`: pickle migration helpers for legacy module-path compatibility

## Compatibility

- `gaze_data.py` and `spike_data.py` are compatibility shims for historic pickle module paths.
- `migrations/pickle_modules.py` can rewrite stored pickles to canonical modules.

## Usage Notes

- Prefer `data.records.*` and `data.loaders.*` as canonical imports in new code.
- Keep dataclasses lightweight and serialization-friendly.
