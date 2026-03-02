# scripts/ephys/preprocessing

CLI entrypoints for ephys preprocessing.

Scripts:
- `add_date_column_from_session_name.py`
  Adds/updates a `date` column in the unit-level spike pickle by copying values
  from `session_name` (current compatibility step for nomenclature alignment).
- `migrate_legacy_pickle_modules.py`
  Rewrites legacy pickle module references (`data.gaze_data`, `data.spike_data`)
  to current module paths.

Typical usage:
- `python scripts/ephys/preprocessing/add_date_column_from_session_name.py`
- add `--dry-run` to validate only
- add `--overwrite-existing` if `date` already exists but differs
- `python scripts/ephys/preprocessing/migrate_legacy_pickle_modules.py --dry-run`
- `python scripts/ephys/preprocessing/migrate_legacy_pickle_modules.py`
