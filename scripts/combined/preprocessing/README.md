# scripts/combined/preprocessing

Combined behavioral+ephys preprocessing utilities.

Scripts:
- `migrate_legacy_pickle_modules.py`
  One-time migration that rewrites pickles referencing legacy module paths
  (`dal_monte_2022_analysis.data.gaze_data`, `...data.spike_data`) so they
  deserialize directly against current modules.

Typical usage:
- dry run:
  `PYTHONPATH=src python scripts/combined/preprocessing/migrate_legacy_pickle_modules.py --dry-run`
- apply migration:
  `PYTHONPATH=src python scripts/combined/preprocessing/migrate_legacy_pickle_modules.py`
