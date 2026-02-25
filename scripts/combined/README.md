# scripts/combined

Combined behavioral+ephys CLI entrypoints.

Subfolders:
- `preprocessing/`
- `features/`
- `analysis/`
- `plotting/`
- `modeling/`
- `bash/`

These scripts call into `src/dal_monte_2022_analysis/combined/*`.

Note:
- `preprocessing/migrate_legacy_pickle_modules.py` is a cross-domain migration
  helper that rewrites old pickles to current module paths.
