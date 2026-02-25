# ephys

Ephys-domain package.

Subfolders are workflow stages:
- `preprocessing/`
- `features/`
- `analysis/`
- `plotting/`
- `modeling/`

Use shared modules from:
- `dal_monte_2022_analysis.data`
- `dal_monte_2022_analysis.config`
- `dal_monte_2022_analysis.utils`

Current preprocessing utility:
- `preprocessing/spike_data.py`
  Adds a `date` column to the unit-level spike table pickle from `session_name`
  for nomenclature consistency.

Current feature utility:
- `features/fixation_psth.py`
  Builds fixation-triggered unit PSTHs at trial level (session outputs) and
  date-level averaged PSTHs.

Current plotting utility:
- `plotting/fixation_psth.py`
  Builds per-unit fixation PSTH figures with condition-specific rasters and
  average firing-rate (mean ± SEM) traces.
