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
- `features/period_psth.py`
  Builds period-centered (interactive/non-interactive) unit PSTHs at trial
  level for each date/session.

Current analysis utility:
- `analysis/fixation_selectivity.py`
  Tests per-unit fixation-category selectivity for interactive face,
  non-interactive face, and object across multiple PSTH windows.
- `analysis/fixation_neural_cross_correlation.py`
  Computes fixation-level neural PSTH cross-correlations for within-region
  and cross-region unit pairs (FFT-based xcorr per fixation/pair).

Current plotting utility:
- `plotting/fixation_psth.py`
  Builds per-unit fixation PSTH figures with condition-specific rasters and
  average firing-rate (mean ± SEM) traces.
- `plotting/period_psth.py`
  Builds per-unit interactive/non-interactive period PSTH figures with rasters
  and average firing-rate (mean ± SEM) traces.
- `plotting/fixation_selectivity_venn.py`
  Builds region-level Venn diagrams for fixation selectivity pair overlaps.
  Also supports selective-unit example overlays (per-bin significance ticks)
  through the selective plotting CLI script.
- `plotting/fixation_neural_cross_correlation.py`
  Builds date-level and all-date neural xcorr summary plots for within-region
  and cross-region pairs with condition-specific overlays.
  Exposes date-aggregated and kind-specific plotting entrypoints.
