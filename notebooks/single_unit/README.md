# notebooks/single_unit

Single-neuron characterization of fixation responses in BLA, ACCg, dmPFC and OFC.

| File | What it is |
|---|---|
| `single_unit_fixation_responses.ipynb` | The notebook. Example-unit PSTH + raster grids, population fractions with stats, and temporal-specificity metrics. |
| `_build_notebook.py` | Authors the notebook from plain-Python source strings. |

## Regenerating

The notebook is *generated*, not hand-edited — the long code cells live in
`_build_notebook.py` so they stay diffable and lintable. Edit that file, then:

```bash
conda run -n gaze_processing python notebooks/single_unit/_build_notebook.py
conda run -n gaze_processing python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=3600 \
    notebooks/single_unit/single_unit_fixation_responses.ipynb
```

Everything runs in the **`gaze_processing`** conda env.

## Inputs

All precomputed by builders under `scripts/ephys/analysis/`, read from
`analysis_output_root/ephys/psth/`:

| Directory | Builder |
|---|---|
| `fixation_psth_selectivity/` | `build_fixation_selective_units.py` |
| `fixation_condition_dominance/` | `build_fixation_condition_dominance.py` |
| `fixation_peakiness/` | `build_fixation_peakiness.py` |
| `fixation_temporal_specificity/` | `build_fixation_temporal_specificity.py` |

The §2 example grids additionally read the 1 ms spike-train trial store for rasters.
Those renders take a few minutes each and are cached to
`fixation_psth_selective_unit_plots/nb_*.png`; set `FORCE_RERENDER = True` in the
notebook to rebuild them (needed after changing the exemplar unit lists in
`configs/ephys_fixation_psth.yaml`).

Summary tables are written to `analysis_output_root/ephys/psth/single_unit_notebook/`.
