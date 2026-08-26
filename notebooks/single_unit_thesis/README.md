# notebooks/single_unit_thesis

Thesis-chapter notebooks for the single-unit fixation analyses. The exploratory
notebook in [`notebooks/single_unit/`](../single_unit) is kept as-is; these two are the
streamlined, figure-first versions written for the chapter itself.

| File | Role |
|---|---|
| `single_unit_thesis_chapter_results.ipynb` | Main chapter results: unit yield, fixation-pair selectivity Venn, high/low Dominant-Peak Prominence examples with the score's construction drawn on BLA 600, the DPP distribution, coefficient-of-variation violins, and preferred-category bars. |
| `single_unit_trace_metrics_appendix.ipynb` | Appendix: the full set of mean firing-rate trace metrics, their distributions across the modulated subpopulation, which are redundant, and what DPP does and does not capture. |
| `_build_notebooks.py` | Authors both notebooks from plain-Python source strings. |

## Dominant-Peak Prominence (DPP)

The chapter's temporal-structure score, previously called *peakiness* /
*temporal specificity*. On each condition-average trace, divided by √(mean firing rate):

- **P₁** — largest topographic peak prominence
- **P₂** — largest prominence at least 250 ms away from P₁
- **DPP** = P₁ / (1 + λ·P₂/P₁), with λ = 0.5

A unit scores high only when one tall peak has no comparable rival. The stored
column is still `peakiness_score`; `thesis_common.DPP_COLUMN` is the single place
that mapping lives.

## Regenerating

All plotting and data assembly lives in `src/` (per `AGENTS.md`); the notebooks only
orchestrate and display. Edit `_build_notebooks.py` or the `src` modules, then:

```bash
conda run -n gaze_processing python notebooks/single_unit_thesis/_build_notebooks.py

conda run -n gaze_processing python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=3600 \
    notebooks/single_unit_thesis/single_unit_thesis_chapter_results.ipynb

conda run -n gaze_processing python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=3600 \
    notebooks/single_unit_thesis/single_unit_trace_metrics_appendix.ipynb
```

The main notebook reads the 1 ms spike-train store for the example-unit rasters. Only
the dates the example units come from are scanned, so a full run takes a couple of
minutes rather than a full sweep.

## Source modules

| Module | Contents |
|---|---|
| `ephys/plotting/thesis_common.py` | Figure style, region/condition vocabulary, DPP naming, analysis-window bars, significance brackets, dual PDF+PNG export. |
| `ephys/plotting/thesis_single_unit.py` | The six main-chapter figure panels. |
| `ephys/plotting/thesis_single_unit_data.py` | Table joins, exemplar resolution, trial loading for the example panels. |
| `ephys/plotting/thesis_trace_metrics.py` | Appendix metric panels. |
| `ephys/analysis/fixation_peakiness.py` | `decompose_dominant_peak_prominence` — recomputes a trace's DPP keeping peak positions and prominence reference levels, so the schematic draws the same P₁/P₂ the stored score used. |
| `core/stats/proportions.py` | Wilson intervals, two-proportion and goodness-of-fit tests, star annotation. |

## Figure conventions

- **One row of four region panels.** Extra rows only where a panel genuinely needs them.
- **Illustrator-editable PDF plus 400 dpi PNG** for every figure, written to
  `analysis_output_root/ephys/psth/single_unit_thesis/` (appendix figures in the
  `appendix/` subfolder). PDFs embed TrueType subsets (`pdf.fonttype=42`) and carry no
  clipping masks, so text stays editable and artwork ungrouped.
- **Analysis windows are horizontal bars** beneath each trace, not dotted background
  rectangles — the old rectangles spanned the panel height and read as data.
- **Fixation category is the only categorical colour**; region is encoded on the axis.
