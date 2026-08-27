# notebooks/single_unit_thesis

Thesis-chapter notebooks for the single-unit fixation analyses. The exploratory
notebook in [`notebooks/single_unit/`](../single_unit) is kept as-is; these two are the
streamlined, figure-first versions written for the chapter itself.

| File | Role |
|---|---|
| `single_unit_thesis_chapter_results.ipynb` | Main chapter results, in narrative order: interactive-face example units → preferred-category distribution → UpSet of fixation-pair selectivity → the two metrics and their schematic → the halfwidth × isolation space with example traces inset → halfwidth by category (supplementary) and the trial-matched CV control. |
| `single_unit_trace_metrics_appendix.ipynb` | Appendix: the full set of mean firing-rate trace metrics, their distributions across the modulated subpopulation, which are redundant, and what DPP does and does not capture. |
| `_build_notebooks.py` | Authors both notebooks from plain-Python source strings. |

## The two trace metrics

Each unit's response is described by two independent quantities measured on the
trace of its preferred fixation category:

| Metric | Definition | Meaning |
|---|---|---|
| **Dominant-peak width** | FWHM of the excess response (ms) | how narrow or wide the dominant peak is |
| **Dominant-peak prominence** | `1 - P₂/P₁` | 1 = a single clear peak; 0 = a second peak of equal prominence |

Corners of that space are labelled **narrow / wide** × **single-peak / multi-peak** —
the plain-language poles of the two axes. The prominence poles avoid re-using
"dominant", which is already in the metric's own name.

The stored columns keep their original names (`response_duration_ms`,
`peak_isolation`); `thesis_metric_space` holds the display labels, with
`HALFWIDTH_LABEL` / `ISOLATION_LABEL` kept as aliases.

where **P₁** is the largest topographic peak prominence of the √-rate-normalised
trace and **P₂** the largest prominence at least 250 ms away.

These replace the earlier composite `peakiness_score` = `P₁ / (1 + λ·P₂/P₁)`, which
should **not** be used for comparisons across fixation categories. It correlates
**0.99 with P₁ alone** — the discount term spans too little range to matter — so it
measured peak height while being described as measuring isolation, and peak height
is precisely the quantity trial count corrupts (ρ = −0.80 with log trial count,
against −0.001 for peak isolation and +0.12 for duration).

## The trial-count confound

Interactive-face fixations are about five times more numerous than the other
categories (median 846 trials per unit versus 168 and 168). The coefficient of
variation of the mean trace across time bins looks dramatically lower for
interactive face, **but that is an artefact**: each bin is itself a trial average,
so `Var_t[m] ≈ Var_t[s] + E[σ²]/N` and a condition seen with fewer trials inherits
extra across-bin variance from estimation noise. Subsampling interactive-face
trials while holding condition and signal fixed inflates its own CV up to 2.4×.

`fixation_cv_trial_matched` recomputes every condition's CV from equal-sized trial
subsamples. The effect does not survive. The chapter therefore uses **response
duration**, which is a width rather than an amplitude and is confound-free.

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
| `ephys/plotting/thesis_single_unit.py` | Unit yield, UpSet, example-unit rows, preferred category, per-condition metric comparison. |
| `ephys/plotting/thesis_metric_space.py` | Metric schematic, width × prominence space with inset traces, CV control. |
| `ephys/plotting/thesis_chapter_text.py` | Builds the chapter's numbers-and-methods block from the persisted tables. |
| `ephys/plotting/thesis_single_unit_data.py` | Table joins, exemplar resolution, trial loading for the example panels. |
| `ephys/plotting/thesis_trace_metrics.py` | Appendix metric panels. |
| `ephys/analysis/fixation_peakiness.py` | `decompose_dominant_peak_prominence` — recomputes a trace's P₁/P₂ keeping peak positions and prominence reference levels, so the schematic draws exactly what the stored numbers describe. |
| `ephys/analysis/fixation_cv_trial_matched.py` | Trial-count-matched CV control and the inflation curve. |
| `core/stats/proportions.py` | Wilson intervals, two-proportion and goodness-of-fit tests, star annotation. |

## Figure conventions

- **One row of four region panels.** Extra rows only where a panel genuinely needs them.
- **Illustrator-editable PDF plus 400 dpi PNG** for every figure, written to
  `analysis_output_root/ephys/psth/single_unit_thesis/` (appendix figures in the
  `appendix/` subfolder). PDFs embed TrueType subsets (`pdf.fonttype=42`) and carry no
  clipping masks, so text stays editable and artwork ungrouped.
- **Analysis windows are horizontal bars** beneath each trace, not dotted background
  rectangles — the old rectangles spanned the panel height and read as data.
- **UpSet, not Venn**, for set intersections: three-set Venns cannot in general be
  drawn area-proportionally, and per-region scaling would break cross-region
  comparison. Region is the series there and uses the prominent colourblind-safe
  `REGION_COLORS` palette; unit counts are printed on each bar. Sized to work both
  standalone and as one panel row of a larger paper figure.
- **Violins follow the behavioural convention** — seaborn, `inner="quart"`,
  `cut=0`, so quartile lines sit inside the body and the kernel is truncated at
  the observed range.
- **Fixation category is the only categorical colour**; region is encoded on the axis.
