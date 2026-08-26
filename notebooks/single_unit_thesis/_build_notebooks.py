"""Author the two thesis-chapter single-unit notebooks from source strings.

Per ``AGENTS.md`` the notebooks are thin: every function and class they use lives
in ``src/dal_monte_2022_analysis``. This script only assembles narrative and
call sites, so the long code stays diffable as plain Python.

    conda run -n gaze_processing python notebooks/single_unit_thesis/_build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

SETUP = '''
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

repo_root = Path.cwd()
if not (repo_root / "src").exists():
    repo_root = next(parent for parent in Path.cwd().parents if (parent / "src").exists())
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting import thesis_common as style
from dal_monte_2022_analysis.ephys.plotting import thesis_single_unit as figs
from dal_monte_2022_analysis.ephys.plotting import thesis_single_unit_data as data
from dal_monte_2022_analysis.ephys.plotting import thesis_metric_space as space
from dal_monte_2022_analysis.ephys.plotting import thesis_trace_metrics as metrics
from dal_monte_2022_analysis.ephys.analysis.fixation_peakiness import (
    decompose_dominant_peak_prominence,
)

DATASET_CFG_PATH = repo_root / "configs" / "dataset.yaml"
PLOTTING_CFG_PATH = repo_root / "configs" / "plotting.yaml"
PSTH_CFG_PATH = repo_root / "configs" / "ephys_fixation_psth.yaml"

dataset_cfg = load_config(DATASET_CFG_PATH)
plotting_cfg = load_config(PLOTTING_CFG_PATH)
psth_cfg = load_config(PSTH_CFG_PATH)

style.apply_thesis_plot_style(plotting_cfg)
pd.set_option("display.width", 190)
pd.set_option("display.max_columns", 60)

ANALYSIS_ROOT = data.resolve_psth_analysis_root(dataset_cfg)
FIGURE_SETTINGS = style.ThesisFigureSettings(output_dir=ANALYSIS_ROOT / "single_unit_thesis")
FIGURE_SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)

FIGURE_MANIFEST: dict[str, dict[str, Path]] = {}
'''

def _cell(kind: str, source: str) -> dict:
    lines = source.strip("\n").splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "gaze_processing",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# =========================================================================== #
# Notebook 1 - chapter results                                                #
# =========================================================================== #

MAIN: list[dict] = []


def md(source: str) -> None:
    MAIN.append(_cell("markdown", source))


def code(source: str) -> None:
    MAIN.append(_cell("code", source))


md(
    """
# Single-unit responses to social and non-social fixations

**Thesis chapter results — part 1.**

Neurons were recorded simultaneously from the basolateral amygdala (BLA) together
with one of three prefrontal regions — gyral anterior cingulate cortex (ACCg),
dorsomedial prefrontal cortex (dmPFC), or orbitofrontal cortex (OFC) — while two
macaques freely viewed one another. Every fixation falls into one of three
categories: **interactive face**, **non-interactive face**, or **object**.

The chapter argues in five steps:

| § | Claim |
|---|---|
| 2 | Individual neurons fire preferentially for interactive face fixations. |
| 3 | But units are not simply "interactive-face cells" — they separate every pair of categories. |
| 4 | Across the modulated population, interactive face is nevertheless the most frequently preferred category. |
| 5 | Interactive-face responses are held longer, in BLA and OFC. |
| 6 | Responses vary along two independent axes — duration and peak isolation — with no clean cell types. |

Every figure is one row of four region panels, saved as an Illustrator-editable
PDF (embedded TrueType, no clipping masks) plus a 400 dpi PNG. All plotting lives
in `src/dal_monte_2022_analysis/ephys/plotting/`; this notebook orchestrates and
displays only.
"""
)

md("## 1 · Setup")

code(SETUP.strip())

code(
    '''
units = data.load_thesis_unit_table(ANALYSIS_ROOT)
pair_selectivity = data.load_pair_selectivity(ANALYSIS_ROOT)
condition_traces = data.load_condition_traces(ANALYSIS_ROOT)
trace_shape = data.load_trace_shape_table(ANALYSIS_ROOT, units)
trace_shape_all = data.load_trace_shape_table(ANALYSIS_ROOT, units, condition="all")
matched_cv, matched_cv_stats, cv_inflation = data.load_trial_matched_cv(ANALYSIS_ROOT)

unit_plot_settings = data.build_unit_plot_settings(
    DATASET_CFG_PATH, PLOTTING_CFG_PATH, psth_cfg
)
peakiness_settings = data.build_peakiness_settings(DATASET_CFG_PATH, psth_cfg)

print(f"{len(units)} units | {units['is_selective'].sum()} fixation-category-modulated")
print(f"{units['date'].nunique()} sessions | figures -> {FIGURE_SETTINGS.output_dir}")
'''
)

md(
    """
## 2 · Neurons fire preferentially for interactive face fixations

Four example units, one per region, each significant on **both** interactive-face
contrasts (int vs non-int face, and int vs object) after FDR correction, and each
firing most for interactive face across the whole fixation window. Panel subtitles
give the mean rate for interactive / non-interactive / object.

Rasters are subsampled to 70 trials per category so individual spikes stay
visible; the firing-rate traces use all trials (mean ± SEM, 10 ms bins, 20 ms
Gaussian smoothing). The three grey bars mark the analysis windows used in §3.
"""
)

code(
    '''
INT_FACE_EXAMPLES = {"bla": "602", "accg": "118", "dmpfc": "1515", "ofc": "1038"}

example_units = data.build_exemplar_table(
    units, {region: {"int_face": uuid} for region, uuid in INT_FACE_EXAMPLES.items()}
)
window_means = pd.read_csv(
    ANALYSIS_ROOT / "fixation_psth_selectivity" / "condition_window_means.csv"
)
window_means = window_means.loc[window_means["window_name"] == "full_fix"]
example_units = example_units.merge(
    window_means.loc[
        :,
        ["unit_key", "mean_fr_face_interactive_hz", "mean_fr_face_non_interactive_hz",
         "mean_fr_object_hz"],
    ],
    on="unit_key",
    how="left",
)
example_units["subtitle"] = example_units.apply(
    lambda row: (
        f"{row.mean_fr_face_interactive_hz:.1f} vs "
        f"{row.mean_fr_face_non_interactive_hz:.1f} vs "
        f"{row.mean_fr_object_hz:.1f} Hz"
    ),
    axis=1,
)

example_specs = data.build_example_unit_panel_specs(
    example_units,
    style="int_face",
    unit_settings=unit_plot_settings,
    peakiness_settings=peakiness_settings,
    condition_traces=condition_traces,
    subtitle_column="subtitle",
)
fig = figs.plot_example_unit_panel(example_specs, show_window_legend=True)
FIGURE_MANIFEST["fig01_interactive_face_examples"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig01_interactive_face_examples"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(
    example_units.loc[
        :, ["region_label", "uuid", "date", "subtitle", "dominant_condition", "n_selective_pairs"]
    ]
)
'''
)

md(
    """
## 3 · Units are not simply "interactive-face cells"

A unit counts as **fixation-category-modulated** if its firing rate differs
significantly between at least one category pair in at least one of three 500 ms
windows (Welch's *t*-test on per-trial window means, FDR corrected across all
pair × window tests within the unit).
"""
)

code(
    '''
fig, yield_table = figs.plot_unit_yield_panel(units)
FIGURE_MANIFEST["fig02_unit_yield"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig02_unit_yield"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(yield_table.round(3))
'''
)

md(
    """
Which pairs a unit separates is shown below as an **UpSet plot** rather than a Venn.
Three-set Venns cannot in general be drawn area-proportionally — some combinations
of the seven subset sizes admit no valid circle geometry — and per-region area
scaling would give each of the four panels a different shape, defeating the
comparison. An UpSet reads the same information off an ordinary bar chart with an
explicit membership matrix beneath. Bars are the fraction of each region's
*selective* units so the regions are comparable despite differing yields.
"""
)

code(
    '''
pair_membership = figs.compute_pair_selectivity_membership(pair_selectivity)
fig, upset_counts = figs.plot_pair_selectivity_upset_panel(
    pair_membership,
    region_totals=units.groupby("region").size().to_dict(),
)
FIGURE_MANIFEST["fig03_pair_selectivity_upset"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig03_pair_selectivity_upset"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(upset_counts)
'''
)

md(
    """
## 4 · Interactive face is still the preferred category

For each modulated unit, which category evoked the highest mean rate over the full
−500 to +500 ms window. Bars carry Wilson 95% intervals and a binomial test against
chance (1/3), FDR corrected across the region × category family; the pie inset shows
the composition, which cannot carry uncertainty and so is secondary.
"""
)

code(
    '''
fig, preference_table = figs.plot_preferred_condition_panel(units.loc[units["is_selective"]])
FIGURE_MANIFEST["fig04_preferred_condition"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig04_preferred_condition"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(
    preference_table.loc[
        :, ["region_label", "condition", "k", "n", "fraction", "ci_low", "ci_high",
            "p_adj", "stars"]
    ].round(4)
)
'''
)

md(
    """
## 5 · Interactive-face responses are held longer

### Why not the coefficient of variation

The obvious statistic — the CV of the mean firing-rate trace across time bins — is
**not usable for comparing fixation categories here**. Interactive-face fixations are
about five times more numerous than the others (median 846 trials per unit versus
168 and 168). The number of time bins is identical in every condition, but that does
not protect the statistic: each bin of the mean trace is itself a trial average, so
writing `m(t) = s(t) + e(t)` gives

$$\\operatorname{Var}_t[m] \\approx \\operatorname{Var}_t[s] + \\mathbb{E}[\\sigma^2]/N$$

and a condition observed with fewer trials inherits extra across-bin variance from
estimation noise alone.

The control below equalises trial counts within each unit and recomputes every
condition's CV from 25 matched subsamples. **The apparent interactive-face advantage
does not survive**: raw CV differs by a factor of two, matched CV is
indistinguishable. The right panel shows the mechanism directly — subsampling
interactive-face trials, holding condition and neural signal fixed, inflates that
condition's own CV by up to 2.4×.
"""
)

code(
    '''
fig = space.plot_cv_trial_matched_control(matched_cv, cv_inflation)
FIGURE_MANIFEST["fig05_cv_trial_matched_control"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig05_cv_trial_matched_control"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(
    matched_cv_stats.loc[
        matched_cv_stats["condition_a"] == "face_interactive",
        ["region", "variant", "condition_b", "n_units_paired", "median_a", "median_b",
         "statistic", "p_value_adjusted", "significant_adjusted"],
    ].round(4)
)
'''
)

md(
    """
### Response duration, which is confound-free

**Response duration** — the full width at half maximum of the excess response — is a
width, not an amplitude, and is therefore near-insensitive to trial count
(Spearman ρ with log trial count = +0.12, against −0.76 for any prominence-based
score). Compared within unit across categories, it gives the defensible version of
the stability claim.
"""
)

code(
    '''
fig, duration_table = figs.plot_condition_metric_panel(
    trace_shape_all.loc[trace_shape_all["is_selective"]],
    metric="response_duration_ms",
    metric_label=space.DURATION_LABEL,
)
FIGURE_MANIFEST["fig06_response_duration"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig06_response_duration"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(
    duration_table.loc[
        :, ["region_label", "condition_b", "n_units", "median_a", "median_b",
            "statistic", "p_adj", "stars"]
    ].round(3)
)
'''
)

md(
    """
## 6 · Two axes describe the response, and there are no cell types

Each unit's response is summarised by two quantities measured on the trace of its
preferred category:

- **Response duration** — FWHM of the excess response (ms).
- **Peak isolation** — $1 - P_2/P_1$, where $P_1$ is the dominant peak's topographic
  prominence and $P_2$ the largest prominence at least 250 ms away. 1 means the peak
  has no rival; 0 means an equally strong second peak exists.

These replace the single composite score used previously. That composite,
$P_1/(1+\\lambda P_2/P_1)$, correlates **0.99 with $P_1$ alone** — the discount term
spans too little range to matter — so it measured peak *height* while being
described as measuring isolation, and peak height is exactly the quantity trial
count corrupts. Duration and isolation are near-independent (ρ ≈ −0.26) and both
are trial-count insensitive (ρ = +0.12 and −0.001).
"""
)

code(
    '''
schematic_row = condition_traces.loc[
    (condition_traces["unit_uuid"] == "unit_uuid__600")
    & (condition_traces["condition"] == "face_interactive")
].iloc[0]
schematic_decomposition = decompose_dominant_peak_prominence(
    np.asarray(schematic_row["trace_hz"], dtype=float),
    np.asarray(schematic_row["bin_centers_s_rel"], dtype=float),
    peakiness_settings,
)
schematic_duration = float(
    trace_shape.loc[trace_shape["uuid"] == "600", "response_duration_ms"].iloc[0]
)
fig = space.plot_isolation_schematic(
    schematic_decomposition, unit_label="BLA unit 600", duration_ms=schematic_duration
)
FIGURE_MANIFEST["fig07_metric_schematic"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig07_metric_schematic"
)
display(Image(data=style.figure_to_png_bytes(fig)))
'''
)

md(
    """
Plotted against each other, the modulated population forms a **single unimodal
cloud** in every region. There is no peaky/sustained dichotomy to threshold — which
is why the chapter treats these as descriptive axes rather than a taxonomy. Naming
the corners is still useful, and the units that occupy them are shown below.
"""
)

code(
    '''
corner_units = pd.concat(
    [space.select_corner_units(trace_shape, region=region) for region in style.REGION_ORDER],
    ignore_index=True,
)
fig, metric_space_summary = space.plot_metric_space_panel(trace_shape, corners=corner_units)
FIGURE_MANIFEST["fig08_metric_space"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig08_metric_space"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(metric_space_summary.round(3))
'''
)

code(
    '''
CORNER_REGION = "bla"
fig = space.plot_corner_example_traces(
    corner_units.loc[corner_units["region"] == CORNER_REGION], condition_traces
)
FIGURE_MANIFEST["fig09_corner_examples"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig09_corner_examples"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(
    corner_units.loc[
        :, ["region", "corner", "uuid", "date", "response_duration_ms", "peak_isolation",
            "mean_fr_hz"]
    ].round(2)
)
'''
)

md("## 7 · Persist tables and figure manifest")

code(
    '''
exports = {
    "unit_yield_by_region.csv": yield_table,
    "pair_selectivity_upset_counts.csv": upset_counts,
    "preferred_condition_by_region.csv": preference_table,
    "response_duration_by_condition.csv": duration_table,
    "cv_trial_matched_stats.csv": matched_cv_stats,
    "metric_space_summary.csv": metric_space_summary,
    "metric_space_corner_units.csv": corner_units.loc[
        :, ["region", "corner", "uuid", "date", "response_duration_ms", "peak_isolation"]
    ],
    "interactive_face_examples.csv": example_units.loc[
        :, ["region", "uuid", "date", "subtitle", "dominant_condition"]
    ],
}
for filename, frame in exports.items():
    frame.to_csv(FIGURE_SETTINGS.output_dir / filename, index=False)
    print(f"wrote {filename:40s} ({len(frame)} rows)")

print()
for stem in FIGURE_MANIFEST:
    print(stem)
'''
)

md(
    """
## Summary

1. **Individual neurons prefer interactive face fixations** (§2). The four example
   units each separate interactive face from both other categories and fire most for
   it throughout the fixation.
2. **But they are not category-specific detectors** (§3). Most modulated units
   separate two or three of the three pairs; the single-pair combinations are the
   minority everywhere.
3. **Interactive face is the modal preference** (§4), significantly above chance in
   BLA and OFC and numerically leading in ACCg and dmPFC.
4. **Interactive-face responses last longer** (§5) — BLA 350 vs 300/310 ms, OFC 350
   vs 290 ms. The coefficient of variation, which appears to show a far larger
   effect, is not usable: the difference is created by interactive face having ~5×
   more trials, and it vanishes under trial matching.
5. **Response shape is continuous, not categorical** (§6). Duration and peak
   isolation are independent and both unimodal; naming corners of that space is
   descriptive convenience, not a cell-type claim.

The appendix (`single_unit_trace_metrics_appendix.ipynb`) reports the wider set of
trace metrics, which of them are redundant, and how each relates to these two.
"""
)


# =========================================================================== #
# Notebook 2 - appendix                                                       #
# =========================================================================== #

APPENDIX: list[dict] = []


def amd(source: str) -> None:
    APPENDIX.append(_cell("markdown", source))


def acode(source: str) -> None:
    APPENDIX.append(_cell("code", source))


amd(
    """
# Appendix · Quantifying the shape of mean firing-rate traces

The chapter summarises each unit's temporal structure with a single number,
**Dominant-Peak Prominence (DPP)**. That is a deliberate simplification: DPP asks
only *"is there one tall peak with no comparable rival?"* and is silent about how
**wide** the response is and how **ragged** the trace is between peaks.

This appendix reports the fuller set of trace statistics, shows their distributions,
and establishes what DPP does and does not capture.

**Scope.** All panels use the **fixation-category-modulated** subpopulation only. A
width or raggedness statistic computed on an unmodulated trace describes the noise
floor rather than a response, so including those units would blur every distribution
towards the same shape.

Every figure is a single embeddable panel (one row of four regions), written as an
Illustrator-editable PDF and a 400 dpi PNG.
"""
)

amd("## A1 · Setup")

acode(SETUP.strip())

acode(
    '''
units = data.load_thesis_unit_table(ANALYSIS_ROOT)
selective = data.load_trace_metric_table(ANALYSIS_ROOT, units, selective_only=True)

exemplars = data.build_exemplar_table(units, data.parse_config_exemplar_map(psth_cfg))
exemplar_metrics = data.load_trace_metric_table(
    ANALYSIS_ROOT, exemplars, selective_only=False
)

FIGURE_SETTINGS = style.ThesisFigureSettings(
    output_dir=ANALYSIS_ROOT / "single_unit_thesis" / "appendix"
)
FIGURE_SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)

print(f"{len(selective)} fixation-category-modulated units with trace metrics")
print(f"figures -> {FIGURE_SETTINGS.output_dir}")
'''
)

amd(
    """
## A2 · The metrics

Each metric is computed on the condition-average firing-rate trace over −500 to
+500 ms, with the baseline taken as the 10th percentile of the in-window trace. They
are grouped by the property they measure, so it is clear which are alternative
estimates of the same thing and which are genuinely independent.
"""
)

acode(
    '''
METRIC_ORDER = [
    "mass_width_frac_50",
    "effective_width_ms",
    "lifetime_sparseness",
    "peak_dominance",
    "n_prominent_peaks",
    "fwhm_frac",
    "sustained_frac",
    "roughness",
    "autocorr_width_ms",
    "peak_z",
]
definitions = metrics.build_metric_definition_table(METRIC_ORDER)
display(definitions)
'''
)

amd(
    """
## A3 · Distribution of each metric

One panel per metric, four regions per panel, with the chapter's high- and low-DPP
example units marked. Bin edges are shared across regions so the panel shapes can be
compared directly; the axis is trimmed at the 99.5th percentile because several
metrics have a thin right tail that would otherwise flatten the body.
"""
)

acode(
    '''
FIGURE_MANIFEST = {}
for index, metric in enumerate(METRIC_ORDER, start=1):
    fig = metrics.plot_metric_distribution_panel(
        selective, metric, exemplars=exemplar_metrics
    )
    stem = f"figA{index:02d}_{metric}"
    FIGURE_MANIFEST[stem] = style.save_thesis_figure(fig, FIGURE_SETTINGS, stem)
    display(Markdown(f"**{metrics.metric_axis_label(metric)}**"))
    display(Image(data=style.figure_to_png_bytes(fig)))
'''
)

acode(
    '''
metric_summary = metrics.build_metric_summary_table(selective, metrics=METRIC_ORDER)
display(metric_summary.round(3))
'''
)

amd(
    """
## A4 · Which metrics are redundant

Several of these measure the same underlying property by different routes. The
rank-correlation matrix below identifies those groups, so only one member of each
needs to be reported in the chapter.
"""
)

acode(
    '''
CORRELATION_METRICS = METRIC_ORDER + [style.DPP_COLUMN]
fig, correlation = metrics.plot_metric_correlation_panel(
    selective, metrics=CORRELATION_METRICS
)
FIGURE_MANIFEST["figA11_metric_correlation"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "figA11_metric_correlation"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(correlation.round(2))
'''
)

amd(
    """
## A5 · What DPP does and does not capture

Each panel plots one trace metric against DPP within region. A metric that tracks DPP
closely is redundant with it; one that is orthogonal describes a property of the same
trace that the chapter's single score discards.
"""
)

acode(
    '''
RELATIONSHIP_METRICS = ["peak_dominance", "fwhm_frac", "roughness", "peak_z"]
relationship_tables = []
for index, metric in enumerate(RELATIONSHIP_METRICS, start=12):
    fig, table = metrics.plot_metric_vs_dpp_panel(
        selective, metric, exemplars=exemplar_metrics
    )
    stem = f"figA{index:02d}_{metric}_vs_dpp"
    FIGURE_MANIFEST[stem] = style.save_thesis_figure(fig, FIGURE_SETTINGS, stem)
    display(Markdown(f"**{metrics.metric_axis_label(metric)} vs {style.DPP_ABBREV}**"))
    display(Image(data=style.figure_to_png_bytes(fig)))
    relationship_tables.append(table)

relationships = pd.concat(relationship_tables, ignore_index=True)
display(relationships.round(4))
'''
)

amd(
    """
## A6 · Where the example units fall on every metric

Percentile rank within each unit's own region. This is the audit of the chapter's
example selection: a high-DPP example should sit high on peak dominance and low on
the width measures, and a low-DPP example should do the opposite.
"""
)

acode(
    '''
rank_table = metrics.build_exemplar_metric_rank_table(
    selective, exemplar_metrics, metrics=RELATIONSHIP_METRICS + ["mass_width_frac_50"]
)
display(rank_table.round(2))
'''
)

amd("## A7 · Persist tables and figure manifest")

acode(
    '''
exports = {
    "trace_metric_definitions.csv": definitions,
    "trace_metric_summary_by_region.csv": metric_summary,
    "trace_metric_correlation.csv": correlation.reset_index(names="metric"),
    "trace_metric_vs_dpp.csv": relationships,
    "exemplar_metric_percentiles.csv": rank_table,
}
for filename, frame in exports.items():
    frame.to_csv(FIGURE_SETTINGS.output_dir / filename, index=False)
    print(f"wrote {filename:40s} ({len(frame)} rows)")

print()
for stem in FIGURE_MANIFEST:
    print(stem)
'''
)

amd(
    """
## Appendix summary

- The metrics fall into a small number of correlated groups (A4): width measures
  (`mass_width_frac_50`, `effective_width_ms`, `fwhm_frac`, `sustained_frac`,
  `lifetime_sparseness`) are largely interchangeable, as are the two composites.
  `roughness` and `peak_dominance` carry information the width measures do not.
- **DPP measures peak isolation, not response width** (A5). Its strongest
  association is with `peak_dominance` (rho approximately 0.3) and its next with
  `roughness` and the prominent-peak count (both approximately -0.3); it is close to
  independent of every width measure (|rho| < 0.15 against `mass_width_frac_50`,
  `effective_width_ms`, `fwhm_frac` and `sustained_frac`). The chapter's single score
  should therefore not be described as measuring how *narrow* a response is - it says
  a peak stands alone, not that it is brief. Note also that no single metric explains
  DPP: even the strongest correlation is modest, because DPP combines peak height and
  rival suppression in a way none of the individual statistics reproduces.
- The high-DPP example units rank at the top of their region on peak dominance and at
  the bottom on the width measures, as their label claims (A6). Among the low-DPP
  examples, **dmPFC 1516 does not**: it ranks high on peak dominance and low on FWHM,
  behaving like a high-DPP unit on the shape metrics even though its DPP score is
  only mid-range. Its high `roughness` is what holds its DPP down.
"""
)


def main() -> None:
    directory = Path(__file__).resolve().parent
    targets = {
        "single_unit_thesis_chapter_results.ipynb": MAIN,
        "single_unit_trace_metrics_appendix.ipynb": APPENDIX,
    }
    for filename, cells in targets.items():
        path = directory / filename
        path.write_text(
            json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
