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
from dal_monte_2022_analysis.ephys.plotting import thesis_trace_metrics as metrics

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
macaques freely viewed one another.

Every fixation is assigned to one of three categories:

| Category | Definition |
|---|---|
| **Interactive face** | fixation on the partner's face during a detected interactive period |
| **Non-interactive face** | fixation on the partner's face outside interactive periods |
| **Object** | fixation on a non-social object |

The chapter asks four questions in order:

1. **How many neurons care about fixation category at all?** (§2, §3)
2. **What does an individual response look like, and how do responses differ in
   their temporal structure?** (§4–§6)
3. **Is the response to interactive face fixations more temporally reliable?** (§7)
4. **Which category do neurons actually prefer?** (§8)

Every figure is one row of four region panels and is written to disk as an
Illustrator-editable PDF (embedded TrueType text, no clipping masks) alongside a
400 dpi PNG. All plotting functions live in
`src/dal_monte_2022_analysis/ephys/plotting/thesis_single_unit.py`; this notebook
only orchestrates and displays.
"""
)

md("## 1 · Setup")

code(SETUP.strip())

code(
    '''
units = data.load_thesis_unit_table(ANALYSIS_ROOT)
pair_selectivity = data.load_pair_selectivity(ANALYSIS_ROOT)
variability, variability_stats = data.load_condition_variability(ANALYSIS_ROOT)
condition_traces = data.load_condition_traces(ANALYSIS_ROOT)

unit_plot_settings = data.build_unit_plot_settings(
    DATASET_CFG_PATH, PLOTTING_CFG_PATH, psth_cfg
)
peakiness_settings = data.build_peakiness_settings(DATASET_CFG_PATH, psth_cfg)

print(f"{len(units)} units | {units['is_selective'].sum()} fixation-category-modulated")
print(f"{units['date'].nunique()} sessions | regions: {sorted(units['region'].unique())}")
print(f"figures -> {FIGURE_SETTINGS.output_dir}")
'''
)

md(
    """
## 2 · How many neurons differentiate fixation categories?

A unit is called **fixation-category-modulated** if its firing rate differs
significantly between at least one of the three category pairs in at least one of
three 500 ms analysis windows — pre-fixation (−500 to 0 ms), peri-fixation (−250 to
+250 ms) and post-fixation (0 to +500 ms). Significance is a Welch's *t*-test on
per-trial window-mean rates, Benjamini–Hochberg FDR corrected across all
pair × window tests within a unit.
"""
)

code(
    '''
fig, yield_table = figs.plot_unit_yield_panel(units)
FIGURE_MANIFEST["fig01_unit_yield"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig01_unit_yield"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(yield_table.round(3))
'''
)

md(
    """
## 3 · Which category contrasts carry the effect

A modulated unit can separate one, two or all three category pairs. The Venn
diagrams below partition each region's modulated units by which pairs they
separate; circle geometry is fixed across regions so the panels are directly
comparable (an area-scaled Venn would give each region a different shape).
"""
)

code(
    '''
pair_membership = figs.compute_pair_selectivity_membership(pair_selectivity)
fig, venn_counts = figs.plot_pair_selectivity_venn_panel(
    pair_membership,
    region_totals=units.groupby("region").size().to_dict(),
)
FIGURE_MANIFEST["fig02_pair_selectivity_venn"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig02_pair_selectivity_venn"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(venn_counts)
'''
)

md(
    """
## 4 · Quantifying the temporal structure of a response — Dominant-Peak Prominence

Individual responses differ in whether the firing-rate change is concentrated into
a single transient or spread across the fixation. To quantify this on one axis, each
unit's condition-average trace is scored by its **Dominant-Peak Prominence (DPP)**:

1. The trace is divided by √(mean firing rate). This variance-stabilising step makes
   the score comparable between a 2 Hz and a 30 Hz unit; without it DPP would mostly
   report firing rate.
2. All local peaks at least 30 ms apart are detected, and each is assigned its
   topographic **prominence** — the height it rises above the highest saddle
   separating it from any taller peak.
3. **P₁** is the largest prominence. **P₂** is the largest prominence at least
   250 ms away from P₁, i.e. the strongest genuinely separate rival.
4. The score discounts P₁ by how close its rival comes:

$$\\mathrm{DPP} \;=\; \\frac{P_1}{1 + \\lambda\\, P_2 / P_1}, \\qquad \\lambda = 0.5$$

A unit scores high only if it has **one tall peak with no comparable competitor**.
A trace with several similar peaks is penalised even when one of them is tall, and a
trace with no peak at all scores near zero.

The BLA panel below draws the construction on that unit's interactive-face trace.
"""
)

code(
    '''
exemplar_map = data.parse_config_exemplar_map(psth_cfg)
exemplars = data.build_exemplar_table(units, exemplar_map)
display(
    exemplars.loc[
        :, ["style", "region_label", "uuid", "date", style.DPP_COLUMN, "dpp_percentile"]
    ]
    .rename(columns={style.DPP_COLUMN: "DPP", "dpp_percentile": "within-region percentile"})
    .round(3)
)
'''
)

md("### 4a · High-DPP example units")

code(
    '''
high_specs = data.build_example_unit_panel_specs(
    exemplars,
    style="phasic",
    unit_settings=unit_plot_settings,
    peakiness_settings=peakiness_settings,
    condition_traces=condition_traces,
    schematic_region="bla",
)
fig = figs.plot_example_unit_panel(
    high_specs, schematic_index=0, show_window_legend=True
)
FIGURE_MANIFEST["fig03_high_dpp_examples"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig03_high_dpp_examples"
)
display(Image(data=style.figure_to_png_bytes(fig)))
'''
)

md(
    """
Rasters are subsampled to 70 trials per category so individual spikes stay visible;
at the full several-hundred trials every tick overlaps its neighbours and the raster
prints as a solid block. Firing-rate traces use all trials (mean ± SEM, 10 ms bins,
20 ms Gaussian smoothing).

The three grey bars below each trace mark the analysis windows used in §2 — they
replace the dotted background rectangles used previously, which spanned the full
panel height and read as data.
"""
)

md("### 4b · Low-DPP example units")

code(
    '''
low_specs = data.build_example_unit_panel_specs(
    exemplars,
    style="tonic",
    unit_settings=unit_plot_settings,
    peakiness_settings=peakiness_settings,
    condition_traces=condition_traces,
)
fig = figs.plot_example_unit_panel(low_specs, show_window_legend=True)
FIGURE_MANIFEST["fig04_low_dpp_examples"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig04_low_dpp_examples"
)
display(Image(data=style.figure_to_png_bytes(fig)))
'''
)

md(
    """
## 5 · Where the example units sit in the population

The example units are not extremes chosen to flatter the score — the panel below
places each of them inside its own region's DPP distribution, with its percentile
printed.
"""
)

code(
    '''
fig, dpp_summary = figs.plot_dpp_distribution_panel(units, exemplars)
FIGURE_MANIFEST["fig05_dpp_distribution"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig05_dpp_distribution"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(dpp_summary.round(3))
'''
)

md(
    """
### 5a · Metric-selected alternates

Two of the configured low-DPP examples do not in fact sit low in their region:
ACCg 118 is near the region median and dmPFC 1516 is above it. The panel below shows
the units the score itself picks out — highest- and lowest-DPP modulated unit per
region, restricted to a firing-rate range that stays plottable — as candidate
replacements. The main-text figures above keep the originally configured units.
"""
)

code(
    '''
alternates = data.select_metric_driven_exemplars(units)
display(
    alternates.loc[
        :, ["style", "region_label", "uuid", "date", style.DPP_COLUMN, "dpp_percentile",
            "face_interactive_mean_fr_hz"]
    ]
    .rename(columns={style.DPP_COLUMN: "DPP", "dpp_percentile": "percentile"})
    .round(3)
)

fig, _ = figs.plot_dpp_distribution_panel(units, alternates)
FIGURE_MANIFEST["fig05b_dpp_distribution_alternates"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig05b_dpp_distribution_alternates"
)
display(Image(data=style.figure_to_png_bytes(fig)))
'''
)

md(
    """
## 6 · Interactive-face responses are less variable over time

If interactive face fixations drive a more consistent response, the unit's mean
firing-rate trace should fluctuate *less* over the fixation for that category. The
coefficient of variation (SD ÷ mean of the trace across time bins) tests this
directly, and being a ratio it is not confounded by the higher firing rates
interactive face fixations often evoke.

Comparisons are paired within unit (each unit contributes all three categories),
Benjamini–Hochberg corrected within region. Brackets are drawn from the stored test
table rather than recomputed, so figure and statistics cannot drift apart.
"""
)

code(
    '''
fig, cv_summary = figs.plot_condition_cv_panel(variability, variability_stats)
FIGURE_MANIFEST["fig06_condition_cv"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig06_condition_cv"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(cv_summary.round(4))

display(
    variability_stats.loc[variability_stats["metric_key"] == "cv"]
    .loc[:, ["region", "condition_pair", "n_units_paired", "statistic", "p_value_adjusted",
             "significant_adjusted"]]
    .round(4)
)
'''
)

md(
    """
## 7 · Interactive face is still the preferred category

Lower temporal variability is not the same as lower firing. The panel below counts,
for each modulated unit, which category evoked the highest mean rate over the full
−500 to +500 ms window.
"""
)

code(
    '''
fig, preference_table = figs.plot_preferred_condition_panel(units.loc[units["is_selective"]])
FIGURE_MANIFEST["fig07_preferred_condition"] = style.save_thesis_figure(
    fig, FIGURE_SETTINGS, "fig07_preferred_condition"
)
display(Image(data=style.figure_to_png_bytes(fig)))
display(preference_table.round(3))
'''
)

md("## 8 · Persist tables and figure manifest")

code(
    '''
exports = {
    "unit_yield_by_region.csv": yield_table,
    "pair_selectivity_venn_counts.csv": venn_counts,
    "dpp_distribution_summary.csv": dpp_summary,
    "dpp_exemplars_configured.csv": exemplars.loc[
        :, ["style", "region", "uuid", "date", style.DPP_COLUMN, "dpp_percentile"]
    ],
    "dpp_exemplars_metric_selected.csv": alternates.loc[
        :, ["style", "region", "uuid", "date", style.DPP_COLUMN, "dpp_percentile"]
    ],
    "condition_cv_summary.csv": cv_summary,
    "preferred_condition_by_region.csv": preference_table,
}
for filename, frame in exports.items():
    frame.to_csv(FIGURE_SETTINGS.output_dir / filename, index=False)
    print(f"wrote {filename:44s} ({len(frame)} rows)")

print()
for stem, paths in FIGURE_MANIFEST.items():
    print(f"{stem:38s} " + ", ".join(f"{ext}" for ext in sorted(paths)))
'''
)

md(
    """
## Summary

1. **Yield.** A substantial minority of neurons in every region differentiate at
   least one fixation-category pair, and the fraction is not uniform: BLA is the most
   responsive, dmPFC the least (§2).
2. **Contrast.** Comparisons involving interactive face fixations dominate. Very few
   units separate non-interactive face from object without also separating one of the
   interactive-face contrasts (§3).
3. **Temporal structure.** Dominant-Peak Prominence separates units with a single
   isolated transient from units whose modulation is spread across the fixation
   (§4–§5). The configured high-DPP examples all sit in the top decile of their
   region; two of the four low-DPP examples do not sit low, and metric-selected
   replacements are given in §5a.
4. **Reliability.** The coefficient of variation of the mean firing-rate trace is
   significantly lower for interactive face fixations than for either other category,
   in every region (§6).
5. **Preference.** That greater temporal reliability does not come at the cost of
   drive: interactive face is the most frequently preferred category among modulated
   units in every region (§7).

Appendix quantification of trace shape — the metrics behind the DPP score, and what
each one adds — is in `single_unit_trace_metrics_appendix.ipynb`.
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
