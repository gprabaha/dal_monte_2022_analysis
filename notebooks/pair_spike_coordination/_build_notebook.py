"""Author the neural pair spike-coordination notebook from source strings.

Per ``AGENTS.md`` the notebook is thin: every function it calls lives in
``src/dal_monte_2022_analysis``.  This script assembles narrative and call sites
only, so the analysis cannot drift into the notebook.

    conda run -n gaze_processing python notebooks/pair_spike_coordination/_build_notebook.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

FILENAME = "pair_spike_coordination.ipynb"
TITLE = "Spike coordination in simultaneously recorded neural pairs"


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
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.ephys.analysis import fixation_pair_spike_coordination as psc
from dal_monte_2022_analysis.ephys.plotting import fixation_pair_spike_coordination as viz

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

DATASET_CFG_PATH = repo_root / "configs" / "dataset.yaml"
COORDINATION_CFG_PATH = repo_root / "configs" / "ephys_fixation_pair_spike_coordination.yaml"
cfg = load_config(str(DATASET_CFG_PATH))

SUMMARY_DIR = build_analysis_output_dir(cfg, psc.DEFAULT_OUTPUT_SUBDIR) / "summary"
FIGURE_DIR = build_analysis_output_dir(cfg, psc.DEFAULT_OUTPUT_SUBDIR) / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
figs = viz.PairCoordinationPlotSettings(output_dir=FIGURE_DIR)

MIN_PAIRS = psc.DEFAULT_MIN_PAIRS_FOR_REPORTING

settings = psc.build_pair_spike_coordination_settings_from_config(
    dataset_cfg_path=str(DATASET_CFG_PATH),
    coordination_cfg_path=str(COORDINATION_CFG_PATH),
)
print("summary dir:", SUMMARY_DIR)
'''

BUILD = '''
# Submission is opt-in: the array costs real cluster time and a notebook cell is
# easy to re-run by accident. False only reports what is missing.
SUBMIT_JOBS = False
REBUILD_SUMMARIES = True

per_date = psc.ensure_pair_coordination_built(
    settings, submit=SUBMIT_JOBS, wait=True,
    sbatch_path=repo_root / psc.DEFAULT_SBATCH_PATH, repo_root=repo_root,
)
if REBUILD_SUMMARIES:
    _ = psc.run_summary_build(settings)
'''

LOAD = '''
traces = pd.read_pickle(SUMMARY_DIR / "traces_by_region.pkl")
traces_selective = pd.read_pickle(SUMMARY_DIR / "traces_by_region_selective.pkl")

inventory = pd.read_csv(SUMMARY_DIR / "region_pair_inventory.csv")
display(inventory)

dropped = pd.read_csv(SUMMARY_DIR / "dropped_artifact_dates.csv")
if len(dropped):
    display(Markdown(
        f"**{len(dropped)} date(s) removed for the zero-lag artifact:** "
        f"{sorted(dropped['date'].astype(str))}"
    ))
else:
    display(Markdown("_No date was flagged for the zero-lag artifact._"))
'''


def within(measure: str = "norm") -> str:
    return f'''
fig, paths = viz.plot_observed_and_null_grid(
    traces, figs, scope="within_region", measure="{measure}", min_pairs=MIN_PAIRS
)
display(Image(filename=str(paths["png"])))
'''


WITHIN_OBSERVED_SELECTIVE = '''
fig, paths = viz.plot_observed_and_null_grid(
    traces_selective, figs, scope="within_region", min_pairs=MIN_PAIRS,
    label="FDR-selective pairs",
)
display(Image(filename=str(paths["png"])))
'''

WITHIN_CORRECTED = '''
fig, paths = viz.plot_null_corrected_grid(
    traces, figs, scope="within_region", min_pairs=MIN_PAIRS
)
display(Image(filename=str(paths["png"])))

contrasts = pd.read_csv(SUMMARY_DIR / "condition_comparisons_by_region.csv")
display(contrasts.loc[contrasts["scope"] == "within_region"].round(4))

fig, paths = viz.plot_condition_contrasts(contrasts, figs, scope="within_region")
display(Image(filename=str(paths["png"])))
'''

WITHIN_CORRECTED_SELECTIVE = '''
fig, paths = viz.plot_null_corrected_grid(
    traces_selective, figs, scope="within_region", min_pairs=MIN_PAIRS,
    label="FDR-selective pairs",
)
display(Image(filename=str(paths["png"])))

contrasts_selective = pd.read_csv(
    SUMMARY_DIR / "condition_comparisons_by_region_selective.csv"
)
display(contrasts_selective.loc[contrasts_selective["scope"] == "within_region"].round(4))

fig, paths = viz.plot_condition_contrasts(
    contrasts_selective, figs, scope="within_region", label="FDR-selective pairs"
)
display(Image(filename=str(paths["png"])))
'''

CROSS_OBSERVED = '''
fig, paths = viz.plot_observed_and_null_grid(
    traces, figs, scope="cross_region", min_pairs=MIN_PAIRS
)
display(Image(filename=str(paths["png"])))

fig, paths = viz.plot_observed_and_null_grid(
    traces_selective, figs, scope="cross_region", min_pairs=MIN_PAIRS,
    label="FDR-selective pairs",
)
display(Image(filename=str(paths["png"])))
'''

CROSS_CORRECTED = '''
fig, paths = viz.plot_null_corrected_grid(
    traces, figs, scope="cross_region", min_pairs=MIN_PAIRS
)
display(Image(filename=str(paths["png"])))

display(contrasts.loc[contrasts["scope"] == "cross_region"].round(4))
fig, paths = viz.plot_condition_contrasts(contrasts, figs, scope="cross_region")
display(Image(filename=str(paths["png"])))
'''

CROSS_CORRECTED_SELECTIVE = '''
fig, paths = viz.plot_null_corrected_grid(
    traces_selective, figs, scope="cross_region", min_pairs=MIN_PAIRS,
    label="FDR-selective pairs",
)
display(Image(filename=str(paths["png"])))

display(contrasts_selective.loc[contrasts_selective["scope"] == "cross_region"].round(4))
fig, paths = viz.plot_condition_contrasts(
    contrasts_selective, figs, scope="cross_region", label="FDR-selective pairs"
)
display(Image(filename=str(paths["png"])))
'''

SECOND_MEASURE = '''
for scope in ("within_region", "cross_region"):
    fig, paths = viz.plot_null_corrected_grid(
        traces, figs, scope=scope, measure="count", min_pairs=MIN_PAIRS
    )
    display(Image(filename=str(paths["png"])))

counts = pd.read_csv(SUMMARY_DIR / "condition_comparisons_by_region_counts.csv")
merged = contrasts.merge(
    counts, on=["scope", "region_pair", "condition_a", "condition_b"],
    suffixes=("_norm", "_count"),
)
display(
    merged.loc[
        :,
        ["scope", "region_pair", "condition_a", "condition_b",
         "mean_difference_norm", "significant_norm",
         "mean_difference_count", "significant_count"],
    ].round(4)
)
'''

ABOVE_NULL = '''
vs_null = pd.read_csv(SUMMARY_DIR / "vs_null_by_region.csv")
display(vs_null.round(4))
'''

ARTIFACT = '''
diagnostics = pd.read_csv(SUMMARY_DIR / "zero_lag_diagnostics.csv")
display(diagnostics.round(4))

fig, paths = viz.plot_zero_lag_diagnostics(diagnostics, figs)
display(Image(filename=str(paths["png"])))
'''


_CELL_IDS = count(1)


def _next_id() -> str:
    return f"cell-{next(_CELL_IDS):02d}"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    markdown(f"""
# {TITLE}

Do two neurons recorded at the same time coordinate their spiking differently
during **interactive-face**, **non-interactive-face** and **object** fixations —
within a region, and across regions?

## What is computed

Every cross-correlation is computed on one fixation's two **unsmoothed 1 ms**
spike trains, over the **±500 ms** window around fixation onset, and only then
averaged. Nothing outside that window enters anything. Cross-correlating
condition-averaged PSTHs would measure shared rate structure, not coordination;
smoothing first would blur the timing this exists to measure.

The correlation is **linear** (zero-padded), the statistic
`scipy.signal.correlate` computes and the one the behavioural
cross-correlations use.

## The null

One null: the **trial shuffle**. Unit A's train on fixation *i* is paired with
unit B's train on some other fixation. Both units keep their fixation-locked
rate profiles and their exact spike counts; only the trial-by-trial covariation
is destroyed. An excess over it means the two cells co-fluctuate from fixation
to fixation.

Each draw is a *derangement* of the fixation index, so the null is built from
exactly as many pairings as the observed statistic. Estimating it from all
`F(F−1)` cross-fixation pairings would shrink its standard error and inflate
every comparison.

## The two measures

| measure | definition | role |
|---|---|---|
| **normalised correlation** | correlation ÷ `sqrt(n_A · n_B)` | headline |
| coincidence count | spike pairs at that lag, per fixation | second measure |

The normalisation is the one `core.signal.normalize_cross_correlation_sqrt_bin_count`
applies to the behavioural cross-correlations — for binary vectors `sqrt(n_x·n_y)`
is exactly `‖x‖·‖y‖`, the cosine normalisation.

**Is it appropriate here?** Partly, and it is worth being precise. It does *not*
equate chance levels across conditions with different firing rates: chance
coincidences grow as `n_A·n_B` while this divides by `sqrt(n_A·n_B)`, so a
higher-rate condition still sits higher. What removes the rate is the **null**,
which carries the same spike counts. And because the normaliser depends only on
spike counts, it cancels exactly in `observed − null` — so this choice sets the
y-axis of the observed plots and changes no statistic.

## How results are reported

Per region for within-region pairs, per region pair for cross-region pairs.
**Nothing is averaged across regions.** With these recordings a pooled number
would be a composition of very unequal region contributions rather than a
summary of them.

Each block runs on all pairs first, then on pairs where **both** units are
FDR-selective (`three_condition_core`).
"""),
    code(SETUP),
    markdown("""
## 1. Build state and summaries
"""),
    code(BUILD),
    markdown("""
## 2. What the recordings support

A cross-region pair exists only where both regions were recorded in the **same
session**. All four regions support within-region comparisons; across regions
every well-populated combination involves BLA, ACCg and OFC were never recorded
together, and dmPFC × OFC comes from a handful of sessions. Combinations below
the pair threshold appear in the tables but are excluded from the figures.

Days carrying the zero-lag artifact are removed from everything below.
"""),
    code(LOAD),
    markdown("""
# Within region

## 3. Observed correlation and the trial-shuffle null

Regions down, fixation conditions across. The **gap between the two curves** is
the coordination — the observed curve alone is not, since a correlation scales
with the product of the two firing rates.
"""),
    code(within()),
    markdown("""
### The same, for FDR-selective pairs only
"""),
    code(WITHIN_OBSERVED_SELECTIVE),
    markdown("""
## 4. Null-corrected coordination, and condition comparisons

Observed minus null, conditions overlaid within each region. Zero means no
coordination beyond what each unit's own fixation-locked rate profile predicts.

The contrasts are **paired within pair**: the same two neurons, electrodes and
session, differing only in which fixations were used. Benjamini–Hochberg
corrected; filled markers survived it.
"""),
    code(WITHIN_CORRECTED),
    markdown("""
### The same, for FDR-selective pairs only

Selecting units by a condition contrast and then asking whether coordination
differs by condition is circular, so this is a sensitivity check rather than
independent confirmation. What it can show is whether an effect is carried by
the selective subset or is distributed across the population.
"""),
    code(WITHIN_CORRECTED_SELECTIVE),
    markdown("""
# Across regions

## 5. Observed correlation and the trial-shuffle null

All pairs, then FDR-selective pairs.
"""),
    code(CROSS_OBSERVED),
    markdown("""
## 6. Null-corrected coordination, and condition comparisons
"""),
    code(CROSS_CORRECTED),
    markdown("""
### The same, for FDR-selective pairs only
"""),
    code(CROSS_CORRECTED_SELECTIVE),
    markdown("""
# Supporting

## 7. Second measure: raw coincidence counts

The same null-corrected comparison in coincidences per fixation rather than
normalised units, with the two sets of contrasts side by side. A conclusion that
holds in one measure and not the other is a conclusion about the normalisation,
not about the neurons.
"""),
    code(SECOND_MEASURE),
    markdown("""
## 8. Is there coordination above null at all?

Secondary to the condition comparison, but worth confirming there is something
to compare. One-sample Wilcoxon of the per-pair excess against zero, per region
and condition, using `z` — the right quantity for this question and the wrong
one for comparing conditions, since it scales with the square root of the
fixation count.
"""),
    code(ABOVE_NULL),
    markdown("""
## 9. The zero-lag artifact

Two randomly sampled neurons are essentially never monosynaptically connected,
so a sharp zero-lag peak shared by most pairs on a day is common input —
movement, arousal, or a shared reference/ground artifact. It is a property of
the **day and array**, not of the pair, so flagged days are removed from every
result above rather than noted here.
"""),
    code(ARTIFACT),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = Path(__file__).resolve().parent / FILENAME
    out_path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {out_path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
