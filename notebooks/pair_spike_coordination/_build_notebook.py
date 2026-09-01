"""Author the neural pair spike-coordination notebook from source strings.

Per ``AGENTS.md`` the notebook is thin: every function it calls lives in
``src/dal_monte_2022_analysis``.  This script assembles narrative and call sites
only.

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
pd.set_option("display.max_columns", 50)

cfg = load_config(str(repo_root / "configs" / "dataset.yaml"))
settings = psc.build_pair_spike_coordination_settings_from_config(
    dataset_cfg_path=str(repo_root / "configs" / "dataset.yaml"),
    coordination_cfg_path=str(repo_root / "configs" / "ephys_fixation_pair_spike_coordination.yaml"),
)

SUMMARY_DIR = build_analysis_output_dir(cfg, psc.DEFAULT_OUTPUT_SUBDIR) / "summary"
FIGURE_DIR = build_analysis_output_dir(cfg, psc.DEFAULT_OUTPUT_SUBDIR) / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
figs = viz.PairCoordinationPlotSettings(output_dir=FIGURE_DIR)
MIN_PAIRS = psc.DEFAULT_MIN_PAIRS_FOR_REPORTING

print(f"window        : {settings.signal_window_ms[0]:.0f} to {settings.signal_window_ms[1]:.0f} ms")
print(f"signal        : {settings.signal_input_column} from {settings.trial_input_filename}")
print(f"null          : {settings.n_circular_shift_draws} circular shifts, min {settings.min_circular_shift_ms:.0f} ms")
'''

BUILD = '''
# Submission is opt-in: the array costs cluster time and a cell is easy to
# re-run by accident. False only reports what is missing.
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
traces_sig = pd.read_pickle(SUMMARY_DIR / "traces_by_region_selective.pkl")
contrasts = pd.read_csv(SUMMARY_DIR / "condition_comparisons_by_region.csv")
contrasts_sig = pd.read_csv(SUMMARY_DIR / "condition_comparisons_by_region_selective.csv")

# Confirm every cross-correlation came from the same +/-500 ms window.
one = next(
    (build_analysis_output_dir(cfg, psc.DEFAULT_OUTPUT_SUBDIR)).glob(
        "date=*/session=*/pair_coordination.pkl"
    )
)
meta = pd.read_pickle(one)["meta"]
print(f"window {meta['signal_window_ms']} ms at {meta['bin_size_ms']:.0f} ms bins, "
      f"{meta['correlation_kind']} correlation, null = {meta['null_kind']}")

display(pd.read_csv(SUMMARY_DIR / "region_pair_inventory.csv"))
'''


def observed(traces: str, scope: str, label: str = "") -> str:
    arg = f', label="{label}"' if label else ""
    return f'''
fig, paths = viz.plot_observed_and_null_grid(
    {traces}, figs, scope="{scope}", min_pairs=MIN_PAIRS{arg}
)
display(Image(filename=str(paths["png"])))
'''


def corrected(traces: str, table: str, scope: str, label: str = "") -> str:
    arg = f', label="{label}"' if label else ""
    return f'''
fig, paths = viz.plot_null_corrected_grid(
    {traces}, figs, scope="{scope}", min_pairs=MIN_PAIRS{arg}
)
display(Image(filename=str(paths["png"])))

subset = {table}.loc[{table}["scope"] == "{scope}"]
display(subset.round(4))

fig, paths = viz.plot_condition_contrasts(
    {table}, figs, scope="{scope}"{arg}
)
display(Image(filename=str(paths["png"])))
'''


ABOVE_NULL = '''
display(pd.read_csv(SUMMARY_DIR / "vs_null_by_region.csv").round(4))
'''

ARTIFACT = '''
dropped = pd.read_csv(SUMMARY_DIR / "dropped_artifact_dates.csv")
if len(dropped):
    display(Markdown(
        f"**{len(dropped)} date(s) removed from every result above:** "
        f"{sorted(dropped['date'].astype(str))}"
    ))
else:
    display(Markdown("_No date was flagged._"))

diagnostics = pd.read_csv(SUMMARY_DIR / "zero_lag_diagnostics.csv")
fig, paths = viz.plot_zero_lag_diagnostics(diagnostics, figs)
display(Image(filename=str(paths["png"])))
'''


_CELL_IDS = count(1)


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"cell-{next(_CELL_IDS):02d}",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"cell-{next(_CELL_IDS):02d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    markdown(f"""
# {TITLE}

Does spike coordination between two simultaneously recorded neurons differ
between **interactive-face**, **non-interactive-face** and **object** fixations
— within a region, and across regions?

## What is computed

Every cross-correlation is computed on one fixation's two **unsmoothed 1 ms**
spike trains, over the **±500 ms** window around fixation onset, and only then
averaged across fixations. Nothing outside that window enters anything.

The correlation is **linear** (zero-padded), the same statistic
`scipy.signal.correlate` computes. Values are **coincidences per fixation**:
spike pairs at that lag, averaged over fixations. No normalisation is applied —
the null already carries both units' firing rates and exact spike counts, so
`observed − null` is rate-controlled by construction.

## The null

One null, the **circular shift**: unit B's train is rotated by a random offset
(at least 50 ms) **within its own fixation**. Each fixation keeps its own spike
counts and its slow envelope, so two cells whose excitability simply rises and
falls together across fixations produce *no* excess over it. What is destroyed
is fine temporal alignment.

This replaces the cross-trial shuffle, which pairs fixation *i* with fixation
*j* and so destroys across-fixation rate covariation as well — too low a bar. A
synthetic pair with shared gain and no timing relationship reaches z = 2.0
against the cross-trial null and 1.3 against this one, level with an uncoupled
pair.

**Known bias, stated up front.** The rotation also misaligns the two units'
*fixation-locked rate profiles*. Two cells that both respond to fixation onset
show a broad correlation from that alone, and the shift removes it — so the null
sits low at every lag and a uniform positive excess appears with no timing
structure behind it. On a synthetic uncoupled pair with a shared rate bump this
bias is about z = 1.2. Read the **shape** of the excess, not its offset: a peak
standing above its own flanks is coordination, a flat elevation is this bias.

## Structure

Per region, per fixation type. **Nothing is averaged across regions.**

1. Within region — all pairs
2. Within region — FDR-selective pairs
3. Across regions — all pairs
4. Across regions — FDR-selective pairs

Each block shows the observed correlation against the null, then the
null-corrected difference with the condition contrasts.
"""),
    code(SETUP),
    markdown("""
## Build state and summaries
"""),
    code(BUILD),
    markdown("""
## What the recordings support

A cross-region pair exists only where both regions were recorded in the **same
session**. All four regions support within-region comparisons; across regions
every well-populated combination involves BLA. Combinations below the pair
threshold appear in the table but are excluded from the figures.

Days carrying the zero-lag artifact are removed from everything below.
"""),
    code(LOAD),
    markdown("""
# 1. Within region — all pairs

## Observed correlation and the cross-trial null

Regions down, fixation conditions across. The **gap between the curves** is the
coordination; the observed curve alone is not, since a correlation scales with
the product of the two firing rates.
"""),
    code(observed("traces", "within_region")),
    markdown("""
## Null-corrected, and condition comparisons

Observed minus null, conditions overlaid within each region. Zero means no
coordination beyond what each unit's own fixation-locked rate profile predicts.

Contrasts are **paired within pair** — the same two neurons, electrodes and
session, differing only in which fixations were used. Benjamini–Hochberg
corrected; filled markers survived it.
"""),
    code(corrected("traces", "contrasts", "within_region")),
    markdown("""
# 2. Within region — FDR-selective pairs

Pairs where **both** units are significantly selective for at least one
fixation-condition contrast (FDR-corrected, `three_condition_core`).

Selecting units by a condition contrast and then asking whether coordination
differs by condition is circular, so this is a sensitivity check rather than
independent confirmation. What it can show is whether an effect is carried by
the selective subset or is distributed across the population.
"""),
    code(observed("traces_sig", "within_region", "FDR-selective pairs")),
    code(corrected("traces_sig", "contrasts_sig", "within_region", "FDR-selective pairs")),
    markdown("""
# 3. Across regions — all pairs
"""),
    code(observed("traces", "cross_region")),
    code(corrected("traces", "contrasts", "cross_region")),
    markdown("""
# 4. Across regions — FDR-selective pairs
"""),
    code(observed("traces_sig", "cross_region", "FDR-selective pairs")),
    code(corrected("traces_sig", "contrasts_sig", "cross_region", "FDR-selective pairs")),
    markdown("""
# Supporting

## Is there coordination above null at all?

Secondary to the condition comparison, but worth confirming there is something
to compare. One-sample Wilcoxon of the per-pair excess against zero, per region
and condition, using `z` — the right quantity for this question and the wrong
one for comparing conditions, since it scales with the square root of the
fixation count.
"""),
    code(ABOVE_NULL),
    markdown("""
## The zero-lag artifact

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
