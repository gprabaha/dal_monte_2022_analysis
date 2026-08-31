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


ORCHESTRATE = '''
settings = psc.build_pair_spike_coordination_settings_from_config(
    dataset_cfg_path=str(DATASET_CFG_PATH),
    coordination_cfg_path=str(repo_root / "configs" / "ephys_fixation_pair_spike_coordination.yaml"),
)

# Submission is opt-in. The array job costs real cluster time and a notebook
# cell is easy to re-run by accident, so leaving this False only *reports* what
# is missing and prints the sbatch line to run.
SUBMIT_JOBS = False
WAIT_FOR_JOBS = True

per_date = psc.ensure_pair_coordination_built(
    settings,
    submit=SUBMIT_JOBS,
    wait=WAIT_FOR_JOBS,
    sbatch_path=repo_root / psc.DEFAULT_SBATCH_PATH,
    repo_root=repo_root,
)
display(per_date)
'''

REBUILD_SUMMARY = '''
# Regenerate the summary tables from whatever is on disk. Skip if they are
# already current -- this reads every session file, so it is the slow step.
REBUILD_SUMMARIES = True

if REBUILD_SUMMARIES:
    _ = psc.run_summary_build(settings, metric=EFFECT_METRIC)
else:
    display(Markdown("_Using the summary tables already on disk._"))
'''

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
from dal_monte_2022_analysis.ephys.plotting import thesis_common as style

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

DATASET_CFG_PATH = repo_root / "configs" / "dataset.yaml"
cfg = load_config(str(DATASET_CFG_PATH))

SUMMARY_DIR = build_analysis_output_dir(cfg, psc.DEFAULT_OUTPUT_SUBDIR) / "summary"
FIGURE_DIR = build_analysis_output_dir(cfg, psc.DEFAULT_OUTPUT_SUBDIR) / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
fig_settings = viz.PairCoordinationPlotSettings(output_dir=FIGURE_DIR)

#: The excess is standardised in single-fixation null units so that it does not
#: scale with how many fixations a condition contains.  Interactive-face
#: fixations outnumber non-interactive-face ones roughly five to one, so a
#: z-based comparison would rank conditions largely by trial count.
EFFECT_METRIC = "trial_shuffle_mean_effect_pm10ms"

print("summary dir:", SUMMARY_DIR)
print("figure dir :", FIGURE_DIR)
'''

LOAD = '''
pairs, lags_ms = psc.load_pair_coordination(str(DATASET_CFG_PATH))
print(f"pair-condition rows: {len(pairs):,}")
print(f"lag axis: {lags_ms.size} bins, {lags_ms.min():.0f} to {lags_ms.max():.0f} ms")

inventory = psc.build_pair_inventory(pairs)
display(inventory)

# Which region and region-pair comparisons the recordings actually support.
# A cross-region pair exists only where both regions were recorded in the same
# session, and that was far from uniform.
region_inventory = psc.build_region_pair_inventory(pairs)
display(region_inventory)

USABLE = psc.sufficient_region_pairs(pairs)
MIN_PAIRS = psc.DEFAULT_MIN_PAIRS_FOR_REPORTING
print(f"region pairs with >= {MIN_PAIRS} pairs in every condition: {USABLE}")
thin = region_inventory.loc[~region_inventory["sufficient_pairs"], "region_pair"].tolist()
if thin:
    print(f"reported but not interpreted (too few pairs): {thin}")
'''

OBSERVED_AND_NULLS = '''
by_scope = pd.read_pickle(SUMMARY_DIR / "group_traces_by_scope.pkl")

for scope in ("within_region", "cross_region"):
    fig, paths = viz.plot_observed_and_nulls(by_scope, fig_settings, scope=scope)
    display(Image(filename=str(paths["png"])))
'''

REGION_TRACES = '''
by_region = pd.read_pickle(SUMMARY_DIR / "group_traces_by_region.pkl")

for scope in ("within_region", "cross_region"):
    for null_name in ("trial_shuffle", "circular_shift"):
        fig, paths = viz.plot_region_traces(
            by_region, fig_settings, scope=scope, null_name=null_name,
            min_pairs=MIN_PAIRS,
        )
        display(Image(filename=str(paths["png"])))
'''

REGION_TESTS = '''
vs_null_by_region = pd.read_csv(SUMMARY_DIR / "coordination_vs_null_by_region.csv")
display(vs_null_by_region.round(4))

fig, paths = viz.plot_region_condition_tests(
    vs_null_by_region, fig_settings, min_pairs=MIN_PAIRS
)
display(Image(filename=str(paths["png"])))
'''

REGION_EFFECTS = '''
summary_by_region = pd.read_csv(SUMMARY_DIR / "coordination_summary_by_region.csv")
display(summary_by_region.round(4))

fig, paths = viz.plot_condition_effects(summary_by_region, fig_settings)
display(Image(filename=str(paths["png"])))
'''

REGION_CONTRASTS = '''
by_region_contrasts = pd.read_csv(SUMMARY_DIR / "condition_comparisons_by_region.csv")
display(by_region_contrasts.round(4))

fig, paths = viz.plot_condition_contrasts(
    by_region_contrasts, fig_settings,
    label="All pairs, per region", stem="fig05_condition_contrasts_by_region",
)
display(Image(filename=str(paths["png"])))
'''

POOLED = '''
fig, paths = viz.plot_group_z_traces(by_scope, fig_settings, null_name="trial_shuffle")
display(Image(filename=str(paths["png"])))

vs_null = pd.read_csv(SUMMARY_DIR / "coordination_vs_null.csv")
display(vs_null.round(4))
fig, paths = viz.plot_excess_vs_null(vs_null, fig_settings)
display(Image(filename=str(paths["png"])))

comparisons = pd.read_csv(SUMMARY_DIR / "condition_comparisons.csv")
display(comparisons.round(4))
fig, paths = viz.plot_condition_contrasts(comparisons, fig_settings, label="All pairs, pooled", stem="fig08_condition_contrasts_pooled")
display(Image(filename=str(paths["png"])))
'''

MATCHED = '''
matched_metric = EFFECT_METRIC + "_matched"
if matched_metric in pairs.columns and pairs[matched_metric].notna().any():
    matched = psc.compare_conditions(
        pairs, metric=matched_metric, group_columns=("scope", "region_pair")
    )
    display(matched.round(4))
    fig, paths = viz.plot_condition_contrasts(
        matched, fig_settings,
        label="Trial-count matched, per region", stem="fig09_contrasts_matched",
    )
    display(Image(filename=str(paths["png"])))
    merged = by_region_contrasts.merge(
        matched, on=["scope", "region_pair", "condition_a", "condition_b"],
        suffixes=("_full", "_matched"),
    )
    display(
        merged.loc[
            :,
            ["scope", "region_pair", "condition_a", "condition_b",
             "mean_difference_full", "mean_difference_matched",
             "significant_full", "significant_matched"],
        ].round(4)
    )
else:
    display(Markdown("_Trial-count-matched columns are absent; rebuild with `trial_match_conditions: true`._"))
'''

SELECTIVE = '''
selective = pairs.loc[pairs["both_selective"]]
print(f"pairs with both units FDR-selective: {len(selective):,} of {len(pairs):,} "
      f"({len(selective) / max(len(pairs), 1):.1%})")
display(psc.build_pair_inventory(selective))

sel_by_scope = pd.read_pickle(SUMMARY_DIR / "group_traces_selective.pkl")
sel_by_region = pd.read_pickle(SUMMARY_DIR / "group_traces_by_region_selective.pkl")

for scope in ("within_region", "cross_region"):
    fig, paths = viz.plot_observed_and_nulls(
        sel_by_scope, fig_settings, scope=scope, stem="fig10_observed_and_nulls_selective"
    )
    display(Image(filename=str(paths["png"])))
    fig, paths = viz.plot_region_traces(
        sel_by_region, fig_settings, scope=scope, label="FDR-selective",
        min_pairs=MIN_PAIRS, stem="fig11_region_traces_selective",
    )
    display(Image(filename=str(paths["png"])))
'''

SELECTIVE_TESTS = '''
sel_contrasts = pd.read_csv(SUMMARY_DIR / "condition_comparisons_by_region_selective.csv")
display(sel_contrasts.round(4))

fig, paths = viz.plot_condition_contrasts(
    sel_contrasts, fig_settings,
    label="Both units FDR-selective, per region", stem="fig12_contrasts_selective",
)
display(Image(filename=str(paths["png"])))

# Side by side with the all-pairs result on the same rows.
side_by_side = by_region_contrasts.merge(
    sel_contrasts, on=["scope", "region_pair", "condition_a", "condition_b"],
    suffixes=("_all", "_selective"),
)
display(
    side_by_side.loc[
        :,
        ["scope", "region_pair", "condition_a", "condition_b",
         "n_pairs_all", "mean_difference_all", "significant_all",
         "n_pairs_selective", "mean_difference_selective", "significant_selective"],
    ].round(4)
)
'''

ZERO_LAG = '''
diagnostics = psc.build_zero_lag_diagnostics(pairs)
display(diagnostics.round(4))

flagged = diagnostics.loc[diagnostics["suspected_zero_lag_artifact"].fillna(False)]
if len(flagged):
    display(Markdown(
        f"**{len(flagged)} date/scope combinations flagged.** "
        f"Dates: {sorted(flagged['date'].unique())}"
    ))
else:
    display(Markdown("_No date stands out as a zero-lag outlier._"))

fig, paths = viz.plot_zero_lag_diagnostics(diagnostics, fig_settings)
display(Image(filename=str(paths["png"])))
'''

EXCLUDE = '''
flagged_dates = set(
    psc.build_zero_lag_diagnostics(pairs)
    .pipe(lambda d: d.loc[d["suspected_zero_lag_artifact"].fillna(False), "date"])
    .astype(str)
)
if flagged_dates:
    clean = pairs.loc[~pairs["date"].astype(str).isin(flagged_dates)]
    print(f"dropping {len(flagged_dates)} flagged date(s): {sorted(flagged_dates)}")
    print(f"pairs remaining: {len(clean):,} of {len(pairs):,}")
    clean_comparisons = psc.compare_conditions(
        clean, metric=EFFECT_METRIC, group_columns=("scope", "region_pair")
    )
    display(clean_comparisons.round(4))
    display(Markdown(
        "If the condition effects above match the all-days result, the conclusion "
        "does not rest on the flagged days."
    ))
else:
    display(Markdown("_No days were flagged, so there is nothing to exclude._"))
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

Do two neurons recorded at the same time coordinate their spiking more during
**interactive-face** fixations than during non-interactive-face or object
fixations, and does that differ **within** a region versus **across** regions?

Everything rests on per-fixation spike trains. Cross-correlating
condition-averaged PSTHs would measure shared rate structure, not coordination,
so every cross-correlation is computed on one fixation's two 1 ms trains and
only then averaged. The trains are **unsmoothed** — smoothing before
cross-correlation blurs exactly the fine timing this analysis exists to measure.

## Linear correlation, everything inside ±500 ms

The correlation is **linear** (zero-padded transform) — the statistic
`scipy.signal.correlate` computes, and the one the behavioural
cross-correlations use. At lag L only the `N − |L|` genuinely overlapping bins
contribute, so no spike is ever paired with one a full window away.

Observed and both nulls all use the same 1000 bins of 1 ms spike counts spanning
−500 to +500 ms around fixation onset. Nothing outside that window enters any
computation: 95% of ±5 s surrounds contain at least one *other* analysed
fixation (median 5), so outside the window is not baseline.

Linear correlation tapers — fewer bins contribute as |lag| grows — but both
nulls carry the identical taper, so it cancels in every excess and z-score.

## Reading the two nulls

| null | destroys | keeps | an excess means |
|---|---|---|---|
| **trial shuffle** | trial-by-trial covariation | each unit's fixation-locked rate profile | the cells co-fluctuate across fixations |
| **circular shift** | fine temporal alignment | that fixation's spike count and slow envelope | coordination finer than the shift |

The shift null rotates a train within the window, which wraps. That is fine in a
null — destroying alignment is the point — and is exactly why it is not fine in
the observed statistic.

## Which number to compare

- **`z`** — excess in units of the null SD of the fixation-averaged statistic.
  Grows with `sqrt(n_fixations)`. Use for *is this coordinated at all*.
- **`effect`** — `z / sqrt(n_fixations)`, in single-fixation null units. Does not
  depend on trial count. **Use to compare conditions**, since interactive-face
  fixations outnumber non-interactive ones about five to one.

## How results are reported

Per **region** for within-region pairs and per **region pair** for cross-region
pairs, *before* any pooling. Pooling first would let one region with many pairs
carry a conclusion that does not hold in the others. Pooled tables follow as a
summary. Everything is run on all recorded pairs first, then repeated on pairs
where both units are FDR-selective.

Which comparisons are available is set by the recordings, not by choice — see
the region-pair inventory in section 2. All four regions support within-region
comparisons; across regions, only the combinations involving BLA are
well populated.
"""),
    code(SETUP),
    markdown("""
## 1. Build state

Per-session tables are built by a SLURM array over the 42 recording dates. This
reports what exists; it queues nothing unless `SUBMIT_JOBS = True`, in which
case it submits only the incomplete dates and waits.
"""),
    code(ORCHESTRATE),
    markdown("""
Aggregate the per-session tables into the summary files the rest of the notebook
reads. This touches every session file, so set `REBUILD_SUMMARIES = False` to
reuse what is on disk.
"""),
    code(REBUILD_SUMMARY),
    markdown("""
## 2. What is in the analysis

**Read the region-pair inventory before any result.** A cross-region pair exists
only where both regions were recorded in the *same session*, and the recording
configuration was far from uniform: every well-populated cross-region
combination involves BLA, ACCg and OFC were never recorded together at all, and
dmPFC × OFC comes from a handful of sessions. All four regions support
within-region comparisons.

Combinations below the pair threshold are reported in the tables but excluded
from the region figures, so a hundred-pair curve is never drawn beside a
thirty-thousand-pair one.
"""),
    code(LOAD),
    markdown("""
## 3. What the correlation and its nulls actually look like

Read this first. The black curve is the mean cross-correlation across pairs, in
coincidences per fixation; the two dashed curves are where each null sits on the
same axes. The excess is something you can see rather than something to take on
trust from a z-score. Bands are standard error across pairs.

The two nulls should **not** coincide. The trial-shuffle null keeps each unit's
fixation-locked rate profile, so it sits at the level shared rate structure
alone produces. The circular-shift null keeps each fixation's own spike count
but destroys alignment. Observed above both is coordination that neither
explains.
"""),
    code(OBSERVED_AND_NULLS),
    markdown("""
## 4. Excess over null, per region and per region pair

The same comparison as a standardised excess, resolved per region (within) and
per region pair (across), for both nulls. Zero is the null's own expectation.

Structure present against the trial-shuffle null but absent against the
circular-shift null is slow co-fluctuation rather than fine synchrony.
"""),
    code(REGION_TRACES),
    markdown("""
### Is coordination above null, region by region?

One-sample Wilcoxon signed-rank test of the per-pair excess against zero, per
region and condition. This is the *is there anything there* question, answered
before any pooling.
"""),
    code(REGION_TESTS),
    markdown("""
## 5. Condition effects per region

Bootstrap confidence intervals on the per-pair effect, by region and region
pair.
"""),
    code(REGION_EFFECTS),
    markdown("""
### Does interactive face change coordination, region by region?

Comparisons are **paired within pair**: the same two neurons, the same
electrodes, the same session, differing only in which fixations were used. That
removes pair identity, firing rate and recording quality in one step.
Benjamini–Hochberg corrected across the reported contrasts.
"""),
    code(REGION_CONTRASTS),
    markdown("""
## 6. Pooled across regions

The same quantities pooled to scope level, as a summary of the region-resolved
results above. Read these *after* section 5, and treat a pooled effect that no
individual region shows as a composition artifact rather than a finding.
"""),
    code(POOLED),
    markdown("""
## 7. Control: trial-count matching

The `effect` metric is already built not to scale with trial count. This is the
direct check: every pair recomputed on a common fixation count across
conditions. **If a condition difference survives here, trial count is not
driving it.**
"""),
    code(MATCHED),
    markdown("""
## 8. Pairs where both units are FDR-selective

Everything above used **all** recorded pairs. This repeats it on pairs where
both units are significantly selective for at least one fixation-condition
contrast (FDR-corrected, `three_condition_core`).

This is a sensitivity check, not independent confirmation — selecting units by a
condition contrast and then asking whether coordination differs by condition is
circular. What it can legitimately show is whether an effect is carried by the
selective subset or is distributed across the population.
"""),
    code(SELECTIVE),
    markdown("""
### Selective-pair contrasts, against the all-pair result

The final table puts the two side by side on the same rows, so a difference in
conclusion is visible directly rather than inferred by comparing two tables.
"""),
    code(SELECTIVE_TESTS),
    markdown("""
## 9. The zero-lag artifact

Earlier runs showed a sharp zero-lag peak on some days. That is almost certainly
**not** a pairwise interaction: the chance that two randomly sampled neurons are
monosynaptically connected is near zero, so a zero-lag peak shared by most pairs
on a day is common input — movement, arousal, or a shared reference/ground
artifact.

The signature separating an artifact from a real effect is that it is a property
of the **day and array**, not of the pair: on a contaminated day nearly every
simultaneously recorded pair shows it, including pairs sharing nothing else.
"""),
    code(ZERO_LAG),
    markdown("""
## 10. Does the conclusion survive dropping flagged days?

The honest test of an artifact: remove the suspect days and see whether the
condition effects hold.
"""),
    code(EXCLUDE),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
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
