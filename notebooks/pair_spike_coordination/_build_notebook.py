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
'''

VALIDATE = '''
identities = psc.verify_null_identities()
display(identities)
assert identities["passes"].all(), "Null construction identities failed."

sensitivity = psc.verify_null_sensitivity()
display(sensitivity.round(3))

fig, paths = viz.plot_null_validation(identities, sensitivity, fig_settings)
display(Image(filename=str(paths["png"])))
'''

ABOVE_NULL = '''
vs_null = psc.test_against_null(pairs, metric=EFFECT_METRIC)
display(vs_null.round(4))

fig, paths = viz.plot_excess_vs_null(vs_null, fig_settings)
display(Image(filename=str(paths["png"])))
'''

TRACES = '''
by_scope = pd.read_pickle(SUMMARY_DIR / "group_z_traces_by_scope.pkl")

for null_name in ("trial_shuffle", "circular_shift"):
    fig, paths = viz.plot_group_z_traces(by_scope, fig_settings, null_name=null_name)
    display(Markdown(f"**{viz.NULL_LABELS[null_name]}** — {viz.NULL_MEANINGS[null_name]}"))
    display(Image(filename=str(paths["png"])))
'''

EFFECTS = '''
summary = psc.summarize_coordination(pairs, metric=EFFECT_METRIC)
display(summary.round(4))

fig, paths = viz.plot_condition_effects(summary, fig_settings)
display(Image(filename=str(paths["png"])))
'''

CONTRASTS = '''
comparisons = psc.compare_conditions(pairs, metric=EFFECT_METRIC)
display(comparisons.round(4))

fig, paths = viz.plot_condition_contrasts(comparisons, fig_settings, label="All pairs")
display(Image(filename=str(paths["png"])))
'''

MATCHED = '''
matched_metric = EFFECT_METRIC + "_matched"
if matched_metric in pairs.columns and pairs[matched_metric].notna().any():
    matched = psc.compare_conditions(pairs, metric=matched_metric)
    display(matched.round(4))
    fig, paths = viz.plot_condition_contrasts(
        matched, fig_settings, label="Trial-count matched", stem="fig04b_condition_contrasts_matched"
    )
    display(Image(filename=str(paths["png"])))
    merged = comparisons.merge(
        matched, on=["scope", "condition_a", "condition_b"], suffixes=("_full", "_matched")
    )
    display(
        merged.loc[
            :,
            ["scope", "condition_a", "condition_b",
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

selective_comparisons = psc.compare_conditions(selective, metric=EFFECT_METRIC)
display(selective_comparisons.round(4))

fig, paths = viz.plot_condition_contrasts(
    selective_comparisons, fig_settings,
    label="Both units FDR-selective", stem="fig04c_condition_contrasts_selective",
)
display(Image(filename=str(paths["png"])))

selective_traces = pd.read_pickle(SUMMARY_DIR / "group_z_traces_selective.pkl")
fig, paths = viz.plot_selectivity_comparison(by_scope, selective_traces, fig_settings)
display(Image(filename=str(paths["png"])))
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
    clean_comparisons = psc.compare_conditions(clean, metric=EFFECT_METRIC)
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

Everything here rests on per-fixation spike trains. Cross-correlating
condition-averaged PSTHs would measure shared rate structure, not coordination,
so every cross-correlation is computed on one fixation's two 1 ms trains and
only then averaged. The trains are **unsmoothed** — smoothing before
cross-correlation blurs exactly the fine timing this analysis exists to measure.

## Reading the two nulls

Coordination is only meaningful relative to a null, and the two nulls answer
different questions:

| null | what it destroys | what it keeps | an excess means |
|---|---|---|---|
| **trial shuffle** | trial-by-trial covariation | each unit's fixation-locked rate profile | the two cells co-fluctuate from fixation to fixation |
| **circular shift** | fine temporal alignment within a fixation | that fixation's spike count and slow envelope | coordination finer than the shift, beyond slow co-modulation |

Each null draw is a **derangement** of the fixation index, so it is built from
exactly as many terms as the observed statistic. Estimating the null from all
`F*(F-1)` cross-fixation pairings would shrink its standard error and inflate
every z-score.

## Which number to compare

Two standardised quantities appear below and they are not interchangeable:

- **`z`** — the excess in units of the null SD of the fixation-averaged
  statistic. Grows with the square root of the fixation count. Use it for *is
  this coordinated at all*.
- **`effect`** — `z / sqrt(n_fixations)`, the excess in single-fixation null
  units. Its expectation does not depend on trial count. **Use it to compare
  conditions.** Interactive-face fixations outnumber non-interactive-face ones
  about five to one, so ranking conditions by `z` would rank them largely by
  trial count.

## Running this notebook

Section 1 checks what is built and can submit the SLURM array itself, then
section 2 rebuilds the summary tables. Everything after that reads those
summaries. The equivalent from a shell is:

    sbatch --array=0-41 hpc/ephys/run_fixation_pair_spike_coordination.sbatch
    python scripts/ephys/analysis/build_fixation_pair_spike_coordination_summary.py
"""),
    code(SETUP),
    markdown("""
## 1. Build state

The per-session pair tables are built by a SLURM array over the 42 recording
dates. This reports what exists and what is missing; it does **not** queue
anything unless `SUBMIT_JOBS` is set to `True`, in which case it submits only
the incomplete dates and waits for the array to finish.
"""),
    code(ORCHESTRATE),
    markdown("""
Once every date is built, aggregate the per-session tables into the summary
files the rest of the notebook reads. This is the slow step — it touches every
session file — so set `REBUILD_SUMMARIES = False` to reuse what is on disk.
"""),
    code(REBUILD_SUMMARY),
    markdown("""
## 2. Load

One row per (pair, condition). Traces stay on disk; the scalar summaries are
what every test below uses.
"""),
    code(LOAD),
    markdown("""
## 3. Do the nulls behave?

Two checks, in order. The **identities** confirm the fast frequency-domain path
returns exactly what brute-force cross-correlation would — the speed-ups are
only worth having if they are provably the same number.

The **sensitivity** check is the one that matters for interpretation. Four
synthetic pairs are pushed through the real computation:

- `independent` — no coupling. Both nulls should sit at zero. If they do not,
  every z below is inflated.
- `shared_rate` — the units share a per-fixation gain but are otherwise
  independent within a fixation. The trial-shuffle null **should** detect this;
  the circular-shift null should **not**.
- `synchronous` — a fraction of spikes copied at a fixed 4 ms lag. Both should
  detect it, at −4 ms.
- `common_zero_lag` — spikes injected into both units in the same 1 ms bin: the
  artifact signature, not a pairwise interaction.

Note `peak_z` is a maximum over ~200 lags, so it is inflated under the null by
construction (an uncoupled pair reaches ≈3). The windowed `mean_z_pm10ms`
columns are not maxima and are what the tests use.
"""),
    code(VALIDATE),
    markdown("""
## 4. Is there any coordination above null to begin with?

Before asking whether conditions differ, ask whether there is anything to
differ. A one-sample Wilcoxon signed-rank test of the per-pair excess against
zero — zero being the null's own expectation.
"""),
    code(ABOVE_NULL),
    markdown("""
## 5. Where in lag does the coordination sit?

Mean per-lag excess across pairs, one panel per scope. Distance from zero *is*
the coordination; bands are standard error across pairs.

Compare the two nulls: structure present against the trial-shuffle null but
absent against the circular-shift null is slow co-fluctuation rather than fine
synchrony.
"""),
    code(TRACES),
    markdown("""
## 6. Condition effects, by region pair

Bootstrap confidence intervals on the per-pair effect, broken out by region and
region pair.
"""),
    code(EFFECTS),
    markdown("""
## 7. Does interactive face change coordination?

The comparison is **paired within pair**: the same two neurons, the same
electrodes, the same session, differing only in which fixations were used. That
removes pair identity, firing rate and recording quality as explanations in one
step, which a comparison across separate pair populations could not do.
Benjamini–Hochberg correction across the reported contrasts.
"""),
    code(CONTRASTS),
    markdown("""
## 8. Control: trial-count matching

The `effect` metric is already constructed not to scale with trial count. This
is the direct check on the same confound: every pair recomputed on a common
fixation count across conditions.

**If a condition difference survives here, trial count is not driving it.**
"""),
    code(MATCHED),
    markdown("""
## 9. Restricting to FDR-selective units

Pairs where **both** units are significantly selective for at least one
fixation-condition contrast (FDR-corrected, `three_condition_core`).

This is a sensitivity check, not independent confirmation — selecting units by a
condition contrast and then asking whether coordination differs by condition is
circular. What it can legitimately show is whether any effect is carried by the
selective subset or is distributed across the population.
"""),
    code(SELECTIVE),
    markdown("""
## 10. The zero-lag artifact

Earlier runs showed a sharp zero-lag peak on some days. That is almost certainly
**not** a pairwise interaction: the chance that two randomly sampled neurons are
monosynaptically connected is near zero, so a zero-lag peak shared by most pairs
on a day is common input — movement, arousal, or a shared reference/ground
artifact on the recording system.

The signature that separates an artifact from a real effect is that it is a
property of the **day and array**, not of the pair: on a contaminated day nearly
every simultaneously recorded pair shows it, including pairs sharing nothing
else. This flags such days as outliers instead of averaging them in.
"""),
    code(ZERO_LAG),
    markdown("""
## 11. Does the conclusion survive dropping flagged days?

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
