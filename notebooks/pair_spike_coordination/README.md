# Neural pair spike coordination

Does spike coordination between **simultaneously recorded** neuron pairs change
with fixation condition — interactive face vs non-interactive face vs object —
and does that differ within a region versus across regions?

## Contents

- `pair_spike_coordination.ipynb` — the analysis notebook.
- `_build_notebook.py` — authors the notebook from source strings. Edit this,
  never the `.ipynb`, then re-run it. Per `AGENTS.md` the notebook is a thin
  display layer; the analysis lives in `src/`.

## Running it

The notebook drives the whole pipeline. Section 1 reports which of the 42 dates
are built and, with `SUBMIT_JOBS = True`, submits a SLURM array for **only the
incomplete dates** and waits for it. Section 2 rebuilds the summary tables.
Everything after reads those summaries.

Submission is opt-in on purpose: the array costs real cluster time and a
notebook cell is easy to re-run by accident. With `SUBMIT_JOBS = False` the cell
only inspects and prints the `sbatch` line it would have run.

The equivalent from a shell:

```bash
# 1. Per-session pair tables (SLURM array over the 42 recording dates)
sbatch --array=0-41 hpc/ephys/run_fixation_pair_spike_coordination.sbatch

# ...or one date locally
python scripts/ephys/analysis/build_fixation_pair_spike_coordination.py --date 01312018

# 2. Aggregate into the tables the notebook reads
python scripts/ephys/analysis/build_fixation_pair_spike_coordination_summary.py
```

After editing `_build_notebook.py`, regenerate the `.ipynb`:

```bash
conda run -n gaze_processing python notebooks/pair_spike_coordination/_build_notebook.py
```

Sanity-check the machinery without touching the data:

```bash
python scripts/ephys/analysis/build_fixation_pair_spike_coordination.py --verify-only
```

## Design decisions worth knowing before reading the figures

**Per-fixation, never per-average.** Coordination is a trial-by-trial quantity.
Cross-correlating condition-averaged PSTHs measures shared rate structure, not
coordination. Every cross-correlation is computed on one fixation's two 1 ms
spike trains and only then averaged.

**Unsmoothed input.** Smoothing before cross-correlation blurs the fine timing
the analysis exists to measure. Smooth the saved traces afterwards if a figure
needs it.

**Linear correlation, everything inside ±500 ms.** The correlation is linear
(zero-padded transform) — the same statistic `scipy.signal.correlate` computes
and the one the behavioural cross-correlations use. At lag L only the `N − |L|`
genuinely overlapping bins contribute, so no spike is ever paired with one a
full window away. A wrapping transform would pair a spike at −500 ms with one at
+500 ms and call it a coincidence at lag L, which is not a physical measurement.

The taper that linear correlation introduces needs no correction: both nulls
carry it identically, so it cancels in the excess and in every z-score. Per-lag
overlap counts are stored with each output for figures wanting comparable raw
magnitudes.

Widening the window is not an option: 95% of ±5 s surrounds contain at least one
other analysed fixation (median 5). Outside the window is not baseline.

**Two nulls, because they answer different questions.**

| null | destroys | keeps | an excess means |
|---|---|---|---|
| trial shuffle | trial-by-trial covariation | each unit's fixation-locked rate profile | the cells co-fluctuate across fixations |
| circular shift | fine temporal alignment | that fixation's count and slow envelope | coordination finer than the shift |

The shift null rotates a train within the window, which wraps. That is fine in a
null — destroying alignment is the point — and is exactly why it is not fine in
the observed statistic.

**Count-matched nulls.** There are `F*(F-1)` possible cross-fixation pairings but
only `F` real ones. A null estimated from all of them would have a far smaller
standard error than the observed statistic and would inflate every z-score. Each
draw is therefore a *derangement*: exactly `F` pairings, each fixation used once
per side.

**`z` vs `effect`.** `z` is the excess in null-SD units of the fixation-averaged
statistic and grows with `sqrt(n_fixations)`; use it for "is this coordinated at
all". `effect = z / sqrt(n_fixations)` does not depend on trial count; **use it
to compare conditions.** Interactive-face fixations outnumber non-interactive
ones roughly five to one, so ranking conditions by `z` would rank them largely
by trial count. Section 7 additionally recomputes everything on a common
fixation count as a direct control.

**Paired condition contrasts.** Every pair contributes all three conditions, so
comparisons are within-pair: the same two neurons, electrodes and session,
differing only in which fixations were used. Pair identity, firing rate and
recording quality cannot explain a difference.

**The zero-lag artifact.** Earlier runs showed a sharp zero-lag peak on some
days. The chance that two randomly sampled neurons are monosynaptically
connected is near zero, so a zero-lag peak shared by most pairs on a day is
common input — movement, arousal, or a shared reference/ground artifact. It is a
property of the *day and array*, not of the pair, so section 9 reports per-date
zero-lag prevalence and flags outlier days, and section 10 re-runs the condition
tests with those days dropped.

## Data provenance

Input is `fixations_spike_train_1ms.pkl`, the same extraction that produces
`fixations_psth_10ms.pkl`. The fixations, units and trials are identical to
those used by the single-unit, population-PCA and mRNN analyses, so a pair
result can be joined to a unit result without translation.

## Which comparisons the recordings support

A cross-region pair exists only where both regions were recorded in the **same
session**, and that was far from uniform. Across all 417 sessions:

| within region | pairs | | across regions | pairs |
|---|---|---|---|---|
| BLA | 49,172 | | BLA × dmPFC | 34,190 |
| OFC | 23,820 | | ACCg × BLA | 25,154 |
| ACCg | 19,893 | | BLA × OFC | 24,750 |
| dmPFC | 16,130 | | ACCg × dmPFC | 5,690 |
| | | | dmPFC × OFC | 140 |
| | | | **ACCg × OFC** | **0** |

All four regions support within-region comparisons. Across regions, every
well-populated combination involves **BLA** — ACCg and OFC were never recorded
together, and dmPFC × OFC comes from 10 sessions. `build_region_pair_inventory`
reports this and flags combinations below the pair threshold; the region figures
exclude them so a hundred-pair curve is never drawn beside a
thirty-thousand-pair one. Nothing is dropped silently.

## How results are reported

Per **region** for within-region pairs and per **region pair** for cross-region
pairs, *before* any pooling — pooling first would let one region with many pairs
carry a conclusion that does not hold in the others. Pooled tables follow as a
summary. Everything runs on all recorded pairs first, then repeats on pairs
where both units are FDR-selective, with the two shown side by side on the same
rows so a difference in conclusion is visible rather than inferred.

The first figure is the raw one: the mean cross-correlation with **both nulls
drawn on the same axes**, in coincidences per fixation, so the excess is visible
rather than only inferable from a z-score.

## Outputs

Written under `analysis_output_root/ephys/psth/fixation_pair_spike_coordination/`:

- `date=*/session=*/pair_coordination.pkl` — per-session pair tables with traces
  (~2 GB total).
- `summary/*_by_region.csv` — the primary tables: coordination vs null,
  condition comparisons, and effect summaries per region / region pair, for all
  pairs and for FDR-selective pairs.
- `summary/*.csv` — the same quantities pooled to scope level, plus the pair
  inventory and zero-lag diagnostics.
- `summary/group_traces_*.pkl` — group-mean lag traces carrying the raw observed
  correlation, each null's level, and the standardised excess, accumulated by
  streaming so the notebook never holds every pair's traces in memory.
- `figures/` — every panel as editable PDF plus high-resolution PNG.
