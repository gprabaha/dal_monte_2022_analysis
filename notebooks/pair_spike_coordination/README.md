# Neural pair spike coordination

Does spike coordination between **simultaneously recorded** neuron pairs change
with fixation condition — interactive face vs non-interactive face vs object —
and does that differ within a region versus across regions?

## Contents

- `pair_spike_coordination.ipynb` — the analysis notebook.
- `_build_notebook.py` — authors the notebook from source strings. Edit this,
  never the `.ipynb`, then re-run it. Per `AGENTS.md` the notebook is a thin
  display layer; the analysis lives in `src/`.

## Regenerating

```bash
# 1. Per-session pair tables (SLURM array over the 42 recording dates)
sbatch --array=0-41 hpc/ephys/run_fixation_pair_spike_coordination.sbatch

# ...or one date locally
python scripts/ephys/analysis/build_fixation_pair_spike_coordination.py --date 01312018

# 2. Aggregate into the tables the notebook reads
python scripts/ephys/analysis/build_fixation_pair_spike_coordination_summary.py

# 3. Rebuild and run the notebook
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

**Circular correlation.** The window is a fixed ±500 ms. Treating it as periodic
makes every lag use all 1000 bins instead of a triangular taper, and makes the
circular-shift null exact: shifting a train by `s` rotates the correlation by
`-s`, so a shift null costs no extra transforms.

**Two nulls, because they answer different questions.**

| null | destroys | keeps | an excess means |
|---|---|---|---|
| trial shuffle | trial-by-trial covariation | each unit's fixation-locked rate profile | the cells co-fluctuate across fixations |
| circular shift | fine temporal alignment | that fixation's count and slow envelope | coordination finer than the shift |

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

## Outputs

Written under `analysis_output_root/ephys/psth/fixation_pair_spike_coordination/`:

- `date=*/session=*/pair_coordination.pkl` — per-session pair tables with traces
  (~2 GB total).
- `summary/*.csv` — inventory, coordination vs null, condition comparisons,
  zero-lag diagnostics.
- `summary/group_z_traces_*.pkl` — group-mean lag traces, accumulated by
  streaming so the notebook never holds every pair's traces in memory.
- `figures/` — every panel as editable PDF plus high-resolution PNG.
