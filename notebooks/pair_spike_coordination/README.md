# Neural pair spike coordination

Does spike coordination between **simultaneously recorded** neuron pairs change
with fixation condition — interactive face vs non-interactive face vs object —
within a region and across regions?

## Contents

- `pair_spike_coordination.ipynb` — the analysis notebook.
- `_build_notebook.py` — authors the notebook from source strings. Edit this,
  never the `.ipynb`, then re-run it. Per `AGENTS.md` the notebook is a thin
  display layer; the analysis lives in `src/`.

## Running it

Section 1 reports which of the 42 dates are built and, with `SUBMIT_JOBS = True`,
submits a SLURM array for **only the incomplete dates** and waits. Submission is
opt-in: the array costs real cluster time and a notebook cell is easy to re-run
by accident.

```bash
# equivalent from a shell
sbatch --array=0-41 hpc/ephys/run_fixation_pair_spike_coordination.sbatch
python scripts/ephys/analysis/build_fixation_pair_spike_coordination_summary.py

# after editing _build_notebook.py
conda run -n gaze_processing python notebooks/pair_spike_coordination/_build_notebook.py
```

## What is computed

**Per fixation, never per average.** Every cross-correlation is computed on one
fixation's two **unsmoothed 1 ms** spike trains and only then averaged.
Cross-correlating condition-averaged PSTHs would measure shared rate structure,
not coordination; smoothing first would blur the timing this exists to measure.

**Everything inside ±500 ms.** Observed and null alike use the same 1000 bins
around fixation onset. Widening it is not an option: 95% of ±5 s surrounds
contain at least one other analysed fixation (median 5), so outside the window
is not neutral baseline.

**Linear correlation** (zero-padded transform) — the statistic
`scipy.signal.correlate` computes and the one the behavioural
cross-correlations use. At lag L only the `N − |L|` genuinely overlapping bins
contribute, so no spike is paired with one a full window away. The taper this
introduces needs no correction: the null carries it identically.

## One null

The **cross-trial shuffle**: unit A's train on fixation *i* paired with unit B's on
some other fixation. Both units keep their fixation-locked rate profiles and
their exact spike counts; only trial-by-trial covariation is destroyed.

Each draw is a *derangement* of the fixation index, so the null is built from
exactly as many pairings as the observed statistic. Estimating it from all
`F(F−1)` cross-fixation pairings would shrink its standard error and inflate
every comparison.

## The measure

**Coincidences per fixation** — spike pairs at that lag, averaged over
fixations — and `observed − null` as the comparison. No normalisation is
applied: the cross-trial null already carries both units' firing rates and their
exact spike counts, so the difference is rate-controlled by construction.
Dividing by a function of the spike counts on top of that would change the
y-axis and nothing else.

## How results are reported

Per region for within-region pairs, per region pair for cross-region pairs.
**Nothing is averaged across regions** — with these recordings a pooled number
would be a composition of very unequal region contributions rather than a
summary. Each block runs on all pairs, then on pairs where both units are
FDR-selective.

Comparisons are **paired within pair**: the same two neurons, electrodes and
session, differing only in which fixations were used.

## Which comparisons the recordings support

A cross-region pair exists only where both regions were recorded in the **same
session**. Across all 417 sessions:

| within region | pairs | | across regions | pairs |
|---|---|---|---|---|
| BLA | 49,172 | | BLA × dmPFC | 34,190 |
| OFC | 23,820 | | ACCg × BLA | 25,154 |
| ACCg | 19,893 | | BLA × OFC | 24,750 |
| dmPFC | 16,130 | | ACCg × dmPFC | 5,690 |
| | | | dmPFC × OFC | 140 |
| | | | **ACCg × OFC** | **0** |

All four regions support within-region comparisons. Across regions every
well-populated combination involves **BLA**. Combinations below the pair
threshold appear in the tables but are excluded from the figures.

## The zero-lag artifact

Two randomly sampled neurons are essentially never monosynaptically connected,
so a sharp zero-lag peak shared by most pairs on a day is common input —
movement, arousal, or a shared reference/ground artifact. It is a property of
the day and array, not of the pair, so flagged days are **removed from every
result**, not noted at the end.

## Outputs

Under `analysis_output_root/ephys/psth/fixation_pair_spike_coordination/`:

- `date=*/session=*/pair_coordination.pkl` — per-session pair tables with traces
- `summary/*.csv` — inventory, per-region coordination and condition contrasts
  (all pairs and FDR-selective), the count-measure contrasts, zero-lag
  diagnostics, and the dropped dates
- `summary/traces_by_region*.pkl` — group-mean lag traces, streamed
- `figures/` — every panel as editable PDF plus high-resolution PNG
