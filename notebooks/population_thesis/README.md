# notebooks/population_thesis

Chapter-style report on the population geometry of fixation type in BLA, ACCg,
dmPFC and OFC. Written to be read start to finish by someone who has not seen
the exploratory work.

| File | Role |
|---|---|
| `population_geometry_main_chapter.ipynb` | **The chapter.** Intro / Methods / Results / Discussion with five main figures, then an appendix on the population subspaces. The five main figures are sized and ordered to be stitched into one composite paper figure. |
| `population_geometry_chapter.ipynb` | Long-form version: every method, verification and control, 13 figures. Use it to look something up; use the main chapter to write from. |
| `_build_main_chapter.py`, `_build_chapter.py` | Author each notebook from plain-Python source strings. |

## The five main figures

| Figure | Question | Statistic |
|---|---|---|
| 1 | What kind of signal distinguishes the three fixation types? | variance shares, pooled over all 1,201 neurons, neuron-bootstrap CIs, FDR contrasts |
| 2 | How many dimensions does the analysis need? | cumulative variance, per-region thresholds |
| 3 | What do the trajectories look like? | — |
| 4 | Which axis carries which distinction? | mean abs. difference per component, fixed-axis neuron bootstrap, FDR contrasts |
| 5 | Do the conditions move together? | deviation correlation with exact circular-shift null |

Appendix A1–A4: cross-condition variance explained (rows = fitted condition,
columns = region), alignment matrix, principal angle spectra, and the summary
with neuron-subsampling intervals and FDR contrasts; plus the controls.

The exploratory notebooks it draws on are in
[`../population_pc_subspaces/`](../population_pc_subspaces) — those carry the full
method inventory, every verification check, and the pair-selective-unit variant.
This chapter reports the subset that tells the story.

## The argument

1. **The separation is overwhelmingly a static offset.** Pooled over all 1,201
   neurons the condition offset carries 47% of the raw variance and **83% after
   removing the SEM-derived noise energy**, against 8–9% for the condition × time
   term (both contrasts p = 0.007). The trajectories are compact clouds far apart
   relative to their own size.
2. **Individual axes carry specific distinctions.** In BLA, ACCg and OFC, PC1
   separates faces from object while barely separating the two faces, and PC2 does
   the reverse — a factorised code, not one graded salience dimension. 18 of 36
   pairwise contrasts survive FDR. dmPFC does not follow this arrangement.
3. **Time-locked co-movement separates faces from objects decisively.** The two
   face conditions move through their neighbourhoods in step (z = +10.1 BLA,
   +4.7 ACCg, +3.0 OFC); face–object pairs are indistinguishable from
   circularly-shifted controls. dmPFC is the exception.
4. **Subspace overlap agrees on the ordering but not the size.** The face pair is
   the most-aligned pair in 4/4 regions (8/8 contrasts significant) but leads by
   only ~0.06 on a 0–1 scale. The dissociation — same ordering, wholly different
   effect size — is the central result: face and object have comparable *access*
   to similar directions and differ in whether a common process drives them along
   those directions at the same moments.
5. **The interactive-face state is compact and low-dimensional** in 4/4 regions,
   after correcting for the 5:1 trial-count imbalance.
6. **dmPFC is a different regime** throughout: largest offset share, highest
   alignment for every pair, the only area where face–object pairs co-move, a
   different axis arrangement, and the least reliable geometry under split-half
   resampling.

## Statistics used

| Quantity | Resampling unit | Why that one |
|---|---|---|
| Variance shares | neurons, with replacement | the share is a population quantity; neurons are the exchangeable units |
| Per-axis separation | neurons, with replacement, **axes held fixed** | refitting per resample lets similar-variance components rotate into one another, so "PC1" would not name the same axis twice |
| Co-movement | circular time shifts (exact, all 81) | preserves each trajectory's own smoothness and destroys only their alignment in time; two smooth curves correlate by accident |
| Subspace overlap | neurons, **subsampled without replacement** | duplicated neurons are perfectly correlated, which lowers covariance rank and inflates every overlap measure |

Time bins are never a resampling unit: 20 ms smoothing makes them heavily
autocorrelated. Subspace intervals are reverse-percentile, because subsampling
shifts those metrics systematically.

## Two things the chapter explains carefully

**The three-part response split, without tensor notation.** Figure 1 works it
through on a single neuron: average the three condition curves together (shared
time course), take each curve's own level (condition offset — the centroid), and
look at what is left (condition × time). The three parts add back to the observed
curves exactly, which is what lets their variances be quoted as percentages
summing to 100%.

**What time-locked co-movement measures, and why it needs its own null.**
Figure 8 shows the computation for a face–face pair and a face–object pair side
by side: the centroid-removed trajectories, the per-bin dot product whose average
*is* the correlation, and the observed value against all 81 circular time shifts.
The intuition is two dancers — subspace overlap asks whether they know the same
moves, co-movement asks whether they are dancing in sync. The circular-shift null
is needed because two smooth trajectories correlate substantially by accident.

## Regenerating

```bash
for name in main_chapter chapter; do
  conda run -n gaze_processing python notebooks/population_thesis/_build_${name/main_chapter/main_chapter}.py
done

conda run -n gaze_processing python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=3600 \
    notebooks/population_thesis/population_geometry_main_chapter.ipynb
```

Each takes 80–90 seconds. Figures (editable PDF + PNG) and tables go to
`analysis_output_root/ephys/psth/fixation_population_pc_subspace/main_chapter/`
and `.../chapter/`.

Every number in the prose is computed in the cell above it, so the text stays
correct when the analysis is rerun. All reusable code is in `src/` per
`AGENTS.md`; the notebook only orchestrates and displays.

## Caveats carried in the Discussion

- Object fixations are **not matched for interactive state** (pooled across both),
  so part of the face–object difference could be context mixing.
- **No cross-validated effect sizes**: trial averaging removed the repeats a
  crossnobis-style distance would need.
- **Nothing about fine-timescale dynamics**: the 20 ms smoothing correlates the
  noise at the scale a 10 ms derivative probes, and measured step energy is near
  the noise floor. No speed or rotation claims.
- Split-half distance *ordering* is stable in BLA/ACCg/OFC but not dmPFC.
