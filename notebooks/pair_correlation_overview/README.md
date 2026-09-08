# Mean signal and per-trial spike correlation in simultaneously recorded selective pairs

Written as a **thesis chapter**: introduction, a methods section with the full
equations, results and discussion, with the figures inline. Figures are also
written as editable PDFs for stitching into a paper figure.

Every analysed pair is **two units recorded simultaneously, both selective** —
each significant for at least one of the three pairwise fixation-type contrasts
after multiple-comparison correction. That rule is applied twice per pair, so it
costs roughly its own square: BLA has 53% of its units selective and keeps 38% of
its pairs, and across groups 20–38% of recorded pairs survive. Figures 2 and 6
show this. Two things are then measured on those same pairs, so they can be compared
pair for pair.

| | per-trial spike correlation | mean signal correlation |
|---|---|---|
| computed on | per-fixation 1 ms spike trains | condition-averaged rate timelines |
| asks | do they fire together **within** a fixation | do their **mean responses** share a shape |
| null | circular shift within fixation, 50 draws | unit of the same region, different session, 20 draws |
| summarised as | mean excess over ±250 ms | mean null-corrected ρ over ±250 ms |
| trial-count matched | yes | no — see caveat |

Averaging over fixations removes trial-by-trial co-firing entirely. That is the
only difference between the two, and it is why a pair can have either without the
other.

## Why not "noise correlation"

The second measure is conventionally called noise correlation and that name is
deliberately not used. Classical noise correlation is **one number per pair**:
the Pearson correlation, across trials, of the two units' spike *counts* after
each unit's condition mean is removed. What is computed here is a **full
cross-correlogram within each fixation**, averaged afterwards — it resolves
timing that a count correlation integrates away, and being unnormalised it does
not live on [−1, 1]. Reporting it under the other name would invite comparing
magnitudes against a literature that measured something else.

## Naming

The two measures are named after the operation that defines them and the names are
used verbatim throughout: **mean signal correlation** (average the fixations, then
correlate) and **per-trial spike correlation** (correlate within each fixation,
then average). Every figure title, axis label and paragraph uses one of those two.

## Structure

Nine numbered figures, each with a caption below it in the notebook. The
within-region and cross-region results are two self-contained sections: each opens
with its own denominators, so neither has to be read against the other.

1. Method schematic — one set of trials, two orders of operation

**Within region**

2. Donuts: pairs recorded per region and the fraction analysed
3. **Mean signal**: null-corrected correlation across lags
4. **Per-trial**: observed against null, every region and fixation type
5. **Both measures side by side**, each reduced to a ±250 ms mean minus its null,
   with the paired contrasts marked

**Across regions**

6–9. The same four figures, for BLA × ACCg, BLA × dmPFC and BLA × OFC

The Spearman correlation relating the two measures is reported as a table, not a
figure: the coefficients are positive in some region-and-condition combinations
and absent or negative in others, and bars would give a scattered set of values
the visual weight of a result.

## Why the two y-axes are not comparable

Mean signal correlation is a Pearson coefficient, bounded in [−1, 1]. The
per-trial measure is **spike pairs per fixation**: at each 1 ms lag, the number of spike
pairs separated by that lag. Chance is roughly `rate₁ × rate₂ × bin width` —
about 0.05 for two 7 Hz units — which is why the observed traces sit where they
do, and why the null-subtracted excess is around 10⁻³ and the bars are drawn
×10⁻³. They are different units and only their *ranks* are compared, in the
Spearman table. Figures 5 and 9 put the two on adjacent axes precisely because
only the *pattern across fixation types* is comparable, not the magnitudes.

**Every bar panel is scaled to its own decade and names it in the axis label**
(`, ×10⁻³`). The panels cannot share an axis — a shared one would flatten the
cross-region bars to a line at zero — so the decade in the label is what makes
magnitudes comparable *between* figures: the per-trial panel reads ×10⁻³ in
Figure 5 and ×10⁻⁴ in Figure 9, and the ten-fold drop is a symbol rather than a
count of leading zeros.

## Two lag windows, two questions

- **±250 ms** is the reporting window, matching the signal correlation's, and it
  answers *how much excess co-firing does this condition carry in total*.
- **±10 ms** is used only for whether **one pair** is individually coordinated: a
  monosynaptic or common-input peak lives inside ±10 ms, and averaging z over
  half a second of empty lags dilutes it to nothing. Run at ±250 ms, zero pairs
  are individually significant anywhere — a statement about the window, not the
  pairs.

## The trial-count caveat

Interactive-face fixations outnumber the others about six to one, so
interactive-face estimates are more precise and correlate better with anything.
Every **per-trial** condition contrast runs on the trial-count-matched
recomputation and does not carry this. The mean-signal contrasts cannot be
matched, because no matched average exists, so their absolute sizes are upper
bounds. `notebooks/signal_correlation/` has the stratification that bounds it
directly.

## Reading the statistics

With thousands of pairs per region almost any difference reaches significance,
and a session with 20 simultaneous units contributes 190 non-independent pairs.
**Read the rank-biserial effect sizes, not the asterisks.**

## Rebuilding

```
python _build_notebook.py          # authors the .ipynb from source
jupyter nbconvert --to notebook --execute --inplace pair_correlation_overview.ipynb
```

The figures land outside the repo, under
`<analysis_outputs>/ephys/psth/fixation_pair_spike_coordination/overview/`, and
are overwritten on every run.
