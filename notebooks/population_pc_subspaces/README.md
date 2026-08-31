# notebooks/population_pc_subspaces

Two notebooks on the neural population PC geometry of the three fixation
conditions (`face_interactive`, `face_non_interactive`, `object`) in BLA, ACCg,
dmPFC and OFC.

| File | Role |
|---|---|
| `population_pc_subspaces_all_units.ipynb` | The reference analysis, on every unit with all three conditions. |
| `population_pc_subspaces_pair_selective_units.ipynb` | The same cells restricted to fixation-pair selective units. A sensitivity check, not an independent confirmation — see the caveat below. |
| `_build_notebooks.py` | Authors both notebooks from plain-Python source strings, so they cannot drift apart. |

## Input, and a provenance discrepancy

Both notebooks read
`fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl` — firing rate in
Hz, converted per trial before averaging, Gaussian-smoothed at σ = 20 ms. This
is the file the mRNN targets and the fixation selectivity analysis also use.

The stored `ephys/psth/fixation_population_pca/results.pkl` reads a **different,
older** export (`fixations_split_by_interactive_state.pkl`, written before the
current fixation-detection pass). The two select the same units but the older
one carries roughly 25–35% fewer fixations per unit, and it is also capped at 50
components, so for regions needing more than that its cumulative variance curve
never reaches 95%. Section 3c of each notebook quantifies the gap. **Reconciling
`configs/ephys_fixation_psth.yaml:population_pca_input_filename` with the 10 ms
export is outstanding work.**

## Choosing the retained dimension

Visualisation uses 3 PCs; every quantitative result uses the full retained set.

**42 PCs** reach 95% of the concatenated-fit variance in all four regions
(BLA 42, ACCg 40, OFC 38, dmPFC 33 — BLA is binding). This reproduces the
`pca_n_components` the mRNN target builder derives with the same max-over-regions
rule, so both analyses work in comparably sized spaces. On the pair-selective
subset the same rule gives **34**.

Note that these counts come from the *concatenated* fit, which spans three
conditions of dynamics. A single condition's own fit needs about half as many.

## What the notebooks check

Concatenating conditions along time and slicing the projection back apart is the
step most likely to fail silently, so it is asserted rather than assumed:

- Seven numerical identities per region (orthonormality, centring, slice
  round-trip, reconstruction, energy conservation, no one-bin time shift).
- A **label** check the identities cannot give: each condition's PC trajectory is
  back-projected into firing-rate space and scored against *every* condition's
  true rates. The winning match must sit on the diagonal for all three.

## The measures, and which question each answers

The notebook gives each of these its full definition; this is the map.

| Measure | Asks | Sees offsets? |
|---|---|---|
| Cross-condition variance explained | how much of B survives projection onto A's PCs | no |
| Alignment index | ...relative to the *best possible* k-dim capture of B | no |
| Principal angles | do the *directions* differ, regardless of amplitude | no |
| Offset / dynamics split | how much of the separation is static vs moving | both, separately |
| Deviation correlation | do the conditions move **at the same time**? | no |
| Trajectory geometry | centroid distance, state distance, Procrustes shape | yes |

**Per-condition PCA centres on each condition's own mean**, so the alignment
index and the principal angles are already blind to the static offsets — they
are covariance-structure measures, not positional ones. §7d checks this
numerically rather than asserting it.

### The floor is k/N

A random k-dimensional subspace of R^N captures a fraction k/N of any
covariance's trace in expectation, and the simulated null matches that to three
decimals. At k = 42 the floor is 42/537 = 0.08 in BLA but 42/187 = 0.22 in
dmPFC, so **raw alignment indices must not be compared across regions.** Use
`alignment_above_floor` = (A − k/N)/(1 − k/N), which rescales chance to zero.

The upper reference is a within-condition ceiling from an odd/even time-bin
split. It is optimistic — smoothing makes adjacent bins dependent — so it bounds
how high alignment could plausibly go rather than estimating noise.

## What the statistics can and cannot support

There is **no unbiased cross-validated distance** available here. Cross-validated
(crossnobis-style) distances need independent repeats of the same measurement,
and trial averaging removed them. In its place:

- **Condition-label shuffle** (permuted within each unit). Note the direction:
  this does *not* test whether conditions are separated — a shuffled population
  has *more* dissimilar subspaces, because the real conditions share a large
  common fixation-locked response. It tests whether the observed *shared*
  subspace is real. Both tail probabilities are reported.
- **Unit bootstrap** for the separation bands — an interval, not a test.
- **Split-half reliability** over units: the strongest internal-validity check
  available from condition-averaged data.

## Headline results (all-unit notebook)

**1. The separation is mostly positional.** A two-way (condition × time) ANOVA
of the population tensor splits into three orthogonal terms summing to 100%:
shared time course 18–21%, **static condition offset 45–54%**, condition × time
28–36%. 55–75% of the squared separation between any pair is offset. The clean
clustering in the 3D figure is a statement about *where* the population sits,
not about different trajectories.

Removing the offset is exactly centroid subtraction — each condition's own
time-average, per neuron. Figure 8 refits after removing both marginals so the
interaction dynamics are visible alone.

**2. The two face conditions share a time-locked dynamic that neither shares
with object.** This is the sharpest result and it does *not* follow from the
subspace measures. Floor-corrected alignment ranks IF/NF highest in all four
regions but only narrowly (e.g. BLA 0.196 vs 0.151 and 0.109). The deviation
correlation — do they move *at the same moments*? — separates them decisively:

| Region | IF/NF | IF/OB | NF/OB |
|---|---|---|---|
| BLA | z = +10.1 * | +0.5 | +1.1 |
| ACCg | z = +4.7 * | −0.3 | −1.7 |
| OFC | z = +3.0 * | −0.1 | −0.2 |
| dmPFC | z = +2.0 | +2.5 * | +1.7 * |

(circular-shift null, * = p < 0.05). Figure 15 plots alignment against
co-movement and shows the pairs are not ordered by the alignment axis at all.

**3. The interactive-face state is more compact and lower-dimensional.** After
correcting for the 5:1 trial imbalance its excursion is roughly half that of the
other conditions in every region and its dynamics participation ratio is lower.

**dmPFC is the exception throughout** — highest floor-corrected alignment for
every pair, and the only region where face-vs-object pairs also co-move. A
different regime, not a weaker version of the same one.

## The trial-count confound, and how it is handled

Interactive-face fixations outnumber the other two roughly **5:1** (median ~850
vs ~168 per unit), and a noisier average has more apparent excursion and speed.
Every dynamics quantity is therefore noise-corrected using the stored `psth_sem`:
the noise covariance in the PC basis is V' diag(s²) V, and its energy is
subtracted from the excursion, the offset and the temporal covariance.

The 20 ms smoothing makes that noise **autocorrelated**, with
ρ(ℓ) = exp(−ℓ²/4σ²) at σ = 2 bins. Treating it as independent overstates the
noise in a first difference by more than an order of magnitude and drives every
corrected speed to zero regardless of the data. With the kernel-aware model,
`speed_over_noise` is near 1 in most cases: **bin-to-bin speed is not resolvable
and should not be quoted.** Excursion and dimensionality integrate over the whole
window and survive correction comfortably.

For the deviation correlation the correction is a disattenuation of the
denominator only — the numerator is unbiased because noise is independent across
conditions. Note the direction: the uncorrected value is biased *toward zero*,
hardest for the conditions with fewest trials, so skipping it would make a real
shared dynamic look like an absence of one.

## A note on the stored `explained_variance_fraction` column

Its own metadata names it `projection_variance_fraction_within_retained_pcs`: it
divides by variance *inside* the retained PCs, so it approaches 1.0 for every
condition pairing once enough PCs are kept. §7f shows both normalisations side
by side.

## Selective-unit caveat

Selecting units by a condition contrast and then measuring condition separation is
circular by construction, and the selective subset does show higher alignment and
larger normalised separation. What that notebook can legitimately answer is
whether the geometry is *carried by* the selective minority or distributed across
the whole population — not whether the separation is real.

## Regenerating

All reusable code lives in `src/` per `AGENTS.md`; the notebooks only orchestrate
and display.

```bash
conda run -n gaze_processing python notebooks/population_pc_subspaces/_build_notebooks.py

for nb in all_units pair_selective_units; do
  conda run -n gaze_processing python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=3600 \
    notebooks/population_pc_subspaces/population_pc_subspaces_${nb}.ipynb
done
```

Each notebook takes about 1–2 minutes. Figures (editable PDF + PNG) and tables
are written to
`analysis_output_root/ephys/psth/fixation_population_pc_subspace/{all,pair_selective}_units/`.

## Source modules

| Module | Contents |
|---|---|
| `ephys/analysis/fixation_population_pc_subspace.py` | Tensor assembly (with SEMs), full-rank PCA, verification suite, ANOVA marginalisation, alignment / principal-angle / trajectory metrics, the offset–dynamics split, the SEM noise model, nulls and resampling. |
| `ephys/plotting/fixation_population_pc_subspace.py` | All figures, including the per-region 3D viewing-angle search. |
| `ephys/plotting/thesis_common.py` | Shared style, region/condition palette and labels. |
| `ephys/modeling/fixation_mrnn_bridge.py` | Reused to load and reshape the combined export, which keeps this analysis on the mRNN's unit set. |

## Relationship to `fixation_population_pca`

The older builder stays as-is; it produces the manuscript trajectory and pairwise
geometry figures. This module differs in three ways: canonical 10 ms input,
full-rank fits (so the retained dimension is measured rather than clipped by
`population_pca_max_components`), and an explicit verification suite.
