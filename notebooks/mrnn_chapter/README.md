# notebooks/mrnn_chapter

The final mRNN series: three notebooks, fitted on the minimax objective and scored on the
ceiling-matched R². Runs live under `analysis_output_root/ephys/modeling/fixation_mrnn/final/`,
never mixed with the `chapter/` rebuild — the two are not on the same objective and their
numbers are not comparable.

| | Notebook | Question | Runs |
|---|---|---|---|
| 01 | `01_ladder.ipynb` | Does a region need the network to reproduce itself? Singles, pairs, triples, full. | 14 × 5 + full × 10 = **80** |
| 02 | `02_rank_grid.ipynb` | Which is the low-dimensional channel — self-recurrence or inter-regional input? Within × cross rank grid plus both marginals. | 35 × 5 = **175** |
| 02b | `02b_bottleneck_properties.ipynb` | Which bottleneck costs fit, and does it change which fixation type is hardest? Cost per fixation type along each marginal, regions pooled, every seed shown; where a region's drive comes from; supplementary network properties. | none (reads 01 and 02) |
| 03 | `03_ensemble.ipynb` | The selected model fitted ten times against the ten dense fits: what the constraint cost each fixation type, whether the two faces are alike (data, state, drives, lesion profile, fixed points), drive and flow, dynamics (fixed points, flow fields, local linearisation), lesions (pairs, per-condition ranking, lesioned dynamics), and a consistency scoreboard. | 10 (done) |
| 04 | `04_chapter.ipynb` | **The chapter.** Introduction · Methods · Results · Discussion, nine composite figures with statistics: schematics (including how the ceiling is measured), the ladder in one region's traces, within- vs inter-regional connections, the rank grid, the cost of every constraint to every fixation type, pair lesions and what they do to the local dynamics (eigenvalue spectra, Wasserstein distance), and the dynamical portrait of interactive face. Reads 01–03; fits nothing. | none |

Task 01 writes `final/base_model.yaml`; 02, 02b and 03 read it. Task 02b caches its replay tables under `final/02b_bottleneck_properties/tables/` — delete them to recompute. Task 02 writes
`selected_bottleneck.yaml`; 03 reads it and refuses to queue without it. Task 04 caches its
own tables under `final/04_chapter/tables/` (ladder traces and component gains for the trace
figure; the lesion spectra summary over all twenty fits, which takes about a quarter of an
hour to recompute, and the raw eigenvalues of the two representative fits) and writes its
figures to `plots/final_04_chapter/`; its code is
`src/.../ephys/{analysis,plotting}/fixation_mrnn_chapter.py`.

Chapter figure conventions (set by the author's review of the first draft): groups of five to
ten fits are box plots with every fit a dot, larger groups are violins; no horizontal grid
lines, only reference values (zero, chance, the dense network, the selection bar where a
selection is made); comparisons are paired t within an arm, Welch's t between arms, the
"widens the gap" interaction as Welch's t on per-fit gaps against the dense fits, Holm
within each panel, and **only significant comparisons are marked**. "Cell" is avoided
because the units are neurons: a region × fixation-type block is a *combination*, a point
of the rank grid a *configuration*.

## What changed from the rebuild, and why

| | rebuild (`chapter/`) | here (`final/`) |
|---|---|---|
| objective | weighted sum, balanced condition weighting | **minimax** over the 12 region × condition cells: every cell scored by its unexplained fraction, the worst optimised. Conditions degrade together; no region dominates by unit count. Balanced weighting becomes redundant. |
| R² centring | grand mean of the flattened (time × PC) block per condition | **per component over pooled condition × time** — the centring the noise ceiling uses |
| within-region rank | not available | `within_region_bottleneck_dim` — L·R factorisation of each region's own block |
| subset regions | not supported (raised) | `region_order` may name any subset; PCA per region and a normalisation scale pooled over all four recorded regions keep a region's target bit-identical at every rung |
| sparsity mask | tied to the run seed | `sparsity_seed` pinnable (not used in this series) |

The convergence check behind the objective change (2000 iterations, compressed cosine):
same mean unexplained fraction under sum and minimax (0.427 / 0.425), worst cell 0.635 under
the sum against **0.442** under minimax, cells within 0.017 of one another instead of 0.21.

## Lessons carried from the rebuild's audit (`../mrnn_thesis/06`, `07`)

- Score fit against the noise ceiling on the worst cell, not the mean.
- Any agreement measure needs its untrained-architecture floor; report distances as well
  as similarities; exclude within-region self-drive from "inter-regional" measures.
- Per-condition damage in absolute squared error, never per-condition R².
- Lesions need a matched random-weight control.
- The claim to make about an ensemble is at the highest level of description that clears
  its null: weights, spectra, state geometry, pathway ranking, condition contrast.

Task 03's analyses live in `src/.../ephys/analysis/fixation_mrnn_ensemble.py` and its figures in `src/.../ephys/plotting/fixation_mrnn_ensemble.py`; its tables are cached under `final/03_ensemble/tables/` (delete a file to recompute it).

Regenerate a notebook with its `_build_*.py`, then

```bash
conda run -n gaze_processing python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=2400 notebooks/mrnn_chapter/<name>.ipynb
```
