# notebooks/mrnn_chapter

The final mRNN series: three notebooks, fitted on the minimax objective and scored on the
ceiling-matched R². Runs live under `analysis_output_root/ephys/modeling/fixation_mrnn/final/`,
never mixed with the `chapter/` rebuild — the two are not on the same objective and their
numbers are not comparable.

| | Notebook | Question | Runs |
|---|---|---|---|
| 01 | `01_ladder.ipynb` | Does a region need the network to reproduce itself? Singles, pairs, triples, full. | 14 × 5 + full × 10 = **80** |
| 02 | `02_rank_grid.ipynb` | Which is the low-dimensional channel — self-recurrence or inter-regional input? Within × cross rank grid plus both marginals. | 35 × 5 = **175** |
| 03 | `03_ensemble.ipynb` | What do ten fits of the most constrained adequate model agree on? | 10 (after 02) |

Task 01 writes `final/base_model.yaml`; 02 and 03 read it. Task 02 writes
`selected_bottleneck.yaml`; 03 reads it and refuses to queue without it.

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

Regenerate a notebook with its `_build_*.py`, then

```bash
conda run -n gaze_processing python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=2400 notebooks/mrnn_chapter/<name>.ipynb
```
