# notebooks/mrnn_thesis

The multi-regional RNN (mRNN) analysis: the audit of the legacy fits, and the rebuild.

## The rebuild

Everything the chapter will rest on is being refitted from scratch. The legacy tree is
kept as a guide to *what to do*, not as a source of results — its families differ in
learning rate, iteration count, activation and loss weights, so most of its comparisons
are not comparable, and none of its checkpoints is the best iterate its run reached.

Rebuilt runs live under `analysis_output_root/ephys/modeling/fixation_mrnn/chapter/<task>/`,
never mixed with the legacy `scratch/` tree. One notebook per task; each is a control
surface, not a script — it shows its design, the state of every run, and submits cluster
work only when you set `SUBMIT = True`.

| | Notebook | Question | Status |
|---|---|---|---|
| 00 | `00_training_protocol.ipynb` | What optimiser settings converge reliably? | **built**; sweep not yet submitted |
| 01 | `01_target_and_loss.ipynb` | How many PCs, weighted how; derivative and curvature weights | to build |
| 02 | `02_capacity.ipynb` | Units per region, at fixed settings | to build |
| 03 | `03_architecture.ipynb` | full vs cross+diagonal vs within-region, ensembled and capacity-matched | to build |
| 04 | `04_bottleneck_rank.ipynb` | Ranks 1–30, epoch-matched end to end | to build |
| 05 | `05_seed_ensembles.ipynb` | 100 seeds — does stable fitting change the reproducibility result? | to build |
| 06 | `06_invariants.ipynb` | Condition contrast; channel identity | to build |
| 07 | `07_dynamics.ipynb` | Fixed points, Jacobians, basins | to build |
| 08 | `08_generalization.ipynb` | Held-out post-fixation half | to build |
| 09 | `09_chapter.ipynb` | Collated report | to build |

The order is forced by what feeds what: 00 fixes the recipe, 01–02 fix the target and
size, 03–04 are the structural claims, and 05 onwards are the science.

## The audit of the legacy fits

| File | Role |
|---|---|
| `mrnn_synthesis.ipynb` | Reads all 273 legacy fits, scores them on one footing, and separates what they establish from what they only suggest. This is the motivation for the rebuild and the record of what is *not* being carried forward. |
| `_build_summary.py` | Authors it. |

Regenerate any notebook with its `_build_*.py`, then:

```bash
conda run -n gaze_processing python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=2400 notebooks/mrnn_thesis/<name>.ipynb
```

All reusable code is in `src/` per `AGENTS.md` —
`ephys/analysis/fixation_mrnn_{synthesis,protocol}.py` and the matching
`ephys/plotting/` modules. Every number in the prose is computed in the cell above it.

The exploratory notebooks this draws on are the loose `fixation_mrnn_*.ipynb` files
in [`../`](..).

---

## 1. What is being modelled

| | |
|---|---|
| Data | `fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl` — the same export the population-geometry chapter uses |
| Regions | ofc (241 units), bla (537), dmpfc (187), accg (236) |
| Conditions | `face_interactive`, `face_non_interactive`, `object` |
| Target | per-region PC scores, 42 PCs/region (smallest shared count reaching 95% variance in every region; per-region requirement is 38/42/33/40) |
| Tensor | 3 conditions × 100 bins (10 ms, −500…+500 ms) × 4 regions × 42 PCs = **50,400 numbers** |
| Model | Elman mRNN, 4 region blocks of 50 units, condition one-hot input only (no temporal basis), trained initial state per condition |
| Objective | pointwise PC MSE + first-difference + second-difference, optional PC→FR backprojection terms, optional within-region L1 |

## 2. Run inventory

60 scratch runs under
`analysis_output_root/ephys/modeling/fixation_mrnn/scratch/`.

| Family | Runs | What it settled |
|---|---|---|
| `smoke_*` | 6 | pipeline; condition-only vs 20 Gaussian temporal-basis channels |
| `hidden_unit_sweep_*` | 8 | h = 5,10,20,25,30,40,50,60 |
| `loss_sweep_*` | 12 | 4 loss mixes × {h40,h50} × 3 LRs |
| `loss_necessity_*` | 8 | derivative/curvature weight pairs at 50k / 100k / 200k iters |
| `cca_42pc_tanh_*` | 6 | 3 connectivity architectures × {h40, h50}, 50k iters, **no bottleneck** |
| `bottleneck_dim_*` | 8 | ranks 1,2,3,4,5 (100k iters) and 5,10,20,30 (50k iters) |
| `interregional_current_*` | 2 | the single model used for the current-geometry analysis |
| `ensemble_d5_c3_50u_100k_100init*` | 3 × 100 | **100 seeds**, unbottlenecked, d=5/c=3, 100k iters |
| `bottleneck1d_l1within_50u_100k_10init` | 10 | rank-1 bottleneck ensemble, 100k iters |
| `bottleneck3d_l1within_50u_250k_10init` | 10 | rank-3 bottleneck ensemble, 250k iters |

**These families are not mutually comparable.** They differ in iteration count
(20k → 250k), learning rate, activation (softplus vs tanh), L1 (0 vs 0.01),
derivative/curvature weights, and parameterization. Any cross-family plot has to
carry those differences on its face or it is not a comparison.

---

## 3. Established results

### R1. The hidden-unit sweep is confounded by learning rate — the capacity question is open
The sweep lowered the learning rate as the models got wider (1e-3 for 5–25 units, 3e-4
at 30, 1e-4 at 40–60), so width and step size vary together. At 50 units and 20,000
iterations — width and training length both fixed — a learning rate of 1e-4 reaches mean
PC R² **0.815** while 1e-3 reaches **0.985** (n = 4, range 0.983–0.987). **The entire
apparent capacity effect is the learning rate.**

What the fits do establish is that 50 units/region is *sufficient*. Whether 20 or 30
would also do has never been tested at matched settings (G5).

Related: 4 of the 34 50-unit runs in the tree diverged, at learning rates 1e-3, 2e-3 and
1e-2. 1e-3 is a working band, not a safe one.

### R2. Inter-regional coupling is required; dense within-region coupling is not
At 50 units, 50k iterations, one seed each (`model_selection` notebook, Part 1):

| architecture | pooled PC R² |
|---|---|
| `full` | 0.9954 |
| `cross_region_with_self_diagonal` | 0.9926 |
| `within_region` only | 0.9500 |

All 12 region × condition cells order identically (three Wilcoxon comparisons at the
exact test floor, W = 0, n = 12, Holm p = 0.0015). Removing cross-region blocks costs
~0.045 mean R²; reducing within-region coupling to a **diagonal** costs ~0.003. The
interesting half of this is the second: regions barely need dense internal recurrence
as long as they can talk to each other.

### R3. The fitted solution is not decomposable, and it is fragile
Lesioning the trained `full` model: deleting *any one* of the 16 blocks drops pooled
R² from 0.998 to ≈ −2, and deleting all 12 cross blocks is no worse than deleting one
— saturated, so it cannot rank pathways. Graded attenuation is the usable measure: a
**5% attenuation already costs several R² points**, and 20% destroys the fit. At 10%
attenuation the four within-region blocks are individually more load-bearing than the
twelve inter-regional ones (Cliff's δ ≈ −1, clean separation, but 16 perturbations of
one seed — descriptive, not inferential).

### R4. Unconstrained cross-region blocks stay full rank, but that is slack
Every fitted inter-regional block in the `full` model is numerically full rank and
spreads 90% of its energy over ~half its singular directions. Yet an explicitly
rank-constrained fit reaches the same quality at rank 3 (R5). So the broad spectrum
is what an unpenalized optimizer produces, not what the function needs.

### R5. A rank-3 inter-regional bottleneck is free
Epoch-matched block (ranks 1–5, 100k iters, `l1_weight_scale = 0.01`), per-region PC R²:

| rank | ofc | bla | dmpfc | accg |
|---|---|---|---|---|
| 1 | 0.973 | 0.986 | 0.957 | 0.947 |
| 2 | 0.973 | 0.985 | 0.959 | 0.948 |
| **3** | **0.981** | **0.989** | **0.972** | **0.968** |
| 4 | 0.980 | 0.989 | 0.973 | 0.965 |
| 5 | 0.982 | 0.989 | 0.976 | 0.970 |

Backprojected firing-rate R² is ≥ 0.999 at every rank. Rank 3 is the knee; 1→3 buys
~0.02, 3→5 buys ~0.002. **All four cross-region blocks, all 12 pathways, carry the
whole story through 3 dimensions each.**

The ranks 10/20/30 runs were trained for 50k iterations against 100k for ranks 1–5, so
the apparent collapse at rank 20 (PC R² 0.77–0.83) is a training-length artifact and
those three points are **not currently readable**.

### R6. Fits are reproducible; the circuit read off them is not
This is the problem the chapter has to confront.

100 seeds, unbottlenecked, 100k iters: **25/100 flagged bad** (divergence or failure to
improve). Among the 75 healthy runs the best loss still spans a factor of 5 at identical
settings.

Separately, across the whole tree **89 of 273 saved checkpoints (33%) sit more than 1.5×
above the lowest loss their own run reached** — the training loop saves the final
iterate, not the best one (G2).

Cross-initialization consistency (mean off-diagonal correlation over inits; 1 = identical
dynamics), 10 seeds each:

| measure | rank-1 bottleneck (100k) | rank-3 bottleneck (250k) |
|---|---|---|
| inter-regional current magnitude | 0.20 | **0.60** |
| relative source contribution | 0.08 | **0.35** |
| latent geometry (drive RDM / RSA) | 0.35 | 0.41 |
| — RSA ofc / bla / dmpfc / accg | 0.36 / 0.44 / 0.30 / 0.30 | 0.42 / 0.43 / 0.42 / 0.37 |

The bottleneck **raises** consistency substantially — but the two ensembles also differ
in iteration count (100k vs 250k), so **the comparison is confounded and cannot be
reported as it stands** (see G1). Even at its best, 0.35–0.60 means the circuit
description is not a property of the data.

### R7. The invariants that *do* survive the ensemble
From the 100-seed ensemble's saved per-seed current tables (75 healthy runs), the
time-averaged relative source contribution to each region's drive direction is
**≈ 0.25 for every source** — the model spreads drive uniformly across the four
sources. That near-uniformity is itself the first invariant.

The systematic deviations from it are the second, and they are condition-specific.
Paired across seeds (Wilcoxon, n = 75):

| source | interactive face | non-interactive face | object | FI vs FN | FI vs OBJ |
|---|---|---|---|---|---|
| ofc | 0.251 | 0.227 | 0.239 | **p = 1.3e−3** | 0.21 |
| bla | 0.247 | 0.281 | 0.281 | **p = 1.6e−7** | **p = 1.2e−4** |
| dmpfc | 0.242 | 0.228 | 0.227 | p = 0.030 | 0.090 |
| accg | 0.259 | 0.264 | 0.253 | 0.36 | 0.41 |

**BLA's share of the network drive falls during interactive face fixations**, and it
falls into *every* target region (bla→ofc p = 1.1e−7, bla→bla 2.4e−7, bla→dmpfc 7.0e−5,
bla→accg 1.2e−5). OFC's share rises. Non-interactive face and object are
indistinguishable from each other on every source (all p > 0.5) — interactive face is
the odd condition out.

*(Computed 2026-09-02 from the stored `within_current_projection_by_seed.pkl`; the
100-seed ensemble notebook saves the mean ± SEM but does not run this contrast.)*

### R8. Interactive face fixations are what inter-regional coupling is *for*
Per-condition median R² from the architecture comparison:

| condition | full | cross+self-diag | within-region only | cost of dropping cross-region |
|---|---|---|---|---|
| face_interactive | 0.9905 | 0.9864 | 0.9334 | **0.056** |
| face_non_interactive | 0.9980 | 0.9971 | 0.9872 | 0.011 |
| object | 0.9982 | 0.9953 | 0.9889 | 0.009 |

Removing inter-regional coupling costs **5–6× more for interactive face fixations**
than for either other condition, in every region. Interactive face is also the hardest
condition for the intact model. Single seed — this needs the ensemble before it can be
claimed (G3).

This lines up with the population-geometry chapter's finding that the interactive-face
state is the compact, low-dimensional one in 4/4 regions: a state the population enters
from a distributed drive rather than one each region can generate alone.

### R9. Data-side: regional PC trajectories are near-perfectly linearly related
Pairwise CCA on the 42-PC regional trajectories gives first canonical correlations
> 0.999 for all 6 pairs, still > 0.96 at dimension 20. **This is uninterpretable as
stated** — 42×42 CCA on 300 condition-time observations is nearly rank-saturated, and
no cross-validation or permutation null was run. Treat as a motivating observation
only; it needs the fix in G6 before it appears in the chapter.

---

## 4. Gaps — what has to be run or fixed

Ordered by what blocks the chapter.

**G1. Iteration-matched bottleneck ensembles (blocks R6).**
The headline consistency claim currently confounds rank with training length.
Run 100 seeds at **rank 1, 3 and 5, all at 250k iterations, identical settings
otherwise**. This is the single most important missing run. It answers: does
constraining the channel actually make the circuit identifiable, and is there a rank
at which the current pattern converges?

**G2. Best-iterate checkpointing, everywhere.**
Runs currently save the *final* iterate. The 40-unit `cross+self-diagonal` run ended
~4× above its own best loss with a negative tail slope; 25/100 ensemble seeds are
flagged bad. Every fit compared in the chapter must save the best iterate, and the
convergence table (`final_over_best`, `rel_improvement_last_2k`, transient count) has
to be a reported figure panel, not a diagnostic.

**G3. Ensemble the architecture comparison (blocks R2, R3, R8).**
One seed per architecture. Run 20+ seeds per architecture at 50 units so R2's ordering
and R8's interactive-face effect get an inferential test, and so the lesion group
comparison (R3) has independent replicates instead of 16 perturbations of one network.

**G4. Capacity-matched architectures (blocks R2).**
`within_region` at 50 units has 40% of `full`'s parameters, so its worse fit is
confounded with its size. Section 7 of the model-selection notebook already solves for
the equalizing sizes: **50 / 57 / 90** units for full / cross+self-diag / within-region.
Not yet trained.

**G5. A clean hidden-unit sweep** at fixed learning rate (1e-3) and fixed iterations,
10–60 units. R1's question is currently unanswered. Cheap.

**G6. Any held-out data at all.**
Every number above is in-sample, on a model with k/n ≈ 1. The training code already
supports the fix with no new code: set `post_fixation_loss_weight = 0.0`, train on
pre-fixation bins only, and score the post-fixation half the model must generate by
rolling its own dynamics forward. Do this for the chosen architecture at rank 3.

**G6b. Fix the CCA (blocks R9).**
Cross-validated canonical correlations (fit on a subset of condition-time bins, score
on held-out) plus a circular-shift null, matching the null convention already used in
`../population_thesis/` and `../signal_correlation/`. Or drop R9 from the chapter.

**G7. Refit ranks 10/20/30 at 100k iterations (blocks R5).**
Three runs, so the bottleneck sweep can be read end to end rather than in two blocks.

**G8. Condition-resolved current analysis on the bottlenecked ensembles.**
The 100-seed condition contrast (R7) exists only for the *unbottlenecked* ensemble.
A first pass over the 10 rank-3 seeds reproduces the direction of every significant
effect — ofc +0.042, bla −0.040, dmpfc +0.019 for interactive vs non-interactive face
(healthy 7 seeds) — but n = 10 is below the power the contrast needs. Falls out of G1
for free once those seeds exist.

**G10. Extend the channel-identity test** (R10) to the 100-seed rank-3 ensemble, and ask
whether the union of a pathway's three channels is more stable than each individually.
Decides whether R10 is "not identifiable" or "identifiable but underpowered". Falls out
of G1.

**G9. Dynamical-systems characterization is missing entirely.**
The current-geometry work projects currents onto progression directions. Nothing yet
characterizes the *system*: fixed points and their stability (Jacobian eigenspectra
per condition), whether the three conditions sit in one basin or three, the dimension
and orientation of the slow manifold, and whether the interactive-face state's
compactness (population chapter) shows up as a distinct attractor. This is the natural
mechanistic payoff and it is entirely unstarted.

### R10. The rank-3 channels are not identifiable across seeds *(new, 2026-09-02)*
Mapping each pathway's read and write factors into the relevant region's PC space —
fixed by the data, therefore shared across fits — and comparing by principal angles over
the 10 rank-3 seeds: cross-seed alignment averages **0.296** against a chance level of
0.218 (95th percentile 0.313); only 5 of 24 pathway × direction combinations clear that
percentile. The sharper test is decisive: **a pathway's channel is no more similar to
itself in another seed than to a different pathway in that seed** (write 0.284 vs 0.284,
p = 0.27; read 0.309 vs 0.314, p = 0.99).

So the bottleneck constrains *how much* passes between regions without pinning down
*what* passes. Ten seeds is small, so this stays open pending the 100-seed rank-3
ensemble (G1 → G10), but a weak effect is not what a mechanistic claim about a
three-dimensional channel would need.

---

## 5. Proposed chapter

Working title: *What the regions have to tell each other, and when.*

| Fig | Claim | Rests on | Status |
|---|---|---|---|
| 1 | Architecture and target: what the model sees, what it must produce | — | schematic exists (`plots/model_selection/architecture_schematics_and_performance.pdf`) |
| 2 | Regions cannot reproduce each other's trajectories alone; dense internal recurrence is dispensable | R2 | needs G3, G4 |
| 3 | Three dimensions per pathway suffice | R5, R4 | needs G7 |
| 4 | What those three dimensions *are* — and that seeds do not agree on them | R10 | done for 10 seeds; needs G1 |
| 5 | Fits replicate, circuits do not — and what survives anyway | R6, R7 | needs G1, G2 |
| 6 | Interactive face fixations are the condition that requires the coupling | R8, R7 | needs G3, G8 |
| 7 | The dynamical system: fixed points, basins, and the interactive-face state | G9 | **unstarted** |

Discussion has to carry, prominently: in-sample only unless G5 runs; k/n ≈ 1 so no
parameter-counting criterion is interpretable (AIC/BIC/F-tests appear as a caveat, not
a selection rule); the model is finely tuned rather than robust, so perturbation results
describe the fit and not the brain; object fixations are not matched for interactive
state; and the honest framing of R6 — that a well-fitting mRNN is a hypothesis generator
about routing, not a measurement of it.

---

## 6. Recommended order

1. **G2** (best-iterate checkpointing) — a code change; everything downstream depends on it.
2. **G1** (100 seeds × ranks 1/3/5 at 250k) — the big cluster job; launch it early. Gives G8 free.
3. **G5** (clean hidden-unit sweep) — cheap, and closes the one question that has been reopened.
4. **G3 + G4** (architecture ensemble at matched capacity) — second cluster job.
5. **G9** (fixed points / Jacobians) — new `src/` analysis code, on the G1 ensemble.
6. **G5**, **G6**, **G7** — the three cleanup runs.
7. Author the chapter notebook here.
