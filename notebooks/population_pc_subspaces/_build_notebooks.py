"""Author the two population PC subspace notebooks from source strings.

Per ``AGENTS.md`` the notebooks are thin: every function and class they call
lives in ``src/dal_monte_2022_analysis``.  This script assembles narrative and
call sites only, so the two notebooks cannot drift apart -- they are the same
cells with one parameter changed, which is the point of the comparison.

    conda run -n gaze_processing python notebooks/population_pc_subspaces/_build_notebooks.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path


SCOPES: dict[str, dict[str, object]] = {
    "all_units": {
        "filename": "population_pc_subspaces_all_units.ipynb",
        "title": "Population PC subspaces — all recorded units",
        "unit_scope": "all",
        "blurb": (
            "This notebook uses **every recorded unit** with an average PSTH for all three "
            "fixation conditions. It is the reference version: the dimensionality, the "
            "trajectories and the subspace comparisons reported here are the population-level "
            "result for each region."
        ),
    },
    "pair_selective_units": {
        "filename": "population_pc_subspaces_pair_selective_units.ipynb",
        "title": "Population PC subspaces — fixation-pair selective units",
        "unit_scope": "pair_selective",
        "blurb": (
            "This notebook repeats the all-unit analysis on units that are **significantly "
            "selective for at least one fixation-condition pair** (FDR-corrected, "
            "`three_condition_core` family). Selecting units by a condition contrast and then "
            "measuring condition separation is circular by construction, so this is a "
            "sensitivity check, not an independent confirmation. What it can legitimately "
            "answer is whether the population geometry is carried by the selective minority or "
            "distributed across the whole population."
        ),
    },
}


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
from dal_monte_2022_analysis.ephys.analysis import fixation_population_pc_subspace as pcs
from dal_monte_2022_analysis.ephys.plotting import fixation_population_pc_subspace as viz
from dal_monte_2022_analysis.ephys.plotting import thesis_common as style

DATASET_CFG_PATH = repo_root / "configs" / "dataset.yaml"
PLOTTING_CFG_PATH = repo_root / "configs" / "plotting.yaml"

style.apply_thesis_plot_style(load_config(PLOTTING_CFG_PATH))
pd.set_option("display.width", 210)
pd.set_option("display.max_columns", 90)

# ---- analysis parameters -------------------------------------------------- #
UNIT_SCOPE = "__UNIT_SCOPE__"          # "all" or "pair_selective"
WINDOW_MS = (-500.0, 500.0)
VARIANCE_THRESHOLD = 0.95
N_NULL_SUBSPACES = 200                  # random-subspace baseline draws
N_LABEL_SHUFFLES = 100                  # condition-label shuffles
N_UNIT_BOOTSTRAPS = 200                 # unit resamples for separation bands
N_SPLIT_HALVES = 50                     # unit split-halves for reliability
RANDOM_SEED = 20260827

FIGURE_DIR = pcs.resolve_output_dir(DATASET_CFG_PATH, scope=UNIT_SCOPE + "_units")
FIGURE_MANIFEST: dict[str, Path] = {}

SHORT = {"face_interactive": "IF", "face_non_interactive": "NF", "object": "OB"}
PAIR_SHORT = {
    "face_interactive__vs__face_non_interactive": "IF/NF",
    "face_interactive__vs__object": "IF/OB",
    "face_non_interactive__vs__object": "NF/OB",
}


def show(fig, stem: str, *, dpi: int = 190) -> None:
    """Save one figure as editable PDF plus PNG, then display the PNG."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = FIGURE_DIR / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None, bbox_inches="tight", transparent=False)
        FIGURE_MANIFEST[f"{stem}.{extension}"] = path
    display(Image(filename=str(FIGURE_DIR / f"{stem}.png")))
    import matplotlib.pyplot as plt

    plt.close(fig)


def save_table(frame: pd.DataFrame, stem: str) -> pd.DataFrame:
    """Write a table next to the figures and return it for display."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{stem}.csv"
    frame.to_csv(path, index=False)
    FIGURE_MANIFEST[f"{stem}.csv"] = path
    return frame


def unordered_pair(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse ordered (pc_condition, eval_condition) rows onto one pair label."""
    labels = [
        pcs.pair_label(*sorted([a, b]))
        for a, b in zip(frame["pc_condition"], frame["eval_condition"])
    ]
    return frame.assign(pair_label=labels)


print("scope:", UNIT_SCOPE)
print("outputs:", FIGURE_DIR)
'''


PROVENANCE_TEXT = r'''
## 1. Where the data comes from

Every population matrix is built from
`fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl`, the combined
10 ms average export. Three properties matter downstream:

- Values are **firing rate in Hz**, converted per trial before averaging.
- Traces are Gaussian-smoothed at $\sigma = 20$ ms **before** averaging. At a
  10 ms bin that is $\sigma = 2$ bins, so adjacent bins are not independent
  samples — this comes back in §8, where it decides what can be said about speed.
- The conditions come from two average partitions: `face_interactive` and
  `face_non_interactive` from the interactive-state **split** partition, `object`
  from the **unsplit** partition (object fixations pooled across interactive
  state, because there are too few per state to average).

### A provenance discrepancy worth knowing about

The stored `ephys/psth/fixation_population_pca/results.pkl` was **not** built
from this file. Its configured input is
`fixations_split_by_interactive_state.pkl`, an earlier export written before the
current fixation-detection pass. The unit sets match, but the older file carries
roughly 25–35% fewer fixations per unit. §3c quantifies the consequence. The
analysis below uses the current file, so it agrees with the mRNN and the
selectivity analysis rather than with the stored pickle.
'''


LOAD_CODE = '''
all_populations = pcs.load_region_populations(DATASET_CFG_PATH, window_ms=WINDOW_MS)

if UNIT_SCOPE == "pair_selective":
    selective_units, selective_pairs = pcs.load_pair_selective_units(DATASET_CFG_PATH)
    populations = pcs.load_region_populations(
        DATASET_CFG_PATH, window_ms=WINDOW_MS, unit_keys=selective_units["unit_key"]
    )
    selectivity_summary = save_table(
        pcs.summarize_selective_units(selective_units, selective_pairs, populations=all_populations),
        "selective_unit_summary",
    )
    display(Markdown("**Fixation-pair selective units retained**"))
    display(selectivity_summary)
else:
    selective_units = selective_pairs = selectivity_summary = None
    populations = all_populations

inventory = save_table(pcs.build_unit_inventory(populations), "unit_inventory")
display(inventory[[
    "region", "n_units", "n_sessions", "n_time_bins",
    "fr_matrix_units_by_time", "concatenated_matrix_units_by_time", "max_pca_rank",
]])
show(viz.plot_unit_inventory(inventory), "fig00_unit_inventory")
'''


NOTATION_TEXT = r'''
## 2. Notation, and exactly what the PCA does

### The matrices

For region $R$ with $N$ units, $T = 100$ time bins of 10 ms spanning
$[-500, +500]$ ms, and conditions $c \in \{\mathrm{IF}, \mathrm{NF}, \mathrm{OB}\}$:

$$X_c \in \mathbb{R}^{N \times T}, \qquad
[X_c]_{n t} = \text{mean firing rate of unit } n \text{ at bin } t \text{ for condition } c$$

The concatenation is along **time**, in a fixed condition order:

$$X = \begin{bmatrix} X_{\mathrm{IF}} & X_{\mathrm{NF}} & X_{\mathrm{OB}}\end{bmatrix}
\in \mathbb{R}^{N \times 3T}$$

### The PCA

Time bins are **samples**, units are **features**. Let $Y = X^{\top} \in \mathbb{R}^{3T \times N}$,

$$\mu = \frac{1}{3T}\sum_{s=1}^{3T} Y_{s\cdot} \in \mathbb{R}^{N},
\qquad \tilde{Y} = Y - \mathbf{1}\mu^{\top},
\qquad \tilde{Y} = U \Sigma V^{\top}$$

The **components** are the leading right singular vectors, $W = V_{:,1:k}^{\top} \in \mathbb{R}^{k \times N}$,
with $W W^{\top} = I_k$. Each row is a direction in neuron space — a weighting
over the $N$ neurons.

The **scores** (trajectories) are

$$Z_c = W\left(X_c - \mu \mathbf{1}^{\top}\right) \in \mathbb{R}^{k \times T}$$

so $Z_c(t) \in \mathbb{R}^{k}$ is the population state at bin $t$, and the
back-projection is $\hat{X}_c = W^{\top} Z_c + \mu\mathbf{1}^{\top}$.

Variance explained by component $j$:
$\lambda_j = \sigma_j^2 / (3T - 1)$, ratio $\lambda_j / \sum_i \lambda_i$.

### Two centrings, and why the distinction runs through everything below

There are two different means in play, and nearly every interpretive question in
this notebook turns on which one was used.

| Fit | Centred on | Consequence |
| --- | --- | --- |
| **Concatenated** ($W$, $\mu$ above) | the grand mean over all $3T$ samples | Each condition keeps its own offset from that grand mean. This is why the 3D trajectories appear as three separated clouds. |
| **Per condition** ($W_c$, $\mu_c = \frac{1}{T}\sum_t X_c(:,t)$) | that condition's **own centroid** | The offset is removed before the basis is computed. $W_c$ describes only how condition $c$ *varies about its own mean*. |

Everything positional — centroid distances, the 3D pictures — uses the
concatenated fit. Everything in §7 that compares *subspaces* uses per-condition
fits, and is therefore already blind to the offsets. §7d checks that numerically
rather than asserting it.

### The concatenation contract

Column block $i$ of $Z$ must map back to condition $i$, in the same order in and
out. This is the step most likely to fail silently — a block swap would relabel
every trajectory while leaving every figure looking plausible — so §3 asserts it
as a numerical identity.
'''


CONTRACT_CODE = '''
for region, population in populations.items():
    concatenated = population.concatenated_units_by_time()
    print(f"{region:>6}  N={population.n_units:4d} units, T={population.n_time} bins")
    print(f"        X_c  (units x time)          = {population.condition_matrix(population.conditions[0]).shape}")
    print(f"        X    (units x 3T)            = {concatenated.shape}")
    print(f"        column blocks                = {population.condition_slices()}")
    print(f"        max PCA rank min(N, 3T-1)    = {min(population.n_units, 3 * population.n_time - 1)}")
'''


VERIFY_TEXT = r'''
## 3. Verification

### 3a. Numerical identities

Seven checks per region. Six are exact linear-algebra facts; the seventh guards
against a one-bin time shift when the concatenated projection is sliced apart.

| Check | Identity | A failure would mean |
| --- | --- | --- |
| `components_orthonormal` | $WW^{\top} = I_k$ | the basis is not orthonormal, so projections double-count variance |
| `fit_mean_is_sample_mean` | $\mu = \frac{1}{3T}\sum_s Y_{s\cdot}$ | conditions centred on the wrong mean, shifting every trajectory |
| `concatenation_slice_roundtrip` | $Z[:, (i-1)T{:}\,iT] = W(X_{c_i} - \mu\mathbf{1}^{\top})$ | **the slice step returns the wrong block — trajectories mislabelled** |
| `concatenated_scores_match_manual_projection` | stored $Z$ equals $W(X-\mu\mathbf{1}^\top)$ | stored scores stale relative to the stored basis |
| `full_rank_reconstruction_exact` | $W^{\top}WX_c^{\text{ctr}} = X_c^{\text{ctr}}$ at full rank | back-projection does not invert the projection |
| `projection_preserves_total_energy` | $\|W\tilde{X}\|_F^2 = \|\tilde{X}\|_F^2$ (Parseval) | variance lost or created in the projection |
| `time_order_not_shifted` | zero-lag alignment beats a one-bin roll | a condition block written back off-by-one in time |
'''


VERIFY_CODE = '''
fits = {region: pcs.fit_all_scopes(population) for region, population in populations.items()}

checks = save_table(
    pd.concat(
        [
            pcs.verify_pca_identities(population, fits[region]["concatenated"])
            for region, population in populations.items()
        ],
        ignore_index=True,
    ),
    "pca_identity_checks",
)
n_failed = int((~checks["passed"]).sum())
display(Markdown(
    f"**{len(checks) - n_failed} / {len(checks)} identity checks pass**"
    + ("" if n_failed == 0 else f" — **{n_failed} FAILED**")
))
show(viz.plot_verification_summary(checks), "fig01_identity_checks")
display(checks.loc[~checks["passed"]] if n_failed else checks.head(7)[["check", "criterion", "value", "tolerance"]])
'''


IDENTITY_TEXT = r'''
### 3b. Does the interactive-face trajectory actually belong to interactive face?

The identities above prove the arithmetic is self-consistent. They do **not**
prove the labels are right: a pipeline that swapped two conditions consistently
would pass every one of them.

This check goes the other way round. Reconstruct each condition from its own
truncated scores, $\hat{X}_a = W^{\top}Z_a + \mu\mathbf{1}^{\top}$, then score
that reconstruction against **every** condition's true rates:

$$R^2(a \to b) = 1 - \frac{\|X_b - \hat{X}_a\|_F^2}{\|X_b - \bar{X}_b\|_F^2}$$

If the labelling is correct the winning match must lie on the diagonal for all
three conditions.

**Reading the numbers.** The diagonal is the fraction of a condition's own
temporal variance that the shared $k$-dimensional subspace captures — a real
quantity, not a formality. The off-diagonal entries are large and *negative*,
which is expected rather than a bug: reconstructing one condition and scoring it
against another does worse than predicting that other condition's own mean, so
$R^2 < 0$. The more negative, the more distinct the two conditions are.
'''


IDENTITY_CODE = '''
dimensionality = save_table(pcs.build_dimensionality_table(fits), "dimensionality_by_region")
SHARED_N_COMPONENTS, per_region_components = pcs.resolve_shared_n_components(
    fits, threshold=VARIANCE_THRESHOLD
)

confusion = save_table(
    pd.concat(
        [
            pcs.condition_identity_confusion(
                population, fits[region]["concatenated"], n_components=SHARED_N_COMPONENTS
            )
            for region, population in populations.items()
        ],
        ignore_index=True,
    ),
    "condition_identity_confusion",
)
diagonal = confusion[confusion["reconstructed_from"] == confusion["compared_against"]]
display(Markdown(
    f"**Condition identity recovered in {int(diagonal['identity_recovered'].sum())} / {len(diagonal)} "
    f"region x condition cases** (reconstruction at k = {SHARED_N_COMPONENTS})."
))
for region in confusion["region"].unique():
    display(Markdown(f"`{region}` — R² of reconstruction (rows) against true rates (columns)"))
    display(
        confusion[confusion["region"] == region]
        .pivot(index="reconstructed_from", columns="compared_against", values="r2")
        .round(3)
    )
'''


STALE_TEXT = r'''
### 3c. This analysis against the stored `results.pkl`

Quantifies the provenance gap from §1. `unit_sets_identical` should be true — the
two pipelines select the same neurons. The dimensionality columns are where they
part company, and `stored_max_components` shows a second issue: the stored run
caps PCA at 50 components (`population_pca_max_components: 50`), so for regions
needing more than that the stored cumulative curve never reaches 95% and the
retained-dimension question cannot be answered from it at all.
'''


STALE_CODE = '''
stored_comparison = pcs.verify_against_stored_pca(populations, fits, cfg_path=DATASET_CFG_PATH)
if len(stored_comparison):
    display(save_table(stored_comparison, "stored_pca_comparison"))
else:
    display(Markdown("_No stored `results.pkl` found; skipping._"))
'''


DIMS_TEXT = r'''
## 4. How many PCs to keep

### The quantity

$$\text{cum}(k) = \frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{\min(N,\,3T-1)}\lambda_j},
\qquad k^{*}(R) = \min\{k : \text{cum}(k) \ge 0.95\}$$

The denominator runs over **all** components, so this is the ordinary fraction of
total variance. The fit here is full-rank (no configured cap), which is what makes
$k^{*}$ a measurement rather than a number clipped by a setting.

### The shared dimension

$$k = \max_{R} k^{*}(R)$$

The maximum, not the mean: it guarantees the retained subspace reaches the
threshold in **every** region, so no area is analysed in a space too small for
it. This is the same rule the mRNN target builder applies, so the two analyses
work in comparably sized spaces.

### Why $k \approx 40$ and not $\approx 20$

These counts come from the **concatenated** fit, which spans three conditions'
worth of structure. A single condition's own fit needs roughly half as many — the
per-condition rows in the table below show the difference directly. Neither
number is "the dimensionality of the region"; they answer different questions.

### Why 3 PCs are used for pictures and $k$ for everything else

Three PCs capture only around half the concatenated variance. Any subspace
comparison computed on them would be a comparison of the visualisation, not of
the populations, so every quantitative result below uses all $k$.
'''


DIMS_CODE = '''
binding = max(per_region_components, key=per_region_components.get)
display(Markdown(
    f"**k = {SHARED_N_COMPONENTS} PCs reach {VARIANCE_THRESHOLD:.0%} of the concatenated variance "
    f"in every region.**  Per region: "
    + ", ".join(f"{style.region_label(r)} {c}" for r, c in sorted(per_region_components.items()))
    + f".  Binding region: {style.region_label(binding)}."
))
show(
    viz.plot_cumulative_variance(fits, threshold=VARIANCE_THRESHOLD, shared_n_components=SHARED_N_COMPONENTS),
    "fig02_cumulative_variance",
)
display(dimensionality[[
    "region", "fit_scope", "n_units", "n_components_full_rank",
    "n_pcs_for_80pct", "n_pcs_for_90pct", "n_pcs_for_95pct", "n_pcs_for_99pct",
    "participation_ratio", "explained_variance_ratio_pc1",
]])
show(viz.plot_variance_spectra(fits), "fig03_variance_spectra")
'''


PR_TEXT = r'''
### Participation ratio, in the table above

$$\mathrm{PR} = \frac{\left(\sum_j \lambda_j\right)^2}{\sum_j \lambda_j^2}$$

An effective dimension count that needs no threshold. If $d$ components share the
variance equally, $\mathrm{PR} = d$; if one dominates, $\mathrm{PR} \to 1$. It is
much smaller than $k^{*}$ (single digits versus ~40) because it is dominated by
the leading components, whereas $k^{*}$ is set by the long tail. Both are correct
answers to different questions: **PR** says how many dimensions carry most of the
action, **$k^{*}$** says how many you must keep to avoid discarding 5% of it.
'''


TRAJ_TEXT = r'''
## 5. What the trajectories look like

All three conditions are drawn in the **concatenated** basis, so the axes mean
the same thing for every curve and the visible distances are real distances.
Drawing each condition in its own basis would give each curve its own axes and
the distances between them would mean nothing.

Viewing angles are searched per region. The conditions separate along different
PC combinations in each area, so one fixed camera cannot serve all four. The
search maximises the **product** of two normalised terms — the closest gap
between any two trajectories, and how unfolded each trajectory stays. A sum lets
a view win on separation alone, and the winner is then always a near-top-down
camera that spreads the conditions apart while collapsing every trajectory into
a scribble.

Because a camera can always be chosen to flatter the data, the fixed PC-plane
projections that follow are the honest cross-check: they have no free parameters.
'''


TRAJ_CODE = '''
figure, resolved_views = viz.plot_pc_trajectories_3d(fits)
show(figure, "fig04_pc_trajectories_3d")
display(Markdown(
    "Selected views (elev°, azim°): "
    + ", ".join(f"`{r}` {a[0]:.0f}, {a[1]:.0f}" for r, a in resolved_views.items())
))
show(viz.plot_pc_plane_projections(fits), "fig05_pc_planes")
show(viz.plot_pc_timecourses(fits), "fig06_pc_timecourses")
'''


DECOMP_TEXT = r'''
## 6. Separating the offset from the dynamics

The PC time courses above make a structural point the 3D view hides: the leading
PCs are dominated by a near-**constant offset** between conditions. Each condition
sits in a different part of the state space and wanders comparatively little
within its own neighbourhood.

That matters, because *"the conditions occupy separated subspaces"* and *"the
conditions follow different dynamics"* are different claims and the figures so
far support only the first.

### The decomposition (two-way ANOVA on the population tensor)

Write the tensor $r_{c,t,n}$ (condition $\times$ time $\times$ unit) and let
$\bar{r}_{\cdot\cdot n}$ be the grand mean over conditions and time for unit $n$.
Define three terms:

$$
\underbrace{a_{t n} = \bar{r}_{\cdot t n} - \bar{r}_{\cdot\cdot n}}_{\text{shared time course}}
\qquad
\underbrace{b_{c n} = \bar{r}_{c \cdot n} - \bar{r}_{\cdot\cdot n}}_{\text{condition offset}}
\qquad
\underbrace{e_{c t n} = r_{c t n} - \bar{r}_{\cdot\cdot n} - a_{t n} - b_{c n}}_{\text{condition} \times \text{time interaction}}
$$

Because the design is balanced (every condition has the same $T$), the three are
**mutually orthogonal** and the sums of squares add exactly:

$$\underbrace{\sum_{c,t,n}(r_{ctn}-\bar{r}_{\cdot\cdot n})^2}_{\text{SS}_{\text{total}}}
= \underbrace{C\sum_{t,n} a_{tn}^2}_{\text{SS}_{\text{time}}}
+ \underbrace{T\sum_{c,n} b_{cn}^2}_{\text{SS}_{\text{offset}}}
+ \underbrace{\sum_{c,t,n} e_{ctn}^2}_{\text{SS}_{\text{interaction}}}$$

The reported shares are $\text{SS}_{\bullet} / \text{SS}_{\text{total}}$, so they
sum to 1 by construction (a useful check on the table below).

### What each share means

| Term | What it is | What a large value means |
| --- | --- | --- |
| **Shared time course** | the fixation-locked response every condition follows | the population responds to *fixating*, regardless of what is fixated |
| **Condition offset** | a static per-condition displacement, constant over the window | conditions differ by *where the population sits*, not by what it does over time |
| **Condition × time** | everything left: condition-specific dynamics | conditions differ in *how the response unfolds* |

### Yes — the offset removal is centroid subtraction

`marginalize_population` implements exactly the terms above:

- `remove_condition_offset` subtracts $b_{cn}$, i.e. **each condition's own
  time-average, per neuron**. In the PC space this is
  $d_c(t) = Z_c(t) - m_c$ with $m_c = \frac{1}{T}\sum_t Z_c(t)$ — the centroid.
  So: yes, it is centroid subtraction, done per condition and per neuron.
- `remove_shared_time_course` subtracts $a_{tn}$, the across-condition mean at
  each time bin.
- `condition_dynamics_only` subtracts both, leaving $e_{ctn}$ alone.

Refitting PCA on the residual is what makes the condition-specific dynamics
visible; in the raw fit they are buried under terms three to five times larger.
'''


DECOMP_CODE = '''
decomposition = save_table(
    pd.concat(
        [pcs.condition_variance_decomposition(population) for population in populations.values()],
        ignore_index=True,
    ),
    "variance_decomposition",
)
show(viz.plot_variance_decomposition(decomposition), "fig07_variance_decomposition")
shares = decomposition.pivot(index="region", columns="component", values="fraction_of_total")
display(shares.round(3))
display(Markdown(
    "Orthogonality check — the three components sum to "
    + ", ".join(
        f"`{r}` {shares.loc[r, ['condition_independent_time', 'condition_main_effect', 'condition_by_time_interaction']].sum():.4f}"
        for r in shares.index
    )
    + " (should be 1.0000)."
))

dynamics_fits = {
    region: {
        "concatenated": pcs.fit_population_pca(
            pcs.marginalize_population(population, mode="condition_dynamics_only")
        )
    }
    for region, population in populations.items()
}
figure, _ = viz.plot_pc_trajectories_3d(dynamics_fits)
show(figure, "fig08_condition_dynamics_only_3d")
display(Markdown(
    "Above: the same populations after removing **both** the shared time course and each "
    "condition's offset. What remains is the interaction term alone. Compare with fig04 — "
    "the clean separation there was carried by the offsets."
))
'''


SUBSPACE_TEXT = r'''
## 7. Comparing the condition subspaces

Everything in this section uses **per-condition** fits, which centre on each
condition's own mean. They are therefore measures of *covariance structure*, not
of position — §7d verifies that.

### 7a. Cross-condition variance explained

Let $\tilde{X}_c = X_c - \mu_c\mathbf{1}^{\top}$ (condition $c$ about its own
centroid) and

$$C_c = \frac{1}{T-1}\tilde{X}_c\tilde{X}_c^{\top} \in \mathbb{R}^{N \times N}$$

its temporal covariance across neurons. Let $V_a \in \mathbb{R}^{N \times k}$ hold
the top-$k$ eigenvectors of $C_a$. Then

$$\text{captured}(a \to b) = \operatorname{tr}\!\left(V_a^{\top} C_b V_a\right),
\qquad
\text{fraction}(a \to b) = \frac{\operatorname{tr}(V_a^{\top} C_b V_a)}{\operatorname{tr}(C_b)}$$

In words: **take condition $a$'s principal directions, project condition $b$'s
activity onto them, and ask what fraction of $b$'s variance survives.**

It is **asymmetric**: $\text{fraction}(a\to b) \ne \text{fraction}(b\to a)$,
because a condition with broadly spread variance makes a more general basis. That
asymmetry is informative and is not averaged away in the tables.

### 7b. Alignment index — and why the raw fraction is not enough

The fraction above has no natural scale: how much *could* any $k$-dimensional
subspace capture of $C_b$? At most the top-$k$ eigenvalues. Normalising by that
ceiling gives the alignment index (Elsayed et al., 2016):

$$A(a \to b) = \frac{\operatorname{tr}\left(V_a^{\top} C_b V_a\right)}{\sum_{j=1}^{k}\lambda_j(C_b)} \in [0, 1]$$

- $A = 1$: $a$'s directions are **as good for $b$** as $b$'s own optimal
  $k$-dimensional subspace. The two conditions vary in the same subspace.
- $A = 0$: $a$'s directions carry none of $b$'s variance — orthogonal subspaces.

The normalisation is what makes values comparable across conditions: a low raw
fraction could just mean $b$ is low-variance, whereas a low $A$ means genuinely
different directions.

#### The floor is $k/N$, and it is not the same in every region

A **uniformly random** $k$-dimensional subspace of $\mathbb{R}^{N}$ captures, in
expectation, a fraction $k/N$ of the trace of *any* covariance. So

$$\mathbb{E}\left[A_{\text{random}}\right] \approx \frac{k}{N}$$

This is checked against simulation below (they agree to three decimals). It has a
sharp consequence: **raw alignment indices must not be compared across regions.**
At $k = 42$ the floor is $42/537 = 0.08$ in BLA but $42/187 = 0.22$ in dmPFC, so
dmPFC's higher alignment values are partly just its smaller population. The
comparable quantity rescales chance to zero:

$$A^{\dagger} = \frac{A - k/N}{1 - k/N}$$

**Read `alignment_above_floor` for anything cross-region.**

#### The ceiling

$A = 1$ is unreachable in practice because a subspace estimated from finite noisy
data does not perfectly capture even its own condition. The within-condition
ceiling (fit on odd time bins, evaluate on even) bounds how high $A$ could
plausibly go. Because the traces are smoothed at $\sigma = 2$ bins, odd and even
bins are not independent and this ceiling is **optimistic** — it is an upper
bound on plausible alignment, not a noise estimate.

### 7c. Principal angles

The scale-free counterpart. For orthonormal bases $V_a, V_b \in \mathbb{R}^{N\times k}$,
the principal angles $0 \le \theta_1 \le \cdots \le \theta_k \le 90°$ satisfy

$$\cos\theta_j = s_j\!\left(V_a^{\top} V_b\right)$$

the singular values of the $k\times k$ overlap matrix. Reading them:

| Quantity | Meaning |
| --- | --- |
| $\theta_1 = 0$ | at least one direction is shared exactly |
| $\theta_j = 90°$ | direction $j$ of $a$ is orthogonal to **all** of $b$ |
| $\sum_j \cos^2\theta_j = \|V_a^{\top}V_b\|_F^2$ | the effective **number of shared dimensions** (a real-valued count out of $k$) |
| $\overline{\cos^2\theta} = \frac{1}{k}\sum_j\cos^2\theta_j$ | shared fraction of the subspace |
| $d_{\text{Grassmann}} = \sqrt{\sum_j \theta_j^2}$ | a proper metric on subspaces |
| $d_{\text{chordal}} = \sqrt{\sum_j \sin^2\theta_j}$ | equals $\sqrt{k - \|V_a^\top V_b\|_F^2}$: the "unshared" dimension count |

Angles differ from the alignment index in one important way: they weight every
direction equally, while $A$ weights by how much variance actually lies along
each. Two conditions can share their high-variance directions (high $A$) while
their low-variance directions are unrelated (large mean angle). **Angles answer
"are the directions the same?"; $A$ answers "do the directions that matter carry
the other condition's activity?"**

The same $k/N$ caveat applies: random $k$-subspaces in a small $N$ are closer
together, so mean angles are not comparable across regions either — compare
against each region's own null.

### 7d. These measures never saw the offsets

A per-condition PCA centres on that condition's own time-average, so $V_c$
already describes deviations from the centroid, and $C_c$ is built the same way.
The alignment index and the principal angles are therefore **already** pure
dynamics measures and cannot be inflated by the conditions sitting in different
places.

That is easy to assume and easy to get wrong, so it is checked: every metric is
recomputed after explicitly subtracting each condition's centroid, and the
difference must be zero to numerical precision.
'''


SUBSPACE_CODE = '''
subspace_table = save_table(
    pcs.build_region_subspace_summary(
        populations, fits,
        n_components=SHARED_N_COMPONENTS,
        n_null_subspaces=N_NULL_SUBSPACES,
        seed=RANDOM_SEED,
    ),
    "pairwise_subspace_metrics",
)
variance_curves = save_table(
    pd.concat(
        [
            pcs.cross_condition_variance_curve(population, fits[region], max_components=SHARED_N_COMPONENTS)
            for region, population in populations.items()
        ],
        ignore_index=True,
    ),
    "cross_condition_variance_curves",
)
alignment_ceiling = save_table(
    pd.concat(
        [
            pcs.within_condition_alignment_ceiling(population, n_components=SHARED_N_COMPONENTS)
            for population in populations.values()
        ],
        ignore_index=True,
    ),
    "within_condition_alignment_ceiling",
)
offset_invariance = save_table(
    pcs.verify_subspace_offset_invariance(populations, n_components=SHARED_N_COMPONENTS),
    "subspace_offset_invariance",
)

show(viz.plot_cross_condition_variance_curves(variance_curves, ceiling=alignment_ceiling), "fig09_cross_condition_variance_curves")
show(viz.plot_alignment_matrix(subspace_table), "fig10_alignment_matrix")
show(viz.plot_principal_angle_spectra(fits, n_components=SHARED_N_COMPONENTS), "fig11_principal_angle_spectra")
show(viz.plot_subspace_distance_map(fits, n_components=SHARED_N_COMPONENTS), "fig12_subspace_distance_map")

off = subspace_table.loc[~subspace_table["is_within_condition"]]
display(Markdown("**Simulated vs analytic random-subspace floor** (should agree)"))
display(
    off.groupby("region")[["n_units", "alignment_null_mean", "alignment_floor_analytic"]]
    .mean().round(4)
)
display(Markdown("**Cross-condition subspace metrics** (ordered pairs)"))
display(off[[
    "region", "pc_condition", "eval_condition",
    "variance_explained_fraction", "alignment_index", "alignment_floor_analytic",
    "alignment_above_floor", "angle_mean_principal_angle", "angle_mean_cos2",
    "angle_chordal_distance",
]].round(3))
display(Markdown(
    f"**Offset-invariance check:** alignment and principal angles are unchanged by explicit "
    f"centroid removal in {int(offset_invariance['offset_invariant'].sum())} / {len(offset_invariance)} "
    f"region x pair cases (max difference "
    f"{offset_invariance['alignment_difference'].max():.2e} in alignment, "
    f"{offset_invariance['angle_difference_deg'].max():.2e}° in angle)."
))
'''


PAIRSUM_TEXT = r'''
### 7e. Collapsed to unordered pairs

The ordered table above is the primary record. This is the same data averaged
over direction, which is what to quote when the asymmetry is not the point.
`alignment_above_floor` is the cross-region-comparable column; the effective
shared dimension count is $\|V_a^\top V_b\|_F^2 = k\,\overline{\cos^2\theta}$.
'''


PAIRSUM_CODE = '''
pair_summary = unordered_pair(off).groupby(["region", "pair_label"]).agg(
    alignment_index=("alignment_index", "mean"),
    alignment_floor=("alignment_floor_analytic", "mean"),
    alignment_above_floor=("alignment_above_floor", "mean"),
    mean_principal_angle=("angle_mean_principal_angle", "mean"),
    mean_cos2=("angle_mean_cos2", "mean"),
    chordal_distance=("angle_chordal_distance", "mean"),
).reset_index()
pair_summary["shared_dimensions"] = pair_summary["mean_cos2"] * SHARED_N_COMPONENTS
pair_summary["pair"] = pair_summary["pair_label"].map(PAIR_SHORT)
display(save_table(pair_summary, "pairwise_subspace_summary").round(3))
'''


LEGACY_TEXT = r'''
### 7f. A note on the stored `explained_variance_fraction` column

The existing `fixation_population_pca` builder writes a column named
`explained_variance_fraction` whose own metadata field calls it
`projection_variance_fraction_within_retained_pcs`. It normalises by

$$\frac{\sum_{j\le k} v_j^{\top}C_b v_j}{\sum_{j \le k} v_j^{\top} C_b v_j} \to 1$$

— that is, by the variance *inside* the retained PCs rather than by $b$'s total
variance. It therefore approaches 1.0 for every condition pairing once enough PCs
are retained, regardless of how different the subspaces are. It measures where
variance sits *within* the retained set; it is not a cross-condition
variance-explained measure and should not be read as one.
'''


LEGACY_CODE = '''
at_retained = variance_curves[variance_curves["n_components"] == SHARED_N_COMPONENTS]
display(
    at_retained.loc[~at_retained["is_within_condition"], [
        "region", "pc_condition", "eval_condition",
        "cumulative_variance_explained", "alignment_index", "fraction_within_retained_pcs",
    ]].round(3).head(12)
)
'''


DYNAMICS_TEXT = r'''
## 8. How the conditions differ *in their dynamics*

§7 asked whether the conditions vary along the same directions. This section asks
the two questions that answer are they moving differently, and it is where the
face-versus-object distinction actually shows up.

### 8a. Splitting the separation into a static part and a moving part

Write $Z_c(t) = m_c + d_c(t)$ with $m_c = \frac{1}{T}\sum_t Z_c(t)$ and
$\sum_t d_c(t) = 0$. Then

$$\left\langle \|Z_a(t) - Z_b(t)\|^2 \right\rangle_t
= \underbrace{\|m_a - m_b\|^2}_{\text{offset}}
+ \underbrace{\left\langle \|d_a(t) - d_b(t)\|^2 \right\rangle_t}_{\text{dynamics}}$$

exactly — the cross term carries a factor $\langle d_a - d_b\rangle_t = 0$ and
vanishes. So the separation splits with no residual, and reporting only the total
conflates *"the conditions sit in different places"* with *"the conditions move
differently"*.

### 8b. Do they move *together*? The deviation correlation

The dynamics term above is large whenever the two conditions move differently
**for any reason** — including simply moving by different amounts. The sharper
question is whether they move in step:

$$\rho_{ab} = \frac{\left\langle d_a(t)^{\top} d_b(t) \right\rangle_t}
{\sqrt{\left\langle\|d_a\|^2\right\rangle_t \left\langle\|d_b\|^2\right\rangle_t}} \in [-1, 1]$$

A vector-valued time-series correlation.

- $\rho = 1$: the two conditions trace the **same excursion at the same time**.
- $\rho = 0$: they move independently.
- $\rho < 0$: they move oppositely.

This is a strictly different question from the alignment index. $A$ asks whether
the conditions *can* move along the same directions; $\rho$ asks whether they
*do so simultaneously*. Two conditions could share a subspace perfectly ($A$
high) yet visit it at unrelated times ($\rho = 0$).

### 8c. Correcting for the trial-count imbalance

Interactive-face fixations outnumber the other two roughly **five to one**
(median ~850 vs ~168 per unit). A noisier average has more apparent excursion and
more apparent speed, so an uncorrected comparison of dynamics would be partly a
comparison of trial counts. This is the same confound the single-unit chapter
documents for the coefficient of variation.

Each stored average carries a standard error $s_c(t,n)$. Treating estimation noise
as independent across neurons, its covariance in unit space at time $t$ is
$\operatorname{diag}(s_c(t,\cdot)^2)$, and in the PC basis $V$ its energy is

$$\nu_c(t) = \operatorname{tr}\!\left(V^{\top}\operatorname{diag}(s_c(t,\cdot)^2)V\right)
= \sum_{j\le k}\sum_n V_{nj}^2\, s_c(t,n)^2$$

Two corrections follow, using $\bar\nu_c = \langle \nu_c(t)\rangle_t$:

$$\left\langle\|d_c\|^2\right\rangle_t^{\text{signal}}
= \left\langle\|d_c\|^2\right\rangle_t - \bar\nu_c\,(1 - \kappa),
\qquad
\|m_a - m_b\|^2_{\text{signal}} = \|m_a-m_b\|^2 - (\bar\nu_a + \bar\nu_b)\,\kappa$$

where $\kappa$ is the fraction of noise energy surviving into the time mean.

**The smoothing matters here.** For independent noise $\kappa = 1/T$. But
convolving white noise with a Gaussian of width $\sigma$ leaves noise with
autocorrelation $\rho_{\text{noise}}(\ell) = \exp(-\ell^2/4\sigma^2)$, which
averages down more slowly, so $\kappa$ is computed from that kernel with
$\sigma = 2$ bins (read off the average files' metadata, not fitted).

For $\rho_{ab}$ the correction is a **disattenuation**: the numerator
$\langle d_a^{\top}d_b\rangle$ is already unbiased, because noise is independent
*across conditions*; only the normalising energies are inflated. Note the
direction of the bias — uncorrected $\rho$ is biased **toward zero**, hardest for
the conditions with fewest trials, so failing to correct would make a real shared
dynamic look like an absence of one.

### 8d. Speed, and what cannot be claimed about it

$$\text{step energy} = \left\langle \|Z_c(t{+}1) - Z_c(t)\|^2\right\rangle_t,
\qquad \text{noise expectation} = 2\bar\nu_c\left(1 - e^{-1/4\sigma^2}\right)$$

`speed_over_noise` is their ratio. Values at or below 1 mean bin-to-bin motion is
**not resolvable** above the noise floor.

This is a real limit, not a formality: the 20 ms smoothing correlates the noise at
exactly the timescale a 10 ms derivative probes. Excursion and dimensionality are
integrated over the whole window and survive correction comfortably; instantaneous
speed largely does not. Read the excursion, treat the speed as indicative.

### 8e. The test for shared dynamics

Is $\rho_{ab}$ bigger than chance? The null circularly shifts one condition's
deviation trajectory against the other. This preserves each condition's own
temporal autocorrelation and its excursion energy, and destroys **only** the
temporal correspondence between them — exactly what a non-zero $\rho$ claims. A
white-noise null would be far too permissive, since two smooth slowly-varying
trajectories correlate substantially by accident.

The test is exact rather than sampled: all 81 admissible shifts are used (shifts
under 10 bins are excluded because a smoothed trajectory still overlaps itself
there). With 81 shifts the smallest attainable $p$ is $1/82 = 0.012$, so **read
the $z$-score as the graded measure** and the $p$-value as a threshold.
'''


DYNAMICS_CODE = '''
offset_dynamics = save_table(
    pcs.build_offset_vs_dynamics_table(populations, fits, n_components=SHARED_N_COMPONENTS),
    "offset_vs_dynamics",
)
shift_test = save_table(
    pcs.test_deviation_correlation(populations, fits, n_components=SHARED_N_COMPONENTS),
    "deviation_correlation_shift_test",
)
dynamics_summary = save_table(
    pd.concat(
        [
            pcs.condition_dynamics_summary(
                population, fits[region]["concatenated"], n_components=SHARED_N_COMPONENTS
            )
            for region, population in populations.items()
        ],
        ignore_index=True,
    ),
    "condition_dynamics_summary",
)

show(viz.plot_offset_vs_dynamics(offset_dynamics, shift_test=shift_test), "fig13_offset_vs_dynamics")
display(offset_dynamics[[
    "region", "pair_label", "offset_share_of_separation", "offset_share_corrected",
    "dynamics_share_corrected", "deviation_correlation", "deviation_correlation_corrected",
]].round(3))

display(Markdown("**Circular-shift test for time-locked shared dynamics**"))
display(shift_test[[
    "region", "pair_label", "observed_deviation_correlation",
    "shift_null_mean", "shift_null_sd", "z_against_shift_null", "p_value",
]].round(3))

show(viz.plot_condition_dynamics(dynamics_summary), "fig14_condition_dynamics")
display(dynamics_summary[[
    "region", "condition", "median_n_trials", "excursion_rms", "excursion_rms_corrected",
    "signal_fraction_of_excursion", "speed_over_noise",
    "dynamics_participation_ratio", "dynamics_participation_ratio_corrected",
]].round(2))
'''


COMOVE_TEXT = r'''
### 8f. The two dynamics questions, plotted against each other

Horizontal axis: alignment index — *do the conditions vary along the same
directions at all?* Vertical axis: deviation correlation — *do they travel those
directions at the same time?*

These are genuinely independent, and the data separates on one of them and not
the other. Watch which axis orders the pairs. Ringed markers exceed the
circular-shift null at $p < 0.05$.

Note that $z$ and $p$ can disagree at the margin (dmPFC IF/NF has $z = 2.0$ but
$p = 0.061$): the shift null is not Gaussian, and with 81 admissible shifts the
$p$-value is a coarse rank statistic. The $z$-score is the better graded measure;
significance is called on $p$ throughout.
'''


COMOVE_CODE = '''
show(
    viz.plot_alignment_vs_comovement(subspace_table, offset_dynamics, shift_test=shift_test),
    "fig15_alignment_vs_comovement",
)
'''


NULL_TEXT = r'''
## 9. Is the geometry more than chance, and does it replicate?

**Condition-label shuffle.** Permutes condition labels independently within each
unit, so every neuron keeps all three of its real traces and only the
population's *agreement* about which trace is which is destroyed.

Note which way this null points. It does **not** test whether the conditions are
separated — a shuffled population has three subspaces that are, if anything,
*more* different from each other, because the real conditions share a large common
fixation-locked response. It tests whether the observed **shared** subspace is
real. Both tail probabilities are reported so the direction never has to be
inferred from a sign.

**Unit bootstrap.** Resamples units with replacement and refits. With
trial-averaged input there are no trials left to resample, so this asks "would
another sample of neurons from this region show the same separation" — the
generalisation a population claim needs. It is an interval, not a test.

**Split-half reliability.** Splits units into two disjoint halves and reruns the
pipeline in each. The halves span different unit spaces so their bases are not
comparable, but the *geometry* they imply is.

**What is not available.** There is no unbiased cross-validated distance here.
Cross-validated (crossnobis-style) distances need independent repeats of the same
measurement, and trial averaging removed them. That is why this section reports a
label-shuffle null and resampling intervals rather than a cross-validated effect
size — and why §8e uses a circular-shift null, which needs no repeats.
'''


NULL_CODE = '''
null_comparison = save_table(
    pd.concat(
        [
            pcs.build_pairwise_null_comparison(
                population, n_components=SHARED_N_COMPONENTS,
                n_shuffles=N_LABEL_SHUFFLES, seed=RANDOM_SEED,
            )
            for population in populations.values()
        ],
        ignore_index=True,
    ),
    "condition_shuffle_null",
)
show(viz.plot_null_comparison(null_comparison, ceiling=alignment_ceiling), "fig16_shuffle_null")
display(null_comparison[[
    "region", "pair_label",
    "observed_alignment_index", "shuffled_alignment_index", "p_alignment_higher_than_shuffle",
    "observed_mean_principal_angle", "shuffled_mean_principal_angle", "p_angle_smaller_than_shuffle",
]].round(3))

separation = save_table(
    pd.concat(
        [
            pcs.time_resolved_separation(
                population, fits[region]["concatenated"],
                n_components=SHARED_N_COMPONENTS, n_bootstrap=N_UNIT_BOOTSTRAPS, seed=RANDOM_SEED,
            )
            for region, population in populations.items()
        ],
        ignore_index=True,
    ),
    "time_resolved_separation",
)
show(viz.plot_time_resolved_separation(separation), "fig17_time_resolved_separation")

reliability_summary = save_table(
    pcs.summarize_split_half_reliability(
        pd.concat(
            [
                pcs.split_half_geometry_reliability(
                    population, n_components=SHARED_N_COMPONENTS,
                    n_splits=N_SPLIT_HALVES, seed=RANDOM_SEED,
                )
                for population in populations.values()
            ],
            ignore_index=True,
        )
    ),
    "split_half_reliability",
)
display(Markdown("**Split-half reliability of the condition geometry**"))
display(reliability_summary.round(3))
'''


CLAIMS_TEXT = r'''
## 10. What can be claimed

The cell below reads the saved tables and states each claim with its supporting
numbers, so it stays correct when the analysis is rerun. The narrative that
follows interprets it.
'''


CLAIMS_CODE = '''
lines: list[str] = []
regions = sorted(populations)

lines.append("### Dimensionality\\n")
lines.append(
    f"- **k = {SHARED_N_COMPONENTS}** PCs reach {VARIANCE_THRESHOLD:.0%} of the concatenated variance in "
    "every region ("
    + ", ".join(f"{style.region_label(r)} {per_region_components[r]}" for r in regions)
    + f"); {style.region_label(binding)} is binding."
)
pr = dimensionality[dimensionality["fit_scope"] == "concatenated"].set_index("region")["participation_ratio"]
lines.append(
    "- But the participation ratio is only "
    + ", ".join(f"{style.region_label(r)} {pr[r]:.1f}" for r in regions)
    + " — most of the action is in a handful of dimensions; the other ~35 carry the 5% tail."
)

lines.append("\\n### Where the variance lives\\n")
for region in regions:
    row = shares.loc[region]
    lines.append(
        f"- **{style.region_label(region)}**: shared time course {row['condition_independent_time']:.0%}, "
        f"condition offset {row['condition_main_effect']:.0%}, "
        f"condition x time {row['condition_by_time_interaction']:.0%}."
    )
lines.append(
    "- Across regions the **static offset is the single largest term**, so the visual separation "
    "in fig04 is mostly positional."
)

lines.append("\\n### Subspace overlap (floor-corrected, cross-region comparable)\\n")
ranks = []
for region in regions:
    sub = pair_summary[pair_summary["region"] == region].set_index("pair")
    ordered = sub["alignment_above_floor"].sort_values(ascending=False)
    ranks.append(str(ordered.index[0]))
    lines.append(
        f"- **{style.region_label(region)}**: "
        + ", ".join(f"{p} {sub.loc[p, 'alignment_above_floor']:.3f}" for p in ["IF/NF", "IF/OB", "NF/OB"] if p in sub.index)
        + f"  (raw floor k/N = {sub['alignment_floor'].iloc[0]:.3f})"
    )
lines.append(
    f"- **IF/NF is the most-aligned pair in {ranks.count('IF/NF')} of {len(regions)} regions.** "
    "The two face conditions share more subspace structure than either shares with object — "
    "but the margins are modest."
)

lines.append("\\n### Time-locked shared dynamics — the clearest dissociation\\n")
for region in regions:
    sub = shift_test[shift_test["region"] == region].set_index("pair_label")
    parts = []
    for label, short in PAIR_SHORT.items():
        if label not in sub.index:
            continue
        z = float(sub.loc[label, "z_against_shift_null"])
        significant = float(sub.loc[label, "p_value"]) < 0.05
        parts.append(f"{short} z={z:+.1f}{'*' if significant else ''}")
    lines.append(f"- **{style.region_label(region)}**: " + ", ".join(parts))
significant = shift_test[shift_test["p_value"] < 0.05]
face_face = significant[significant["pair_label"] == "face_interactive__vs__face_non_interactive"]
lines.append(
    f"- IF/NF exceeds the shift null in **{len(face_face)} of {len(regions)}** regions; "
    f"face-vs-object pairs do so in "
    f"**{len(significant) - len(face_face)} of {2 * len(regions)}** cases."
)

lines.append("\\n### Dynamic amplitude, after correcting for the 5:1 trial imbalance\\n")
for region in regions:
    sub = dynamics_summary[dynamics_summary["region"] == region].set_index("condition")
    lines.append(
        f"- **{style.region_label(region)}** excursion (corrected): "
        + ", ".join(f"{SHORT[c]} {sub.loc[c, 'excursion_rms_corrected']:.1f}" for c in SHORT if c in sub.index)
        + "  |  dynamics dimensionality: "
        + ", ".join(f"{SHORT[c]} {sub.loc[c, 'dynamics_participation_ratio_corrected']:.1f}" for c in SHORT if c in sub.index)
    )
resolvable = dynamics_summary[dynamics_summary["speed_over_noise"] > 1.0] if "speed_over_noise" in dynamics_summary else dynamics_summary.iloc[:0]
lines.append(
    f"- Bin-to-bin speed is resolvable above the noise floor in only "
    f"{len(resolvable)} of {len(dynamics_summary)} region x condition cases — do not quote speed."
)

display(Markdown("\\n".join(lines)))
save_table(pd.DataFrame({"claim": lines}), "claims")
'''


NARRATIVE_TEXT = r'''
### Reading it together

Three statements the analysis supports, in decreasing order of confidence.

**1. The three conditions occupy clearly distinct population states, and the
distinction is mostly positional.** The offset term is the largest single
component of the variance in every region, and 55–75% of the squared separation
between any pair is static. The clean clustering in fig04 is real, but it is a
statement about *where* the population sits during each kind of fixation, not
about different trajectories through state space.

**2. Interactive and non-interactive face share a time-locked dynamic that
neither shares with object.** This is the sharpest result and it does not follow
from the subspace measures. Floor-corrected alignment ranks IF/NF highest in every
region, but only narrowly. The deviation correlation separates the pairs
decisively: IF/NF is well above its circular-shift null in BLA, ACCg and OFC,
while both face-vs-object pairs sit at zero in those regions. The two face
conditions do not merely have access to similar directions — they move along them
*at the same moments*, which object fixations do not.

Note this could not have been seen from the alignment index alone: fig15 shows the
pairs are not ordered by the horizontal axis at all.

**3. The interactive-face state is more compact and lower-dimensional.** After
noise correction its excursion is roughly half that of the other conditions in
every region, and its dynamics participation ratio is consistently lower. This
survives the 5:1 trial-count correction, which is the obvious artefactual
explanation and the one to rule out first.

**dmPFC is the exception throughout.** It has the highest floor-corrected
alignment for every pair, and it is the one region where face-vs-object deviation
correlations are also above the shift null. Its conditions share more structure
overall and are separated less sharply — a different regime, not a weaker version
of the same one.

### What is *not* supported

- **Any claim about rotational dynamics, speed differences, or trajectory
  shape at fine timescales.** `speed_over_noise` is near 1 in most region ×
  condition cases: bin-to-bin motion is not resolvable above the noise the
  20 ms smoothing leaves behind.
- **Cross-region comparison of raw alignment indices or principal angles.**
  The chance floor is $k/N$ and $N$ ranges from 187 to 537. Use
  `alignment_above_floor`.
- **Anything requiring a cross-validated effect size.** Trial averaging removed
  the repeats. The nulls here are permutation-based and the intervals are
  resampling-based; neither is a cross-validated distance.
- **A causal or directional reading of the asymmetry** in
  `variance_explained_fraction`. That a condition's PCs capture more of another
  than the reverse reflects how broadly its variance is spread, nothing more.
'''


MANIFEST_CODE = '''
display(Markdown("**Saved outputs**"))
display(Markdown("\\n".join(f"- `{name}`" for name in sorted(FIGURE_MANIFEST))))
print(FIGURE_DIR)
'''


_CELL_COUNTER = count(1)


def _cell(kind: str, source: str) -> dict:
    lines = source.strip("\n").splitlines(keepends=True)
    cell = {
        "cell_type": kind,
        "id": f"cell-{next(_CELL_COUNTER):02d}",
        "metadata": {},
        "source": lines,
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "gaze_processing",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(scope_key: str) -> dict:
    global _CELL_COUNTER
    _CELL_COUNTER = count(1)
    spec = SCOPES[scope_key]
    header = f"""# {spec['title']}

{spec['blurb']}

**Structure**

| § | Question |
|---|---|
| 1 | Where the data comes from, and one provenance discrepancy |
| 2 | Notation, and exactly what operation the PCA performs |
| 3 | Verification: identities, condition labels, and the stored pickle |
| 4 | How many PCs to keep, and why that number |
| 5 | What the trajectories look like |
| 6 | Separating the static offset from the dynamics |
| 7 | Comparing the condition subspaces — four measures, with their maths |
| 8 | How the conditions differ *in their dynamics* |
| 9 | Nulls, intervals and replication |
| 10 | What can and cannot be claimed |

Regenerate with
`conda run -n gaze_processing python notebooks/population_pc_subspaces/_build_notebooks.py`.
Reusable code lives in
`src/dal_monte_2022_analysis/ephys/analysis/fixation_population_pc_subspace.py`
and `.../ephys/plotting/fixation_population_pc_subspace.py`.
"""

    cells = [
        _cell("markdown", header),
        _cell("code", SETUP.replace("__UNIT_SCOPE__", str(spec["unit_scope"]))),
        _cell("markdown", PROVENANCE_TEXT),
        _cell("code", LOAD_CODE),
        _cell("markdown", NOTATION_TEXT),
        _cell("code", CONTRACT_CODE),
        _cell("markdown", VERIFY_TEXT),
        _cell("code", VERIFY_CODE),
        _cell("markdown", IDENTITY_TEXT),
        _cell("code", IDENTITY_CODE),
        _cell("markdown", STALE_TEXT),
        _cell("code", STALE_CODE),
        _cell("markdown", DIMS_TEXT),
        _cell("code", DIMS_CODE),
        _cell("markdown", PR_TEXT),
        _cell("markdown", TRAJ_TEXT),
        _cell("code", TRAJ_CODE),
        _cell("markdown", DECOMP_TEXT),
        _cell("code", DECOMP_CODE),
        _cell("markdown", SUBSPACE_TEXT),
        _cell("code", SUBSPACE_CODE),
        _cell("markdown", PAIRSUM_TEXT),
        _cell("code", PAIRSUM_CODE),
        _cell("markdown", LEGACY_TEXT),
        _cell("code", LEGACY_CODE),
        _cell("markdown", DYNAMICS_TEXT),
        _cell("code", DYNAMICS_CODE),
        _cell("markdown", COMOVE_TEXT),
        _cell("code", COMOVE_CODE),
        _cell("markdown", NULL_TEXT),
        _cell("code", NULL_CODE),
        _cell("markdown", CLAIMS_TEXT),
        _cell("code", CLAIMS_CODE),
        _cell("markdown", NARRATIVE_TEXT),
        _cell("code", MANIFEST_CODE),
    ]
    return _notebook(cells)


def main() -> None:
    here = Path(__file__).resolve().parent
    for scope_key, spec in SCOPES.items():
        path = here / str(spec["filename"])
        path.write_text(json.dumps(build(scope_key), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
