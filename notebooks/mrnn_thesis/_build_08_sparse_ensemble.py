"""Author the sparse-ensemble notebook (task 08 of the rebuilt mRNN analysis).

    conda run -n gaze_processing python notebooks/mrnn_thesis/_build_08_sparse_ensemble.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

OUTPUT_FILENAME = "08_sparse_ensemble.ipynb"
_CELL_COUNTER = count(1)


HEADER = r"""# 08 · A sparse network, fitted ten times

*The last experiment. Everything before this measured how much the solution set can be
narrowed; this asks whether the narrowest network that still fits has anything to say that
its ten fits agree on.*

**Why sparsity and not something else.** Every constraint family tried so far was scored on
a measure that turned out to include within-region self-drive alongside the inter-regional
currents it was named for (task 06, E2). Corrected, the low-rank family *reverses* — a rank-1
bottleneck agrees at 0.016, which is chance. Sparsity is the only family that survives the
correction pointing the right way:

| arm | as reported | inter-regional only |
|---|---|---|
| dense baseline | 0.551 | **0.539** |
| within 5% | 0.882 | **0.626** |
| within 50% | 0.770 | **0.682** |
| cross 25% | 0.903 | **0.602** |
| rank-1 | 0.816 | **0.016** |

A lift from 0.54 to about 0.6–0.68, on three seeds, which is three pairs. That is not
resolvable and it is exactly what ten seeds is for.

**One thing had to be fixed before the question could be asked.** The sparsity mask was tied
to the run seed, so every seed of every previous sparsity arm was fitted with a *different
surviving topology*. That is the right default for asking whether sparsity as a family gives
reproducible fits, and it is what the earlier arms measured — but it cannot answer whether
ten fits of one sparse network agree, because those ten networks were not the same network.
`sparsity_seed` is now separable from the run seed, and this notebook runs **both** arms:

- **shared mask** — one topology, ten weight initialisations. *Is a fixed sparse network's
  solution unique?* This is the identifiability question proper.
- **varying mask** — ten topologies, ten initialisations. *Does sparsity as a design produce
  a consistent circuit description?* This is what the earlier arms accidentally asked.

The pair is more informative than either. If the shared-mask arm agrees and the varying-mask
arm does not, the circuit is determined by the wiring and not by the data — which is a result,
and not the one the chapter wants.

| Section | |
|---|---|
| 1 | The adequacy bar, restated |
| 2 | The density screen — 4 densities × 3 seeds |
| 3 | Choosing the density |
| 4 | The ensemble — 10 seeds, shared and varying mask |
| 5 | Does the ensemble fit? |
| 6 | Agreement, corrected and against its floor |
| 7 | Lesions: directed, bidirectional, and whole-region |
| 8 | Do the seeds agree on which connections matter? |
| 9 | What is special about interactive face |
| 10 | Reading the result |

**Nothing here submits a job unless a `SUBMIT_*` flag is set to `True`.**
"""


S1 = r"""## 1. The adequacy bar, restated

Task 04 marked a variant adequate when its **worst region × condition cell** sat within
twice the baseline's own seed-to-seed spread of the noise ceiling. That spread is 0.00067,
so the bar was $R^2/\text{ceiling} \ge 0.9987$ — the worst single cell had to be within
0.13% of the data's own reliability.

That is a bar calibrated to reproducibility, not to adequacy, and it is far stricter than
anything the chapter needs. Under it, 5% cross-region density was marked a failure at
**0.9914** — a fit within 0.9% of the noise ceiling on its worst cell.

So this notebook states its bar explicitly and reports both:

| bar | rule | what it is for |
|---|---|---|
| **strict** | within 2× the baseline seed spread (≈ 0.999) | "indistinguishable from the unconstrained fit" |
| **adequate** | worst cell ≥ 0.99 of ceiling | "reproduces the data" |

The strict bar is the right one for asking whether a constraint is *free*. The adequate bar
is the right one here, because the question is not whether sparsity costs nothing — it is
whether a sparse network still reproduces the data well enough that its circuit is worth
reading. On the adequate bar, 5% density is in play.
"""


SETUP = r'''
%load_ext autoreload
%autoreload 2

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from IPython.display import Image, Markdown, display

repo_root = Path.cwd()
if not (repo_root / "src").exists():
    repo_root = next(parent for parent in Path.cwd().parents if (parent / "src").exists())
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_audit as audit
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_circuit as circ
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_protocol as protocol
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_sweep as sweep
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_synthesis as syn
from dal_monte_2022_analysis.ephys.analysis import fixation_psth_noise_ceiling as ceiling_mod
from dal_monte_2022_analysis.ephys.plotting import fixation_mrnn_audit as aviz
from dal_monte_2022_analysis.ephys.plotting import fixation_mrnn_sweep as viz
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    ThesisFigureSettings, apply_thesis_plot_style, figure_to_png_bytes, save_thesis_figure,
)

DATASET_CFG_PATH = repo_root / "configs" / "dataset.yaml"
MRNN_CFG_PATH = repo_root / "configs" / "ephys_fixation_mrnn.yaml"
apply_thesis_plot_style(load_config(repo_root / "configs" / "plotting.yaml"))

PROTOCOL_ROOT = protocol.resolve_chapter_root(DATASET_CFG_PATH, task="00_training_protocol")
BASE_ROOT = sweep.resolve_task_root("02_model_selection", DATASET_CFG_PATH)
TASK_ROOT = sweep.resolve_task_root("08_sparse_ensemble", DATASET_CFG_PATH)
CEILING_DIR = ceiling_mod.resolve_output_dir(DATASET_CFG_PATH)
FIGURE_DIR = syn.resolve_output_dir(DATASET_CFG_PATH, scope="08_sparse_ensemble")
FIGURES = ThesisFigureSettings(output_dir=FIGURE_DIR)

SELECTED_PROTOCOL, PROTOCOL_IS_PROVISIONAL = sweep.load_selected_protocol_or_provisional(
    PROTOCOL_ROOT / "selected_protocol.yaml"
)
BASE_MODEL = yaml.safe_load((BASE_ROOT / "selected_base_model.yaml").read_text())

#: Uniform density applied to **all sixteen blocks**, within and cross alike. The earlier
#: prongs varied one side at a time, which is what makes a coordinate-wise conclusion
#: unsafe: 5% within alone and 25% cross alone each fit, and the corner where they meet
#: does not. One density everywhere cannot fall into that gap.
DENSITY_GRID = (0.05, 0.10, 0.15, 0.25)
SCREEN_SEEDS = 3
ENSEMBLE_SEEDS = 10
#: The topology every seed of the shared-mask arm is fitted on. Any fixed integer will do;
#: it is recorded so the arm can be reproduced exactly.
SHARED_MASK_SEED = 20260905

#: The bar this notebook judges on. See Section 1 -- 0.99 of the measured noise ceiling on
#: the worst region x condition cell.
ADEQUATE_BAR = 0.99

ceiling_by_region = (
    pd.read_csv(CEILING_DIR / "pc_space_ceiling.csv")
    .groupby("region")["reliability"].mean().to_dict()
)

base_overrides = {
    "hidden_units": int(BASE_MODEL["hidden_units"]),
    "condition_loss_weighting": str(BASE_MODEL["condition_loss_weighting"]),
    "l1_weight_scale": float(BASE_MODEL["l1_weight_scale"]),
    "recurrent_connectivity": str(BASE_MODEL["recurrent_connectivity"]),
    "recurrent_bottleneck_dim": None,
}


def show(figure, stem: str) -> None:
    save_thesis_figure(figure, FIGURES, stem)
    display(Image(data=figure_to_png_bytes(figure, dpi=190)))


def cached(name: str, build):
    path = TASK_ROOT / "tables" / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    frame = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


print("recipe   :", SELECTED_PROTOCOL["selected_label"])
print("base     :", base_overrides)
print("task root:", TASK_ROOT)
'''


S2_TEXT = r"""## 2. The density screen

Four densities, three seeds each, applied uniformly to all sixteen blocks. This is a screen,
not the experiment: it exists to pick the density the ensemble runs at, and three seeds is
enough to see where the fit falls off.
"""

S2_CODE = r'''
screen_variants = [
    sweep.ModelVariant(
        label=f"global_d{density:g}".replace(".", "p"),
        arm="uniform density screen",
        overrides={**base_overrides,
                   "within_region_density": float(density),
                   "cross_region_density": float(density)},
    )
    for density in DENSITY_GRID
]
screen_seeds = protocol.protocol_seeds(n_seeds=SCREEN_SEEDS)

display(pd.DataFrame([v.describe() for v in screen_variants]))
display(Markdown(
    f"**{len(screen_variants)} densities × {SCREEN_SEEDS} seeds = "
    f"{len(screen_variants) * SCREEN_SEEDS} runs.** The mask varies with the seed here, "
    "which is what a screen wants: a density that only fits on one lucky topology is not a "
    "density that fits."
))
'''

S2B_CODE = r'''
SUBMIT_SCREEN = False   # <-- set to True to submit the density screen

flight = sweep.in_flight_run_dirs(TASK_ROOT / "_jobs_screen", TASK_ROOT / "_jobs_ensemble")
screen_commands, _ = sweep.variant_job_commands(
    screen_variants, screen_seeds, root=TASK_ROOT, repo_root=repo_root,
    protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH,
    exclude_run_dirs=flight["run_dirs"],
)
screen_inventory = sweep.index_variant_runs(
    TASK_ROOT, screen_variants, screen_seeds, in_flight=flight["run_dirs"]
)
display(Markdown(
    f"**{int(screen_inventory['complete'].sum())} complete**, "
    f"**{int(screen_inventory['queued'].sum())} queued**, "
    f"**{int(screen_inventory['pending'].sum())} unqueued** of {len(screen_inventory)}."
))

if SUBMIT_SCREEN and screen_commands:
    from datetime import datetime
    from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

    jobs_dir = TASK_ROOT / "_jobs_screen"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_file = jobs_dir / f"screen_{stamp}.txt"
    write_job_file(job_file, screen_commands)
    submitted = [Path(line.split("--run-dir ")[1].split()[0]) for line in screen_commands]
    screen_id = submit_dsq_array_job(
        job_file_path=job_file, sbatch_script_path=jobs_dir / f"screen_{stamp}.sh",
        log_dir=jobs_dir / "logs", job_name="mrnn_density_screen", partition="psych_gpu",
        cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
    )
    sweep.record_submission(jobs_dir, job_id=screen_id, run_dirs=submitted,
                            label=f"density screen {stamp}")
    (jobs_dir / "job_id.txt").write_text(str(screen_id) + "\n")
    display(Markdown(f"Submitted **{len(screen_commands)}** runs as array **{screen_id}**."))
elif screen_commands:
    display(Markdown("`SUBMIT_SCREEN` is **False** — nothing was submitted."))
else:
    display(Markdown("The density screen is fully trained."))
'''


S3_TEXT = r"""## 3. Choosing the density

Two numbers decide it, and they pull in opposite directions.

**Fit** has to clear the adequate bar of Section 1 on the worst region × condition cell.
**Agreement** is what the whole exercise is for, and it is measured on the inter-regional
blocks only — the correction from task 06 that reversed the low-rank result.

The rule: **the sparsest density that clears the bar.** Agreement is reported alongside and
breaks ties, but it does not override fit — reading a circuit off a network that does not
reproduce the data is the mistake task 05 made when it ran its gate on the failed corner.
"""

S3_CODE = r'''
screen_done = sweep.index_variant_runs(TASK_ROOT, screen_variants, screen_seeds)
if not screen_done["complete"].any():
    display(Markdown("*The density screen has not been trained yet — Section 2 submits it.*"))
    SELECTED_DENSITY = None
else:
    fit = sweep.score_variant_fit(screen_done[screen_done["complete"]], ceiling_by_region)
    summary = sweep.adequacy_table(fit, baseline_label="global_d0p25")
    rows = []
    for variant in screen_variants:
        dirs = audit.seed_run_dirs(TASK_ROOT / variant.label)
        if len(dirs) < 2:
            continue
        block = audit.current_agreement_decomposition({variant.label: TASK_ROOT / variant.label})
        rows.append({"label": variant.label,
                     "inter_only_agreement": float(block["agreement_inter_only"].iloc[0]),
                     "self_energy_share": float(block["self_energy_share"].iloc[0])})
    agreement = pd.DataFrame(rows)
    table = summary.merge(agreement, on="label", how="left")
    table["clears_bar"] = table["worst_condition"] >= ADEQUATE_BAR
    display(table[["label", "worst_condition", "mean_all", "clears_bar",
                   "inter_only_agreement", "self_energy_share"]].round(4))

    passing = table[table["clears_bar"]]
    SELECTED_DENSITY = None
    if not passing.empty:
        pick = passing.iloc[passing["label"].map(
            lambda s: float(s.replace("global_d", "").replace("p", "."))).argmin()]
        SELECTED_DENSITY = float(pick["label"].replace("global_d", "").replace("p", "."))
        (TASK_ROOT / "selected_density.yaml").write_text(yaml.safe_dump({
            "selected_density": SELECTED_DENSITY,
            "selection_rule": f"sparsest uniform density with worst cell >= {ADEQUATE_BAR} of ceiling",
            "worst_condition": float(pick["worst_condition"]),
            "inter_only_agreement": float(pick["inter_only_agreement"]),
            "shared_mask_seed": SHARED_MASK_SEED,
        }))
        display(Markdown(
            f"**Selected density: {SELECTED_DENSITY:g}** — worst cell "
            f"{pick['worst_condition']:.4f} of ceiling, inter-regional agreement "
            f"{pick['inter_only_agreement']:.3f}. Written to `selected_density.yaml`."))
    else:
        display(Markdown("**No density clears the bar.** The ensemble should not be run."))
'''


S4_TEXT = r"""## 4. The ensemble — ten seeds, two arms

The two arms differ in one setting and ask different questions.

**Shared mask.** `sparsity_seed` pinned, so all ten fits have the *same* surviving
connections and differ only in weight initialisation. Disagreement here is disagreement
about the solution, which is the identifiability question the chapter has been asking since
task 05.

**Varying mask.** `sparsity_seed` left to follow the run seed, so each fit has its own
topology. This is what every earlier sparsity arm did. Disagreement here mixes "different
solutions" with "different networks", so on its own it is uninterpretable — but *beside* the
shared-mask arm it becomes the measurement of how much of the disagreement is the wiring.

The comparison is the point. Three outcomes, all worth having:

| shared mask | varying mask | reading |
|---|---|---|
| agrees | disagrees | the solution is fixed by the topology, not by the data |
| agrees | agrees | sparsity has made the circuit identifiable — the result the chapter wanted |
| disagrees | disagrees | the solution set is large even at one fixed sparse topology |
"""

S4_CODE = r'''
SELECTED = None
if (TASK_ROOT / "selected_density.yaml").exists():
    SELECTED = yaml.safe_load((TASK_ROOT / "selected_density.yaml").read_text())
    SELECTED_DENSITY = float(SELECTED["selected_density"])

if SELECTED_DENSITY is None:
    display(Markdown("*No density selected — Section 3 has to run first.*"))
    ensemble_variants = []
else:
    ensemble_variants = [
        sweep.ModelVariant(
            label=f"ensemble_shared_mask_d{SELECTED_DENSITY:g}".replace(".", "p"),
            arm="shared mask",
            overrides={**base_overrides,
                       "within_region_density": SELECTED_DENSITY,
                       "cross_region_density": SELECTED_DENSITY,
                       "sparsity_seed": int(SHARED_MASK_SEED)},
        ),
        sweep.ModelVariant(
            label=f"ensemble_varying_mask_d{SELECTED_DENSITY:g}".replace(".", "p"),
            arm="varying mask",
            overrides={**base_overrides,
                       "within_region_density": SELECTED_DENSITY,
                       "cross_region_density": SELECTED_DENSITY},
        ),
    ]
    display(pd.DataFrame([v.describe() for v in ensemble_variants]))
    display(Markdown(
        f"**2 arms × {ENSEMBLE_SEEDS} seeds = {2 * ENSEMBLE_SEEDS} runs** at density "
        f"{SELECTED_DENSITY:g}."))
'''

S4B_CODE = r'''
SUBMIT_ENSEMBLE = False   # <-- set to True to submit both ensemble arms

ensemble_seeds = protocol.protocol_seeds(n_seeds=ENSEMBLE_SEEDS)
if ensemble_variants:
    flight = sweep.in_flight_run_dirs(TASK_ROOT / "_jobs_screen", TASK_ROOT / "_jobs_ensemble")
    ensemble_commands, _ = sweep.variant_job_commands(
        ensemble_variants, ensemble_seeds, root=TASK_ROOT, repo_root=repo_root,
        protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH,
        exclude_run_dirs=flight["run_dirs"],
    )
    inventory = sweep.index_variant_runs(
        TASK_ROOT, ensemble_variants, ensemble_seeds, in_flight=flight["run_dirs"])
    display(Markdown(
        f"**{int(inventory['complete'].sum())} complete**, "
        f"**{int(inventory['queued'].sum())} queued**, "
        f"**{int(inventory['pending'].sum())} unqueued** of {len(inventory)}."))

    if SUBMIT_ENSEMBLE and ensemble_commands:
        from datetime import datetime
        from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

        jobs_dir = TASK_ROOT / "_jobs_ensemble"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_file = jobs_dir / f"ensemble_{stamp}.txt"
        write_job_file(job_file, ensemble_commands)
        submitted = [Path(line.split("--run-dir ")[1].split()[0]) for line in ensemble_commands]
        ensemble_id = submit_dsq_array_job(
            job_file_path=job_file, sbatch_script_path=jobs_dir / f"ensemble_{stamp}.sh",
            log_dir=jobs_dir / "logs", job_name="mrnn_sparse_ensemble", partition="psych_gpu",
            cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
        )
        sweep.record_submission(jobs_dir, job_id=ensemble_id, run_dirs=submitted,
                                label=f"sparse ensemble {stamp}")
        (jobs_dir / "job_id.txt").write_text(str(ensemble_id) + "\n")
        display(Markdown(f"Submitted **{len(ensemble_commands)}** runs as array **{ensemble_id}**."))
    elif ensemble_commands:
        display(Markdown("`SUBMIT_ENSEMBLE` is **False** — nothing was submitted."))
    else:
        display(Markdown("Both ensemble arms are fully trained."))
'''


S5_TEXT = r"""## 5. Does the ensemble fit?

Convergence before fit, and fit before anything about the circuit. A collection of badly
fitted networks can agree perfectly and the agreement means nothing.
"""

S5_CODE = r'''
ARMS = {v.arm: TASK_ROOT / v.label for v in ensemble_variants}
ARMS = {name: path for name, path in ARMS.items() if audit.seed_run_dirs(path)}

if not ARMS:
    display(Markdown("*The ensemble has not been trained yet.*"))
else:
    done = sweep.index_variant_runs(TASK_ROOT, ensemble_variants, ensemble_seeds)
    display(sweep.convergence_table(done[done["complete"]]).round(5))
    fit = sweep.score_variant_fit(done[done["complete"]], ceiling_by_region)
    per_condition = fit.groupby(["label", "condition"])["r2_vs_ceiling"].agg(["mean", "std", "min"])
    display(per_condition.round(4))
    parameters = pd.DataFrame([
        {"label": label, **sweep.count_trainable_parameters(
            done[(done["label"] == label) & done["complete"]].iloc[0]["run_dir"])}
        for label in done.loc[done["complete"], "label"].unique()
    ])
    show(viz.plot_fit_versus_constraint(fit, parameters), "fig01_ensemble_fit")
'''


S6_TEXT = r"""## 6. Agreement, corrected and against its floor

The full battery from task 07 — seven measures tiered by what each ignores, three of them
distances — plus the inter-regional current agreement with the E2 correction applied. Every
one against fits of the untrained architecture with the same mask.

The comparison that matters is **shared mask against varying mask**, read together with the
dense baseline from task 07.
"""

S6_CODE = r'''
if ARMS:
    battery = cached("agreement_battery", lambda: pd.concat(
        [audit.agreement_battery(audit.seed_run_dirs(p)).assign(arm=name) for name, p in ARMS.items()]
        + [audit.untrained_agreement_battery(audit.seed_run_dirs(p)[0], n_draws=5).assign(arm=name)
           for name, p in ARMS.items()], ignore_index=True))
    for name in ARMS:
        show(aviz.plot_agreement_battery(battery, audit.AGREEMENT_BATTERY, arm=name),
             f"fig02_battery_{name.replace(' ', '_')}")

    decomposition = cached("current_agreement",
                           lambda: audit.current_agreement_decomposition(ARMS))
    display(decomposition.round(3))
    display(Markdown(
        "`agreement_inter_only` is the number to read. `agreement_all_blocks` is what the "
        "earlier sweeps reported, and it is inflated by the within-region self-drive the "
        "sparsity mask also thins — so under a *uniform* mask the two should sit closer "
        "together than they did for the one-sided prongs."))
'''


S7_TEXT = r"""## 7. Lesions on the trained network

The trained model is left alone and one connection group is silenced, so nothing re-adapts.
This asks what a route was carrying in the solution the network found — a different question
from task 03, which refitted without the connection and asked whether the data needs it. The
pair is more informative than either.

**Three things this has to do that the earlier lesion analysis did not.**

*Damage in comparable units.* `reconstruction_accuracy` divides by each condition's own
variance, and interactive face has about half the variance of the other two, so an identical
perturbation scores twice the damage (task 06, E4). Every per-condition comparison here is in
absolute squared error.

*A matched random control.* Under a sparsity mask, different blocks keep different numbers of
connections by chance. Ablating a pathway removes $k$ live weights; the control removes $k$
live weights drawn at random from elsewhere in the network. Without it, "this pathway
matters" can mean no more than "this pathway kept more weights".

*Directed lesions as well as bidirectional.* In the dense model a single directed removal
cost 0.0001–0.0012 against a 0.0002 seed spread, because eleven other blocks reroute around
it. Sparsity removes most of that rerouting capacity, so the directed lesions may be readable
here for the first time. Post-hoc ablation costs nothing, so all of them are run: 12 directed,
6 bidirectional, 4 within-region, and 4 whole-region isolations.
"""

S7_CODE = r'''
if ARMS:
    lesions = cached("lesion_battery", lambda: pd.concat(
        [audit.lesion_battery(p).assign(arm=name)
         for name, path in ARMS.items() for p in audit.seed_run_dirs(path)],
        ignore_index=True))
    display(lesions.groupby(["arm", "lesion_kind"])["damage_sse"].describe().round(3))
    show(aviz.plot_lesion_battery(lesions), "fig03_lesion_battery")
'''


S8_TEXT = r"""## 8. Do the seeds agree on which connections matter?

This is the question the ensemble exists to answer, and it is not "how big is the damage".

Weight-level agreement is settled and negative. But a *ranking* can survive where the
weights do not: ten networks can disagree about every connection and still agree that
severing BLA↔ACCg costs more than severing dmPFC↔OFC. That is a claim about routing which
does not require the routing to be identifiable, and it is the strongest form of result
still available.

Measured as Kendall's $\tau$ between each pair of seeds' damage rankings, against a null of
randomly permuted rankings, and separately for the directed, bidirectional and isolation
families.
"""

S8_CODE = r'''
if ARMS:
    ranking = cached("lesion_ranking_agreement",
                     lambda: audit.lesion_ranking_agreement(lesions))
    display(ranking.round(3))
    show(aviz.plot_lesion_ranking_agreement(ranking), "fig04_ranking_agreement")
    display(Markdown(
        "A $\\tau$ that clears its permutation null says the ten fits agree on the ordering "
        "of connections by importance even though they disagree about the weights. A $\\tau$ "
        "at the null says the chapter cannot rank pathways at all, and that is the honest "
        "end of the line for a circuit claim."))
'''


S9_TEXT = r"""## 9. What is special about interactive face

Every per-condition comparison in absolute squared error, so the variance confound of task 06
cannot reappear. Three questions, in order of how much they would be worth:

1. Is any condition more damaged by lesions overall, once units are comparable?
2. Does any *pathway* show a condition-specific effect — damage concentrated in interactive
   face and not in the other two?
3. Do the ten seeds agree about it?

The third is what turns an observation into a claim. In the dense model the answer to (2) was
no: all twelve pathways sat between 0.61 and 0.83 with no separation. Sparsity is the reason
to ask again.
"""

S9_CODE = r'''
if ARMS:
    condition_effect = cached("lesion_condition_effect",
                              lambda: audit.lesion_condition_contrast(lesions))
    display(condition_effect.round(4))
    show(aviz.plot_lesion_condition_contrast(condition_effect),
         "fig05_lesion_condition_contrast")

    screen = cached("invariant_screen", lambda: audit.invariant_screen(ARMS))
    consistency = cached("screen_consistency", lambda: audit.screen_consistency(screen))
    show(aviz.plot_invariant_screen(consistency, audit.SCREEN_PROPERTIES),
         "fig06_invariant_screen_sparse")
'''


S10 = r"""## 10. Reading the result

In order, and the first one gates the rest.

**Did it fit?** If no density clears 0.99 of the ceiling on its worst cell, the experiment
stops there and the result is that a uniformly sparse network cannot reproduce these
trajectories — which is itself the cleanest structural statement the chapter could make,
given that 5% *within-region* density costs almost nothing.

**Shared mask versus varying mask.** If the shared-mask arm agrees and the varying-mask arm
does not, then what the earlier sparsity arms measured was topology, and the apparent lift
from 0.54 to 0.6–0.68 was the mask varying rather than sparsity helping. If neither agrees,
the solution set is large even at one fixed sparse topology, and the chapter's negative
result is as strong as it can be made.

**The ranking, not the damage.** Section 8 is where a positive result would come from, and it
is scored against a permutation null rather than against zero. Report $\tau$ with that null on
the same axis, and report it separately for directed, bidirectional and whole-region lesions —
they are three different granularities and there is no reason they should agree.

**What survives about interactive face.** Section 9 asks in absolute units, which is the
correction that removed the previous version of this result. Anything that survives here has
survived the confound that produced the last one.

---

### What this can and cannot settle

It cannot make the circuit identifiable if it is not. What it can do is establish **which
level of description is stable**: weights, spectra, state geometry, pathway ranking, or
condition contrast. The chapter's claim should be pitched at the highest level that clears
its null, and this notebook is what locates that level.

**Next:** `09_chapter.ipynb` collates tasks 01–08 into the figure set — the fit, the
constraint ladder, the identifiability battery, the invariant screen, and whatever Section 8
returns.
"""


def _cell(kind: str, source: str) -> dict:
    cell = {
        "cell_type": kind,
        "id": f"cell-{next(_CELL_COUNTER):02d}",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build() -> dict:
    cells = [
        _cell("markdown", HEADER),
        _cell("code", SETUP),
        _cell("markdown", S1),
        _cell("markdown", S2_TEXT), _cell("code", S2_CODE), _cell("code", S2B_CODE),
        _cell("markdown", S3_TEXT), _cell("code", S3_CODE),
        _cell("markdown", S4_TEXT), _cell("code", S4_CODE), _cell("code", S4B_CODE),
        _cell("markdown", S5_TEXT), _cell("code", S5_CODE),
        _cell("markdown", S6_TEXT), _cell("code", S6_CODE),
        _cell("markdown", S7_TEXT), _cell("code", S7_CODE),
        _cell("markdown", S8_TEXT), _cell("code", S8_CODE),
        _cell("markdown", S9_TEXT), _cell("code", S9_CODE),
        _cell("markdown", S10),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "gaze_processing", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
                "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    path = Path(__file__).resolve().parent / OUTPUT_FILENAME
    path.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
