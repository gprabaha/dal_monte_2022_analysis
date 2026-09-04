"""Author the model-selection synthesis notebook (task 02 of the rebuilt mRNN analysis).

    conda run -n gaze_processing python notebooks/mrnn_thesis/_build_02_model_selection.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

OUTPUT_FILENAME = "02_model_selection.ipynb"
_CELL_COUNTER = count(1)


HEADER = r"""# 02 · Model selection — a minimal network that fits every fixation type

*Synthesis of tasks 00–01, and the one change they imply.*

Everything downstream — which connections are necessary, how narrow inter-regional
communication can be, whether independently fitted networks agree about the circuit —
is measured against a **base model**. This notebook settles what that model is.

The requirement is not "fits best". With enough width these trajectories can be fitted
several ways over, so the useful target is **the smallest network that reproduces every
fixation type to the limit the data supports**. Minimal matters because every constraint
result downstream is a statement about what can be removed, and a network with spare
capacity has less to remove.

Tasks 00 and 01 got most of the way there and then ran into something that width cannot
fix: **interactive-face fixations are systematically the hardest to reproduce, and not
because the network is too small.** That is a result in its own right, and it is what this
notebook diagnoses and then addresses.

| Section | |
|---|---|
| 1 | What the training sweep established, and the corrections found along the way |
| 2 | What the capacity sweep established |
| 3 | Why interactive face fits worst — the diagnosis |
| 4 | The proposed change, in words and in maths |
| 5 | What has been ruled out |
| 6 | The experiment, and its falsifiable prediction |
| 7 | Run state and submission (off by default) |
| 8 | Results |
| 9 | The base model |
"""


SETUP = r'''
# The analysis code these notebooks call lives in src/ and is edited between runs. Without
# autoreload a kernel keeps whatever it imported first.
%load_ext autoreload
%autoreload 2

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from IPython.display import Image, Markdown, display

repo_root = Path.cwd()
if not (repo_root / "src").exists():
    repo_root = next(parent for parent in Path.cwd().parents if (parent / "src").exists())
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_protocol as protocol
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_sweep as sweep
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_synthesis as syn
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_target_loss as tl
from dal_monte_2022_analysis.ephys.analysis import fixation_psth_noise_ceiling as ceiling_mod
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import replay_fixation_mrnn_run
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_training import condition_loss_weights
from dal_monte_2022_analysis.ephys.plotting import fixation_mrnn_sweep as viz
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    ThesisFigureSettings,
    apply_thesis_plot_style,
    figure_to_png_bytes,
    save_thesis_figure,
)

DATASET_CFG_PATH = repo_root / "configs" / "dataset.yaml"
MRNN_CFG_PATH = repo_root / "configs" / "ephys_fixation_mrnn.yaml"
apply_thesis_plot_style(load_config(repo_root / "configs" / "plotting.yaml"))

PROTOCOL_ROOT = protocol.resolve_chapter_root(DATASET_CFG_PATH, task="00_training_protocol")
CAPACITY_ROOT = sweep.resolve_task_root("01_capacity", DATASET_CFG_PATH)
TASK_ROOT = sweep.resolve_task_root("02_model_selection", DATASET_CFG_PATH)
CEILING_DIR = ceiling_mod.resolve_output_dir(DATASET_CFG_PATH)
FIGURE_DIR = syn.resolve_output_dir(DATASET_CFG_PATH, scope="02_model_selection")
FIGURES = ThesisFigureSettings(output_dir=FIGURE_DIR)

SELECTED_PROTOCOL, PROTOCOL_IS_PROVISIONAL = sweep.load_selected_protocol_or_provisional(
    PROTOCOL_ROOT / "selected_protocol.yaml"
)
ALLOW_PROVISIONAL_PROTOCOL = False

HIDDEN_UNIT_GRID = (20, 30, 40, 50)
SWEEP_SEEDS = 3
GALLERY_REGION = "ofc"

pc_ceiling = pd.read_csv(CEILING_DIR / "pc_space_ceiling.csv")
ceiling_by_region = pc_ceiling.groupby("region")["reliability"].mean().to_dict()


def show(figure, stem: str) -> None:
    save_thesis_figure(figure, FIGURES, stem)
    display(Image(data=figure_to_png_bytes(figure, dpi=190)))


print("recipe    :", SELECTED_PROTOCOL["selected_label"])
print("task root :", TASK_ROOT)
'''


S1_TEXT = r"""## 1. What the training sweep established

**The loss landscape is rough.** These fits are full-batch and deterministic — three
conditions, no input or activation noise — so a rise in the loss is not sampling variance,
it is the optimiser stepping uphill on a fixed surface. Under a constant learning rate that
happens throughout training and never stops: the worst configuration ended **16.6×** above
the lowest loss it had reached.

**A decaying learning rate is what makes it converge.** With a cosine schedule the step
size at the end is small enough that overshooting is impossible, so the trajectory settles
and the final iterate *is* the best iterate. That is not a cosmetic property — a run that
never settles has a best iterate that is the luckiest point on a jittery walk, and
different seeds land at different lucky points.

**The optimum is architecture-dependent**, which is worth carrying forward as a caution
rather than a footnote:
"""

S1_CODE = r'''
sweep_root = PROTOCOL_ROOT / "sweep"
protocol_histories = {}
for path in sorted(sweep_root.glob("*/seed=*/history.csv")):
    if not (path.parent / "checkpoint_best.pth").exists():
        continue
    protocol_histories.setdefault(path.parent.parent.name, []).append(pd.read_csv(path))

if not protocol_histories:
    display(Markdown("Task 00 has no trained runs on disk."))
else:
    protocol_convergence = sweep.convergence_table(protocol_histories).sort_values(
        ["n_converged", "median_best_loss"], ascending=[False, True]
    )
    display(protocol_convergence.round(6))
    display(Markdown(
        f"Selected: **`{SELECTED_PROTOCOL['selected_label']}`** — "
        f"lr {SELECTED_PROTOCOL['optimizer']['lr']:g}, {SELECTED_PROTOCOL['optimizer']['lr_schedule']} "
        f"schedule, gradient clip {SELECTED_PROTOCOL['optimizer']['gradient_clip_norm']}, "
        f"{SELECTED_PROTOCOL['optimizer']['activation']}, spectral radius "
        f"{SELECTED_PROTOCOL['optimizer']['spectral_radius']}, "
        f"{SELECTED_PROTOCOL['epochs']:,} iterations.\n\n"
        f"An earlier pass, on a model that still carried a rank-3 bottleneck and a within-region "
        f"L1 penalty, selected **lr 1e-3**. On the unconstrained model that same rate converges on "
        f"1 of 3 seeds. Removing two structural constraints moved the optimal step size by a factor "
        f"of three — so any recipe carried into a *differently* constrained model has to be "
        f"re-checked, which is why tasks 03 and 04 retry a failing variant at a lower rate before "
        f"scoring it."
    ))
'''

S1B_TEXT = r"""### 1a. Three things that were silently wrong

Found while setting the sweeps up. Each one had been inherited rather than chosen, and each
changes how earlier numbers should be read.

| | what it was | what it actually did |
|---|---|---|
| `gradient_clip_norm = 1.0` | meant as a stability guard | measured gradient norms have median 0.02, so it bound on **0.1%** of steps and had never constrained anything. 0.05 is the first threshold that binds. |
| `spectral_radius` | a swept axis | never applied to the block parameterization at all — `sr = 0.9` and `sr = 1.1` produced **bit-identical** trained weights |
| `l1_weight_scale = 0.01` | a mild sparsity prior on within-region weights | drove those blocks to **~1e-6** against ~1e-1 for the cross-region blocks — five orders of magnitude — **with no change in fit**. An ablation, not a prior. It silently removed a whole class of connection from every model in the project. |

The last one also meant `model.parameters()` was being miscounted: `mrnn.W_rec` is a copy
the forward pass overwrites from the block parameters, receives no gradient, and inflated
every parameter count about six-fold. The corrected figure for a 40-unit model is **33,448**
free parameters against 50,400 target numbers.
"""


S2_TEXT = r"""## 2. What the capacity sweep established

Widths 20–60 units per region, dense all-to-all connectivity, no rank constraint, no
within-region penalty, three seeds each. Every width converged on every seed, so the recipe
transferred across the range.

Fit is scored against the **noise ceiling** — the split-half reliability of each region's PC
trajectories — rather than against 1.0. Scoring against 1.0 would ask the model to reproduce
sampling noise, and a value above 1.0 means it is doing exactly that.
"""

S2_CODE = r'''
capacity_variants = [
    sweep.ModelVariant(label=f"h{u:02d}",
                       overrides={"hidden_units": u, "recurrent_bottleneck_dim": None,
                                  "l1_weight_scale": 0.0},
                       arm="uniform")
    for u in (20, 30, 40, 50, 60)
]
seeds = protocol.protocol_seeds(n_seeds=SWEEP_SEEDS)
capacity_inventory = sweep.index_variant_runs(CAPACITY_ROOT, capacity_variants, seeds)

if not capacity_inventory["complete"].any():
    display(Markdown("Task 01 has no trained runs on disk."))
else:
    capacity_fit = sweep.score_variant_fit(capacity_inventory, ceiling_by_region)
    capacity_params = pd.DataFrame([
        {"label": v.label, **sweep.count_trainable_parameters(
            capacity_inventory[(capacity_inventory["label"] == v.label)
                               & capacity_inventory["complete"]].iloc[0]["run_dir"])}
        for v in capacity_variants
        if capacity_inventory[(capacity_inventory["label"] == v.label) & capacity_inventory["complete"]].shape[0]
    ])
    table = (capacity_fit.groupby(["label", "condition"])["r2_vs_ceiling"].mean().unstack()
             .join(capacity_params.set_index("label")[["total", "parameters_per_datum"]]))
    display(table.round(4))
'''

S2B_CODE = r'''
if capacity_inventory["complete"].any():
    recovery = pd.concat([
        tl.spectral_recovery(capacity_inventory[(capacity_inventory["label"] == v.label)
                                                & capacity_inventory["complete"]].iloc[0]["run_dir"]).assign(label=v.label)
        for v in capacity_variants
        if capacity_inventory[(capacity_inventory["label"] == v.label) & capacity_inventory["complete"]].shape[0]
    ], ignore_index=True)
    display(Markdown("**Fraction of observed spectral power the model reproduces**, by band:"))
    for condition in ("face_interactive", "face_non_interactive"):
        display(Markdown(f"*{condition.replace('_', ' ')}*"))
        display(recovery[recovery["condition"] == condition]
                .pivot_table(index="band", columns="label", values="power_ratio").round(3))
'''

S2C_CODE = r'''
if capacity_inventory["complete"].any():
    by_condition = capacity_fit.groupby(["label", "condition"])["r2_vs_ceiling"].mean().unstack()
    at_h40 = by_condition.loc["h40"]
    hf = recovery[(recovery["band"] == "10-20 Hz")].pivot_table(
        index="label", columns="condition", values="power_ratio")
    display(Markdown(
        f"**The result that decides this notebook: width moves the three conditions together, "
        f"it does not equalise them.**\n\n"
        f"At 40 units interactive face sits at **{at_h40['face_interactive']:.3f}** of the ceiling "
        f"while the other two are already at **{at_h40['face_non_interactive']:.3f}** and "
        f"**{at_h40['object']:.3f}** — *past* it, which means reproducing sampling noise. Getting "
        f"interactive face to the ceiling needs 50 units, where the others sit at "
        f"{by_condition.loc['h50','face_non_interactive']:.3f}. So width buys interactive-face fit "
        f"only by buying more overfitting everywhere else, and doubles parameters per data point "
        f"from {float(capacity_params.set_index('label').loc['h40','parameters_per_datum']):.2f} to "
        f"{float(capacity_params.set_index('label').loc['h50','parameters_per_datum']):.2f} to do it.\n\n"
        f"The high-frequency numbers say the same thing more sharply. At 10–20 Hz interactive face "
        f"is reproduced at **{hf.loc['h40','face_interactive']:.3f}** where non-interactive face is "
        f"already at **{hf.loc['h40','face_non_interactive']:.3f}**; interactive face does not catch "
        f"up until 60 units ({hf.loc['h60','face_interactive']:.3f}). The model spends capacity in "
        f"order of gradient magnitude, and interactive face is served last."
    ))
'''


S3_TEXT = r"""## 3. Why interactive face fits worst

The natural guess is that interactive face is intrinsically harder — more fast structure to
capture, or fewer trials behind it. Both are false, and the truth points somewhere else.
"""

S3_CODE = r'''
combined = pd.read_pickle(
    load_config(DATASET_CFG_PATH)["analysis_output_root"]
    + "/ephys/psth/fixation_psth_averages/fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl"
) if False else pd.read_pickle(
    Path(load_config(DATASET_CFG_PATH)["analysis_output_root"])
    / "ephys/psth/fixation_psth_averages/fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl"
)
trials = combined.groupby(["average_partition", "fixation_category", "interactive_state"]).n_trials.median()
display(Markdown("**Trials per unit** in the export the model is fit to:"))
display(trials.to_frame("median trials"))

reference_run = capacity_inventory[(capacity_inventory["label"] == "h40") & capacity_inventory["complete"]].iloc[0]["run_dir"]
replay = replay_fixation_mrnn_run(reference_run, device="cpu")
checkpoint = replay["checkpoint"]
conditions = list(replay["condition_order"])
freqs = np.fft.rfftfreq(100, d=0.01)

power = {c: {"total": 0.0, "high": 0.0, "energy": 0.0, "residual": 0.0} for c in conditions}
for region in replay["region_order"]:
    observed = np.asarray(checkpoint["target_by_region"][region], dtype=float)
    predicted = replay["output_by_region"][region].detach().cpu().numpy()
    for index, condition in enumerate(conditions):
        centred = observed[index] - observed[index].mean(axis=0, keepdims=True)
        spectrum = (np.abs(np.fft.rfft(centred * np.hanning(100)[:, None], axis=0)) ** 2).sum(axis=1)
        power[condition]["total"] += spectrum.sum()
        power[condition]["high"] += spectrum[freqs >= 10].sum()
        power[condition]["energy"] += float(np.sum(centred ** 2))
        power[condition]["residual"] += float(np.sum((observed[index] - predicted[index]) ** 2))

spectral = pd.DataFrame(power).T
spectral["high_fraction"] = spectral["high"] / spectral["total"]
total_energy, total_residual = spectral["energy"].sum(), spectral["residual"].sum()
spectral["pct_of_loss_energy"] = 100 * spectral["energy"] / total_energy
spectral["pct_of_residual"] = 100 * spectral["residual"] / total_residual
display(spectral[["total", "high", "high_fraction", "pct_of_loss_energy", "pct_of_residual"]].round(4))
'''

S3B_CODE = r'''
row = spectral.loc["face_interactive"]
other = spectral.loc["face_non_interactive"]
display(Markdown(
    f"**Interactive face has the *most* trials and the *least* signal power.** "
    f"846 trials per unit against 168 for non-interactive face and ~85 for each half of object — "
    f"five times more. Its PSTH is therefore the cleanest estimate in the set, and consequently "
    f"the one with the least variance: **{row['total'] / other['total']:.2f}×** the total power of "
    f"non-interactive face.\n\n"
    f"**It also has less high-frequency content, not more** — "
    f"{row['high'] / other['high']:.2f}× the absolute power above 10 Hz, and a smaller fraction of "
    f"its own power up there ({row['high_fraction']:.3f} against {other['high_fraction']:.3f}). So "
    f"the failure is not that there is more fast structure to capture.\n\n"
    f"**And the objective sees it accordingly.** Interactive face commands "
    f"**{row['pct_of_loss_energy']:.1f}%** of the loss energy while producing "
    f"**{row['pct_of_residual']:.1f}%** of the residual — {row['pct_of_residual'] / row['pct_of_loss_energy']:.1f} "
    f"times more error than gradient. Equal weighting would be 33.3% each."
))
'''


S4_TEXT = r"""## 4. The proposed change

The objective is a weighted squared error over conditions $c$, time bins $t$ and target
components $j$:

$$\mathcal{L}=\sum_{c}w_c\sum_{t,j}\big(\hat y_{c,t,j}-y_{c,t,j}\big)^2$$

with $w_c=1$ throughout so far. The gradient a condition contributes scales with the size of
its residual, which scales with its target energy $E_c=\sum_{t,j}(y_{c,t,j}-\bar y_c)^2$. So
under uniform weighting **a condition's influence on the fit is proportional to its
variance**.

That would be harmless if variance reflected how much structure a condition has. It does
not. Writing each trial-averaged bin as signal plus estimation noise, $m(t)=s(t)+e(t)$ with
$N_c$ trials,

$$E_c \;\approx\; \underbrace{\textstyle\sum_t s(t)^2}_{\text{signal}} \;+\; \frac{K}{N_c}$$

and on this dataset the second term dominates — $E_c\times N_c$ is near-constant across a
five-fold range in trial count. So $E_c\propto 1/N_c$, and therefore

$$\boxed{\;\text{uniform MSE weights each condition in inverse proportion to the number of trials it was estimated from.}\;}$$

The best-measured condition receives the least gradient. That is the mechanism, and it is a
property of the objective rather than of the brain.

**The fix is one multiplier per condition**, $w_c\propto 1/E_c$ normalised to mean 1, which
makes the three contributions equal — and, by the relation above, is approximately
$w_c\propto N_c$. Nothing else changes: the targets, the PCA basis and the trajectories the
model is fit to are untouched.
"""

S4_CODE = r'''
target = torch.as_tensor(
    np.concatenate([np.asarray(checkpoint["target_by_region"][r], dtype=float)
                    for r in replay["region_order"]], axis=-1),
    dtype=torch.float32,
)
weights = condition_loss_weights(target, mode="balanced", device="cpu").flatten().numpy()
energy = (target - target.mean(dim=1, keepdim=True)).pow(2).mean(dim=(1, 2)).numpy()
counts = {
    "face_interactive": float(trials[("split", "face", "interactive")]),
    "face_non_interactive": float(trials[("split", "face", "non_interactive")]),
    "object": float(trials[("split", "object", "interactive")] + trials[("split", "object", "non_interactive")]),
}
mean_count = float(np.mean(list(counts.values())))
display(pd.DataFrame({
    "trials N": [counts[c] for c in conditions],
    "energy E": energy,
    "E x N": [energy[i] * counts[c] for i, c in enumerate(conditions)],
    "balanced weight": weights,
    "N / mean(N)": [counts[c] / mean_count for c in conditions],
}, index=conditions).round(4))
display(Markdown(
    "`E × N` is near-constant, which is the empirical content of the derivation above. The last "
    "two columns are the balanced weight computed two ways — from the target variance, and from "
    "the trial counts — and they agree."
))
'''


S5_TEXT = r"""## 5. What has been ruled out

Each of these was a plausible route to the same goal, and each was tested rather than
assumed.

| candidate | why not |
|---|---|
| **More width** | Section 2. Moves all conditions together; reaching the ceiling on interactive face requires the other two to overfit further, at 1.5× the parameters. |
| **A firing-rate reconstruction loss** | The PCA components are orthonormal to 2e−8, so $\|r V\|^2=\|r\|^2$ — the firing-rate residual *is* the PC residual, up to a per-region constant of $\sqrt{42/n_\text{units}}$ (verified numerically). It carries no information about which components or timescales matter. |
| **Component whitening** | Equalises the 42 PCs rather than the conditions, so it does not address this problem — and the noise ceiling falls from 0.991 for the top ten components to 0.954 for the bottom ten, so it would spend capacity where the data is weakest. |
| **Spectral radius** | Now that it is actually applied: 0.183 vs 0.180 recovery at 10–20 Hz for sr 0.9 vs 1.1. No effect. |
| **Rescaling the input** | The condition input is a constant one-hot and $W_{\text{in}}$ is learned, so any rescaling is absorbed into it. It sets a per-condition constant drive — an offset — and nothing else. |
| **Shared dynamics between the face conditions** | Correlation between the two face conditions' observed PC trajectories is **0.066**. They are near-independent; interactive face is not inheriting a solution shaped for its neighbour. |
| **Temporal basis input channels** | Would supply the fast structure directly, and thereby make the model input-driven rather than dynamics-driven — which destroys the question the whole chapter is asking. |
"""


S6_TEXT = r"""## 6. The experiment

Widths 20–50 under **balanced** condition weighting, against the uniform runs task 01
already fitted. Everything else is identical — same recipe, same seeds, same architecture —
so the only difference is $w_c$.

**The prediction, stated before the runs.** If the diagnosis is right, balancing should let a
*smaller* network reach the ceiling on interactive face, and should do so **without pushing
the other two conditions further past it**. Concretely: h40 balanced should reach
$R^2/\text{ceiling}\approx1.0$ on interactive face while non-interactive face and object stay
near 1.0 rather than climbing to 1.02.

If instead every condition simply moves up together, or interactive face improves only by the
others getting worse, the diagnosis is wrong and width really was the binding constraint.
"""

S6_CODE = r'''
balanced_variants = [
    sweep.ModelVariant(
        label=f"h{u:02d}_balanced",
        overrides={"hidden_units": u, "recurrent_bottleneck_dim": None,
                   "l1_weight_scale": 0.0, "condition_loss_weighting": "balanced"},
        arm="balanced",
    )
    for u in HIDDEN_UNIT_GRID
]
display(pd.DataFrame([v.describe() for v in balanced_variants]))
display(Markdown(
    f"**{len(balanced_variants)} variants × {len(seeds)} seeds = {len(balanced_variants) * len(seeds)} runs.** "
    f"The uniform arm is read from task 01 rather than refitted, since it is the same "
    f"configuration at the same seeds."
))
'''


S7_TEXT = r"""## 7. Run state and submission"""

S7_CODE = r'''
SUBMIT = False   # <-- set to True to actually submit the missing cells

commands, run_dirs = sweep.variant_job_commands(
    balanced_variants, seeds, root=TASK_ROOT, repo_root=repo_root,
    protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH,
)
inventory = sweep.index_variant_runs(TASK_ROOT, balanced_variants, seeds)
job_state = protocol.running_job_state(TASK_ROOT / "_jobs")

if PROTOCOL_IS_PROVISIONAL:
    display(Markdown(
        "⚠️ **Task 00 has not frozen a recipe**, so the optimiser here is provisional. "
        + ("`ALLOW_PROVISIONAL_PROTOCOL` is set, so submission is allowed."
           if ALLOW_PROVISIONAL_PROTOCOL else
           "Submission is blocked; set `ALLOW_PROVISIONAL_PROTOCOL = True` to queue anyway.")
    ))
display(Markdown(
    f"**{int(inventory['complete'].sum())} complete**, **{int(inventory['diverged'].sum())} diverged**, "
    f"**{int(inventory['pending'].sum())} not yet run** of {len(inventory)} cells."
))
if job_state["active"]:
    display(Markdown(
        f"⚠️ **Job array `{job_state['job_id']}` is still on the queue** "
        f"({', '.join(f'{n} {s.lower()}' for s, n in sorted(job_state['states'].items()))})."
    ))
elif commands:
    display(Markdown(f"{len(commands)} run(s) would be submitted."))
    print("first command:\n")
    print(commands[0])
'''

S7B_CODE = r'''
if job_state["active"]:
    display(Markdown(f"Nothing submitted: job array `{job_state['job_id']}` is still running."))
elif SUBMIT and commands and PROTOCOL_IS_PROVISIONAL and not ALLOW_PROVISIONAL_PROTOCOL:
    display(Markdown("**Not submitted**: task 00 has not frozen a recipe."))
elif SUBMIT and commands:
    from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

    jobs_dir = TASK_ROOT / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / "balanced.txt"
    write_job_file(job_file, commands)
    job_id = submit_dsq_array_job(
        job_file_path=job_file, sbatch_script_path=jobs_dir / "balanced.sh",
        log_dir=jobs_dir / "logs", job_name="mrnn_balanced", partition="psych_gpu",
        cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
    )
    (jobs_dir / "job_id.txt").write_text(str(job_id) + "\n")
    display(Markdown(f"Submitted **{len(commands)}** runs as job array **{job_id}**."))
elif commands:
    display(Markdown("`SUBMIT` is **False** — nothing was submitted."))
else:
    display(Markdown("Every cell is already trained; go on to Section 8."))
'''


S8_TEXT = r"""## 8. Results

Convergence first — a reweighted objective is a different objective, so it has to be shown
to optimise before its fit is read.
"""

S8_CODE = r'''
histories = sweep.load_histories(inventory)
if not histories:
    display(Markdown("No balanced runs yet — this section fills in as the sweep lands."))
else:
    display(sweep.convergence_table(histories).round(6))
    show(viz.plot_sweep_loss_trajectories(histories, convergence=sweep.convergence_table(histories),
                                          n_columns=4), "fig01_balanced_loss_trajectories")
'''

S8B_CODE = r'''
if histories:
    balanced_fit = sweep.score_variant_fit(inventory, ceiling_by_region)
    both = pd.concat([
        capacity_fit.assign(weighting="uniform",
                            width=capacity_fit["label"].str.extract(r"h(\d+)")[0].astype(int)),
        balanced_fit.assign(weighting="balanced",
                            width=balanced_fit["label"].str.extract(r"h(\d+)")[0].astype(int)),
    ], ignore_index=True)
    table = (both.groupby(["width", "weighting", "condition"])["r2_vs_ceiling"].mean()
             .unstack("condition").round(4))
    display(table)
'''

S8C_CODE = r'''
if histories:
    balanced_recovery = pd.concat([
        tl.spectral_recovery(inventory[(inventory["label"] == v.label) & inventory["complete"]].iloc[0]["run_dir"]).assign(
            width=int(v.overrides["hidden_units"]), weighting="balanced")
        for v in balanced_variants
        if inventory[(inventory["label"] == v.label) & inventory["complete"]].shape[0]
    ], ignore_index=True)
    uniform_recovery = recovery.assign(
        width=recovery["label"].str.extract(r"h(\d+)")[0].astype(int), weighting="uniform")
    hf = pd.concat([uniform_recovery, balanced_recovery], ignore_index=True)
    hf = hf[(hf["band"] == "10-20 Hz") & (hf["condition"] == "face_interactive")]
    display(Markdown("**Interactive-face power reproduced at 10–20 Hz**, by width and weighting:"))
    display(hf.pivot_table(index="width", columns="weighting", values="power_ratio").round(3))
'''

S8D_CODE = r'''
if histories:
    pivot = table.copy()
    verdict = []
    for width in sorted({w for w, _ in pivot.index}):
        if ("uniform" not in [g for w, g in pivot.index if w == width]
                or "balanced" not in [g for w, g in pivot.index if w == width]):
            continue
        u, b = pivot.loc[(width, "uniform")], pivot.loc[(width, "balanced")]
        verdict.append({
            "width": width,
            "int face uniform": u["face_interactive"], "int face balanced": b["face_interactive"],
            "others uniform": u[["face_non_interactive", "object"]].mean(),
            "others balanced": b[["face_non_interactive", "object"]].mean(),
        })
    verdict = pd.DataFrame(verdict).set_index("width")
    verdict["int face gain"] = verdict["int face balanced"] - verdict["int face uniform"]
    verdict["others overshoot change"] = (verdict["others balanced"] - 1.0).abs() - (verdict["others uniform"] - 1.0).abs()
    display(verdict.round(4))
    display(Markdown(
        "The prediction in Section 6 is confirmed if **int face gain** is positive while "
        "**others overshoot change** is zero or negative — interactive face rising toward the "
        "ceiling without the other conditions moving further past it. A positive value in the last "
        "column would mean the fit was simply redistributed, not improved."
    ))
'''

S8E_CODE = r'''
if histories:
    gallery = pd.concat([
        sweep.gallery_traces(inventory, region=GALLERY_REGION, space="pc", indices=(0, 1, 2)),
        sweep.gallery_traces(capacity_inventory, region=GALLERY_REGION, space="pc", indices=(0, 1, 2)),
    ], ignore_index=True)
    order = [f"h{u:02d}" for u in HIDDEN_UNIT_GRID] + [f"h{u:02d}_balanced" for u in HIDDEN_UNIT_GRID]
    order = [label for label in order if label in set(gallery["label"])]
    show(viz.plot_fit_gallery(gallery, order=order,
                              title=f"{GALLERY_REGION.upper()} — top three PCs, uniform above, balanced below"),
         "fig02_gallery_uniform_vs_balanced")
'''


S9_TEXT = r"""## 9. The base model

The rule follows the aim: **the narrowest configuration that reaches the noise ceiling on
every condition, without exceeding it on any.** Exceeding the ceiling is not a better fit;
it is reproducing sampling noise, and it costs the parameters that later constraint results
have to be measured against.
"""

S9_CODE = r'''
if not histories:
    display(Markdown("Selection is deferred until the balanced sweep completes."))
else:
    TOLERANCE = 0.015
    candidates = []
    for (width, weighting), row in table.iterrows():
        reaches = float(row.min()) >= 1.0 - TOLERANCE
        exceeds = float(row.max()) > 1.0 + TOLERANCE
        candidates.append({"width": width, "weighting": weighting,
                           "worst_condition": float(row.min()), "best_condition": float(row.max()),
                           "reaches_ceiling": reaches, "exceeds_ceiling": exceeds})
    candidates = pd.DataFrame(candidates)
    display(candidates.round(4))
    usable = candidates[candidates["reaches_ceiling"] & ~candidates["exceeds_ceiling"]]
    if usable.empty:
        display(Markdown(
            "**No configuration reaches every condition's ceiling without exceeding it on another.** "
            "That is itself the result: on this data the conditions cannot be fitted to the same "
            "standard simultaneously, and any downstream claim has to carry that."
        ))
    else:
        winner = usable.sort_values("width").iloc[0]
        selected = {
            "hidden_units": int(winner["width"]),
            "condition_loss_weighting": str(winner["weighting"]),
            "recurrent_bottleneck_dim": None,
            "l1_weight_scale": 0.0,
            "recurrent_connectivity": "full",
            "inherited_protocol": str(SELECTED_PROTOCOL["selected_label"]),
            "epochs": int(SELECTED_PROTOCOL["epochs"]),
            "selection_rule": "narrowest configuration reaching every condition's noise ceiling "
                              f"without exceeding it, tolerance {TOLERANCE}",
            "worst_condition_r2_vs_ceiling": float(winner["worst_condition"]),
        }
        (TASK_ROOT / "selected_base_model.yaml").write_text(yaml.safe_dump(selected, sort_keys=False))
        display(Markdown(
            f"**Base model: {int(winner['width'])} units per region, {winner['weighting']} condition "
            f"weighting**, dense all-to-all connectivity, no within-region penalty. Worst condition "
            f"sits at {float(winner['worst_condition']):.3f} of its ceiling.\n\n"
            f"Frozen to `{TASK_ROOT / 'selected_base_model.yaml'}`, which tasks 03 and 04 import."
        ))
'''


S10 = r"""## 10. What this settles

**The base model** every later task is measured against: how wide, and how the objective
weights the conditions — both chosen against a measured noise ceiling rather than against a
loss value, and both minimal in the sense that nothing smaller reaches that ceiling.

**Two results in their own right**, independent of what follows:

1. **Interactive-face fixations are systematically the hardest to reproduce, and not because
   the network is too small.** They carry the most trials, the least variance and the least
   high-frequency power, and an absolute-error objective therefore weights them in inverse
   proportion to how well they were measured. Width can compensate only by letting the other
   conditions overfit.
2. **The optimisation landscape is rough, and the recipe is architecture-dependent.** Under a
   constant learning rate these fits never settle; the optimal rate moved threefold when two
   structural constraints were removed. Both facts constrain how any later comparison between
   architectures can be read, which is why the constraint sweeps retry a failing variant at a
   lower rate before scoring it.

**Next:** `03_connectivity.ipynb` — with the base model fixed, which connections are
necessary. Then `04_bottleneck_rank.ipynb` for how narrow inter-regional communication can be.

Both import `selected_base_model.yaml`, which carries **the condition weighting as well as
the width**. That matters more than it looks: a constraint result measured under an
objective that under-fits one condition fivefold is partly a statement about the objective.
Fixing the weighting here means every later comparison — which connections are necessary,
how narrow the inter-regional channel can be, whether independently seeded fits agree — is
made on a model that reproduces all three fixation types to the same standard.
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
        _cell("markdown", S1_TEXT),
        _cell("code", S1_CODE),
        _cell("markdown", S1B_TEXT),
        _cell("markdown", S2_TEXT),
        _cell("code", S2_CODE),
        _cell("code", S2B_CODE),
        _cell("code", S2C_CODE),
        _cell("markdown", S3_TEXT),
        _cell("code", S3_CODE),
        _cell("code", S3B_CODE),
        _cell("markdown", S4_TEXT),
        _cell("code", S4_CODE),
        _cell("markdown", S5_TEXT),
        _cell("markdown", S6_TEXT),
        _cell("code", S6_CODE),
        _cell("markdown", S7_TEXT),
        _cell("code", S7_CODE),
        _cell("code", S7B_CODE),
        _cell("markdown", S8_TEXT),
        _cell("code", S8_CODE),
        _cell("code", S8B_CODE),
        _cell("code", S8C_CODE),
        _cell("code", S8D_CODE),
        _cell("code", S8E_CODE),
        _cell("markdown", S9_TEXT),
        _cell("code", S9_CODE),
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
