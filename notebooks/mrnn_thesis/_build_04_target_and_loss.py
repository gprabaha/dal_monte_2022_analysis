"""Author the target-and-loss notebook (task 01 of the rebuild).

Nothing is submitted unless the reader sets ``SUBMIT = True``. Like task 00, the notebook
is the control surface: it shows the diagnosis, the variants, the state of every run, and
the selection, and is readable while the sweep is half finished.

    conda run -n gaze_processing python notebooks/mrnn_thesis/_build_01_target_and_loss.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

OUTPUT_FILENAME = "04_target_and_loss.ipynb"
_CELL_COUNTER = count(1)


HEADER = r"""# 04 · Target and loss — making the model fit the data

*Task 04 of the rebuilt mRNN analysis. Task 00 settled **how** to train; this settles
**what to train on**.*

The recipe from task 00 converges cleanly — every seed ends on its best iterate, no late
loss spikes — and reproduces the region PC trajectories well in aggregate. It has two
specific failures, and both trace to the objective rather than to the architecture.

**Interactive face is under-fitted.** The loss is an absolute MSE, so each condition's
influence is proportional to its target energy. Interactive face carries about **11%** of
that energy while producing about **half** the residual. The reason it has so little
energy is not that the neural response is weaker — it has **five times more trials** than
the other conditions, so its PSTH is the cleanest and lowest-variance estimate in the set.

**Fast structure is dropped, and only there.** The fitted model reproduces 77–99% of the
observed power above 10 Hz for the other two conditions and about **5%** for interactive
face. That content is real: split-half reliability of the region PC trajectories runs from
0.998 down to 0.90 across all 42 components.

| Section | |
|---|---|
| 1 | The objective, written out term by term |
| 2 | Where the task-00 model stands, on three axes |
| 3 | What to change, and the variants that test it |
| 4 | Run state and submission (off by default) |
| 5 | Loss trajectories |
| 6 | Results: fit, fast structure, and seed agreement |
| 7 | Selection |

**Nothing here submits a job unless you set `SUBMIT = True` in Section 4.**
"""


SETUP = r'''
# The analysis code these notebooks call lives in src/ and is edited between runs. Without
# autoreload a kernel keeps whatever it imported first, so a function added to src after
# the kernel started raises AttributeError until it is restarted -- which is easy to
# misread as a bug in the code rather than in the kernel's cache.
%load_ext autoreload
%autoreload 2

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
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_protocol as protocol
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_sweep as sweep
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_synthesis as syn
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_target_loss as tl
from dal_monte_2022_analysis.ephys.analysis import fixation_psth_noise_ceiling as ceiling_mod
from dal_monte_2022_analysis.ephys.plotting import fixation_mrnn_target_loss as viz
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
SELECTED_PROTOCOL_PATH = PROTOCOL_ROOT / "selected_protocol.yaml"
TASK_ROOT = tl.resolve_task_root(DATASET_CFG_PATH)
CEILING_DIR = ceiling_mod.resolve_output_dir(DATASET_CFG_PATH)
FIGURE_DIR = syn.resolve_output_dir(DATASET_CFG_PATH, scope="04_target_and_loss")
FIGURES = ThesisFigureSettings(output_dir=FIGURE_DIR)

#: Five seeds rather than three: seed agreement is one of the three selection axes, and
#: three seeds give only three pairs to average over.
VARIANT_SEEDS = 5

SELECTED_PROTOCOL, PROTOCOL_IS_PROVISIONAL = sweep.load_selected_protocol_or_provisional(
    SELECTED_PROTOCOL_PATH
)
ALLOW_PROVISIONAL_PROTOCOL = False


def show(figure, stem: str) -> None:
    save_thesis_figure(figure, FIGURES, stem)
    display(Image(data=figure_to_png_bytes(figure, dpi=190)))


print("inherited recipe :", SELECTED_PROTOCOL["selected_label"])
if PROTOCOL_IS_PROVISIONAL:
    print("                   ** provisional: task 00 has not frozen a recipe **")
print("task root        :", TASK_ROOT)
'''


S1_TEXT = r"""## 1. The objective, written out

Every run so far has optimised the same scalar. Written out, with the scales the task-00
recipe uses:

$$\mathcal{L}
= \underbrace{\big\langle w\,(\hat y - y)^2 \big\rangle}_{\text{pointwise, PC space}}
+ 1.0\underbrace{\big\langle w\,(\Delta\hat y - \Delta y)^2 \big\rangle}_{\text{first difference}}
+ 0.5\underbrace{\big\langle w\,(\Delta^2\hat y - \Delta^2 y)^2 \big\rangle}_{\text{second difference}}
+ 0.01\underbrace{\textstyle\sum_r \langle |W^{rr}_{\text{rec}}| \rangle}_{\text{within-region } L_1}$$

where $y$ is the 42-dimensional PC target for each region, $\Delta$ is the difference
between adjacent 10 ms bins, and $w$ is a weight over the (condition, time, component)
axes that has so far been **uniform** — that is the thing this notebook changes.

Every other term the training code supports is switched off in the current recipe:
correlation loss, variance loss, the three firing-rate backprojection terms, activity
$L_1$, and both $L_2$ penalties are all at scale 0. The cell below reads that off the
frozen config rather than restating it, so this stays true if the recipe changes.
"""

S1_CODE = r'''
import yaml

# Read the objective off a run that was actually trained under the selected recipe, so
# this section cannot drift from what the models were fitted with. Falls back to the
# frozen protocol itself when task 00 has not produced runs yet.
_winning = sorted((PROTOCOL_ROOT / "sweep" / str(SELECTED_PROTOCOL["selected_label"])).glob("seed=*"))
if _winning and (_winning[0] / "run_config.yaml").exists():
    run_config = yaml.safe_load((_winning[0] / "run_config.yaml").read_text())
else:
    display(Markdown(
        "⚠️ Task 00 has no trained runs, so the objective below is read from the frozen "
        "protocol rather than from a fitted model."
    ))
    run_config = {**SELECTED_PROTOCOL["architecture"], **SELECTED_PROTOCOL["optimizer"],
                  "epochs": SELECTED_PROTOCOL["epochs"], "loss_fn": "mse",
                  "condition_order": ("face_interactive", "face_non_interactive", "object"),
                  "correlation_loss_scale": 0.0, "variance_loss_scale": 0.0,
                  "fr_reconstruction_loss_scale": 0.0, "fr_temporal_derivative_loss_scale": 0.0,
                  "fr_temporal_curvature_loss_scale": 0.0, "l1_rate_scale": 0.0,
                  "l2_weight_scale": 0.0, "l2_rate_scale": 0.0}

loss_terms = {
    "pointwise reconstruction (PC space)": 1.0,
    "first temporal difference": run_config["temporal_derivative_loss_scale"],
    "second temporal difference": run_config["temporal_curvature_loss_scale"],
    "temporal correlation": run_config["correlation_loss_scale"],
    "variance matching": run_config["variance_loss_scale"],
    "firing-rate reconstruction": run_config["fr_reconstruction_loss_scale"],
    "firing-rate first difference": run_config["fr_temporal_derivative_loss_scale"],
    "firing-rate second difference": run_config["fr_temporal_curvature_loss_scale"],
    "within-region weight L1": run_config["l1_weight_scale"],
    "activity L1": run_config["l1_rate_scale"],
    "weight L2": run_config["l2_weight_scale"],
    "activity L2": run_config["l2_rate_scale"],
}
terms = pd.DataFrame({"term": list(loss_terms), "scale": list(loss_terms.values())})
terms["active"] = terms["scale"] > 0
display(terms)

model_parameters = {
    "hidden units per region": run_config["hidden_units"],
    "activation": run_config["activation"],
    "initial spectral radius": run_config["spectral_radius"],
    "recurrent connectivity": run_config["recurrent_connectivity"],
    "inter-regional bottleneck rank": run_config["recurrent_bottleneck_dim"],
    "PCs per region": run_config["pca_n_components"],
    "input channels": f"{len(run_config['condition_order'])} condition one-hot, {run_config['temporal_basis_count']} temporal basis",
    "trained initial state": run_config["train_initial_state"],
    "learning rate": run_config["lr"],
    "schedule": f"{run_config['lr_schedule']} to {run_config['lr_min_factor']:g} x peak",
    "gradient clip norm": run_config["gradient_clip_norm"],
    "iterations": run_config["epochs"],
    "loss function": run_config["loss_fn"],
}
display(pd.Series(model_parameters, name="task-00 model").to_frame())
'''


S2_TEXT = r"""## 2. Where the task-00 model stands

Three axes, because they disagree and the disagreement is the point.

**Fit, against the noise ceiling rather than against 1.0.** Scoring against 1.0 asks the
model to reproduce sampling noise. The ceiling is the split-half reliability of each
region's PC trajectories, built through the same smoothing and windowing the target uses,
with the basis fitted on one half only and a noise control that lands at ~0.00.

**Fast-structure recovery** — the fraction of observed spectral power the model
reproduces, by band. This is what the trace plots showed by eye, made into a number.

**Seed agreement** — how much independently seeded fits of the *same* configuration
resemble each other. A variant that fits better while making seeds disagree has not
improved the model; it has made it easier to overfit.
"""

S2_CODE = r'''
pc_ceiling = pd.read_csv(CEILING_DIR / "pc_space_ceiling.csv")
ceiling_by_region = pc_ceiling.groupby("region")["reliability"].mean().to_dict()

display(Markdown(
    "**Noise ceiling of the region PC trajectories** (mean split-half reliability over the "
    "42 components, with the pure-noise control alongside):"
))
display(
    pc_ceiling.groupby("region")
    .agg(mean_reliability=("reliability", "mean"),
         lowest_pc=("reliability", "min"),
         noise_control=("noise_control", "mean"))
    .round(4)
)

baseline_runs = sorted(
    str(d) for d in (PROTOCOL_ROOT / "sweep" / str(SELECTED_PROTOCOL["selected_label"])).glob("seed=*")
    if (d / "checkpoint_best.pth").exists()
)
HAVE_BASELINE = bool(baseline_runs)
if not HAVE_BASELINE:
    display(Markdown(
        "**Task 00 has no trained runs under the selected recipe**, so there is no baseline to "
        "characterise yet. Run task 00, then re-run this section — the diagnosis it sets up is "
        "what the variants in Section 3 are designed against."
    ))
else:
    baseline_fit = pd.concat([tl.ceiling_relative_fit(d, ceiling_by_region) for d in baseline_runs])
    baseline_recovery = pd.concat([tl.spectral_recovery(d) for d in baseline_runs])
    baseline_agreement = tl.inter_seed_agreement(baseline_runs)

    display(Markdown("**Fit, as a fraction of what the data determines:**"))
    display(baseline_fit.groupby("condition")[["r2", "ceiling", "r2_vs_ceiling"]].mean().round(4))
    display(Markdown("**Fraction of observed power reproduced:**"))
    display(baseline_recovery.pivot_table(index="band", columns="condition", values="power_ratio").round(3))
    display(Markdown("**Agreement between the seeds of this one configuration:**"))
    display(baseline_agreement.round(4))
'''

S2B_CODE = r'''
if not HAVE_BASELINE:
    display(Markdown("Deferred: no trained baseline runs."))
else:
  interactive_high = baseline_recovery[
    (baseline_recovery["condition"] == "face_interactive") & (baseline_recovery["band"] == "10-20 Hz")
]["power_ratio"].mean()
  other_high = baseline_recovery[
    (baseline_recovery["condition"] != "face_interactive") & (baseline_recovery["band"] == "10-20 Hz")
]["power_ratio"].mean()
  geometry = float(baseline_agreement.set_index("feature").loc["latent drive geometry", "mean_agreement"])
  outputs = float(baseline_agreement.set_index("feature").loc["output trajectories", "mean_agreement"])

  display(Markdown(
      f"**Read together, three things.**\n\n"
      f"1. In variance terms the model is essentially at the ceiling — interactive face scores "
      f"{baseline_fit[baseline_fit['condition'] == 'face_interactive']['r2_vs_ceiling'].mean():.3f}, "
      f"the other conditions slightly *above* 1.0, which means a little sampling noise is being "
      f"reproduced. So $R^2$ says the fit is close to as good as the data allows.\n\n"
      f"2. In spectral terms it is not. At 10–20 Hz the model reproduces "
      f"**{100 * interactive_high:.0f}%** of interactive-face power against "
      f"**{100 * other_high:.0f}%** for the other conditions. The reason $R^2$ misses this is that "
      f"high-frequency content carries very little variance: the deficit is **small in variance and "
      f"large in shape**, which is why it was visible by eye before it was visible in a number.\n\n"
      f"3. Seeds agree almost perfectly on *what the model outputs* "
      f"({outputs:.3f}) and much less on *how it produces it* — latent drive geometry "
      f"{geometry:.3f}. Task 00 fixed convergence; it did **not** fix this. Any circuit-level "
      f"claim still rests on something the seeds do not agree about, and carrying that number "
      f"forward as a selection axis is how this notebook avoids making it worse."
  ))
'''


S3_TEXT = r"""## 3. What to change

Two knobs act on the diagnosed imbalance, and two on what the dynamics can express.

**Condition weighting.** `balanced` rescales each condition so all three contribute
equally to the objective, instead of in proportion to their target energy. This is the
direct fix for interactive face commanding 11% of the loss.

**Component weighting.** Under a uniform loss the leading PCs dominate simply by carrying
more variance, and fast temporal structure lives disproportionately in the lower
components. `whiten` rescales each component by its own standard deviation so all 42
count equally; `sqrt_whiten` goes half-way. A floor of 5% of the largest scale keeps a
near-empty component from being amplified without bound.

**Spectral radius and hidden width.** These ask a different question: whether the model
*can* express the missing structure at all. The spectral radius sets how long the
network's modes persist, and width sets how many modes there are.

> **A bug found while setting this up.** `spectral_radius` was passed to the underlying
> `mrnntorch` object whose `W_rec` this wrapper overwrites, so it had **no effect on any
> run using the block parameterization** — the task-00 sweep's `sr=0.9` and `sr=1.1` cells
> produced bit-identical trained weights. It now rescales the assembled recurrent blocks
> at initialization. That makes it a usable knob for the first time, and it means the
> task-00 sweep's spectral-radius arm should be read as a duplicate of its neighbour
> rather than as evidence that the radius does not matter.

The dynamics probe runs at the best-guess weighting rather than crossed with everything:
if the weighting arm closes the gap the probe is unnecessary, and if it does not, the
probe is the next thing to try.
"""

S3_CODE = r'''
variants = tl.target_loss_variants()
seeds = protocol.protocol_seeds(n_seeds=VARIANT_SEEDS)
BASELINE = tl.baseline_label(variants)

design = pd.DataFrame([
    {"label": v.label, "condition weighting": v.condition_loss_weighting,
     "component weighting": v.pc_loss_weighting, "spectral radius": v.spectral_radius,
     "hidden units": v.hidden_units,
     "arm": "weighting factorial" if v.label.startswith("cond") else "dynamics probe"}
    for v in variants
])
display(design)
display(Markdown(
    f"**{len(variants)} variants × {len(seeds)} seeds = {len(variants) * len(seeds)} runs** at "
    f"{SELECTED_PROTOCOL['epochs']:,} iterations, everything else inherited from "
    f"`{SELECTED_PROTOCOL['selected_label']}`. The baseline cell is **`{BASELINE}`** — the task-00 "
    f"configuration refitted here under identical conditions, so every comparison has a reference "
    f"rather than importing one from another sweep."
))
'''


S4_TEXT = r"""## 4. Run state and submission

Submission happens **only** if you set `SUBMIT = True`, and is blocked while a previously
submitted array is still on the queue. Completed cells are never resubmitted, so re-running
this section is also how you top up a partial sweep.
"""

S4_CODE = r'''
SUBMIT = False   # <-- set to True to actually submit the missing cells

commands, run_dirs = tl.variant_job_commands(
    variants, seeds,
    root=TASK_ROOT,
    repo_root=repo_root,
    protocol=SELECTED_PROTOCOL,
    mrnn_cfg_path=MRNN_CFG_PATH,
)
inventory = tl.index_variant_runs(TASK_ROOT, variants, seeds)
job_state = protocol.running_job_state(TASK_ROOT / "_jobs")

if PROTOCOL_IS_PROVISIONAL:
    display(Markdown(
        "⚠️ **Task 00 has not frozen a recipe**, so the optimiser here is provisional. "
        + ("`ALLOW_PROVISIONAL_PROTOCOL` is set, so submission is allowed."
           if ALLOW_PROVISIONAL_PROTOCOL else
           "Submission is blocked; set `ALLOW_PROVISIONAL_PROTOCOL = True` to queue anyway.")
    ))

display(Markdown(
    f"**{int(inventory['complete'].sum())} complete**, "
    f"**{int(inventory['diverged'].sum())} diverged**, "
    f"**{int(inventory['pending'].sum())} not yet run** of {len(inventory)} cells."
))
if job_state["active"]:
    display(Markdown(
        f"⚠️ **Job array `{job_state['job_id']}` is still on the queue** "
        f"({', '.join(f'{n} {s.lower()}' for s, n in sorted(job_state['states'].items()))}). "
        f"Submission is blocked — the missing cells are already training."
    ))
elif commands:
    display(Markdown(f"{len(commands)} run(s) would be submitted."))
    print("first command:\n")
    print(commands[0])
'''

S4B_CODE = r'''
if job_state["active"]:
    display(Markdown(f"Nothing submitted: job array `{job_state['job_id']}` is still running."))
elif SUBMIT and commands and PROTOCOL_IS_PROVISIONAL and not ALLOW_PROVISIONAL_PROTOCOL:
    display(Markdown("**Not submitted**: task 00 has not frozen a recipe."))
elif SUBMIT and commands:
    from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

    jobs_dir = TASK_ROOT / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / "variants.txt"
    write_job_file(job_file, commands)
    job_id = submit_dsq_array_job(
        job_file_path=job_file,
        sbatch_script_path=jobs_dir / "variants.sh",
        log_dir=jobs_dir / "logs",
        job_name="mrnn_target_loss",
        partition="psych_gpu",
        cpus_per_task=1,
        mem_per_cpu="12G",
        time_limit="06:00:00",
        gres="gpu:1",
    )
    (jobs_dir / "job_id.txt").write_text(str(job_id) + "\n")
    display(Markdown(f"Submitted **{len(commands)}** runs as job array **{job_id}**."))
elif commands:
    display(Markdown("`SUBMIT` is **False** — nothing was submitted."))
else:
    display(Markdown("Every cell is already trained; go on to Section 5."))
'''


S5_TEXT = r"""## 5. Loss trajectories

One panel per variant, all seeds overlaid, with the total objective and its components.

**The panels are not comparable to each other.** Each variant optimises a different
objective — reweighting the conditions or the components changes what the number means —
so a lower curve here is *not* a better model. That is what Section 6 is for. What these
show is whether each fit converged, and which term was still moving when training stopped.
"""

S5_CODE = r'''
completed = inventory[inventory["complete"].astype(bool)]
histories = {}
for label, block in completed.groupby("label", sort=False):
    runs = []
    for run_dir in block["run_dir"]:
        history_path = Path(run_dir) / "history.csv"
        if history_path.exists():
            runs.append(pd.read_csv(history_path))
    if runs:
        histories[label] = runs

if not histories:
    display(Markdown("No completed runs yet — this section fills in as the sweep lands."))
else:
    convergence = pd.DataFrame([
        {
            "label": label,
            "n_seeds": len(runs),
            "worst_final_over_best": max(
                float(h["loss"].to_numpy()[-1] / np.nanmin(h["loss"].to_numpy())) for h in runs
            ),
            "worst_late_spikes": max(
                int(protocol.evaluate_convergence(h["loss"].to_numpy())["late_spikes"]) for h in runs
            ),
        }
        for label, runs in histories.items()
    ])
    display(convergence.round(5))
    show(viz.plot_loss_trajectories(histories), "fig01_loss_trajectories")
'''


S6_TEXT = r"""## 6. Results

Fit against the ceiling, fast-structure recovery, and seed agreement — reported
separately, because they trade against one another and which trade is acceptable is a
judgement rather than something a weighted sum can settle.
"""

S6_CODE = r'''
if not histories:
    display(Markdown("Nothing to score yet."))
else:
    fit = pd.concat([
        tl.ceiling_relative_fit(row["run_dir"], ceiling_by_region).assign(label=row["label"])
        for _, row in completed.iterrows()
    ], ignore_index=True)
    recovery = pd.concat([
        tl.spectral_recovery(row["run_dir"]).assign(label=row["label"])
        for _, row in completed.iterrows()
    ], ignore_index=True)
    agreement = pd.concat([
        tl.inter_seed_agreement(list(block["run_dir"])).assign(label=label)
        for label, block in completed.groupby("label", sort=False)
    ], ignore_index=True)

    show(viz.plot_ceiling_relative_fit(fit, baseline=BASELINE), "fig02_ceiling_relative_fit")
    show(viz.plot_spectral_recovery(recovery), "fig03_spectral_recovery")
    show(viz.plot_seed_agreement(agreement), "fig04_seed_agreement")
'''

S6B_CODE = r'''
if histories:
    scores = tl.score_variants(inventory, ceiling_by_region)
    display(scores.round(4))
    show(viz.plot_variant_tradeoff(scores, baseline=BASELINE), "fig05_variant_tradeoff")
'''


S7_TEXT = r"""## 7. Selection

The choice is made on three conditions, in this order:

1. **Seed agreement on latent drive geometry must not fall below the baseline.** This is
   the quantity every downstream circuit claim depends on, and a variant that trades it
   for fit is not an improvement for this project.
2. **Interactive-face fit must not exceed the noise ceiling.** Above 1.0 is
   noise-reproduction, not accuracy.
3. Among what remains, **maximise recovery of interactive-face power at 10–20 Hz**, which
   is the deficit this task exists to close.
"""

S7_CODE = r'''
if not histories:
    display(Markdown("Selection is deferred until the sweep completes."))
else:
    recovery_column = next(c for c in scores.columns if c.startswith("power_recovered_"))
    reference = scores.set_index("label").loc[BASELINE]
    eligible = scores[
        (scores["seed_agreement_geometry"] >= float(reference["seed_agreement_geometry"]) - 0.02)
        & (scores["interactive_r2_vs_ceiling"] <= 1.0)
    ]
    if eligible.empty:
        display(Markdown(
            "**No variant satisfies both guards.** Report that as the result rather than relaxing "
            "them: it would mean the fast structure can only be bought with either seed agreement "
            "or noise reproduction, which is itself a finding about what this model class can do."
        ))
    else:
        winner = eligible.sort_values(recovery_column, ascending=False).iloc[0]
        display(Markdown(
            f"**Selected: `{winner['label']}`.** Interactive-face power recovered at 10–20 Hz rises "
            f"from **{float(reference[recovery_column]):.3f}** (baseline) to "
            f"**{float(winner[recovery_column]):.3f}**, with interactive-face $R^2$/ceiling at "
            f"{float(winner['interactive_r2_vs_ceiling']):.3f} and seed agreement on drive geometry "
            f"at {float(winner['seed_agreement_geometry']):.3f} against the baseline's "
            f"{float(reference['seed_agreement_geometry']):.3f}."
        ))
        selected = {
            "selected_label": str(winner["label"]),
            "selection_rule": "seed agreement not below baseline, fit not above the noise ceiling, "
                              "then maximum interactive-face power recovery at 10-20 Hz",
            "inherited_protocol": str(SELECTED_PROTOCOL["selected_label"]),
            "epochs": int(SELECTED_PROTOCOL["epochs"]),
            "overrides": next(v for v in variants if v.label == winner["label"]).overrides(),
            "scores": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                       for k, v in winner.items()},
        }
        import yaml as _yaml
        path = TASK_ROOT / "selected_target_loss.yaml"
        path.write_text(_yaml.safe_dump(selected, sort_keys=False))
        display(Markdown(f"Frozen to `{path}`."))
'''


S8 = r"""## 8. What this settles, and what comes next

**Settled.** What the model is fit to and how the objective weights it, chosen against a
measured noise ceiling rather than against 1.0, and against seed agreement rather than fit
alone. Frozen to `selected_target_loss.yaml`, which task 02 onwards import.

**Reportable in its own right.**

1. The PSTH export's condition groups differ by more than 12× in trial count, and PSTH
   variance scales as 1/N across them — so a uniform absolute-error loss weights the
   conditions by how noisily they were measured. Interactive face, the best-measured
   condition, receives the least gradient.
2. Region PC trajectories are highly reliable (0.90–0.998 across all 42 components) even
   though single-unit PSTHs are not (median 0.05–0.13). Averaging a few hundred units is
   what makes the modelling target well determined at all — and it means the fast
   structure the model was dropping is signal, not noise.
3. `spectral_radius` had been inert for every run using the block parameterization.

**Still open.**

- **Seed agreement on the latent geometry stays well below 1 even for cleanly converged
  fits.** Task 00 removed the optimiser as an explanation. Whether it is the architecture,
  the bottleneck rank, or something irreducible about fitting an RNN to three trajectories
  is what tasks 03–05 have to decide, and it remains the central risk to every
  circuit-level claim.
- Whether the recovered fast structure changes the inter-regional current geometry at all.
  That is the question this task was run to make answerable, and it is answered in task 05.

**Next:** `05_seed_ensembles.ipynb`.
sweep could not answer because it varied the learning rate along with the width.
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
        _cell("markdown", S2_TEXT),
        _cell("code", S2_CODE),
        _cell("code", S2B_CODE),
        _cell("markdown", S3_TEXT),
        _cell("code", S3_CODE),
        _cell("markdown", S4_TEXT),
        _cell("code", S4_CODE),
        _cell("code", S4B_CODE),
        _cell("markdown", S5_TEXT),
        _cell("code", S5_CODE),
        _cell("markdown", S6_TEXT),
        _cell("code", S6_CODE),
        _cell("code", S6B_CODE),
        _cell("markdown", S7_TEXT),
        _cell("code", S7_CODE),
        _cell("markdown", S8),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "gaze_processing", "language": "python", "name": "python3"},
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


def main() -> None:
    path = Path(__file__).resolve().parent / OUTPUT_FILENAME
    path.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
