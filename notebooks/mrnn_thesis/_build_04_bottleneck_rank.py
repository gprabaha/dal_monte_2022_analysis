"""Author the bottleneck-rank notebook (task 03 of the rebuilt mRNN analysis).

    conda run -n gaze_processing python notebooks/mrnn_thesis/_build_03_bottleneck_rank.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

OUTPUT_FILENAME = "04_bottleneck_rank.ipynb"
_CELL_COUNTER = count(1)


HEADER = r"""# 04 · Inter-regional bottleneck — how narrow can the channel be?

*Task 04 of the rebuilt mRNN analysis. Task 02 fixed the base model, task 03 established
which blocks have to exist; this asks how much has to pass through the ones that do.*

Each inter-regional block is written $W_{s\to t}=LR$ with $L$ of shape
$(n_s\times r)$ and $R$ of shape $(r\times n_t)$, so all communication from one region to
another passes through $r$ dimensions. The **dense, unconstrained block is the baseline**,
and every rank is measured against it.

**What the legacy analysis got wrong here, and what this fixes.**

| | legacy | here |
|---|---|---|
| baseline | none — every run was already factorized | dense blocks, fitted alongside |
| training length | ranks 1–5 at 100k, ranks 10–30 at 50k | every rank at the same iteration count |
| checkpoint | final iterate | best iterate |
| seeds | one per rank | five per rank |
| scored against | 1.0 | the measured noise ceiling |

The legacy sweep concluded that fit is *non-monotone* in rank — flat from 3 to 5, worse at
10 and 20, partially recovering at 30 — which a correctly converged sweep cannot produce,
since a rank-20 model contains every rank-5 model as a special case. That was a
training-length artefact. Epoch-matching is the point of redoing it.

**Note that `rank = hidden width` is not the dense baseline.** The product parameterization
reaches full rank but carries twice the parameters and optimizes differently, so both are
fitted and reported separately.

| Section | |
|---|---|
| 1 | What the earlier tasks settled, and the variants |
| 2 | Run state and submission (off by default) |
| 3 | Loss trajectories and convergence |
| 4 | Fit against the ceiling, by rank |
| 5 | The visual check across ranks |
| 6 | Seed agreement — does a narrower channel make the circuit identifiable? |
| 7 | What rank the unconstrained model actually used |
| 8 | Reading the rank result |
| 9 | Sparsity and low rank as a matched-budget comparison |
| 10 | The combination, verified jointly |
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
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import load_fixation_mrnn_checkpoint
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
MODEL_SELECTION_ROOT = sweep.resolve_task_root("02_model_selection", DATASET_CFG_PATH)
CONNECTIVITY_ROOT = sweep.resolve_task_root("02_connectivity", DATASET_CFG_PATH)
TASK_ROOT = sweep.resolve_task_root("03_bottleneck_rank", DATASET_CFG_PATH)
CEILING_DIR = ceiling_mod.resolve_output_dir(DATASET_CFG_PATH)
FIGURE_DIR = syn.resolve_output_dir(DATASET_CFG_PATH, scope="04_bottleneck_rank")
FIGURES = ThesisFigureSettings(output_dir=FIGURE_DIR)

SELECTED_PROTOCOL, PROTOCOL_IS_PROVISIONAL = sweep.load_selected_protocol_or_provisional(
    PROTOCOL_ROOT / "selected_protocol.yaml"
)

#: Set True to queue this sweep before task 00 has frozen a recipe. Only the optimiser is
#: provisional in that case -- the architecture comes from the corrected
#: PROTOCOL_ARCHITECTURE either way -- and task 00's own confirmation grid is what would
#: overturn it. Worth doing when the cluster is the bottleneck; the run configs record
#: exactly what was used, so a mismatch stays detectable.
ALLOW_PROVISIONAL_PROTOCOL = False

# The base model comes from task 02, which fixes both the width and how the objective
# weights the fixation conditions. Both matter here: a constraint result measured under an
# objective that under-fits one condition fivefold is a statement about the objective as
# much as about the constraint, so every variant below inherits the weighting too.
BASE_MODEL_PATH = MODEL_SELECTION_ROOT / "selected_base_model.yaml"
CAPACITY_PATH = CAPACITY_ROOT / "selected_capacity.yaml"
if BASE_MODEL_PATH.exists():
    BASE_MODEL = yaml.safe_load(BASE_MODEL_PATH.read_text())
    HIDDEN_UNITS = int(BASE_MODEL["hidden_units"])
    CONDITION_WEIGHTING = str(BASE_MODEL["condition_loss_weighting"])
    SELECTED_CAPACITY = BASE_MODEL
    CAPACITY_SOURCE = f"task 02 (`{HIDDEN_UNITS}` units, `{CONDITION_WEIGHTING}` weighting)"
elif CAPACITY_PATH.exists():
    BASE_MODEL = None
    SELECTED_CAPACITY = yaml.safe_load(CAPACITY_PATH.read_text())
    HIDDEN_UNITS = int(SELECTED_CAPACITY["hidden_units"])
    CONDITION_WEIGHTING = "uniform"
    CAPACITY_SOURCE = (f"task 01 (`{SELECTED_CAPACITY['selected_label']}`) with **provisional "
                       f"uniform weighting** — task 02 has not finished")
else:
    BASE_MODEL = None
    SELECTED_CAPACITY = None
    HIDDEN_UNITS = 50
    CONDITION_WEIGHTING = "uniform"
    CAPACITY_SOURCE = "**provisional fallback** — tasks 01 and 02 have not finished"

#: Set True to queue this sweep before task 01 has chosen a width. Cluster time is the
#: scarce resource and these arrays take hours, so waiting for a clean dependency can cost
#: more than the risk: if task 01 selects a different width, the runs fitted here are at
#: the wrong one and Section 1 will say so. The width every cell was actually trained at is
#: recorded in its own run_config.yaml either way, so a mismatch is always detectable.
#: Set True to queue before task 02 has fixed the base model. Both the width and the
#: condition weighting would then be provisional.
ALLOW_PROVISIONAL_WIDTH = False

#: Ranks to fit, plus a dense baseline. Chosen to span from a single shared dimension up
#: to and past the hidden width, so the curve can be read end to end at one iteration count.
RANK_GRID = (1, 2, 3, 5, 10, 20)
SWEEP_SEEDS = 3
GALLERY_REGION = "ofc"


def show(figure, stem: str) -> None:
    save_thesis_figure(figure, FIGURES, stem)
    display(Image(data=figure_to_png_bytes(figure, dpi=190)))


if PROTOCOL_IS_PROVISIONAL:
    print("recipe       : ** provisional: task 00 has not frozen one **")
print("hidden units :", HIDDEN_UNITS)
print("task root    :", TASK_ROOT)
'''


S1_TEXT = r"""## 1. The variants

Every cell shares the width from task 01, the connectivity task 02 supports, and the
recipe from task 00 — **including the iteration count**, which is the thing the legacy
sweep failed to hold fixed.
"""

S1_CODE = r'''
CONNECTIVITY = "full"
findings_path = CONNECTIVITY_ROOT / "connectivity_findings.csv"
if findings_path.exists():
    display(Markdown(
        f"Task 02 findings are available at `{findings_path}`. Connectivity here is `{CONNECTIVITY}`; "
        f"if task 02 showed some blocks are dispensable, restrict this sweep to the surviving ones "
        f"by setting `recurrent_blocked_pairs` in the overrides below."
    ))

# l1_weight_scale is set to 0 rather than inherited. The task-00 recipe carries 0.01,
# which came from the legacy ensembles and was never chosen -- and at that value it is not
# a sparsity prior but an ablation: in the fitted h40 model it drives the within-region
# blocks to ~1e-6 against ~1e-1 for the cross-region blocks, five orders of magnitude
# down. That would make `within_region_only` below a model with almost no recurrence at
# all, and would make `full` and `cross_plus_self_diagonal` near-duplicates. Sparsity is
# swept properly in task 04.
base_overrides = {"hidden_units": HIDDEN_UNITS, "recurrent_connectivity": CONNECTIVITY,
                  "l1_weight_scale": 0.0,
                  "condition_loss_weighting": CONDITION_WEIGHTING}

variants = [
    sweep.ModelVariant(label="dense", arm="baseline",
                       overrides={**base_overrides, "recurrent_bottleneck_dim": None}),
]
variants += [
    sweep.ModelVariant(label=f"rank{rank:02d}", arm="rank constrained",
                       overrides={**base_overrides, "recurrent_bottleneck_dim": int(rank)})
    for rank in RANK_GRID
]
seeds = protocol.protocol_seeds(n_seeds=SWEEP_SEEDS)
BASELINE = "dense"

display(Markdown(f"Width **{HIDDEN_UNITS}** units per region, from {CAPACITY_SOURCE}."))

# What the *trained* runs were fitted at, which may predate task 01's answer. Only runs
# with a checkpoint count: generating the job commands writes a run_config.yaml for every
# cell whether or not it is ever submitted, so staged configs are not evidence of anything.
_trained = sorted(
    path for path in TASK_ROOT.glob("*/seed=*/run_config.yaml")
    if (path.parent / "checkpoint_best.pth").exists()
)
if _trained:
    # Compare every setting the base model fixes, not just the width. A run fitted under a
    # different objective answers a different question, and pooling it with the others
    # would be invisible in the results table.
    _configs = [yaml.safe_load(path.read_text()) for path in _trained]
    _expected = {"hidden_units": HIDDEN_UNITS, "condition_loss_weighting": CONDITION_WEIGHTING,
                 "l1_weight_scale": 0.0, "recurrent_bottleneck_dim": None}
    _stale = {
        key: sorted({str(cfg.get(key)) for cfg in _configs})
        for key, want in _expected.items()
        if {str(cfg.get(key)) for cfg in _configs} != {str(want)}
    }
    if _stale:
        display(Markdown(
            f"🔴 **{len(_trained)} trained run(s) do not match the base model.** "
            + "; ".join(f"`{k}` is {v} but should be `{_expected[k]}`" for k, v in _stale.items())
            + ". Those runs answer the question under different settings and must be refitted "
              "before anything below is read — move them aside and re-submit."
        ))
    else:
        display(Markdown(
            f"{len(_trained)} trained run(s), all matching the base model "
            f"({HIDDEN_UNITS} units, `{CONDITION_WEIGHTING}` weighting)."
        ))
display(pd.DataFrame([v.describe() for v in variants]))
display(Markdown(
    f"**{len(variants)} variants × {len(seeds)} seeds = {len(variants) * len(seeds)} runs**, all at "
    f"{SELECTED_PROTOCOL['epochs']:,} iterations."
))
'''


S2_TEXT = r"""## 2. Run state and submission"""

S2_CODE = r'''
SUBMIT = False   # <-- set to True to actually submit the missing cells

# Every job directory this task submits into. A live array in any of them owns cells that
# are indistinguishable on disk from cells that were never run -- a checkpoint is written
# only when training ends -- so all of them have to be consulted before deciding what is
# genuinely unqueued. That includes the sparsity arm in Section 9, which submits
# separately and may well be in flight while this section is re-run.
JOBS_DIRS = [TASK_ROOT / "_jobs", TASK_ROOT / "_jobs_sparsity", TASK_ROOT / "_jobs_retry"]
flight = sweep.in_flight_run_dirs(*JOBS_DIRS)

commands, run_dirs = sweep.variant_job_commands(
    variants, seeds, root=TASK_ROOT, repo_root=repo_root,
    protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH,
    exclude_run_dirs=flight["run_dirs"],
)
inventory = sweep.index_variant_runs(TASK_ROOT, variants, seeds, in_flight=flight["run_dirs"])

if PROTOCOL_IS_PROVISIONAL:
    display(Markdown(
        "⚠️ **Task 00 has not frozen a recipe**, so the optimiser here is the provisional one "
        "(`lr 3e-4`, cosine, clip 0.05). The architecture is not provisional. "
        + ("`ALLOW_PROVISIONAL_PROTOCOL` is set, so submission is allowed."
           if ALLOW_PROVISIONAL_PROTOCOL else
           "Submission is blocked; set `ALLOW_PROVISIONAL_PROTOCOL = True` to queue anyway.")
    ))

if BASE_MODEL is None:
    display(Markdown(
        f"⚠️ **Task 02 has not fixed the base model**, so these cells would be fitted at the "
        f"provisional width of **{HIDDEN_UNITS}** units with **{CONDITION_WEIGHTING}** condition "
        f"weighting. "
        + ("`ALLOW_PROVISIONAL_WIDTH` is set, so submission is allowed."
           if ALLOW_PROVISIONAL_WIDTH else
           "Submission is blocked; set `ALLOW_PROVISIONAL_WIDTH = True` to queue anyway.")
    ))

display(Markdown(
    f"**{int(inventory['complete'].sum())} complete**, **{int(inventory['queued'].sum())} on the "
    f"queue**, **{int(inventory['diverged'].sum())} diverged**, "
    f"**{int(inventory['pending'].sum())} unqueued** of {len(inventory)} cells."
))
display(Markdown(sweep.describe_in_flight(flight)))
if commands:
    display(Markdown(
        f"{len(commands)} run(s) would be submitted — the unqueued cells only. Cells a live array "
        f"already owns are skipped, so this is safe to re-run while jobs are on the queue."
    ))
    print("first command:\n")
    print(commands[0])
'''

S2B_CODE = r'''
if SUBMIT and commands and BASE_MODEL is None and not ALLOW_PROVISIONAL_WIDTH:
    display(Markdown(
        f"**Not submitted**: task 02 has not fixed the base model, so these cells would be fitted at "
        f"the provisional {HIDDEN_UNITS} units. Set `ALLOW_PROVISIONAL_WIDTH = True` to queue them "
        f"anyway — worth doing when the cluster is the bottleneck, since a mismatch is detectable "
        f"afterwards and only costs a refit."
    ))
elif SUBMIT and commands and PROTOCOL_IS_PROVISIONAL and not ALLOW_PROVISIONAL_PROTOCOL:
    display(Markdown(
        "**Not submitted**: task 00 has not frozen a recipe. Set "
        "`ALLOW_PROVISIONAL_PROTOCOL = True` to queue on the provisional optimiser."
    ))
elif SUBMIT and commands:
    from datetime import datetime

    from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

    jobs_dir = TASK_ROOT / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    # Timestamped, so a submission made while an earlier array is still live keeps its own
    # command list and logs instead of overwriting the record of what is running.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_file = jobs_dir / f"bottleneck_{stamp}.txt"
    write_job_file(job_file, commands)
    submitted = [Path(line.split("--run-dir ")[1].split()[0]) for line in commands]
    job_id = submit_dsq_array_job(
        job_file_path=job_file, sbatch_script_path=jobs_dir / f"bottleneck_{stamp}.sh",
        log_dir=jobs_dir / "logs", job_name="mrnn_bottleneck", partition="psych_gpu",
        cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
    )
    sweep.record_submission(jobs_dir, job_id=job_id, run_dirs=submitted, label=f"rank {stamp}")
    (jobs_dir / "job_id.txt").write_text(str(job_id) + "\n")
    display(Markdown(f"Submitted **{len(commands)}** runs as job array **{job_id}**."))
elif commands:
    display(Markdown("`SUBMIT` is **False** — nothing was submitted."))
elif int(inventory["queued"].sum()):
    display(Markdown(
        f"Nothing to submit: the {int(inventory['queued'].sum())} outstanding cells are already "
        f"on the queue."
    ))
else:
    display(Markdown("Every cell is already trained; go on to Section 3."))
'''


S3_TEXT = r"""## 3. Loss trajectories and convergence

The legacy sweep's non-monotone rank curve was a training-length artefact. With the
iteration count held fixed, any remaining non-monotonicity has to be explained here — a
higher rank that fits worse than a lower one is either an optimisation failure, visible in
these trajectories, or a genuine finding about the loss landscape.
"""

S3_CODE = r'''
histories = sweep.load_histories(inventory)
labels = [v.label for v in variants if v.label in histories]

if not histories:
    display(Markdown("No completed runs yet — this section fills in as the sweep lands."))
else:
    convergence = sweep.convergence_table(histories)
    display(convergence.round(5))
    show(viz.plot_sweep_loss_trajectories(histories, convergence=convergence, n_columns=4),
         "fig01_loss_trajectories")
'''


S3B_TEXT = r"""### 3a. Constraints that need a smaller step

The recipe is selected on the **unconstrained** model, and the optimum is
architecture-dependent — on this problem the best learning rate moved by a factor of three
when the rank constraint and the within-region penalty were removed. So a constrained
variant that fails the bar has two possible explanations, and they mean opposite things:

- the data cannot be reproduced under that constraint — the result this sweep is for;
- the inherited recipe cannot *optimise* it — a statement about the optimiser, not the brain.

Scoring a failure without separating them would report the second as the first. Any variant
that fails is therefore refitted at successively lower learning rates, and which rate each
one needed is reported rather than hidden: "this constraint required a smaller step" is
itself informative about the loss landscape it induces.
"""

S3B_CODE = r'''
if histories:
    retries = sweep.retry_variants_for_nonconverged(
        variants, convergence, inherited_lr=float(SELECTED_PROTOCOL["optimizer"]["lr"])
    )
    if not retries:
        display(Markdown("Every variant converged at the inherited learning rate; no retries needed."))
    else:
        retry_commands, _ = sweep.variant_job_commands(
            retries, seeds, root=TASK_ROOT, repo_root=repo_root,
            protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH,
            exclude_run_dirs=flight["run_dirs"],
        )
        retry_inventory = sweep.index_variant_runs(
            TASK_ROOT, retries, seeds, in_flight=flight["run_dirs"]
        )
        display(Markdown(
            f"**{len(set(r.label.split('__')[0] for r in retries))} variant(s) failed the bar**, "
            f"giving {len(retries)} retry configurations "
            f"({int(retry_inventory['complete'].sum())} already trained, {len(retry_commands)} to run). "
            f"Submit them the same way as Section 2, then re-run this notebook."
        ))
        if retry_commands and SUBMIT:
            from datetime import datetime

            from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

            jobs_dir = TASK_ROOT / "_jobs_retry"
            jobs_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_file = jobs_dir / f"retry_{stamp}.txt"
            write_job_file(job_file, retry_commands)
            retry_dirs = [Path(line.split("--run-dir ")[1].split()[0]) for line in retry_commands]
            retry_id = submit_dsq_array_job(
                job_file_path=job_file, sbatch_script_path=jobs_dir / f"retry_{stamp}.sh",
                log_dir=jobs_dir / "logs", job_name="mrnn_retry", partition="psych_gpu",
                cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
            )
            sweep.record_submission(
                jobs_dir, job_id=retry_id, run_dirs=retry_dirs, label=f"retry {stamp}"
            )
            (jobs_dir / "job_id.txt").write_text(str(retry_id) + "\n")
            display(Markdown(f"Submitted **{len(retry_commands)}** retries as job array **{retry_id}**."))

        retry_histories = sweep.load_histories(retry_inventory)
        if retry_histories:
            combined = pd.concat([convergence, sweep.convergence_table(retry_histories)], ignore_index=True)
            resolved = sweep.resolve_best_converged(combined, base_labels=[v.label for v in variants])
            display(resolved.round(6))
            inventory = pd.concat([inventory, retry_inventory], ignore_index=True)
            histories = {**histories, **retry_histories}
            labels = list(resolved.loc[resolved["converged"], "used_label"])
'''


S4_TEXT = r"""## 4. Fit by rank

The question is where the curve flattens: the lowest rank whose fit is indistinguishable
from dense is the width of the inter-regional channel the data actually requires.
"""

S4_CODE = r'''
pc_ceiling = pd.read_csv(CEILING_DIR / "pc_space_ceiling.csv")
ceiling_by_region = pc_ceiling.groupby("region")["reliability"].mean().to_dict()

if not histories:
    display(Markdown("Nothing to score yet."))
else:
    fit = sweep.score_variant_fit(inventory, ceiling_by_region)
    parameters = pd.DataFrame([
        {"label": label, **sweep.count_trainable_parameters(
            inventory[(inventory["label"] == label) & inventory["complete"]].iloc[0]["run_dir"])}
        for label in labels
    ])
    summary = (
        fit.groupby("label")["r2_vs_ceiling"].agg(["mean", "min"])
        .join(parameters.set_index("label")[["inter_region", "total", "parameters_per_datum"]])
        .loc[labels]
    )
    reference = float(summary.loc[BASELINE, "mean"]) if BASELINE in summary.index else np.nan
    summary["cost_vs_dense"] = reference - summary["mean"]
    display(summary.round(4))
    show(viz.plot_fit_versus_constraint(fit, parameters, order=labels,
                                        x_label="inter-regional rank"), "fig02_fit_vs_rank")
    display(Markdown("By condition:"))
    display(fit.groupby(["label", "condition"])["r2_vs_ceiling"].mean().unstack().loc[labels].round(4))
'''


S5_TEXT = r"""## 5. The visual check across ranks

A rank low enough to distort the trajectories will usually show it as smoothing before it
shows it in $R^2$, so this gallery is the check that decides whether a "free" rank
reduction is really free.
"""

S5_CODE = r'''
if histories:
    pc_traces = sweep.gallery_traces(inventory, region=GALLERY_REGION, space="pc", indices=(0, 1, 2))
    show(viz.plot_fit_gallery(pc_traces, order=labels,
                              title=f"{GALLERY_REGION.upper()} — top three PCs, observed against mRNN"),
         "fig03_gallery_pc")
'''

S5B_CODE = r'''
if histories:
    fr_traces = sweep.gallery_traces(inventory, region=GALLERY_REGION, space="fr", indices=(0, 80, 160))
    show(viz.plot_fit_gallery(fr_traces, order=labels,
                              title=f"{GALLERY_REGION.upper()} — three example units, backprojected firing rate"),
         "fig04_gallery_fr")
'''


S6_TEXT = r"""## 6. Does a narrower channel make the circuit identifiable?

This is the reason the bottleneck was introduced in the first place. Independently seeded
fits of the unconstrained model agree on their outputs and not on their inter-regional
currents; squeezing communication through $r$ dimensions removes ways of producing the
same output, so it should raise agreement if anything can.

The legacy ensembles hinted at this — rank-3 scored higher than rank-1 on every
consistency measure — but the two were trained for different numbers of iterations, so
the comparison was confounded. Here they are not.
"""

S6_CODE = r'''
if histories:
    agreement = pd.concat([
        tl.inter_seed_agreement(list(block["run_dir"])).assign(label=label)
        for label, block in inventory[inventory["complete"].astype(bool)].groupby("label", sort=False)
    ], ignore_index=True)
    display(agreement.pivot_table(index="label", columns="feature", values="mean_agreement")
            .loc[labels].round(4))
    show(viz.plot_seed_agreement_by_variant(agreement, order=labels), "fig05_seed_agreement")
'''


S7_TEXT = r"""## 7. What rank did the unconstrained model use?

The dense baseline was free to use any rank. Its singular spectrum says what it chose,
and is the natural companion to the constrained sweep: if the constrained fits are as good
at rank 3 while the dense blocks spread their energy over many more directions, then that
spread was slack rather than signal — an unpenalised optimiser has no reason to
concentrate a block.
"""

S7_CODE = r'''
if histories and BASELINE in labels:
    dense_runs = inventory[(inventory["label"] == BASELINE) & inventory["complete"]]
    rows = []
    for _, run in dense_runs.iterrows():
        model, _ = load_fixation_mrnn_checkpoint(Path(run["run_dir"]), device="cpu")
        slices = model.hidden_region_slices()
        weight = model.recurrent_weight_matrix().detach().cpu().numpy()
        for source in model.region_order:
            for target in model.region_order:
                if source == target:
                    continue
                block = weight[slices[target], slices[source]]
                singular = np.linalg.svd(block, compute_uv=False)
                energy = np.cumsum(singular**2) / max(np.sum(singular**2), 1e-12)
                rows.append({
                    "seed": int(run["seed"]), "pathway": f"{source}→{target}",
                    "numerical_rank": int(np.linalg.matrix_rank(block)),
                    "dims_for_50pct": int(np.searchsorted(energy, 0.50) + 1),
                    "dims_for_90pct": int(np.searchsorted(energy, 0.90) + 1),
                })
    spectra = pd.DataFrame(rows)
    display(spectra.groupby("pathway")[["numerical_rank", "dims_for_50pct", "dims_for_90pct"]]
            .median().round(1))
    display(Markdown(
        f"Across {spectra['seed'].nunique()} seeds and {spectra['pathway'].nunique()} pathways, the "
        f"unconstrained blocks are numerically rank "
        f"{spectra['numerical_rank'].median():.0f} and need a median of "
        f"**{spectra['dims_for_90pct'].median():.0f}** directions to carry 90% of their energy — "
        f"against **{spectra['dims_for_50pct'].median():.0f}** for half of it. Read that as an upper "
        f"bound rather than a requirement: it says the unconstrained fit did not spontaneously "
        f"discover a bottleneck, not that a bottleneck would hurt. Section 4 is what tests that."
    ))
'''


S8_TEXT = r"""## 8. Reading the result

Three statements the sweep can support, written to `bottleneck_findings.csv`:

1. **The lowest rank indistinguishable from dense.** Reported with the baseline's own
   seed-to-seed spread as the yardstick, and cross-checked against the gallery in Section 5,
   since a rank can look free in $R^2$ and visibly smooth the trajectories.
2. **Whether the curve is monotone once training length is held fixed.** The legacy sweep's
   non-monotonicity was an artefact; if it survives here it is a real finding and needs the
   convergence table in Section 3 to interpret.
3. **Whether constraining the channel buys identifiability.** If seed agreement on the
   latent drive geometry rises as rank falls, the bottleneck is doing what it was
   introduced to do — and the rank at which agreement peaks, not the rank at which fit
   peaks, is the model tasks 05–07 should use.
"""

S8_CODE = r'''
if not histories:
    display(Markdown("Deferred until the sweep completes."))
else:
    spread = float(fit[fit["label"] == BASELINE].groupby("seed")["r2_vs_ceiling"].mean().std())
    geometry = agreement[agreement["feature"] == "latent drive geometry"].set_index("label")["mean_agreement"]
    findings = summary.join(geometry.rename("seed_agreement_geometry"))
    findings["indistinguishable_from_dense"] = findings["cost_vs_dense"] <= 2 * spread
    findings.to_csv(TASK_ROOT / "bottleneck_findings.csv")
    display(findings.round(4))

    constrained = findings.drop(index=BASELINE, errors="ignore")
    free = constrained[constrained["indistinguishable_from_dense"]]
    lowest = free.index[0] if len(free) else None
    peak_agreement = geometry.drop(index=BASELINE, errors="ignore").idxmax() if len(geometry) > 1 else None
    display(Markdown(
        f"Baseline seed-to-seed spread **{spread:.4f}**; a rank counts as indistinguishable when it "
        f"costs less than twice that.\n\n"
        f"**Lowest rank indistinguishable from dense:** "
        f"{lowest if lowest is not None else 'none — every constraint costs'}.\n\n"
        f"**Rank with the highest seed agreement on drive geometry:** "
        f"{peak_agreement if peak_agreement is not None else 'n/a'} "
        f"(dense: {float(geometry.get(BASELINE, float('nan'))):.3f}).\n\n"
        f"Written to `{TASK_ROOT / 'bottleneck_findings.csv'}`."
    ))
'''


S9_TEXT = r"""
## 9. Sparsity and low rank as a matched-budget comparison

Sections 4-8 constrain the inter-regional channel one way: force each block through a
product of a left and a right factor, so everything passing from one region to another
passes through $r$ shared dimensions. That restricts the **rank** of the pathway. It is not
the only way to make a pathway cheap, and the alternative asks a different question.

**Sparsity and low rank are independent.** A block that keeps a quarter of its entries is
still full rank; a rank-5 block is still fully dense. Measured directly on this
architecture:

| constraint | density | rank | live entries |
|---|---|---|---|
| dense (baseline) | 100.0% | 20 | 1600 |
| 25% of entries kept | 25.5% | 20 | 1008 |
| rank 5 (same cost) | 100.0% | 5 | 1600 |
| 10% of entries kept | 10.2% | 16 | 879 |
| rank 2 (same cost) | 100.0% | 2 | 1600 |

So "make the pathway smaller" is really two claims that one sweep cannot separate:
*communication is confined to few dimensions* (rank) versus *communication runs over few
connections* (sparsity). They can be made to cost the same. On a $40\times40$ block, rank
$r$ costs $2\times40\times r = 80r$ free parameters and density $d$ costs $1600d$, so the
two are matched when **$d = r/20$**. Any difference at matched budget is attributable to
the *structure* of the constraint rather than to how much of it there is.

**Three prongs, each measured against the same dense baseline.**

| prong | what is constrained | grid |
|---|---|---|
| A · within-region sparsity | each region's own recurrent block | density 0.05, 0.10, 0.25, 0.50 |
| B · cross-region sparsity | the 12 inter-regional blocks | density 0.05, 0.10, 0.25, 0.50 |
| C · cross-region low rank | the 12 inter-regional blocks | rank 1, 2, 5, 10 — **already fitted in Section 4** |

Prong C costs nothing extra: `RANK_GRID` already contains the matched ranks, so those runs
are read out of the sweep above rather than repeated. Only A and B are submitted here, and
neither depends on the rank sweep's answer, so both can be queued while Section 4 is still
running.

Prong B is what makes prong C interpretable. Cross-region sparsity and cross-region low
rank spend an identical budget on an identical set of blocks, so if the low-rank models
turn out to be the more reproducible across seeds, that is because rank is the right
description of inter-regional communication — not because the constraint happened to be
the tighter of the two.

**Why the L1 arm was dropped.** An earlier version of this section swept an L1 penalty
scale. That measured the wrong thing twice over: the penalty is an *input* and sparsity is
the *outcome*, and on this model the map between them is not gradual — at 0.01 the
within-region blocks land five orders of magnitude below the cross-region ones, which is
not a sparse network but an ablated one. A density mask states the constraint directly, so
the surviving fraction is set rather than hoped for, and leaves the surviving weights free
to take whatever magnitude they need. `l1_weight_scale` stays pinned at 0 throughout.

**Sparsity is still measured, not assumed.** Two outcomes that look identical to any
scale-free measure — a few large weights surviving, or every weight shrinking together —
are separated by the **within-to-cross norm ratio**: 1.03 when the blocks are intact,
0.00001 when they have been erased. Gini moves only from 0.37 to 0.43 across that whole
difference, which is why it is reported but not relied on.
"""

S9_CODE = r'''
#: Matched budgets. On a 40x40 block, rank r costs 80r free parameters and density d costs
#: 1600d, so d = r/20 spends the same. Prong C is read from RANK_GRID, not refitted.
MATCHED_RANKS = tuple(rank for rank in (1, 2, 5, 10) if rank in RANK_GRID)
DENSITY_GRID = tuple(rank / 20.0 for rank in MATCHED_RANKS)   # 0.05, 0.10, 0.25, 0.50
BLOCK_ENTRIES = HIDDEN_UNITS * HIDDEN_UNITS


def matched_budget(rank: int) -> int:
    """Free parameters a rank-r factorization of one block costs."""
    return 2 * HIDDEN_UNITS * int(rank)


sparsity_variants = [
    sweep.ModelVariant(
        label=f"within_d{density:g}".replace(".", "p"),
        arm="A · within-region sparsity",
        overrides={**base_overrides, "recurrent_bottleneck_dim": None,
                   "within_region_density": float(density)},
    )
    for density in DENSITY_GRID
] + [
    sweep.ModelVariant(
        label=f"cross_d{density:g}".replace(".", "p"),
        arm="B · cross-region sparsity",
        overrides={**base_overrides, "recurrent_bottleneck_dim": None,
                   "cross_region_density": float(density)},
    )
    for density in DENSITY_GRID
]

display(Markdown("**Matched budget.** Each row costs the same however it is spent."))
display(pd.DataFrame({
    "rank (prong C)": MATCHED_RANKS,
    "density (prongs A, B)": DENSITY_GRID,
    "free parameters per block": [matched_budget(rank) for rank in MATCHED_RANKS],
    "of dense": [f"{matched_budget(rank) / BLOCK_ENTRIES:.0%}" for rank in MATCHED_RANKS],
}))
display(pd.DataFrame([v.describe() for v in sparsity_variants]))
display(Markdown(
    f"**{len(sparsity_variants)} cells × {len(seeds)} seeds = {len(sparsity_variants) * len(seeds)} "
    f"runs** for prongs A and B. Prong C adds nothing to submit: ranks {MATCHED_RANKS} are already "
    f"in `RANK_GRID`, and all three prongs share the dense baseline."
))
'''


S9B_CODE = r'''
SUBMIT_SPARSITY = False   # <-- set to True to submit prongs A and B

sparsity_commands, _ = sweep.variant_job_commands(
    sparsity_variants, seeds, root=TASK_ROOT, repo_root=repo_root,
    protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH,
    exclude_run_dirs=flight["run_dirs"],
)
sparsity_inventory = sweep.index_variant_runs(
    TASK_ROOT, sparsity_variants, seeds, in_flight=flight["run_dirs"]
)
display(Markdown(
    f"**{int(sparsity_inventory['complete'].sum())} complete**, "
    f"**{int(sparsity_inventory['queued'].sum())} on the queue**, "
    f"**{int(sparsity_inventory['pending'].sum())} unqueued** of {len(sparsity_inventory)} cells."
))

if SUBMIT_SPARSITY and sparsity_commands and BASE_MODEL is None and not ALLOW_PROVISIONAL_WIDTH:
    display(Markdown(
        "**Not submitted**: task 02 has not fixed the base model. Set "
        "`ALLOW_PROVISIONAL_WIDTH = True` to queue at the provisional width anyway."
    ))
elif SUBMIT_SPARSITY and sparsity_commands and PROTOCOL_IS_PROVISIONAL and not ALLOW_PROVISIONAL_PROTOCOL:
    display(Markdown(
        "**Not submitted**: task 00 has not frozen a recipe. Set "
        "`ALLOW_PROVISIONAL_PROTOCOL = True` to queue on the provisional optimiser."
    ))
elif SUBMIT_SPARSITY and sparsity_commands:
    from datetime import datetime

    from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

    jobs_dir = TASK_ROOT / "_jobs_sparsity"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_file = jobs_dir / f"sparsity_{stamp}.txt"
    write_job_file(job_file, sparsity_commands)
    submitted = [Path(line.split("--run-dir ")[1].split()[0]) for line in sparsity_commands]
    sparsity_id = submit_dsq_array_job(
        job_file_path=job_file, sbatch_script_path=jobs_dir / f"sparsity_{stamp}.sh",
        log_dir=jobs_dir / "logs", job_name="mrnn_sparsity", partition="psych_gpu",
        cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
    )
    sweep.record_submission(
        jobs_dir, job_id=sparsity_id, run_dirs=submitted, label=f"sparsity {stamp}"
    )
    (jobs_dir / "job_id.txt").write_text(str(sparsity_id) + "\n")
    display(Markdown(
        f"Submitted **{len(sparsity_commands)}** runs as job array **{sparsity_id}**. They queue "
        f"behind whatever is already in flight; the per-user GPU cap decides when they start."
    ))
elif sparsity_commands:
    display(Markdown("`SUBMIT_SPARSITY` is **False** — nothing was submitted."))
elif int(sparsity_inventory["queued"].sum()):
    display(Markdown("Nothing to submit: the outstanding cells are already on the queue."))
else:
    display(Markdown("Prongs A and B are fully trained."))
'''


S9C_CODE = r'''
#: Which agreement features count towards "is the solution unique". "output trajectories"
#: is excluded on purpose: it is what training optimises, so two seeds agreeing on it says
#: the optimiser worked, not that the circuits match. It is displayed, never scored.
TAUTOLOGICAL_FEATURES = ("output trajectories",)


def _prong_and_budget(label: str) -> tuple[str, float]:
    """Constraint family and free parameters per constrained block, from the label."""
    text = str(label)
    if text.startswith("within_d"):
        return "A · within-region sparsity", float(text[8:].replace("p", ".")) * BLOCK_ENTRIES
    if text.startswith("cross_d"):
        return "B · cross-region sparsity", float(text[7:].replace("p", ".")) * BLOCK_ENTRIES
    if text.startswith("rank"):
        return "C · cross-region low rank", float(matched_budget(int(text[4:])))
    return "dense", float(BLOCK_ENTRIES)


def _scored_agreement(inventory_frame: pd.DataFrame) -> pd.DataFrame:
    complete = inventory_frame[inventory_frame["complete"].astype(bool)]
    if complete.empty:
        return pd.DataFrame(columns=["label", "feature", "mean_agreement"])
    return pd.concat([
        tl.inter_seed_agreement(list(block["run_dir"])).assign(label=label)
        for label, block in complete.groupby("label", sort=False)
    ], ignore_index=True)


sparsity_histories = sweep.load_histories(sparsity_inventory)
if not sparsity_histories:
    display(Markdown(
        "No prong A or B runs have finished yet — this section fills in as the arm lands. "
        "Prong C is already reported in Section 4."
    ))
else:
    sparsity_convergence = sweep.convergence_table(sparsity_histories)
    display(sparsity_convergence.round(6))
    # Always look at the trajectories before the numbers: a constraint that merely makes
    # the problem harder to optimise and one the data cannot tolerate produce the same
    # final loss, and only the shape of the descent tells them apart.
    show(viz.plot_sweep_loss_trajectories(sparsity_histories, convergence=sparsity_convergence),
         "fig06_sparsity_losses")

    sparsity_fit = sweep.score_variant_fit(sparsity_inventory, ceiling_by_region)
    achieved = sweep.summarize_sparsity(sparsity_inventory)
    display(Markdown(
        "**Achieved sparsity.** The mask sets the surviving fraction; this checks what the "
        "surviving weights did with it. A ratio near 1 means the blocks are intact and merely "
        "thinner; a ratio near 0 means they were erased, whatever the fit says."
    ))
    display(achieved.groupby("label")[
        ["within_to_cross_norm", "within_gini", "within_fraction_near_zero"]].mean().round(5))

    # Prong C is read out of Section 4 rather than refitted, so all three families are
    # scored from runs trained under identical settings apart from the constraint itself.
    combined_fit = pd.concat([fit, sparsity_fit], ignore_index=True)
    agreement_all = pd.concat(
        [_scored_agreement(inventory), _scored_agreement(sparsity_inventory)], ignore_index=True
    )
    scored = agreement_all[~agreement_all["feature"].isin(TAUTOLOGICAL_FEATURES)]
    agreement_by_label = scored.groupby("label")["mean_agreement"].mean()

    adequacy = sweep.adequacy_table(combined_fit, baseline_label=BASELINE)
    adequacy["prong"] = [_prong_and_budget(label)[0] for label in adequacy["label"]]
    adequacy["budget"] = [_prong_and_budget(label)[1] for label in adequacy["label"]]
    adequacy["agreement"] = [float(agreement_by_label.get(label, np.nan))
                             for label in adequacy["label"]]

    matched_only = adequacy[
        adequacy["prong"].isin(["A · within-region sparsity", "B · cross-region sparsity",
                                "C · cross-region low rank"])
        & adequacy["budget"].round(2).isin([round(matched_budget(r), 2) for r in MATCHED_RANKS])
    ].copy()
    matched_only = matched_only.rename(columns={"worst_condition": "fit_vs_ceiling"})
    if BASELINE in set(adequacy["label"]):
        matched_only["dense_fit"] = float(
            adequacy.loc[adequacy["label"] == BASELINE, "worst_condition"].iloc[0])
        matched_only["dense_agreement"] = float(agreement_by_label.get(BASELINE, np.nan))

    display(Markdown(
        "**All three prongs at matched budget.** `fit_vs_ceiling` is the *worst* condition, not "
        "the mean — a constraint that keeps the average by giving up interactive-face structure "
        "has not been tolerated by the data.\n\n"
        "**The bar is 1.0, the noise ceiling — not the dense baseline.** The scale is normalized "
        "so that 1.0 means reproducing exactly the reproducible part of the signal and above 1.0 "
        "means reproducing sampling noise. The unconstrained model lands *above* the ceiling, so "
        "asking a constrained model to match it is asking it to overfit by the same margin. "
        "`cost_vs_baseline` is reported because the gap is worth seeing; it is not the criterion."
    ))
    display(matched_only[["prong", "label", "budget", "fit_vs_ceiling", "cost_vs_baseline",
                          "adequate", "agreement"]].sort_values(["prong", "budget"]).round(4))
    show(viz.plot_matched_budget_prongs(matched_only), "fig07_matched_budget_prongs")
'''


S9D_CODE = r'''
if sparsity_histories:
    # The selection rule, stated once. Fit is a *constraint*: a model has to reach the noise
    # ceiling on its worst condition, within the seed-to-seed spread of the fits themselves.
    # It is deliberately not "match the dense baseline": past the ceiling further loss
    # reduction is fitting sampling error, the dense model does exactly that, and scoring
    # against it rejects every constraint that declines to overfit. Among the models that
    # clear the bar, the one to keep is the most reproducible across seeds at the smallest
    # budget -- reproducibility is the objective, because a circuit claim that changes with
    # the random seed is not a claim about the brain.
    knees = []
    for prong, block in matched_only.groupby("prong"):
        ok = block[block["adequate"]]
        if ok.empty:
            knees.append({"prong": prong, "tightest_adequate": None, "budget": np.nan,
                          "fit_vs_ceiling": np.nan, "agreement": np.nan})
            continue
        pick = ok.loc[ok["budget"].idxmin()]
        knees.append({"prong": prong, "tightest_adequate": pick["label"],
                      "budget": float(pick["budget"]),
                      "fit_vs_ceiling": float(pick["fit_vs_ceiling"]),
                      "agreement": float(pick["agreement"])})
    knee_table = pd.DataFrame(knees)
    display(Markdown(
        "**Tightest setting each family can sustain**, and what it buys. \"Sustain\" means its "
        "worst condition still reaches the noise ceiling; the smallest budget that does is the "
        "knee."
    ))
    display(knee_table.round(4))

    # The comparison the arm exists for. B and C spend the same budget on the same blocks
    # and differ only in the structure of the constraint, so a gap between them is evidence
    # about what inter-regional communication *is*, not about how much of it was removed.
    b = knee_table[knee_table["prong"].str.startswith("B")]
    c = knee_table[knee_table["prong"].str.startswith("C")]
    if len(b) and len(c) and np.isfinite(b["budget"].iloc[0]) and np.isfinite(c["budget"].iloc[0]):
        paired = matched_only[matched_only["prong"].str.startswith(("B", "C"))]
        contrast = paired.pivot_table(index="budget", columns="prong",
                                      values=["fit_vs_ceiling", "agreement"])
        display(Markdown(
            "**Sparsity against low rank, budget for budget.** Same parameters, same blocks, "
            "different structure:"
        ))
        display(contrast.round(4))

    findings_path = TASK_ROOT / "sparsity_findings.csv"
    matched_only.to_csv(findings_path, index=False)
    knee_table.to_csv(TASK_ROOT / "prong_knees.csv", index=False)
    display(Markdown(
        f"Written to `{findings_path}` and `{TASK_ROOT / 'prong_knees.csv'}`. Section 10 uses "
        f"the knees to define the one combination that gets verified jointly."
    ))
'''


S10_TEXT = r"""
## 10. The combination, verified jointly

Sections 4 and 9 explore the three families **coordinate-wise**: each is varied on its own
against the same dense baseline, which is what makes the comparison between them clean.
Coordinate-wise exploration does not license a coordinate-wise *conclusion*. The
constraints plausibly interact — with a wide inter-regional channel a region can route
around its own thinned internal block, and with a narrow one it cannot — so the setting
each family tolerates alone is not necessarily the setting it tolerates alongside the
others.

So the corner defined by the three knees is fitted as its own model and reported next to
what the coordinate-wise result predicted for it. Two outcomes, both worth having:

- **It holds.** The constraints are separable at these budgets, the coordinate-wise sweep
  was the right experiment, and the combined model is the one the rest of the chapter uses.
- **It does not.** The gap is the interaction, measured rather than assumed, and the
  selection falls back to the tightest corner that does clear the bar.

Only this one corner is fitted, not the full grid. The grid would be the right experiment
if the interaction were the question; here it is a risk to be checked, and one corner
checks it.
"""


S10_CODE = r'''
SUBMIT_COMBINED = False   # <-- set to True to submit the joint corner

combined_variants: list = []
if not sparsity_histories:
    display(Markdown(
        "Section 9 has to report before the corner is defined. Nothing to submit yet."
    ))
elif knee_table["tightest_adequate"].isna().all():
    display(Markdown(
        "🔴 **No family sustained any constraint at matched budget.** There is no corner to "
        "verify: the dense model is the selection, and that itself is the result — the data "
        "does not tolerate a cheaper inter-regional channel at this width."
    ))
else:
    corner: dict[str, object] = {**base_overrides, "recurrent_bottleneck_dim": None}
    chosen: list[str] = []
    for _, row in knee_table.iterrows():
        if row["tightest_adequate"] is None or not isinstance(row["tightest_adequate"], str):
            continue
        label = str(row["tightest_adequate"])
        prong = str(row["prong"])
        if prong.startswith("A"):
            corner["within_region_density"] = float(label[8:].replace("p", "."))
        elif prong.startswith("B"):
            corner["cross_region_density"] = float(label[7:].replace("p", "."))
        elif prong.startswith("C"):
            corner["recurrent_bottleneck_dim"] = int(label[4:])
        chosen.append(f"{prong} → `{label}`")

    # B and C constrain the same blocks two different ways. Imposing both would confound the
    # comparison the arm was built to make, so the corner keeps whichever of the two survived
    # at the smaller budget and drops the other.
    cross_prongs = knee_table[knee_table["prong"].str.startswith(("B", "C"))].dropna(subset=["budget"])
    if len(cross_prongs) == 2:
        loser = cross_prongs.loc[cross_prongs["budget"].idxmax(), "prong"]
        if str(loser).startswith("B"):
            corner.pop("cross_region_density", None)
        else:
            corner["recurrent_bottleneck_dim"] = None
        display(Markdown(
            f"Both cross-region families cleared the bar; the corner keeps the one that did it at "
            f"the smaller budget and drops **{loser}**, since imposing both would confound them."
        ))

    combined_variants = [sweep.ModelVariant(label="combined", arm="joint corner", overrides=corner)]
    display(Markdown("**The corner:** " + "; ".join(chosen)))
    display(pd.DataFrame([combined_variants[0].describe()]))

if combined_variants:
    combined_commands, _ = sweep.variant_job_commands(
        combined_variants, seeds, root=TASK_ROOT, repo_root=repo_root,
        protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH,
        exclude_run_dirs=flight["run_dirs"],
    )
    combined_inventory = sweep.index_variant_runs(
        TASK_ROOT, combined_variants, seeds, in_flight=flight["run_dirs"]
    )
    display(Markdown(
        f"**{int(combined_inventory['complete'].sum())} complete**, "
        f"**{int(combined_inventory['queued'].sum())} on the queue**, "
        f"**{int(combined_inventory['pending'].sum())} unqueued** of {len(combined_inventory)} cells."
    ))
    if SUBMIT_COMBINED and combined_commands:
        from datetime import datetime

        from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

        jobs_dir = TASK_ROOT / "_jobs_combined"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_file = jobs_dir / f"combined_{stamp}.txt"
        write_job_file(job_file, combined_commands)
        submitted = [Path(line.split("--run-dir ")[1].split()[0]) for line in combined_commands]
        combined_id = submit_dsq_array_job(
            job_file_path=job_file, sbatch_script_path=jobs_dir / f"combined_{stamp}.sh",
            log_dir=jobs_dir / "logs", job_name="mrnn_combined", partition="psych_gpu",
            cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
        )
        sweep.record_submission(
            jobs_dir, job_id=combined_id, run_dirs=submitted, label=f"combined {stamp}"
        )
        (jobs_dir / "job_id.txt").write_text(str(combined_id) + "\n")
        display(Markdown(f"Submitted **{len(combined_commands)}** runs as array **{combined_id}**."))
    elif combined_commands:
        display(Markdown("`SUBMIT_COMBINED` is **False** — nothing was submitted."))

    combined_histories = sweep.load_histories(combined_inventory)
    if combined_histories:
        display(sweep.convergence_table(combined_histories).round(6))
        show(viz.plot_sweep_loss_trajectories(combined_histories), "fig08_combined_losses")
        combined_score = sweep.score_variant_fit(combined_inventory, ceiling_by_region)
        joint = sweep.adequacy_table(
            pd.concat([fit, combined_score], ignore_index=True), baseline_label=BASELINE
        )
        row = joint[joint["label"] == "combined"]
        predicted = float(knee_table["fit_vs_ceiling"].min())
        display(joint[joint["label"].isin([BASELINE, "combined"])].round(4))
        display(Markdown(
            f"**Coordinate-wise prediction {predicted:.4f} against joint result "
            f"{float(row['worst_condition'].iloc[0]):.4f}.** A gap here is the interaction between "
            f"the constraints; it is reported whichever way it falls, and it decides whether the "
            f"corner or a looser setting goes forward to task 05."
        ))

        selection = {
            "hidden_units": HIDDEN_UNITS,
            "condition_loss_weighting": CONDITION_WEIGHTING,
            "l1_weight_scale": 0.0,
            **{k: v for k, v in combined_variants[0].overrides.items()
               if k in ("within_region_density", "cross_region_density", "recurrent_bottleneck_dim")},
            "inherited_protocol": SELECTED_PROTOCOL.get("selected_label"),
            "epochs": int(SELECTED_PROTOCOL["epochs"]),
            "worst_condition_vs_ceiling": float(row["worst_condition"].iloc[0]),
            "adequate": bool(row["adequate"].iloc[0]),
            "selection_rule": ("fit is a constraint (worst condition reaches the noise ceiling "
                               "within twice the seed spread); among adequate models, most "
                               "reproducible at least cost"),
        }
        path = TASK_ROOT / "selected_constrained_model.yaml"
        path.write_text(yaml.safe_dump(selection, sort_keys=False))
        display(Markdown(f"Selection written to `{path}` — task 05 reads it from there."))
'''


S11 = r"""
## 11. What comes next

Tasks 01–04 together fix the model the rest of the chapter uses: how wide, what is
connected, how much passes between regions, and over how many connections — each chosen
because the data still supports the fit under that constraint, not because it minimised a
loss.

**Next:** `05_seed_ensembles.ipynb` takes the selected model and asks the question this
whole rebuild exists to answer — whether independently fitted networks agree about the
circuit once the fitting itself is sound. It runs a small ensemble first, and only scales
up if the small one agrees: a hundred seeds of an unidentifiable model is a hundred copies
of the same non-result. Agreement is scored against a null rather than a threshold, and on
quantities that are invariant to the signed permutation of hidden units — the exact
symmetry of a tanh network with a linear readout — so that relabelling units cannot be
mistaken for a difference in circuits.
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
        _cell("markdown", S3B_TEXT),
        _cell("code", S3B_CODE),
        _cell("markdown", S4_TEXT),
        _cell("code", S4_CODE),
        _cell("markdown", S5_TEXT),
        _cell("code", S5_CODE),
        _cell("code", S5B_CODE),
        _cell("markdown", S6_TEXT),
        _cell("code", S6_CODE),
        _cell("markdown", S7_TEXT),
        _cell("code", S7_CODE),
        _cell("markdown", S8_TEXT),
        _cell("code", S8_CODE),
        _cell("markdown", S9_TEXT),
        _cell("code", S9_CODE),
        _cell("code", S9B_CODE),
        _cell("code", S9C_CODE),
        _cell("code", S9D_CODE),
        _cell("markdown", S10_TEXT),
        _cell("code", S10_CODE),
        _cell("markdown", S11),
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
