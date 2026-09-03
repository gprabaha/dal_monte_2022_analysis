"""Author the bottleneck-rank notebook (task 03 of the rebuilt mRNN analysis).

    conda run -n gaze_processing python notebooks/mrnn_thesis/_build_03_bottleneck_rank.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

OUTPUT_FILENAME = "03_bottleneck_rank.ipynb"
_CELL_COUNTER = count(1)


HEADER = r"""# 03 · Inter-regional bottleneck — how narrow can the channel be?

*Task 03 of the rebuilt mRNN analysis. Task 01 fixed the width, task 02 established which
blocks have to exist; this asks how much has to pass through the ones that do.*

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
| 8 | Reading the result |
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
CONNECTIVITY_ROOT = sweep.resolve_task_root("02_connectivity", DATASET_CFG_PATH)
TASK_ROOT = sweep.resolve_task_root("03_bottleneck_rank", DATASET_CFG_PATH)
CEILING_DIR = ceiling_mod.resolve_output_dir(DATASET_CFG_PATH)
FIGURE_DIR = syn.resolve_output_dir(DATASET_CFG_PATH, scope="03_bottleneck_rank")
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

CAPACITY_PATH = CAPACITY_ROOT / "selected_capacity.yaml"
if CAPACITY_PATH.exists():
    SELECTED_CAPACITY = yaml.safe_load(CAPACITY_PATH.read_text())
    HIDDEN_UNITS = int(SELECTED_CAPACITY["hidden_units"])
    CAPACITY_SOURCE = f"task 01 (`{SELECTED_CAPACITY['selected_label']}`)"
else:
    SELECTED_CAPACITY = None
    HIDDEN_UNITS = 50
    CAPACITY_SOURCE = "**provisional fallback** — task 01 has not finished"

#: Set True to queue this sweep before task 01 has chosen a width. Cluster time is the
#: scarce resource and these arrays take hours, so waiting for a clean dependency can cost
#: more than the risk: if task 01 selects a different width, the runs fitted here are at
#: the wrong one and Section 1 will say so. The width every cell was actually trained at is
#: recorded in its own run_config.yaml either way, so a mismatch is always detectable.
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
                  "l1_weight_scale": 0.0}

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
    _widths = {int(yaml.safe_load(path.read_text())["hidden_units"]) for path in _trained}
    if _widths != {HIDDEN_UNITS}:
        display(Markdown(
            f"🔴 **Width mismatch.** {len(_trained)} trained run(s) used {sorted(_widths)} units, "
            f"but the current selection is {HIDDEN_UNITS}. Those runs answer the question at the "
            f"wrong width and should be refitted before the results below are used."
        ))
    else:
        display(Markdown(f"{len(_trained)} trained run(s), all at {HIDDEN_UNITS} units."))
display(pd.DataFrame([v.describe() for v in variants]))
display(Markdown(
    f"**{len(variants)} variants × {len(seeds)} seeds = {len(variants) * len(seeds)} runs**, all at "
    f"{SELECTED_PROTOCOL['epochs']:,} iterations."
))
'''


S2_TEXT = r"""## 2. Run state and submission"""

S2_CODE = r'''
SUBMIT = False   # <-- set to True to actually submit the missing cells

commands, run_dirs = sweep.variant_job_commands(
    variants, seeds, root=TASK_ROOT, repo_root=repo_root,
    protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH,
)
inventory = sweep.index_variant_runs(TASK_ROOT, variants, seeds)
job_state = protocol.running_job_state(TASK_ROOT / "_jobs")

if PROTOCOL_IS_PROVISIONAL:
    display(Markdown(
        "⚠️ **Task 00 has not frozen a recipe**, so the optimiser here is the provisional one "
        "(`lr 1e-3`, cosine, clip 0.05). The architecture is not provisional. "
        + ("`ALLOW_PROVISIONAL_PROTOCOL` is set, so submission is allowed."
           if ALLOW_PROVISIONAL_PROTOCOL else
           "Submission is blocked; set `ALLOW_PROVISIONAL_PROTOCOL = True` to queue anyway.")
    ))

if SELECTED_CAPACITY is None:
    display(Markdown(
        f"⚠️ **Task 01 has not finished**, so these cells would be fitted at the provisional "
        f"width of **{HIDDEN_UNITS}** units. "
        + ("`ALLOW_PROVISIONAL_WIDTH` is set, so submission is allowed."
           if ALLOW_PROVISIONAL_WIDTH else
           "Submission is blocked; set `ALLOW_PROVISIONAL_WIDTH = True` to queue anyway.")
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

S2B_CODE = r'''
if job_state["active"]:
    display(Markdown(f"Nothing submitted: job array `{job_state['job_id']}` is still running."))
elif SUBMIT and commands and SELECTED_CAPACITY is None and not ALLOW_PROVISIONAL_WIDTH:
    display(Markdown(
        f"**Not submitted**: task 01 has not selected a width, so these cells would be fitted at "
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
    from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

    jobs_dir = TASK_ROOT / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / "bottleneck.txt"
    write_job_file(job_file, commands)
    job_id = submit_dsq_array_job(
        job_file_path=job_file, sbatch_script_path=jobs_dir / "bottleneck.sh",
        log_dir=jobs_dir / "logs", job_name="mrnn_bottleneck", partition="psych_gpu",
        cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
    )
    (jobs_dir / "job_id.txt").write_text(str(job_id) + "\n")
    display(Markdown(f"Submitted **{len(commands)}** runs as job array **{job_id}**."))
elif commands:
    display(Markdown("`SUBMIT` is **False** — nothing was submitted."))
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


S9 = r"""## 9. What comes next

Tasks 01–03 together fix the model the rest of the chapter uses: how wide, what is
connected, and how much passes between regions — each chosen because the data still
supports the fit under that constraint, not because it minimised a loss.

**Next:** `05_seed_ensembles.ipynb` fits that model at a hundred initializations and asks
the question this whole rebuild exists to answer — whether independently fitted networks
agree about the circuit once the fitting itself is sound. Task 04 (target and loss) is
available if the fast-structure deficit turns out to matter for that answer.
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
        _cell("markdown", S5_TEXT),
        _cell("code", S5_CODE),
        _cell("code", S5B_CODE),
        _cell("markdown", S6_TEXT),
        _cell("code", S6_CODE),
        _cell("markdown", S7_TEXT),
        _cell("code", S7_CODE),
        _cell("markdown", S8_TEXT),
        _cell("code", S8_CODE),
        _cell("markdown", S9),
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
