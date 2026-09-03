"""Author the capacity notebook (task 01 of the rebuilt mRNN analysis).

    conda run -n gaze_processing python notebooks/mrnn_thesis/_build_01_capacity.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

OUTPUT_FILENAME = "01_capacity.ipynb"
_CELL_COUNTER = count(1)


HEADER = r"""# 01 · Capacity — how many units does each region need?

*Task 01 of the rebuilt mRNN analysis. Task 00 settled how to train; this is the first of
the structural questions.*

**The framing.** With roughly twenty thousand free parameters against fifty thousand
target numbers, plenty of configurations fit these trajectories. Ranking them by loss is
therefore not very interesting. The useful question is the opposite one: **which
constraints can the data tolerate?** A model that still reproduces the trajectories while
being restricted tells us something; a model that fits because it was given enough
freedom to fit anything does not.

This notebook applies the first constraint — network width — and asks how far it can be
pushed before the fit gives way.

**Why the range stops at 60.** Each region reads out to 42 principal components, so a
region with many more than 42 units is not being asked for anything its width could
supply. The sweep covers **20, 30, 40, 50, 60 units per region**, which brackets the
target dimensionality from well below to somewhat above.

**Why there is no bottleneck here.** The task-00 recipe carried
`recurrent_bottleneck_dim = 3`, a rank constraint on every inter-regional block that was
inherited rather than tested. That is itself one of the constraints this chapter is about,
so it does not belong in the baseline. Every model here uses **dense, full-rank
inter-regional connectivity**, and the rank constraint is imposed and measured in task 03.

> Expressing that required a code change: inter-region blocks were *always* factorized as
> `left @ right`, so a model with no rank constraint could not be built. Setting the rank
> equal to the hidden width is not a substitute — the product parameterization carries
> twice the parameters and optimizes differently.

| Section | |
|---|---|
| 1 | The sweep, and what it inherits |
| 2 | Run state and submission (off by default) |
| 3 | Loss trajectories and convergence |
| 4 | Fit against the noise ceiling, and the parameters that bought it |
| 5 | The visual check, across the whole sweep |
| 6 | Do the seeds agree? |
| 7 | Selection |
"""


SETUP = r'''
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
TASK_ROOT = sweep.resolve_task_root("01_capacity", DATASET_CFG_PATH)
CEILING_DIR = ceiling_mod.resolve_output_dir(DATASET_CFG_PATH)
FIGURE_DIR = syn.resolve_output_dir(DATASET_CFG_PATH, scope="01_capacity")
FIGURES = ThesisFigureSettings(output_dir=FIGURE_DIR)

SELECTED_PROTOCOL = sweep.load_selected_protocol(PROTOCOL_ROOT / "selected_protocol.yaml")

#: Widths to test. Each region reads out 42 PCs, so far above that adds freedom the
#: target cannot use.
HIDDEN_UNIT_GRID = (20, 30, 40, 50, 60)
#: Five seeds: seed agreement is one of the selection axes and three seeds give three pairs.
SWEEP_SEEDS = 5
#: Region whose traces the gallery shows. Fixed across the sweep so rows compare.
GALLERY_REGION = "ofc"


def show(figure, stem: str) -> None:
    save_thesis_figure(figure, FIGURES, stem)
    display(Image(data=figure_to_png_bytes(figure, dpi=190)))


print("inherited recipe :", SELECTED_PROTOCOL["selected_label"])
print("task root        :", TASK_ROOT)
'''


S1_TEXT = r"""## 1. The sweep

Everything except width and the rank constraint is inherited from the recipe task 00
froze, so exactly one thing differs between cells.
"""

S1_CODE = r'''
variants = [
    sweep.ModelVariant(
        label=f"h{units:02d}",
        overrides={"hidden_units": int(units), "recurrent_bottleneck_dim": None},
        arm="capacity",
    )
    for units in HIDDEN_UNIT_GRID
]
seeds = protocol.protocol_seeds(n_seeds=SWEEP_SEEDS)

display(pd.DataFrame([v.describe() for v in variants]))
display(pd.Series({
    **{k: v for k, v in SELECTED_PROTOCOL["optimizer"].items()},
    "iterations": SELECTED_PROTOCOL["epochs"],
    "connectivity": "full (all regions to all regions)",
    "inter-regional rank": "unconstrained (dense)",
    "PCs per region": SELECTED_PROTOCOL["architecture"]["pca_n_components"],
}, name="inherited").to_frame())
display(Markdown(
    f"**{len(variants)} widths × {len(seeds)} seeds = {len(variants) * len(seeds)} runs** at "
    f"{SELECTED_PROTOCOL['epochs']:,} iterations, submitted as one SLURM array job."
))
'''


S2_TEXT = r"""## 2. Run state and submission

Submission happens **only** if you set `SUBMIT = True`, and is blocked while a previously
submitted array is still on the queue.
"""

S2_CODE = r'''
SUBMIT = False   # <-- set to True to actually submit the missing cells

commands, run_dirs = sweep.variant_job_commands(
    variants, seeds,
    root=TASK_ROOT, repo_root=repo_root,
    protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH,
)
inventory = sweep.index_variant_runs(TASK_ROOT, variants, seeds)
job_state = protocol.running_job_state(TASK_ROOT / "_jobs")

display(Markdown(
    f"**{int(inventory['complete'].sum())} complete**, **{int(inventory['diverged'].sum())} diverged**, "
    f"**{int(inventory['pending'].sum())} not yet run** of {len(inventory)} cells."
))
if job_state["active"]:
    display(Markdown(
        f"⚠️ **Job array `{job_state['job_id']}` is still on the queue** "
        f"({', '.join(f'{n} {s.lower()}' for s, n in sorted(job_state['states'].items()))}). "
        f"Submission is blocked."
    ))
elif commands:
    display(Markdown(f"{len(commands)} run(s) would be submitted."))
    print("first command:\n")
    print(commands[0])
'''

S2B_CODE = r'''
if job_state["active"]:
    display(Markdown(f"Nothing submitted: job array `{job_state['job_id']}` is still running."))
elif SUBMIT and commands:
    from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

    jobs_dir = TASK_ROOT / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / "capacity.txt"
    write_job_file(job_file, commands)
    job_id = submit_dsq_array_job(
        job_file_path=job_file,
        sbatch_script_path=jobs_dir / "capacity.sh",
        log_dir=jobs_dir / "logs",
        job_name="mrnn_capacity",
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
    display(Markdown("Every cell is already trained; go on to Section 3."))
'''


S3_TEXT = r"""## 3. Loss trajectories and convergence

Read this before any fit comparison. A width that merely makes the model harder to
optimise looks the same in the final loss as a width the data genuinely cannot use, and
only the trajectory separates them. Panel titles are green where every seed cleared the
task-00 convergence bar and red where they did not.
"""

S3_CODE = r'''
histories = sweep.load_histories(inventory)
labels = [v.label for v in variants if v.label in histories]

if not histories:
    display(Markdown("No completed runs yet — this section fills in as the sweep lands."))
else:
    convergence = sweep.convergence_table(histories)
    display(convergence.round(5))
    show(viz.plot_sweep_loss_trajectories(histories, convergence=convergence), "fig01_loss_trajectories")
'''


S4_TEXT = r"""## 4. Fit, and what it cost in parameters

Fit is scored against the **measured noise ceiling** — the split-half reliability of each
region's PC trajectories — rather than against 1.0, because scoring against 1.0 asks the
model to reproduce sampling noise.

The parameter count sits beside it because the two have to be read together. A fit
obtained with a tenth of the data's degrees of freedom means considerably more than the
same fit obtained with all of them.
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
    display(parameters[["label", "within_region", "inter_region", "readout", "initial_state",
                        "total", "target_numbers", "parameters_per_datum"]].round(3))
    display(fit.groupby(["label", "condition"])["r2_vs_ceiling"].mean().unstack().round(4))
    show(viz.plot_fit_versus_constraint(fit, parameters, order=labels,
                                        x_label="hidden units per region"), "fig02_fit_vs_capacity")
'''


S5_TEXT = r"""## 5. The visual check, across the whole sweep

Looking at every trace of every model is not feasible, and a fit that scores well can
still be visibly wrong — that is how the interactive-face smoothing was found. The
compromise is a gallery: one row per width, all showing the **same three components of
the same region**, sized for scanning rather than for reading detail.

The model selected at the end gets the full-detail trace plots; everything else gets a
row here.
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


S6_TEXT = r"""## 6. Do the seeds agree?

The reason to care about width beyond fit. Seeds of the task-00 model agree at 0.997 on
what the model outputs and only 0.52 on its latent drive geometry — so the trajectories
are determined by the data while the mechanism producing them is not. A narrower network
has fewer ways to produce the same output, so if any constraint is going to make the
solution more identifiable, this is where it should start to show.
"""

S6_CODE = r'''
if histories:
    agreement = pd.concat([
        tl.inter_seed_agreement(list(block["run_dir"])).assign(label=label)
        for label, block in inventory[inventory["complete"].astype(bool)].groupby("label", sort=False)
    ], ignore_index=True)
    display(agreement.pivot_table(index="label", columns="feature", values="mean_agreement").round(4))
    show(viz.plot_seed_agreement_by_variant(agreement, order=labels), "fig05_seed_agreement")
'''


S7_TEXT = r"""## 7. Selection

The rule follows the framing: **the narrowest width whose fit is statistically
indistinguishable from the widest**, rather than the width with the best fit. Where two
widths fit equally, the smaller is the stronger result — it says the data can be
reproduced under a tighter constraint.

Seed agreement is reported alongside but does not override: if a narrower model is
markedly more identifiable at a small cost in fit, that is a judgement worth making
explicitly rather than by formula.
"""

S7_CODE = r'''
if not histories:
    display(Markdown("Selection is deferred until the sweep completes."))
else:
    summary = (
        fit.groupby("label")["r2_vs_ceiling"].agg(["mean", "min"])
        .join(parameters.set_index("label")[["total", "parameters_per_datum"]])
        .join(agreement[agreement["feature"] == "latent drive geometry"]
              .set_index("label")["mean_agreement"].rename("seed_agreement_geometry"))
        .join(convergence.set_index("label")[["n_converged", "n_seeds"]])
        .loc[labels]
    )
    display(summary.round(4))

    converged = summary[summary["n_converged"] == summary["n_seeds"]]
    if converged.empty:
        display(Markdown("**No width converged on every seed.** That is the result; report it."))
    else:
        best = float(converged["mean"].max())
        # "Indistinguishable" is set at one percent of the ceiling-relative score, which is
        # comfortably inside the seed-to-seed spread observed in task 00.
        adequate = converged[converged["mean"] >= best - 0.01]
        winner = adequate.index[0]
        display(Markdown(
            f"**Selected: `{winner}`** — the narrowest width within 0.01 of the best "
            f"ceiling-relative fit ({float(adequate.loc[winner, 'mean']):.3f} against {best:.3f}), "
            f"using {int(adequate.loc[winner, 'total']):,} parameters "
            f"({float(adequate.loc[winner, 'parameters_per_datum']):.2f} per target number) and "
            f"reaching {float(adequate.loc[winner, 'seed_agreement_geometry']):.3f} seed agreement on "
            f"latent drive geometry."
        ))
        import yaml as _yaml
        path = TASK_ROOT / "selected_capacity.yaml"
        path.write_text(_yaml.safe_dump({
            "selected_label": str(winner),
            "hidden_units": int(str(winner).lstrip("h")),
            "selection_rule": "narrowest width within 0.01 of the best ceiling-relative fit, "
                              "among widths that converged on every seed",
            "inherited_protocol": str(SELECTED_PROTOCOL["selected_label"]),
            "recurrent_bottleneck_dim": None,
            "scores": {k: float(v) for k, v in summary.loc[winner].items()},
        }, sort_keys=False))
        display(Markdown(f"Frozen to `{path}`."))
'''


S8 = r"""## 8. What this settles, and what comes next

**Settled.** How wide each region needs to be, measured against the noise ceiling and
with the parameter cost reported alongside — and, for the first time in this project,
without a rank constraint quietly in place.

**Two corrections carried in from setting this up**, both of which change how earlier
numbers should be read:

1. **`spectral_radius` was inert** for every run using the block parameterization, so the
   task-00 sweep's two spectral-radius arms were duplicates.
2. **The models are far smaller than reported.** `mrnn.W_rec` is a 200×200 copy the
   forward pass overwrites from the block parameters; it is registered as a parameter and
   receives no gradient. Counting it — as `model.parameters()` does — inflates the total
   about six-fold. The 50-unit model has **23,368** effective parameters against 50,400
   target numbers, a ratio of **0.46**, not the ~1 the legacy audit reported. The
   "as many parameters as data points" caveat should be retired.

**Next:** `02_connectivity.ipynb` — with width fixed, which connections are actually
necessary. Full all-to-all is the baseline; within-region only, cross-region only, and
single-pathway lesions are the constraints. Then `03_bottleneck_rank.ipynb` puts the rank
constraint back and asks how narrow inter-regional communication can be.
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
