"""Author the constrained-ensemble notebook (task 03 of the final mRNN series).

    conda run -n gaze_processing python notebooks/mrnn_chapter/_build_03_ensemble.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

OUTPUT_FILENAME = "03_ensemble.ipynb"
_CELL_COUNTER = count(1)


HEADER = r"""# 03 · The most constrained adequate model, fitted ten times

*Task 03 of the final series. Depends on task 02's `selected_bottleneck.yaml`; its ensemble
cannot be queued until that exists.*

Task 02 finds the tightest $(r_w, r_c)$ whose worst region × condition cell still clears the
bar. This task refits that one model with ten seeds and asks what the ten fits agree on —
against the ten-seed **dense** ensemble task 01 already produced, which is the baseline every
agreement number here is compared to.

The measurement machinery is the rebuild's, and the lessons carried over from it are the
reason this notebook is structured as it is:

- Agreement is measured with a **battery** tiered by what each statistic is blind to, three
  of them distances, every one against fits of the *untrained* architecture. A number above
  its floor is agreement; a number at its floor is the architecture.
- The current-agreement measure excludes within-region self-drive (rebuild E2).
- Per-condition lesion damage is scored in **absolute squared error** (rebuild E4). Under the
  minimax objective the conditions degrade together by construction, so what is asked of them
  here is whether the *seeds agree*, not which is damaged most.
- Lesions carry a **matched random control** — the same number of live weights removed at
  random — because under a rank constraint different blocks carry different weight.
- The statistic a positive result would appear in is the **cross-seed ranking of lesions**
  (Kendall's $\tau$ against a permutation null), not the damage itself: a ranking can be
  reproducible where the weights are not.

The specific consistencies to test are to be listed in Section 7; the sections before it
produce what any of them would need.

| Section | |
|---|---|
| 1 | The selected model |
| 2 | Run state and submission (off by default) |
| 3 | Does the ensemble fit? |
| 4 | Agreement against its floor — constrained versus dense |
| 5 | Lesions on the trained network |
| 6 | Do the seeds agree on which connections matter? |
| 7 | Consistencies *(to be specified)* |
| 8 | Reading the result |

**Nothing here submits a job unless `SUBMIT` is set to `True`.**
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
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    ThesisFigureSettings, apply_thesis_plot_style, figure_to_png_bytes, save_thesis_figure,
)

DATASET_CFG_PATH = repo_root / "configs" / "dataset.yaml"
MRNN_CFG_PATH = repo_root / "configs" / "ephys_fixation_mrnn.yaml"
apply_thesis_plot_style(load_config(repo_root / "configs" / "plotting.yaml"))

TREE = "final"
PROTOCOL_ROOT = protocol.resolve_chapter_root(DATASET_CFG_PATH, task="00_training_protocol")
LADDER_ROOT = sweep.resolve_task_root("01_ladder", DATASET_CFG_PATH, tree=TREE)
GRID_ROOT = sweep.resolve_task_root("02_rank_grid", DATASET_CFG_PATH, tree=TREE)
TASK_ROOT = sweep.resolve_task_root("03_ensemble", DATASET_CFG_PATH, tree=TREE)
CEILING_DIR = ceiling_mod.resolve_output_dir(DATASET_CFG_PATH)
FIGURE_DIR = syn.resolve_output_dir(DATASET_CFG_PATH, scope="final_03_ensemble")
FIGURES = ThesisFigureSettings(output_dir=FIGURE_DIR)

SELECTED_PROTOCOL, _ = sweep.load_selected_protocol_or_provisional(PROTOCOL_ROOT / "selected_protocol.yaml")
BASE = yaml.safe_load((TASK_ROOT.parent / "base_model.yaml").read_text())
REGION_ORDER = tuple(BASE.pop("region_order"))
BASE.pop("inherited_protocol", None); BASE.pop("r2_centring", None)

ENSEMBLE_SEEDS = 10
#: The matched per-cell ceiling task 01 computes (see its Section 1); the old per-region
#: ceiling only as a fallback before it exists.
_cell_path = TASK_ROOT.parent / "pc_space_ceiling_by_cell.csv"
if _cell_path.exists():
    _cell = pd.read_csv(_cell_path)
    ceiling_by_region = _cell[_cell["condition"] == "all"].set_index("region")["reliability"].to_dict()
    ceiling_by_cell = {(r, c): v for r, c, v in _cell[_cell["condition"] != "all"]
                       [["region", "condition", "reliability"]].itertuples(index=False)}
else:
    ceiling_by_region = (pd.read_csv(CEILING_DIR / "pc_space_ceiling.csv")
                         .groupby("region")["reliability"].mean().to_dict())
    ceiling_by_cell = None
ADEQUATE_BAR = 0.98

SELECTED = None
if (GRID_ROOT / "selected_bottleneck.yaml").exists():
    SELECTED = yaml.safe_load((GRID_ROOT / "selected_bottleneck.yaml").read_text())


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


print("selected :", SELECTED or "— task 02 has not selected a bottleneck yet")
print("task root:", TASK_ROOT)
'''


S1_CODE = r'''
if SELECTED is None:
    display(Markdown("*Task 02 has not written `selected_bottleneck.yaml`. Nothing below can run.*"))
    variant = None
else:
    variant = sweep.ModelVariant(
        label=f"ensemble_{SELECTED['label']}", arm="constrained ensemble",
        overrides={**BASE,
                   "within_region_bottleneck_dim": int(SELECTED["within_region_bottleneck_dim"]),
                   "recurrent_bottleneck_dim": int(SELECTED["recurrent_bottleneck_dim"])})
    display(pd.DataFrame([variant.describe()]))
    display(Markdown(f"Selection rule: *{SELECTED['selection_rule']}*. Worst cell at selection: "
                     f"**{SELECTED['worst_condition']:.4f}**."))
seeds = protocol.protocol_seeds(n_seeds=ENSEMBLE_SEEDS)
'''


S2_CODE = r'''
SUBMIT = False   # <-- set to True to queue the ensemble

if variant is not None:
    flight = sweep.in_flight_run_dirs(TASK_ROOT / "_jobs")
    commands, _ = sweep.variant_job_commands(
        [variant], seeds, root=TASK_ROOT, repo_root=repo_root,
        protocol=SELECTED_PROTOCOL, mrnn_cfg_path=MRNN_CFG_PATH, exclude_run_dirs=flight["run_dirs"])
    inventory = sweep.index_variant_runs(TASK_ROOT, [variant], seeds, in_flight=flight["run_dirs"])
    display(Markdown(
        f"**{int(inventory['complete'].sum())} complete**, **{int(inventory['queued'].sum())} queued**, "
        f"**{int(inventory['pending'].sum())} unqueued** of {len(inventory)}."))
    if SUBMIT and commands:
        from datetime import datetime
        from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

        jobs_dir = TASK_ROOT / "_jobs"; jobs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_file = jobs_dir / f"ensemble_{stamp}.txt"
        write_job_file(job_file, commands)
        submitted = [Path(line.split("--run-dir ")[1].split()[0]) for line in commands]
        job_id = submit_dsq_array_job(
            job_file_path=job_file, sbatch_script_path=jobs_dir / f"ensemble_{stamp}.sh",
            log_dir=jobs_dir / "logs", job_name="mrnn_final_ensemble", partition="psych_gpu",
            cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1")
        sweep.record_submission(jobs_dir, job_id=job_id, run_dirs=submitted, label=f"ensemble {stamp}")
        (jobs_dir / "job_id.txt").write_text(str(job_id) + "\n")
        display(Markdown(f"Submitted **{len(commands)}** runs as array **{job_id}**."))
    elif commands:
        display(Markdown("`SUBMIT` is **False** — nothing was submitted."))
    else:
        display(Markdown("The ensemble is fully trained."))
'''


S3_TEXT = r"""## 3. Does the ensemble fit?

Consistency is only interesting among models that reproduce the data.
"""

S3_CODE = r'''
ARMS = {}
if variant is not None and audit.seed_run_dirs(TASK_ROOT / variant.label):
    ARMS["constrained"] = TASK_ROOT / variant.label
if audit.seed_run_dirs(LADDER_ROOT / "full"):
    ARMS["dense"] = LADDER_ROOT / "full"

if "constrained" not in ARMS:
    display(Markdown("*The ensemble has not been trained yet.*"))
    fit = None
else:
    done = inventory[inventory["complete"]]
    display(sweep.convergence_table(sweep.load_histories(done)).round(5))
    fit = cached("ensemble_fit", lambda: sweep.score_variant_fit(done, ceiling_by_region, ceiling_by_cell=ceiling_by_cell))
    display(fit.groupby("condition")["r2_vs_ceiling"].agg(["mean", "std", "min"]).round(4))
    worst = fit.groupby("seed")["r2_vs_ceiling"].min()
    display(Markdown(f"Worst cell per seed: mean **{worst.mean():.4f}**, min {worst.min():.4f}; "
                     f"{int((worst >= ADEQUATE_BAR).sum())}/{len(worst)} seeds clear {ADEQUATE_BAR}."))
'''


S4_TEXT = r"""## 4. Agreement against its floor — constrained versus dense

The battery from the rebuild (seven measures, three distances, tiered by invariance group),
each against fits of the untrained architecture with the same rank constraints, for the
constrained ensemble and the ten-seed dense network side by side. The question is not
"is agreement high" but "is the constrained margin above the dense margin".
"""

S4_CODE = r'''
if fit is not None and "dense" in ARMS:
    battery = cached("agreement_battery", lambda: pd.concat(
        [audit.agreement_battery(audit.seed_run_dirs(p)).assign(arm=name) for name, p in ARMS.items()]
        + [audit.untrained_agreement_battery(audit.seed_run_dirs(p)[0], n_draws=6).assign(arm=name)
           for name, p in ARMS.items()], ignore_index=True))
    for name in ARMS:
        show(aviz.plot_agreement_battery(battery, audit.AGREEMENT_BATTERY, arm=name),
             f"fig01_battery_{name}")
    sim = [n for n, _, k in audit.AGREEMENT_BATTERY if k == "similarity"]
    show(aviz.plot_battery_across_arms(battery, sim, arm_order=list(ARMS)), "fig02_margins_constrained_vs_dense")
    display(cached("current_agreement", lambda: audit.current_agreement_decomposition(ARMS)).round(3))
'''


S5_TEXT = r"""## 5. Lesions on the trained network

Directed, bidirectional, within-region and whole-region isolation, each silenced in the
converged model, each with a matched random-weight control, damage in absolute squared error.
"""

S5_CODE = r'''
if fit is not None:
    lesions = cached("lesion_battery", lambda: pd.concat(
        [audit.lesion_battery(p).assign(arm=name)
         for name, path in ARMS.items() for p in audit.seed_run_dirs(path)], ignore_index=True))
    for name in ARMS:
        show(aviz.plot_lesion_battery(lesions, arm=name), f"fig03_lesions_{name}")
'''


S6_TEXT = r"""## 6. Do the seeds agree on which connections matter?

Kendall's $\tau$ between seeds' lesion-damage rankings, per lesion family and per arm, against
a permutation null. A ranking that clears its null in the constrained ensemble and not in the
dense one is the result this series was built to find.
"""

S6_CODE = r'''
if fit is not None:
    ranking = cached("lesion_ranking_agreement", lambda: audit.lesion_ranking_agreement(lesions))
    display(ranking.round(3))
    show(aviz.plot_lesion_ranking_agreement(ranking), "fig04_ranking_agreement")
    contrast = cached("lesion_condition_contrast", lambda: audit.lesion_condition_contrast(lesions))
    display(contrast.round(4))
'''


S7 = r"""## 7. Consistencies

*To be specified.* The candidates the rebuild's machinery already computes, any of which can
be dropped in here as a cell:

| candidate | function | what it would show |
|---|---|---|
| within-model condition contrasts (drive alignment, current dimensionality, …) | `audit.invariant_screen`, `audit.screen_consistency` | properties constant across the ten fits, with the target-side control (`audit.target_side_control`) to separate findings from echoes |
| channel alignment between conditions on the same pathway | `audit.condition_channel_alignment`, `audit.untrained_channel_alignment` | whether fixation types share inter-regional channels |
| slow-point stability per condition | `audit.slow_point_stability` | with the corrected Jacobian |
| pathway ranking | Section 6 | already computed |
"""


S8 = r"""## 8. Reading the result

**Fit first.** If the ten fits do not clear the bar, task 02's selection was too tight and
this notebook says so before anything else.

**Constrained against dense.** Every agreement number here is a *difference* between the
constrained ensemble's margin over its floor and the dense ensemble's margin over its. The
rebuild found that constraining the map without reducing the state dimension did not help;
this is the test of whether constraining *both* recurrences at once does.

**The level that is stable.** Weights, spectra, state geometry, pathway ranking, condition
contrast — the chapter's claim is pitched at the highest level that clears its null, and this
notebook is what locates it.
"""


def _cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "id": f"cell-{next(_CELL_COUNTER):02d}", "metadata": {},
            "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build() -> dict:
    cells = [
        _cell("markdown", HEADER), _cell("code", SETUP),
        _cell("code", S1_CODE), _cell("code", S2_CODE),
        _cell("markdown", S3_TEXT), _cell("code", S3_CODE),
        _cell("markdown", S4_TEXT), _cell("code", S4_CODE),
        _cell("markdown", S5_TEXT), _cell("code", S5_CODE),
        _cell("markdown", S6_TEXT), _cell("code", S6_CODE),
        _cell("markdown", S7), _cell("markdown", S8),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "gaze_processing", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
                          "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
                          "pygments_lexer": "ipython3", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    path = Path(__file__).resolve().parent / OUTPUT_FILENAME
    path.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
