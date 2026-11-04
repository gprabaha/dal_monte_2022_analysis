"""Author the partner-identity notebook (task 01b of the final mRNN series).

    conda run -n gaze_processing python notebooks/mrnn_chapter/_build_01b_partner_identity.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

OUTPUT_FILENAME = "01b_partner_identity.ipynb"
_CELL_COUNTER = count(1)


HEADER = r"""# 01b · Does a region need this partner, or any partner?

*Companion to task 01. Reads nothing from it and writes nothing it depends on -- a
self-contained control for the ladder's central claim.*

The ladder asks whether a region reproduces its own trajectories better inside the network
than alone, and finds that it does. But a "pair" network has strictly more parameters than
a "single" network, so part of that gain could be generic extra capacity -- a second
recurrent block helping regardless of what it represents -- rather than anything specific
to the partner region's identity. This task builds the matched control: a **self-pair**
duplicates one region into two blocks that both read its *entire* neuron population -- same
PCA, bit-identical target, independently trained weights per block -- so it has exactly the
architecture and parameter count of a **cross-pair** (two different areas, each also fitted
on its own full population). The only thing that differs is whether a block's partner
carries that area's actual information or a second copy of the scored region's own.

**Every region's PCA is fitted on all of its neurons.** An earlier version of this task
instead split each region's units into two independent, non-overlapping halves, precisely so
that a self-pair would not be a literal duplicate. That bought independence at the cost of
fitting PCs on half the population and complicating the matched ceiling with a per-seed
split; here, fitting on the full population is worth a different confound instead: a
self-pair's two blocks can partly reproduce their target by mirroring each other rather than
by using their own recurrence, so **"self beats cross" is weaker evidence against partner
identity mattering than it would be under independent halves, and "cross beats self" is
correspondingly a *stronger* finding.** Hold that in mind wherever the self-vs-cross contrast
is read below.

Four regions give **ten unique two-block networks**: four self-pairs and six cross-pairs.
Each network is scored on both of its blocks, filling a **4x4 matrix of 16 (scored region,
partner) cells** from those ten fits. If a region reproduces itself equally well paired with
a duplicate of itself as paired with a real different area, the ladder's "the network helps"
result is mostly capacity; if cross-pairs reliably beat self-pairs *despite* the self-pair's
mirroring advantage, the partner's identity matters -- and this task also asks *for which
fixation type* and *for which region* that is truest, and whether a cross-pair does not just
fit as well but gets there faster.

**Design.** Five seeds, one five-seed grid -- 50 fits. Nothing about the target depends on
the training seed any more (no per-seed data split), so a seed now varies only the network's
weight initialisation, the same as every other task in the series. Every fit uses the same
recipe as the rest of the final series: minimax objective, 40 units per block, full dense
connectivity (no rank bottleneck -- this task is about identity, not dimensionality), lr 3e-4
cosine, tanh, spectral radius 1.1, 100k iterations.

| Section | |
|---|---|
| 1 | The grid, and where its data live |
| 2 | Run state and submission (off by default) |
| 3 | The matched ceiling (full-population, needs no checkpoint) |
| 4 | Convergence and adequacy: is every cell actually fitting well? |
| 5 | The 4x4 matrix: does a region need this partner, or any partner? |
| 6 | Which fixation type is most affected? |
| 6b | Does a cross-pair reproduce different frequencies than a self-pair? |
| 7 | Which region -- or which pair -- is most affected? |
| 8 | Does a cross-pair not just fit as well, but converge faster? |
| 9 | What the fits look like: PC and firing-rate reconstruction |
| 10 | Reading the result |

**Nothing here submits a job unless `SUBMIT` or `SUBMIT_RETRY` is set to `True`.**
"""


SETUP = r'''
%load_ext autoreload
%autoreload 2

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

repo_root = Path.cwd()
if not (repo_root / "src").exists():
    repo_root = next(parent for parent in Path.cwd().parents if (parent / "src").exists())
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_audit as audit
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_chapter as ch
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_ensemble as ens
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_partner_identity as pid
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_protocol as protocol
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_sweep as sweep
from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_synthesis as syn
from dal_monte_2022_analysis.ephys.plotting import fixation_mrnn_audit as aviz
from dal_monte_2022_analysis.ephys.plotting import fixation_mrnn_ensemble as eviz
from dal_monte_2022_analysis.ephys.plotting import fixation_mrnn_partner_identity as pviz
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_SHORT_LABELS, REGION_LABELS, ThesisFigureSettings, apply_thesis_plot_style,
    figure_to_png_bytes, save_thesis_figure,
)

DATASET_CFG_PATH = repo_root / "configs" / "dataset.yaml"
apply_thesis_plot_style(load_config(repo_root / "configs" / "plotting.yaml"))
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

TREE = "final"
#: A new task name -- not ``01b_partner_identity`` -- so this full-population grid never
#: shares a directory with the original half-split grid's checkpoints (90 trained, several
#: still in flight at the time this task was rebuilt). Nothing here reads or deletes those;
#: they remain available separately if the half-split design is ever wanted again.
TASK_ROOT = sweep.resolve_task_root("01b_partner_identity_full_population", DATASET_CFG_PATH, tree=TREE)
FIGURE_DIR = syn.resolve_output_dir(DATASET_CFG_PATH, scope="final_01b_partner_identity_full_population")
FIGURES = ThesisFigureSettings(output_dir=FIGURE_DIR)
COND = CONDITION_SHORT_LABELS

N_SEEDS = 5
ARCHITECTURES = pid.partner_identity_architectures()
SEEDS = protocol.protocol_seeds(n_seeds=N_SEEDS)
EPOCHS = 100_000

UNIT_CEILING_PATH = repo_root.parent / "local_data" / "dal_monte_2022" / "analysis_outputs" / "ephys" / "psth" / "fixation_noise_ceiling" / "unit_noise_ceiling.pkl"


def show(figure, stem: str) -> None:
    save_thesis_figure(figure, FIGURES, stem)
    display(Image(data=figure_to_png_bytes(figure, dpi=190)))
    plt.close(figure)


def cached(name: str, build):
    path = TASK_ROOT / "tables" / f"{name}.csv"
    if path.exists() and path.stat().st_size > 1:
        frame = pd.read_csv(path)
        if "seed" in frame.columns:
            frame["seed"] = frame["seed"].astype(str)
        return frame
    frame = build()
    if frame is not None and len(frame):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return frame


def md(text: str) -> None:
    display(Markdown(text))


print(f"{len(ARCHITECTURES)} architectures x {len(SEEDS)} seeds = {len(ARCHITECTURES) * len(SEEDS)} cells")
print("task root:", TASK_ROOT)
'''


S1_TEXT = r"""## 1. The grid, and where its data live

Every architecture is one of ten unique two-block networks, built directly from the
canonical combined fixation-PSTH export -- no pre-written per-seed dataset, since nothing
about a target depends on the training seed any more. A **self-pair**'s two blocks both read
*all* of the scored region's neurons (`pid.duplicate_region_dataframe`: the region's rows are
copied under two virtual labels, `<region>_1` and `<region>_2`, bit-identical); a
**cross-pair**'s two blocks are just its two canonical region names, the same mechanism the
ladder's own region-pairs already use. Every architecture, self or cross, is normalised to
the same scale (`pid.build_architecture_targets`'s `normalization_scale_override`), pooled
once from the untouched four-region dataframe -- otherwise a self-pair's duplicated region
would pool a slightly different scale than a cross-pair's untouched one.
"""

S1_CODE = r'''
table = pd.DataFrame([{"label": a.label, "arm": a.arm, "scored_region": a.scored_region,
                       "partner_region": a.partner_region, "virtual_regions": " + ".join(a.virtual_regions())}
                      for a in ARCHITECTURES])
display(table)
cells = pid.scored_cells(ARCHITECTURES)
md(f"**{len(ARCHITECTURES)} architectures** (4 self-pairs, 6 cross-pairs) **x {len(SEEDS)} seeds = "
   f"{len(ARCHITECTURES) * len(SEEDS)} fits**, filling **{len(cells)} scored (region, partner) cells** "
   "of the 4x4 matrix -- a self-pair fills one diagonal cell, a cross-pair fills two off-diagonal cells "
   "from the same trained network.")
'''


S2_TEXT = r"""## 2. Run state and submission"""

S2_CODE = r'''
SUBMIT = False   # <-- set to True to queue every missing (architecture, seed) cell as one array

flight = sweep.in_flight_run_dirs(TASK_ROOT / "_jobs")
commands, run_dirs = pid.job_commands(ARCHITECTURES, SEEDS, root=TASK_ROOT, repo_root=repo_root,
                                      epochs=EPOCHS, exclude_run_dirs=flight["run_dirs"])
inventory = pid.index_runs(TASK_ROOT, ARCHITECTURES, SEEDS)
display(Markdown(
    f"**{int(inventory['complete'].sum())} complete**, **{len(commands)} queued or unqueued** of {len(inventory)}."))

if SUBMIT and commands:
    from datetime import datetime
    from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

    jobs_dir = TASK_ROOT / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_file = jobs_dir / f"partner_identity_{stamp}.txt"
    write_job_file(job_file, commands)
    job_id = submit_dsq_array_job(
        job_file_path=job_file, sbatch_script_path=jobs_dir / f"partner_identity_{stamp}.sh",
        log_dir=jobs_dir / "logs", job_name="mrnn_partner_identity", partition="psych_gpu",
        cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
    )
    sweep.record_submission(jobs_dir, job_id=job_id, run_dirs=run_dirs, label=f"partner_identity {stamp}")
    (jobs_dir / "job_id.txt").write_text(str(job_id) + "\n")
    display(Markdown(f"Submitted **{len(commands)}** runs as array **{job_id}**."))
elif commands:
    display(Markdown("`SUBMIT` is **False** -- nothing was submitted."))
elif int(inventory["complete"].sum()) == len(inventory):
    display(Markdown("The grid is fully trained."))
else:
    display(Markdown("Every missing cell already has a job in flight (excluded by `in_flight_run_dirs`); "
                     "nothing new to queue."))
'''


S3_TEXT = r"""## 3. The matched ceiling (full-population, needs no checkpoint)

Unlike the half-split design, the full-population PCA basis for a canonical region does not
depend on the training seed -- it is a fixed function of the data and the recipe's own PCA
settings. The ceiling can therefore be computed once, directly from the combined PSTH export,
before any cell finishes training (`pid.full_population_ceiling_by_cell`); a self-pair's two
blocks (`<region>_1`, `<region>_2`) read off their canonical region's value
(`pid.ceiling_lookup`), since a duplicate shares that region's PCA basis exactly.
"""

S3_CODE = r'''
unit_ceiling = pd.read_pickle(UNIT_CEILING_PATH)
ceiling_cell = cached("ceiling_by_cell", lambda: pid.full_population_ceiling_by_cell(DATASET_CFG_PATH, unit_ceiling))
display(ceiling_cell.pivot(index="region", columns="condition", values="reliability").round(4))
lookup = pid.ceiling_lookup(ceiling_cell, ARCHITECTURES)
'''


S4_TEXT = r"""## 4. Convergence and adequacy: is every cell actually fitting well?

The minimax objective optimises the *worst* region x condition cell, but a well-optimised
worst cell is not the same claim as a well-fit one: minimax tells the optimiser what to
prioritise, not whether it succeeded. Two different failures would otherwise look alike in
a bare fit table, so they are checked separately, in order:

1. **Convergence** (`sweep.convergence_table`) -- did training settle, or is the best
   iterate just the luckiest point on a jittery walk? A configuration that merely confuses
   the optimiser looks the same in the final loss as one the data genuinely cannot
   tolerate, and only a flagged convergence failure tells them apart.
2. **Adequacy** (`pid.adequacy_table`) -- *given* a converged fit, does its worst condition
   actually clear the noise ceiling (within the grid's own seed-to-seed spread)? This is the
   "all the models are fitting well" check: not just that the objective's worst cell was
   targeted, but that it ended up near ceiling once training finished.

Any architecture failing step 1 is queued for a retry at a lower learning rate (`SUBMIT_RETRY`
below, off by default) rather than folded into the comparison as-is -- a non-converged fit
would masquerade as "this architecture cannot fit the data" when it may only mean the
inherited learning rate could not optimise it.
"""

S4_CODE = r'''
done = inventory[inventory["complete"]]
if done.empty:
    display(Markdown("*Nothing trained yet.*"))
    fit = matrix_fit = None
else:
    histories_by_seed = pid.load_histories_by_seed(done)
    convergence = sweep.convergence_table({label: list(by_seed.values()) for label, by_seed in histories_by_seed.items()})
    md("**Convergence, per architecture** (`n_converged` out of `n_seeds` on the bar task 00 fixed):")
    display(convergence.round(4))

    SUBMIT_RETRY = False   # <-- set to True to queue a lower-learning-rate retry for every architecture that failed

    retry_flight = sweep.in_flight_run_dirs(TASK_ROOT / "_jobs")
    retry_commands, retry_run_dirs = pid.retry_commands_for_nonconverged(
        ARCHITECTURES, convergence, SEEDS, root=TASK_ROOT, repo_root=repo_root, epochs=EPOCHS,
        exclude_run_dirs=retry_flight["run_dirs"],
    )
    if not retry_commands:
        display(Markdown("Every architecture that has finished training converged; no retry needed."))
    elif SUBMIT_RETRY:
        from datetime import datetime
        from dal_monte_2022_analysis.runtime.hpc.jobs import submit_dsq_array_job, write_job_file

        jobs_dir = TASK_ROOT / "_jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_file = jobs_dir / f"partner_identity_retry_{stamp}.txt"
        write_job_file(job_file, retry_commands)
        job_id = submit_dsq_array_job(
            job_file_path=job_file, sbatch_script_path=jobs_dir / f"partner_identity_retry_{stamp}.sh",
            log_dir=jobs_dir / "logs", job_name="mrnn_partner_identity_retry", partition="psych_gpu",
            cpus_per_task=1, mem_per_cpu="12G", time_limit="06:00:00", gres="gpu:1",
        )
        sweep.record_submission(jobs_dir, job_id=job_id, run_dirs=retry_run_dirs, label=f"partner_identity_retry {stamp}")
        display(Markdown(f"Submitted **{len(retry_commands)}** retry runs as array **{job_id}**."))
    else:
        display(Markdown(f"**{len(retry_commands)}** cells failed convergence and would be retried at a lower "
                         "learning rate -- `SUBMIT_RETRY` is **False**, so nothing was submitted."))

    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import score_variant_fit
    fit = cached("fit", lambda: score_variant_fit(done, ceiling_by_region={}, ceiling_by_cell=lookup))
    matrix_fit = cached("matrix_fit", lambda: pid.matrix_fit_table(fit, ARCHITECTURES))

    adequacy = pid.adequacy_table(matrix_fit)
    md("**Adequacy, per architecture** (worst region x condition cell against ceiling; `adequate` uses the grid's own seed spread as tolerance):")
    display(adequacy.round(4))
    if not adequacy["adequate"].all():
        md(f"**{int((~adequacy['adequate']).sum())} of {len(adequacy)} architectures fall outside tolerance on their worst "
           "condition** -- read §5 onward with that in mind; a failing architecture's own cells are not excluded "
           "below; they are flagged here so a comparison can be reread in light of them.")
'''


S5_TEXT = r"""## 5. The 4x4 matrix: does a region need this partner, or any partner?

The matrix, and the direct test: for each region, is its self-pair fit reliably lower than
the mean of its three cross-pair fits, paired by seed? Recall the self-pair's mirroring
advantage (§ header): this contrast is conservative in the direction that matters -- a
self-pair can partly fit by copying its duplicate partner, so any residual cross-pair
advantage found here is not explained by that advantage.
"""

S5_CODE = r'''
if matrix_fit is not None:
    show(pviz.plot_partner_identity_matrix(matrix_fit), "fig01_matrix")
    display(matrix_fit.groupby(["scored_region", "partner_region", "arm"])["r2_vs_ceiling"].mean()
            .unstack("partner_region").round(4))

    self_vs_cross = pid.self_vs_cross_per_seed(matrix_fit)
    long = self_vs_cross.melt(id_vars=["scored_region", "seed"], value_vars=["self", "cross"],
                              var_name="arm", value_name="value")
    region_tests = ch.paired_contrasts(long, value="value", group="arm", unit="seed",
                                       by="scored_region", pairs=[("cross", "self")])
    show(pviz.plot_self_vs_cross_by_region(long, region_tests), "fig02_self_vs_cross_by_region")
    md("**Self-pair vs. cross-pair, per region** (mean over fits; positive `difference` means cross beats self):")
    display(self_vs_cross.groupby("scored_region")[["self", "cross"]].agg(["mean", "std"]).round(4))
    md("**Tests:**")
    display(region_tests[["scored_region", "n", "mean_a", "mean_b", "difference", "statistic", "p", "p_holm", "stars"]].round(5)
            if not region_tests.empty else region_tests)
'''


S6_TEXT = r"""## 6. Which fixation type is most affected?

The same self-vs-cross contrast, now per fixation type with regions pooled: does
interactive face lose more from being paired with a duplicate of itself than the other two
fixation types do?
"""

S6_CODE = r'''
if matrix_fit is not None:
    cond_per_fit = matrix_fit.groupby(["arm", "seed", "condition"])["r2_vs_ceiling"].mean().reset_index()
    cond_tests = ch.paired_contrasts(cond_per_fit, value="r2_vs_ceiling", group="arm", unit="seed",
                                     by="condition", pairs=[("cross", "self")])
    show(pviz.plot_self_vs_cross_by_condition(matrix_fit, cond_tests), "fig03_self_vs_cross_by_condition")
    md("**Self vs. cross per fixation type** (regions pooled):")
    display(cond_per_fit.groupby(["condition", "arm"])["r2_vs_ceiling"].agg(["mean", "std"]).unstack("arm").round(4))
    md("**Tests:**")
    display(cond_tests[["condition", "n", "mean_a", "mean_b", "difference", "statistic", "p", "p_holm", "stars"]].round(5)
            if not cond_tests.empty else cond_tests)
'''


S6B_TEXT = r"""## 6b. Does a cross-pair reproduce different frequencies than a self-pair?

§5 asks about total fit; this asks *where in the spectrum* self and cross differ, the same
band-resolved recovery the ladder reports (`fixation_mrnn_audit.band_resolved_recovery`):
the fraction of each region's target power, per frequency band, left in the residual.
Regions pooled, paired by seed. A region's own slow envelope is easy either way; if the
self-pair's mirroring advantage (header) shows up anywhere, it should be at the fast bands,
where a block has the least to gain from copying a duplicate of itself and the most to gain
from genuine inter-regional structure -- or, if cross still wins there despite that
advantage, that is the clearest sign the partner's actual identity carries fast structure a
duplicate cannot supply.
"""

S6B_CODE = r'''
if matrix_fit is not None:
    band_recovery = cached("band_recovery", lambda: pd.concat(
        [audit.band_resolved_recovery(r.run_dir).assign(label=r.label) for r in done.itertuples()],
        ignore_index=True))
    band_matrix = pid.matrix_fit_table(band_recovery, ARCHITECTURES)
    band_per_fit = band_matrix.groupby(["arm", "seed", "band"])["residual_fraction"].mean().reset_index()
    band_tests = ch.paired_contrasts(band_per_fit, value="residual_fraction", group="arm", unit="seed",
                                     by="band", pairs=[("self", "cross")])
    show(aviz.plot_band_recovery_self_vs_cross(band_per_fit, band_tests), "fig03b_band_recovery_self_vs_cross")
    md("**Residual (unreproduced) fraction of target power, self-pair vs. cross-pair** "
       "(regions pooled; `difference` = self − cross; positive means cross recovers more of that band):")
    display(band_per_fit.groupby(["band", "arm"])["residual_fraction"].mean().unstack("arm").round(4))
    md("**Tests:**")
    display(band_tests[["band", "n", "mean_a", "mean_b", "difference", "statistic", "p", "p_holm", "stars"]]
            .rename(columns={"mean_a": "self", "mean_b": "cross"}).round(5)
            if not band_tests.empty else band_tests)
'''


S7_TEXT = r"""## 7. Which region -- or which pair -- is most affected?

Figure 2 already shows whether the self-vs-cross gap is uniform across regions or
concentrated in one or two. This section adds the finer-grained read: for each scored
region, which *specific* partner helps most, and does the identity of the partner (not just
"a partner exists") matter within the six cross-pairs?
"""

S7_CODE = r'''
if matrix_fit is not None:
    per_partner = matrix_fit.groupby(["scored_region", "partner_region", "arm", "seed"])["r2_vs_ceiling"].mean().reset_index()
    md("**Every partner, per scored region** (mean over fits and fixation types; the diagonal is the self-pair):")
    display(per_partner.groupby(["scored_region", "partner_region"])["r2_vs_ceiling"].mean().unstack("partner_region").round(4))

    cross_only = per_partner[per_partner["arm"] == "cross"]
    if cross_only["scored_region"].nunique() and (cross_only.groupby("scored_region")["partner_region"].nunique() >= 2).any():
        partner_tests = []
        for region, block in cross_only.groupby("scored_region"):
            wide = block.pivot(index="seed", columns="partner_region", values="r2_vs_ceiling")
            if wide.shape[1] >= 2 and wide.shape[0] >= 3:
                partners = list(wide.columns)
                partner_tests.append(
                    ch.welch_contrasts(block, value="r2_vs_ceiling", group="partner_region",
                                       pairs=[(partners[i], partners[j]) for i in range(len(partners)) for j in range(i + 1, len(partners))])
                    .assign(scored_region=region)
                )
        if partner_tests:
            partner_tests = pd.concat(partner_tests, ignore_index=True)
            md("**Does the specific cross-region partner matter, within the three real partners of one region?** "
               "(Welch's t between partners, Holm-corrected within each region; empty means no contrast reached significance):")
            display(partner_tests[partner_tests["significant"]][["scored_region", "a", "b", "difference", "p_holm", "stars"]]
                    if not partner_tests.empty else partner_tests)
'''


S8_TEXT = r"""## 8. Does a cross-pair not just fit as well, but converge faster?

Fitting equally well (§5) and getting there equally fast are different claims. Two fits can
land at the same final adequacy while one took a fraction of the iterations to get there --
not visible in the final cell alone, only in the training trajectory. Scored on
`worst_cell_unexplained`, the same per-iteration quantity the minimax objective optimises, so
"converges faster" means faster on the objective's own terms.
"""

S8_CODE = r'''
if matrix_fit is not None and done is not None and not done.empty:
    speed = pid.convergence_speed_table(histories_by_seed, ARCHITECTURES)
    show(pviz.plot_convergence_curves(histories_by_seed, ARCHITECTURES), "fig05_convergence_curves")
    speed_tests = ch.paired_contrasts(speed, value="iterations_to_tolerance", group="arm", unit="seed",
                                      by="scored_region", pairs=[("cross", "self")])
    show(pviz.plot_convergence_speed(speed, speed_tests), "fig06_convergence_speed")
    md("**Iterations to converge, self vs. cross, per region** (lower = faster):")
    display(speed.groupby(["scored_region", "arm"])["iterations_to_tolerance"].agg(["mean", "std"]).unstack("arm").round(1))
    md("**Tests:**")
    display(speed_tests[["scored_region", "n", "mean_a", "mean_b", "difference", "statistic", "p", "p_holm", "stars"]].round(5)
            if not speed_tests.empty else speed_tests)
'''


S9_TEXT = r"""## 9. What the fits look like: PC and firing-rate reconstruction

Target-prominent, model-thin overlays (the chapter's Figure 2 convention), in true native
units with no vertical offset between components or units -- one small panel per
component/unit x fixation type, every seed's reconstruction overlaid on the one observed
trace. Shown in both spaces: the region-PC space the model is actually trained in, and the
PC-backprojected firing-rate space the neurons were recorded in, for one region's leading
components/units, self-paired against its best cross-pair, across every seed.
"""

S9_CODE = r'''
GALLERY_REGION = "bla"

if matrix_fit is not None:
    best_partner = (matrix_fit[(matrix_fit["scored_region"] == GALLERY_REGION) & (matrix_fit["arm"] == "cross")]
                    .groupby("partner_region")["r2_vs_ceiling"].mean().idxmax())
    self_label = f"self_{GALLERY_REGION}"
    cross_label = next(a.label for a in ARCHITECTURES if a.arm == "cross"
                       and {a.scored_region, a.partner_region} == {GALLERY_REGION, best_partner})
    md(f"Best cross-partner for {REGION_LABELS.get(GALLERY_REGION, GALLERY_REGION)}: "
       f"**{REGION_LABELS.get(best_partner, best_partner)}** (`{cross_label}`).")

    for label, arm in ((self_label, "self"), (cross_label, "cross")):
        block = done[done["label"] == label]
        arch = next(a for a in ARCHITECTURES if a.label == label)
        virtual = arch.scored_region if arm == "cross" else arch.virtual_regions()[0]
        for space, fig_stem in (("pc", "fig07_gallery_pc"), ("fr", "fig08_gallery_fr")):
            traces = cached(f"gallery_{GALLERY_REGION}_{arm}_{space}",
                            lambda block=block, virtual=virtual, space=space: ens.reconstruction_traces(
                                list(block["run_dir"]), region=virtual, space=space))
            if traces is not None and len(traces):
                space_label = "region PCs" if space == "pc" else "PC-backprojected firing rate"
                show(eviz.plot_reconstruction_overlay(traces, arm=arm,
                     title=f"{REGION_LABELS.get(GALLERY_REGION, GALLERY_REGION)} ({arm}), {space_label} · "
                           "target (black) and every seed's fit"),
                     f"{fig_stem}_{GALLERY_REGION}_{arm}")
'''


S10 = r"""## 10. Reading the result

Three things this task can support, in order of what they would mean:

1. **Whether the ladder's "the network helps" result survives matched parameter count.**
   The self-vs-cross contrast in §5, per region and Holm-corrected across all four, read
   against the self-pair's mirroring advantage (header): if self-pairs reproduce a region
   about as well as cross-pairs do *despite* that advantage, the ladder's gain was mostly
   extra capacity; if cross-pairs still win, the partner's content matters, and more clearly
   so than an independent-halves design could show.
2. **Whether that matters more for one fixation type, or gets there faster.** §6 mirrors how
   the chapter reads "does this constraint cost interactive face more"; §8 asks the
   complementary question of training speed, not just final adequacy.
3. **Whether it is one region, or a specific pair, carrying the effect.** §7's partner
   breakdown is descriptive at five seeds; a genuine ranking among the three real partners
   of one region needs more seeds than a coarse self-vs-cross average does.

Read all three only in light of §4: an architecture flagged as non-converged or inadequate
there is not a clean read of "the data cannot tolerate this architecture" until it has had
its retry.

**Deferred: the rank-bottleneck probe.** The half-split design's final sections asked whether
a dense network's spare capacity was masking a larger partner-identity effect, by retraining
at a tight cross-region rank. That probe is not rebuilt here -- revisit it once the dense
full-population grid's own read is in hand, since (1) the self-pair's mirroring advantage
changes what a "mask" would even mean here, and (2) the half-split design's region-pair
size-asymmetry motivation does not carry over the same way: a full-population self-pair is
trivially size-matched (a region against all of itself), where the half-split one was an
independent half against an independent half. The legacy functions
(`pid.relabel_combined_dataframe`, `pid.region_pair_asymmetry`, ...) and the half-split
grid's own checkpoints remain available if this is picked back up.
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
        _cell("markdown", S1_TEXT), _cell("code", S1_CODE),
        _cell("markdown", S2_TEXT), _cell("code", S2_CODE),
        _cell("markdown", S3_TEXT), _cell("code", S3_CODE),
        _cell("markdown", S4_TEXT), _cell("code", S4_CODE),
        _cell("markdown", S5_TEXT), _cell("code", S5_CODE),
        _cell("markdown", S6_TEXT), _cell("code", S6_CODE),
        _cell("markdown", S6B_TEXT), _cell("code", S6B_CODE),
        _cell("markdown", S7_TEXT), _cell("code", S7_CODE),
        _cell("markdown", S8_TEXT), _cell("code", S8_CODE),
        _cell("markdown", S9_TEXT), _cell("code", S9_CODE),
        _cell("markdown", S10),
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
