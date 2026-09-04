"""Shared machinery for the constraint sweeps.

Tasks 01-04 all ask the same shape of question -- take the training recipe task 00 froze,
vary one structural choice, fit several seeds, and ask whether the model still reproduces
the data under that constraint -- so they share one implementation. A task supplies only
its list of :class:`ModelVariant` overrides.

The framing matters for what gets scored. These sweeps are not looking for the
lowest-loss model: with roughly as many free parameters as the dataset has numbers,
plenty of configurations fit. They are looking for **which constraints the data can
tolerate**, so a variant is interesting when it fits *despite* being restricted. That is
why every variant is scored against the measured noise ceiling rather than against 1.0,
carries its parameter count next to its fit, and reports agreement across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Collection, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_protocol import (
    ConvergenceBar,
    evaluate_convergence,
    resolve_chapter_root,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
    backproject_replay_outputs_to_firing_rates,
    reconstruction_accuracy,
    replay_fixation_mrnn_run,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_training import (
    FixationMRNNRunSettings,
    load_fixation_mrnn_config,
    settings_from_config,
)

CONDITION_ORDER: tuple[str, ...] = ("face_interactive", "face_non_interactive", "object")


@dataclass(frozen=True)
class ModelVariant:
    """One structural configuration: a label plus the settings it overrides."""

    label: str
    overrides: Mapping[str, object] = field(default_factory=dict)
    #: Free-text group used to split arms of a sweep in tables and figures.
    arm: str = "main"

    def describe(self) -> dict[str, object]:
        return {"label": self.label, "arm": self.arm, **dict(self.overrides)}


#: Optimiser used when task 00 has not yet frozen a recipe. These match what task 00
#: selected on the corrected architecture. Kept as a fallback rather than removed because
#: the notebooks have to stay readable while their dependency is running -- but using it is
#: flagged rather than silent, because the optimum is architecture-dependent: it moved from
#: 1e-3 to 3e-4 when the rank constraint and the within-region penalty were removed.
PROVISIONAL_OPTIMIZER: dict[str, object] = {
    "lr": 3e-4,
    "gradient_clip_norm": 0.05,
    "lr_schedule": "cosine",
    "lr_warmup_iterations": 0,
    "lr_min_factor": 0.01,
    "activation": "tanh",
    "spectral_radius": 1.1,
}
PROVISIONAL_EPOCHS = 100_000


def load_selected_protocol(path: str | Path) -> dict[str, object]:
    """The training recipe task 00 froze, which every sweep here inherits."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_selected_protocol_or_provisional(path: str | Path) -> tuple[dict[str, object], bool]:
    """The frozen recipe if task 00 has produced one, otherwise a flagged fallback.

    Returns ``(protocol, is_provisional)``. Loading eagerly and crashing when the file is
    absent makes a downstream notebook unreadable while its dependency is still running,
    which is the opposite of useful -- these sweeps take hours and the notebooks are meant
    to be reviewed before they are submitted.

    The fallback pairs the *current* :data:`PROTOCOL_ARCHITECTURE` with the optimiser task
    00's first pass chose, so only the optimiser is provisional. Callers are expected to
    surface that and to gate submission on it.
    """
    path = Path(path)
    if path.exists():
        return load_selected_protocol(path), False
    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_protocol import PROTOCOL_ARCHITECTURE

    return (
        {
            "selected_label": "provisional (task 00 has not run)",
            "epochs": PROVISIONAL_EPOCHS,
            "optimizer": dict(PROVISIONAL_OPTIMIZER),
            "architecture": dict(PROTOCOL_ARCHITECTURE),
        },
        True,
    )


def build_variant_settings(
    variant: ModelVariant,
    *,
    protocol: Mapping[str, object],
    mrnn_cfg_path: str | Path,
    seed: int,
) -> FixationMRNNRunSettings:
    """Task-00 recipe with this variant's overrides applied on top.

    The variant always wins: the protocol supplies the optimiser and the parts of the
    architecture the variant does not name, so exactly one thing differs between cells.
    """
    base = settings_from_config(load_fixation_mrnn_config(mrnn_cfg_path))
    inherited = {**dict(protocol["architecture"]), **dict(protocol["optimizer"])}
    for key in variant.overrides:
        inherited.pop(key, None)
    return replace(
        base,
        **inherited,
        **dict(variant.overrides),
        epochs=int(protocol["epochs"]),
        seed=int(seed),
        # A sweep that counts failures must not have them rescued by the guardrails.
        divergence_loss_threshold=None,
    )


def variant_run_dir(root: str | Path, variant: ModelVariant, seed: int) -> Path:
    return Path(root) / variant.label / f"seed={int(seed)}"


def variant_job_commands(
    variants: Sequence[ModelVariant],
    seeds: Sequence[int],
    *,
    root: str | Path,
    repo_root: str | Path,
    protocol: Mapping[str, object],
    mrnn_cfg_path: str | Path,
    conda_env: str = "gaze_processing",
    skip_existing: bool = True,
    exclude_run_dirs: Collection[str | Path] = (),
) -> tuple[list[str], list[Path]]:
    """One shell command per missing (variant, seed) cell, each with a frozen config.

    ``exclude_run_dirs`` holds cells that some other array has already claimed but has not
    finished, from :func:`in_flight_run_dirs`. They are invisible on disk -- a queued cell
    and a never-submitted one look identical, because the checkpoint is written only at the
    end -- so without it a second submission while an array is in flight would queue a
    duplicate of every unfinished cell, and the duplicate would train into the same
    directory as the original.
    """
    repo_root = Path(repo_root)
    claimed = {str(Path(path).resolve()) for path in exclude_run_dirs}
    commands: list[str] = []
    run_dirs: list[Path] = []
    for variant in variants:
        for seed in seeds:
            run_dir = variant_run_dir(root, variant, seed)
            run_dirs.append(run_dir)
            if skip_existing and (run_dir / "checkpoint_best.pth").exists():
                continue
            if str(run_dir.resolve()) in claimed:
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            settings = build_variant_settings(
                variant, protocol=protocol, mrnn_cfg_path=mrnn_cfg_path, seed=seed
            )
            payload = {key: value for key, value in settings.__dict__.items()}
            payload["region_order"] = list(payload["region_order"])
            payload["condition_order"] = list(payload["condition_order"])
            config_path = run_dir / "run_config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=True)
            commands.append(
                " ".join(
                    [
                        f"cd {repo_root} &&",
                        "FIXATION_MRNN_PROGRESS=off",
                        f"conda run -n {conda_env} python",
                        "scripts/ephys/modeling/train_fixation_mrnn_into_run_dir.py",
                        f"--mrnn-cfg {config_path}",
                        f"--run-dir {run_dir}",
                        f"--seed {int(seed)}",
                        "--device auto",
                        "--overwrite",
                    ]
                )
            )
    return commands, run_dirs


def index_variant_runs(
    root: str | Path,
    variants: Sequence[ModelVariant],
    seeds: Sequence[int],
    *,
    in_flight: Collection[str | Path] = (),
) -> pd.DataFrame:
    """One row per (variant, seed) cell with its completion state.

    ``pending`` counts only cells that nothing has claimed, so it is what a submission
    would actually queue; cells sitting in a live array are reported as ``queued`` instead.
    """
    claimed = {str(Path(path).resolve()) for path in in_flight}
    rows: list[dict[str, object]] = []
    for variant in variants:
        for seed in seeds:
            run_dir = variant_run_dir(root, variant, seed)
            complete = (run_dir / "checkpoint_best.pth").exists()
            failed = (run_dir / "training_failed.json").exists()
            queued = not complete and not failed and str(run_dir.resolve()) in claimed
            rows.append(
                {
                    **variant.describe(),
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "complete": bool(complete),
                    "diverged": bool(failed and not complete),
                    "queued": bool(queued),
                    "pending": not complete and not failed and not queued,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Which cells a live array has already claimed
# --------------------------------------------------------------------------------------
#
# A cell writes its checkpoint only when training ends, so on disk a cell that is queued
# looks exactly like a cell that was never submitted. The first version of these notebooks
# handled that by refusing to submit anything at all while any array was live, which is
# safe but blocks every independent arm behind the slowest one -- with a 20-GPU cap and
# six-hour runs that costs days. Recording what each submission claimed makes the
# distinction explicit, so a rerun can queue the arms that nothing owns.


def submission_record_path(jobs_dir: str | Path, job_id: str | int) -> Path:
    return Path(jobs_dir) / f"submitted_{job_id}.json"


def record_submission(
    jobs_dir: str | Path,
    *,
    job_id: str | int,
    run_dirs: Sequence[str | Path],
    label: str = "",
) -> Path:
    """Write down which cells a submitted array owns."""
    import json
    from datetime import datetime, timezone

    path = submission_record_path(jobs_dir, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": str(job_id),
        "label": label,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dirs": [str(Path(run_dir).resolve()) for run_dir in run_dirs],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _run_dirs_in_job_file(job_file: str | Path) -> list[str]:
    """The ``--run-dir`` arguments of every command in a dSQ job file."""
    import shlex

    run_dirs: list[str] = []
    for line in Path(job_file).read_text().splitlines():
        tokens = shlex.split(line)
        for index, token in enumerate(tokens[:-1]):
            if token == "--run-dir":
                run_dirs.append(str(Path(tokens[index + 1]).resolve()))
    return run_dirs


def backfill_submission_records(jobs_dir: str | Path) -> list[Path]:
    """Reconstruct records for arrays submitted before submissions were recorded.

    The job file dSQ was handed is the exact list of commands that were queued, and
    ``job_id.txt`` names the array they became, so the pairing is recoverable rather than
    lost. Without this, an array submitted by the earlier code would look unclaimed and be
    duplicated by the first rerun.
    """
    jobs_dir = Path(jobs_dir)
    id_file = jobs_dir / "job_id.txt"
    if not id_file.exists():
        return []
    job_id = id_file.read_text().strip()
    if not job_id or submission_record_path(jobs_dir, job_id).exists():
        return []
    candidates = [path for path in sorted(jobs_dir.glob("*.txt")) if path.name != "job_id.txt"]
    if not candidates:
        return []
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return [
        record_submission(
            jobs_dir,
            job_id=job_id,
            run_dirs=_run_dirs_in_job_file(newest),
            label=f"backfilled from {newest.name}",
        )
    ]


def _array_job_states(job_ids: Sequence[str]) -> dict[str, dict[str, int]]:
    """Per-array task-state counts from one ``squeue`` call.

    ``%F`` is the base array id, so tasks of ``29250099_[18-20]`` are attributed to
    ``29250099`` rather than counted as separate jobs.
    """
    import subprocess

    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            ["squeue", "-j", ",".join(sorted(set(job_ids))), "-h", "-o", "%F %T"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    states: dict[str, dict[str, int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        base, state = parts
        states.setdefault(base, {})
        states[base][state] = states[base].get(state, 0) + 1
    return states


#: Slurm states that mean a cell is still owned by its array.
LIVE_JOB_STATES: frozenset[str] = frozenset(
    {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
)


def in_flight_run_dirs(*jobs_dirs: str | Path) -> dict[str, object]:
    """Cells owned by an array that is still on the queue, across several job directories.

    Returns ``{"run_dirs": set[str], "jobs": [...]}``. Arms of one task submit into
    separate job directories, and a rerun has to know about all of them at once, so this
    takes as many as it is given.
    """
    import json

    records: list[dict[str, object]] = []
    for jobs_dir in jobs_dirs:
        jobs_dir = Path(jobs_dir)
        if not jobs_dir.exists():
            continue
        backfill_submission_records(jobs_dir)
        for path in sorted(jobs_dir.glob("submitted_*.json")):
            with path.open("r", encoding="utf-8") as handle:
                records.append(json.load(handle))
    states = _array_job_states([str(record["job_id"]) for record in records])
    claimed: set[str] = set()
    jobs: list[dict[str, object]] = []
    for record in records:
        job_id = str(record["job_id"])
        job_states = states.get(job_id, {})
        active = any(state in LIVE_JOB_STATES for state in job_states)
        if active:
            claimed.update(str(path) for path in record["run_dirs"])
        jobs.append(
            {
                "job_id": job_id,
                "label": record.get("label", ""),
                "active": active,
                "cells": len(record["run_dirs"]),
                "states": job_states,
            }
        )
    return {"run_dirs": claimed, "jobs": jobs}


def describe_in_flight(state: Mapping[str, object]) -> str:
    """One markdown line per live array, for the submission cell to display."""
    lines = []
    for job in state["jobs"]:  # type: ignore[index]
        if not job["active"]:
            continue
        detail = ", ".join(f"{n} {s.lower()}" for s, n in sorted(job["states"].items()))
        label = f" ({job['label']})" if job["label"] else ""
        lines.append(f"- array `{job['job_id']}`{label}: {job['cells']} cells — {detail}")
    if not lines:
        return "No array from this task is on the queue."
    return "**Live arrays owning cells in this task:**\n" + "\n".join(lines)


def load_histories(inventory: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
    """Loss history per variant, one frame per completed seed."""
    histories: dict[str, list[pd.DataFrame]] = {}
    for label, block in inventory[inventory["complete"].astype(bool)].groupby("label", sort=False):
        runs = []
        for run_dir in block["run_dir"]:
            path = Path(run_dir) / "history.csv"
            if path.exists():
                runs.append(pd.read_csv(path))
        if runs:
            histories[label] = runs
    return histories


def convergence_table(
    histories: Mapping[str, Sequence[pd.DataFrame]],
    bar: ConvergenceBar = ConvergenceBar(),
) -> pd.DataFrame:
    """Did each variant converge, on the bar task 00 fixed?

    A constraint that merely makes the model harder to optimise looks the same in the
    final loss as a constraint the data genuinely cannot tolerate. Separating them is
    what this table is for, and it has to be read before any fit comparison.
    """
    rows: list[dict[str, object]] = []
    for label, runs in histories.items():
        scored = [evaluate_convergence(h["loss"].to_numpy(dtype=float), bar) for h in runs]
        best = [float(np.nanmin(h["loss"].to_numpy(dtype=float))) for h in runs]
        rows.append(
            {
                "label": label,
                "n_seeds": len(runs),
                "n_converged": int(sum(s["passes"] for s in scored)),
                "worst_late_spikes": int(max(s["late_spikes"] for s in scored)),
                "worst_final_over_best": float(max(s["final_over_best"] for s in scored)),
                "median_best_loss": float(np.median(best)),
                "best_loss_spread": float(max(best) / min(best)) if min(best) > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


#: Learning rates tried, in order, when a variant fails to converge at the inherited one.
#: The ladder exists because the optimum is architecture-dependent: on this problem the
#: best learning rate moved by a factor of three when the rank constraint and the
#: within-region penalty were removed, so a recipe tuned on the unconstrained model is not
#: guaranteed to optimise a constrained one.
LEARNING_RATE_LADDER: tuple[float, ...] = (3e-4, 1e-4, 3e-5)


def retry_variants_for_nonconverged(
    variants: Sequence[ModelVariant],
    convergence: pd.DataFrame,
    *,
    inherited_lr: float,
    ladder: Sequence[float] = LEARNING_RATE_LADDER,
) -> list[ModelVariant]:
    """Re-fit variants that failed the convergence bar, at successively lower learning rates.

    Without this a constraint sweep cannot distinguish its two possible failures. A variant
    that scores badly may be telling us the data cannot be reproduced under that
    constraint -- the result the sweep is for -- or merely that the inherited recipe cannot
    optimise it, which is a statement about the optimiser and not about the brain. Since the
    recipe is selected on the *unconstrained* model, the second failure is the more likely
    one, and it would masquerade as the first.

    Retrying only the failures, and recording which learning rate each variant needed,
    turns that confound into a reported quantity: "this constraint required a smaller step"
    is itself informative about the loss landscape it induces.
    """
    if convergence.empty:
        return []
    failed = set(
        convergence.loc[convergence["n_converged"] < convergence["n_seeds"], "label"]
    )
    lower = [rate for rate in ladder if float(rate) < float(inherited_lr)]
    retries: list[ModelVariant] = []
    for variant in variants:
        if variant.label not in failed:
            continue
        for rate in lower:
            retries.append(
                ModelVariant(
                    label=f"{variant.label}__lr{rate:g}".replace(".", "p").replace("-", "m"),
                    overrides={**dict(variant.overrides), "lr": float(rate)},
                    arm=f"{variant.arm} (retry)",
                )
            )
    return retries


def resolve_best_converged(
    convergence: pd.DataFrame,
    *,
    base_labels: Sequence[str],
) -> pd.DataFrame:
    """For each base variant, the run that converged, whatever learning rate it needed.

    Retry labels carry a ``__lr`` suffix, so a base variant and its retries collapse to one
    row here. Reporting the learning rate each variant ended up needing keeps the
    comparison honest: variants fitted at different step sizes are still comparable on fit,
    but the difference has to be visible rather than hidden in the label.
    """
    rows: list[dict[str, object]] = []
    for base in base_labels:
        candidates = convergence[
            (convergence["label"] == base) | (convergence["label"].str.startswith(f"{base}__lr"))
        ]
        if candidates.empty:
            continue
        converged = candidates[candidates["n_converged"] == candidates["n_seeds"]]
        chosen = (
            converged.sort_values("median_best_loss").iloc[0]
            if len(converged)
            else candidates.sort_values("n_converged", ascending=False).iloc[0]
        )
        label = str(chosen["label"])
        rows.append(
            {
                "variant": base,
                "used_label": label,
                "needed_lower_lr": "__lr" in label,
                "converged": int(chosen["n_converged"]) == int(chosen["n_seeds"]),
                "n_converged": int(chosen["n_converged"]),
                "n_seeds": int(chosen["n_seeds"]),
                "median_best_loss": float(chosen["median_best_loss"]),
            }
        )
    return pd.DataFrame(rows)


def count_trainable_parameters(run_dir: str | Path, *, device: str = "cpu") -> dict[str, int]:
    """Free parameters that actually influence the output, split by role.

    ``model.parameters()`` cannot be used directly. The wrapper assembles its recurrent
    matrix from block parameters and copies the result into ``mrnn.W_rec`` on every
    forward pass, so that 200x200 tensor is registered as a parameter, is never in the
    gradient path, and receives no gradient -- counting it inflates the total roughly
    six-fold. The mask and sign tensors are likewise structural. This function therefore
    runs one forward/backward pass and counts only what receives a gradient, plus the
    trained initial state, which is optimized alongside the model but held outside it.

    The ratio of this to ``target_numbers`` is what decides how much a good fit means.
    """
    import torch

    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    model = replay["model"]
    inputs = replay["inp"].detach().clone()
    initial_state = replay["h0"].detach().clone()

    model.zero_grad(set_to_none=True)
    outputs = model(inputs, initial_state, noise=False)
    probe = sum(value.pow(2).mean() for value in outputs["output_by_region"].values())
    probe.backward()

    counts = {"within_region": 0, "inter_region": 0, "input": 0, "readout": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        size = int(parameter.numel())
        if name.startswith("_within_region_param"):
            counts["within_region"] += size
        elif name.startswith(("_inter_left", "_inter_right", "_inter_dense")):
            counts["inter_region"] += size
        elif name.startswith("output_heads"):
            counts["readout"] += size
        elif "W_inp" in name:
            counts["input"] += size
        else:
            counts["other"] += size
    model.zero_grad(set_to_none=True)

    counts["initial_state"] = int(np.asarray(replay["checkpoint"]["h0"]).size)
    counts["total"] = int(sum(v for k, v in counts.items() if k != "total"))
    target = replay["checkpoint"]["target_by_region"]
    counts["target_numbers"] = int(sum(np.asarray(v).size for v in target.values()))
    counts["parameters_per_datum"] = counts["total"] / max(counts["target_numbers"], 1)
    return counts


def score_variant_fit(
    inventory: pd.DataFrame,
    ceiling_by_region: Mapping[str, float],
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Region x condition fit for every completed run, against the noise ceiling."""
    frames = []
    for _, run in inventory[inventory["complete"].astype(bool)].iterrows():
        replay = replay_fixation_mrnn_run(Path(run["run_dir"]), device=device)
        accuracy = reconstruction_accuracy(replay)
        accuracy["ceiling"] = [float(ceiling_by_region.get(str(r), np.nan)) for r in accuracy["region"]]
        accuracy["r2_vs_ceiling"] = accuracy["r2"] / accuracy["ceiling"]
        frames.append(accuracy.assign(label=run["label"], seed=run["seed"], run_dir=run["run_dir"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def representative_traces(
    run_dir: str | Path,
    *,
    region: str,
    space: str = "pc",
    indices: Sequence[int] = (0, 1, 2),
    device: str = "cpu",
) -> pd.DataFrame:
    """Observed and reconstructed traces for a fixed set of components or units.

    ``indices`` is supplied by the caller and held constant across every model in a
    sweep, which is what makes the gallery rows comparable -- picking each model's own
    best-fitting component would compare different quantities.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    checkpoint = replay["checkpoint"]
    timeline = np.asarray(checkpoint["timeline_s"], dtype=float)
    if space == "fr":
        observed = np.asarray(checkpoint["pc_reconstructed_raw_by_region"][region], dtype=float)
        predicted = np.asarray(backproject_replay_outputs_to_firing_rates(replay)[region], dtype=float)
    else:
        observed = np.asarray(checkpoint["target_by_region"][region], dtype=float)
        predicted = replay["output_by_region"][region].detach().cpu().numpy().astype(float)

    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(replay["condition_order"]):
        for index in indices:
            if index >= observed.shape[-1]:
                continue
            for time_index, time_s in enumerate(timeline):
                rows.append(
                    {
                        "region": region,
                        "condition": condition,
                        "space": space,
                        "index": int(index),
                        "time_s": float(time_s),
                        "observed": float(observed[condition_index, time_index, index]),
                        "predicted": float(predicted[condition_index, time_index, index]),
                    }
                )
    return pd.DataFrame(rows)


def gallery_traces(
    inventory: pd.DataFrame,
    *,
    region: str,
    space: str = "pc",
    indices: Sequence[int] = (0, 1, 2),
    seed: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """One representative seed's traces for every variant, ready to plot as a gallery.

    Visual verification is the check that matters and it does not scale: looking at every
    trace of every model is not feasible. The compromise is one row per variant on a
    fixed set of traces, scannable at a glance, with the full-detail plots reserved for
    whichever model the sweep selects.
    """
    completed = inventory[inventory["complete"].astype(bool)]
    frames = []
    for label, block in completed.groupby("label", sort=False):
        chosen = block[block["seed"] == seed] if seed is not None else block
        if chosen.empty:
            chosen = block
        run_dir = chosen.sort_values("seed").iloc[0]["run_dir"]
        frames.append(
            representative_traces(run_dir, region=region, space=space, indices=indices, device=device)
            .assign(label=label)
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def decompose_isolation_fit(
    fit: pd.DataFrame,
    *,
    isolated_region_by_label: Mapping[str, str],
    baseline_label: str,
) -> pd.DataFrame:
    """Split an isolation variant's fit into the region cut off and the ones left talking.

    Cutting every connection into and out of one region changes two things at once, and
    the loss is a sum over all four regions' readouts, so a single pooled score cannot
    say which changed:

    - the isolated region now has to reproduce its own trajectories **autonomously**,
      driven only by the condition input and its trained initial state;
    - the remaining three have to reproduce theirs **without** that region's input.

    Those are different questions with different answers, and only the second speaks to
    whether the region is necessary to the rest of the network. Reporting them separately
    is the whole point: an isolation that leaves the remaining regions untouched while
    wrecking the isolated one says that region depends on the others, not the reverse.
    """
    reference = fit[fit["label"] == baseline_label]
    if reference.empty:
        raise ValueError(f"baseline {baseline_label!r} not present in the fit table")
    baseline_by_region = reference.groupby("region")["r2_vs_ceiling"].mean()

    rows: list[dict[str, object]] = []
    for label, region in isolated_region_by_label.items():
        block = fit[fit["label"] == label]
        if block.empty:
            continue
        by_region = block.groupby("region")["r2_vs_ceiling"].mean()
        remaining = [r for r in by_region.index if str(r) != str(region)]
        rows.append(
            {
                "label": label,
                "isolated_region": region,
                "isolated_fit": float(by_region.get(region, np.nan)),
                "isolated_baseline": float(baseline_by_region.get(region, np.nan)),
                "isolated_cost": float(baseline_by_region.get(region, np.nan) - by_region.get(region, np.nan)),
                "remaining_fit": float(by_region[remaining].mean()) if remaining else np.nan,
                "remaining_baseline": float(baseline_by_region[remaining].mean()) if remaining else np.nan,
                "remaining_cost": float(
                    baseline_by_region[remaining].mean() - by_region[remaining].mean()
                ) if remaining else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Within-region sparsity
# --------------------------------------------------------------------------------------


def within_region_sparsity(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """How concentrated each region's internal recurrent block actually is.

    An L1 penalty is an *input*; sparsity is the *outcome*, and the map between them is
    steep enough that the scale cannot stand in for the result -- on this model a penalty
    of 0.01 produces not a sparse block but an absent one, with weights five orders of
    magnitude below the cross-region blocks.

    Three measures, because each fails differently:

    - ``participation_ratio`` -- :math:`(\sum|w|)^2/\sum w^2`, the effective number of
      contributing weights. Threshold-free, and directly comparable to the block size.
    - ``gini`` -- concentration on a 0-1 scale, invariant to overall magnitude, so it
      separates "shrunk uniformly" from "genuinely sparse".
    - ``fraction_near_zero`` -- readable, but depends on the threshold, so it is reported
      alongside rather than alone.

    The cross-region blocks are measured too. They carry no penalty, so they are the
    within-subject control for what an unregularised block of this model looks like.

    **Read the scale-free measures together with the norm.** An L1 penalty can produce two
    very different outcomes that Gini and the participation ratio cannot tell apart: a few
    large weights surviving (genuine sparsity, Gini rises) or every weight shrinking
    together (an ablation, Gini barely moves). Measured on this model, a penalty of 0.01
    gives the second -- Gini moves 0.37 to 0.43 while the block norm collapses by five
    orders of magnitude. :func:`summarize_sparsity` therefore reports the within-to-cross
    norm ratio alongside, which is the measure that separates them.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    model = replay["model"]
    slices = model.hidden_region_slices()
    weight = model.recurrent_weight_matrix().detach().cpu().numpy()

    def describe(block: np.ndarray, reference: float) -> dict[str, float]:
        magnitude = np.abs(block).ravel()
        total = float(magnitude.sum())
        squared = float((magnitude ** 2).sum())
        ordered = np.sort(magnitude)
        n = ordered.size
        gini = (
            float((2.0 * np.sum(np.arange(1, n + 1) * ordered)) / (n * total) - (n + 1) / n)
            if total > 0 else np.nan
        )
        return {
            "n_weights": int(n),
            "frobenius": float(np.linalg.norm(block)),
            "median_abs": float(np.median(magnitude)),
            "participation_ratio": float(total ** 2 / squared) if squared > 0 else 0.0,
            "gini": gini,
            "fraction_near_zero": float(np.mean(magnitude < 0.01 * reference)) if reference > 0 else np.nan,
        }

    reference = float(np.abs(weight).max())
    rows: list[dict[str, object]] = []
    for region in model.region_order:
        rows.append({"block": "within", "region": region,
                     **describe(weight[slices[region], slices[region]], reference)})
    for source in model.region_order:
        for target in model.region_order:
            if source == target:
                continue
            rows.append({"block": "cross", "region": f"{source}→{target}",
                         **describe(weight[slices[target], slices[source]], reference)})
    frame = pd.DataFrame(rows)
    frame["effective_fraction"] = frame["participation_ratio"] / frame["n_weights"]
    return frame.assign(run_dir=str(run_dir))


def summarize_sparsity(inventory: pd.DataFrame, *, device: str = "cpu") -> pd.DataFrame:
    """Achieved within-region sparsity per variant, with the cross-region blocks alongside."""
    rows: list[dict[str, object]] = []
    for _, run in inventory[inventory["complete"].astype(bool)].iterrows():
        measured = within_region_sparsity(run["run_dir"], device=device)
        within = measured[measured["block"] == "within"]
        cross = measured[measured["block"] == "cross"]
        cross_norm = float(cross["frobenius"].mean())
        rows.append({
            "label": run["label"],
            "seed": run["seed"],
            # The headline: how much within-region weight survives, relative to the
            # unpenalised cross-region blocks of the same model. Near 0 means the penalty
            # removed the blocks rather than sparsified them.
            "within_to_cross_norm": float(within["frobenius"].mean() / cross_norm) if cross_norm > 0 else np.nan,
            "within_effective_fraction": float(within["effective_fraction"].mean()),
            "within_gini": float(within["gini"].mean()),
            "within_fraction_near_zero": float(within["fraction_near_zero"].mean()),
            "cross_effective_fraction": float(cross["effective_fraction"].mean()),
            "cross_gini": float(cross["gini"].mean()),
        })
    return pd.DataFrame(rows)


def adequacy_table(
    fit: pd.DataFrame,
    *,
    baseline_label: str = "dense",
    tolerance_multiple: float = 2.0,
) -> pd.DataFrame:
    """Per-variant fit against an unconstrained baseline, with an adequacy flag.

    Adequacy is judged on the **worst condition**, not the mean. A constraint that leaves
    the average fit intact by trading interactive-face structure for object structure has
    not been tolerated by the data; it has been absorbed by the condition the objective
    already finds easy. The tolerance scales with the baseline's own seed-to-seed spread
    rather than being a fixed number, so it tightens as the fits become more reproducible.

    This is the *constraint* half of the selection rule. Among the variants it marks
    adequate, the one to keep is chosen on reproducibility and cost -- never on fit, which
    beyond the ceiling is measuring noise.
    """
    per_seed = (
        fit.groupby(["label", "seed", "condition"])["r2_vs_ceiling"].mean().reset_index()
    )
    worst = per_seed.groupby(["label", "seed"])["r2_vs_ceiling"].min().reset_index()
    summary = worst.groupby("label")["r2_vs_ceiling"].agg(
        worst_condition="mean", seed_spread="std"
    )
    summary["mean_all"] = fit.groupby("label")["r2_vs_ceiling"].mean()
    if baseline_label in summary.index:
        reference = float(summary.loc[baseline_label, "worst_condition"])
        spread = float(summary.loc[baseline_label, "seed_spread"])
    else:
        reference, spread = float("nan"), float("nan")
    tolerance = tolerance_multiple * spread
    summary["cost_vs_baseline"] = reference - summary["worst_condition"]
    summary["tolerance"] = tolerance
    summary["adequate"] = summary["cost_vs_baseline"] <= tolerance
    return summary.reset_index()


def lowest_adequate_rank(
    fit: pd.DataFrame,
    *,
    baseline_label: str = "dense",
    tolerance_multiple: float = 2.0,
) -> int | None:
    """The lowest constrained rank whose fit is indistinguishable from the dense baseline.

    "Indistinguishable" is measured against the baseline's own seed-to-seed spread rather
    than a fixed threshold, so the criterion scales with how reproducible the fits are.
    Returns ``None`` when no constrained rank qualifies.
    """
    baseline = fit[fit["label"] == baseline_label]
    if baseline.empty:
        return None
    spread = float(baseline.groupby("seed")["r2_vs_ceiling"].mean().std())
    reference = float(baseline["r2_vs_ceiling"].mean())
    ranks: list[tuple[int, float]] = []
    for label, block in fit.groupby("label"):
        if not str(label).startswith("rank"):
            continue
        ranks.append((int(str(label).replace("rank", "").split("__")[0]),
                      float(block["r2_vs_ceiling"].mean())))
    adequate = [r for r, value in sorted(ranks) if reference - value <= tolerance_multiple * spread]
    return adequate[0] if adequate else None


def resolve_task_root(task: str, cfg_path: str | Path = "configs/dataset.yaml") -> Path:
    """Output root for one sweep task."""
    return resolve_chapter_root(cfg_path, task=task)


__all__ = [
    "CONDITION_ORDER",
    "ModelVariant",
    "build_variant_settings",
    "LEARNING_RATE_LADDER",
    "convergence_table",
    "resolve_best_converged",
    "retry_variants_for_nonconverged",
    "count_trainable_parameters",
    "decompose_isolation_fit",
    "gallery_traces",
    "index_variant_runs",
    "load_histories",
    "PROVISIONAL_OPTIMIZER",
    "load_selected_protocol",
    "lowest_adequate_rank",
    "summarize_sparsity",
    "within_region_sparsity",
    "load_selected_protocol_or_provisional",
    "representative_traces",
    "resolve_task_root",
    "score_variant_fit",
    "variant_job_commands",
    "variant_run_dir",
]
