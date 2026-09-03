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
from typing import Mapping, Sequence

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


def load_selected_protocol(path: str | Path) -> dict[str, object]:
    """The training recipe task 00 froze, which every sweep here inherits."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
) -> tuple[list[str], list[Path]]:
    """One shell command per missing (variant, seed) cell, each with a frozen config."""
    repo_root = Path(repo_root)
    commands: list[str] = []
    run_dirs: list[Path] = []
    for variant in variants:
        for seed in seeds:
            run_dir = variant_run_dir(root, variant, seed)
            run_dirs.append(run_dir)
            if skip_existing and (run_dir / "checkpoint_best.pth").exists():
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
) -> pd.DataFrame:
    """One row per (variant, seed) cell with its completion state."""
    rows: list[dict[str, object]] = []
    for variant in variants:
        for seed in seeds:
            run_dir = variant_run_dir(root, variant, seed)
            complete = (run_dir / "checkpoint_best.pth").exists()
            failed = (run_dir / "training_failed.json").exists()
            rows.append(
                {
                    **variant.describe(),
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "complete": bool(complete),
                    "diverged": bool(failed and not complete),
                    "pending": not complete and not failed,
                }
            )
    return pd.DataFrame(rows)


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


def resolve_task_root(task: str, cfg_path: str | Path = "configs/dataset.yaml") -> Path:
    """Output root for one sweep task."""
    return resolve_chapter_root(cfg_path, task=task)


__all__ = [
    "CONDITION_ORDER",
    "ModelVariant",
    "build_variant_settings",
    "convergence_table",
    "count_trainable_parameters",
    "gallery_traces",
    "index_variant_runs",
    "load_histories",
    "load_selected_protocol",
    "representative_traces",
    "resolve_task_root",
    "score_variant_fit",
    "variant_job_commands",
    "variant_run_dir",
]
