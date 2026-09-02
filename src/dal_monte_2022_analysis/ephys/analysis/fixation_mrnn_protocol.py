"""Training-protocol sweep for the rebuilt fixation mRNN analysis.

Everything the chapter rests on is being refitted, and the first thing to settle is how
to train at all. The existing runs show two symptoms of an unstable optimiser: a quarter
of seeds fail outright at identical settings, and the loss trajectories spike by one to
two orders of magnitude right through to the final iteration.

This module defines the sweep that fixes the recipe. It varies learning rate, gradient
clipping and learning-rate schedule at a fixed architecture, runs several seeds per
configuration, and scores configurations on **reliability across seeds** rather than on
the best single fit -- which is the property the downstream ensembles actually need.

Rebuilt runs live under a ``chapter/`` root, kept separate from the legacy ``scratch/``
tree so the two can never be pooled by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_training import (
    FixationMRNNRunSettings,
    load_fixation_mrnn_config,
    settings_from_config,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir

DEFAULT_MODELING_SUBDIR = "ephys/modeling/fixation_mrnn"

#: Architecture held fixed across the whole protocol sweep. These are the settings the
#: legacy work converged on and that the downstream tasks will vary one at a time; the
#: point of this sweep is the optimiser, so nothing structural moves here.
PROTOCOL_ARCHITECTURE: dict[str, object] = {
    "target_mode": "region_pcs",
    "pca_n_components": 42,
    "hidden_units": 50,
    "recurrent_connectivity": "full",
    "recurrent_bottleneck_dim": 3,
    "temporal_basis_count": 0,
    "temporal_derivative_loss_scale": 1.0,
    "temporal_curvature_loss_scale": 0.5,
    "l1_weight_scale": 0.01,
    "train_initial_state": True,
}


@dataclass(frozen=True)
class ProtocolConfig:
    """One training configuration in the sweep.

    Carries the optimiser axes plus the two architecture choices that plausibly drive
    the instability: the activation (``tanh`` is bounded and zero-centred, matching
    zero-centred PC targets; ``softplus`` is positive and unbounded, so an overshoot has
    no ceiling) and the initial spectral radius (above ~1 the Jacobian product over 100
    recurrent steps is what sharpens the landscape).
    """

    label: str
    lr: float
    gradient_clip_norm: float | None
    lr_schedule: str = "constant"
    lr_warmup_iterations: int = 0
    lr_min_factor: float = 0.01
    activation: str = "tanh"
    spectral_radius: float = 1.1

    def overrides(self) -> dict[str, object]:
        return {
            "lr": float(self.lr),
            "gradient_clip_norm": None if self.gradient_clip_norm is None else float(self.gradient_clip_norm),
            "lr_schedule": str(self.lr_schedule),
            "lr_warmup_iterations": int(self.lr_warmup_iterations),
            "lr_min_factor": float(self.lr_min_factor),
            "activation": str(self.activation),
            "spectral_radius": float(self.spectral_radius),
        }


def _clip_token(value: float | None) -> str:
    return "none" if value is None else f"{value:g}".replace(".", "p")


def _number_token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _config(
    *,
    lr: float,
    schedule: str,
    clip: float | None,
    activation: str,
    spectral_radius: float,
) -> ProtocolConfig:
    label = "_".join(
        [
            f"lr{_number_token(lr)}",
            schedule,
            f"clip{_clip_token(clip)}",
            activation,
            f"sr{_number_token(spectral_radius)}",
        ]
    )
    return ProtocolConfig(
        label=label,
        lr=float(lr),
        gradient_clip_norm=clip,
        lr_schedule=str(schedule),
        activation=str(activation),
        spectral_radius=float(spectral_radius),
    )


def protocol_sweep_grid(
    *,
    learning_rates: Sequence[float] = (3e-4, 1e-3, 3e-3, 1e-2),
    activations: Sequence[str] = ("tanh", "softplus"),
    spectral_radii: Sequence[float] = (0.9, 1.1),
    reference_activation: str = "tanh",
    reference_spectral_radius: float = 1.1,
    binding_clip: float = 0.05,
) -> list[ProtocolConfig]:
    """The staged screening design: one factorial plus two control arms.

    A full product over every axis would be 64 configurations, most of them spent on
    questions the existing runs have already answered. The design instead is:

    **Main arm** -- peak learning rate x activation x initial spectral radius, all on a
    cosine schedule with no clipping. Cosine is fixed here rather than swept because a
    completed seed-matched block already shows it drives final/best to exactly 1.000 on
    every seed; what is open is the *peak* rate it can safely carry, since decaying the
    rate also costs fit.

    **Schedule control** -- the same learning rates on a constant schedule at the
    reference architecture, so the cosine claim is demonstrated inside this sweep rather
    than imported from the pilot.

    **Clipping control** -- the same learning rates with a clip norm of 0.05. The value
    matters: measured gradient norms on this problem have median 0.024 and p90 0.078, so
    the historical threshold of 1.0 bound on 0.1% of steps and was inert. 0.05 is the
    first threshold that actually binds.
    """
    configs: list[ProtocolConfig] = []
    for lr in learning_rates:
        for activation in activations:
            for spectral_radius in spectral_radii:
                configs.append(
                    _config(lr=lr, schedule="cosine", clip=None,
                            activation=activation, spectral_radius=spectral_radius)
                )
    for lr in learning_rates:
        configs.append(
            _config(lr=lr, schedule="constant", clip=None,
                    activation=reference_activation, spectral_radius=reference_spectral_radius)
        )
        configs.append(
            _config(lr=lr, schedule="cosine", clip=float(binding_clip),
                    activation=reference_activation, spectral_radius=reference_spectral_radius)
        )
    return configs


def protocol_seeds(*, n_seeds: int = 5, base_seed: int = 20260902) -> list[int]:
    """A fixed seed list, shared by every configuration in the sweep.

    Sharing seeds across configurations makes the comparison paired: a configuration
    that fails on seed 3 can be compared against another configuration on the same
    seed 3, rather than against a different random draw.
    """
    rng = np.random.default_rng(int(base_seed))
    return [int(v) for v in rng.integers(1, 2**31 - 1, size=int(n_seeds))]


def resolve_chapter_root(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    task: str,
    modeling_subdir: str = DEFAULT_MODELING_SUBDIR,
) -> Path:
    """Output root for one rebuild task, kept apart from the legacy ``scratch/`` tree."""
    cfg = load_config(cfg_path)
    path = build_analysis_output_dir(cfg, modeling_subdir) / "chapter" / str(task)
    path.mkdir(parents=True, exist_ok=True)
    return path


def protocol_run_dir(root: str | Path, config: ProtocolConfig, seed: int) -> Path:
    """Run directory for one (configuration, seed) cell of the sweep."""
    return Path(root) / config.label / f"seed={int(seed)}"


def build_protocol_settings(
    config: ProtocolConfig,
    *,
    mrnn_cfg_path: str | Path,
    epochs: int,
    seed: int,
    architecture: Mapping[str, object] = PROTOCOL_ARCHITECTURE,
) -> FixationMRNNRunSettings:
    """Settings for one sweep cell: fixed architecture plus this configuration's optimiser."""
    base = settings_from_config(load_fixation_mrnn_config(mrnn_cfg_path))
    return replace(
        base,
        **dict(architecture),
        **config.overrides(),
        epochs=int(epochs),
        seed=int(seed),
        # The sweep measures how often a configuration fails, so the guardrails must not
        # rescue it. Divergence is recorded, never retried, and never reseeded.
        divergence_loss_threshold=None,
    )


def write_protocol_job(
    config: ProtocolConfig,
    *,
    root: str | Path,
    mrnn_cfg_path: str | Path,
    epochs: int,
    seed: int,
    architecture: Mapping[str, object] = PROTOCOL_ARCHITECTURE,
) -> tuple[Path, Path]:
    """Write the per-cell config YAML and return ``(config_path, run_dir)``.

    Each cell carries its own frozen config file rather than a pile of CLI flags, so the
    run directory is a self-contained record of what was trained.
    """
    run_dir = protocol_run_dir(root, config, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = build_protocol_settings(
        config, mrnn_cfg_path=mrnn_cfg_path, epochs=epochs, seed=seed, architecture=architecture
    )
    payload = {key: value for key, value in asdict(settings).items()}
    payload["region_order"] = list(payload["region_order"])
    payload["condition_order"] = list(payload["condition_order"])
    payload["scratch_id"] = run_dir.name
    config_path = run_dir / "run_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)
    return config_path, run_dir


def protocol_job_commands(
    configs: Sequence[ProtocolConfig],
    seeds: Sequence[int],
    *,
    root: str | Path,
    repo_root: str | Path,
    mrnn_cfg_path: str | Path,
    epochs: int,
    conda_env: str = "gaze_processing",
    architecture: Mapping[str, object] = PROTOCOL_ARCHITECTURE,
    skip_existing: bool = True,
) -> tuple[list[str], list[Path]]:
    """One shell command per missing sweep cell, plus the run directories they write to.

    The dedicated ``train_fixation_mrnn_into_run_dir.py`` entry point is used rather than
    the general training CLI for two reasons: it writes where it is told rather than into
    the legacy ``scratch/`` tree, and it never retries a diverged run under a new seed,
    which would quietly convert the failures this sweep exists to count into successes
    under a different initialization.
    """
    repo_root = Path(repo_root)
    commands: list[str] = []
    run_dirs: list[Path] = []
    for config in configs:
        for seed in seeds:
            run_dir = protocol_run_dir(root, config, seed)
            run_dirs.append(run_dir)
            if skip_existing and (run_dir / "checkpoint_best.pth").exists():
                continue
            config_path, _ = write_protocol_job(
                config,
                root=root,
                mrnn_cfg_path=mrnn_cfg_path,
                epochs=epochs,
                seed=seed,
                architecture=architecture,
            )
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


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def count_loss_transients(losses: np.ndarray, *, jump_threshold_log10: float = 0.5) -> int:
    """Number of upward jumps in the loss larger than ``jump_threshold_log10`` decades.

    A converging optimiser should not repeatedly triple its own loss. Counting the spikes
    separates "ended at a good value" from "ended at a good value by luck", which is the
    distinction the final-versus-best gap is a symptom of.
    """
    finite = np.isfinite(losses) & (losses > 0)
    safe = np.where(finite, losses, np.nan)
    log_loss = np.log10(np.clip(safe, 1e-12, None))
    delta = np.diff(log_loss)
    return int(np.sum(np.nan_to_num(delta, nan=0.0) > float(jump_threshold_log10)))


def index_protocol_runs(root: str | Path, configs: Sequence[ProtocolConfig], seeds: Sequence[int]) -> pd.DataFrame:
    """One row per sweep cell: whether it completed, failed, or has not run."""
    rows: list[dict[str, object]] = []
    for config in configs:
        for seed in seeds:
            run_dir = protocol_run_dir(root, config, seed)
            complete = (run_dir / "checkpoint_best.pth").exists()
            failed = (run_dir / "training_failed.json").exists()
            rows.append(
                {
                    **asdict(config),
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "complete": bool(complete),
                    "diverged": bool(failed and not complete),
                    "pending": not complete and not failed,
                }
            )
    return pd.DataFrame(rows)


def score_protocol_runs(inventory: pd.DataFrame) -> pd.DataFrame:
    """Per-cell training quality: best loss, drift after the best, and spike count."""
    rows: list[dict[str, object]] = []
    for _, run in inventory.iterrows():
        history_path = Path(run["run_dir"]) / "history.csv"
        record = {key: run[key] for key in ("label", "lr", "gradient_clip_norm", "lr_schedule", "seed", "diverged")}
        if not history_path.exists():
            rows.append({**record, "best_loss": np.nan, "final_over_best": np.nan,
                         "n_transients": np.nan, "iterations": 0})
            continue
        history = pd.read_csv(history_path)
        losses = history["loss"].to_numpy(dtype=float)
        finite = losses[np.isfinite(losses)]
        best = float(finite.min()) if finite.size else np.nan
        final = float(losses[-1]) if losses.size else np.nan
        rows.append(
            {
                **record,
                "best_loss": best,
                "final_over_best": float(final / best) if best and np.isfinite(best) and best > 0 else np.nan,
                "n_transients": count_loss_transients(losses),
                "iterations": int(len(history)),
            }
        )
    return pd.DataFrame(rows)


def rank_protocols(scores: pd.DataFrame, *, n_seeds: int | None = None) -> pd.DataFrame:
    """Collapse per-seed scores to one row per configuration, ordered by reliability.

    The ordering is deliberately not "lowest median loss". A configuration that fits
    beautifully on three seeds and diverges on two is useless for an ensemble, so
    failures are the first sort key, the spread of best loss across seeds is the second,
    and only then the typical loss itself.
    """
    rows: list[dict[str, object]] = []
    for label, block in scores.groupby("label", sort=False):
        total = int(n_seeds or len(block))
        usable = block[np.isfinite(block["best_loss"]) & ~block["diverged"].astype(bool)]
        best = usable["best_loss"].to_numpy(dtype=float)
        rows.append(
            {
                "label": label,
                "lr": float(block["lr"].iloc[0]),
                "gradient_clip_norm": block["gradient_clip_norm"].iloc[0],
                "lr_schedule": block["lr_schedule"].iloc[0],
                "n_seeds": total,
                "n_failed": int(total - len(usable)),
                "failure_rate": float((total - len(usable)) / total) if total else np.nan,
                "median_best_loss": float(np.median(best)) if best.size else np.nan,
                "iqr_best_loss": float(np.subtract(*np.percentile(best, [75, 25]))) if best.size else np.nan,
                "spread_ratio": float(best.max() / best.min()) if best.size and best.min() > 0 else np.nan,
                "median_final_over_best": float(usable["final_over_best"].median()) if len(usable) else np.nan,
                "median_transients": float(usable["n_transients"].median()) if len(usable) else np.nan,
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["failure_rate", "spread_ratio", "median_best_loss"], ascending=True
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def write_selected_protocol(
    ranked: pd.DataFrame,
    configs: Sequence[ProtocolConfig],
    *,
    path: str | Path,
    epochs: int,
    architecture: Mapping[str, object] = PROTOCOL_ARCHITECTURE,
) -> Path:
    """Freeze the winning configuration to a YAML the downstream tasks import.

    Writing the choice to a file rather than restating it in each notebook is what stops
    the rebuild from drifting the way the legacy tree did, where every family ended up
    with its own learning rate and iteration count.
    """
    winner_label = str(ranked.iloc[0]["label"])
    winner = next(config for config in configs if config.label == winner_label)
    payload = {
        "selected_label": winner_label,
        "selection_rule": "fewest seed failures, then smallest across-seed spread, then lowest median loss",
        "epochs": int(epochs),
        "optimizer": winner.overrides(),
        "architecture": dict(architecture),
        "sweep_summary": ranked.head(3).to_dict(orient="records"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return path


# --------------------------------------------------------------------------------------
# Diagnosing the instability
# --------------------------------------------------------------------------------------


def objective_is_deterministic(settings: FixationMRNNRunSettings) -> dict[str, object]:
    """Report whether the training objective carries any stochasticity.

    This decides what a loss spike can mean. With three conditions trained full-batch and
    no input or activation noise, the loss is a deterministic function of the parameters,
    so an upward jump is the optimizer stepping uphill on a fixed landscape -- not
    sampling variance. That rules out "just noise" and rules in step-size remedies.
    """
    return {
        "batch_size": len(settings.condition_order),
        "full_batch": True,
        "input_noise": float(settings.inp_noise),
        "activation_noise": float(settings.act_noise),
        "deterministic": float(settings.inp_noise) == 0.0 and float(settings.act_noise) == 0.0,
    }


def measure_gradient_norms(
    settings: FixationMRNNRunSettings,
    *,
    run_dir: str | Path,
    iterations: int = 2000,
    seed: int = 31,
) -> pd.DataFrame:
    """Train briefly with clipping effectively disabled, recording the gradient norm.

    ``torch.nn.utils.clip_grad_norm_`` returns the pre-clip norm, so setting the
    threshold far above any plausible value turns it into a measurement. The point is to
    find out whether the clip threshold in use ever binds: Adam scales each parameter by
    its own second moment, so a global norm clip only does anything when the norm
    actually exceeds the threshold.
    """
    import torch

    from dal_monte_2022_analysis.ephys.modeling import fixation_mrnn_training as training

    norms: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, **kwargs):
        value = original(parameters, max_norm, **kwargs)
        norms.append(float(value))
        return value

    torch.nn.utils.clip_grad_norm_ = recording_clip
    try:
        probe = replace(settings, epochs=int(iterations), gradient_clip_norm=1e9)
        result = training.train_one_initialization(probe, run_dir=Path(run_dir), seed=int(seed), overwrite=True)
    finally:
        torch.nn.utils.clip_grad_norm_ = original

    history = result["history"]
    return pd.DataFrame(
        {
            "iteration": history["iteration"].to_numpy()[: len(norms)],
            "loss": history["loss"].to_numpy()[: len(norms)],
            "gradient_norm": np.asarray(norms, dtype=float),
        }
    )


def clip_threshold_binding_rate(gradient_norms: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    """Fraction of steps each candidate clip threshold would actually constrain."""
    values = gradient_norms["gradient_norm"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "clip_norm": float(threshold),
                "fraction_of_steps_clipped": float(np.mean(values > float(threshold))),
            }
            for threshold in thresholds
        ]
    )


def spike_positions(losses: np.ndarray, *, jump_threshold_log10: float = 0.5) -> np.ndarray:
    """Where in a run its upward loss jumps occur, as a fraction of total iterations."""
    losses = np.asarray(losses, dtype=float)
    safe = np.where(np.isfinite(losses) & (losses > 0), losses, np.nan)
    delta = np.diff(np.log10(np.clip(safe, 1e-12, None)))
    indices = np.flatnonzero(np.nan_to_num(delta, nan=0.0) > float(jump_threshold_log10))
    return indices / max(len(losses), 1)


def survey_legacy_instability(
    scratch_root: str | Path,
    *,
    min_iterations: int = 5000,
) -> pd.DataFrame:
    """Spike rate and final-versus-best gap for every legacy run, with its settings.

    The legacy tree is an unplanned natural experiment: 271 runs spanning six learning
    rates, two activations and two clip settings. It cannot separate confounded axes --
    which is what the sweep is for -- but it does show which axis moves the instability
    at all, and it is the evidence that picks the sweep's ranges.
    """
    import torch

    root = Path(scratch_root)
    rows: list[dict[str, object]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        candidates = [entry] if (entry / "history.csv").exists() else sorted(entry.glob("init=*"))
        for run in candidates:
            history_path = run / "history.csv"
            checkpoint_path = run / "checkpoint_final.pth"
            if not history_path.exists() or not checkpoint_path.exists():
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            stored = checkpoint.get("settings")
            settings = dict(stored) if isinstance(stored, Mapping) else dict(vars(stored))
            losses = pd.read_csv(history_path)["loss"].to_numpy(dtype=float)
            finite = losses[np.isfinite(losses)]
            if len(losses) < int(min_iterations) or finite.size == 0:
                continue
            positions = spike_positions(losses)
            rows.append(
                {
                    "run": str(run.relative_to(root)),
                    "iterations": int(len(losses)),
                    "lr": settings.get("lr"),
                    "activation": settings.get("activation"),
                    "spectral_radius": settings.get("spectral_radius"),
                    "gradient_clip_norm": settings.get("gradient_clip_norm"),
                    "n_spikes": int(positions.size),
                    "spikes_per_10k": float(1e4 * positions.size / len(losses)),
                    "late_spikes": int(np.sum(positions > 0.5)),
                    "best_loss": float(finite.min()),
                    "final_over_best": float(losses[-1] / finite.min()) if finite.min() > 0 else np.nan,
                    "final_is_best": bool(finite.min() > 0 and losses[-1] / finite.min() < 1.01),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# The pass bar
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvergenceBar:
    """What a configuration has to achieve to be usable downstream.

    Stated before the sweep runs so the choice cannot be reverse-engineered from the
    result. The bar is about *convergence*, not cosmetics: a spiky trajectory means the
    optimizer never settled, so its best iterate is the luckiest point on a jittery walk.
    Different seeds then land at different lucky points -- which is a candidate
    explanation for the seed-to-seed disagreement the whole analysis turns on.
    """

    #: No upward jump larger than this many decades, anywhere in the scored window.
    jump_threshold_log10: float = 0.5
    #: Fraction of training the bar is applied over, measured from the end.
    late_fraction: float = 0.5
    #: How far the final iterate may sit above the best one.
    max_final_over_best: float = 1.01
    #: Every seed must pass; a configuration that works on most seeds is not usable.
    require_all_seeds: bool = True


def evaluate_convergence(losses: np.ndarray, bar: ConvergenceBar = ConvergenceBar()) -> dict[str, object]:
    """Score one run against the bar."""
    losses = np.asarray(losses, dtype=float)
    finite = losses[np.isfinite(losses)]
    if finite.size == 0:
        return {"late_spikes": np.nan, "final_over_best": np.nan, "passes": False}
    start = int(len(losses) * (1.0 - float(bar.late_fraction)))
    late = spike_positions(losses, jump_threshold_log10=bar.jump_threshold_log10)
    late_count = int(np.sum(late > (1.0 - float(bar.late_fraction))))
    final_over_best = float(losses[-1] / finite.min()) if finite.min() > 0 else np.inf
    return {
        "scored_from_iteration": start,
        "late_spikes": late_count,
        "final_over_best": final_over_best,
        "passes": bool(late_count == 0 and final_over_best <= float(bar.max_final_over_best)),
    }


def configurations_meeting_bar(
    inventory: pd.DataFrame,
    bar: ConvergenceBar = ConvergenceBar(),
) -> pd.DataFrame:
    """Apply the bar to every seed, then to every configuration."""
    rows: list[dict[str, object]] = []
    for _, run in inventory.iterrows():
        history_path = Path(run["run_dir"]) / "history.csv"
        record = {
            "label": run["label"],
            "lr": run["lr"],
            "lr_schedule": run["lr_schedule"],
            "gradient_clip_norm": run["gradient_clip_norm"],
            "activation": run.get("activation"),
            "spectral_radius": run.get("spectral_radius"),
            "seed": run["seed"],
        }
        if not history_path.exists():
            rows.append({**record, "late_spikes": np.nan, "final_over_best": np.nan,
                         "best_loss": np.nan, "passes": False, "complete": False})
            continue
        losses = pd.read_csv(history_path)["loss"].to_numpy(dtype=float)
        finite = losses[np.isfinite(losses)]
        rows.append({
            **record,
            **evaluate_convergence(losses, bar),
            "best_loss": float(finite.min()) if finite.size else np.nan,
            "complete": True,
        })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed
    aggregated = (
        per_seed.groupby("label", sort=False)
        .agg(
            lr=("lr", "first"),
            lr_schedule=("lr_schedule", "first"),
            gradient_clip_norm=("gradient_clip_norm", "first"),
            activation=("activation", "first"),
            spectral_radius=("spectral_radius", "first"),
            n_seeds=("seed", "size"),
            n_complete=("complete", "sum"),
            n_passing=("passes", "sum"),
            max_late_spikes=("late_spikes", "max"),
            max_final_over_best=("final_over_best", "max"),
            median_best_loss=("best_loss", "median"),
        )
        .reset_index()
    )
    aggregated["passes_bar"] = (
        (aggregated["n_passing"] == aggregated["n_seeds"])
        if bar.require_all_seeds
        else (aggregated["n_passing"] > 0)
    ) & (aggregated["n_complete"] == aggregated["n_seeds"])
    return aggregated.sort_values(["passes_bar", "median_best_loss"], ascending=[False, True]).reset_index(drop=True)


__all__ = [
    "survey_legacy_instability",
    "spike_positions",
    "objective_is_deterministic",
    "measure_gradient_norms",
    "evaluate_convergence",
    "configurations_meeting_bar",
    "clip_threshold_binding_rate",
    "ConvergenceBar",
    "PROTOCOL_ARCHITECTURE",
    "ProtocolConfig",
    "build_protocol_settings",
    "count_loss_transients",
    "index_protocol_runs",
    "protocol_job_commands",
    "protocol_run_dir",
    "protocol_seeds",
    "protocol_sweep_grid",
    "rank_protocols",
    "resolve_chapter_root",
    "score_protocol_runs",
    "write_protocol_job",
    "write_selected_protocol",
]
