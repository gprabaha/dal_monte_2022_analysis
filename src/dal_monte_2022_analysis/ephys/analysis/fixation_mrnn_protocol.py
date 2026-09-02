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
    "activation": "tanh",
    "spectral_radius": 1.1,
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
    """One optimiser configuration in the sweep."""

    label: str
    lr: float
    gradient_clip_norm: float | None
    lr_schedule: str = "constant"
    lr_warmup_iterations: int = 0
    lr_min_factor: float = 0.01

    def overrides(self) -> dict[str, object]:
        return {
            "lr": float(self.lr),
            "gradient_clip_norm": None if self.gradient_clip_norm is None else float(self.gradient_clip_norm),
            "lr_schedule": str(self.lr_schedule),
            "lr_warmup_iterations": int(self.lr_warmup_iterations),
            "lr_min_factor": float(self.lr_min_factor),
        }


def _clip_token(value: float | None) -> str:
    return "none" if value is None else f"{value:g}".replace(".", "p")


def protocol_sweep_grid(
    *,
    learning_rates: Sequence[float] = (3e-4, 1e-3, 3e-3),
    gradient_clips: Sequence[float | None] = (None, 1.0),
    schedules: Sequence[str] = ("constant", "cosine"),
    warmup_iterations: int = 0,
) -> list[ProtocolConfig]:
    """The full optimiser grid, as a list of labelled configurations.

    Defaults give 3 x 2 x 2 = 12 configurations. The learning rates bracket the working
    band the legacy runs suggest (1e-3 fits well, 1e-4 underfits, 2e-3 and above
    sometimes diverges), so the sweep spans a regime where failures are expected --
    a sweep that never fails would tell us nothing about reliability.
    """
    configs: list[ProtocolConfig] = []
    for lr in learning_rates:
        for clip in gradient_clips:
            for schedule in schedules:
                label = f"lr{lr:g}".replace(".", "p").replace("-", "m")
                label = f"{label}_clip{_clip_token(clip)}_{schedule}"
                configs.append(
                    ProtocolConfig(
                        label=label,
                        lr=float(lr),
                        gradient_clip_norm=clip,
                        lr_schedule=str(schedule),
                        lr_warmup_iterations=int(warmup_iterations),
                    )
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


__all__ = [
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
