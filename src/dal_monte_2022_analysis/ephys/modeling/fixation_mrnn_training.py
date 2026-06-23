"""Minimal fixation mRNN training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import yaml

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_model import (
    FixationMRNNModel,
    build_model_spec,
    normalize_recurrent_connectivity,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_targets import (
    CONDITION_ORDER,
    FixationMRNNTargets,
    build_fixation_mrnn_targets,
    normalize_target_mode,
    serialize_pca_metadata,
    summarize_targets,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


class TrainingDivergedError(RuntimeError):
    """Raised when a training run exceeds the configured divergence guardrails."""


@dataclass
class FixationMRNNRunSettings:
    """Settings for target creation, model construction, and training."""

    dataset_cfg_path: str = "configs/dataset.yaml"
    input_subdir: str = "ephys/psth/fixation_psth_averages"
    dataframe_filename: str = "fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl"
    timeline_filename: str = "fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl"
    output_subdir: str = "ephys/modeling/fixation_mrnn"
    region_order: tuple[str, ...] = ("ofc", "bla", "dmpfc", "accg")
    condition_order: tuple[str, ...] = CONDITION_ORDER
    target_mode: str = "raw_fr"
    normalize_targets: bool = True
    normalization_stabilizer: float = 5.0
    pca_variance_threshold: float = 0.95
    pca_n_components: int | None = None
    temporal_basis_count: int = 0
    hidden_units: int | dict[str, int] = 50
    activation: str = "softplus"
    spectral_radius: float | None = 1.2
    rec_constrained: bool = False
    inp_constrained: bool = False
    recurrent_connectivity: str = "full"
    batch_first: bool = True
    inp_noise: float = 0.0
    act_noise: float = 0.0
    epochs: int = 10_000
    lr: float = 1e-3
    loss_fn: str = "mse"
    temporal_derivative_loss_scale: float = 1.0
    temporal_curvature_loss_scale: float = 1.0
    correlation_loss_scale: float = 0.0
    variance_loss_scale: float = 0.0
    fr_reconstruction_loss_scale: float = 0.0
    fr_temporal_derivative_loss_scale: float = 0.0
    fr_temporal_curvature_loss_scale: float = 0.0
    pre_fixation_loss_weight: float = 1.0
    post_fixation_loss_weight: float = 1.0
    l1_weight_scale: float = 0.0
    l1_rate_scale: float = 0.0
    l2_weight_scale: float = 0.0
    l2_rate_scale: float = 0.0
    gradient_clip_norm: float | None = None
    divergence_loss_threshold: float | None = None
    divergence_patience: int = 100
    divergence_min_iteration: int = 100
    train_initial_state: bool = True
    initial_state_scale: float = 0.01
    seed: int = 123456
    initialization_mode: str = "single"
    n_initializations: int = 100
    overwrite_seed_plan: bool = False
    seed_plan_filename: str = "seed_plan.json"
    device: str = "auto"


def load_fixation_mrnn_config(path: str | Path = "configs/ephys_fixation_mrnn.yaml") -> dict[str, Any]:
    """Load the fixation mRNN YAML config."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def settings_from_config(
    cfg: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> FixationMRNNRunSettings:
    """Create settings from config plus optional overrides."""
    data = dict(cfg)
    if "canonical_region_order" in data and "region_order" not in data:
        data["region_order"] = data.pop("canonical_region_order")
    for key in ("region_order", "condition_order"):
        if key in data:
            data[key] = tuple(data[key])
    for key, value in dict(overrides or {}).items():
        if value is not None:
            data[key] = value
    allowed = set(FixationMRNNRunSettings.__dataclass_fields__)
    return FixationMRNNRunSettings(**{key: value for key, value in data.items() if key in allowed})


def resolve_device(device: str) -> str:
    """Resolve auto/cuda/cpu."""
    token = str(device).strip().lower()
    if token == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if token.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return token


def resolve_fixation_mrnn_output_root(settings: FixationMRNNRunSettings) -> Path:
    """Resolve output root from dataset config."""
    cfg = load_config(settings.dataset_cfg_path)
    return build_analysis_output_dir(cfg, settings.output_subdir)


def make_targets(settings: FixationMRNNRunSettings) -> FixationMRNNTargets:
    """Build targets using settings."""
    return build_fixation_mrnn_targets(
        settings.dataset_cfg_path,
        input_subdir=settings.input_subdir,
        dataframe_filename=settings.dataframe_filename,
        timeline_filename=settings.timeline_filename,
        region_order=tuple(settings.region_order),
        condition_order=tuple(settings.condition_order),
        normalize_targets=bool(settings.normalize_targets),
        normalization_stabilizer=float(settings.normalization_stabilizer),
        pca_variance_threshold=float(settings.pca_variance_threshold),
        pca_n_components=settings.pca_n_components,
        temporal_basis_count=int(settings.temporal_basis_count),
    )


def _seed_plan_path(directory: str | Path, settings: FixationMRNNRunSettings) -> Path:
    return Path(directory) / str(settings.seed_plan_filename)


def load_or_create_seed_plan(
    directory: str | Path,
    settings: FixationMRNNRunSettings,
    *,
    n_seeds: int,
    overwrite: bool = False,
) -> list[int]:
    """Load persistent seeds or create them once."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = _seed_plan_path(directory, settings)
    if path.exists() and not overwrite:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        seeds = [int(seed) for seed in payload["seeds"]]
        if len(seeds) < int(n_seeds):
            raise ValueError(f"{path} has {len(seeds)} seeds, but {n_seeds} were requested.")
        return seeds[: int(n_seeds)]
    rng = np.random.default_rng(int(settings.seed))
    seeds = [
        int(seed)
        for seed in rng.integers(1, np.iinfo(np.int32).max, size=int(n_seeds), dtype=np.int64)
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "base_seed": int(settings.seed),
                "seeds": seeds,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return seeds


def _target_tensors(
    targets: FixationMRNNTargets,
    *,
    target_mode: str,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    arrays = targets.targets_for_mode(target_mode)
    by_region = {
        region: torch.as_tensor(arrays[region], dtype=torch.float32, device=device)
        for region in targets.region_order
    }
    return by_region, torch.cat([by_region[region] for region in targets.region_order], dim=-1)


def _loss_fn(name: str) -> nn.Module:
    token = str(name).strip().lower()
    if token == "mse":
        return nn.MSELoss()
    if token == "mae":
        return nn.L1Loss()
    raise ValueError("loss_fn must be 'mse' or 'mae'.")


def _elementwise_loss(name: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    token = str(name).strip().lower()
    if token == "mse":
        return (prediction - target) ** 2
    if token == "mae":
        return torch.abs(prediction - target)
    raise ValueError("loss_fn must be 'mse' or 'mae'.")


def _weighted_loss(
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = _elementwise_loss(name, prediction, target)
    if weights is not None:
        loss = loss * weights
    return torch.mean(loss)


def _time_weights(
    timeline_s: Sequence[float],
    *,
    pre_fixation_weight: float,
    post_fixation_weight: float,
    device: str,
) -> torch.Tensor:
    weights = np.where(
        np.asarray(timeline_s, dtype=float) < 0.0,
        float(pre_fixation_weight),
        float(post_fixation_weight),
    ).astype(np.float32)
    mean_weight = float(np.mean(weights))
    if mean_weight > 0:
        weights = weights / mean_weight
    return torch.as_tensor(weights, dtype=torch.float32, device=device).reshape(1, -1, 1)


def _temporal_difference_loss(
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    order: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape[1] <= int(order):
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    pred_diff = prediction
    target_diff = target
    for _ in range(int(order)):
        pred_diff = pred_diff[:, 1:, :] - pred_diff[:, :-1, :]
        target_diff = target_diff[:, 1:, :] - target_diff[:, :-1, :]
    diff_weights = weights[:, int(order) :, :] if weights is not None else None
    return _weighted_loss(name, pred_diff, target_diff, weights=diff_weights)


def _l1(parameters: Sequence[torch.Tensor], scale: float) -> torch.Tensor:
    if not parameters:
        return torch.zeros(())
    penalty = torch.zeros((), device=parameters[0].device)
    for param in parameters:
        penalty = penalty + torch.mean(torch.abs(param))
    return penalty * float(scale)


def _l2(parameters: Sequence[torch.Tensor], scale: float) -> torch.Tensor:
    if not parameters:
        return torch.zeros(())
    penalty = torch.zeros((), device=parameters[0].device)
    for param in parameters:
        penalty = penalty + torch.mean(param**2)
    return penalty * float(scale)


def _temporal_correlation_loss(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    pred_centered = prediction - torch.mean(prediction, dim=1, keepdim=True)
    target_centered = target - torch.mean(target, dim=1, keepdim=True)
    numerator = torch.sum(pred_centered * target_centered, dim=1)
    denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1) + eps)
    correlation = numerator / torch.clamp(denominator, min=eps)
    return torch.mean(1.0 - correlation)


def _variance_loss(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    pred_var = torch.var(prediction, dim=(0, 1), unbiased=False)
    target_var = torch.var(target, dim=(0, 1), unbiased=False)
    return torch.mean((torch.log(pred_var + eps) - torch.log(target_var + eps)) ** 2)


def _pc_to_firing_rate_tensor(
    pcs: torch.Tensor,
    *,
    components: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    n_components = min(int(pcs.shape[-1]), int(components.shape[0]))
    return pcs[..., :n_components] @ components[:n_components] + mean


def _pc_backprojection_tensors(
    targets: FixationMRNNTargets,
    *,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    raw_from_pcs = targets.pc_reconstructed_raw_by_region()
    target_fr = {
        region: torch.as_tensor(raw_from_pcs[region], dtype=torch.float32, device=device)
        for region in targets.region_order
    }
    components = {
        region: torch.as_tensor(targets.pca_by_region[region].components, dtype=torch.float32, device=device)
        for region in targets.region_order
    }
    means = {
        region: torch.as_tensor(targets.pca_by_region[region].mean, dtype=torch.float32, device=device)
        for region in targets.region_order
    }
    return target_fr, components, means


def _firing_rate_reconstruction_loss(
    settings: FixationMRNNRunSettings,
    output_by_region: Mapping[str, torch.Tensor],
    target_fr_by_region: Mapping[str, torch.Tensor],
    components_by_region: Mapping[str, torch.Tensor],
    means_by_region: Mapping[str, torch.Tensor],
    *,
    region_order: Sequence[str],
    time_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_losses = []
    derivative_losses = []
    curvature_losses = []
    for region in region_order:
        pred_fr = _pc_to_firing_rate_tensor(
            output_by_region[region],
            components=components_by_region[region],
            mean=means_by_region[region],
        )
        target_fr = target_fr_by_region[region]
        reconstruction_losses.append(_weighted_loss(settings.loss_fn, pred_fr, target_fr, weights=time_weights))
        derivative_losses.append(
            _temporal_difference_loss(settings.loss_fn, pred_fr, target_fr, order=1, weights=time_weights)
        )
        curvature_losses.append(
            _temporal_difference_loss(settings.loss_fn, pred_fr, target_fr, order=2, weights=time_weights)
        )
    return (
        torch.mean(torch.stack(reconstruction_losses)),
        torch.mean(torch.stack(derivative_losses)),
        torch.mean(torch.stack(curvature_losses)),
    )


def _write_failed_training_manifest(
    *,
    run_dir: Path,
    settings: FixationMRNNRunSettings,
    seed: int,
    target_mode: str,
    history: Sequence[Mapping[str, object]],
    reason: str,
    iteration: int,
    loss_value: float,
) -> None:
    """Persist a failed run record without writing a usable checkpoint."""
    history_df = pd.DataFrame(history)
    if not history_df.empty:
        history_df.to_csv(run_dir / "history.csv", index=False)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "failure_reason": str(reason),
        "failure_iteration": int(iteration),
        "failure_loss": float(loss_value),
        "seed": int(seed),
        "target_mode": str(target_mode),
        "run_dir": str(run_dir),
        "settings": asdict(settings),
    }
    for filename in ("manifest.json", "training_failed.json"):
        with (run_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def train_one_initialization(
    settings: FixationMRNNRunSettings,
    *,
    run_dir: str | Path,
    seed: int,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train one mRNN initialization and save artifacts."""
    run_dir = Path(run_dir)
    if run_dir.exists() and (run_dir / "checkpoint_final.pth").exists() and not overwrite:
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    settings = FixationMRNNRunSettings(**{**asdict(settings), "seed": int(seed)})
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    device = resolve_device(settings.device)
    target_mode = normalize_target_mode(settings.target_mode)
    targets = make_targets(settings)
    targets_by_region, target = _target_tensors(targets, target_mode=target_mode, device=device)
    target_fr_by_region, fr_components_by_region, fr_means_by_region = _pc_backprojection_tensors(
        targets,
        device=device,
    )
    inp = torch.as_tensor(targets.input_tensor, dtype=torch.float32, device=device)
    output_dims = targets.output_dims_for_mode(target_mode)
    model_spec = build_model_spec(
        region_order=tuple(settings.region_order),
        output_dims_by_region=output_dims,
        hidden_units=settings.hidden_units,
        device=device,
        input_dim=int(inp.shape[-1]),
        activation=settings.activation,
        spectral_radius=settings.spectral_radius,
        rec_constrained=settings.rec_constrained,
        inp_constrained=settings.inp_constrained,
        recurrent_connectivity=normalize_recurrent_connectivity(settings.recurrent_connectivity),
        batch_first=settings.batch_first,
        inp_noise=settings.inp_noise,
        act_noise=settings.act_noise,
    )
    model = FixationMRNNModel(model_spec).to(device)
    h0 = settings.initial_state_scale * torch.randn(inp.shape[0], model.total_num_units, device=device)
    if settings.train_initial_state:
        h0 = nn.Parameter(h0)
        opt_params = [*model.parameters(), h0]
    else:
        opt_params = list(model.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=float(settings.lr))
    time_weights = _time_weights(
        targets.timeline_s,
        pre_fixation_weight=float(settings.pre_fixation_loss_weight),
        post_fixation_weight=float(settings.post_fixation_loss_weight),
        device=device,
    )
    history = []
    divergence_count = 0

    for iteration in tqdm(range(1, int(settings.epochs) + 1), desc=f"{target_mode} seed={seed}", unit="iter"):
        optimizer.zero_grad()
        out = model(inp, h0, noise=False)
        reconstruction = _weighted_loss(settings.loss_fn, out["output"], target, weights=time_weights)
        derivative = _temporal_difference_loss(
            settings.loss_fn,
            out["output"],
            target,
            order=1,
            weights=time_weights,
        )
        curvature = _temporal_difference_loss(
            settings.loss_fn,
            out["output"],
            target,
            order=2,
            weights=time_weights,
        )
        correlation = _temporal_correlation_loss(out["output"], target)
        variance = _variance_loss(out["output"], target)
        if target_mode == "region_pcs":
            fr_reconstruction, fr_derivative, fr_curvature = _firing_rate_reconstruction_loss(
                settings,
                out["output_by_region"],
                target_fr_by_region,
                fr_components_by_region,
                fr_means_by_region,
                region_order=targets.region_order,
                time_weights=time_weights,
            )
        else:
            fr_reconstruction = torch.zeros((), dtype=target.dtype, device=device)
            fr_derivative = torch.zeros((), dtype=target.dtype, device=device)
            fr_curvature = torch.zeros((), dtype=target.dtype, device=device)
        rate = torch.mean(torch.abs(out["h_seq"])) * float(settings.l1_rate_scale)
        l2_rate = torch.mean(out["h_seq"] ** 2) * float(settings.l2_rate_scale)
        weight = _l1([param for param in model.mrnn.parameters()], settings.l1_weight_scale)
        l2_weight = _l2([param for param in model.mrnn.parameters()], settings.l2_weight_scale)
        loss = (
            reconstruction
            + float(settings.temporal_derivative_loss_scale) * derivative
            + float(settings.temporal_curvature_loss_scale) * curvature
            + float(settings.correlation_loss_scale) * correlation
            + float(settings.variance_loss_scale) * variance
            + float(settings.fr_reconstruction_loss_scale) * fr_reconstruction
            + float(settings.fr_temporal_derivative_loss_scale) * fr_derivative
            + float(settings.fr_temporal_curvature_loss_scale) * fr_curvature
            + rate
            + weight
            + l2_rate
            + l2_weight
        )
        row = {
            "iteration": iteration,
            "loss": float(loss.detach().cpu()),
            "reconstruction_loss": float(reconstruction.detach().cpu()),
            "mse_loss": float(reconstruction.detach().cpu()),
            "temporal_derivative_loss": float(derivative.detach().cpu()),
            "temporal_curvature_loss": float(curvature.detach().cpu()),
            "correlation_loss": float(correlation.detach().cpu()),
            "variance_loss": float(variance.detach().cpu()),
            "fr_reconstruction_loss": float(fr_reconstruction.detach().cpu()),
            "fr_temporal_derivative_loss": float(fr_derivative.detach().cpu()),
            "fr_temporal_curvature_loss": float(fr_curvature.detach().cpu()),
            "rate_loss": float(rate.detach().cpu()),
            "weight_loss": float(weight.detach().cpu()),
            "l2_rate_loss": float(l2_rate.detach().cpu()),
            "l2_weight_loss": float(l2_weight.detach().cpu()),
        }
        history.append(row)

        loss_value = float(row["loss"])
        loss_is_finite = bool(np.isfinite(loss_value))
        threshold = settings.divergence_loss_threshold
        threshold_hit = (
            threshold is not None
            and int(iteration) >= int(settings.divergence_min_iteration)
            and loss_value > float(threshold)
        )
        if not loss_is_finite or threshold_hit:
            divergence_count += 1
        else:
            divergence_count = 0
        if not loss_is_finite or divergence_count >= int(settings.divergence_patience):
            reason = "non_finite_loss" if not loss_is_finite else "loss_above_divergence_threshold"
            _write_failed_training_manifest(
                run_dir=run_dir,
                settings=settings,
                seed=int(seed),
                target_mode=target_mode,
                history=history,
                reason=reason,
                iteration=int(iteration),
                loss_value=loss_value,
            )
            raise TrainingDivergedError(
                f"Training diverged for seed={seed} at iteration={iteration}: "
                f"{reason}, loss={loss_value:g}."
            )
        loss.backward()
        if settings.gradient_clip_norm is not None and float(settings.gradient_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=float(settings.gradient_clip_norm))
        optimizer.step()

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    summarize_targets(targets, target_mode=target_mode).to_csv(run_dir / "target_summary.csv", index=False)
    target_payload = {
        region: np.asarray(targets_by_region[region].detach().cpu())
        for region in targets.region_order
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "h0": h0.detach().cpu(),
        "model_spec": asdict(model_spec),
        "settings": asdict(settings),
        "seed": int(seed),
        "target_mode": target_mode,
        "region_order": list(targets.region_order),
        "condition_order": list(targets.condition_order),
        "timeline_s": targets.timeline_s,
        "input_tensor": targets.input_tensor,
        "target_by_region": target_payload,
        "pc_reconstructed_raw_by_region": {
            region: values
            for region, values in targets.pc_reconstructed_raw_by_region().items()
        },
        "features_by_region": {
            region: list(features)
            for region, features in targets.features_for_mode(target_mode).items()
        },
        "normalization_scale": targets.normalization_scale,
        "pca_by_region": serialize_pca_metadata(targets.pca_by_region),
    }
    torch.save(checkpoint, run_dir / "checkpoint_final.pth")
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": int(seed),
                "target_mode": target_mode,
                "run_dir": str(run_dir),
                "final_loss": float(history_df["loss"].iloc[-1]),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return {
        "run_dir": run_dir,
        "checkpoint_path": run_dir / "checkpoint_final.pth",
        "history": history_df,
    }


def train_fixation_mrnn(
    settings: FixationMRNNRunSettings,
    *,
    output_dir: str | Path,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Train in single or multiple initialization mode."""
    mode = str(settings.initialization_mode).strip().lower()
    if mode not in {"single", "multiple"}:
        raise ValueError("initialization_mode must be 'single' or 'multiple'.")
    n_seeds = 1 if mode == "single" else int(settings.n_initializations)
    seeds = load_or_create_seed_plan(
        output_dir,
        settings,
        n_seeds=n_seeds,
        overwrite=bool(settings.overwrite_seed_plan),
    )
    results = []
    for idx, seed in enumerate(seeds):
        run_dir = Path(output_dir) if mode == "single" else Path(output_dir) / f"init={idx:03d}_seed={seed}"
        results.append(train_one_initialization(settings, run_dir=run_dir, seed=seed, overwrite=overwrite))
    return results


def train_fixation_mrnn_scratch(
    settings: FixationMRNNRunSettings,
    *,
    scratch_id: str = "latest",
    overwrite: bool = True,
) -> dict[str, object]:
    """Train a scratch run; returns one result for single mode or an aggregate for multiple mode."""
    output_dir = resolve_fixation_mrnn_output_root(settings) / "scratch" / str(scratch_id)
    results = train_fixation_mrnn(settings, output_dir=output_dir, overwrite=overwrite)
    if len(results) == 1:
        return results[0]
    index = pd.DataFrame(
        [
            {
                "run_dir": str(result["run_dir"]),
                "checkpoint_path": str(result["checkpoint_path"]),
                "final_loss": float(result["history"]["loss"].iloc[-1]),
            }
            for result in results
        ]
    )
    index.to_csv(output_dir / "index.csv", index=False)
    return {"run_dir": output_dir, "index": index, "results": results}


__all__ = [
    "FixationMRNNRunSettings",
    "TrainingDivergedError",
    "load_fixation_mrnn_config",
    "load_or_create_seed_plan",
    "make_targets",
    "resolve_device",
    "resolve_fixation_mrnn_output_root",
    "settings_from_config",
    "train_fixation_mrnn",
    "train_fixation_mrnn_scratch",
    "train_one_initialization",
]
