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
    temporal_basis_count: int = 20
    hidden_units: int | dict[str, int] = 100
    activation: str = "softplus"
    spectral_radius: float | None = 1.3
    rec_constrained: bool = False
    inp_constrained: bool = False
    batch_first: bool = True
    inp_noise: float = 0.0
    act_noise: float = 0.0
    epochs: int = 10_000
    lr: float = 1e-3
    loss_fn: str = "mse"
    l1_weight_scale: float = 0.0
    l1_rate_scale: float = 0.0
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


def _l1(parameters: Sequence[torch.Tensor], scale: float) -> torch.Tensor:
    if not parameters:
        return torch.zeros(())
    penalty = torch.zeros((), device=parameters[0].device)
    for param in parameters:
        penalty = penalty + torch.mean(torch.abs(param))
    return penalty * float(scale)


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
    criterion = _loss_fn(settings.loss_fn)
    history = []

    for iteration in tqdm(range(1, int(settings.epochs) + 1), desc=f"{target_mode} seed={seed}", unit="iter"):
        optimizer.zero_grad()
        out = model(inp, h0, noise=False)
        mse = criterion(out["output"], target)
        rate = torch.mean(torch.abs(out["h_seq"])) * float(settings.l1_rate_scale)
        weight = _l1([param for param in model.mrnn.parameters()], settings.l1_weight_scale)
        loss = mse + rate + weight
        loss.backward()
        optimizer.step()
        row = {
            "iteration": iteration,
            "loss": float(loss.detach().cpu()),
            "mse_loss": float(mse.detach().cpu()),
            "rate_loss": float(rate.detach().cpu()),
            "weight_loss": float(weight.detach().cpu()),
        }
        history.append(row)

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
