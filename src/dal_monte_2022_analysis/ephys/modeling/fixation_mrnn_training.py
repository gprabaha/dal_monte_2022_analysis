"""Run planning and training for fixation mRNN models."""

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
import yaml

from dal_monte_2022_analysis.config.load import load_config, resolve_repo_path
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_model import (
    FixationMRNNModel,
    build_model_spec,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_targets import (
    CANONICAL_CONDITION_ORDER,
    FixationMRNNTargets,
    build_condition_input,
    build_fixation_mrnn_targets,
    normalize_target_mode,
    serialize_pca_metadata,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path


RUN_PLAN_JSON_COLUMNS = (
    "canonical_region_order",
    "internal_region_order",
    "canonical_feature_order_by_region",
    "internal_feature_order_by_region",
    "hidden_units_by_region",
)


@dataclass
class FixationMRNNRunSettings:
    """Settings for one fixation mRNN training run."""

    dataset_cfg_path: str = "configs/dataset.yaml"
    input_subdir: str = "ephys/psth/fixation_psth_averages"
    dataframe_filename: str = "fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl"
    timeline_filename: str = "fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl"
    output_subdir: str = "ephys/modeling/fixation_mrnn"
    target_mode: str = "raw_fr"
    canonical_region_order: tuple[str, ...] = ("ofc", "bla", "dmpfc", "accg")
    internal_region_order: tuple[str, ...] | None = None
    internal_feature_order_by_region: dict[str, tuple[str, ...]] | None = None
    hidden_units: int | dict[str, int] = 100
    activation: str = "softplus"
    dt: float = 10.0
    tau: float = 100.0
    spectral_radius: float | None = 1.3
    batch_first: bool = True
    rec_constrained: bool = False
    inp_constrained: bool = False
    inp_noise: float = 0.0
    act_noise: float = 0.0
    input_region_name: str = "input"
    normalize_targets: bool = True
    normalization_stabilizer: float = 5.0
    pca_variance_threshold: float = 0.95
    epochs: int = 500
    lr: float = 1e-3
    loss_fn: str = "mse"
    l1_weight_scale: float = 1e-4
    l1_rate_scale: float = 1e-4
    initial_state: str = "zeros"
    initial_state_scale: float = 0.01
    train_initial_state: bool = True
    seed: int = 123456
    region_order_shuffle_seed: int | None = None
    feature_order_shuffle_seed: int | None = None
    device: str = "auto"
    checkpoint_every: int = 0


def load_fixation_mrnn_config(path: str | Path = "configs/ephys_fixation_mrnn.yaml") -> dict[str, Any]:
    """Load the generic fixation mRNN config."""
    return load_config(path, config_type="generic")


def settings_from_config(
    cfg: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> FixationMRNNRunSettings:
    """Create run settings from a config dictionary and explicit overrides."""
    data = dict(cfg)
    if "dataset_cfg_path" in data:
        data["dataset_cfg_path"] = str(data["dataset_cfg_path"])
    if "canonical_region_order" in data:
        data["canonical_region_order"] = tuple(data["canonical_region_order"])
    overrides = dict(overrides or {})
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    allowed = set(FixationMRNNRunSettings.__dataclass_fields__)
    return FixationMRNNRunSettings(
        **{key: value for key, value in data.items() if key in allowed}
    )


def resolve_device(device: str) -> str:
    """Resolve auto/cuda/cpu device strings for torch and mrnntorch."""
    token = str(device).strip().lower()
    if token == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if token.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return token


def resolve_fixation_mrnn_output_root(settings: FixationMRNNRunSettings) -> Path:
    """Resolve the canonical fixation mRNN output root."""
    cfg = load_config(settings.dataset_cfg_path)
    return build_analysis_output_dir(cfg, settings.output_subdir)


def derive_internal_region_order(
    canonical_region_order: Sequence[str],
    *,
    seed: int,
    shuffle: bool = True,
) -> tuple[str, ...]:
    """Derive a deterministic internal mRNN region construction order."""
    regions = tuple(str(region) for region in canonical_region_order)
    if not shuffle:
        return regions
    rng = np.random.default_rng(int(seed))
    return tuple(rng.permutation(np.asarray(regions, dtype=object)).tolist())


def derive_internal_feature_order_by_region(
    canonical_feature_order_by_region: Mapping[str, Sequence[str]],
    *,
    seed: int,
    shuffle: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Derive deterministic within-region feature orders."""
    rng = np.random.default_rng(int(seed))
    out: dict[str, tuple[str, ...]] = {}
    for region, features in canonical_feature_order_by_region.items():
        feature_tuple = tuple(str(feature) for feature in features)
        if shuffle and len(feature_tuple) > 1:
            out[str(region)] = tuple(
                rng.permutation(np.asarray(feature_tuple, dtype=object)).tolist()
            )
        else:
            out[str(region)] = feature_tuple
    return out


def _hidden_units_by_region(
    hidden_units: int | Mapping[str, int],
    canonical_region_order: Sequence[str],
) -> dict[str, int]:
    if isinstance(hidden_units, Mapping):
        return {region: int(hidden_units[region]) for region in canonical_region_order}
    return {region: int(hidden_units) for region in canonical_region_order}


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _json_loads(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _make_targets(settings: FixationMRNNRunSettings) -> FixationMRNNTargets:
    return build_fixation_mrnn_targets(
        settings.dataset_cfg_path,
        input_subdir=settings.input_subdir,
        dataframe_filename=settings.dataframe_filename,
        timeline_filename=settings.timeline_filename,
        canonical_region_order=tuple(settings.canonical_region_order),
        condition_names=CANONICAL_CONDITION_ORDER,
        normalize_targets=bool(settings.normalize_targets),
        normalization_stabilizer=float(settings.normalization_stabilizer),
        pca_variance_threshold=float(settings.pca_variance_threshold),
    )


def prepare_fixation_mrnn_run_plan(
    settings: FixationMRNNRunSettings,
    *,
    experiment_id: str,
    n_runs: int,
    seed_start: int,
    target_modes: Sequence[str] | None = None,
    shuffle_region_order: bool = True,
    shuffle_feature_order: bool = True,
    overwrite: bool = False,
) -> Path:
    """Create an indexed run plan for a fixation mRNN experiment."""
    target_modes = tuple(target_modes or (settings.target_mode,))
    output_root = resolve_fixation_mrnn_output_root(settings)
    experiment_dir = output_root / "experiments" / str(experiment_id)
    if experiment_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Experiment directory already exists: {experiment_dir}. "
            "Pass overwrite=True to replace the plan files."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    target_cache: dict[str, FixationMRNNTargets] = {}
    rows: list[dict[str, object]] = []
    run_idx = 0
    for target_mode_raw in target_modes:
        target_mode = normalize_target_mode(target_mode_raw)
        if target_mode not in target_cache:
            target_cache[target_mode] = _make_targets(settings)
        targets = target_cache[target_mode]
        canonical_features = targets.feature_order_for_mode(target_mode)
        output_dims = targets.output_dims_for_mode(target_mode)

        for offset in range(int(n_runs)):
            seed = int(seed_start) + int(offset)
            region_seed = seed
            feature_seed = seed + 1_000_003
            internal_region_order = derive_internal_region_order(
                settings.canonical_region_order,
                seed=region_seed,
                shuffle=shuffle_region_order,
            )
            internal_feature_order = derive_internal_feature_order_by_region(
                canonical_features,
                seed=feature_seed,
                shuffle=shuffle_feature_order,
            )
            hidden_units_by_region = _hidden_units_by_region(
                settings.hidden_units,
                settings.canonical_region_order,
            )
            rows.append(
                {
                    "run_idx": run_idx,
                    "seed": seed,
                    "target_mode": target_mode,
                    "init_kind": settings.initial_state,
                    "hidden_units": int(settings.hidden_units)
                    if not isinstance(settings.hidden_units, Mapping)
                    else _json_dumps(hidden_units_by_region),
                    "hidden_units_by_region": _json_dumps(hidden_units_by_region),
                    "canonical_region_order": _json_dumps(list(settings.canonical_region_order)),
                    "internal_region_order": _json_dumps(list(internal_region_order)),
                    "canonical_feature_order_by_region": _json_dumps(
                        {k: list(v) for k, v in canonical_features.items()}
                    ),
                    "internal_feature_order_by_region": _json_dumps(
                        {k: list(v) for k, v in internal_feature_order.items()}
                    ),
                    "region_order_shuffle_seed": region_seed,
                    "feature_order_shuffle_seed": feature_seed,
                    "output_dims_by_region": _json_dumps(output_dims),
                    "status": "pending",
                }
            )
            run_idx += 1

    run_plan = pd.DataFrame(rows)
    run_plan.to_csv(experiment_dir / "run_plan.csv", index=False)
    experiment_payload = asdict(settings)
    experiment_payload.update(
        {
            "experiment_id": str(experiment_id),
            "n_runs": int(n_runs),
            "seed_start": int(seed_start),
            "target_modes": [normalize_target_mode(mode) for mode in target_modes],
            "shuffle_region_order": bool(shuffle_region_order),
            "shuffle_feature_order": bool(shuffle_feature_order),
        }
    )
    with (experiment_dir / "experiment.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(experiment_payload, f, sort_keys=True)
    return experiment_dir / "run_plan.csv"


def read_fixation_mrnn_run_plan(path: str | Path) -> pd.DataFrame:
    """Read a run plan and JSON-decode structured columns."""
    df = pd.read_csv(path)
    for column in RUN_PLAN_JSON_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(_json_loads)
    if "output_dims_by_region" in df.columns:
        df["output_dims_by_region"] = df["output_dims_by_region"].map(_json_loads)
    return df


def _row_for_run_idx(run_plan: pd.DataFrame, run_idx: int) -> dict[str, object]:
    rows = run_plan.loc[run_plan["run_idx"].astype(int) == int(run_idx)]
    if rows.empty:
        raise KeyError(f"run_idx={run_idx} was not found in the run plan.")
    if len(rows) > 1:
        raise ValueError(f"run_idx={run_idx} appears more than once in the run plan.")
    return rows.iloc[0].to_dict()


def _run_dir_name(run_idx: int, seed: int) -> str:
    return f"run={int(run_idx):04d}_seed={int(seed)}"


def _scratch_run_dir(settings: FixationMRNNRunSettings, scratch_id: str) -> Path:
    return resolve_fixation_mrnn_output_root(settings) / "scratch" / str(scratch_id)


def _experiment_run_dir(
    settings: FixationMRNNRunSettings,
    *,
    experiment_id: str,
    run_idx: int,
    seed: int,
) -> Path:
    return (
        resolve_fixation_mrnn_output_root(settings)
        / "experiments"
        / str(experiment_id)
        / "runs"
        / _run_dir_name(run_idx, seed)
    )


def _feature_indices(
    canonical: Sequence[str],
    internal: Sequence[str],
) -> list[int]:
    index = {str(feature): idx for idx, feature in enumerate(canonical)}
    return [index[str(feature)] for feature in internal]


def _tensor_targets_for_training(
    targets: FixationMRNNTargets,
    *,
    target_mode: str,
    internal_feature_order_by_region: Mapping[str, Sequence[str]],
    device: str,
) -> dict[str, torch.Tensor]:
    canonical_targets = targets.targets_for_mode(target_mode)
    canonical_features = targets.feature_order_for_mode(target_mode)
    out: dict[str, torch.Tensor] = {}
    for region in targets.canonical_region_order:
        indices = _feature_indices(
            canonical_features[region],
            internal_feature_order_by_region[region],
        )
        out[region] = torch.tensor(
            canonical_targets[region][..., indices],
            dtype=torch.float32,
            device=device,
        )
    return out


def _concatenate_region_tensors(
    values_by_region: Mapping[str, torch.Tensor],
    region_order: Sequence[str],
) -> torch.Tensor:
    return torch.cat([values_by_region[region] for region in region_order], dim=-1)


def _get_loss_fn(loss_fn: str) -> nn.Module:
    token = str(loss_fn).strip().lower()
    if token == "mse":
        return nn.MSELoss()
    if token == "mae":
        return nn.L1Loss()
    if token == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError("Unsupported loss_fn. Expected mse, mae, or smooth_l1.")


def _initial_state(
    kind: str,
    shape: tuple[int, int],
    *,
    device: str,
    scale: float,
) -> torch.Tensor:
    token = str(kind).strip().lower()
    if token == "zeros":
        return torch.zeros(shape, device=device)
    if token == "normal":
        return float(scale) * torch.randn(shape, device=device)
    if token == "uniform":
        return torch.empty(shape, device=device).uniform_(-float(scale), float(scale))
    raise ValueError("Unsupported initial_state. Expected zeros, normal, or uniform.")


def _l1_weight(rnn, scale: float) -> torch.Tensor:
    penalty = torch.zeros((), device=next(rnn.parameters()).device)
    for param in rnn.parameters():
        penalty = penalty + torch.mean(torch.abs(param.flatten()))
    return penalty * float(scale)


def _l1_rate(activity: torch.Tensor, scale: float) -> torch.Tensor:
    return float(scale) * torch.mean(torch.abs(activity.flatten()))


def _write_status(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True, default=str)


def _run_manifest(
    *,
    settings: FixationMRNNRunSettings,
    targets: FixationMRNNTargets,
    model_spec,
    run_idx: int | None,
    experiment_id: str | None,
    scratch_id: str | None,
    internal_feature_order_by_region: Mapping[str, Sequence[str]],
    run_dir: Path,
) -> dict[str, object]:
    target_mode = normalize_target_mode(settings.target_mode)
    canonical_features = targets.feature_order_for_mode(target_mode)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_idx": run_idx,
        "experiment_id": experiment_id,
        "scratch_id": scratch_id,
        "target_mode": target_mode,
        "seed": int(settings.seed),
        "run_dir": str(run_dir),
        "dataframe_path": str(targets.dataframe_path) if targets.dataframe_path else None,
        "timeline_path": str(targets.timeline_path) if targets.timeline_path else None,
        "condition_names": list(targets.condition_names),
        "canonical_region_order": list(targets.canonical_region_order),
        "internal_region_order": list(model_spec.internal_region_order),
        "canonical_feature_order_by_region": {
            region: list(features) for region, features in canonical_features.items()
        },
        "internal_feature_order_by_region": {
            region: list(features)
            for region, features in internal_feature_order_by_region.items()
        },
        "region_order_shuffle_seed": settings.region_order_shuffle_seed,
        "feature_order_shuffle_seed": settings.feature_order_shuffle_seed,
        "model_spec": asdict(model_spec),
        "training_settings": asdict(settings),
    }


def train_fixation_mrnn_run(
    settings: FixationMRNNRunSettings,
    *,
    run_dir: str | Path,
    run_idx: int | None = None,
    experiment_id: str | None = None,
    scratch_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train one fixation mRNN run and persist run artifacts."""
    run_dir = Path(run_dir)
    if run_dir.exists() and not overwrite:
        if (run_dir / "checkpoint_final.pth").exists():
            raise FileExistsError(f"Run already has a final checkpoint: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(settings.seed))
    np.random.seed(int(settings.seed) % (2**32 - 1))
    device = resolve_device(settings.device)
    targets = _make_targets(settings)
    target_mode = normalize_target_mode(settings.target_mode)
    canonical_features = targets.feature_order_for_mode(target_mode)

    internal_region_order = settings.internal_region_order
    region_seed = settings.region_order_shuffle_seed
    if internal_region_order is None:
        region_seed = int(settings.seed) if region_seed is None else int(region_seed)
        internal_region_order = derive_internal_region_order(
            settings.canonical_region_order,
            seed=region_seed,
            shuffle=True,
        )

    internal_feature_order = settings.internal_feature_order_by_region
    feature_seed = settings.feature_order_shuffle_seed
    if internal_feature_order is None:
        feature_seed = int(settings.seed) + 1_000_003 if feature_seed is None else int(feature_seed)
        internal_feature_order = derive_internal_feature_order_by_region(
            canonical_features,
            seed=feature_seed,
            shuffle=True,
        )
    internal_feature_order = {
        region: tuple(features) for region, features in internal_feature_order.items()
    }

    settings.region_order_shuffle_seed = region_seed
    settings.feature_order_shuffle_seed = feature_seed
    settings.internal_region_order = tuple(internal_region_order)
    settings.internal_feature_order_by_region = dict(internal_feature_order)

    output_dims = {
        region: len(internal_feature_order[region])
        for region in settings.canonical_region_order
    }
    model_spec = build_model_spec(
        canonical_region_order=tuple(settings.canonical_region_order),
        internal_region_order=tuple(internal_region_order),
        output_dims_by_region=output_dims,
        hidden_units=settings.hidden_units,
        device=device,
        input_dim=3,
        input_region_name=settings.input_region_name,
        activation=settings.activation,
        dt=settings.dt,
        tau=settings.tau,
        inp_noise=settings.inp_noise,
        act_noise=settings.act_noise,
        rec_constrained=settings.rec_constrained,
        inp_constrained=settings.inp_constrained,
        batch_first=settings.batch_first,
        spectral_radius=settings.spectral_radius,
    )

    status_path = run_dir / "status.json"
    _write_status(
        status_path,
        {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(settings.seed),
        },
    )

    try:
        model = FixationMRNNModel(model_spec).to(device)
        train_targets_by_region = _tensor_targets_for_training(
            targets,
            target_mode=target_mode,
            internal_feature_order_by_region=internal_feature_order,
            device=device,
        )
        target = _concatenate_region_tensors(
            train_targets_by_region,
            settings.canonical_region_order,
        )
        inp = torch.tensor(targets.input_tensor, dtype=torch.float32, device=device)
        x0_tensor = _initial_state(
            settings.initial_state,
            (1, model.total_num_units),
            device=device,
            scale=settings.initial_state_scale,
        )
        if settings.train_initial_state:
            x0_param = nn.Parameter(x0_tensor)
            opt_params = [*model.parameters(), x0_param]
        else:
            x0_param = x0_tensor
            opt_params = list(model.parameters())

        optimizer = torch.optim.Adam(opt_params, lr=float(settings.lr))
        criterion = _get_loss_fn(settings.loss_fn)
        history: list[dict[str, float | int]] = []
        batch_size = int(inp.shape[0])

        for epoch in range(1, int(settings.epochs) + 1):
            model.train()
            optimizer.zero_grad()
            x0_batch = x0_param.expand(batch_size, -1)
            out = model(inp, x0_batch, noise=True)
            pred = out["output"]
            mse_loss = criterion(pred, target)
            rate_loss = _l1_rate(out["h_seq"], settings.l1_rate_scale)
            weight_loss = _l1_weight(model.mrnn, settings.l1_weight_scale)
            loss = mse_loss + rate_loss + weight_loss
            loss.backward()
            optimizer.step()

            record = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "mse_loss": float(mse_loss.detach().cpu()),
                "rate_loss": float(rate_loss.detach().cpu()),
                "weight_loss": float(weight_loss.detach().cpu()),
            }
            history.append(record)

            if settings.checkpoint_every and epoch % int(settings.checkpoint_every) == 0:
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "x0": x0_param.detach().cpu(),
                        "model_spec": asdict(model_spec),
                        "settings": asdict(settings),
                    },
                    checkpoint_dir / f"epoch={epoch:06d}.pth",
                )

        history_df = pd.DataFrame(history)
        history_df.to_csv(run_dir / "history.csv", index=False)
        manifest = _run_manifest(
            settings=settings,
            targets=targets,
            model_spec=model_spec,
            run_idx=run_idx,
            experiment_id=experiment_id,
            scratch_id=scratch_id,
            internal_feature_order_by_region=internal_feature_order,
            run_dir=run_dir,
        )
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True, default=str)
        save_pickle_path(targets.training_dataframe, run_dir / "training_dataframe.pkl")
        save_pickle_path(
            serialize_pca_metadata(targets.pca_metadata_by_region),
            run_dir / "pca_metadata.pkl",
        )
        pd.DataFrame(
            [
                {"region": region, "feature": feature, "feature_index": idx}
                for region, features in canonical_features.items()
                for idx, feature in enumerate(features)
            ]
        ).to_csv(run_dir / "canonical_feature_inventory.csv", index=False)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "x0": x0_param.detach().cpu(),
                "model_spec": asdict(model_spec),
                "settings": asdict(settings),
                "manifest": manifest,
                "condition_names": list(targets.condition_names),
                "timeline_s_rel": targets.timeline_s_rel,
                "input_tensor": targets.input_tensor,
                "target_mode": target_mode,
                "canonical_region_order": list(settings.canonical_region_order),
                "internal_region_order": list(internal_region_order),
                "canonical_feature_order_by_region": {
                    region: list(features) for region, features in canonical_features.items()
                },
                "internal_feature_order_by_region": {
                    region: list(features)
                    for region, features in internal_feature_order.items()
                },
                "pca_metadata_by_region": serialize_pca_metadata(targets.pca_metadata_by_region),
            },
            run_dir / "checkpoint_final.pth",
        )
        _write_status(
            status_path,
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": int(settings.seed),
                "final_loss": float(history[-1]["loss"]) if history else None,
                "epochs": int(settings.epochs),
            },
        )
        return {
            "run_dir": run_dir,
            "history": history_df,
            "manifest": manifest,
            "checkpoint_path": run_dir / "checkpoint_final.pth",
        }
    except Exception as exc:
        _write_status(
            status_path,
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": int(settings.seed),
                "error": repr(exc),
            },
        )
        raise


def train_fixation_mrnn_scratch(
    settings: FixationMRNNRunSettings,
    *,
    scratch_id: str = "latest",
    overwrite: bool = True,
) -> dict[str, object]:
    """Train a replaceable scratch run."""
    return train_fixation_mrnn_run(
        settings,
        run_dir=_scratch_run_dir(settings, scratch_id),
        scratch_id=scratch_id,
        overwrite=overwrite,
    )


def settings_for_experiment_run(
    base_settings: FixationMRNNRunSettings,
    *,
    run_plan_path: str | Path,
    run_idx: int,
) -> FixationMRNNRunSettings:
    """Merge one run-plan row into base settings."""
    row = _row_for_run_idx(read_fixation_mrnn_run_plan(run_plan_path), run_idx)
    hidden_units_by_region = row.get("hidden_units_by_region")
    settings = FixationMRNNRunSettings(**asdict(base_settings))
    settings.target_mode = normalize_target_mode(str(row["target_mode"]))
    settings.seed = int(row["seed"])
    settings.initial_state = str(row.get("init_kind", settings.initial_state))
    settings.hidden_units = {
        str(region): int(value)
        for region, value in dict(hidden_units_by_region).items()
    }
    settings.canonical_region_order = tuple(row["canonical_region_order"])
    settings.internal_region_order = tuple(row["internal_region_order"])
    settings.internal_feature_order_by_region = {
        str(region): tuple(features)
        for region, features in dict(row["internal_feature_order_by_region"]).items()
    }
    settings.region_order_shuffle_seed = int(row["region_order_shuffle_seed"])
    settings.feature_order_shuffle_seed = int(row["feature_order_shuffle_seed"])
    return settings


def train_fixation_mrnn_experiment_run(
    base_settings: FixationMRNNRunSettings,
    *,
    experiment_id: str,
    run_idx: int,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train one run from an experiment run plan."""
    experiment_dir = (
        resolve_fixation_mrnn_output_root(base_settings)
        / "experiments"
        / str(experiment_id)
    )
    run_plan_path = experiment_dir / "run_plan.csv"
    settings = settings_for_experiment_run(
        base_settings,
        run_plan_path=run_plan_path,
        run_idx=run_idx,
    )
    return train_fixation_mrnn_run(
        settings,
        run_dir=_experiment_run_dir(
            settings,
            experiment_id=experiment_id,
            run_idx=run_idx,
            seed=settings.seed,
        ),
        run_idx=run_idx,
        experiment_id=experiment_id,
        overwrite=overwrite,
    )


def rebuild_fixation_mrnn_experiment_index(
    settings: FixationMRNNRunSettings,
    *,
    experiment_id: str,
) -> Path:
    """Rebuild index.csv by scanning run manifests and status files."""
    experiment_dir = (
        resolve_fixation_mrnn_output_root(settings)
        / "experiments"
        / str(experiment_id)
    )
    rows: list[dict[str, object]] = []
    for run_dir in sorted((experiment_dir / "runs").glob("run=*")):
        status = {}
        manifest = {}
        if (run_dir / "status.json").exists():
            with (run_dir / "status.json").open("r", encoding="utf-8") as f:
                status = json.load(f)
        if (run_dir / "manifest.json").exists():
            with (run_dir / "manifest.json").open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_idx": manifest.get("run_idx"),
                "seed": manifest.get("seed", status.get("seed")),
                "target_mode": manifest.get("target_mode"),
                "status": status.get("status", "unknown"),
                "final_loss": status.get("final_loss"),
                "epochs": status.get("epochs"),
                "checkpoint_path": str(run_dir / "checkpoint_final.pth")
                if (run_dir / "checkpoint_final.pth").exists()
                else None,
            }
        )
    index_path = experiment_dir / "index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path


__all__ = [
    "FixationMRNNRunSettings",
    "derive_internal_feature_order_by_region",
    "derive_internal_region_order",
    "load_fixation_mrnn_config",
    "prepare_fixation_mrnn_run_plan",
    "read_fixation_mrnn_run_plan",
    "rebuild_fixation_mrnn_experiment_index",
    "resolve_device",
    "resolve_fixation_mrnn_output_root",
    "settings_for_experiment_run",
    "settings_from_config",
    "train_fixation_mrnn_experiment_run",
    "train_fixation_mrnn_run",
    "train_fixation_mrnn_scratch",
]
