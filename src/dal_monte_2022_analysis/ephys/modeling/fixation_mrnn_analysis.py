"""Minimal replay and analysis for fixation Elman mRNNs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_model import (
    FixationMRNNModel,
    FixationMRNNModelSpec,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_targets import backproject_region_pcs
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_training import resolve_device


def _model_spec_from_checkpoint(checkpoint: Mapping[str, object], *, device: str) -> FixationMRNNModelSpec:
    spec = dict(checkpoint["model_spec"])
    spec["region_order"] = tuple(spec["region_order"])
    spec["hidden_units_by_region"] = {str(k): int(v) for k, v in dict(spec["hidden_units_by_region"]).items()}
    spec["output_dims_by_region"] = {str(k): int(v) for k, v in dict(spec["output_dims_by_region"]).items()}
    spec["device"] = str(device)
    return FixationMRNNModelSpec(**spec)


def load_fixation_mrnn_checkpoint(
    run_dir: str | Path,
    *,
    device: str = "cpu",
) -> tuple[FixationMRNNModel, dict[str, object]]:
    """Load a trained model and checkpoint."""
    resolved_device = resolve_device(device)
    checkpoint = torch.load(
        Path(run_dir) / "checkpoint_final.pth",
        map_location=resolved_device,
        weights_only=False,
    )
    model = FixationMRNNModel(_model_spec_from_checkpoint(checkpoint, device=resolved_device)).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def replay_fixation_mrnn_run(
    run_dir: str | Path,
    *,
    device: str = "cpu",
    noise: bool = False,
) -> dict[str, object]:
    """Replay a trained model on its saved inputs."""
    model, checkpoint = load_fixation_mrnn_checkpoint(run_dir, device=device)
    resolved_device = next(model.parameters()).device
    inp = torch.as_tensor(checkpoint["input_tensor"], dtype=torch.float32, device=resolved_device)
    h0 = checkpoint["h0"].to(resolved_device)
    with torch.no_grad():
        out = model(inp, h0, noise=noise)
    return {
        **out,
        "model": model,
        "checkpoint": checkpoint,
        "inp": inp,
        "h0": h0,
        "condition_order": tuple(checkpoint["condition_order"]),
        "region_order": tuple(checkpoint["region_order"]),
        "output_by_region": out["output_by_region"],
    }


def _run_model_replay(
    model: FixationMRNNModel,
    checkpoint: Mapping[str, object],
    *,
    noise: bool = False,
) -> dict[str, object]:
    resolved_device = next(model.parameters()).device
    inp = torch.as_tensor(checkpoint["input_tensor"], dtype=torch.float32, device=resolved_device)
    h0 = checkpoint["h0"].to(resolved_device)
    with torch.no_grad():
        out = model(inp, h0, noise=noise)
    return {
        **out,
        "model": model,
        "checkpoint": checkpoint,
        "inp": inp,
        "h0": h0,
        "condition_order": tuple(checkpoint["condition_order"]),
        "region_order": tuple(checkpoint["region_order"]),
        "output_by_region": out["output_by_region"],
    }


def replay_fixation_mrnn_run_with_ablations(
    run_dir: str | Path,
    *,
    ablations: Sequence[tuple[str, str]],
    device: str = "cpu",
    noise: bool = False,
) -> dict[str, object]:
    """Replay a checkpoint after zeroing selected recurrent region blocks.

    Each ablation is a ``(source_region, target_region)`` tuple. The
    corresponding n_source x n_target recurrent block is set to zero before
    replay, deleting that directed source-to-target current.
    """
    model, checkpoint = load_fixation_mrnn_checkpoint(run_dir, device=device)
    mrnn = model.mrnn
    with torch.no_grad():
        for source_region, target_region in ablations:
            source_start, source_stop = mrnn.get_region_indices(source_region)
            target_start, target_stop = mrnn.get_region_indices(target_region)
            mrnn.W_rec[target_start:target_stop, source_start:source_stop] = 0.0
    replay = _run_model_replay(model, checkpoint, noise=noise)
    replay["ablated_connections"] = tuple((str(source), str(target)) for source, target in ablations)
    return replay


def backproject_replay_outputs_to_firing_rates(replay: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Back-project PC-space model outputs into normalized firing-rate space."""
    checkpoint = replay["checkpoint"]
    if checkpoint.get("target_mode") != "region_pcs":
        raise ValueError("Firing-rate backprojection is only defined for region_pcs checkpoints.")
    return {
        region: backproject_region_pcs(
            replay["output_by_region"][region].detach().cpu().numpy().astype(float, copy=False),
            checkpoint["pca_by_region"][region],
        )
        for region in replay["region_order"]
    }


def pc_reconstructed_firing_rate_accuracy(replay: Mapping[str, object]) -> pd.DataFrame:
    """Compare predicted and target PC-backprojected firing-rate trajectories."""
    checkpoint = replay["checkpoint"]
    target_fr = checkpoint.get("pc_reconstructed_raw_by_region")
    if target_fr is None:
        target_fr = {
            region: backproject_region_pcs(
                np.asarray(checkpoint["target_by_region"][region], dtype=float),
                checkpoint["pca_by_region"][region],
            )
            for region in replay["region_order"]
        }
    predicted_fr = backproject_replay_outputs_to_firing_rates(replay)
    rows = []
    for region in replay["region_order"]:
        observed = np.asarray(target_fr[region], dtype=float)
        predicted = np.asarray(predicted_fr[region], dtype=float)
        for cond_idx, condition in enumerate(replay["condition_order"]):
            y = observed[cond_idx].reshape(-1)
            yhat = predicted[cond_idx].reshape(-1)
            err = y - yhat
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            ss_res = float(np.sum(err**2))
            rows.append(
                {
                    "region": region,
                    "condition": condition,
                    "mse": float(np.mean(err**2)),
                    "mae": float(np.mean(np.abs(err))),
                    "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                    "correlation": float(np.corrcoef(y, yhat)[0, 1])
                    if np.std(y) > 0 and np.std(yhat) > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def reconstruction_accuracy(replay: Mapping[str, object]) -> pd.DataFrame:
    """Compute reconstruction metrics by region and condition."""
    checkpoint = replay["checkpoint"]
    rows = []
    for region in replay["region_order"]:
        observed = np.asarray(checkpoint["target_by_region"][region], dtype=float)
        predicted = replay["output_by_region"][region].detach().cpu().numpy().astype(float, copy=False)
        for cond_idx, condition in enumerate(replay["condition_order"]):
            y = observed[cond_idx].reshape(-1)
            yhat = predicted[cond_idx].reshape(-1)
            err = y - yhat
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            ss_res = float(np.sum(err**2))
            rows.append(
                {
                    "region": region,
                    "condition": condition,
                    "mse": float(np.mean(err**2)),
                    "mae": float(np.mean(np.abs(err))),
                    "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                    "correlation": float(np.corrcoef(y, yhat)[0, 1])
                    if np.std(y) > 0 and np.std(yhat) > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def variance_comparison(replay: Mapping[str, object]) -> pd.DataFrame:
    """Compare target and reconstruction variance by region."""
    checkpoint = replay["checkpoint"]
    rows = []
    for region in replay["region_order"]:
        observed = np.asarray(checkpoint["target_by_region"][region], dtype=float)
        predicted = replay["output_by_region"][region].detach().cpu().numpy().astype(float, copy=False)
        observed_var = float(np.var(observed))
        predicted_var = float(np.var(predicted))
        rows.append(
            {
                "region": region,
                "observed_variance": observed_var,
                "reconstructed_variance": predicted_var,
                "reconstructed_to_observed_ratio": predicted_var / observed_var if observed_var > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _effective_recurrent_weight(model: FixationMRNNModel) -> torch.Tensor:
    mrnn = model.mrnn
    if mrnn.rec_constrained:
        return mrnn.apply_dales_law(mrnn.W_rec, mrnn.W_rec_mask, mrnn.W_rec_sign_matrix)
    return mrnn.W_rec * mrnn.W_rec_mask


def _effective_input_weight(model: FixationMRNNModel) -> torch.Tensor:
    mrnn = model.mrnn
    if mrnn.inp_constrained:
        return mrnn.apply_dales_law(mrnn.W_inp, mrnn.W_inp_mask, mrnn.W_inp_sign_matrix)
    return mrnn.W_inp * mrnn.W_inp_mask


def extract_region_currents(replay: Mapping[str, object]) -> tuple[pd.DataFrame, dict[tuple[str, str], torch.Tensor]]:
    """Extract Elman recurrent currents.

    For target region r and source region s, the code takes the block
    W_rec[r, s] from the effective recurrent matrix and multiplies it by
    the previous source hidden activity h_s(t - 1):

        I_{s -> r}(t) = W_rec[r, s] h_s(t - 1)

    Rows report vector norms, signed projections onto the target-region next
    activity vector, and signed relative contributions. The signed projection is
    positive when a source current aligns with the next activity vector and
    negative when it opposes that vector.
    """
    model = replay["model"]
    mrnn = model.mrnn
    h_seq = replay["h_seq"]
    h0 = replay["h0"]
    h_prev = torch.cat([h0.unsqueeze(1), h_seq[:, :-1]], dim=1)
    inp = replay["inp"]
    w_rec = _effective_recurrent_weight(model)
    w_inp = _effective_input_weight(model)
    rows = []
    vectors: dict[tuple[str, str], torch.Tensor] = {}
    region_order = tuple(replay["region_order"])
    condition_order = tuple(replay["condition_order"])
    flat_inp = inp.reshape(-1, inp.shape[-1])

    for target_region in region_order:
        target_start, target_stop = mrnn.get_region_indices(target_region)
        target_slice = slice(target_start, target_stop)
        recurrent_norms = []
        recurrent_projections = []
        target_next = h_seq[..., target_slice]
        target_next_norm = torch.linalg.vector_norm(target_next, dim=-1)
        for source_region in region_order:
            source_start, source_stop = mrnn.get_region_indices(source_region)
            source_slice = slice(source_start, source_stop)
            block = w_rec[target_slice, source_slice]
            source_h = h_prev[..., source_slice]
            current = (block @ source_h.reshape(-1, source_h.shape[-1]).T).T.reshape(
                source_h.shape[0],
                source_h.shape[1],
                -1,
            )
            vectors[(target_region, source_region)] = current.detach().cpu()
            recurrent_norms.append(torch.linalg.vector_norm(current, dim=-1))
            projection = (current * target_next).sum(dim=-1) / torch.clamp(
                target_next_norm,
                min=1e-8,
            )
            recurrent_projections.append(projection)

        norm_stack = torch.stack(recurrent_norms, dim=0)
        projection_stack = torch.stack(recurrent_projections, dim=0)
        projection_denom = torch.sum(torch.abs(projection_stack), dim=0, keepdim=True)
        rel = projection_stack / torch.where(
            projection_denom > 0,
            projection_denom,
            torch.ones_like(projection_denom),
        )
        rel = torch.where(projection_denom > 0, rel, torch.zeros_like(rel))
        external = (w_inp[target_slice, :] @ flat_inp.T).T.reshape(inp.shape[0], inp.shape[1], -1)
        vectors[(target_region, "external_input")] = external.detach().cpu()
        vectors[(target_region, "bias")] = mrnn.tonic_inp[target_slice].detach().cpu()

        for source_idx, source_region in enumerate(region_order):
            for cond_idx, condition in enumerate(condition_order):
                for time_idx in range(h_seq.shape[1]):
                    rows.append(
                        {
                            "target_region": target_region,
                            "source_region": source_region,
                            "condition": condition,
                            "time_idx": int(time_idx),
                            "current_norm": float(norm_stack[source_idx, cond_idx, time_idx].detach().cpu()),
                            "signed_projection": float(
                                projection_stack[source_idx, cond_idx, time_idx].detach().cpu()
                            ),
                            "target_next_norm": float(
                                target_next_norm[cond_idx, time_idx].detach().cpu()
                            ),
                            "relative_contribution": float(rel[source_idx, cond_idx, time_idx].detach().cpu()),
                        }
                    )
    return pd.DataFrame(rows), vectors


def _fit_hidden_pca(values: np.ndarray, *, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    n_conditions, n_time, n_features = values.shape
    flat = values.reshape(n_conditions * n_time, n_features)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((int(n_components), n_features), dtype=float)
    n_fit = min(int(n_components), vt.shape[0])
    if n_fit:
        components[:n_fit] = vt[:n_fit]
    return mean.squeeze(0), components


def compute_global_flow_field(
    replay: Mapping[str, object],
    *,
    condition: str,
    time_idx: int,
    grid_points: int = 15,
    radius: float = 1.0,
) -> dict[str, object]:
    """Compute a 2D Elman flow field in global hidden-state PC space.

    The PCA is fit on the full hidden state concatenated across all regions.
    At each grid point, the whole hidden state is replaced by a point in that
    global PC plane, one Elman update is applied with the selected condition
    and time-bin input held fixed, and the displacement is projected back into
    the same global PC plane.
    """
    model = replay["model"]
    mrnn = model.mrnn
    condition_order = tuple(replay["condition_order"])
    cond_idx = condition_order.index(condition)
    time_idx = int(time_idx)
    h_seq = replay["h_seq"]
    if time_idx < 0 or time_idx >= int(h_seq.shape[1]):
        raise IndexError(f"time_idx={time_idx} outside replay length {h_seq.shape[1]}.")

    hidden_values = h_seq.detach().cpu().numpy().astype(float, copy=False)
    mean, components = _fit_hidden_pca(hidden_values, n_components=2)
    scores = (hidden_values.reshape(-1, hidden_values.shape[-1]) - mean) @ components.T
    center = (hidden_values[cond_idx, time_idx] - mean) @ components.T
    spread = float(np.nanstd(scores[:, :2]))
    if not np.isfinite(spread) or spread <= 0:
        spread = 1.0
    extent = float(radius) * spread
    x_values = np.linspace(center[0] - extent, center[0] + extent, int(grid_points))
    y_values = np.linspace(center[1] - extent, center[1] + extent, int(grid_points))
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    h_base = h_seq[cond_idx, time_idx].detach().clone()
    inp_t = replay["inp"][cond_idx, time_idx]
    w_rec = _effective_recurrent_weight(model)
    w_inp = _effective_input_weight(model)
    velocities = np.zeros((*grid_x.shape, 2), dtype=float)

    with torch.no_grad():
        for row in range(grid_x.shape[0]):
            for col in range(grid_x.shape[1]):
                pc_point = np.asarray([grid_x[row, col], grid_y[row, col]], dtype=float)
                h_full_np = mean + pc_point @ components
                h_full = torch.as_tensor(h_full_np, dtype=h_base.dtype, device=h_base.device)
                pre_next = w_rec @ h_full + w_inp @ inp_t + mrnn.tonic_inp
                h_next = mrnn.activation(pre_next)
                next_hidden = h_next.detach().cpu().numpy().astype(float, copy=False)
                next_pc = (next_hidden - mean) @ components.T
                velocities[row, col] = next_pc - pc_point

    return {
        "region": "global",
        "condition": condition,
        "time_idx": int(time_idx),
        "grid_x": grid_x,
        "grid_y": grid_y,
        "u": velocities[..., 0],
        "v": velocities[..., 1],
        "speed": np.linalg.norm(velocities, axis=-1),
    }


def compute_region_flow_field(
    replay: Mapping[str, object],
    *,
    region: str,
    condition: str,
    time_idx: int,
    grid_points: int = 15,
    radius: float = 1.0,
) -> dict[str, object]:
    """Compute a region-specific flow contribution in global hidden PC space.

    The PC axes are fit once on the full hidden state across all regions. At
    each grid point, the model is advanced one Elman step, but only the
    selected region's slice of the hidden-state displacement is projected back
    onto those global PC axes. This keeps every region in a comparable PC
    coordinate system while showing each region's contribution separately.
    """
    model = replay["model"]
    mrnn = model.mrnn
    if region not in tuple(replay["region_order"]):
        raise ValueError(f"Unknown region: {region}")
    condition_order = tuple(replay["condition_order"])
    cond_idx = condition_order.index(condition)
    time_idx = int(time_idx)
    h_seq = replay["h_seq"]
    if time_idx < 0 or time_idx >= int(h_seq.shape[1]):
        raise IndexError(f"time_idx={time_idx} outside replay length {h_seq.shape[1]}.")

    hidden_values = h_seq.detach().cpu().numpy().astype(float, copy=False)
    mean, components = _fit_hidden_pca(hidden_values, n_components=2)
    scores = (hidden_values.reshape(-1, hidden_values.shape[-1]) - mean) @ components.T
    center = (hidden_values[cond_idx, time_idx] - mean) @ components.T
    spread = float(np.nanstd(scores[:, :2]))
    if not np.isfinite(spread) or spread <= 0:
        spread = 1.0
    extent = float(radius) * spread
    x_values = np.linspace(center[0] - extent, center[0] + extent, int(grid_points))
    y_values = np.linspace(center[1] - extent, center[1] + extent, int(grid_points))
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    region_start, region_stop = mrnn.get_region_indices(region)
    region_slice = slice(region_start, region_stop)
    h_base = h_seq[cond_idx, time_idx].detach().clone()
    inp_t = replay["inp"][cond_idx, time_idx]
    w_rec = _effective_recurrent_weight(model)
    w_inp = _effective_input_weight(model)
    velocities = np.zeros((*grid_x.shape, 2), dtype=float)

    with torch.no_grad():
        for row in range(grid_x.shape[0]):
            for col in range(grid_x.shape[1]):
                pc_point = np.asarray([grid_x[row, col], grid_y[row, col]], dtype=float)
                h_full_np = mean + pc_point @ components
                h_full = torch.as_tensor(h_full_np, dtype=h_base.dtype, device=h_base.device)
                pre_next = w_rec @ h_full + w_inp @ inp_t + mrnn.tonic_inp
                h_next = mrnn.activation(pre_next)
                delta = h_next.detach().cpu().numpy().astype(float, copy=False) - h_full_np
                region_delta = np.zeros_like(delta)
                region_delta[region_slice] = delta[region_slice]
                velocities[row, col] = region_delta @ components.T

    return {
        "region": region,
        "condition": condition,
        "time_idx": int(time_idx),
        "grid_x": grid_x,
        "grid_y": grid_y,
        "u": velocities[..., 0],
        "v": velocities[..., 1],
        "speed": np.linalg.norm(velocities, axis=-1),
    }


def output_pc_scores(
    replay: Mapping[str, object],
    *,
    region: str,
    n_components: int = 3,
) -> np.ndarray:
    """Compute PCs from model output trajectories for one region."""
    values = replay["output_by_region"][region].detach().cpu().numpy().astype(float, copy=False)
    n_conditions, n_time, n_features = values.shape
    flat = values.reshape(n_conditions * n_time, n_features)
    centered = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_fit = min(int(n_components), vt.shape[0])
    scores = np.zeros((flat.shape[0], int(n_components)), dtype=float)
    if n_fit:
        scores[:, :n_fit] = centered @ vt[:n_fit].T
    return scores.reshape(n_conditions, n_time, int(n_components))


__all__ = [
    "extract_region_currents",
    "compute_global_flow_field",
    "compute_region_flow_field",
    "backproject_replay_outputs_to_firing_rates",
    "load_fixation_mrnn_checkpoint",
    "output_pc_scores",
    "pc_reconstructed_firing_rate_accuracy",
    "reconstruction_accuracy",
    "replay_fixation_mrnn_run",
    "replay_fixation_mrnn_run_with_ablations",
    "variance_comparison",
]
