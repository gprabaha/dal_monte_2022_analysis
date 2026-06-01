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

    Rows report vector norms and relative recurrent contributions.
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

        norm_stack = torch.stack(recurrent_norms, dim=0)
        denom = norm_stack.sum(dim=0, keepdim=True)
        rel = norm_stack / torch.where(denom > 0, denom, torch.ones_like(denom))
        rel = torch.where(denom > 0, rel, torch.zeros_like(rel))
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
                            "relative_contribution": float(rel[source_idx, cond_idx, time_idx].detach().cpu()),
                        }
                    )
    return pd.DataFrame(rows), vectors


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
    "load_fixation_mrnn_checkpoint",
    "output_pc_scores",
    "reconstruction_accuracy",
    "replay_fixation_mrnn_run",
    "variance_comparison",
]
