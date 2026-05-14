"""Replay and analysis helpers for trained fixation mRNN runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_model import (
    FixationMRNNModel,
    FixationMRNNModelSpec,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_targets import (
    build_condition_input,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_training import resolve_device


def _model_spec_from_checkpoint(checkpoint: Mapping[str, object], *, device: str) -> FixationMRNNModelSpec:
    spec = dict(checkpoint["model_spec"])
    spec["canonical_region_order"] = tuple(spec["canonical_region_order"])
    spec["internal_region_order"] = tuple(spec["internal_region_order"])
    spec["hidden_units_by_region"] = {
        str(region): int(value)
        for region, value in dict(spec["hidden_units_by_region"]).items()
    }
    spec["output_dims_by_region"] = {
        str(region): int(value)
        for region, value in dict(spec["output_dims_by_region"]).items()
    }
    spec["device"] = device
    return FixationMRNNModelSpec(**spec)


def load_fixation_mrnn_checkpoint(
    run_dir: str | Path,
    *,
    device: str = "cpu",
) -> tuple[FixationMRNNModel, dict[str, object]]:
    """Load a trained fixation mRNN model and checkpoint payload."""
    resolved_device = resolve_device(device)
    checkpoint_path = Path(run_dir) / "checkpoint_final.pth"
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model = FixationMRNNModel(
        _model_spec_from_checkpoint(checkpoint, device=resolved_device)
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _canonicalize_region_output(
    output: torch.Tensor,
    *,
    canonical_features: Sequence[str],
    internal_features: Sequence[str],
) -> torch.Tensor:
    index = {str(feature): idx for idx, feature in enumerate(internal_features)}
    restore_indices = [index[str(feature)] for feature in canonical_features]
    return output[..., restore_indices]


def canonicalize_outputs_by_region(
    output_by_region: Mapping[str, torch.Tensor],
    checkpoint: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Restore per-region model outputs from run-internal to canonical feature order."""
    canonical_features = checkpoint["canonical_feature_order_by_region"]
    internal_features = checkpoint["internal_feature_order_by_region"]
    return {
        region: _canonicalize_region_output(
            output_by_region[region],
            canonical_features=canonical_features[region],
            internal_features=internal_features[region],
        )
        for region in checkpoint["canonical_region_order"]
    }


def replay_fixation_mrnn_run(
    run_dir: str | Path,
    *,
    device: str = "cpu",
    noise: bool = False,
    stim_input: torch.Tensor | None = None,
) -> dict[str, object]:
    """Replay a trained fixation mRNN for all saved fixation conditions."""
    model, checkpoint = load_fixation_mrnn_checkpoint(run_dir, device=device)
    resolved_device = next(model.parameters()).device
    condition_names = tuple(checkpoint["condition_names"])
    if "input_tensor" in checkpoint:
        input_tensor = np.asarray(checkpoint["input_tensor"], dtype=np.float32)
    else:
        input_tensor = build_condition_input(
            condition_names,
            timesteps=len(checkpoint["timeline_s_rel"]),
            dtype=np.float32,
        )
    inp = torch.tensor(input_tensor, dtype=torch.float32, device=resolved_device)
    x0 = checkpoint["x0"].to(resolved_device).expand(inp.shape[0], -1)
    if stim_input is not None:
        stim_input = stim_input.to(resolved_device)
    with torch.no_grad():
        replay = model(inp, x0, stim_input=stim_input, noise=noise)
    canonical_output_by_region = canonicalize_outputs_by_region(
        replay["output_by_region"],
        checkpoint,
    )
    canonical_output = torch.cat(
        [
            canonical_output_by_region[region]
            for region in checkpoint["canonical_region_order"]
        ],
        dim=-1,
    )
    replay.update(
        {
            "model": model,
            "checkpoint": checkpoint,
            "condition_names": condition_names,
            "inp": inp,
            "x0": x0,
            "canonical_output_by_region": canonical_output_by_region,
            "canonical_output": canonical_output,
        }
    )
    return replay


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


def compute_fixation_mrnn_currents(replay: Mapping[str, object]) -> tuple[pd.DataFrame, dict[tuple[str, str], torch.Tensor]]:
    """Compute source-region recurrent currents into each target region."""
    model = replay["model"]
    mrnn = model.mrnn
    x_seq = replay["x_seq"]
    h_seq = replay["h_seq"]
    inp = replay["inp"]
    x0 = replay["x0"]
    h0 = mrnn.activation(x0)
    x_prev = torch.cat([x0.unsqueeze(1), x_seq[:, :-1]], dim=1)
    h_prev = torch.cat([h0.unsqueeze(1), h_seq[:, :-1]], dim=1)
    w_rec = _effective_recurrent_weight(model)
    w_inp = _effective_input_weight(model)
    baseline = mrnn.tonic_inp
    regions = list(checkpoint_region_order(replay))
    condition_names = tuple(replay["condition_names"])

    rows: list[dict[str, object]] = []
    vectors: dict[tuple[str, str], torch.Tensor] = {}
    flat_inp = inp.reshape(-1, inp.shape[-1])
    for target_region in regions:
        target_start, target_end = mrnn.get_region_indices(target_region)
        target_slice = slice(target_start, target_end)
        target_next = h_seq[..., target_slice]
        leak = -x_prev[..., target_slice]
        base = baseline[target_slice].expand_as(target_next)
        external = (w_inp[target_slice, :] @ flat_inp.T).T.reshape(
            inp.shape[0], inp.shape[1], -1
        )
        for source, current in {
            "leak": leak,
            "baseline": base,
            "external_input": external,
        }.items():
            vectors[(target_region, source)] = current.detach().cpu()
            rows.extend(
                _current_summary_rows(
                    current,
                    target_next,
                    condition_names,
                    target_region,
                    source,
                )
            )
        for source_region in regions:
            source_start, source_end = mrnn.get_region_indices(source_region)
            source_slice = slice(source_start, source_end)
            block = w_rec[target_slice, source_slice]
            source_h = h_prev[..., source_slice]
            current = (block @ source_h.reshape(-1, source_h.shape[-1]).T).T.reshape(
                source_h.shape[0], source_h.shape[1], -1
            )
            vectors[(target_region, source_region)] = current.detach().cpu()
            rows.extend(
                _current_summary_rows(
                    current,
                    target_next,
                    condition_names,
                    target_region,
                    source_region,
                )
            )
    return pd.DataFrame(rows), vectors


def checkpoint_region_order(replay: Mapping[str, object]) -> tuple[str, ...]:
    """Return canonical region order from replay/checkpoint payload."""
    checkpoint = replay["checkpoint"]
    return tuple(str(region) for region in checkpoint["canonical_region_order"])


def _current_summary_rows(
    current: torch.Tensor,
    target_next: torch.Tensor,
    condition_names: Sequence[str],
    target_region: str,
    source: str,
) -> list[dict[str, object]]:
    norm = torch.linalg.vector_norm(current, dim=-1)
    target_norm = torch.linalg.vector_norm(target_next, dim=-1)
    cosine = F.cosine_similarity(current, target_next, dim=-1, eps=1e-8)
    dot = (current * target_next).sum(dim=-1)
    rows: list[dict[str, object]] = []
    for cond_idx, condition in enumerate(condition_names):
        for time_idx in range(current.shape[1]):
            rows.append(
                {
                    "condition": str(condition),
                    "time_idx": int(time_idx),
                    "target_region": str(target_region),
                    "source": str(source),
                    "current_norm": float(norm[cond_idx, time_idx].detach().cpu()),
                    "target_next_norm": float(target_norm[cond_idx, time_idx].detach().cpu()),
                    "dot_to_next_activity": float(dot[cond_idx, time_idx].detach().cpu()),
                    "cosine_to_next_activity": float(cosine[cond_idx, time_idx].detach().cpu()),
                }
            )
    return rows


def compute_fixation_mrnn_eigenvalues(
    replay: Mapping[str, object],
    *,
    regions: Sequence[str] | None = None,
    dh: bool = False,
) -> pd.DataFrame:
    """Compute local Jacobian eigenvalues over replayed conditions and time."""
    try:
        from mrnntorch.analysis.linear.leaky_linear import mLinearization
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Installed 'mrnntorch' linear-analysis modules are required."
        ) from exc

    model = replay["model"]
    region_args = tuple(regions or checkpoint_region_order(replay))
    linearization = mLinearization(model.mrnn, *region_args)
    inp = replay["inp"]
    x_seq = replay["x_seq"]
    h_seq = replay["h_seq"]
    x0 = replay["x0"]
    h0 = model.mrnn.activation(x0)
    x_prev = torch.cat([x0.unsqueeze(1), x_seq[:, :-1]], dim=1)
    h_prev = torch.cat([h0.unsqueeze(1), h_seq[:, :-1]], dim=1)

    rows: list[dict[str, object]] = []
    for cond_idx, condition in enumerate(replay["condition_names"]):
        for time_idx in range(inp.shape[1]):
            reals, ims, _ = linearization.eigendecomposition(
                inp[cond_idx, time_idx],
                x_prev[cond_idx, time_idx],
                h=h_prev[cond_idx, time_idx] if dh else None,
                dh=dh,
            )
            for eig_idx, (real, imag) in enumerate(zip(reals, ims)):
                rows.append(
                    {
                        "condition": str(condition),
                        "time_idx": int(time_idx),
                        "eig_idx": int(eig_idx),
                        "regions": ",".join(region_args),
                        "real": float(real),
                        "imag": float(imag),
                    }
                )
    return pd.DataFrame(rows)


def compute_fixation_mrnn_flow_fields(
    replay: Mapping[str, object],
    *,
    region: str,
    condition: str | int = 0,
    num_points: int = 7,
    x_offset: float = 1.0,
    y_offset: float = 1.0,
    cancel_other_regions: bool = False,
):
    """Compute installed-mrnntorch flow fields for one region and condition."""
    try:
        from mrnntorch.analysis.flow_fields.leaky_flow_field_finder import (
            mFlowFieldFinder,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Installed 'mrnntorch' flow-field modules and 'rnntoolkit' are required."
        ) from exc

    model = replay["model"]
    condition_names = tuple(replay["condition_names"])
    cond_idx = condition_names.index(condition) if isinstance(condition, str) else int(condition)
    x_seq = replay["x_seq"][cond_idx : cond_idx + 1]
    inp = replay["inp"][cond_idx : cond_idx + 1]
    fit_states = model.mrnn.get_region_activity(x_seq, region)
    finder = mFlowFieldFinder(
        model.mrnn,
        fit_states,
        num_points=int(num_points),
        x_offset=float(x_offset),
        y_offset=float(y_offset),
        region_list=[region],
        cancel_other_regions=bool(cancel_other_regions),
    )
    return {
        "region": region,
        "condition": condition_names[cond_idx],
        "flow_fields": finder.find_nonlinear_flow(x_seq, inp),
    }


def build_region_stimulus(
    replay: Mapping[str, object],
    *,
    regions: Sequence[str],
    start_idx: int,
    stop_idx: int,
    strength: float,
    ramp_up: int = 0,
    ramp_down: int = 0,
) -> torch.Tensor:
    """Build an additive stimulus tensor targeting one or more mRNN regions."""
    model = replay["model"]
    mrnn = model.mrnn
    inp = replay["inp"]
    stim = torch.zeros(
        (inp.shape[0], inp.shape[1], mrnn.total_num_units),
        dtype=inp.dtype,
        device=inp.device,
    )
    mask = torch.zeros((mrnn.total_num_units,), dtype=inp.dtype, device=inp.device)
    for region in regions:
        start, stop = mrnn.get_region_indices(str(region))
        mask[start:stop] = 1.0
    start_idx = max(0, int(start_idx))
    stop_idx = min(int(stop_idx), int(inp.shape[1]))
    if stop_idx <= start_idx:
        return stim
    values = torch.full((stop_idx - start_idx,), float(strength), device=inp.device)
    if ramp_up > 0:
        n = min(int(ramp_up), values.shape[0])
        values[:n] = torch.linspace(0.0, float(strength), n, device=inp.device)
    if ramp_down > 0:
        n = min(int(ramp_down), values.shape[0])
        values[-n:] = torch.linspace(float(strength), 0.0, n, device=inp.device)
    stim[:, start_idx:stop_idx, :] = values[None, :, None] * mask[None, None, :]
    return stim


__all__ = [
    "build_region_stimulus",
    "canonicalize_outputs_by_region",
    "compute_fixation_mrnn_currents",
    "compute_fixation_mrnn_eigenvalues",
    "compute_fixation_mrnn_flow_fields",
    "load_fixation_mrnn_checkpoint",
    "replay_fixation_mrnn_run",
]
