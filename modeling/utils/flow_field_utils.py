"""Flow-field helpers for trained mRNN checkpoints."""

import numpy as np
import torch
from mRNNTorch.analysis import flow_field

from modeling.utils.current_analysis import condition_label, region_slice


def compute_region_flow_fields(
    replay,
    region,
    condition,
    *,
    num_points=20,
    x_offset=10,
    y_offset=10,
    cancel_other_regions=False,
):
    """Compute regional flow fields using the mRNNTorch flow_field helper."""
    model = replay["model"]
    condition_names = replay["condition_columns"]
    condition_idx = condition_names.index(condition) if isinstance(condition, str) else int(condition)
    condition_name = condition_names[condition_idx]

    data_coords, x_vels, y_vels, speeds = flow_field(
        model.mrnn,
        replay["h_seq"][condition_idx : condition_idx + 1],
        replay["inp"][condition_idx : condition_idx + 1],
        region,
        num_points=num_points,
        x_offset=x_offset,
        y_offset=y_offset,
        cancel_other_regions=cancel_other_regions,
        follow_traj=False,
    )
    return {
        "region": region,
        "condition": condition_name,
        "data_coords": data_coords,
        "x_vels": x_vels,
        "y_vels": y_vels,
        "speeds": speeds,
    }


def plot_flow_field_snapshots(flow_result, *, time_indices=None, plt_module=None):
    """Plot selected flow-field snapshots returned by compute_region_flow_fields."""
    if plt_module is None:
        import matplotlib.pyplot as plt_module

    data_coords = flow_result["data_coords"]
    x_vels = flow_result["x_vels"]
    y_vels = flow_result["y_vels"]
    if time_indices is None:
        max_t = len(x_vels) - 1
        time_indices = np.linspace(0, max_t, num=min(5, len(x_vels)), dtype=int)

    fig, axes = plt_module.subplots(
        1, len(time_indices), figsize=(3.4 * len(time_indices), 3.2)
    )
    axes = np.atleast_1d(axes)
    for ax, time_idx in zip(axes, time_indices):
        coords = data_coords[time_idx]
        ax.streamplot(
            coords[:, :, 0],
            coords[:, :, 1],
            x_vels[time_idx],
            y_vels[time_idx],
            color="black",
            linewidth=1.2,
            arrowsize=1.2,
            zorder=0,
        )
        ax.set_title(f"t={time_idx}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(
        f"{flow_result['region']}: {condition_label(flow_result['condition'])}",
        y=1.04,
    )
    fig.tight_layout()
    return fig


def fit_region_state_pcs(replay, model, regions=None, n_components=2):
    """Fit state PCs for custom local-flow diagnostics."""
    regions = regions or list(model.mrnn.region_dict)
    pc_by_region = {}
    x_seq = replay["x_seq"].detach().cpu().numpy()
    for region in regions:
        sl = region_slice(model, region)
        region_state = x_seq[..., sl]
        flat = region_state.reshape(-1, region_state.shape[-1])
        mean = flat.mean(axis=0, keepdims=True)
        centered = flat - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:n_components]
        scores = centered @ components.T
        pc_by_region[region] = {
            "scores": scores.reshape(region_state.shape[0], region_state.shape[1], -1),
            "components": components,
            "mean": mean.squeeze(0),
        }
    return pc_by_region


def previous_x_sequence(replay):
    """Return x(t) aligned with each transition to x(t+1)."""
    x_seq = replay["x_seq"]
    inp = replay["inp"]
    x0 = replay["xn_0"].to(x_seq.device).expand(inp.shape[0], -1)
    return torch.cat([x0.unsqueeze(1), x_seq[:, :-1]], dim=1)


def plot_local_region_flow_timeseries(
    replay,
    region,
    condition,
    *,
    time_indices=None,
    grid_n=11,
    span_scale=2.0,
    plt_module=None,
):
    """Plot local one-step mRNN flow around a trajectory in regional state-PC space."""
    if plt_module is None:
        import matplotlib.pyplot as plt_module

    model = replay["model"]
    condition_names = replay["condition_columns"]
    condition_idx = condition_names.index(condition) if isinstance(condition, str) else int(condition)
    condition_name = condition_names[condition_idx]
    x_prev = previous_x_sequence(replay)
    inp = replay["inp"]
    state_pcs = fit_region_state_pcs(replay, model, regions=[region], n_components=2)[region]
    components = torch.tensor(state_pcs["components"], dtype=x_prev.dtype, device=x_prev.device)
    mean = torch.tensor(state_pcs["mean"], dtype=x_prev.dtype, device=x_prev.device)
    scores = state_pcs["scores"]

    if time_indices is None:
        max_t = x_prev.shape[1] - 1
        time_indices = np.linspace(0, max_t, num=min(5, x_prev.shape[1]), dtype=int).tolist()

    all_scores = scores.reshape(-1, scores.shape[-1])
    span = np.percentile(np.abs(all_scores), 90, axis=0)
    span = np.maximum(span, 1e-6) * span_scale

    fig, axes = plt_module.subplots(
        1, len(time_indices), figsize=(3.4 * len(time_indices), 3.2)
    )
    axes = np.atleast_1d(axes)
    sl = region_slice(model, region)

    for ax, time_idx in zip(axes, time_indices):
        center_score = torch.tensor(scores[condition_idx, time_idx], dtype=x_prev.dtype, device=x_prev.device)
        gx = torch.linspace(center_score[0] - span[0], center_score[0] + span[0], grid_n, device=x_prev.device)
        gy = torch.linspace(center_score[1] - span[1], center_score[1] + span[1], grid_n, device=x_prev.device)
        grid_x, grid_y = torch.meshgrid(gx, gy, indexing="xy")
        grid_scores = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

        x_grid = x_prev[condition_idx, time_idx].repeat(grid_scores.shape[0], 1)
        x_grid[:, sl] = mean + grid_scores @ components
        inp_grid = inp[condition_idx : condition_idx + 1, time_idx : time_idx + 1].repeat(
            grid_scores.shape[0], 1, 1
        )

        with torch.no_grad():
            x_next, _ = model.mrnn(x_grid, inp_grid, noise=False)
        delta = x_next[:, 0, sl] - x_grid[:, sl]
        delta_scores = delta @ components.T

        ax.quiver(
            grid_scores[:, 0].cpu(),
            grid_scores[:, 1].cpu(),
            delta_scores[:, 0].cpu(),
            delta_scores[:, 1].cpu(),
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.004,
            alpha=0.75,
        )
        traj = scores[condition_idx]
        ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=1.0, alpha=0.5)
        ax.scatter(traj[time_idx, 0], traj[time_idx, 1], color="red", s=24, zorder=3)
        ax.set_title(f"t={time_idx}")
        ax.set_xlabel("state PC1")
        ax.set_ylabel("state PC2")
    fig.suptitle(f"{region}: {condition_label(condition_name)}", y=1.04)
    fig.tight_layout()
    return fig
