import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
from utils.models import Model
from utils.exp_utils import pca_batched, save_fig, load_hp, load_model, initial_state
from utils.train_utils import get_mean_fixation_data, interactivity_input
from utils.plt_utils import ax_3d_no_grid
from utils.current_analysis import condition_label, region_slice
import os
import numpy as np


def plot_pca(act, exp_path, region, data_type):
    fig, ax = ax_3d_no_grid()

    label_str = ["interactive face", "noninteractive face", "object"]
    ax.plot(
        act[0, :, 0],
        act[0, :, 1],
        act[0, :, 2],
        linewidth=4,
        label=label_str[0],
        color="blue",
    )
    ax.plot(
        act[1, :, 0],
        act[1, :, 1],
        act[1, :, 2],
        linewidth=4,
        label=label_str[1],
        color="maroon",
    )
    ax.plot(
        act[2, :, 0],
        act[2, :, 1],
        act[2, :, 2],
        linewidth=4,
        label=label_str[2],
        color="green",
    )
    # ax.legend(loc="best")

    save_path = os.path.join(exp_path, f"{region}_{data_type}_pca")
    save_fig(save_path, eps=False)


def fit_region_pcs(activity, model, regions=None, n_components=3):
    """Fit PCA to each region's activity across all conditions and time bins."""
    regions = regions or list(model.mrnn.region_dict)
    pc_by_region = {}
    for region in regions:
        sl = region_slice(model, region)
        region_activity = activity[..., sl].detach().cpu().numpy()
        flat = region_activity.reshape(-1, region_activity.shape[-1])
        mean = flat.mean(axis=0, keepdims=True)
        centered = flat - mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:n_components]
        scores = centered @ components.T
        denom = max(centered.shape[0] - 1, 1)
        explained = (singular_values[:n_components] ** 2) / denom
        total = (singular_values**2).sum() / denom
        explained_ratio = explained / total if total > 0 else np.zeros_like(explained)
        pc_by_region[region] = {
            "scores": scores.reshape(
                region_activity.shape[0], region_activity.shape[1], -1
            ),
            "components": components,
            "mean": mean.squeeze(0),
            "explained_ratio": explained_ratio,
        }
    return pc_by_region


def plot_region_pc_timeseries(
    pc_by_region, condition_names, pc_indices=(0, 1, 2), *, plt_module=None
):
    """Plot regional PC score time series by fixation condition."""
    if plt_module is None:
        import matplotlib.pyplot as plt_module

    n_regions = len(pc_by_region)
    fig, axes = plt_module.subplots(
        n_regions,
        len(pc_indices),
        figsize=(4.0 * len(pc_indices), 2.6 * n_regions),
        sharex=True,
    )
    axes = np.asarray(axes)
    if n_regions == 1:
        axes = axes[None, :]
    if len(pc_indices) == 1:
        axes = axes[:, None]

    for row_idx, (region, data) in enumerate(pc_by_region.items()):
        scores = data["scores"]
        for col_idx, pc_idx in enumerate(pc_indices):
            ax = axes[row_idx, col_idx]
            for cond_idx, condition in enumerate(condition_names):
                ax.plot(scores[cond_idx, :, pc_idx], label=condition_label(condition))
            pct = 100 * data["explained_ratio"][pc_idx]
            ax.set_title(f"{region} PC{pc_idx + 1} ({pct:.1f}%)")
            ax.set_ylabel("score")
            if row_idx == n_regions - 1:
                ax.set_xlabel("time bin")
            if row_idx == 0 and col_idx == len(pc_indices) - 1:
                ax.legend(
                    frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left"
                )
    fig.tight_layout()
    return fig


def plot_all_pcs(model, exp_path, x, data_type, dataset=None):
    """Plot the PCs for each region in REGIONS"""
    l_idx = 0
    for region in model.out_order:
        if data_type == "hidden_activity":
            act = model.mrnn.get_region_activity(x, region)
        elif data_type == "data":
            if dataset is None:
                raise Exception
            r_idx = dataset.get_region_indices(region)
            act = x[..., r_idx]
        elif data_type == "output":
            if model.latent_training:
                act = x[..., l_idx : l_idx + model.n_components]
                l_idx += model.n_components
            else:
                if dataset is None:
                    raise Exception
                r_idx = dataset.get_region_indices(region)
                act = x[..., r_idx]
        else:
            raise ValueError

        act_reduced = pca_batched(act)
        plot_pca(act_reduced, exp_path, region, data_type)


def model_pca(model_path, data_type):
    hp = load_hp(model_path)
    exp_path = f"results/{hp['model_save_name']}/pca"

    dataset = get_mean_fixation_data("~/naturalistic_social_gaze_mech/social_gaze")

    model = load_model(hp, dataset)

    # Start training
    batch, _ = dataset.sample_batch()
    inp = interactivity_input(dataset.group_by_columns, batch.shape[1])
    inp = inp.cpu()

    xn, _ = initial_state(hp, model, batch.shape[0])

    with torch.no_grad():
        out, hn = model(xn, inp, noise=False)

    if data_type == "data":
        plot_all_pcs(model, exp_path, batch, "data", dataset)
    elif data_type == "output":
        plot_all_pcs(model, exp_path, out, "output", dataset)
    elif data_type == "hidden_activity":
        plot_all_pcs(model, exp_path, hn, "hidden_activity")
