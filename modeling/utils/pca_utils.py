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
import os


def plot_pca(act, exp_path, region, data_type):
    fig, ax = ax_3d_no_grid()

    label_str = ["high_interactivity_face", "low_interactivity_face", "object"]
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

    xn, hn = initial_state(hp, model, batch.shape[0])

    with torch.no_grad():
        out, hn = model(inp, xn, hn, noise=False)

    if data_type == "data":
        plot_all_pcs(model, exp_path, batch, "data", dataset)
    elif data_type == "output":
        plot_all_pcs(model, exp_path, out, "output", dataset)
    elif data_type == "hidden_activity":
        plot_all_pcs(model, exp_path, hn, "hidden_activity")
