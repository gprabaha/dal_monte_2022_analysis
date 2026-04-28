import os
import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
from utils.plt_utils import standard_2d_ax
from utils.exp_utils import (
    load_hp,
    save_fig,
    stim_inp,
    get_other_regions,
)
from utils.train_utils import get_mean_fixation_data, interactivity_input
from utils.models import Model
from mrnntorch.analysis.linear.linearization import Linearization
import numpy as np
from scipy.stats import wasserstein_distance
from utils.exp_utils import pvalues

# Supress warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_eigenvalues(model_path, condition, *args, stim_inp=None):
    hp = load_hp(model_path)

    dataset = get_mean_fixation_data("~/naturalistic_social_gaze_mech/social_gaze")

    model = Model(
        hp["mrnn_config_file"],
        100,
        dataset.units_per_region["dmpfc"],
        dataset.units_per_region["accg"],
        dataset.units_per_region["ofc"],
        dataset.units_per_region["bla"],
        hp["dt"],
        hp["tau"],
        hp["inp_noise"],
        hp["act_noise"],
        hp["rec_constrained"],
        hp["inp_constrained"],
        hp["batch_first"],
        hp["spectral_radius"],
        output_layer=hp["output_layer"],
        latent_training=hp["latent_training"],
        n_components=hp["n_components"],
        device="cpu",
    ).cpu()

    checkpoint = torch.load(
        os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth")
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch, _ = dataset.sample_batch()
    inp = interactivity_input(dataset.group_by_columns, batch.shape[1])

    xn = checkpoint["xn_0"].cpu()
    hn = checkpoint["hn_0"].cpu()

    with torch.no_grad():
        if stim_inp is not None:
            out, hn = model(inp, xn, hn, stim_inp, noise=False)
        else:
            out, hn = model(inp, xn, hn, noise=False)

    out = out.detach().cpu()
    hn = hn.detach().cpu()

    hn_cond = hn[condition]
    linearization = Linearization(model.mrnn)

    max_eigs = []
    eigs_r, eigs_im = [], []
    for hn_t in hn_cond:
        reals, ims, _ = linearization.eigendecomposition(hn_t, *args)
        max_real = max(reals)
        eigs_r.append(reals)
        eigs_im.append(ims)
        max_eigs.append(max_real)

    return max_eigs, eigs_r, eigs_im


def get_ablation_stims(model_path, region, start_silence=50, end_silence=75):
    hp = load_hp(model_path)

    dataset = get_mean_fixation_data("~/naturalistic_social_gaze_mech/social_gaze")

    model = Model(
        hp["mrnn_config_file"],
        100,
        dataset.units_per_region["dmpfc"],
        dataset.units_per_region["accg"],
        dataset.units_per_region["ofc"],
        dataset.units_per_region["bla"],
        hp["dt"],
        hp["tau"],
        hp["inp_noise"],
        hp["act_noise"],
        hp["rec_constrained"],
        hp["inp_constrained"],
        hp["batch_first"],
        hp["spectral_radius"],
        output_layer=hp["output_layer"],
        latent_training=hp["latent_training"],
        n_components=hp["n_components"],
        device="cpu",
    ).cpu()

    checkpoint = torch.load(
        os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth")
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch, _ = dataset.sample_batch()
    inp = interactivity_input(dataset.group_by_columns, batch.shape[1])

    regions = get_other_regions(model, region)

    stim_list = []
    for s in range(len(regions)):
        stim_list.append(
            stim_inp(
                model.mrnn,
                start_silence,
                end_silence,
                inp.shape[1],
                0,
                -5,
                inp.shape[0],
                5,
                10,
                regions[s],
            )
        )

    return stim_list, regions


def plot_max_eigs(model_path, region):
    hp = load_hp(model_path)
    model_save_name = hp["model_save_name"]
    exp_path = f"results/{model_save_name}/linear_analysis"

    fig, ax = standard_2d_ax()
    max_eigs_high_int, _, _ = get_eigenvalues(model_path, 0, region)
    max_eigs_low_int, _, _ = get_eigenvalues(model_path, 1, region)
    max_eigs_object, _, _ = get_eigenvalues(model_path, 2, region)

    ax.plot(max_eigs_high_int, linewidth=4, color="red")
    ax.plot(max_eigs_low_int, linewidth=4, color="blue")
    ax.plot(max_eigs_object, linewidth=4, color="green")
    save_fig(os.path.join(exp_path, f"{region}_max_eigs_all_conds"))


def plot_eig_dist(model_path, region):
    def _make_scatters(data_r, data_im, cond):
        for t, (eigs_r, eigs_im) in enumerate(zip(data_r, data_im)):
            fig, ax = standard_2d_ax()
            ax.scatter(
                eigs_r, eigs_im, color="red", alpha=0.75, edgecolors="black", s=100
            )
            theta = np.linspace(0, 2 * np.pi, 500)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.set_xticks([-1, 1])
            ax.set_yticks([-1, 1])
            ax.plot(x, y, linewidth=1, color="black")
            save_fig(
                os.path.join(exp_path, f"{region}", f"cond{cond}", f"eig_dist_t{t}"),
                eps=True,
            )

    hp = load_hp(model_path)
    model_save_name = hp["model_save_name"]
    exp_path = f"results/{model_save_name}/linear_analysis"

    _, eigs_hi_r, eigs_hi_im = get_eigenvalues(model_path, 0, region)
    _, eigs_li_r, eigs_li_im = get_eigenvalues(model_path, 1, region)
    _, eigs_o_r, eigs_o_im = get_eigenvalues(model_path, 2, region)

    _make_scatters(eigs_hi_r, eigs_hi_im, 0)
    _make_scatters(eigs_li_r, eigs_li_im, 1)
    _make_scatters(eigs_o_r, eigs_o_im, 2)


# ABLATION
def plot_max_eigs_ablation(model_path, region):
    hp = load_hp(model_path)
    model_save_name = hp["model_save_name"]
    exp_path = f"results/{model_save_name}/linear_analysis"
    stim_list, regions = get_ablation_stims(model_path, region)

    for s in range(3):
        fig, ax = standard_2d_ax()
        max_eigs_high_int, _, _ = get_eigenvalues(
            model_path, 0, region, stim_inp=stim_list[s]
        )
        max_eigs_low_int, _, _ = get_eigenvalues(
            model_path, 1, region, stim_inp=stim_list[s]
        )
        max_eigs_object, _, _ = get_eigenvalues(
            model_path, 2, region, stim_inp=stim_list[s]
        )

        ax.plot(max_eigs_high_int, linewidth=4, color="red")
        ax.plot(max_eigs_low_int, linewidth=4, color="blue")
        ax.plot(max_eigs_object, linewidth=4, color="green")
        ax.axvline(x=50, linestyle="--", color="grey", linewidth=2)
        ax.axvline(x=100, linestyle="--", color="grey", linewidth=2)
        save_fig(
            os.path.join(exp_path, f"{region}_max_eigs_all_conds_ablate_{regions[s]}")
        )


def eigs_all_models(region, model_path, stim=False, start_silence=50, end_silence=75):
    all_model_dists_high_int = {}
    all_model_dists_low_int = {}
    all_model_dists_object = {}

    # This is horrible
    if stim:
        stim_list, regions = get_ablation_stims(
            model_path, region, start_silence=start_silence, end_silence=end_silence
        )
    else:
        _, regions = get_ablation_stims(model_path, region)

    # intialize lists for each region
    for r in regions:
        if r not in all_model_dists_high_int:
            all_model_dists_high_int[r] = []
        if r not in all_model_dists_low_int:
            all_model_dists_low_int[r] = []
        if r not in all_model_dists_object:
            all_model_dists_object[r] = []

    for s in range(len(regions)):
        stim_cur = stim_list[s] if stim else None

        _, eigs_high_int_ab, _ = get_eigenvalues(
            model_path, 0, region, stim_inp=stim_cur
        )
        _, eigs_low_int_ab, _ = get_eigenvalues(
            model_path, 1, region, stim_inp=stim_cur
        )
        _, eigs_object_ab, _ = get_eigenvalues(model_path, 2, region, stim_inp=stim_cur)

        all_model_dists_high_int[regions[s]].append(
            eigs_high_int_ab[start_silence:end_silence]
        )
        all_model_dists_low_int[regions[s]].append(
            eigs_low_int_ab[start_silence:end_silence]
        )
        all_model_dists_object[regions[s]].append(
            eigs_object_ab[start_silence:end_silence]
        )

    for r in regions:
        high_int_eigs_arr = np.array(all_model_dists_high_int[r])
        low_int_eigs_arr = np.array(all_model_dists_low_int[r])
        object_eigs_arr = np.array(all_model_dists_object[r])

        high_int_eigs_arr = np.reshape(
            high_int_eigs_arr, (-1, high_int_eigs_arr.shape[-1])
        )
        low_int_eigs_arr = np.reshape(
            low_int_eigs_arr, (-1, low_int_eigs_arr.shape[-1])
        )
        object_eigs_arr = np.reshape(object_eigs_arr, (-1, object_eigs_arr.shape[-1]))

        all_model_dists_high_int[r] = high_int_eigs_arr
        all_model_dists_low_int[r] = low_int_eigs_arr
        all_model_dists_object[r] = object_eigs_arr

    return (
        all_model_dists_high_int,
        all_model_dists_low_int,
        all_model_dists_object,
        regions,
    )


def plot_eigs_ablation_all_models(model_path, region):
    exp_path = f"results/all_social_mrnn/linear_analysis"

    # regions stuff is trash rn
    hi_eigs_ctrl, li_eigs_ctrl, o_eigs_ctrl, regions = eigs_all_models(
        region, model_path, stim=False
    )
    hi_eigs_stim, li_eigs_stim, o_eigs_stim, _ = eigs_all_models(
        region, model_path, stim=True
    )

    def _kl_div_regions(stim_data, ctrl_data):
        dists = {}
        for r in stim_data:
            dists[r] = []
            for t in range(stim_data[r].shape[0]):
                dists[r].append(wasserstein_distance(stim_data[r][t], ctrl_data[r][t]))
        return dists

    def _make_bar_plot(kl_divs, cond):
        fig, ax = standard_2d_ax()
        ys = []
        errs = []
        for r in kl_divs:
            ys.append(np.mean(kl_divs[r]))
            errs.append(np.std(kl_divs[r]))
        ax.bar(regions, ys, yerr=errs, capsize=10)
        save_fig(
            os.path.join(exp_path, f"{region}_kl_div_ablation_cond{cond}"), eps=True
        )

    hi_kl_divs = _kl_div_regions(hi_eigs_stim, hi_eigs_ctrl)
    li_kl_divs = _kl_div_regions(li_eigs_stim, li_eigs_ctrl)
    o_kl_divs = _kl_div_regions(o_eigs_stim, o_eigs_ctrl)

    # Only doing hi for now
    pvalues(regions, hi_kl_divs)

    _make_bar_plot(hi_kl_divs, 0)
    _make_bar_plot(li_kl_divs, 1)
    _make_bar_plot(o_kl_divs, 2)
