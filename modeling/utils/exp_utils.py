import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(1, "..")

import os
import matplotlib.pyplot as plt
import torch
import json
import pickle
from matplotlib import rcParams
import numpy as np
from sklearn.decomposition import PCA
import itertools
import scipy
from modeling.utils.models import Model

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def control_trial(model_path, condition, stim_inp=None):
    hp = load_hp(model_path)

    dataset = get_mean_fixation_data(
        "/Users/lazza/naturalistic_social_gaze_mech/social_gaze"
    )

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
        hp["constrained"],
        hp["batch_first"],
        hp["spectral_radius"],
        device="cpu",
    )

    checkpoint = torch.load(
        os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth")
    )
    model.load_state_dict(checkpoint)
    model.eval()

    batch, keys, _ = dataset.sample_batch()
    inp = interactivity_input(keys, batch.shape[1])

    xn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")
    hn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")

    with torch.no_grad():
        if stim_inp is not None:
            out, hn = model(inp, xn, hn, stim_inp, noise=False)
        else:
            out, hn = model(inp, xn, hn, noise=False)

    out = out.detach().cpu()
    hn = hn.detach().cpu()

    hn_cond = hn[condition]
    out_cond = out[condition]

    # Should be (time, units)
    return hn_cond, out_cond


def create_dir(save_path):
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, "hp.json")
    with open(fname, "r") as f:
        hp = json.load(f)
    return hp


def load_model(hp, dataset):
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

    return model


def initial_state(hp, model, batch_size):
    checkpoint = torch.load(
        os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth")
    )
    if hp["output_layer"]:
        xn = torch.zeros(size=(batch_size, model.mrnn.total_num_units))
        hn = torch.zeros(size=(batch_size, model.mrnn.total_num_units))
    else:
        xn = checkpoint["xn_0"].cpu()
        hn = checkpoint["hn_0"].cpu()
    return xn, hn


def save_fig(save_path, eps=False):
    # Tell matplotlib to embed fonts as text, not outlines
    rcParams["pdf.fonttype"] = 42  # 42 = TrueType (editable in Illustrator)
    rcParams["ps.fonttype"] = 42
    # Simple function to save figure while creating dir and closing
    dir = os.path.dirname(save_path)
    create_dir(dir)
    if eps:
        plt.savefig(save_path + ".pdf", format="pdf")
    else:
        plt.savefig(save_path + ".png")
    plt.close()


def load_pickle(file):
    try:
        with open(file, "rb") as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", file, ":", e)
        raise
    return data


def stim_inp(
    mrnn,
    start_silence,
    end_silence,
    seq_len,
    extra_steps,
    stim_strength,
    batch_size,
    n_steps_rampup,
    n_steps_rampdown,
    *args,
):
    """
    Get inhibitory or excitatory stimulus for optogenetic replication
    Function will gather the mask for the specified region and cell type then make a stimulus targeting these regions

    Returns:
        rnn:                        mRNN to silence
        regions_cell_types:         List of tuples specifying the region and corresponding cell type to get a mask for
        start_silence:              inteer index of when to start perturbations in the sequence
        end_silence:                integer index of when to stop perturbations in the sequence
        max_seq_len:                max sequence length
        extra_steps:                Number of extra steps to add to the sequence if necessary
        stim_strength:              Floating point value that specifies how strong the perturbation is (- or +)
        batch_size:                 Number of conditions to be included in the sequence
    """

    mask = torch.zeros(size=(mrnn.total_num_units,), device=mrnn.device)
    for region in args:
        mask = mask + mrnn.region_mask_dict[region]

    total_stim_time = (end_silence - start_silence) - n_steps_rampup - n_steps_rampdown
    # Inhibitory/excitatory stimulus to network, designed as an input current
    # It applies the inhibitory stimulus to all of the conditions specified in data (or max_seq_len) equally
    stim_pre = torch.zeros(
        size=(batch_size, start_silence, mrnn.total_num_units), device=mrnn.device
    )
    if n_steps_rampup > 0:
        stim_ramp_up = (
            torch.linspace(0, stim_strength, n_steps_rampup)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(mrnn.device)
        )
        stim_ramp_up = stim_ramp_up.repeat(batch_size, 1, mrnn.total_num_units) * mask
    stim_const = (
        torch.ones(
            size=(batch_size, total_stim_time, mrnn.total_num_units), device=mrnn.device
        )
        * mask
        * stim_strength
    )
    if n_steps_rampdown > 0:
        stim_ramp_down = (
            torch.linspace(stim_strength, 0, n_steps_rampdown)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(mrnn.device)
        )
        stim_ramp_down = (
            stim_ramp_down.repeat(batch_size, 1, mrnn.total_num_units) * mask
        )
    stim_post = torch.zeros(
        size=(
            batch_size,
            (seq_len - start_silence) + extra_steps,
            mrnn.total_num_units,
        ),
        device=mrnn.device,
    )
    stim = torch.cat(
        [stim_pre, stim_ramp_up, stim_const, stim_ramp_down, stim_post], dim=1
    )

    return stim


def get_other_regions(model, region):
    regions = []
    for r in model.mrnn.region_dict:
        if r != region:
            regions.append(r)
    return regions


def vaf_ratio(data_1, data_2, basline_dim=None, num_comps=None, control=True):
    # Only use two muscle PCs for this task, but use three for the one above

    num_comps = 12 if num_comps is None else num_comps
    percentile = 90

    if control:
        baseline_dim = data_1.shape[-1] if basline_dim == None else baseline_dim
        # Create a random manifold as a control
        random_matrices = np.random.randn(5000, baseline_dim, num_comps)
        random_bases = np.empty(shape=(5000, num_comps, baseline_dim))
        for basis in range(5000):
            q, _ = np.linalg.qr(random_matrices[basis])
            random_bases[basis] = q.T

    pca1 = PCA()
    pca2 = PCA()

    task1_data = data_1.reshape((-1, data_1.shape[-1]))
    task2_data = data_2.reshape((-1, data_2.shape[-1]))

    pca1.fit(task1_data)
    pca2.fit(task2_data)

    pca1_comps = pca1.components_[:num_comps]
    pca2_comps = pca2.components_[:num_comps]

    # ------------------------------------ TRUE ACROSS AND WITHIN TASK VAFs

    # Get VAF
    across_task_vaf_task1 = (pca2_comps @ task1_data.T).T.var(
        axis=0
    ).sum() / task1_data.var(axis=0).sum()
    within_task_vaf_task1 = (pca1_comps @ task1_data.T).T.var(
        axis=0
    ).sum() / task1_data.var(axis=0).sum()
    ratio = across_task_vaf_task1 / within_task_vaf_task1

    if control:
        # ------------------------------------ CONTROL ACROSS TASK VAFs

        # Get random VAFs, only for data_1 rn
        across_task_vaf = (random_bases @ task1_data.T).var(axis=2).sum(
            axis=1
        ) / task1_data.var(axis=0).sum()
        vaf_ratio_control = (
            np.percentile(across_task_vaf, percentile) / within_task_vaf_task1
        )

    if control:
        return ratio, vaf_ratio_control
    else:
        return ratio


def pvalues(label_list, data_dict):
    combination_labels = list(itertools.combinations(label_list, 2))
    print("\n")
    # Print out significance here
    for combination in combination_labels:
        result = scipy.stats.mannwhitneyu(
            data_dict[combination[0]], data_dict[combination[1]]
        )
        pvalue = result[1]
        if pvalue < 0.001:
            pvalue_str = f"***, {pvalue}"
        elif pvalue < 0.01:
            pvalue_str = f"**, {pvalue}"
        elif pvalue < 0.05:
            pvalue_str = f"*, {pvalue}"
        else:
            pvalue_str = "Not Significant"
        print(f"pvalue for {combination[0]} and {combination[1]} is: {pvalue_str}")
    print("\n")


def pca_batched(act, batch_first=True, n_components=3):
    """
    Performs pca on 3d act

    Args:
        act: 3d tensor of shape [BxTxN]
    """
    if batch_first:
        batch_shape, time_shape, h_shape = act.shape
    else:
        time_shape, batch_shape, h_shape = act.shape

    h_pca = PCA(n_components=n_components)
    act_reshaped = act.reshape((-1, h_shape))
    reduced_acts = h_pca.fit_transform(act_reshaped)

    if batch_first:
        reduced_acts = reduced_acts.reshape((batch_shape, time_shape, n_components))
    else:
        reduced_acts = reduced_acts.reshape((time_shape, batch_shape, n_components))

    return reduced_acts
