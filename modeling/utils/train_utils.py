import sys
from pathlib import Path
import os

project_root = Path(__file__).resolve().parents[1]
parent_dir = os.path.abspath(os.getcwd())

sys.path.insert(0, str(project_root))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import os
import curate_data
from utils.dataset import MeanFixationDataset
import load_data
import json

import sys
from pathlib import Path

from utils.models import Model
from utils.exp_utils import (
    create_dir,
)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

DEF_HP = {
    "inp_dim": 3,
    "epochs": 25000,
    "lr": 1e-3,
    "dt": 10,
    "tau": 100,
    "l1_weight": 1e-4,
    "l1_rate": 1e-4,
    "rec_constrained": False,
    "inp_constrained": False,
    "batch_first": True,
    "seed": 123456,
    "tol": 1e-3,
    "patience": 50,
    "inp_noise": 0,
    "act_noise": 0,
    "spectral_radius": 1.3,
    "output_layer": True,
    "latent_training": True,
    "n_components": 10,
    "save_dir": "checkpoints/social_mrnn",
    "model_save_name": "social_mrnn",
    "model_specifications_path": "checkpoints/model_specifications",
    "mrnn_config_file": "modeling/configurations/mRNN.json",
}


def train(hp=None):
    def_hp = DEF_HP
    if hp is not None:
        def_hp.update(hp)
    hp = def_hp

    # create model path for saving model and hp
    create_dir(hp["save_dir"])
    save_hp(hp, hp["save_dir"])

    # Load processed dataframe
    dataset = get_mean_fixation_data("~/naturalistic_social_gaze_mech/social_gaze")

    # Training variables
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
    ).cuda()

    xn_0 = nn.Parameter(
        torch.zeros(size=(1, model.mrnn.total_num_units), device="cuda")
    )
    hn_0 = nn.Parameter(
        torch.zeros(size=(1, model.mrnn.total_num_units), device="cuda")
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam([*model.parameters(), xn_0, hn_0], lr=hp["lr"])
    cur_loss = 0
    losses = []

    # Start training
    for epoch in range(hp["epochs"]):
        batch, loss_mask = dataset.sample_batch(
            latent_training=hp["latent_training"], n_components=hp["n_components"]
        )
        inp = interactivity_input(dataset.group_by_columns, batch.shape[1])

        # Put to device
        batch = batch.cuda()
        inp = inp.cuda()
        loss_mask = loss_mask.cuda()

        out, hn = model(inp, xn_0, hn_0)

        out = out * loss_mask

        # Compute all losses
        mse_loss = criterion(out, batch)
        rate_loss = l1_rate(hn, hp["l1_rate"])
        weight_loss = l1_weight(model.mrnn, hp["l1_weight"])
        loss = mse_loss + rate_loss + weight_loss
        cur_loss += loss.item()

        # Training stats
        if epoch % 100 == 0 and epoch > 0:
            # Get the directory part of the save path
            directory = os.path.dirname(hp["save_dir"])
            # Check if the directory exists, and create it if it doesn't
            if not os.path.exists(directory):
                os.makedirs(directory)

            state_dict = {
                "model_state_dict": model.state_dict(),
                "xn_0": xn_0,
                "hn_0": hn_0,
            }

            torch.save(
                state_dict, os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth")
            )

            mean_loss = cur_loss / 100
            losses.append(mean_loss)

            with open(hp["save_dir"] + "losses.txt", "w") as output:
                output.write(str(losses))

            cur_loss = 0

            print("Mean training loss at epoch {}:{}".format(epoch, mean_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def initialize_params(
    remake_firing_rate_df=False,
    neural_data_bin_size=10,
    smooth_spike_counts=True,
    guassian_smoothing_sigma=2,
    time_window_before_event=500,
    is_cluster=False,
    path_name=None,
):
    params = {
        "remake_firing_rate_df": remake_firing_rate_df,
        "neural_data_bin_size": neural_data_bin_size,  # 10 ms in seconds
        "smooth_spike_counts": smooth_spike_counts,
        "gaussian_smoothing_sigma": guassian_smoothing_sigma,
        "time_window_before_event": time_window_before_event,
        "is_cluster": is_cluster,
        "path_name": path_name,
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    return params


def reduce_padding(act, keys):
    # Used to compare with padded vectors
    zero_act = torch.zeros(size=(act.shape[-1],))
    # Ensure we can still access the correct condition
    saccade_act = {}
    fixation_act = {}

    # Loop through activity and get rid of padding, add to dictionaries
    # Keys and act should be same length (batch dim)
    for key, cond in zip(keys, act):
        non_padded_act = []
        for act_t in cond:
            if torch.equal(act_t, zero_act):
                continue
            else:
                non_padded_act.append(act_t)
        non_padded_act = torch.stack(non_padded_act)
        if key[0] == "saccade":
            saccade_act[key] = non_padded_act
        elif key[0] == "fixation":
            fixation_act[key] = non_padded_act
    return saccade_act, fixation_act


def get_mean_fixation_data(path):
    # "/Users/John/naturalistic_social_gaze_mech/social_gaze"
    # Load processed dataframe
    params = initialize_params(path_name=path)
    behav_firing_rate_df_file_path = os.path.join(
        params["processed_data_dir"], "mean_fixation_response_df.pkl"
    )

    print("loading data...")
    df = load_data.get_data_df(behav_firing_rate_df_file_path)
    print("creating fr dataset...")
    dataset = MeanFixationDataset(df)

    return dataset


def interactivity_input(key, timesteps):
    input_series = []
    for cond in key:
        inp = torch.zeros(size=(1, timesteps, 3))
        if cond == "high_interactivity_face":
            inp[..., 0] = 1
        if cond == "low_interactivity_face":
            inp[..., 1] = 1
        if cond == "object":
            inp[..., 2] = 1
        input_series.append(inp)
    input_series = torch.cat(input_series, dim=0)
    return input_series


def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    with open(os.path.join(model_dir, "hp.json"), "w") as f:
        json.dump(hp_copy, f)


def l1_weight(rnn, scale):
    l1 = 0
    for name, param in rnn.named_parameters():
        l1 += torch.mean(torch.abs(torch.flatten(param)))
    l1 *= scale
    return l1


def l1_rate(act, scale):
    l1 = scale * torch.mean(torch.abs(torch.flatten(act)))
    return l1
