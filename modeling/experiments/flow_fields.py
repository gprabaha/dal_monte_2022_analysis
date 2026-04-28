import os
import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from mRNNTorch.analysis import flow_field
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from tqdm import tqdm
import config
from plt_utils import standard_2d_ax, empty_2d_ax
from utils import load_pickle, save_fig, get_mean_fixation_data, load_hp, interactivity_input
from models import Model

# Supress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def _plot_energy(coords, speed, save_path):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(coords[:, :, 0], coords[:, :, 1], speed[:, :], cmap=plt.cm.coolwarm,
                linewidth=0, antialiased=False)
    ax.grid(False)
    # Set custom z-ticks
    ax.set_zticks([0, 1])  # Set the z-ticks to -1, 0, and 1
    # Optionally, you can set labels for those ticks
    ax.set_zticklabels(['0', '1'])
    save_fig(save_path)
    

def _plot_flow(coords, x_vel, y_vel, save_path):

    # Add line collection
    fig, ax = empty_2d_ax()
    # Create plot
    ax.streamplot(coords[:, :, 0], 
                  coords[:, :, 1], 
                  x_vel, 
                  y_vel, 
                  color="black", 
                  linewidth=2, 
                  arrowsize=2, 
                  zorder=0,
    )

    save_fig(save_path, eps=True)

    
def _gen_line_collection(
    data,
    time_skips=10
):

    # Create line collection for plotting
    lines = []
    for t in range(0, data.shape[0]-1, time_skips):
        lines.append(np.array([(data[t, 0], 
                                data[t, 1]), 
                                (data[t, 0], 
                                data[t, 1])]))
    line_collection = LineCollection(lines, cmap='viridis', linewidths=6, zorder=5)
    # Create a scalar array that will control the color (e.g., use line's x-coordinate)
    color_values = [t*0.01 for t in range(data.shape[0])]  # mean x-value of each line
    # Set the color values for the LineCollection
    line_collection.set_array(np.array(color_values))
    # Normalize the color values
    norm = Normalize(vmin=min(color_values), vmax=max(color_values))

    return lines, color_values, line_collection, norm




def plot_flow_fields(
    model_path, 
    condition,
    *args, 
    num_points=20,
    x_offset=10,
    y_offset=10,
    cancel_other_regions=False,
    time_skips=1
    ):
    
    hp = load_hp(model_path)
    exp_path = f"results/{hp["model_save_name"]}/flow_fields"
    
    dataset = get_mean_fixation_data("/Users/lazza/naturalistic_social_gaze_mech/social_gaze") 

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
        hp["constrained"],
        hp["batch_first"],
        hp["spectral_radius"],
        output_layer=hp["output_layer"],
        device="cpu"
    ).cpu()

    checkpoint = torch.load(os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Start training
    batch, keys, _ = dataset.sample_batch()
    inp = interactivity_input(keys, batch.shape[1])
    inp = inp.cpu()
    batch = batch.cpu()

    if hp["output_layer"]:
        xn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")
        hn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")
    else:
        xn = checkpoint["xn_0"].cpu()
        hn = checkpoint["hn_0"].cpu()

    with torch.no_grad():
        out, hn = model(inp, xn, hn, noise=False)

    out = out.detach().cpu()
    hn = hn.detach().cpu()

    data_coords, x_vels, y_vels, speeds = flow_field(
        model.mrnn,
        hn[condition:condition+1],
        inp[condition:condition+1],
        *args,
        num_points=num_points,
        x_offset=x_offset,
        y_offset=y_offset,
        cancel_other_regions=cancel_other_regions,
        follow_traj=False
    )

    # Define the save paths for flow and energy
    save_path_flow = os.path.join(exp_path, f"flow_cancel_{cancel_other_regions}")
    save_path_energy = os.path.join(exp_path, f"energy_cancel_{cancel_other_regions}")

    # Get full path for both
    for region in args:
        save_path_flow = save_path_flow + f"_{region}"
        save_path_energy = save_path_energy + f"_{region}"
    save_path_flow = save_path_flow + f"_cond{condition}/"
    save_path_energy = save_path_energy + f"_cond{condition}/"

    print("Plotting flow fields...")
    for i in tqdm(range(0, len(x_vels))):
        
        # Plot and save energy landscape
        energy_save = save_path_energy + f"t{i*time_skips}_energy"
        _plot_energy(data_coords[i], speeds[i], energy_save)

        # Plot and save flow field
        flow_save = save_path_flow + f"t{i*time_skips}_flow"
        _plot_flow(data_coords[i], x_vels[i], y_vels[i], flow_save)




def plot_flow_pfc_object(model_path):
    plot_flow_fields(model_path, 2, "pfc")
def plot_flow_pfc_low_int(model_path):
    plot_flow_fields(model_path, 1, "pfc")
def plot_flow_pfc_high_int(model_path):
    plot_flow_fields(model_path, 0, "pfc")

def plot_flow_acc_object(model_path):
    plot_flow_fields(model_path, 2, "acc")
def plot_flow_acc_low_int(model_path):
    plot_flow_fields(model_path, 1, "acc")
def plot_flow_acc_high_int(model_path):
    plot_flow_fields(model_path, 0, "acc")

def plot_flow_bla_object(model_path):
    plot_flow_fields(model_path, 2, "bla")
def plot_flow_bla_low_int(model_path):
    plot_flow_fields(model_path, 1, "bla")
def plot_flow_bla_high_int(model_path):
    plot_flow_fields(model_path, 0, "bla")

def plot_flow_ofc_object(model_path):
    plot_flow_fields(model_path, 2, "ofc")
def plot_flow_ofc_low_int(model_path):
    plot_flow_fields(model_path, 1, "ofc")
def plot_flow_ofc_high_int(model_path):
    plot_flow_fields(model_path, 0, "ofc")

def plot_all_flow_fields(model_path):
    plot_flow_pfc_object(model_path)
    plot_flow_pfc_high_int(model_path)
    plot_flow_pfc_low_int(model_path)
    plot_flow_acc_object(model_path)
    plot_flow_acc_high_int(model_path)
    plot_flow_acc_low_int(model_path)
    plot_flow_bla_object(model_path)
    plot_flow_bla_high_int(model_path)
    plot_flow_bla_low_int(model_path)
    plot_flow_ofc_object(model_path)
    plot_flow_ofc_high_int(model_path)
    plot_flow_ofc_low_int(model_path)


def main():

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    
    # Principle Angles
    if args.experiment == "plot_flow_pfc_object":
        plot_flow_pfc_object(args.model_path) 
    elif args.experiment == "plot_flow_pfc_low_int":
        plot_flow_pfc_low_int(args.model_path) 
    elif args.experiment == "plot_flow_pfc_high_int":
        plot_flow_pfc_high_int(args.model_path) 

    elif args.experiment == "plot_flow_acc_object":
        plot_flow_acc_object(args.model_path) 
    elif args.experiment == "plot_flow_acc_low_int":
        plot_flow_acc_low_int(args.model_path) 
    elif args.experiment == "plot_flow_acc_high_int":
        plot_flow_acc_high_int(args.model_path) 

    elif args.experiment == "plot_flow_bla_object":
        plot_flow_bla_object(args.model_path) 
    elif args.experiment == "plot_flow_bla_low_int":
        plot_flow_bla_low_int(args.model_path) 
    elif args.experiment == "plot_flow_bla_high_int":
        plot_flow_bla_high_int(args.model_path) 

    elif args.experiment == "plot_flow_ofc_object":
        plot_flow_ofc_object(args.model_path) 
    elif args.experiment == "plot_flow_ofc_low_int":
        plot_flow_ofc_low_int(args.model_path) 
    elif args.experiment == "plot_flow_ofc_high_int":
        plot_flow_ofc_high_int(args.model_path) 
    
    elif args.experiment == "plot_all_flow_fields":
        plot_all_flow_fields(args.model_path)

    else:
        raise ValueError("Experiment not in this file")
    
if __name__ == "__main__":
    main()