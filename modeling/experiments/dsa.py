import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
import os
import numpy as np
from sklearn.decomposition import PCA
from utils import load_hp, get_mean_fixation_data, interactivity_input, load_pickle, save_fig
from models import Model
from DSA.DSA import DSA
import pickle
import config
from plt_utils import standard_2d_ax
import matplotlib.pyplot as plt
import seaborn as sns

def dsa_similarity_matrix(model_path):

    hp = load_hp(model_path)
    
    dataset = get_mean_fixation_data("/Users/John/naturalistic_social_gaze_mech/social_gaze") 

    # temp
    hp["inp_noise"] = 0.01
    hp["act_noise"] = 0.1

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
        device="cpu"
    )

    checkpoint = torch.load(os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth"))
    model.load_state_dict(checkpoint)

    # Start training
    batch, keys, _ = dataset.sample_batch()
    inp = interactivity_input(keys, batch.shape[1])
    inp = inp.cpu()
    batch = batch.cpu()

    trials = []
    for t in range(50):
        xn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")
        hn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")
        with torch.no_grad():
            out, hn = model(inp, xn, hn, noise=True)
            trials.append(hn)
    trials = torch.cat(trials, dim=0)

    out = out.detach().cpu()
    hn = hn.detach().cpu()

    trial_data_h = []
    trial_data_colors = ["red", "blue", "green", "purple"]        
    for region in model.mrnn.region_dict:
        region_act = model.mrnn.get_region_activity(trials, region)
        pca = PCA(n_components=12)
        reduced_act = pca.fit_transform(region_act.reshape((-1, region_act.shape[-1])))
        reduced_act = reduced_act.reshape((region_act.shape[0], region_act.shape[1], 12))
        trial_data_h.append(reduced_act)

    dsa = DSA(trial_data_h, n_delays=90, rank=120, verbose=True, score_method="euclidean", device="cpu")
    similarities = dsa.fit_score()

    dsa_data = {"similarities": similarities, "colors": trial_data_colors}

    with open(os.path.join(hp["save_dir"], "dsa_similarity.txt"), 'wb') as f:
        pickle.dump(dsa_data, f)





def dsa_similarity_matrix_all_models():

    model_paths = [
        "checkpoints/social_mrnn_0",
        "checkpoints/social_mrnn_1",
        "checkpoints/social_mrnn_2",
        "checkpoints/social_mrnn_3",
        "checkpoints/social_mrnn_4"
    ]
    
    dataset = get_mean_fixation_data("/Users/John/naturalistic_social_gaze_mech/social_gaze") 

    all_model_hs = []
    all_model_colors = []
    for model_p in model_paths:

        hp = load_hp(model_p)

        # temp
        hp["inp_noise"] = 0.01
        hp["act_noise"] = 0.1

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
            device="cpu"
        )

        checkpoint = torch.load(os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth"))
        model.load_state_dict(checkpoint)

        # Start training
        batch, keys, _ = dataset.sample_batch()
        inp = interactivity_input(keys, batch.shape[1])
        inp = inp.cpu()
        batch = batch.cpu()

        trials = []
        for t in range(50):
            xn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")
            hn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")
            with torch.no_grad():
                out, hn = model(inp, xn, hn, noise=True)
                trials.append(hn)
        trials = torch.cat(trials, dim=0)

        out = out.detach().cpu()
        hn = hn.detach().cpu()

        trial_data_h = []
        trial_data_colors = ["red", "blue", "green", "purple"]        
        for region in model.mrnn.region_dict:
            region_act = model.mrnn.get_region_activity(trials, region)
            pca = PCA(n_components=12)
            reduced_act = pca.fit_transform(region_act.reshape((-1, region_act.shape[-1])))
            reduced_act = reduced_act.reshape((region_act.shape[0], region_act.shape[1], 12))
            trial_data_h.append(reduced_act)
        
        all_model_hs.extend(trial_data_h)
        all_model_colors.extend(trial_data_colors)

    dsa = DSA(all_model_hs, n_delays=90, rank=120, verbose=True, score_method="euclidean", device="cpu")
    similarities = dsa.fit_score()

    dsa_data = {"similarities": similarities, "colors": all_model_colors}

    with open(os.path.join("checkpoints/all_social_mrnn", "dsa_similarity.txt"), 'wb') as f:
        pickle.dump(dsa_data, f)




def dsa_scatter_all_models():

    exp_path = f"results/all_social_mrnn/dsa"

    fig, ax = standard_2d_ax()

    dsa_data = load_pickle(os.path.join("checkpoints/all_social_mrnn", "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]
    colors = dsa_data["colors"]

    reduced = PCA(n_components=2).fit_transform(similarities)
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, s=250)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_scatter"))




def dsa_heatmap_all_models():

    exp_path = f"results/all_social_mrnn/dsa"

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    dsa_data = load_pickle(os.path.join("checkpoints/all_social_mrnn", "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]

    sns.heatmap(similarities, cmap="Purples")
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_similarity_vis"))



def dsa_scatter(model_path):

    hp = load_hp(model_path)
    exp_path = f"results/{hp["model_save_name"]}/dsa"

    fig, ax = standard_2d_ax()

    dsa_data = load_pickle(os.path.join(hp["save_dir"], "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]
    colors = dsa_data["colors"]

    reduced = PCA(n_components=2).fit_transform(similarities)
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, s=250)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_scatter"))




def dsa_heatmap(model_path):

    hp = load_hp(model_path)
    exp_path = f"results/{hp["model_save_name"]}/dsa"

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    dsa_data = load_pickle(os.path.join(hp["save_dir"], "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]

    sns.heatmap(similarities, cmap="Purples")
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_similarity_vis"))




def main():
    
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "dsa_similarity_matrix":
        dsa_similarity_matrix(args.model_path)
    elif args.experiment == "dsa_similarity_matrix_all_models":
        dsa_similarity_matrix_all_models()
    elif args.experiment == "dsa_scatter_all_models":
        dsa_scatter_all_models(args.model_path)
    elif args.experiment == "dsa_heatmap_all_models":
        dsa_heatmap_all_models(args.model_path)
    elif args.experiment == "dsa_scatter":
        dsa_scatter(args.model_path)
    elif args.experiment == "dsa_heatmap":
        dsa_heatmap(args.model_path)
    else:
        raise ValueError("Experiment not recognized")

if __name__ == "__main__":
    main()