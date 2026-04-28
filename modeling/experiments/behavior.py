import os
import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
import config
from utils.plt_utils import standard_2d_ax
from utils.exp_utils import (
    load_hp,
    get_mean_fixation_data,
    interactivity_input,
    save_fig,
    stim_inp,
    get_other_regions,
)
from models import Model
import numpy as np
from utils.exp_utils import vaf_ratio

# Supress warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def plot_subspace_ablation_pfc_hi(model_path):
    _plot_subspace_stim(model_path, "pfc", 0)


def plot_subspace_ablation_pfc_li(model_path):
    _plot_subspace_stim(model_path, "pfc", 1)


def plot_subspace_ablation_pfc_obj(model_path):
    _plot_subspace_stim(model_path, "pfc", 2)


def plot_subspace_ablation_acc_hi(model_path):
    _plot_subspace_stim(model_path, "acc", 0)


def plot_subspace_ablation_acc_li(model_path):
    _plot_subspace_stim(model_path, "acc", 1)


def plot_subspace_ablation_acc_obj(model_path):
    _plot_subspace_stim(model_path, "acc", 2)


def plot_subspace_ablation_bla_hi(model_path):
    _plot_subspace_stim(model_path, "bla", 0)


def plot_subspace_ablation_bla_li(model_path):
    _plot_subspace_stim(model_path, "bla", 1)


def plot_subspace_ablation_bla_obj(model_path):
    _plot_subspace_stim(model_path, "bla", 2)


def plot_subspace_ablation_ofc_hi(model_path):
    _plot_subspace_stim(model_path, "ofc", 0)


def plot_subspace_ablation_ofc_li(model_path):
    _plot_subspace_stim(model_path, "ofc", 1)


def plot_subspace_ablation_ofc_obj(model_path):
    _plot_subspace_stim(model_path, "ofc", 2)


def plot_all_subspace_ablation(model_path):
    plot_subspace_ablation_pfc_hi(model_path)
    plot_subspace_ablation_pfc_li(model_path)
    plot_subspace_ablation_pfc_obj(model_path)
    plot_subspace_ablation_acc_hi(model_path)
    plot_subspace_ablation_acc_li(model_path)
    plot_subspace_ablation_acc_obj(model_path)
    plot_subspace_ablation_bla_hi(model_path)
    plot_subspace_ablation_bla_li(model_path)
    plot_subspace_ablation_bla_obj(model_path)
    plot_subspace_ablation_ofc_hi(model_path)
    plot_subspace_ablation_ofc_li(model_path)
    plot_subspace_ablation_ofc_obj(model_path)


def plot_subspace_activation_pfc_hi(model_path):
    _plot_subspace_stim(model_path, "pfc", 0, stim_strength=5)


def plot_subspace_activation_pfc_li(model_path):
    _plot_subspace_stim(model_path, "pfc", 1, stim_strength=5)


def plot_subspace_activation_pfc_obj(model_path):
    _plot_subspace_stim(model_path, "pfc", 2, stim_strength=5)


def plot_subspace_activation_acc_hi(model_path):
    _plot_subspace_stim(model_path, "acc", 0, stim_strength=5)


def plot_subspace_activation_acc_li(model_path):
    _plot_subspace_stim(model_path, "acc", 1, stim_strength=5)


def plot_subspace_activation_acc_obj(model_path):
    _plot_subspace_stim(model_path, "acc", 2, stim_strength=5)


def plot_subspace_activation_bla_hi(model_path):
    _plot_subspace_stim(model_path, "bla", 0, stim_strength=5)


def plot_subspace_activation_bla_li(model_path):
    _plot_subspace_stim(model_path, "bla", 1, stim_strength=5)


def plot_subspace_activation_bla_obj(model_path):
    _plot_subspace_stim(model_path, "bla", 2, stim_strength=5)


def plot_subspace_activation_ofc_hi(model_path):
    _plot_subspace_stim(model_path, "ofc", 0, stim_strength=5)


def plot_subspace_activation_ofc_li(model_path):
    _plot_subspace_stim(model_path, "ofc", 1, stim_strength=5)


def plot_subspace_activation_ofc_obj(model_path):
    _plot_subspace_stim(model_path, "ofc", 2, stim_strength=5)


def plot_all_subspace_activation(model_path):
    plot_subspace_activation_pfc_hi(model_path)
    plot_subspace_activation_pfc_li(model_path)
    plot_subspace_activation_pfc_obj(model_path)
    plot_subspace_activation_acc_hi(model_path)
    plot_subspace_activation_acc_li(model_path)
    plot_subspace_activation_acc_obj(model_path)
    plot_subspace_activation_bla_hi(model_path)
    plot_subspace_activation_bla_li(model_path)
    plot_subspace_activation_bla_obj(model_path)
    plot_subspace_activation_ofc_hi(model_path)
    plot_subspace_activation_ofc_li(model_path)
    plot_subspace_activation_ofc_obj(model_path)


def main():
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "plot_all_subspace_ablation":
        plot_all_subspace_ablation(args.model_path)
    elif args.experiment == "plot_all_subspace_activation":
        plot_all_subspace_activation(args.model_path)
    else:
        raise ValueError("Experiment not recognized")


if __name__ == "__main__":
    main()
