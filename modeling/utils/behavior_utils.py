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
)
from utils.train_utils import get_mean_fixation_data
from models import Model
from utils.exp_utils import vaf_ratio

# Supress warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def _get_ablation_stim(
    model_path, region, start_silence=50, end_silence=75, stim_strength=-5
):
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

    stim = stim_inp(
        model.mrnn,
        start_silence,
        end_silence,
        inp.shape[1],
        0,
        stim_strength,
        inp.shape[0],
        5,
        10,
        region,
    )

    return stim


def _plot_subspace_stim(model_path, region, condition, stim_strength=-5):
    hp = load_hp(model_path)
    exp_path = f"results/{hp['model_save_name']}/behavior"

    _, out_ctrl_hi = _get_act(model_path, 0)
    _, out_ctrl_li = _get_act(model_path, 1)
    _, out_ctrl_obj = _get_act(model_path, 2)

    stim = _get_ablation_stim(model_path, region, stim_strength=stim_strength)
    _, out_stim = _get_act(model_path, condition, stim_inp=stim)

    vaf_hi = vaf_ratio(out_ctrl_hi.numpy(), out_stim.numpy(), control=False)
    vaf_li = vaf_ratio(out_ctrl_li.numpy(), out_stim.numpy(), control=False)
    vaf_obj = vaf_ratio(out_ctrl_obj.numpy(), out_stim.numpy(), control=False)

    x = ["hi", "li", "obj"]
    fig, ax = standard_2d_ax()
    ax.bar(x, [vaf_hi, vaf_li, vaf_obj])
    save_fig(
        os.path.join(exp_path, f"{region}_subspace_stim{stim_strength}_{condition}"),
        eps=True,
    )
