import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import config
from utils.linear_analysis_utils import (
    plot_max_eigs,
    plot_eig_dist,
    plot_max_eigs_ablation,
    plot_eigs_ablation_all_models,
)

# Supress warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def plot_max_eigs_pfc(model_path):
    plot_max_eigs(model_path, "pfc")


def plot_max_eigs_acc(model_path):
    plot_max_eigs(model_path, "acc")


def plot_max_eigs_bla(model_path):
    plot_max_eigs(model_path, "bla")


def plot_max_eigs_ofc(model_path):
    plot_max_eigs(model_path, "ofc")


def run_all_max_eigs(model_path):
    plot_max_eigs_pfc(model_path)
    plot_max_eigs_acc(model_path)
    plot_max_eigs_bla(model_path)
    plot_max_eigs_ofc(model_path)


def plot_eigs_dist_pfc(model_path):
    plot_eig_dist(model_path, "pfc")


def plot_eigs_dist_acc(model_path):
    plot_eig_dist(model_path, "acc")


def plot_eigs_dist_bla(model_path):
    plot_eig_dist(model_path, "bla")


def plot_eigs_dist_ofc(model_path):
    plot_eig_dist(model_path, "ofc")


def run_all_eigs_dist(model_path):
    plot_eigs_dist_pfc(model_path)
    plot_eigs_dist_acc(model_path)
    plot_eigs_dist_bla(model_path)
    plot_eigs_dist_ofc(model_path)


def plot_max_eigs_pfc_ablation(model_path):
    plot_max_eigs_ablation(model_path, "pfc")


def plot_max_eigs_acc_ablation(model_path):
    plot_max_eigs_ablation(model_path, "acc")


def plot_max_eigs_bla_ablation(model_path):
    plot_max_eigs_ablation(model_path, "bla")


def plot_max_eigs_ofc_ablation(model_path):
    plot_max_eigs_ablation(model_path, "ofc")


def run_all_max_eigs_ablation(model_path):
    plot_max_eigs_pfc_ablation(model_path)
    plot_max_eigs_acc_ablation(model_path)
    plot_max_eigs_bla_ablation(model_path)
    plot_max_eigs_ofc_ablation(model_path)


def plot_eigs_ablation_all_models_pfc(model_path):
    plot_eigs_ablation_all_models(model_path, "pfc")


def plot_eigs_ablation_all_models_acc(model_path):
    plot_eigs_ablation_all_models(model_path, "acc")


def plot_eigs_ablation_all_models_bla(model_path):
    plot_eigs_ablation_all_models(model_path, "bla")


def plot_eigs_ablation_all_models_ofc(model_path):
    plot_eigs_ablation_all_models(model_path, "ofc")


def main():
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "plot_max_eigs_pfc":
        plot_max_eigs_pfc(args.model_path)
    elif args.experiment == "plot_max_eigs_acc":
        plot_max_eigs_acc(args.model_path)
    elif args.experiment == "plot_max_eigs_bla":
        plot_max_eigs_bla(args.model_path)
    elif args.experiment == "plot_max_eigs_ofc":
        plot_max_eigs_ofc(args.model_path)
    elif args.experiment == "run_all_max_eigs":
        run_all_max_eigs(args.model_path)

    elif args.experiment == "plot_eigs_dist_pfc":
        plot_eigs_dist_pfc(args.model_path)
    elif args.experiment == "plot_eigs_dist_acc":
        plot_eigs_dist_acc(args.model_path)
    elif args.experiment == "plot_eigs_dist_bla":
        plot_eigs_dist_bla(args.model_path)
    elif args.experiment == "plot_eigs_dist_ofc":
        plot_eigs_dist_ofc(args.model_path)
    elif args.experiment == "run_all_eigs_dist":
        run_all_eigs_dist(args.model_path)

    elif args.experiment == "plot_max_eigs_pfc_ablation":
        plot_max_eigs_pfc_ablation(args.model_path)
    elif args.experiment == "plot_max_eigs_acc_ablation":
        plot_max_eigs_acc_ablation(args.model_path)
    elif args.experiment == "plot_max_eigs_bla_ablation":
        plot_max_eigs_bla_ablation(args.model_path)
    elif args.experiment == "plot_max_eigs_ofc_ablation":
        plot_max_eigs_ofc_ablation(args.model_path)
    elif args.experiment == "run_all_max_eigs_ablation":
        run_all_max_eigs_ablation(args.model_path)

    elif args.experiment == "plot_eigs_ablation_all_models_pfc":
        plot_eigs_ablation_all_models_pfc(args.model_path)
    elif args.experiment == "plot_eigs_ablation_all_models_acc":
        plot_eigs_ablation_all_models_acc(args.model_path)
    elif args.experiment == "plot_eigs_ablation_all_models_bla":
        plot_eigs_ablation_all_models_bla(args.model_path)
    elif args.experiment == "plot_eigs_ablation_all_models_ofc":
        plot_eigs_ablation_all_models_ofc(args.model_path)

    else:
        raise NotImplementedError(f"Experiment {args.experiment} not implemented")


if __name__ == "__main__":
    main()

