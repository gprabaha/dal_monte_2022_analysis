import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from utils.pca_utils import model_pca
import config


def data_pca(model_path):
    model_pca(model_path, "data")


def output_pca(model_path):
    model_pca(model_path, "output")


def hidden_activity_pca(model_path):
    model_pca(model_path, "hidden_activity")


def main():
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "data_pca":
        data_pca(args.model_path)
    elif args.experiment == "output_pca":
        output_pca(args.model_path)
    elif args.experiment == "hidden_activity_pca":
        hidden_activity_pca(args.model_path)
    else:
        raise ValueError("Experiment not recognized")


if __name__ == "__main__":
    main()
