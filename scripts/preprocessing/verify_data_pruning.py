import pickle
import numpy as np
from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.io.index_dataset import index_dataset
from dal_monte_2022_analysis.utils.paths import build_processed_data_path

cfg = load_dataset_config("configs/dataset.yaml")
index = index_dataset(cfg, "neural_timeline")
rng = np.random.default_rng()
row = index.iloc[rng.integers(0, len(index))].to_dict()
agents = cfg["agents"]

print("session:", row["session"], "date:", row["date"])
t = pickle.load(open(build_processed_data_path(cfg, row, "neural_timeline", None), "rb")).t
print("timeline len:", len(t), "nan:", np.isnan(t).sum())

for agent in agents:
    pos = pickle.load(open(build_processed_data_path(cfg, row, "gaze_position", agent), "rb"))
    pupil = pickle.load(open(build_processed_data_path(cfg, row, "pupil_size", agent), "rb"))
    print(agent, "pos len:", len(pos.x), len(pos.y), "pupil len:", len(pupil.d))
    print(agent, "pos nan:", np.isnan(pos.x).sum(), np.isnan(pos.y).sum(), "pupil nan:", np.isnan(pupil.d).sum())
