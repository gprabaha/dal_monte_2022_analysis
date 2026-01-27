import pickle
import pdb
from tqdm import tqdm

from src.config.load import load_dataset_config
from src.io.index_dataset import index_dataset
from src.io.load_mat import load_mat_from_path
from src.data.gaze_data import RecordingContext


def build_agent_dataset(
    cfg_path: str,
    modality: str,
    extractor_fn,
    agent_specific: bool = True,
):
    cfg = load_dataset_config(cfg_path)
    index = index_dataset(cfg, modality)

    out_root = cfg["processed_data_root"] / modality
    out_root.mkdir(parents=True, exist_ok=True)

    for row in tqdm(
        index.itertuples(),
        total=len(index),
        desc=f"Extracting {modality}",
        unit="file",
    ):
        mat = load_mat_from_path(row.path)

        agents = cfg["agents"] if agent_specific else [None]
        for agent in agents:
            ctx = RecordingContext(
                date=row.date,
                session=row.session,
                agent=agent,
            )

            data_obj = extractor_fn(mat, ctx)
            if data_obj is None:
                continue

            out_dir = out_root / f"date={row.date}" / f"session={row.session}"
            out_dir.mkdir(parents=True, exist_ok=True)

            suffix = f"agent={agent}" if agent else "shared"
            out_file = out_dir / f"{suffix}.pkl"

            with open(out_file, "wb") as f:
                pickle.dump(data_obj, f)
