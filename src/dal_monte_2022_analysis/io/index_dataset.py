import re
from pathlib import Path
import pandas as pd


def index_dataset(cfg: dict, modality: str) -> pd.DataFrame:
    modality_cfg = cfg["modalities"][modality]

    root = cfg["raw_data_root"] / modality_cfg["folder"]
    pattern = re.compile(modality_cfg["file_pattern"])

    rows = []

    for mat_file in root.glob("*.mat"):
        match = pattern.match(mat_file.name)
        if not match:
            continue

        rows.append({
            "date": int(match["date"]),
            "session": int(match["session"]),
            "path": mat_file,
        })

    if not rows:
        raise RuntimeError(f"No files found for modality '{modality}'")

    return pd.DataFrame(rows).sort_values(["date", "session"])
