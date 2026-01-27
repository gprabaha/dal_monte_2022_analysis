from pathlib import Path


def build_processed_out_dir(cfg, index_row, modality):
    layout = cfg["processed_data_layout"]["pattern"]
    rel_path = layout.format(
        date=index_row["date"],
        session=index_row["session"],
        modality=modality,
    )
    return Path(cfg["processed_data_root"]) / rel_path
