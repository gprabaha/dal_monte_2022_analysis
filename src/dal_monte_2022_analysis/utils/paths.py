from pathlib import Path


def build_processed_out_dir(cfg, index_row, modality):
    layout = cfg["processed_data_layout"]["pattern"]
    rel_path = layout.format(
        date=index_row["date"],
        session=index_row["session"],
        modality=modality,
    )
    return Path(cfg["processed_data_root"]) / rel_path


def build_processed_data_path(cfg, index_row, modality, agent):
    out_dir = build_processed_out_dir(cfg, index_row, modality)
    suffix = f"agent={agent}" if agent else "shared"
    return out_dir / f"{suffix}.pkl"


def build_processed_output_path(cfg, index_row, modality, agent, *, output_suffix):
    output_modality = f"{modality}{output_suffix}" if output_suffix else modality
    return build_processed_data_path(cfg, index_row, output_modality, agent)
