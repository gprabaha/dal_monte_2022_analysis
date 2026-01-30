"""Path helpers for derived data products."""

from pathlib import Path


def build_processed_out_dir(cfg, index_row, modality):
    """Build the processed output directory for a modality/date/session row.

    Args:
        cfg: Dataset config containing processed_data_root and layout pattern.
        index_row: Row dict with "date" and "session" keys.
        modality: Modality name (e.g., "gaze_position", "fixations").

    Returns:
        Path to the directory that should contain the modality outputs.
    """
    layout = cfg["processed_data_layout"]["pattern"]
    rel_path = layout.format(
        date=index_row["date"],
        session=index_row["session"],
        modality=modality,
    )
    return Path(cfg["processed_data_root"]) / rel_path


def build_processed_data_path(cfg, index_row, modality, agent):
    """Return the path to a per-agent (or shared) pickle for a modality.

    Args:
        cfg: Dataset config with processed_data_root and layout pattern.
        index_row: Row dict with "date" and "session" keys.
        modality: Modality name (e.g., "neural_timeline").
        agent: Agent ID (e.g., "m1") or None for shared outputs.

    Returns:
        Path to the pickle file for this modality/agent.
    """
    out_dir = build_processed_out_dir(cfg, index_row, modality)
    suffix = f"agent={agent}" if agent else "shared"
    return out_dir / f"{suffix}.pkl"


def build_processed_output_path(cfg, index_row, modality, agent, *, output_suffix):
    """Return the output path with a suffix applied to the modality name.

    Args:
        cfg: Dataset config with processed_data_root and layout pattern.
        index_row: Row dict with "date" and "session" keys.
        modality: Base modality name to suffix.
        agent: Agent ID or None for shared outputs.
        output_suffix: Suffix appended to the modality (e.g., "_cleaned").

    Returns:
        Path to the suffixed modality pickle.
    """
    output_modality = f"{modality}{output_suffix}" if output_suffix else modality
    return build_processed_data_path(cfg, index_row, output_modality, agent)
