"""Path helpers for derived data products."""

from pathlib import Path
from typing import Optional, Sequence


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


def _normalize_filter(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    """Normalize optional filter values to a set (or None)."""
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(v) for v in values}


def scan_processed_data_paths(
    cfg: dict,
    modality: str,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    agents: Optional[Sequence[Optional[str]]] = None,
) -> list[dict]:
    """Scan processed-data folders and return available paths for a modality.

    Args:
        cfg: Dataset config dictionary.
        modality: Modality directory name under processed_data_root.
        dates: Optional list of date strings to include (MMDDYYYY).
        sessions: Optional list of session identifiers to include.
        agents: Optional list of agent IDs to include (e.g., "m1", "m2", or None
            for shared outputs). The string "shared" is treated as None.

    Returns:
        List of dicts with keys: date, session, agent, path.
    """
    root = Path(cfg["processed_data_root"])
    dates_filter = _normalize_filter(dates)
    sessions_filter = _normalize_filter(sessions)
    agents_filter = None
    if agents is not None:
        agents_filter = set()
        for agent in agents:
            if agent is None or str(agent).lower() == "shared":
                agents_filter.add(None)
            else:
                agents_filter.add(str(agent))

    pattern = root / "date=*" / "session=*" / modality / "*.pkl"
    rows: list[dict] = []
    for pkl_path in root.glob(str(pattern.relative_to(root))):
        parts = pkl_path.parts
        try:
            date_part = next(part for part in parts if part.startswith("date="))
            session_part = next(part for part in parts if part.startswith("session="))
        except StopIteration:
            continue

        date = date_part.split("=", 1)[1]
        session = session_part.split("=", 1)[1]

        if dates_filter is not None and date not in dates_filter:
            continue
        if sessions_filter is not None and session not in sessions_filter:
            continue

        agent = None
        stem = pkl_path.stem
        if stem.startswith("agent="):
            agent = stem.split("=", 1)[1]
        elif stem == "shared":
            agent = None

        if agents_filter is not None and agent not in agents_filter:
            continue

        rows.append({
            "date": date,
            "session": session,
            "agent": agent,
            "path": pkl_path,
        })

    return rows


def list_processed_modalities(cfg: dict) -> set[str]:
    """List modality directory names found under processed_data_root.

    Args:
        cfg: Dataset config dictionary containing processed_data_root.

    Returns:
        Set of modality directory names observed in the processed data tree.
    """
    root = Path(cfg["processed_data_root"])
    if not root.exists():
        return set()

    modalities: set[str] = set()
    for session_dir in root.glob("date=*/session=*"):
        if not session_dir.is_dir():
            continue
        for modality_dir in session_dir.iterdir():
            if modality_dir.is_dir():
                modalities.add(modality_dir.name)
    return modalities


def build_analysis_output_dir(cfg: dict, subdir: str) -> Path:
    """Build the analysis output directory for a named analysis subfolder.

    Args:
        cfg: Dataset config dictionary containing analysis_output_root.
        subdir: Analysis subfolder name.

    Returns:
        Path to the analysis output subdirectory.
    """
    root = Path(cfg.get("analysis_output_root", cfg["processed_data_root"]))
    return root / subdir
