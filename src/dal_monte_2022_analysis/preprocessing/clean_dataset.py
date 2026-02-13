"""Clean previously extracted data by pruning timelines and interpolating gaps."""

import pickle
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.data.cleaning import prune_and_interpolate_session
from dal_monte_2022_analysis.preprocessing.index_dataset import index_dataset
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import (
    build_processed_data_path,
    build_processed_output_path,
)


def _load_pickle(path: Path):
    """Load a pickled object from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        The unpickled object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path):
    """Serialize an object to a pickle file, creating parent directories.

    Args:
        obj: Object to serialize.
        path: Output path for the pickle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _clean_row(args):
    """Clean one session row and write outputs.

    Args:
        args: Tuple of (row, cfg, agents, output_suffix, window_size, max_nans).

    Returns:
        1 if outputs were written, otherwise 0.
    """
    row, cfg, agents, output_suffix, window_size, max_nans = args

    timeline_path = build_processed_data_path(cfg, row, "neural_timeline", None)
    if not timeline_path.exists():
        return 0

    positions_by_agent = {}
    pupils_by_agent = {}

    for agent in agents:
        pos_path = build_processed_data_path(cfg, row, "gaze_position", agent)
        pupil_path = build_processed_data_path(cfg, row, "pupil_size", agent)

        if pos_path.exists():
            positions_by_agent[agent] = _load_pickle(pos_path)
        if pupil_path.exists():
            pupils_by_agent[agent] = _load_pickle(pupil_path)

    if not positions_by_agent or not pupils_by_agent:
        return 0

    timeline = _load_pickle(timeline_path)

    cleaned_timeline, cleaned_positions, cleaned_pupils = prune_and_interpolate_session(
        timeline,
        positions_by_agent,
        pupils_by_agent,
        window_size=window_size,
        max_nans=max_nans,
    )

    if cleaned_timeline is None:
        return 0

    out_timeline = build_processed_output_path(
        cfg,
        row,
        "neural_timeline",
        None,
        output_suffix=output_suffix,
    )
    _save_pickle(cleaned_timeline, out_timeline)

    for agent, pos in cleaned_positions.items():
        out_pos = build_processed_output_path(
            cfg,
            row,
            "gaze_position",
            agent,
            output_suffix=output_suffix,
        )
        _save_pickle(pos, out_pos)

    for agent, pupil in cleaned_pupils.items():
        out_pupil = build_processed_output_path(
            cfg,
            row,
            "pupil_size",
            agent,
            output_suffix=output_suffix,
        )
        _save_pickle(pupil, out_pupil)

    return 1


def clean_dataset(
    cfg_path: str,
    *,
    output_suffix: str = "_cleaned",
    window_size: int = 10,
    max_nans: int = 3,
):
    """Clean all sessions by pruning timelines and interpolating position/pupil.

    Args:
        cfg_path: Path to dataset config YAML.
        output_suffix: Suffix appended to cleaned modality names.
        window_size: Sliding window size for interpolation.
        max_nans: Max NaNs allowed in a window for interpolation.

    Returns:
        None. Outputs are written to disk.
    """
    cfg = load_dataset_config(cfg_path)
    index = index_dataset(cfg, "neural_timeline")

    rows = index.to_dict(orient="records")
    agents = cfg["agents"]

    worker_args = [
        (row, cfg, agents, output_suffix, window_size, max_nans)
        for row in rows
    ]

    n_proc = get_n_processes(max_procs=8)

    with Pool(processes=n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(_clean_row, worker_args),
            total=len(worker_args),
            desc=f"Cleaning dataset ({n_proc} workers)",
            unit="session",
        ):
            pass
