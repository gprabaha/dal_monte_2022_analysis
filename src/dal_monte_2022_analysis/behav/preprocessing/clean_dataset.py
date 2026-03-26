"""Clean previously extracted data by pruning timelines and interpolating gaps."""

from dal_monte_2022_analysis.config.load import load_config, resolve_dataset_cfg_path
from dal_monte_2022_analysis.core.behav.session_cleaning import prune_and_interpolate_session
from dal_monte_2022_analysis.data.loaders.behavioral import (
    index_behavioral_processed_data_from_cfg,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_processed_pickle,
    save_processed_variant_pickle,
)
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks


def _clean_row(args):
    """Clean one session row and write outputs.

    Args:
        args: Tuple of (row, cfg, agents, output_suffix, window_size, max_nans).

    Returns:
        1 if outputs were written, otherwise 0.
    """
    row, cfg, agents, output_suffix, window_size, max_nans = args

    try:
        timeline = load_processed_pickle(cfg, row, "neural_timeline", None)
    except FileNotFoundError:
        return 0

    positions_by_agent = {}
    pupils_by_agent = {}

    for agent in agents:
        try:
            positions_by_agent[agent] = load_processed_pickle(cfg, row, "gaze_position", agent)
        except FileNotFoundError:
            pass
        try:
            pupils_by_agent[agent] = load_processed_pickle(cfg, row, "pupil_size", agent)
        except FileNotFoundError:
            pass

    if not positions_by_agent or not pupils_by_agent:
        return 0

    cleaned_timeline, cleaned_positions, cleaned_pupils = prune_and_interpolate_session(
        timeline,
        positions_by_agent,
        pupils_by_agent,
        window_size=window_size,
        max_nans=max_nans,
    )

    if cleaned_timeline is None:
        return 0

    save_processed_variant_pickle(
        cleaned_timeline,
        cfg,
        row,
        "neural_timeline",
        None,
        output_suffix=output_suffix,
    )

    for agent, pos in cleaned_positions.items():
        save_processed_variant_pickle(
            pos,
            cfg,
            row,
            "gaze_position",
            agent,
            output_suffix=output_suffix,
        )

    for agent, pupil in cleaned_pupils.items():
        save_processed_variant_pickle(
            pupil,
            cfg,
            row,
            "pupil_size",
            agent,
            output_suffix=output_suffix,
        )

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
    dataset_cfg_path = resolve_dataset_cfg_path(cfg_path)
    cfg = load_config(dataset_cfg_path)
    index = index_behavioral_processed_data_from_cfg(cfg, "neural_timeline", agents=[None])
    rows = index[["date", "session"]].drop_duplicates().to_dict(orient="records")
    agents = cfg["agents"]

    worker_args = [
        (row, cfg, agents, output_suffix, window_size, max_nans)
        for row in rows
    ]

    run_tasks(
        _clean_row,
        worker_args,
        desc="Cleaning dataset",
        unit="session",
        use_parallel=True,
        max_procs=8,
    )
