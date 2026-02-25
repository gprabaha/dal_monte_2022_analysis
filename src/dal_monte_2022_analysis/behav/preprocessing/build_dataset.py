"""Dataset construction utilities that run extractors and persist results."""

import pickle
from multiprocessing import Pool
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.behav.preprocessing.index_dataset import index_dataset
from dal_monte_2022_analysis.behav.preprocessing.load_mat import load_mat_from_path
from dal_monte_2022_analysis.data.behavioral_data import RecordingContext
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_processed_out_dir


def _extract_and_save_row_data(args):
    """Load one .mat file, extract modality data, and persist per-agent outputs.

    Args:
        args: Tuple of (row, cfg, modality, extractor_fn, agent_specific).

    Returns:
        1 to indicate the row was processed.
    """
    row, cfg, modality, extractor_fn, agent_specific = args

    mat = load_mat_from_path(row["path"])

    agents = cfg["agents"] if agent_specific else [None]

    for agent in agents:
        monkey_name = None
        if agent is not None:
            monkey_key = f"monkey_name_{agent.lower()}"
            if monkey_key in row:
                monkey_name = row[monkey_key]

        ctx = RecordingContext(
            date=row["date"],
            session=row["session"],
            agent=agent,
            monkey_name=monkey_name,
        )

        data_obj = extractor_fn(mat, ctx)
        if data_obj is None:
            continue

        out_dir = build_processed_out_dir(cfg, row, modality)
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"agent={agent}" if agent else "shared"
        out_file = out_dir / f"{suffix}.pkl"

        with open(out_file, "wb") as f:
            pickle.dump(data_obj, f)

    return 1


def build_agent_dataset(
    cfg_path: str,
    modality: str,
    extractor_fn,
    agent_specific: bool = True,
    use_parallel: bool = False,
):
    """Index raw data, apply an extractor, and write per-session outputs.

    Args:
        cfg_path: Path to dataset config YAML.
        modality: Modality key to process.
        extractor_fn: Callable that returns a data object or None.
        agent_specific: Whether modality is stored per agent.
        use_parallel: Whether to use multiprocessing.

    Returns:
        None. Outputs are written to disk.
    """
    cfg = load_dataset_config(cfg_path)
    index = index_dataset(cfg, modality)

    out_root = cfg["processed_data_root"]
    out_root.mkdir(parents=True, exist_ok=True)

    rows = index.to_dict(orient="records")
    worker_args = [
        (row, cfg, modality, extractor_fn, agent_specific)
        for row in rows
    ]

    if not use_parallel:
        for args in tqdm(
            worker_args,
            desc=f"Extracting {modality} (serial)",
            unit="file",
        ):
            _extract_and_save_row_data(args)
        return

    n_proc = get_n_processes(max_procs=8)
    with Pool(processes=n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(_extract_and_save_row_data, worker_args),
            total=len(worker_args),
            desc=f"Extracting {modality} ({n_proc} workers)",
            unit="file",
        ):
            pass
