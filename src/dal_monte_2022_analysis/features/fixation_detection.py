"""Fixation and saccade detection pipeline."""

import logging
import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.io.index_dataset import index_dataset
from dal_monte_2022_analysis.utils.fixation_utils import detect_fixations_and_saccades
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_processed_data_path

logger = logging.getLogger(__name__)


@dataclass
class FixationDetectionSettings:
    cfg_path: str
    input_modality: str = "gaze_position"
    output_fixations_modality: str = "fixations"
    output_saccades_modality: str = "saccades"


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _extract_non_nan_chunks(positions: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    non_nan_chunks = []
    start_indices = []
    n = positions.shape[0]
    valid_mask = ~np.isnan(positions).any(axis=1)
    diff = np.diff(valid_mask.astype(int))
    chunk_starts = np.where(diff == 1)[0] + 1
    chunk_ends = np.where(diff == -1)[0] + 1
    if valid_mask[0]:
        chunk_starts = np.insert(chunk_starts, 0, 0)
    if valid_mask[-1]:
        chunk_ends = np.append(chunk_ends, n)
    for start, end in zip(chunk_starts, chunk_ends):
        non_nan_chunks.append(positions[start:end])
        start_indices.append(start)
    return non_nan_chunks, start_indices


def _detect_fix_sacc_in_chunk(args: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray]:
    position_chunk, start_ind = args
    fixation_start_stop_indices, saccade_start_stop_indices = detect_fixations_and_saccades(position_chunk)
    fixation_start_stop_indices += start_ind
    saccade_start_stop_indices += start_ind
    return fixation_start_stop_indices, saccade_start_stop_indices


def _build_event_df(events: np.ndarray, date: str, session: str, agent: str, monkey_name: Optional[str]):
    return pd.DataFrame({
        "date": date,
        "session": session,
        "agent": agent,
        "monkey_name": monkey_name,
        "start": events[:, 0],
        "stop": events[:, 1],
    })


def _load_positions(cfg: dict, row: dict, agent: str, input_modality: str):
    pos_path = build_processed_data_path(cfg, row, input_modality, agent)
    if not pos_path.exists():
        return None
    return _load_pickle(pos_path)


def detect_fixations_for_row(
    settings: FixationDetectionSettings,
    row: dict,
    agent: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    cfg = load_dataset_config(settings.cfg_path)

    pos_data = _load_positions(cfg, row, agent, settings.input_modality)
    if pos_data is None:
        logger.warning("Missing positions for date=%s session=%s agent=%s", row["date"], row["session"], agent)
        return None, None

    positions = np.stack([pos_data.x, pos_data.y], axis=1)
    non_nan_chunks, chunk_start_indices = _extract_non_nan_chunks(positions)
    args = [(chunk, start) for chunk, start in zip(non_nan_chunks, chunk_start_indices)]

    results = [_detect_fix_sacc_in_chunk(arg) for arg in args]
    if not results:
        return None, None

    all_fix_start_stops = np.concatenate([r[0] for r in results], axis=0) if results else np.empty((0, 2), dtype=int)
    all_sacc_start_stops = np.concatenate([r[1] for r in results], axis=0) if results else np.empty((0, 2), dtype=int)

    monkey_name = getattr(pos_data.context, "monkey_name", None)

    fix_df = (
        _build_event_df(all_fix_start_stops, row["date"], row["session"], agent, monkey_name)
        if all_fix_start_stops.size else pd.DataFrame()
    )
    sacc_df = (
        _build_event_df(all_sacc_start_stops, row["date"], row["session"], agent, monkey_name)
        if all_sacc_start_stops.size else pd.DataFrame()
    )
    return fix_df, sacc_df


def detect_and_save_for_row(
    settings: FixationDetectionSettings,
    row: dict,
    agent: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    cfg = load_dataset_config(settings.cfg_path)
    fix_df, sacc_df = detect_fixations_for_row(settings, row, agent)
    if fix_df is None and sacc_df is None:
        return None, None
    _save_detection_outputs(
        cfg,
        row,
        agent,
        fix_df,
        sacc_df,
        settings.output_fixations_modality,
        settings.output_saccades_modality,
    )
    return fix_df, sacc_df


def _save_detection_outputs(
    cfg: dict,
    row: dict,
    agent: str,
    fix_df: Optional[pd.DataFrame],
    sacc_df: Optional[pd.DataFrame],
    output_fixations_modality: str,
    output_saccades_modality: str,
) -> None:
    if fix_df is not None:
        fix_path = build_processed_data_path(cfg, row, output_fixations_modality, agent)
        _save_pickle(fix_df, fix_path)
    if sacc_df is not None:
        sacc_path = build_processed_data_path(cfg, row, output_saccades_modality, agent)
        _save_pickle(sacc_df, sacc_path)


def _detect_and_save_worker(args):
    settings, row, agent = args
    fix_df, sacc_df = detect_and_save_for_row(settings, row, agent)
    if fix_df is None and sacc_df is None:
        return 0
    return 1


def build_tasks(
    settings: FixationDetectionSettings,
    *,
    agents: Optional[Iterable[str]] = None,
    test_single: bool = False,
) -> List[Tuple[FixationDetectionSettings, dict, str]]:
    cfg = load_dataset_config(settings.cfg_path)
    index_df = index_dataset(cfg, "gaze_position")
    rows = index_df.to_dict(orient="records")
    agents = list(agents) if agents is not None else cfg["agents"]
    tasks: List[Tuple[FixationDetectionSettings, dict, str]] = []
    for row in rows:
        for agent in agents:
            tasks.append((settings, row, agent))
    if test_single and tasks:
        return [tasks[0]]
    return tasks


def run_detection(
    settings: FixationDetectionSettings,
    *,
    agents: Optional[Iterable[str]] = None,
    use_parallel: bool = False,
    test_single: bool = False,
) -> None:
    tasks = build_tasks(settings, agents=agents, test_single=test_single)
    if not tasks:
        logger.warning("No tasks found for fixation detection.")
        return

    if not use_parallel:
        for task in tqdm(tasks, desc="Detecting fixations/saccades (serial)", unit="task"):
            _detect_and_save_worker(task)
        return

    n_proc = get_n_processes(max_procs=8)
    with Pool(processes=n_proc) as pool:
        for _ in tqdm(
            pool.imap_unordered(_detect_and_save_worker, tasks),
            total=len(tasks),
            desc=f"Detecting fixations/saccades ({n_proc} workers)",
            unit="task",
        ):
            pass


def annotate_fixation_locations(
    cfg_path: str,
    row: dict,
    agent: str,
    fix_df: pd.DataFrame,
    input_modality: str = "gaze_position",
    roi_modality: str = "roi_vertices",
) -> pd.DataFrame:
    cfg = load_dataset_config(cfg_path)
    pos_data = _load_positions(cfg, row, agent, input_modality)
    roi_path = build_processed_data_path(cfg, row, roi_modality, agent)
    if pos_data is None or not roi_path.exists():
        fix_df["location"] = None
        return fix_df

    roi_data = _load_pickle(roi_path)
    x = np.array(pos_data.x)
    y = np.array(pos_data.y)
    roi_rects = _roi_rects_to_df(roi_data)

    fix_df = fix_df.copy()
    fix_df["location"] = None
    for idx, row_fix in fix_df.iterrows():
        start, stop = int(row_fix["start"]), int(row_fix["stop"])
        mean_x = np.mean(x[start:stop + 1])
        mean_y = np.mean(y[start:stop + 1])
        location = _find_matching_rois((mean_x, mean_y), roi_rects)
        fix_df.at[idx, "location"] = location
    return fix_df


def annotate_saccade_from_to(
    cfg_path: str,
    row: dict,
    agent: str,
    sacc_df: pd.DataFrame,
    input_modality: str = "gaze_position",
    roi_modality: str = "roi_vertices",
) -> pd.DataFrame:
    cfg = load_dataset_config(cfg_path)
    pos_data = _load_positions(cfg, row, agent, input_modality)
    roi_path = build_processed_data_path(cfg, row, roi_modality, agent)
    if pos_data is None or not roi_path.exists():
        sacc_df["from"] = None
        sacc_df["to"] = None
        return sacc_df

    roi_data = _load_pickle(roi_path)
    x = np.array(pos_data.x)
    y = np.array(pos_data.y)
    roi_rects = _roi_rects_to_df(roi_data)

    sacc_df = sacc_df.copy()
    sacc_df["from"] = None
    sacc_df["to"] = None
    for idx, row_sacc in sacc_df.iterrows():
        start, stop = int(row_sacc["start"]), int(row_sacc["stop"])
        from_pos = (x[start], y[start])
        to_pos = (x[stop], y[stop])
        sacc_df.at[idx, "from"] = _find_matching_rois(from_pos, roi_rects)
        sacc_df.at[idx, "to"] = _find_matching_rois(to_pos, roi_rects)
    return sacc_df


def reconcile_fixation_saccade_label_mismatches(
    fix_df: pd.DataFrame,
    sacc_df: pd.DataFrame,
    *,
    max_gap: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fix_df = fix_df.copy()
    sacc_df = sacc_df.copy()

    if "location" not in fix_df.columns or "from" not in sacc_df.columns or "to" not in sacc_df.columns:
        return fix_df, sacc_df

    fix_starts = fix_df["start"].tolist()
    fix_stops = fix_df["stop"].tolist()
    fix_locs = fix_df["location"].tolist()
    sacc_starts = sacc_df["start"].tolist()
    sacc_stops = sacc_df["stop"].tolist()
    sacc_froms = sacc_df["from"].tolist()
    sacc_tos = sacc_df["to"].tolist()

    events = _merge_and_sort_gaze_events(fix_starts, fix_stops, sacc_starts, sacc_stops)
    for i in range(len(events) - 1):
        start1, end1, type1, index1 = events[i]
        start2, end2, type2, index2 = events[i + 1]
        if start2 - end1 > max_gap:
            continue
        if type1 == "fixation" and type2 == "saccade":
            fix_lbl = fix_locs[index1]
            sacc_from_lbl = sacc_froms[index2]
            if set(fix_lbl) != set(sacc_from_lbl):
                if "out_of_roi" in fix_lbl:
                    fix_locs[index1] = sacc_from_lbl
                elif "out_of_roi" in sacc_from_lbl:
                    sacc_froms[index2] = fix_lbl
        elif type1 == "saccade" and type2 == "fixation":
            sacc_to_lbl = sacc_tos[index1]
            fix_lbl = fix_locs[index2]
            if set(sacc_to_lbl) != set(fix_lbl):
                if "out_of_roi" in fix_lbl:
                    fix_locs[index2] = sacc_to_lbl
                elif "out_of_roi" in sacc_to_lbl:
                    sacc_tos[index1] = fix_lbl

    fix_df["location"] = fix_locs
    sacc_df["from"] = sacc_froms
    sacc_df["to"] = sacc_tos
    return fix_df, sacc_df


def _roi_rects_to_df(roi_data) -> pd.DataFrame:
    rows = []
    for name, rect in roi_data.rois.items():
        rect = np.asarray(rect).astype(float)
        if rect.size != 4:
            continue
        x1, y1, x2, y2 = rect
        rows.append({
            "roi_name": name,
            "x_min": min(x1, x2),
            "x_max": max(x1, x2),
            "y_min": min(y1, y2),
            "y_max": max(y1, y2),
        })
    return pd.DataFrame(rows)


def _find_matching_rois(position: Tuple[float, float], roi_df: pd.DataFrame):
    matching_rois = []
    for _, roi in roi_df.iterrows():
        if roi["x_min"] <= position[0] <= roi["x_max"] and roi["y_min"] <= position[1] <= roi["y_max"]:
            matching_rois.append(roi["roi_name"])
    return matching_rois if matching_rois else ["out_of_roi"]


def _merge_and_sort_gaze_events(
    fix_starts: List[int],
    fix_stops: List[int],
    sacc_starts: List[int],
    sacc_stops: List[int],
):
    events = [(s, e, "fixation", i) for i, (s, e) in enumerate(zip(fix_starts, fix_stops))]
    events += [(s, e, "saccade", i) for i, (s, e) in enumerate(zip(sacc_starts, sacc_stops))]
    events.sort(key=lambda tup: tup[0])
    return events
