"""Gaze event (fixation/saccade) detection pipeline."""

import warnings
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.data.loaders.behavioral import (
    index_behavioral_processed_data_from_cfg,
)
from dal_monte_2022_analysis.core.contracts import validate_gaze_event_frame
from dal_monte_2022_analysis.core.behav.fixation_detection import (
    FixationDetectionConfig,
    coerce_fixation_detection_config,
    detect_fixations_and_saccades,
)
from dal_monte_2022_analysis.core.behav.roi_geometry import (
    DEFAULT_ROI_ASSIGNMENT_EXPANSION_FRACTION,
    coerce_roi_expansion_fraction,
    iter_roi_rect_bounds,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_processed_pickle,
    save_processed_pickle,
)
from dal_monte_2022_analysis.runtime.execution.parallel import get_n_processes


@dataclass
class GazeEventDetectionSettings:
    """Configuration bundle for gaze event detection."""
    cfg_path: str
    input_modality: str = "gaze_position"
    output_fixations_modality: str = "fixations"
    output_saccades_modality: str = "saccades"
    roi_assignment_expansion_fraction: float = DEFAULT_ROI_ASSIGNMENT_EXPANSION_FRACTION
    fixation_detection: FixationDetectionConfig = field(default_factory=FixationDetectionConfig)
    use_parallel: bool = True
    test_single: bool = False
    agents: Optional[List[str]] = None


def build_gaze_event_detection_settings(
    cfg_path: str,
    detection_cfg: Mapping[str, Any],
) -> GazeEventDetectionSettings:
    """Build typed gaze-event settings from a YAML config mapping."""
    roi_assignment_expansion_fraction = coerce_roi_expansion_fraction(
        detection_cfg.get("roi_assignment_expansion_fraction")
    )

    return GazeEventDetectionSettings(
        cfg_path=cfg_path,
        input_modality=detection_cfg.get("input_modality", "gaze_position"),
        output_fixations_modality=detection_cfg.get("output_fixations_modality", "fixations"),
        output_saccades_modality=detection_cfg.get("output_saccades_modality", "saccades"),
        roi_assignment_expansion_fraction=roi_assignment_expansion_fraction,
        fixation_detection=coerce_fixation_detection_config(detection_cfg.get("fixation_detection")),
        use_parallel=detection_cfg.get("use_parallel", True),
        test_single=detection_cfg.get("test_single", False),
        agents=detection_cfg.get("agents"),
    )


def _extract_non_nan_chunks(positions: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    """Split positions into contiguous chunks without NaNs.

    Args:
        positions: Array of shape (N, 2) with x/y samples.

    Returns:
        Tuple of (chunks, start_indices) aligned by index.
    """
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


def _detect_events_in_chunk(
    args: Tuple[np.ndarray, int, float, FixationDetectionConfig],
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect events for a chunk and rebase indices to global offsets.

    Args:
        args: Tuple of (position_chunk, start_index, sampling_rate_hz, detection_config).

    Returns:
        Tuple of (fixation_intervals, saccade_intervals) rebased to full indices.
    """
    position_chunk, start_ind, sampling_rate_hz, detection_config = args
    print(f"Detecting fixations for chunk starting at {start_ind}\n")
    fixation_start_stop_indices, saccade_start_stop_indices = detect_fixations_and_saccades(
        position_chunk,
        sampling_rate_hz=sampling_rate_hz,
        config=detection_config,
    )
    fixation_start_stop_indices += start_ind
    saccade_start_stop_indices += start_ind
    return fixation_start_stop_indices, saccade_start_stop_indices


def _build_event_df(events: np.ndarray, date: str, session: str, agent: str, monkey_name: Optional[str]):
    """Build a standardized event DataFrame from index intervals.

    Args:
        events: Array of shape (M, 2) with start/stop indices.
        date: Session date string.
        session: Session identifier.
        agent: Agent ID string.
        monkey_name: Optional monkey name for metadata.

    Returns:
        DataFrame with event metadata and start/stop columns.
    """
    event_df = pd.DataFrame({
        "date": date,
        "session": session,
        "agent": agent,
        "monkey_name": monkey_name,
        "start": events[:, 0],
        "stop": events[:, 1],
    })
    return validate_gaze_event_frame(event_df)


def _load_positions(cfg: dict, row: dict, agent: str, input_modality: str):
    """Load positions data for one row/agent if it exists.

    Returns:
        Loaded PositionData or None if the file is missing.
    """
    try:
        return load_processed_pickle(cfg, row, input_modality, agent)
    except FileNotFoundError:
        return None


def _resolve_sampling_rate_hz(
    settings: GazeEventDetectionSettings,
    pos_data,
) -> float:
    """Resolve sampling rate from data metadata, falling back to config defaults."""
    sample_rate_hz = getattr(pos_data, "sample_rate_hz", None)
    if sample_rate_hz is None:
        context = getattr(pos_data, "context", None)
        sample_rate_hz = getattr(context, "sample_rate_hz", None)
    if sample_rate_hz is None:
        sample_rate_hz = settings.fixation_detection.default_sampling_rate_hz

    sample_rate_hz = float(sample_rate_hz)
    if sample_rate_hz <= 0:
        raise ValueError(f"Resolved sampling rate must be positive, got {sample_rate_hz}.")
    return sample_rate_hz


def detect_gaze_events_for_row(
    settings: GazeEventDetectionSettings,
    row: dict,
    agent: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Detect fixations and saccades for a single session/agent row.

    Args:
        settings: Gaze-event detection settings.
        row: Index row with date/session metadata.
        agent: Agent identifier.

    Returns:
        Tuple of (fixation_df, saccade_df). Each is None if input data missing.
    """
    cfg = load_config(settings.cfg_path)

    pos_data = _load_positions(cfg, row, agent, settings.input_modality)
    if pos_data is None:
        print(f"Missing positions for date={row['date']} session={row['session']} agent={agent}")
        return None, None

    positions = np.stack([pos_data.x, pos_data.y], axis=1)
    sampling_rate_hz = _resolve_sampling_rate_hz(settings, pos_data)
    non_nan_chunks, chunk_start_indices = _extract_non_nan_chunks(positions)
    args = [
        (chunk, start, sampling_rate_hz, settings.fixation_detection)
        for chunk, start in zip(non_nan_chunks, chunk_start_indices)
    ]
    
    if settings.use_parallel and len(args) > 1:
        n_proc = get_n_processes(max_procs=8)
        with Pool(processes=n_proc) as pool:
            results = pool.map(_detect_events_in_chunk, args)
    else:
        results = [_detect_events_in_chunk(arg) for arg in args]
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


def process_and_save_gaze_events_for_row(
    settings: GazeEventDetectionSettings,
    row: dict,
    agent: str,
    *,
    annotate: bool = True,
    reconcile: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Detect, annotate, reconcile, and persist events for one row/agent.

    Args:
        settings: Gaze-event detection settings.
        row: Index row with date/session metadata.
        agent: Agent identifier.
        annotate: Whether to add ROI-based annotations.
        reconcile: Whether to reconcile adjacent label mismatches.

    Returns:
        Tuple of (fixation_df, saccade_df), or (None, None) if missing input.
    """
    cfg = load_config(settings.cfg_path)
    fix_df, sacc_df = detect_gaze_events_for_row(settings, row, agent)
    if fix_df is None and sacc_df is None:
        return None, None
    
    if annotate:
        if fix_df is not None and not fix_df.empty:
            fix_df = annotate_fixation_locations(
                settings.cfg_path,
                row,
                agent,
                fix_df,
                input_modality=settings.input_modality,
                roi_expansion_fraction=settings.roi_assignment_expansion_fraction,
            )
        if sacc_df is not None and not sacc_df.empty:
            sacc_df = annotate_saccade_from_to(
                settings.cfg_path,
                row,
                agent,
                sacc_df,
                input_modality=settings.input_modality,
                roi_expansion_fraction=settings.roi_assignment_expansion_fraction,
            )
    
    if reconcile and fix_df is not None and sacc_df is not None:
        if not fix_df.empty and not sacc_df.empty:
            fix_df, sacc_df = reconcile_fixation_saccade_label_mismatches_until_stable(
                fix_df,
                sacc_df,
            )
    
    print(f"\n\nFinal dataframes for date={row['date']} session={row['session']} agent={agent}:\n\n")
    print(f"Fixation df head:\n{fix_df.head()}\n")
    print(f"Saccade df head:\n{sacc_df.head()}\n")
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


def detect_and_save_gaze_events_for_row(
    settings: GazeEventDetectionSettings,
    row: dict,
    agent: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Deprecated: use process_and_save_gaze_events_for_row."""
    warnings.warn(
        (
            "detect_and_save_gaze_events_for_row is deprecated; "
            "use process_and_save_gaze_events_for_row instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return process_and_save_gaze_events_for_row(settings, row, agent)


def _save_detection_outputs(
    cfg: dict,
    row: dict,
    agent: str,
    fix_df: Optional[pd.DataFrame],
    sacc_df: Optional[pd.DataFrame],
    output_fixations_modality: str,
    output_saccades_modality: str,
) -> None:
    """Persist fixation and saccade DataFrames to processed-data paths.

    Args:
        cfg: Dataset config.
        row: Index row with date/session metadata.
        agent: Agent identifier.
        fix_df: Fixation DataFrame or None.
        sacc_df: Saccade DataFrame or None.
        output_fixations_modality: Modality name for fixations.
        output_saccades_modality: Modality name for saccades.
    """
    if fix_df is not None:
        save_processed_pickle(fix_df, cfg, row, output_fixations_modality, agent)
    if sacc_df is not None:
        save_processed_pickle(sacc_df, cfg, row, output_saccades_modality, agent)


def _detect_and_save_worker(args):
    """Worker wrapper that returns 1 if any output was written."""
    settings, row, agent = args
    fix_df, sacc_df = process_and_save_gaze_events_for_row(settings, row, agent)
    if fix_df is None and sacc_df is None:
        return 0
    return 1


def build_tasks(
    settings: GazeEventDetectionSettings,
    *,
    test_single: bool = False,
) -> List[Tuple[GazeEventDetectionSettings, dict, str]]:
    """Build (settings, row, agent) tasks from processed position files.

    Args:
        settings: Gaze-event detection settings.
        test_single: Whether to keep only the first task.

    Returns:
        List of task tuples.
    """
    cfg = load_config(settings.cfg_path)
    index_df = index_behavioral_processed_data_from_cfg(cfg, settings.input_modality)
    rows = index_df.to_dict(orient="records")
    tasks: List[Tuple[GazeEventDetectionSettings, dict, str]] = []
    for row in rows:
        if row.get("agent") is None:
            continue
        tasks.append((settings, row, row["agent"]))
    if test_single and tasks:
        return [tasks[0]]
    return tasks


def run_gaze_event_detection(
    settings: GazeEventDetectionSettings,
    *,
    test_single: bool = False,
) -> None:
    """Run gaze-event detection across all tasks.

    Args:
        settings: Gaze-event detection settings.
        test_single: Whether to run only one task.
    """
    tasks = build_tasks(settings, test_single=test_single)
    if not tasks:
        print("No tasks found for gaze event detection.")
        return

    for task in tqdm(tasks, desc="Detecting gaze events (serial)", unit="task"):
        _detect_and_save_worker(task)


def annotate_fixation_locations(
    cfg_path: str,
    row: dict,
    agent: str,
    fix_df: pd.DataFrame,
    input_modality: str = "gaze_position",
    roi_modality: str = "roi_vertices",
    roi_expansion_fraction: float = DEFAULT_ROI_ASSIGNMENT_EXPANSION_FRACTION,
) -> pd.DataFrame:
    """Annotate fixation rows with ROI labels based on mean gaze position.

    Returns:
        DataFrame with an added "location" column.
    """
    cfg = load_config(cfg_path)
    pos_data = _load_positions(cfg, row, agent, input_modality)
    try:
        roi_data = load_processed_pickle(cfg, row, roi_modality, agent)
    except FileNotFoundError:
        roi_data = None
    if pos_data is None or roi_data is None:
        fix_df["location"] = None
        return fix_df

    x = np.array(pos_data.x)
    y = np.array(pos_data.y)
    roi_rects = _roi_rects_to_df(
        roi_data,
        expansion_fraction=roi_expansion_fraction,
    )

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
    roi_expansion_fraction: float = DEFAULT_ROI_ASSIGNMENT_EXPANSION_FRACTION,
) -> pd.DataFrame:
    """Annotate saccade rows with ROI labels for start/end positions.

    Returns:
        DataFrame with added "from" and "to" columns.
    """
    cfg = load_config(cfg_path)
    pos_data = _load_positions(cfg, row, agent, input_modality)
    try:
        roi_data = load_processed_pickle(cfg, row, roi_modality, agent)
    except FileNotFoundError:
        roi_data = None
    if pos_data is None or roi_data is None:
        sacc_df["from"] = None
        sacc_df["to"] = None
        return sacc_df

    x = np.array(pos_data.x)
    y = np.array(pos_data.y)
    roi_rects = _roi_rects_to_df(
        roi_data,
        expansion_fraction=roi_expansion_fraction,
    )

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
    return_changes: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile mismatched fixation/saccade ROI labels in nearby events.

    Args:
        fix_df: Fixation DataFrame with a "location" column.
        sacc_df: Saccade DataFrame with "from" and "to" columns.
        max_gap: Max gap (samples) to consider events adjacent.
        return_changes: When True, also returns a boolean for changes made.

    Returns:
        Updated fixation/saccade DataFrames (and optionally a bool when requested).
    """
    fix_df = fix_df.copy()
    sacc_df = sacc_df.copy()

    if "location" not in fix_df.columns or "from" not in sacc_df.columns or "to" not in sacc_df.columns:
        return (fix_df, sacc_df, False) if return_changes else (fix_df, sacc_df)

    fix_starts = fix_df["start"].tolist()
    fix_stops = fix_df["stop"].tolist()
    fix_locs = fix_df["location"].tolist()
    sacc_starts = sacc_df["start"].tolist()
    sacc_stops = sacc_df["stop"].tolist()
    sacc_froms = sacc_df["from"].tolist()
    sacc_tos = sacc_df["to"].tolist()

    events = _merge_and_sort_gaze_events(fix_starts, fix_stops, sacc_starts, sacc_stops)
    changes_made = False
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
                    changes_made = True
                elif "out_of_roi" in sacc_from_lbl:
                    sacc_froms[index2] = fix_lbl
                    changes_made = True
        elif type1 == "saccade" and type2 == "fixation":
            sacc_to_lbl = sacc_tos[index1]
            fix_lbl = fix_locs[index2]
            if set(sacc_to_lbl) != set(fix_lbl):
                if "out_of_roi" in fix_lbl:
                    fix_locs[index2] = sacc_to_lbl
                    changes_made = True
                elif "out_of_roi" in sacc_to_lbl:
                    sacc_tos[index1] = fix_lbl
                    changes_made = True

    fix_df["location"] = fix_locs
    sacc_df["from"] = sacc_froms
    sacc_df["to"] = sacc_tos
    return (fix_df, sacc_df, changes_made) if return_changes else (fix_df, sacc_df)


def reconcile_fixation_saccade_label_mismatches_until_stable(
    fix_df: pd.DataFrame,
    sacc_df: pd.DataFrame,
    *,
    max_gap: int = 100,
    max_iters: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Iteratively reconcile labels until no more changes (or max_iters reached)."""
    for i in range(max_iters):
        print(f"Checking fixation and saccade df mismatches for iteration number {i + 1}")
        fix_df, sacc_df, changed = reconcile_fixation_saccade_label_mismatches(
            fix_df,
            sacc_df,
            max_gap=max_gap,
            return_changes=True,
        )
        if not changed:
            print(f"No clashes found in iteration number {i + 1}")
            break
    return fix_df, sacc_df


def _roi_rects_to_df(
    roi_data,
    *,
    expansion_fraction: float = 0.0,
) -> pd.DataFrame:
    """Convert ROI rectangle data into a normalized DataFrame.

    Args:
        roi_data: ROI data object with a .rois mapping.
        expansion_fraction: Fractional width/height growth applied around
            each ROI center. For example, ``0.2`` makes the rectangle 20%
            larger in width and height, adding 10% padding on each side.

    Returns:
        DataFrame with ROI name and bounding box columns.
    """
    rows = []
    for name, bounds in iter_roi_rect_bounds(
        roi_data,
        expansion_fraction=expansion_fraction,
    ):
        x_min, x_max, y_min, y_max = bounds
        rows.append({
            "roi_name": name,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        })
    return pd.DataFrame(rows)


def _find_matching_rois(position: Tuple[float, float], roi_df: pd.DataFrame):
    """Return matching ROI names for a position (or ['out_of_roi']).

    Args:
        position: (x, y) coordinate in the same space as ROI rectangles.
        roi_df: ROI bounds DataFrame.

    Returns:
        List of ROI names, or ["out_of_roi"] if none matched.
    """
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
    """Merge fixation and saccade intervals into a sorted event list.

    Returns:
        List of (start, stop, event_type, index) tuples sorted by start.
    """
    events = [(s, e, "fixation", i) for i, (s, e) in enumerate(zip(fix_starts, fix_stops))]
    events += [(s, e, "saccade", i) for i, (s, e) in enumerate(zip(sacc_starts, sacc_stops))]
    events.sort(key=lambda tup: tup[0])
    return events
