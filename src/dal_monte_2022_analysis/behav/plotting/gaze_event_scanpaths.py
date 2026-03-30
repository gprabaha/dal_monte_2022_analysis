"""Plot scanpath-style gaze-event summaries for fixation/saccade QC."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from dal_monte_2022_analysis.runtime.io.gaze_event_qc import (
    AgentGazeEventArtifacts,
    DEFAULT_GAZE_EVENT_AGENTS,
    SessionKey,
    load_gaze_event_session_artifacts,
    normalize_gaze_event_agents,
    sample_random_paired_gaze_event_sessions,
)
from dal_monte_2022_analysis.data.records.behavioral import PositionData, ROIRectsData

DEFAULT_FIXATION_COLOR = "#C62828"
DEFAULT_SACCADE_COLOR = "#2E7D32"
DEFAULT_ROI_COLORS: dict[str, str] = {
    "face": "#4C78A8",
    "eyes_nf": "#72B7B2",
    "mouth": "#F58518",
    "left_nonsocial_object": "#B279A2",
    "right_nonsocial_object": "#E45756",
}
_DEFAULT_ROI_COLOR = "#6E6E6E"


def _normalize_session_rows(
    sessions: pd.DataFrame | Sequence[Mapping[str, object]],
) -> list[SessionKey]:
    if isinstance(sessions, pd.DataFrame):
        if not {"date", "session"}.issubset(sessions.columns):
            raise ValueError("Session dataframe must contain date and session columns.")
        rows = sessions[["date", "session"]].drop_duplicates().to_dict("records")
    else:
        rows = list(sessions)

    keys: list[SessionKey] = []
    for row in rows:
        if "date" not in row or "session" not in row:
            raise ValueError("Each session row must include date and session.")
        keys.append(SessionKey(date=str(row["date"]), session=str(row["session"])))
    return keys


def _clamp_interval(start: int, stop: int, n_samples: int) -> tuple[int, int] | None:
    lo = max(0, int(start))
    hi = min(int(stop), n_samples - 1)
    if lo > hi or n_samples <= 0:
        return None
    return lo, hi


def compute_fixation_centers(
    position: PositionData,
    fixations: pd.DataFrame,
) -> np.ndarray:
    """Return fixation centers as mean x/y coordinates for each fixation interval."""
    x = np.asarray(position.x, dtype=float)
    y = np.asarray(position.y, dtype=float)
    centers = np.full((len(fixations), 2), np.nan, dtype=float)
    starts = fixations["start"].to_numpy(dtype=int)
    stops = fixations["stop"].to_numpy(dtype=int)

    for idx, (start, stop) in enumerate(zip(starts, stops)):
        bounds = _clamp_interval(start, stop, len(x))
        if bounds is None:
            continue
        lo, hi = bounds
        xs = x[lo : hi + 1]
        ys = y[lo : hi + 1]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(valid):
            continue
        centers[idx, 0] = float(xs[valid].mean())
        centers[idx, 1] = float(ys[valid].mean())

    return centers


def compute_saccade_segments(
    position: PositionData,
    saccades: pd.DataFrame,
) -> np.ndarray:
    """Return start/end coordinate pairs for each saccade interval."""
    x = np.asarray(position.x, dtype=float)
    y = np.asarray(position.y, dtype=float)
    segments = np.full((len(saccades), 2, 2), np.nan, dtype=float)
    starts = saccades["start"].to_numpy(dtype=int)
    stops = saccades["stop"].to_numpy(dtype=int)

    for idx, (start, stop) in enumerate(zip(starts, stops)):
        bounds = _clamp_interval(start, stop, len(x))
        if bounds is None:
            continue
        lo, hi = bounds
        xs = x[lo : hi + 1]
        ys = y[lo : hi + 1]
        valid_idx = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
        if valid_idx.size == 0:
            continue
        first = int(valid_idx[0])
        last = int(valid_idx[-1])
        segments[idx, 0, :] = [xs[first], ys[first]]
        segments[idx, 1, :] = [xs[last], ys[last]]

    return segments


def _alpha_rgba(
    color: str,
    n_items: int,
    *,
    base_alpha: float,
    encode_event_order: bool,
) -> np.ndarray:
    rgba = np.array(mcolors.to_rgba(color), dtype=float)
    if n_items <= 0:
        return np.empty((0, 4), dtype=float)

    colors = np.tile(rgba, (n_items, 1))
    if encode_event_order and n_items > 1:
        colors[:, 3] = np.linspace(max(0.12, base_alpha * 0.3), base_alpha, n_items)
    else:
        colors[:, 3] = base_alpha
    return colors


def _roi_bounds(rois: ROIRectsData) -> np.ndarray:
    corners: list[np.ndarray] = []
    for rect in rois.rois.values():
        coords = np.asarray(rect, dtype=float).reshape(-1)
        if coords.size < 4 or not np.all(np.isfinite(coords[:4])):
            continue
        x1, y1, x2, y2 = coords[:4]
        corners.append(np.array([[x1, y1], [x2, y2]], dtype=float))
    if not corners:
        return np.empty((0, 2), dtype=float)
    return np.vstack(corners)


def resolve_scanpath_bounds(
    fixation_centers: np.ndarray,
    saccade_segments: np.ndarray,
    rois: ROIRectsData,
    *,
    margin: float = 40.0,
) -> tuple[float, float, float, float]:
    """Resolve x/y axis limits from event coordinates and ROI bounds."""
    parts: list[np.ndarray] = []
    if fixation_centers.size:
        valid_centers = fixation_centers[np.isfinite(fixation_centers).all(axis=1)]
        if valid_centers.size:
            parts.append(valid_centers)
    if saccade_segments.size:
        segment_points = saccade_segments.reshape(-1, 2)
        valid_segments = segment_points[np.isfinite(segment_points).all(axis=1)]
        if valid_segments.size:
            parts.append(valid_segments)
    roi_points = _roi_bounds(rois)
    if roi_points.size:
        parts.append(roi_points)

    if not parts:
        return (0.0, 1.0, 0.0, 1.0)

    coords = np.vstack(parts)
    min_x = float(np.min(coords[:, 0]) - margin)
    max_x = float(np.max(coords[:, 0]) + margin)
    min_y = float(np.min(coords[:, 1]) - margin)
    max_y = float(np.max(coords[:, 1]) + margin)
    return (min_x, max_x, min_y, max_y)


def _draw_rois(
    ax: Axes,
    rois: ROIRectsData,
    *,
    label_rois: bool,
    line_width: float,
) -> None:
    for name, rect in rois.rois.items():
        coords = np.asarray(rect, dtype=float).reshape(-1)
        if coords.size < 4 or not np.all(np.isfinite(coords[:4])):
            continue
        x1, y1, x2, y2 = coords[:4]
        edgecolor = DEFAULT_ROI_COLORS.get(str(name), _DEFAULT_ROI_COLOR)
        patch = Rectangle(
            (x1, y1),
            float(x2 - x1),
            float(y2 - y1),
            linewidth=line_width,
            edgecolor=edgecolor,
            facecolor="none",
            alpha=0.9,
            zorder=1,
        )
        ax.add_patch(patch)
        if label_rois:
            ax.text(
                x1,
                y1,
                str(name),
                fontsize=7,
                color=edgecolor,
                ha="left",
                va="bottom",
                zorder=4,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.0,
                },
            )


def plot_agent_gaze_event_scanpath(
    agent_data: AgentGazeEventArtifacts,
    *,
    ax: Optional[Axes] = None,
    fixation_color: str = DEFAULT_FIXATION_COLOR,
    saccade_color: str = DEFAULT_SACCADE_COLOR,
    fixation_size: float = 18.0,
    fixation_alpha: float = 0.95,
    saccade_line_width: float = 1.1,
    saccade_alpha: float = 0.4,
    roi_line_width: float = 1.4,
    label_rois: bool = True,
    encode_event_order: bool = True,
    invert_y: bool = True,
    bounds: Optional[tuple[float, float, float, float]] = None,
    show_axis_labels: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """Plot one agent's fixations and saccades over ROI boxes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))

    fixation_centers = compute_fixation_centers(agent_data.position, agent_data.fixations)
    saccade_segments = compute_saccade_segments(agent_data.position, agent_data.saccades)

    _draw_rois(
        ax,
        agent_data.rois,
        label_rois=label_rois,
        line_width=roi_line_width,
    )

    valid_segments = saccade_segments[np.isfinite(saccade_segments).all(axis=(1, 2))]
    if valid_segments.size:
        segment_colors = _alpha_rgba(
            saccade_color,
            len(valid_segments),
            base_alpha=saccade_alpha,
            encode_event_order=encode_event_order,
        )
        collection = LineCollection(
            valid_segments,
            colors=segment_colors,
            linewidths=saccade_line_width,
            capstyle="round",
            zorder=2,
        )
        ax.add_collection(collection)

    valid_fixations = fixation_centers[np.isfinite(fixation_centers).all(axis=1)]
    if valid_fixations.size:
        fixation_colors = _alpha_rgba(
            fixation_color,
            len(valid_fixations),
            base_alpha=fixation_alpha,
            encode_event_order=encode_event_order,
        )
        ax.scatter(
            valid_fixations[:, 0],
            valid_fixations[:, 1],
            s=fixation_size,
            c=fixation_colors,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    x0, x1, y0, y1 = bounds or resolve_scanpath_bounds(
        fixation_centers,
        saccade_segments,
        agent_data.rois,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    if show_axis_labels:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels, downward)")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if title is not None:
        ax.set_title(title)
    return ax


def _combine_bounds(
    bounds: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    if not bounds:
        return (0.0, 1.0, 0.0, 1.0)
    min_x = min(item[0] for item in bounds)
    max_x = max(item[1] for item in bounds)
    min_y = min(item[2] for item in bounds)
    max_y = max(item[3] for item in bounds)
    return (min_x, max_x, min_y, max_y)


def plot_gaze_event_example_sessions(
    cfg_or_path: dict | str | Path,
    sessions: pd.DataFrame | Sequence[Mapping[str, object]],
    *,
    agents: Sequence[str] = DEFAULT_GAZE_EVENT_AGENTS,
    figsize_per_panel: tuple[float, float] = (5.2, 4.3),
    label_rois: bool = True,
    encode_event_order: bool = True,
    share_screen_bounds: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot multiple sessions as a date/session by agent QC grid."""
    session_keys = _normalize_session_rows(sessions)
    if not session_keys:
        raise ValueError("No sessions were provided for plotting.")

    agent_names = normalize_gaze_event_agents(agents)
    loaded_sessions = [
        load_gaze_event_session_artifacts(
            cfg_or_path,
            date=key.date,
            session=key.session,
            agents=agent_names,
        )
        for key in session_keys
    ]

    all_bounds: list[tuple[float, float, float, float]] = []
    if share_screen_bounds:
        for session_data in loaded_sessions:
            for agent in agent_names:
                payload = session_data.agents[agent]
                all_bounds.append(
                    resolve_scanpath_bounds(
                        compute_fixation_centers(payload.position, payload.fixations),
                        compute_saccade_segments(payload.position, payload.saccades),
                        payload.rois,
                    )
                )
    shared_bounds = _combine_bounds(all_bounds) if all_bounds else None

    n_rows = len(loaded_sessions)
    n_cols = len(agent_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, session_data in enumerate(loaded_sessions):
        for col_idx, agent in enumerate(agent_names):
            payload = session_data.agents[agent]
            monkey_name = payload.position.context.monkey_name or agent
            title = (
                f"{session_data.key.date} | session {session_data.key.session} | "
                f"{agent} ({monkey_name})"
            )
            plot_agent_gaze_event_scanpath(
                payload,
                ax=axes[row_idx, col_idx],
                label_rois=label_rois,
                encode_event_order=encode_event_order,
                bounds=shared_bounds,
                show_axis_labels=(row_idx == n_rows - 1 or col_idx == 0),
                title=title,
            )
    return fig, axes


def plot_random_gaze_event_example_sessions(
    cfg_or_path: dict | str | Path,
    *,
    n_sessions: int = 5,
    random_state: Optional[int] = None,
    agents: Sequence[str] = DEFAULT_GAZE_EVENT_AGENTS,
    label_rois: bool = True,
    encode_event_order: bool = True,
    share_screen_bounds: bool = True,
) -> tuple[pd.DataFrame, Figure, np.ndarray]:
    """Sample random paired sessions and plot them as a QC grid."""
    sampled = sample_random_paired_gaze_event_sessions(
        cfg_or_path,
        n_sessions=n_sessions,
        random_state=random_state,
        agents=agents,
    )
    fig, axes = plot_gaze_event_example_sessions(
        cfg_or_path,
        sampled,
        agents=agents,
        label_rois=label_rois,
        encode_event_order=encode_event_order,
        share_screen_bounds=share_screen_bounds,
    )
    return sampled, fig, axes
