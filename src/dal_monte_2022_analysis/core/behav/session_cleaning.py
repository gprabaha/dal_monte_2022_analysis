"""Pure behavioral session-cleaning algorithms."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from dal_monte_2022_analysis.data.records.behavioral import (
    NeuralTimelineData,
    PositionData,
    PupilSizeData,
)


def prune_timeline(timeline: NeuralTimelineData):
    """Drop NaNs from a neural timeline and return (timeline, valid indices)."""
    t = np.asarray(timeline.t)
    valid_idx = np.where(~np.isnan(t))[0]
    if valid_idx.size == 0:
        return None, None
    pruned = NeuralTimelineData(context=timeline.context, t=t[valid_idx])
    return pruned, valid_idx


def interpolate_nans(array, *, kind="linear", window_size=10, max_nans=3):
    """Fill NaNs in 1D/2D arrays with linear or sliding-window interpolation."""
    if array.ndim == 1 or kind == "linear":
        mask = np.isnan(array)
        if np.any(mask) and np.any(~mask):
            array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])
        return array

    if array.ndim == 2 and kind == "sliding":
        num_points, num_dims = array.shape
        stride = max_nans
        global_nan_mask = np.isnan(array).any(axis=1)

        for start in range(0, num_points - window_size + 1, stride):
            end = start + window_size
            window_mask = global_nan_mask[start:end]
            nan_count = int(np.sum(window_mask))

            if 0 < nan_count <= max_nans:
                window = array[start:end].copy()
                for col in range(num_dims):
                    col_vals = window[:, col]
                    valid = np.where(~np.isnan(col_vals))[0]
                    if valid.size > 1:
                        interp_func = interp1d(
                            valid,
                            col_vals[valid],
                            kind="cubic",
                            fill_value="extrapolate",
                            bounds_error=False,
                        )
                        to_fill = np.where(window_mask)[0]
                        col_vals[to_fill] = interp_func(to_fill)
                array[start:end] = window
        return array

    raise ValueError("Unsupported interpolation type or array shape.")


def interpolate_position(position: PositionData, valid_idx, *, window_size=10, max_nans=3):
    """Interpolate 2D position samples after timeline pruning."""
    x = np.asarray(position.x)[valid_idx]
    y = np.asarray(position.y)[valid_idx]
    positions = np.stack([x, y], axis=1)
    positions = interpolate_nans(
        positions,
        kind="sliding",
        window_size=window_size,
        max_nans=max_nans,
    )
    return PositionData(context=position.context, x=positions[:, 0], y=positions[:, 1])


def interpolate_pupil(pupil: PupilSizeData, valid_idx):
    """Interpolate 1D pupil samples after timeline pruning."""
    d = np.asarray(pupil.d)[valid_idx]
    d = interpolate_nans(d, kind="linear")
    return PupilSizeData(context=pupil.context, d=d)


def prune_and_interpolate_session(
    timeline: NeuralTimelineData,
    positions_by_agent: dict,
    pupils_by_agent: dict,
    *,
    window_size=10,
    max_nans=3,
):
    """Prune timeline and interpolate position/pupil for shared agents."""
    pruned_timeline, valid_idx = prune_timeline(timeline)
    if pruned_timeline is None:
        return None, None, None

    cleaned_positions = {}
    cleaned_pupils = {}
    agents = set(positions_by_agent) & set(pupils_by_agent)
    for agent in agents:
        cleaned_positions[agent] = interpolate_position(
            positions_by_agent[agent],
            valid_idx,
            window_size=window_size,
            max_nans=max_nans,
        )
        cleaned_pupils[agent] = interpolate_pupil(pupils_by_agent[agent], valid_idx)

    return pruned_timeline, cleaned_positions, cleaned_pupils


__all__ = [
    "interpolate_nans",
    "interpolate_position",
    "interpolate_pupil",
    "prune_and_interpolate_session",
    "prune_timeline",
]
