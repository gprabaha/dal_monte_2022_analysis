"""Combine date-partitioned fixation PSTH averages into one sliced dataframe."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    resolve_bin_centers_from_meta,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import (
    build_analysis_output_dir,
    scan_analysis_date_paths,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
)
from dal_monte_2022_analysis.utils.filenames import ensure_filename


_PARTITION_KEYS: dict[str, tuple[str, str]] = {
    "split": ("averages_split_by_interactive_state", "split_meta"),
    "unsplit": ("averages_unsplit_by_interactive_state", "unsplit_meta"),
}


@dataclass
class FixationPSTHAverageCombinerSettings:
    """Configuration for combining date-level average PSTH files."""

    cfg_path: str
    input_subdir: str = "ephys/psth/fixation_psth_averages"
    input_filename: str = "fixations_psth_10ms.pkl"
    output_subdir: str = "ephys/psth/fixation_psth_averages"
    output_dataframe_filename: str = (
        "fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl"
    )
    output_timeline_filename: str = (
        "fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl"
    )
    window_start_s: float = -0.5
    window_stop_s: float = 0.5
    partitions: tuple[str, ...] = field(default_factory=lambda: ("split", "unsplit"))
    output_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "date",
            "unit_uuid",
            "region",
            "spike_channel",
            "recorded_agent",
            "fixation_category",
            "interactive_state",
            "is_interactive",
            "n_trials",
            "psth_mean",
            "psth_sem",
            "source_fixation_monkeys",
            "average_partition",
        )
    )


def _normalize_partitions(partitions: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in partitions:
        token = str(value).strip().lower()
        if token not in _PARTITION_KEYS:
            raise ValueError(
                f"Unsupported partition '{value}'. Expected one of: split, unsplit."
            )
        if token not in out:
            out.append(token)
    if not out:
        raise ValueError("At least one partition must be requested.")
    return tuple(out)


def _extract_average_partitions(
    obj,
    *,
    requested_partitions: Sequence[str],
) -> list[tuple[str, pd.DataFrame, dict]]:
    requested = _normalize_partitions(requested_partitions)

    if isinstance(obj, dict):
        meta = obj.get("meta", {}) or {}
        meta_dict = meta if isinstance(meta, dict) else {}
        combined_partitions: list[tuple[str, pd.DataFrame, dict]] = []
        for partition in requested:
            df_key, meta_key = _PARTITION_KEYS[partition]
            df = obj.get(df_key)
            if not isinstance(df, pd.DataFrame):
                continue
            merged_meta = dict(meta_dict)
            partition_meta = meta_dict.get(meta_key, {})
            if isinstance(partition_meta, dict):
                merged_meta.update(partition_meta)
            merged_meta["selected_partition"] = partition
            combined_partitions.append((partition, df, merged_meta))
        if combined_partitions:
            return combined_partitions

        df = obj.get("averages")
        if isinstance(df, pd.DataFrame):
            split_flag = bool(meta_dict.get("split_by_interactive_state"))
            partition = "split" if split_flag else "unsplit"
            if partition in requested:
                meta_out = dict(meta_dict)
                meta_out["selected_partition"] = partition
                return [(partition, df, meta_out)]

    if isinstance(obj, pd.DataFrame) and "unsplit" in requested:
        return [("unsplit", obj, {})]
    return []


def _validate_centers(
    centers: np.ndarray,
    *,
    path: Path,
) -> np.ndarray:
    arr = np.asarray(centers, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"No PSTH bin centers found in average file: {path}")
    if arr.size < 2:
        raise ValueError(f"Need at least two PSTH bin centers to validate spacing: {path}")
    diffs = np.diff(arr)
    if not np.all(np.isfinite(diffs)):
        raise ValueError(f"Encountered non-finite PSTH bin centers in average file: {path}")
    if not np.allclose(diffs, diffs[0], atol=1e-9, rtol=1e-6):
        raise ValueError(f"Found non-uniform PSTH bin centers in average file: {path}")
    if not np.isclose(float(diffs[0]), 0.01, atol=1e-6):
        raise ValueError(
            "Expected 10 ms PSTH bin centers, "
            f"found {float(diffs[0]) * 1000.0:.6f} ms in average file: {path}"
        )
    return arr


def _build_window_mask(
    centers_s_rel: np.ndarray,
    *,
    window_start_s: float,
    window_stop_s: float,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if window_stop_s <= window_start_s:
        raise ValueError("window_stop_s must be greater than window_start_s.")

    tol = 1e-12
    mask = (
        (centers_s_rel >= float(window_start_s) - tol)
        & (centers_s_rel <= float(window_stop_s) + tol)
    )
    selected = centers_s_rel[mask]
    if selected.size == 0:
        raise ValueError(
            "Requested PSTH window does not overlap the available bin centers "
            f"for average file: {path}"
        )
    return mask, selected


def _slice_trace(
    values: object,
    *,
    mask: np.ndarray,
    expected_size: int,
    column_name: str,
    path: Path,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != int(expected_size):
        raise ValueError(
            f"Column '{column_name}' has {arr.size} bins but expected {expected_size} "
            f"from metadata in average file: {path}"
        )
    return arr[mask]


def combine_fixation_psth_average_dataframes(
    settings: FixationPSTHAverageCombinerSettings,
    *,
    dates: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Combine date-level average PSTH files and save a sliced dataframe + timeline."""
    cfg = load_config(settings.cfg_path)
    rows = scan_analysis_date_paths(
        cfg,
        settings.input_subdir,
        filename=settings.input_filename,
        dates=dates,
    )
    if not rows:
        raise FileNotFoundError(
            "No fixation PSTH average files were found for "
            f"{settings.input_subdir}/{settings.input_filename}."
        )

    requested_partitions = _normalize_partitions(settings.partitions)
    requested_columns = tuple(str(column).strip() for column in settings.output_columns if str(column).strip())
    if not requested_columns:
        raise ValueError("At least one output column must be requested.")
    combined_frames: list[pd.DataFrame] = []
    timeline_ref: Optional[np.ndarray] = None
    n_files_used = 0
    n_partitions_used = 0

    for row in rows:
        path = Path(row["path"])
        obj = load_pickle_path(path)
        partitions = _extract_average_partitions(
            obj,
            requested_partitions=requested_partitions,
        )
        if not partitions:
            continue

        for partition_name, df, meta in partitions:
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if "psth_mean" not in df.columns or "psth_sem" not in df.columns:
                raise ValueError(
                    "Average dataframe must include both 'psth_mean' and 'psth_sem' "
                    f"columns: {path}"
                )

            centers = resolve_bin_centers_from_meta(meta)
            if centers is None:
                raise ValueError(f"Unable to resolve PSTH bin centers from average file: {path}")
            centers = _validate_centers(centers, path=path)
            mask, timeline_s_rel = _build_window_mask(
                centers,
                window_start_s=float(settings.window_start_s),
                window_stop_s=float(settings.window_stop_s),
                path=path,
            )
            if timeline_ref is None:
                timeline_ref = timeline_s_rel
            elif (
                timeline_s_rel.shape != timeline_ref.shape
                or not np.allclose(timeline_s_rel, timeline_ref, atol=1e-12, rtol=1e-9)
            ):
                raise ValueError(
                    "Encountered inconsistent sliced PSTH timelines while combining "
                    f"average files. Offending file: {path}"
                )

            frame = df.copy()
            frame["average_partition"] = partition_name
            frame["source_average_path"] = str(path)
            if "date" not in frame.columns:
                frame["date"] = str(row["date"])
            frame["psth_mean"] = frame["psth_mean"].map(
                lambda vals: _slice_trace(
                    vals,
                    mask=mask,
                    expected_size=centers.size,
                    column_name="psth_mean",
                    path=path,
                )
            )
            frame["psth_sem"] = frame["psth_sem"].map(
                lambda vals: _slice_trace(
                    vals,
                    mask=mask,
                    expected_size=centers.size,
                    column_name="psth_sem",
                    path=path,
                )
            )
            keep_columns = [column for column in requested_columns if column in frame.columns]
            frame = frame.loc[:, keep_columns].copy()
            combined_frames.append(frame)
            n_partitions_used += 1

        n_files_used += 1

    if not combined_frames or timeline_ref is None:
        raise ValueError(
            "Found fixation PSTH average files, but none produced non-empty partitions "
            "for the requested selection."
        )

    combined_df = pd.concat(combined_frames, ignore_index=True, sort=False)
    sort_columns = [
        column
        for column in (
            "date",
            "average_partition",
            "unit_uuid",
            "fixation_category",
            "interactive_state",
        )
        if column in combined_df.columns
    ]
    if sort_columns:
        combined_df = combined_df.sort_values(sort_columns, kind="stable").reset_index(
            drop=True
        )

    output_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    dataframe_path = output_dir / ensure_filename(
        settings.output_dataframe_filename,
        ".pkl",
    )
    timeline_path = output_dir / ensure_filename(
        settings.output_timeline_filename,
        ".pkl",
    )
    save_pickle_path(combined_df, dataframe_path)
    save_pickle_path(np.asarray(timeline_ref, dtype=float), timeline_path)

    return {
        "dataframe": combined_df,
        "timeline_s_rel": np.asarray(timeline_ref, dtype=float),
        "dataframe_path": str(dataframe_path),
        "timeline_path": str(timeline_path),
        "n_dates_scanned": len(rows),
        "n_files_used": int(n_files_used),
        "n_partitions_used": int(n_partitions_used),
        "n_rows": int(len(combined_df)),
        "window_start_s": float(settings.window_start_s),
        "window_stop_s": float(settings.window_stop_s),
        "partitions": requested_partitions,
    }
