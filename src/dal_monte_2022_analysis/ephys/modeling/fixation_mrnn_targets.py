"""Minimal fixation mRNN target construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_bridge import (
    MRNN_CONDITION_COLUMN_ORDER,
    MRNN_REGION_ORDER,
    build_mrnn_training_dataframe,
    load_combined_fixation_psth,
)

CONDITION_ORDER = ("face_interactive", "face_non_interactive", "object")
CONDITION_TO_COLUMN = {
    "face_interactive": "high_interactivity_face",
    "face_non_interactive": "low_interactivity_face",
    "object": "object",
}
TARGET_MODE_ALIASES = {
    "raw": "raw_fr",
    "raw_fr": "raw_fr",
    "fr": "raw_fr",
    "pcs": "region_pcs",
    "pc": "region_pcs",
    "region_pcs": "region_pcs",
    "fr_pcs": "region_pcs",
}


@dataclass(frozen=True)
class PCAMetadata:
    """PCA parameters for one region."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    source_features: tuple[str, ...]


@dataclass(frozen=True)
class FixationMRNNTargets:
    """Training targets for one fixation mRNN dataset."""

    condition_order: tuple[str, ...]
    region_order: tuple[str, ...]
    timeline_s: np.ndarray
    input_tensor: np.ndarray
    raw_by_region: dict[str, np.ndarray]
    pcs_by_region: dict[str, np.ndarray]
    raw_features_by_region: dict[str, tuple[str, ...]]
    pc_features_by_region: dict[str, tuple[str, ...]]
    pca_by_region: dict[str, PCAMetadata]
    training_dataframe: pd.DataFrame
    normalization_scale: float | None
    dataframe_path: Path | None = None
    timeline_path: Path | None = None

    def targets_for_mode(self, target_mode: str) -> dict[str, np.ndarray]:
        mode = normalize_target_mode(target_mode)
        return self.raw_by_region if mode == "raw_fr" else self.pcs_by_region

    def features_for_mode(self, target_mode: str) -> dict[str, tuple[str, ...]]:
        mode = normalize_target_mode(target_mode)
        return self.raw_features_by_region if mode == "raw_fr" else self.pc_features_by_region

    def output_dims_for_mode(self, target_mode: str) -> dict[str, int]:
        return {
            region: int(values.shape[-1])
            for region, values in self.targets_for_mode(target_mode).items()
        }


def normalize_target_mode(target_mode: str) -> str:
    """Normalize target mode aliases."""
    token = str(target_mode).strip().lower()
    try:
        return TARGET_MODE_ALIASES[token]
    except KeyError as exc:
        raise ValueError(f"Unsupported target_mode: {target_mode!r}") from exc


def build_condition_input(
    condition_order: tuple[str, ...],
    *,
    timesteps: int,
) -> np.ndarray:
    """Return condition x time x one-hot input."""
    channel = {condition: idx for idx, condition in enumerate(CONDITION_ORDER)}
    out = np.zeros((len(condition_order), int(timesteps), len(CONDITION_ORDER)), dtype=np.float32)
    for cond_idx, condition in enumerate(condition_order):
        out[cond_idx, :, channel[condition]] = 1.0
    return out


def global_robust_scale(values: np.ndarray, *, stabilizer: float = 5.0) -> float:
    """Compute one robust scale across all training signals."""
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Cannot normalize non-finite target values.")
    return float(np.percentile(arr, 95) - np.percentile(arr, 5) + float(stabilizer))


def _validate_order(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    out = tuple(str(value).strip().lower() for value in values)
    if not out:
        raise ValueError(f"{name} cannot be empty.")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} contains duplicates: {out}")
    return out


def _condition_matrix(
    frame: pd.DataFrame,
    *,
    region: str,
    condition: str,
) -> np.ndarray:
    column = CONDITION_TO_COLUMN[condition]
    region_frame = frame.loc[frame["region"].astype(str) == region]
    traces = [np.asarray(values, dtype=float).reshape(-1) for values in region_frame[column]]
    if not traces:
        raise ValueError(f"No units found for region {region!r}.")
    lengths = {trace.shape[0] for trace in traces}
    if len(lengths) != 1:
        raise ValueError(f"Region {region!r}, condition {condition!r} has inconsistent lengths.")
    matrix = np.column_stack(traces)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Region {region!r}, condition {condition!r} has non-finite values.")
    return matrix.astype(np.float32, copy=False)


def _fit_region_pcs(
    raw_conditions: np.ndarray,
    *,
    feature_order: tuple[str, ...],
    variance_threshold: float,
) -> tuple[np.ndarray, PCAMetadata]:
    n_conditions, n_time, n_features = raw_conditions.shape
    flat = raw_conditions.reshape(n_conditions * n_time, n_features).astype(float, copy=False)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained = singular_values**2 / max(centered.shape[0] - 1, 1)
    total = float(explained.sum())
    ratios = explained / total if total > 0 else np.zeros_like(explained)
    cumulative = np.cumsum(ratios)
    n_components = int(np.searchsorted(cumulative, float(variance_threshold)) + 1) if total > 0 else 1
    n_components = max(1, min(n_components, vt.shape[0]))
    scores = centered @ vt[:n_components].T
    pcs = scores.reshape(n_conditions, n_time, n_components).astype(np.float32, copy=False)
    metadata = PCAMetadata(
        mean=mean.squeeze(0).astype(np.float32, copy=False),
        components=vt[:n_components].astype(np.float32, copy=False),
        explained_variance_ratio=ratios[:n_components].astype(np.float32, copy=False),
        source_features=feature_order,
    )
    return pcs, metadata


def build_fixation_mrnn_targets_from_dataframe(
    combined_dataframe: pd.DataFrame,
    *,
    timeline_s: np.ndarray,
    region_order: tuple[str, ...] = MRNN_REGION_ORDER,
    condition_order: tuple[str, ...] = CONDITION_ORDER,
    normalize_targets: bool = True,
    normalization_stabilizer: float = 5.0,
    pca_variance_threshold: float = 0.95,
) -> FixationMRNNTargets:
    """Build raw firing-rate and region-PC targets from combined PSTH rows."""
    region_order = _validate_order(region_order, name="region_order")
    condition_order = _validate_order(condition_order, name="condition_order")
    missing_conditions = sorted(set(condition_order) - set(CONDITION_TO_COLUMN))
    if missing_conditions:
        raise ValueError(f"Unsupported conditions: {missing_conditions}")

    frame = build_mrnn_training_dataframe(combined_dataframe)
    region_type = pd.CategoricalDtype(categories=region_order, ordered=True)
    frame["region"] = frame["region"].astype(region_type)
    if frame["region"].isna().any():
        raise ValueError("Training dataframe contains regions outside region_order.")
    frame = frame.sort_values(["region", "date", "uuid"]).reset_index(drop=True)

    raw_by_region: dict[str, np.ndarray] = {}
    raw_features_by_region: dict[str, tuple[str, ...]] = {}
    matrices_by_region: dict[str, list[np.ndarray]] = {}
    for region in region_order:
        region_frame = frame.loc[frame["region"].astype(str) == region]
        raw_features_by_region[region] = tuple(str(uuid) for uuid in region_frame["uuid"])
        matrices_by_region[region] = [
            _condition_matrix(frame, region=region, condition=condition)
            for condition in condition_order
        ]

    normalization_scale = None
    if normalize_targets:
        pooled = np.concatenate(
            [matrix.reshape(-1) for matrices in matrices_by_region.values() for matrix in matrices]
        )
        normalization_scale = global_robust_scale(
            pooled,
            stabilizer=float(normalization_stabilizer),
        )

    pcs_by_region: dict[str, np.ndarray] = {}
    pc_features_by_region: dict[str, tuple[str, ...]] = {}
    pca_by_region: dict[str, PCAMetadata] = {}
    for region in region_order:
        matrices = matrices_by_region[region]
        if normalization_scale is not None:
            matrices = [(matrix / normalization_scale).astype(np.float32, copy=False) for matrix in matrices]
        raw = np.stack(matrices, axis=0).astype(np.float32, copy=False)
        raw_by_region[region] = raw
        pcs, pca = _fit_region_pcs(
            raw,
            feature_order=raw_features_by_region[region],
            variance_threshold=float(pca_variance_threshold),
        )
        pcs_by_region[region] = pcs
        pca_by_region[region] = pca
        pc_features_by_region[region] = tuple(f"{region}_pc{idx + 1}" for idx in range(pcs.shape[-1]))

    timeline = np.asarray(timeline_s, dtype=float).reshape(-1)
    n_time = next(iter(raw_by_region.values())).shape[1]
    if timeline.shape[0] != n_time:
        raise ValueError(f"Timeline has {timeline.shape[0]} bins but targets have {n_time}.")

    return FixationMRNNTargets(
        condition_order=condition_order,
        region_order=region_order,
        timeline_s=timeline,
        input_tensor=build_condition_input(condition_order, timesteps=n_time),
        raw_by_region=raw_by_region,
        pcs_by_region=pcs_by_region,
        raw_features_by_region=raw_features_by_region,
        pc_features_by_region=pc_features_by_region,
        pca_by_region=pca_by_region,
        training_dataframe=frame,
        normalization_scale=normalization_scale,
    )


def build_fixation_mrnn_targets(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    input_subdir: str,
    dataframe_filename: str,
    timeline_filename: str,
    region_order: tuple[str, ...] = MRNN_REGION_ORDER,
    condition_order: tuple[str, ...] = CONDITION_ORDER,
    normalize_targets: bool = True,
    normalization_stabilizer: float = 5.0,
    pca_variance_threshold: float = 0.95,
) -> FixationMRNNTargets:
    """Load combined PSTH files and build targets."""
    loaded = load_combined_fixation_psth(
        cfg_path,
        input_subdir=input_subdir,
        dataframe_filename=dataframe_filename,
        timeline_filename=timeline_filename,
    )
    targets = build_fixation_mrnn_targets_from_dataframe(
        loaded.dataframe,
        timeline_s=loaded.timeline_s_rel,
        region_order=region_order,
        condition_order=condition_order,
        normalize_targets=normalize_targets,
        normalization_stabilizer=normalization_stabilizer,
        pca_variance_threshold=pca_variance_threshold,
    )
    return FixationMRNNTargets(
        **{
            **targets.__dict__,
            "dataframe_path": loaded.dataframe_path,
            "timeline_path": loaded.timeline_path,
        }
    )


def summarize_targets(targets: FixationMRNNTargets, *, target_mode: str) -> pd.DataFrame:
    """Compact target summary table."""
    mode = normalize_target_mode(target_mode)
    values = targets.targets_for_mode(mode)
    rows = []
    for region in targets.region_order:
        arr = values[region]
        rows.append(
            {
                "target_mode": mode,
                "region": region,
                "shape": tuple(int(x) for x in arr.shape),
                "output_dim": int(arr.shape[-1]),
                "variance": float(np.var(arr)),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )
    return pd.DataFrame(rows)


def serialize_pca_metadata(pca_by_region: Mapping[str, PCAMetadata]) -> dict[str, dict[str, object]]:
    """Make PCA metadata torch-save friendly."""
    return {
        region: {
            "mean": meta.mean,
            "components": meta.components,
            "explained_variance_ratio": meta.explained_variance_ratio,
            "source_features": list(meta.source_features),
        }
        for region, meta in pca_by_region.items()
    }


__all__ = [
    "CONDITION_ORDER",
    "CONDITION_TO_COLUMN",
    "FixationMRNNTargets",
    "PCAMetadata",
    "build_condition_input",
    "build_fixation_mrnn_targets",
    "build_fixation_mrnn_targets_from_dataframe",
    "global_robust_scale",
    "normalize_target_mode",
    "serialize_pca_metadata",
    "summarize_targets",
]
