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
    n_components_required: int
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

    def pc_reconstructed_raw_by_region(self) -> dict[str, np.ndarray]:
        """Back-project saved region PCs into normalized firing-rate space."""
        return {
            region: backproject_region_pcs(self.pcs_by_region[region], self.pca_by_region[region])
            for region in self.region_order
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
    temporal_basis_count: int = 20,
) -> np.ndarray:
    """Return condition x time x input with condition and temporal channels."""
    channel = {condition: idx for idx, condition in enumerate(CONDITION_ORDER)}
    n_temporal = max(0, int(temporal_basis_count))
    out = np.zeros(
        (len(condition_order), int(timesteps), len(CONDITION_ORDER) + n_temporal),
        dtype=np.float32,
    )
    for cond_idx, condition in enumerate(condition_order):
        out[cond_idx, :, channel[condition]] = 1.0
    if n_temporal:
        time = np.linspace(-1.0, 1.0, int(timesteps), dtype=np.float32)
        centers = np.linspace(-1.0, 1.0, n_temporal, dtype=np.float32)
        if n_temporal == 1:
            width = 1.0
        else:
            width = float(centers[1] - centers[0])
        basis = np.exp(-0.5 * ((time[:, None] - centers[None, :]) / width) ** 2)
        basis = basis / np.maximum(basis.max(axis=0, keepdims=True), 1e-8)
        out[:, :, len(CONDITION_ORDER) :] = basis[None, :, :]
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


@dataclass(frozen=True)
class _RegionPCAFit:
    mean: np.ndarray
    components_full: np.ndarray
    explained_variance_ratio: np.ndarray
    n_components_required: int
    source_features: tuple[str, ...]
    centered: np.ndarray
    n_conditions: int
    n_time: int


def _fit_region_pca(
    raw_conditions: np.ndarray,
    *,
    feature_order: tuple[str, ...],
    variance_threshold: float,
) -> _RegionPCAFit:
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
    return _RegionPCAFit(
        mean=mean.squeeze(0).astype(np.float32, copy=False),
        components_full=vt.astype(np.float32, copy=False),
        explained_variance_ratio=ratios.astype(np.float32, copy=False),
        n_components_required=int(n_components),
        source_features=feature_order,
        centered=centered.astype(np.float32, copy=False),
        n_conditions=int(n_conditions),
        n_time=int(n_time),
    )


def _project_region_pca(fit: _RegionPCAFit, *, n_components: int) -> tuple[np.ndarray, PCAMetadata]:
    n_components = int(n_components)
    n_fit = min(n_components, fit.components_full.shape[0])
    components = np.zeros((n_components, fit.components_full.shape[1]), dtype=np.float32)
    components[:n_fit] = fit.components_full[:n_fit]
    scores = np.zeros((fit.centered.shape[0], n_components), dtype=np.float32)
    if n_fit:
        scores[:, :n_fit] = fit.centered @ components[:n_fit].T
    pcs = scores.reshape(fit.n_conditions, fit.n_time, n_components).astype(np.float32, copy=False)
    metadata = PCAMetadata(
        mean=fit.mean,
        components=components,
        explained_variance_ratio=fit.explained_variance_ratio[:n_fit],
        n_components_required=fit.n_components_required,
        source_features=fit.source_features,
    )
    return pcs, metadata


def backproject_region_pcs(pcs: np.ndarray, pca: PCAMetadata | Mapping[str, object]) -> np.ndarray:
    """Project region PC scores back into normalized firing-rate space.

    This reconstructs the firing-rate trajectories represented by the retained
    PCs. It intentionally does not restore discarded PCs, so it should be used
    as the firing-rate target when the model itself is trained in PC space.
    """
    scores = np.asarray(pcs, dtype=float)
    if isinstance(pca, PCAMetadata):
        mean = np.asarray(pca.mean, dtype=float)
        components = np.asarray(pca.components, dtype=float)
    else:
        mean = np.asarray(pca["mean"], dtype=float)
        components = np.asarray(pca["components"], dtype=float)
    n_components = min(scores.shape[-1], components.shape[0])
    reconstructed = scores[..., :n_components] @ components[:n_components] + mean
    return reconstructed.astype(np.float32, copy=False)


def backproject_region_pcs_by_region(
    pcs_by_region: Mapping[str, np.ndarray],
    pca_by_region: Mapping[str, PCAMetadata | Mapping[str, object]],
) -> dict[str, np.ndarray]:
    """Back-project region PC score tensors for every region."""
    return {
        region: backproject_region_pcs(pcs, pca_by_region[region])
        for region, pcs in pcs_by_region.items()
    }


def build_fixation_mrnn_targets_from_dataframe(
    combined_dataframe: pd.DataFrame,
    *,
    timeline_s: np.ndarray,
    region_order: tuple[str, ...] = MRNN_REGION_ORDER,
    condition_order: tuple[str, ...] = CONDITION_ORDER,
    normalize_targets: bool = True,
    normalization_stabilizer: float = 5.0,
    pca_variance_threshold: float = 0.95,
    temporal_basis_count: int = 20,
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
    pca_fits_by_region: dict[str, _RegionPCAFit] = {}
    for region in region_order:
        matrices = matrices_by_region[region]
        if normalization_scale is not None:
            matrices = [(matrix / normalization_scale).astype(np.float32, copy=False) for matrix in matrices]
        raw = np.stack(matrices, axis=0).astype(np.float32, copy=False)
        raw_by_region[region] = raw
        pca_fits_by_region[region] = _fit_region_pca(
            raw,
            feature_order=raw_features_by_region[region],
            variance_threshold=float(pca_variance_threshold),
        )

    shared_n_components = max(fit.n_components_required for fit in pca_fits_by_region.values())
    for region in region_order:
        pcs, pca = _project_region_pca(
            pca_fits_by_region[region],
            n_components=shared_n_components,
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
        input_tensor=build_condition_input(
            condition_order,
            timesteps=n_time,
            temporal_basis_count=temporal_basis_count,
        ),
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
    temporal_basis_count: int = 20,
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
        temporal_basis_count=temporal_basis_count,
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
            "n_components_required": meta.n_components_required,
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
    "backproject_region_pcs",
    "backproject_region_pcs_by_region",
    "global_robust_scale",
    "normalize_target_mode",
    "serialize_pca_metadata",
    "summarize_targets",
]
