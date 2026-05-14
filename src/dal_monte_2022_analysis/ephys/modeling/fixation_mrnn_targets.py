"""Build fixation mRNN training targets from combined PSTH exports."""

from __future__ import annotations

from dataclasses import dataclass
import json
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

CANONICAL_CONDITION_ORDER = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
CANONICAL_TO_LEGACY_CONDITION = {
    "face_interactive": "high_interactivity_face",
    "face_non_interactive": "low_interactivity_face",
    "object": "object",
}
LEGACY_TO_CANONICAL_CONDITION = {
    legacy: canonical for canonical, legacy in CANONICAL_TO_LEGACY_CONDITION.items()
}
TARGET_MODE_ALIASES = {
    "raw": "raw_fr",
    "raw_fr": "raw_fr",
    "fr": "raw_fr",
    "region_pcs": "region_pcs",
    "latent_pcs": "region_pcs",
    "latents": "region_pcs",
    "pcs": "region_pcs",
}


@dataclass(frozen=True)
class RegionPCAMetadata:
    """PCA fit metadata for one modeled region."""

    region: str
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    n_components: int
    n_components_required_for_threshold: int
    variance_threshold: float
    source_feature_order: tuple[str, ...]


@dataclass(frozen=True)
class FixationMRNNTargets:
    """Canonical target tensors and metadata for fixation mRNN training."""

    condition_names: tuple[str, ...]
    input_tensor: np.ndarray
    raw_targets_by_region: dict[str, np.ndarray]
    pc_targets_by_region: dict[str, np.ndarray]
    raw_feature_order_by_region: dict[str, tuple[str, ...]]
    pc_feature_order_by_region: dict[str, tuple[str, ...]]
    pca_metadata_by_region: dict[str, RegionPCAMetadata]
    region_unit_counts: dict[str, int]
    canonical_region_order: tuple[str, ...]
    timeline_s_rel: np.ndarray
    training_dataframe: pd.DataFrame
    dataframe_path: Path | None = None
    timeline_path: Path | None = None

    def targets_for_mode(self, target_mode: str) -> dict[str, np.ndarray]:
        """Return canonical per-region target tensors for a normalized target mode."""
        mode = normalize_target_mode(target_mode)
        if mode == "raw_fr":
            return self.raw_targets_by_region
        if mode == "region_pcs":
            return self.pc_targets_by_region
        raise ValueError(f"Unsupported target mode: {target_mode!r}")

    def feature_order_for_mode(self, target_mode: str) -> dict[str, tuple[str, ...]]:
        """Return canonical per-region feature labels for a target mode."""
        mode = normalize_target_mode(target_mode)
        if mode == "raw_fr":
            return self.raw_feature_order_by_region
        if mode == "region_pcs":
            return self.pc_feature_order_by_region
        raise ValueError(f"Unsupported target mode: {target_mode!r}")

    def output_dims_for_mode(self, target_mode: str) -> dict[str, int]:
        """Return per-region output dimensions for a target mode."""
        return {
            region: len(features)
            for region, features in self.feature_order_for_mode(target_mode).items()
        }


def normalize_target_mode(target_mode: str) -> str:
    """Normalize target-mode aliases used by configs and CLIs."""
    token = str(target_mode).strip().lower()
    try:
        return TARGET_MODE_ALIASES[token]
    except KeyError as exc:
        expected = ", ".join(sorted(TARGET_MODE_ALIASES))
        raise ValueError(
            f"Unsupported target_mode={target_mode!r}. Expected one of: {expected}."
        ) from exc


def robust_normalize(values: np.ndarray, *, stabilizer: float = 5.0) -> np.ndarray:
    """Normalize values using the old mRNN robust percentile scale."""
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Cannot normalize arrays containing NaN or infinite values.")
    scale = float(np.percentile(arr, 95) - np.percentile(arr, 5))
    return arr / (scale + float(stabilizer))


def build_condition_input(
    condition_names: tuple[str, ...] = CANONICAL_CONDITION_ORDER,
    *,
    timesteps: int,
    dtype=float,
) -> np.ndarray:
    """Build the three-channel one-hot input tensor."""
    condition_to_channel = {
        "face_interactive": 0,
        "face_non_interactive": 1,
        "object": 2,
    }
    inputs = np.zeros((len(condition_names), int(timesteps), 3), dtype=dtype)
    for row_idx, condition in enumerate(condition_names):
        if condition not in condition_to_channel:
            raise ValueError(f"Unsupported fixation mRNN condition: {condition!r}")
        inputs[row_idx, :, condition_to_channel[condition]] = 1.0
    return inputs


def _validate_region_order(region_order: tuple[str, ...]) -> tuple[str, ...]:
    out = tuple(str(region).strip().lower() for region in region_order)
    if not out:
        raise ValueError("canonical_region_order must contain at least one region.")
    if len(set(out)) != len(out):
        raise ValueError(f"canonical_region_order contains duplicates: {out}")
    return out


def _sort_training_dataframe(
    dataframe: pd.DataFrame,
    *,
    canonical_region_order: tuple[str, ...],
) -> pd.DataFrame:
    ordered = dataframe.copy()
    region_type = pd.CategoricalDtype(categories=canonical_region_order, ordered=True)
    ordered["region"] = ordered["region"].astype(region_type)
    if ordered["region"].isna().any():
        missing = sorted(set(dataframe["region"].astype(str)) - set(canonical_region_order))
        raise ValueError(
            "Training dataframe contains regions outside canonical_region_order: "
            + ", ".join(missing)
        )
    return ordered.sort_values(["region", "date", "uuid"]).reset_index(drop=True)


def _condition_matrix_for_region(
    frame: pd.DataFrame,
    *,
    region: str,
    condition: str,
    normalize: bool,
    stabilizer: float,
) -> np.ndarray:
    legacy_condition = CANONICAL_TO_LEGACY_CONDITION[condition]
    region_frame = frame.loc[frame["region"].astype(str) == region]
    traces = [
        np.asarray(trace, dtype=float).reshape(-1)
        for trace in region_frame[legacy_condition].tolist()
    ]
    if not traces:
        raise ValueError(f"No units were found for modeled region {region!r}.")
    lengths = {trace.shape[0] for trace in traces}
    if len(lengths) != 1:
        raise ValueError(
            f"Region {region!r}, condition {condition!r} has inconsistent trace lengths: "
            f"{sorted(lengths)}"
        )
    matrix = np.column_stack(traces)
    if not np.isfinite(matrix).all():
        nan_count = int(np.isnan(matrix).sum())
        inf_count = int(np.isinf(matrix).sum())
        raise ValueError(
            f"Region {region!r}, condition {condition!r} contains non-finite "
            f"values before normalization: nan={nan_count}, inf={inf_count}."
        )
    if normalize:
        matrix = robust_normalize(matrix, stabilizer=stabilizer)
    return matrix.astype(np.float32, copy=False)


@dataclass(frozen=True)
class _RegionPCAFit:
    region: str
    mean: np.ndarray
    centered: np.ndarray
    components_full: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    n_components_required_for_threshold: int
    source_feature_order: tuple[str, ...]


def _fit_region_pca_full(
    raw_by_condition: list[np.ndarray],
    *,
    region: str,
    feature_order: tuple[str, ...],
    variance_threshold: float,
) -> _RegionPCAFit:
    concat = np.concatenate(raw_by_condition, axis=0).astype(float, copy=False)
    mean = concat.mean(axis=0, keepdims=True)
    centered = concat - mean
    if centered.shape[0] < 2:
        raise ValueError(f"Need at least two time samples to fit PCA for region {region!r}.")

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    denom = max(centered.shape[0] - 1, 1)
    explained = (singular_values**2) / denom
    total = float(explained.sum())
    ratios = explained / total if total > 0 else np.zeros_like(explained)
    cumulative = np.cumsum(ratios)
    if ratios.size == 0 or total <= 0:
        n_components_required = 1
    else:
        n_components_required = int(
            np.searchsorted(cumulative, float(variance_threshold), side="left") + 1
        )
        n_components_required = max(1, min(n_components_required, vt.shape[0]))

    return _RegionPCAFit(
        region=region,
        mean=mean.squeeze(0).astype(np.float32, copy=False),
        centered=centered.astype(np.float32, copy=False),
        components_full=vt.astype(np.float32, copy=False),
        explained_variance_ratio=ratios.astype(np.float32, copy=False),
        cumulative_explained_variance_ratio=cumulative.astype(np.float32, copy=False),
        n_components_required_for_threshold=n_components_required,
        source_feature_order=feature_order,
    )


def _project_region_pca(
    fit: _RegionPCAFit,
    *,
    n_components: int,
    n_conditions: int,
    time_len: int,
    variance_threshold: float,
) -> tuple[np.ndarray, RegionPCAMetadata]:
    n_components = int(n_components)
    n_fit_components = min(n_components, int(fit.components_full.shape[0]))
    components = np.zeros(
        (n_components, int(fit.components_full.shape[1])),
        dtype=np.float32,
    )
    components[:n_fit_components] = fit.components_full[:n_fit_components]
    scores = np.zeros((fit.centered.shape[0], n_components), dtype=np.float32)
    if n_fit_components:
        scores[:, :n_fit_components] = (
            fit.centered @ components[:n_fit_components].T
        ).astype(np.float32, copy=False)
    pc_targets = scores.reshape(int(n_conditions), int(time_len), n_components)
    metadata = RegionPCAMetadata(
        region=fit.region,
        mean=fit.mean.astype(np.float32, copy=False),
        components=components.astype(np.float32, copy=False),
        explained_variance_ratio=fit.explained_variance_ratio,
        cumulative_explained_variance_ratio=fit.cumulative_explained_variance_ratio,
        n_components=int(n_components),
        n_components_required_for_threshold=int(fit.n_components_required_for_threshold),
        variance_threshold=float(variance_threshold),
        source_feature_order=fit.source_feature_order,
    )
    return pc_targets, metadata


def build_fixation_mrnn_targets_from_dataframe(
    combined_dataframe: pd.DataFrame,
    *,
    timeline_s_rel: np.ndarray,
    canonical_region_order: tuple[str, ...] = MRNN_REGION_ORDER,
    condition_names: tuple[str, ...] = CANONICAL_CONDITION_ORDER,
    normalize_targets: bool = True,
    normalization_stabilizer: float = 5.0,
    pca_variance_threshold: float = 0.95,
) -> FixationMRNNTargets:
    """Build raw firing-rate and per-region PC targets from a combined PSTH dataframe."""
    canonical_region_order = _validate_region_order(canonical_region_order)
    condition_names = tuple(str(condition) for condition in condition_names)
    unsupported = sorted(set(condition_names) - set(CANONICAL_TO_LEGACY_CONDITION))
    if unsupported:
        raise ValueError(f"Unsupported condition names: {unsupported}")

    training_df = build_mrnn_training_dataframe(combined_dataframe)
    training_df = _sort_training_dataframe(
        training_df,
        canonical_region_order=canonical_region_order,
    )

    raw_targets_by_region: dict[str, np.ndarray] = {}
    pc_targets_by_region: dict[str, np.ndarray] = {}
    raw_feature_order_by_region: dict[str, tuple[str, ...]] = {}
    pc_feature_order_by_region: dict[str, tuple[str, ...]] = {}
    pca_metadata_by_region: dict[str, RegionPCAMetadata] = {}
    region_unit_counts: dict[str, int] = {}
    condition_matrices_by_region: dict[str, list[np.ndarray]] = {}
    pca_fits_by_region: dict[str, _RegionPCAFit] = {}

    for region in canonical_region_order:
        region_frame = training_df.loc[training_df["region"].astype(str) == region]
        feature_order = tuple(str(uuid) for uuid in region_frame["uuid"].tolist())
        raw_feature_order_by_region[region] = feature_order
        region_unit_counts[region] = len(feature_order)

        condition_matrices = [
            _condition_matrix_for_region(
                training_df,
                region=region,
                condition=condition,
                normalize=bool(normalize_targets),
                stabilizer=float(normalization_stabilizer),
            )
            for condition in condition_names
        ]
        raw_targets_by_region[region] = np.stack(condition_matrices, axis=0)
        condition_matrices_by_region[region] = condition_matrices
        pca_fits_by_region[region] = _fit_region_pca_full(
            condition_matrices,
            region=region,
            feature_order=feature_order,
            variance_threshold=float(pca_variance_threshold),
        )

    global_n_components = max(
        fit.n_components_required_for_threshold for fit in pca_fits_by_region.values()
    )
    for region in canonical_region_order:
        condition_matrices = condition_matrices_by_region[region]
        pc_targets, pca_metadata = _project_region_pca(
            pca_fits_by_region[region],
            n_components=int(global_n_components),
            n_conditions=len(condition_names),
            time_len=condition_matrices[0].shape[0],
            variance_threshold=float(pca_variance_threshold),
        )
        pc_targets_by_region[region] = pc_targets
        pca_metadata_by_region[region] = pca_metadata
        pc_feature_order_by_region[region] = tuple(
            f"{region}_pc{idx + 1}" for idx in range(pca_metadata.n_components)
        )

    timeline = np.asarray(timeline_s_rel, dtype=float).reshape(-1)
    expected_timesteps = next(iter(raw_targets_by_region.values())).shape[1]
    if timeline.shape[0] != expected_timesteps:
        raise ValueError(
            f"Timeline has {timeline.shape[0]} bins but targets have {expected_timesteps}."
        )

    return FixationMRNNTargets(
        condition_names=condition_names,
        input_tensor=build_condition_input(condition_names, timesteps=expected_timesteps),
        raw_targets_by_region=raw_targets_by_region,
        pc_targets_by_region=pc_targets_by_region,
        raw_feature_order_by_region=raw_feature_order_by_region,
        pc_feature_order_by_region=pc_feature_order_by_region,
        pca_metadata_by_region=pca_metadata_by_region,
        region_unit_counts=region_unit_counts,
        canonical_region_order=canonical_region_order,
        timeline_s_rel=timeline,
        training_dataframe=training_df,
    )


def build_fixation_mrnn_targets(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    input_subdir: str = "ephys/psth/fixation_psth_averages",
    dataframe_filename: str = "fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl",
    timeline_filename: str = "fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl",
    canonical_region_order: tuple[str, ...] = MRNN_REGION_ORDER,
    condition_names: tuple[str, ...] = CANONICAL_CONDITION_ORDER,
    normalize_targets: bool = True,
    normalization_stabilizer: float = 5.0,
    pca_variance_threshold: float = 0.95,
) -> FixationMRNNTargets:
    """Load the combined PSTH export and build fixation mRNN targets."""
    loaded = load_combined_fixation_psth(
        cfg_path,
        input_subdir=input_subdir,
        dataframe_filename=dataframe_filename,
        timeline_filename=timeline_filename,
    )
    targets = build_fixation_mrnn_targets_from_dataframe(
        loaded.dataframe,
        timeline_s_rel=loaded.timeline_s_rel,
        canonical_region_order=canonical_region_order,
        condition_names=condition_names,
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


def serialize_pca_metadata(
    pca_metadata_by_region: Mapping[str, RegionPCAMetadata],
) -> dict[str, dict[str, object]]:
    """Convert PCA metadata to a pickle/torch-save friendly dictionary."""
    return {
        region: {
            "region": metadata.region,
            "mean": metadata.mean,
            "components": metadata.components,
            "explained_variance_ratio": metadata.explained_variance_ratio,
            "cumulative_explained_variance_ratio": metadata.cumulative_explained_variance_ratio,
            "n_components": metadata.n_components,
            "n_components_required_for_threshold": metadata.n_components_required_for_threshold,
            "variance_threshold": metadata.variance_threshold,
            "source_feature_order": list(metadata.source_feature_order),
        }
        for region, metadata in pca_metadata_by_region.items()
    }


def summarize_fixation_mrnn_targets(
    targets: FixationMRNNTargets,
    *,
    target_modes: tuple[str, ...] = ("raw_fr", "region_pcs"),
    sample_timepoints: int = 5,
    sample_features: int = 5,
) -> pd.DataFrame:
    """Build a compact finite-value and sample summary for target tensors."""
    rows: list[dict[str, object]] = []
    for target_mode_raw in target_modes:
        target_mode = normalize_target_mode(target_mode_raw)
        by_region = targets.targets_for_mode(target_mode)
        for region in targets.canonical_region_order:
            arr = np.asarray(by_region[region], dtype=float)
            finite = np.isfinite(arr)
            finite_values = arr[finite]
            for condition_idx, condition in enumerate(targets.condition_names):
                condition_arr = arr[condition_idx]
                condition_finite = np.isfinite(condition_arr)
                condition_values = condition_arr[condition_finite]
                sample = condition_arr[
                    : int(sample_timepoints),
                    : min(int(sample_features), condition_arr.shape[-1]),
                ]
                rows.append(
                    {
                        "target_mode": target_mode,
                        "region": region,
                        "condition": condition,
                        "shape": json_shape(arr.shape),
                        "condition_shape": json_shape(condition_arr.shape),
                        "n_values": int(condition_arr.size),
                        "n_finite": int(condition_finite.sum()),
                        "n_nan": int(np.isnan(condition_arr).sum()),
                        "n_inf": int(np.isinf(condition_arr).sum()),
                        "global_n_finite": int(finite.sum()),
                        "global_n_nan": int(np.isnan(arr).sum()),
                        "global_n_inf": int(np.isinf(arr).sum()),
                        "min": float(np.min(condition_values)) if condition_values.size else np.nan,
                        "max": float(np.max(condition_values)) if condition_values.size else np.nan,
                        "mean": float(np.mean(condition_values)) if condition_values.size else np.nan,
                        "std": float(np.std(condition_values)) if condition_values.size else np.nan,
                        "global_min": float(np.min(finite_values)) if finite_values.size else np.nan,
                        "global_max": float(np.max(finite_values)) if finite_values.size else np.nan,
                        "sample_values": json_sample(sample),
                    }
                )
    return pd.DataFrame(rows)


def summarize_fixation_mrnn_pca(targets: FixationMRNNTargets) -> pd.DataFrame:
    """Summarize per-region PCA requirements and the shared retained PC count."""
    rows: list[dict[str, object]] = []
    for region in targets.canonical_region_order:
        metadata = targets.pca_metadata_by_region[region]
        n_required = int(metadata.n_components_required_for_threshold)
        n_components = int(metadata.n_components)
        cumulative = metadata.cumulative_explained_variance_ratio
        retained_index = min(n_components, int(cumulative.size)) - 1
        rows.append(
            {
                "region": region,
                "n_source_features": len(metadata.source_feature_order),
                "n_components": n_components,
                "n_components_required_for_threshold": n_required,
                "variance_threshold": float(metadata.variance_threshold),
                "cumulative_at_required": float(cumulative[n_required - 1])
                if cumulative.size >= n_required
                else np.nan,
                "cumulative_at_retained": float(cumulative[retained_index])
                if retained_index >= 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def json_shape(shape: tuple[int, ...]) -> str:
    """Serialize an array shape for compact summaries."""
    return "[" + ", ".join(str(int(value)) for value in shape) + "]"


def json_sample(values: np.ndarray) -> str:
    """Serialize a small numeric sample block for human-readable diagnostics."""
    rounded = np.asarray(values, dtype=float)
    return json.dumps(np.round(rounded, 6).tolist())


__all__ = [
    "CANONICAL_CONDITION_ORDER",
    "CANONICAL_TO_LEGACY_CONDITION",
    "FixationMRNNTargets",
    "RegionPCAMetadata",
    "build_condition_input",
    "build_fixation_mrnn_targets",
    "build_fixation_mrnn_targets_from_dataframe",
    "normalize_target_mode",
    "robust_normalize",
    "serialize_pca_metadata",
    "summarize_fixation_mrnn_pca",
    "summarize_fixation_mrnn_targets",
]
