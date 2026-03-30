"""ClusterFix-style fixation and saccade detection without resampling."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Tuple

import numpy as np
from scipy import signal
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class FixationDetectionConfig:
    """Configurable thresholds and clustering parameters for fixation detection."""

    default_sampling_rate_hz: float = 1000.0
    minimum_input_duration_ms: float = 500.0
    minimum_feature_samples: int = 3
    fixation_min_duration_ms: float = 25.0
    saccade_min_duration_ms: float = 10.0
    tiny_fixation_reassign_duration_ms: float = 5.0
    lowpass_filter_order: int = 60
    lowpass_cutoff_hz: float = 30.0
    lowpass_buffer_duration_ms: float = 100.0
    global_kmeans_k_min: int = 2
    global_kmeans_k_max: int = 5
    global_decimation_step: int = 10
    global_min_sampled_points: int = 10
    local_recluster_padding_duration_ms: float = 50.0
    local_min_feature_samples: int = 5
    local_kmeans_k_min: int = 2
    local_kmeans_k_max: int = 5
    local_decimation_step: int = 5
    kmeans_n_init: int = 5
    silhouette_threshold_fraction: float = 0.9
    fixation_cluster_velocity_std_multiplier: float = 3.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if float(self.default_sampling_rate_hz) <= 0:
            raise ValueError("default_sampling_rate_hz must be positive.")
        if float(self.minimum_input_duration_ms) <= 0:
            raise ValueError("minimum_input_duration_ms must be positive.")
        if int(self.minimum_feature_samples) < 1:
            raise ValueError("minimum_feature_samples must be at least 1.")
        if float(self.fixation_min_duration_ms) <= 0:
            raise ValueError("fixation_min_duration_ms must be positive.")
        if float(self.saccade_min_duration_ms) <= 0:
            raise ValueError("saccade_min_duration_ms must be positive.")
        if float(self.tiny_fixation_reassign_duration_ms) <= 0:
            raise ValueError("tiny_fixation_reassign_duration_ms must be positive.")
        if int(self.lowpass_filter_order) < 1:
            raise ValueError("lowpass_filter_order must be at least 1.")
        if float(self.lowpass_cutoff_hz) <= 0:
            raise ValueError("lowpass_cutoff_hz must be positive.")
        if float(self.lowpass_buffer_duration_ms) < 0:
            raise ValueError("lowpass_buffer_duration_ms cannot be negative.")
        if int(self.global_kmeans_k_min) < 2:
            raise ValueError("global_kmeans_k_min must be at least 2.")
        if int(self.global_kmeans_k_max) < int(self.global_kmeans_k_min):
            raise ValueError("global_kmeans_k_max must be >= global_kmeans_k_min.")
        if int(self.global_decimation_step) < 1:
            raise ValueError("global_decimation_step must be at least 1.")
        if int(self.global_min_sampled_points) < 1:
            raise ValueError("global_min_sampled_points must be at least 1.")
        if float(self.local_recluster_padding_duration_ms) < 0:
            raise ValueError("local_recluster_padding_duration_ms cannot be negative.")
        if int(self.local_min_feature_samples) < 1:
            raise ValueError("local_min_feature_samples must be at least 1.")
        if int(self.local_kmeans_k_min) < 2:
            raise ValueError("local_kmeans_k_min must be at least 2.")
        if int(self.local_kmeans_k_max) < int(self.local_kmeans_k_min):
            raise ValueError("local_kmeans_k_max must be >= local_kmeans_k_min.")
        if int(self.local_decimation_step) < 1:
            raise ValueError("local_decimation_step must be at least 1.")
        if int(self.kmeans_n_init) < 1:
            raise ValueError("kmeans_n_init must be at least 1.")
        if not (0 < float(self.silhouette_threshold_fraction) <= 1.0):
            raise ValueError("silhouette_threshold_fraction must be in (0, 1].")
        if float(self.fixation_cluster_velocity_std_multiplier) < 0:
            raise ValueError("fixation_cluster_velocity_std_multiplier cannot be negative.")


_FIXATION_DETECTION_CONFIG_FIELDS = {field.name for field in fields(FixationDetectionConfig)}


def coerce_fixation_detection_config(
    config: FixationDetectionConfig | Mapping[str, Any] | None = None,
) -> FixationDetectionConfig:
    """Normalize config-like input into a validated FixationDetectionConfig."""
    if config is None:
        return FixationDetectionConfig()
    if isinstance(config, FixationDetectionConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping, FixationDetectionConfig, or None.")

    unknown_keys = sorted(set(config) - _FIXATION_DETECTION_CONFIG_FIELDS)
    if unknown_keys:
        unknown = ", ".join(unknown_keys)
        raise KeyError(f"Unknown fixation detection config key(s): {unknown}.")

    return FixationDetectionConfig(**dict(config))


def _samples_from_duration_ms(duration_ms: float, sampling_rate_hz: float, *, minimum: int = 1) -> int:
    """Convert a duration in milliseconds into whole samples."""
    return max(minimum, int(math.ceil((float(duration_ms) / 1000.0) * float(sampling_rate_hz))))


def detect_fixations_and_saccades(
    positions: np.ndarray,
    sampling_rate_hz: float | None = None,
    random_state: int | None = None,
    config: FixationDetectionConfig | Mapping[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect fixation and saccade intervals using a ClusterFix-style procedure.

    This version adapts the original MATLAB ClusterFix logic but does NOT
    upsample to 1000 Hz. Instead, all duration thresholds are converted from
    milliseconds into samples at the provided sampling rate.

    Args:
        positions:
            Array of shape (N, 2) with columns [x, y].
        sampling_rate_hz:
            Sampling rate of the input positions, e.g. 200.0 for 200 Hz.
            If omitted, defaults to the configured rate, which is 1000 Hz.
        random_state:
            Random seed for k-means reproducibility.
        config:
            Optional fixation-detection parameter overrides.

    Returns:
        fixation_start_stop:
            (M, 2) array of inclusive [start, stop] indices in the ORIGINAL
            sample space.
        saccade_start_stop:
            (K, 2) array of inclusive [start, stop] indices in the ORIGINAL
            sample space.
    """
    config = coerce_fixation_detection_config(config)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")
    if sampling_rate_hz is None:
        sampling_rate_hz = float(config.default_sampling_rate_hz)
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if random_state is None:
        random_state = int(config.random_state)

    min_samples_required = _samples_from_duration_ms(
        config.minimum_input_duration_ms,
        sampling_rate_hz,
    )
    if positions.shape[0] < min_samples_required:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    x_filt, y_filt = _apply_lowpass_filter(positions, sampling_rate_hz, config)

    # These are the ClusterFix features used for clustering:
    # Dist, Vel, Accel, Angular Velocity
    feature_matrix = _compute_clusterfix_features(x_filt, y_filt)

    if feature_matrix.shape[0] < int(config.minimum_feature_samples):
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    normalized_features = _normalize_features(feature_matrix.copy())

    fixation_indices = _extract_fixation_indices_through_global_kmeans(
        normalized_features,
        config=config,
        random_state=random_state,
    )

    fixation_intervals = _extract_behavior_intervals(fixation_indices)

    not_fixation_indices = _refine_fixation_classification_to_get_notfix_inds(
        fixation_intervals,
        normalized_features,
        sampling_rate_hz=sampling_rate_hz,
        config=config,
        random_state=random_state,
    )

    fixation_indices = np.setdiff1d(fixation_indices, not_fixation_indices)
    saccade_indices = np.setdiff1d(np.arange(normalized_features.shape[0]), fixation_indices)

    # Reproduce the MATLAB-style consolidation logic more closely.
    fixation_intervals = _extract_behavior_intervals(fixation_indices)
    saccade_intervals = _extract_behavior_intervals(saccade_indices)

    fixation_min_samples = _samples_from_duration_ms(
        config.fixation_min_duration_ms,
        sampling_rate_hz,
    )
    saccade_min_samples = _samples_from_duration_ms(
        config.saccade_min_duration_ms,
        sampling_rate_hz,
    )
    tiny_fixation_samples = _samples_from_duration_ms(
        config.tiny_fixation_reassign_duration_ms,
        sampling_rate_hz,
    )

    # MATLAB temporarily reassigns very short fixation fragments into saccades.
    too_short_fix = np.where((fixation_intervals[:, 1] - fixation_intervals[:, 0] + 1) < tiny_fixation_samples)[0]
    if too_short_fix.size > 0:
        extra_nonfix = np.concatenate(
            [np.arange(fixation_intervals[i, 0], fixation_intervals[i, 1] + 1) for i in too_short_fix]
        )
        saccade_indices = np.sort(np.unique(np.concatenate([saccade_indices, extra_nonfix])))

    saccade_intervals = _extract_behavior_intervals(saccade_indices)
    too_short_sacc = np.where((saccade_intervals[:, 1] - saccade_intervals[:, 0] + 1) < saccade_min_samples)[0]
    if too_short_sacc.size > 0:
        extra_fix = np.concatenate(
            [np.arange(saccade_intervals[i, 0], saccade_intervals[i, 1] + 1) for i in too_short_sacc]
        )
        fixation_indices = np.sort(np.unique(np.concatenate([fixation_indices, extra_fix])))

    fixation_intervals = _extract_behavior_intervals(fixation_indices)
    fixation_intervals = _filter_behavior_intervals(fixation_intervals, fixation_min_samples)

    # Rebuild final fixation indices from filtered intervals.
    if fixation_intervals.size > 0:
        fixation_indices = np.concatenate(
            [np.arange(start, stop + 1) for start, stop in fixation_intervals]
        )
    else:
        fixation_indices = np.array([], dtype=int)

    # Final saccades are complement of fixations in feature space.
    all_indices = np.arange(normalized_features.shape[0])
    saccade_indices = np.setdiff1d(all_indices, fixation_indices)
    saccade_intervals = _extract_behavior_intervals(saccade_indices)
    saccade_intervals = _filter_behavior_intervals(saccade_intervals, saccade_min_samples)

    # IMPORTANT:
    # ClusterFix features are defined on x[0:N-2], i.e. feature index i refers
    # roughly to original sample i..i+2. To keep your existing convention of
    # mapping detections back to original samples, shift by +1.
    fixation_start_stop = fixation_intervals + 1 if fixation_intervals.size else np.empty((0, 2), dtype=int)
    saccade_start_stop = saccade_intervals + 1 if saccade_intervals.size else np.empty((0, 2), dtype=int)

    return fixation_start_stop.astype(int), saccade_start_stop.astype(int)


def _apply_lowpass_filter(
    positions: np.ndarray,
    sampling_rate_hz: float,
    config: FixationDetectionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same kind of 30 Hz low-pass filtering used in the MATLAB code,
    but at the native sampling rate rather than after upsampling.
    """
    fltord = int(config.lowpass_filter_order)
    lowpass_freq_hz = float(config.lowpass_cutoff_hz)
    nyquist_hz = sampling_rate_hz / 2.0

    if lowpass_freq_hz >= nyquist_hz:
        raise ValueError(
            f"Low-pass cutoff ({lowpass_freq_hz} Hz) must be below Nyquist ({nyquist_hz:.3f} Hz)."
        )

    # MATLAB uses a 100 ms reflection-like buffer.
    buffer_samples = _samples_from_duration_ms(
        config.lowpass_buffer_duration_ms,
        sampling_rate_hz,
        minimum=0,
    )

    if buffer_samples > 0:
        x = np.pad(positions[:, 0], (buffer_samples, buffer_samples), mode="reflect")
        y = np.pad(positions[:, 1], (buffer_samples, buffer_samples), mode="reflect")
    else:
        x = positions[:, 0]
        y = positions[:, 1]

    flt = signal.firwin(fltord + 1, cutoff=lowpass_freq_hz / nyquist_hz, pass_zero=True)

    x_filt = signal.filtfilt(flt, [1.0], x)
    y_filt = signal.filtfilt(flt, [1.0], y)

    if buffer_samples > 0:
        x_filt = x_filt[buffer_samples:-buffer_samples]
        y_filt = y_filt[buffer_samples:-buffer_samples]

    return x_filt, y_filt


def _compute_clusterfix_features(xss: np.ndarray, yss: np.ndarray) -> np.ndarray:
    """
    Compute the four clustering features used in the original MATLAB code:

    1) distance
    2) velocity
    3) acceleration
    4) angular velocity

    Notes:
    - This intentionally mirrors the MATLAB ClusterFix feature logic.
    - Because we are not upsampling to 1000 Hz, these are computed directly
      at the native sampling rate.
    """
    velx = np.diff(xss)
    vely = np.diff(yss)
    vel = np.sqrt(velx**2 + vely**2)

    accel = np.abs(np.diff(vel))
    angle = np.degrees(np.arctan2(vely, velx))

    # Align to length N-2, as in MATLAB.
    vel = vel[:-1]

    dist = np.sqrt((xss[:-2] - xss[2:]) ** 2 + (yss[:-2] - yss[2:]) ** 2)

    # MATLAB form:
    # rot(a) = abs(angle(a)-angle(a+1));
    # rot(rot > 180) = rot(rot > 180)-180;
    # rot = 360-rot;
    rot = np.abs(np.diff(angle))
    rot = np.where(rot > 180, rot - 180, rot)
    rot = 360 - rot

    return np.column_stack((dist, vel, accel, rot))


def _normalize_features(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each feature to [0, 1] after clipping at mean + 3*std,
    matching the MATLAB logic.
    """
    for col in range(feature_matrix.shape[1]):
        values = feature_matrix[:, col]
        threshold = np.mean(values) + 3.0 * np.std(values)
        values = np.minimum(values, threshold)
        values = values - np.min(values)
        max_val = np.max(values)
        if max_val > 0:
            values = values / max_val
        else:
            values = np.zeros_like(values)
        feature_matrix[:, col] = values
    return feature_matrix


def _extract_fixation_indices_through_global_kmeans(
    feature_matrix: np.ndarray,
    config: FixationDetectionConfig,
    random_state: int,
) -> np.ndarray:
    """
    Global clustering step from ClusterFix.

    Chooses k using a silhouette-like inter-vs-intra metric on a decimated
    subset of columns 1:4 in MATLAB terms => Python [:, 1:4] = Vel, Accel, Rot.
    Then clusters on all four features.
    """
    num_clusters = _determine_optimal_clusters_global(
        feature_matrix[:, 1:4],
        config=config,
        random_state=random_state,
    )

    labels = KMeans(
        n_clusters=num_clusters,
        n_init=int(config.kmeans_n_init),
        random_state=random_state,
    ).fit_predict(feature_matrix)

    unique_clusters = np.unique(labels)
    mean_values = np.array([np.mean(feature_matrix[labels == k], axis=0) for k in unique_clusters])
    std_values = np.array([np.std(feature_matrix[labels == k], axis=0) for k in unique_clusters])

    # Fixation cluster = smallest sum of mean velocity + mean acceleration
    fixation_cluster = int(np.argmin(np.sum(mean_values[:, 1:3], axis=1)))

    fixation_mask = labels == fixation_cluster

    # Secondary fixation clusters based on velocity overlap, matching MATLAB logic.
    secondary_fixation_clusters = np.where(
        mean_values[:, 1] < (
            mean_values[fixation_cluster, 1]
            + float(config.fixation_cluster_velocity_std_multiplier) * std_values[fixation_cluster, 1]
        )
    )[0]
    fixation_mask |= np.isin(labels, secondary_fixation_clusters)

    fixation_indices = np.where(fixation_mask)[0]
    return fixation_indices.astype(int)


def _determine_optimal_clusters_global(
    data: np.ndarray,
    config: FixationDetectionConfig,
    random_state: int,
) -> int:
    """
    Match the MATLAB global-k selection logic:
    evaluate k = 2..5 on decimated data using the InterVSIntraDist metric.
    """
    min_k = int(config.global_kmeans_k_min)
    max_k = int(config.global_kmeans_k_max)
    sampled = data[:: int(config.global_decimation_step)]
    if sampled.shape[0] < int(config.global_min_sampled_points):
        return min_k

    sil = np.full(max_k + 1, -np.inf, dtype=float)
    for k in range(min_k, max_k + 1):
        if sampled.shape[0] < k:
            continue
        labels = KMeans(
            n_clusters=k,
            n_init=int(config.kmeans_n_init),
            random_state=random_state,
        ).fit_predict(sampled)
        silh = _inter_vs_intra_dist(sampled, labels)
        sil[k] = np.mean(silh)

    valid = sil[min_k : max_k + 1]
    valid_finite = valid[np.isfinite(valid)]
    if valid_finite.size == 0:
        return min_k

    max_sil = np.max(valid_finite)
    threshold = float(config.silhouette_threshold_fraction) * max_sil
    sil[min_k : max_k + 1] = np.where(
        sil[min_k : max_k + 1] > threshold,
        1.0,
        sil[min_k : max_k + 1],
    )
    num_clusters = np.where(sil == np.max(sil))[0]
    num_clusters = num_clusters[(num_clusters >= min_k) & (num_clusters <= max_k)]

    if num_clusters.size == 0:
        return min_k
    return int(num_clusters[-1])  # MATLAB takes the last one


def _refine_fixation_classification_to_get_notfix_inds(
    fixation_start_stop: np.ndarray,
    feature_matrix: np.ndarray,
    sampling_rate_hz: float,
    config: FixationDetectionConfig,
    random_state: int,
) -> np.ndarray:
    """
    Local reclustering step, adapted from the MATLAB ClusterFix script.

    Uses medians and cluster ranges rather than global mean/std logic.
    """
    non_fixation_indices = []

    padding_samples = _samples_from_duration_ms(
        config.local_recluster_padding_duration_ms,
        sampling_rate_hz,
        minimum=0,
    )
    min_k = int(config.local_kmeans_k_min)
    max_k = int(config.local_kmeans_k_max)

    for i in range(fixation_start_stop.shape[0]):
        start, stop = fixation_start_stop[i]
        surrounding_indices = np.arange(start - padding_samples, stop + padding_samples + 1)
        surrounding_indices = surrounding_indices[
            (surrounding_indices >= 0) & (surrounding_indices < feature_matrix.shape[0])
        ]

        local_features = feature_matrix[surrounding_indices, :]
        if local_features.shape[0] < int(config.local_min_feature_samples):
            continue

        sil = np.full(max_k + 1, -np.inf, dtype=float)
        local_sampled = local_features[:: int(config.local_decimation_step)]

        for k in range(1, max_k + 1):
            if k == 1:
                sil[k] = 0.0
                continue
            if local_sampled.shape[0] < k:
                sil[k] = -np.inf
                continue

            labels = KMeans(
                n_clusters=k,
                n_init=int(config.kmeans_n_init),
                random_state=random_state,
            ).fit_predict(local_sampled)
            silh = _inter_vs_intra_dist(local_sampled, labels)
            sil[k] = np.mean(silh)

        valid = sil[min_k : max_k + 1]
        valid_finite = valid[np.isfinite(valid)]
        if valid_finite.size == 0:
            optimal_clusters = min_k
        else:
            max_sil = np.max(valid_finite)
            threshold = float(config.silhouette_threshold_fraction) * max_sil
            sil[min_k : max_k + 1] = np.where(
                sil[min_k : max_k + 1] > threshold,
                1.0,
                sil[min_k : max_k + 1],
            )
            num_clusters = np.where(sil == np.max(sil))[0]
            num_clusters = num_clusters[(num_clusters >= min_k) & (num_clusters <= max_k)]
            optimal_clusters = int(math.ceil(np.median(num_clusters))) if num_clusters.size else min_k
            optimal_clusters = max(min_k, optimal_clusters)

        cluster_labels = KMeans(
            n_clusters=optimal_clusters,
            n_init=int(config.kmeans_n_init),
            random_state=random_state,
        ).fit_predict(local_features)

        median_values = np.zeros((optimal_clusters, local_features.shape[1]), dtype=float)
        cluster_ranges = np.zeros((optimal_clusters, 2 * (local_features.shape[1] - 1)), dtype=float)

        for k in range(optimal_clusters):
            cluster_points = local_features[cluster_labels == k]
            if cluster_points.shape[0] == 0:
                median_values[k, :] = np.nan
                cluster_ranges[k, :] = np.nan
            elif cluster_points.shape[0] == 1:
                median_values[k, :] = cluster_points[0]
                cluster_ranges[k, :] = np.ones(2 * (local_features.shape[1] - 1))
            else:
                median_values[k, :] = np.median(cluster_points, axis=0)
                # Match MATLAB rng = [max(POINTS(:,1:end-1)) min(POINTS(:,1:end-1))]
                max_part = np.max(cluster_points[:, :-1], axis=0)
                min_part = np.min(cluster_points[:, :-1], axis=0)
                cluster_ranges[k, :] = np.concatenate([max_part, min_part])

        valid_rows = ~np.isnan(median_values).any(axis=1)
        if not np.any(valid_rows):
            continue

        fixation_cluster = int(np.nanargmin(np.sum(median_values[:, 1:3], axis=1)))
        fixation_mask = cluster_labels == fixation_cluster

        # MATLAB overlap test using velocity and acceleration ranges.
        # cluster_ranges columns for 4-feature input:
        # [max_dist, max_vel, max_accel, min_dist, min_vel, min_accel]
        fixation_vel = median_values[fixation_cluster, 1]
        fixation_accel = median_values[fixation_cluster, 2]

        secondary_clusters = np.where(
            (fixation_vel < cluster_ranges[:, 1]) &
            (fixation_vel > cluster_ranges[:, 4]) &
            (fixation_accel < cluster_ranges[:, 2]) &
            (fixation_accel > cluster_ranges[:, 5])
        )[0]

        secondary_clusters = secondary_clusters[secondary_clusters != fixation_cluster]
        fixation_mask |= np.isin(cluster_labels, secondary_clusters)

        non_fixation_indices.extend(surrounding_indices[~fixation_mask])

    if len(non_fixation_indices) == 0:
        return np.array([], dtype=int)

    return np.unique(np.asarray(non_fixation_indices, dtype=int))


def _extract_behavior_intervals(indices: np.ndarray) -> np.ndarray:
    """
    Convert sorted sample indices into inclusive [start, stop] intervals.
    """
    if indices.size == 0:
        return np.empty((0, 2), dtype=int)

    indices = np.sort(np.unique(indices))
    diffs = np.diff(indices)
    gap_locs = np.where(diffs > 1)[0]

    starts = indices[np.insert(gap_locs + 1, 0, 0)]
    stops = indices[np.append(gap_locs, len(indices) - 1)]

    return np.column_stack((starts, stops)).astype(int)


def _filter_behavior_intervals(intervals: np.ndarray, min_duration_samples: int) -> np.ndarray:
    """
    Keep intervals whose inclusive duration is at least min_duration_samples.
    """
    if intervals.size == 0:
        return intervals

    durations = intervals[:, 1] - intervals[:, 0] + 1
    return intervals[durations >= min_duration_samples]


def _inter_vs_intra_dist(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Python version of the MATLAB InterVSIntraDist helper.

    Computes a silhouette-like quantity using squared Euclidean distances.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    n = X.shape[0]
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    k = unique_labels.size

    counts = np.bincount(inverse, minlength=k)
    membership = np.equal.outer(inverse, np.arange(k))

    avg_within = np.full(n, np.inf, dtype=float)
    avg_between = np.full((n, k), np.inf, dtype=float)

    for j in range(n):
        diff = X - X[j]
        distj = np.sum(diff * diff, axis=1)

        for i in range(k):
            mask = membership[:, i]
            if i == inverse[j]:
                denom = max(counts[i] - 1, 1)
                avg_within[j] = np.sum(distj[mask]) / denom
            else:
                avg_between[j, i] = np.sum(distj[mask]) / counts[i]

    min_avg_between = np.min(avg_between, axis=1)
    denom = np.maximum(avg_within, min_avg_between)

    silh = np.zeros(n, dtype=float)
    valid = denom > 0
    silh[valid] = (min_avg_between[valid] - avg_within[valid]) / denom[valid]
    return silh
