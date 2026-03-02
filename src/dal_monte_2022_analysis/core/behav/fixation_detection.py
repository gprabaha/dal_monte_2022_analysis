"""Fixation and saccade detection domain logic."""

import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def detect_fixations_and_saccades(positions: np.ndarray):
    """Detect fixation and saccade intervals using k-means clustering.

    Args:
        positions: Array of shape (N, 2) containing [x, y] samples. At least
            500 samples are required; shorter inputs return empty outputs.

    Returns:
        Tuple of (fixation_start_stop, saccade_start_stop), each an (M, 2) array
        of inclusive start/stop indices.
    """
    if positions.shape[0] < 500:
        print("Insufficient data points (< 500), returning empty arrays.")
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    print("Preprocessing positions data for fixation detection")
    x, y = _apply_lowpass_filter(positions)

    print("Extracting motion parameters for k-means clustering")
    feature_matrix = _compute_motion_features(x, y)

    print("Normalizing parameters for k-means clustering")
    feature_matrix = _normalize_features(feature_matrix)

    print("Performing global clustering of points for 2 to 5 cluster sizes")
    fixation_indices = _extract_fixation_indices_through_global_k_means(feature_matrix)

    fixation_start_stop = _extract_behavior_intervals(fixation_indices)

    print("Refining fixation start-stop indices using local reclustering")
    not_fixations = _refine_fixation_classification_to_get_notfix_inds(
        fixation_start_stop,
        feature_matrix,
    )
    fixation_indices = np.setdiff1d(fixation_indices, not_fixations)
    saccade_indices = np.setdiff1d(np.arange(len(feature_matrix)), fixation_indices)

    fixation_start_stop = _extract_behavior_intervals(fixation_indices + 1)
    fixation_start_stop = _filter_behavior_intervals(fixation_start_stop, min_duration=25)
    saccade_start_stop = _extract_behavior_intervals(saccade_indices + 1)
    saccade_start_stop = _filter_behavior_intervals(saccade_start_stop, min_duration=10)

    fixation_start_stop = (
        fixation_start_stop if fixation_start_stop.size else np.empty((0, 2), dtype=int)
    )
    saccade_start_stop = (
        saccade_start_stop if saccade_start_stop.size else np.empty((0, 2), dtype=int)
    )

    return fixation_start_stop, saccade_start_stop


def _apply_lowpass_filter(positions: np.ndarray):
    """Apply a low-pass FIR filter to smooth eye movement data.

    Returns:
        Tuple of filtered x and y arrays with the original length.
    """
    fltord = 60
    lowpass_freq = 30
    nyquist_freq = 500
    buffer = 100
    x = np.pad(positions[:, 0], (buffer, buffer), "reflect")
    y = np.pad(positions[:, 1], (buffer, buffer), "reflect")
    flt = signal.firwin(fltord, cutoff=lowpass_freq / nyquist_freq, pass_zero=True)
    x = signal.filtfilt(flt, 1, x)
    y = signal.filtfilt(flt, 1, y)
    x = x[buffer:-buffer]
    y = y[buffer:-buffer]
    return x, y


def _compute_motion_features(xss: np.ndarray, yss: np.ndarray):
    """Compute motion features (displacement, velocity, acceleration, rotation).

    Returns:
        Feature matrix with shape (N-2, 4).
    """
    velx, vely = np.diff(xss), np.diff(yss)
    vel = np.sqrt(velx**2 + vely**2)
    accel = np.abs(np.diff(vel))
    angle = np.degrees(np.arctan2(vely, velx))
    rot = np.zeros(len(xss) - 2)
    displacement = np.zeros(len(xss) - 2)
    for i in range(len(xss) - 2):
        rot[i] = np.abs(angle[i] - angle[i + 1])
        displacement[i] = np.sqrt((xss[i] - xss[i + 2]) ** 2 + (yss[i] - yss[i + 2]) ** 2)
    rot[rot > 180] -= 180
    rot = 360 - rot
    vel = vel[:-1]
    return np.column_stack((displacement, vel, accel, rot))


def _normalize_features(feature_matrix: np.ndarray):
    """Normalize feature values to [0, 1] with simple outlier clipping.

    Note:
        Normalization is performed in place and the same array is returned.
    """
    for i in range(feature_matrix.shape[1]):
        threshold = np.mean(feature_matrix[:, i]) + 3 * np.std(feature_matrix[:, i])
        feature_matrix[feature_matrix[:, i] > threshold, i] = threshold
        feature_matrix[:, i] -= np.min(feature_matrix[:, i])
        feature_matrix[:, i] /= np.max(feature_matrix[:, i])
    return feature_matrix


def _extract_fixation_indices_through_global_k_means(feature_matrix: np.ndarray):
    """Perform global k-means clustering to classify fixations vs saccades."""
    num_clusters = _determine_optimal_clusters(feature_matrix[:, 1:4])
    labels = KMeans(n_clusters=num_clusters, n_init=5).fit_predict(feature_matrix)
    unique_clusters = np.unique(labels)
    mean_values = np.array([np.mean(feature_matrix[labels == k], axis=0) for k in unique_clusters])
    std_values = np.array([np.std(feature_matrix[labels == k], axis=0) for k in unique_clusters])
    fixation_cluster = np.argmin(np.sum(mean_values[:, 1:3], axis=1))
    labels[labels == fixation_cluster] = 100
    secondary_fixation_clusters = np.where(
        mean_values[:, 1] < mean_values[fixation_cluster, 1] + 3 * std_values[fixation_cluster, 1]
    )[0]
    labels[np.isin(labels, secondary_fixation_clusters)] = 100
    labels[labels != 100] = 2
    labels[labels == 100] = 1
    fixation_indices = np.where(labels == 1)[0]
    print("Found fixation indices from global clustering")
    return fixation_indices


def _determine_optimal_clusters(data: np.ndarray):
    """Determine optimal number of clusters using silhouette scoring."""
    best_sil, best_k = -1, 2
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, n_init=5).fit(data)
        sil_score = silhouette_score(data, kmeans.labels_)
        if sil_score > best_sil:
            best_sil, best_k = sil_score, k
    print(f"Optimal k-Means cluster number was found to be: {best_k}")
    return best_k


def _refine_fixation_classification_to_get_notfix_inds(
    fixation_start_stop: np.ndarray,
    feature_matrix: np.ndarray,
):
    """Refine fixation classification using local re-clustering windows."""
    non_fixation_indices = []
    for i in range(fixation_start_stop.shape[0]):
        surrounding_indices = np.arange(fixation_start_stop[i, 0] - 50, fixation_start_stop[i, 1] + 50)
        surrounding_indices = surrounding_indices[
            (surrounding_indices >= 0) & (surrounding_indices < len(feature_matrix))
        ]
        local_features = feature_matrix[surrounding_indices, :]
        silhouette_scores = []
        for num_clusters in range(2, 6):
            kmeans = KMeans(n_clusters=num_clusters, n_init=5, random_state=42)
            labels = kmeans.fit_predict(local_features[::5, :])
            silhouette_scores.append(silhouette_score(local_features[::5, :], labels))
        silhouette_scores = np.array(silhouette_scores)
        silhouette_scores[silhouette_scores > 0.9 * np.max(silhouette_scores)] = 1
        optimal_clusters = np.argmax(silhouette_scores) + 2
        kmeans = KMeans(n_clusters=optimal_clusters, n_init=5, random_state=42)
        cluster_labels = kmeans.fit_predict(local_features)
        cluster_medians = np.array(
            [np.median(local_features[cluster_labels == k, :], axis=0) for k in range(optimal_clusters)]
        )
        cluster_ranges = np.array(
            [
                [
                    np.max(local_features[cluster_labels == k, :], axis=0),
                    np.min(local_features[cluster_labels == k, :], axis=0),
                ]
                if np.any(cluster_labels == k)
                else np.ones((2, local_features.shape[1]))
                for k in range(optimal_clusters)
            ]
        )
        primary_fixation_cluster = np.argmin(np.sum(cluster_medians[:, 1:3], axis=1))
        cluster_labels[cluster_labels == primary_fixation_cluster] = 100
        secondary_fixation_clusters = np.where(
            (cluster_ranges[:, 0, 1] > cluster_medians[primary_fixation_cluster, 1])
            & (cluster_ranges[:, 1, 1] < cluster_medians[primary_fixation_cluster, 1])
            & (cluster_ranges[:, 0, 2] > cluster_medians[primary_fixation_cluster, 2])
            & (cluster_ranges[:, 1, 2] < cluster_medians[primary_fixation_cluster, 2])
        )[0]
        secondary_fixation_clusters = secondary_fixation_clusters[
            secondary_fixation_clusters != primary_fixation_cluster
        ]
        for fixation_cluster in secondary_fixation_clusters:
            cluster_labels[cluster_labels == fixation_cluster] = 100
        cluster_labels[cluster_labels != 100] = 2
        cluster_labels[cluster_labels == 100] = 1
        non_fixation_indices.extend(surrounding_indices[cluster_labels == 2])
    print("Found not-fixation indices from local re-clustering")
    return np.array(non_fixation_indices)


def _extract_behavior_intervals(indices: np.ndarray):
    """Convert sorted index sequences into continuous [start, stop] intervals."""
    if len(indices) == 0:
        return np.empty((0, 2), dtype=int)
    diffs = np.diff(indices)
    gaps = np.where(diffs > 1)[0]
    times = np.vstack(
        (indices[np.insert(gaps + 1, 0, 0)], indices[np.append(gaps, len(indices) - 1)])
    ).T
    return times


def _filter_behavior_intervals(intervals: np.ndarray, min_duration: int):
    """Remove intervals shorter than the minimum duration."""
    if intervals.size == 0:
        return intervals
    return intervals[np.where((intervals[:, 1] - intervals[:, 0]) >= min_duration)]

