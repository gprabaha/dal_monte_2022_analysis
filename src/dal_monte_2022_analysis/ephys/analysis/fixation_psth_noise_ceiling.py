"""How much of the fixation PSTH export is reproducible signal?

Every reconstruction score in the mRNN work is an $R^2$ against a trial-averaged PSTH,
implicitly scored against a ceiling of 1.0. That is only right if the PSTH is noise-free.
It is not: the export's four condition groups differ in trial count by more than 12x
(interactive face ~846 trials per unit, object non-interactive ~68), and across-bin
variance scales almost exactly as 1/N across them, which is the signature of estimation
noise rather than of response structure.

This module measures the ceiling directly. For each unit and condition it splits the
trials in half, builds two independent mean traces through **the same pipeline the target
export uses** -- per-trial Gaussian smoothing at 20 ms, then averaging, then windowing --
and correlates them. Spearman-Brown then gives the reliability of the full-trial average.

The result is the number every model $R^2$ should be read against: a model that reaches
the ceiling has extracted everything the data determines, and a model above it is fitting
sampling noise.

Nothing here refits a model. The expensive step is one pass over the per-session trial
files, which is cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.io import (
    load_pickle_path,
    scan_processed_paths_for_filename,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir

#: The model's three conditions, and how each is selected from the trial table. ``object``
#: is deliberately the *unsplit* pool over interactive state, matching what
#: ``fixation_mrnn_bridge`` feeds the model.
CONDITION_SPECS: dict[str, dict[str, object]] = {
    "face_interactive": {"fixation_category": "face", "interactive_state": "interactive"},
    "face_non_interactive": {"fixation_category": "face", "interactive_state": "non_interactive"},
    "object": {"fixation_category": "object", "interactive_state": None},
}
CONDITION_ORDER: tuple[str, ...] = tuple(CONDITION_SPECS)


@dataclass(frozen=True)
class NoiseCeilingSettings:
    """Pipeline parameters, which must match the target export exactly.

    A ceiling computed under different smoothing or windowing is a ceiling for a
    different quantity, so these default to the values in ``configs/ephys_fixation_psth.yaml``
    that produced the combined export the mRNN is fit to.
    """

    cfg_path: str = "configs/dataset.yaml"
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations_psth_10ms.pkl"
    output_subdir: str = "ephys/psth/fixation_noise_ceiling"
    #: Per-trial Gaussian smoothing applied before averaging, in milliseconds.
    smoothing_sigma_ms: float = 20.0
    bin_size_ms: float = 10.0
    #: Trial PSTHs span +/-1000 ms; the model target is the inner +/-500 ms.
    window_start_s: float = -0.5
    window_stop_s: float = 0.5
    trial_window_start_s: float = -1.0
    #: Independent random halvings averaged per unit and condition. More draws reduce the
    #: variance of the reliability estimate; they cannot reduce its bias.
    n_splits: int = 25
    #: Units with fewer trials than this in a condition give a reliability estimate that is
    #: itself mostly noise.
    min_trials: int = 20
    random_seed: int = 20260903
    max_procs: int = 16


def _condition_for_row(category: object, interactive_state: object) -> str | None:
    """Which model condition a trial belongs to, if any."""
    category_token = str(category).strip().lower()
    state_token = str(interactive_state).strip().lower()
    if category_token == "object":
        return "object"
    if category_token != "face":
        return None
    if state_token == "interactive":
        return "face_interactive"
    if state_token in {"non_interactive", "noninteractive"}:
        return "face_non_interactive"
    return None


def _window_slice(settings: NoiseCeilingSettings, n_bins: int) -> slice:
    """Bins of the trial PSTH that fall inside the target window."""
    bin_size_s = float(settings.bin_size_ms) / 1000.0
    centers = float(settings.trial_window_start_s) + (np.arange(n_bins) + 0.5) * bin_size_s
    keep = np.flatnonzero(
        (centers >= float(settings.window_start_s)) & (centers < float(settings.window_stop_s))
    )
    if keep.size == 0:
        raise ValueError("Target window does not overlap the trial PSTH window.")
    return slice(int(keep[0]), int(keep[-1]) + 1)


def _smoothed_trials(counts: np.ndarray, settings: NoiseCeilingSettings) -> np.ndarray:
    """Smooth each trial exactly as the export does, then window."""
    sigma_bins = float(settings.smoothing_sigma_ms) / float(settings.bin_size_ms)
    smoothed = gaussian_filter1d(counts, sigma=sigma_bins, axis=-1, mode="nearest")
    return smoothed[..., _window_slice(settings, counts.shape[-1])]


def split_half_traces(
    trials: np.ndarray,
    *,
    settings: NoiseCeilingSettings,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Two independent half-averages of one unit's trials for one condition.

    Returns arrays of shape ``(n_splits, n_bins)``. Splitting at the *trial* level, and
    smoothing before averaging, is what makes the two halves independent in the same way
    the full average's noise is independent of the signal.
    """
    windowed = _smoothed_trials(np.asarray(trials, dtype=float), settings)
    n_trials = windowed.shape[0]
    half = n_trials // 2
    first = np.empty((int(settings.n_splits), windowed.shape[1]), dtype=float)
    second = np.empty_like(first)
    for draw in range(int(settings.n_splits)):
        order = rng.permutation(n_trials)
        first[draw] = windowed[order[:half]].mean(axis=0)
        second[draw] = windowed[order[half : 2 * half]].mean(axis=0)
    return first, second


def _spearman_brown(correlation: float, *, factor: float = 2.0) -> float:
    """Reliability of a full-length average from the correlation of two halves."""
    if not np.isfinite(correlation):
        return np.nan
    denominator = 1.0 + (factor - 1.0) * correlation
    if abs(denominator) < 1e-12:
        return np.nan
    return float(factor * correlation / denominator)


def _pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between two ``(n_draws, n_bins)`` arrays."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    numerator = np.sum(a * b, axis=1)
    denominator = np.sqrt(np.sum(a**2, axis=1) * np.sum(b**2, axis=1))
    return np.divide(numerator, np.maximum(denominator, 1e-12), where=denominator > 0)


def _date_worker(args) -> list[dict[str, object]]:
    paths, settings = args
    rng = np.random.default_rng(int(settings.random_seed))
    frames: list[pd.DataFrame] = []
    for path in paths:
        blob = load_pickle_path(path)
        table = blob.get("trials") if isinstance(blob, dict) else blob
        if isinstance(table, pd.DataFrame) and not table.empty:
            frames.append(table)
    if not frames:
        return []
    trials = pd.concat(frames, ignore_index=True)
    if "psth_counts" not in trials.columns or "unit_uuid" not in trials.columns:
        return []

    trials = trials.assign(
        _condition=[
            _condition_for_row(category, state)
            for category, state in zip(trials["fixation_category"], trials["interactive_state"])
        ]
    )
    trials = trials[trials["_condition"].notna()]

    rows: list[dict[str, object]] = []
    for (unit, condition), block in trials.groupby(["unit_uuid", "_condition"], sort=False):
        counts = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in block["psth_counts"]])
        if counts.shape[0] < int(settings.min_trials) or not np.isfinite(counts).all():
            continue
        first, second = split_half_traces(counts, settings=settings, rng=rng)
        half_correlation = float(np.nanmean(_pearson(first, second)))
        rows.append(
            {
                "unit_uuid": str(unit),
                "region": str(block["region"].iloc[0]).strip().lower(),
                "condition": str(condition),
                "n_trials": int(counts.shape[0]),
                "half_correlation": half_correlation,
                "reliability": _spearman_brown(half_correlation),
                # Mean of the two half traces is the best available estimate of the signal;
                # its variance is the numerator of any ceiling-corrected score.
                "mean_trace": 0.5 * (first.mean(axis=0) + second.mean(axis=0)),
                "half_a": first.mean(axis=0),
                "half_b": second.mean(axis=0),
            }
        )
    return rows


def compute_unit_noise_ceiling(
    settings: NoiseCeilingSettings = NoiseCeilingSettings(),
    *,
    cache_path: str | Path | None = None,
    refresh: bool = False,
    dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Split-half reliability per unit and condition, over the whole dataset.

    One row per unit x condition, carrying the reliability and the two half traces the
    downstream ceilings are built from. The pass over the per-session trial files is the
    expensive step and is cached.
    """
    if cache_path is not None and not refresh and Path(cache_path).exists():
        return pd.read_pickle(cache_path)

    cfg = load_config(settings.cfg_path)
    entries = scan_processed_paths_for_filename(
        cfg, modality=settings.trial_input_modality, filename=settings.trial_input_filename
    )
    by_date: dict[str, list[Path]] = {}
    for entry in entries:
        date = str(entry["date"])
        if dates is not None and date not in set(dates):
            continue
        by_date.setdefault(date, []).append(Path(entry["path"]))

    tasks = [(paths, settings) for _, paths in sorted(by_date.items())]
    rows: list[dict[str, object]] = []
    if settings.max_procs and settings.max_procs > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(settings.max_procs)) as pool:
            for result in pool.map(_date_worker, tasks):
                rows.extend(result)
    else:
        for task in tasks:
            rows.extend(_date_worker(task))

    table = pd.DataFrame(rows)
    if cache_path is not None and not table.empty:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_pickle(cache_path)
    return table


def region_condition_ceiling(unit_ceiling: pd.DataFrame) -> pd.DataFrame:
    """Median reliability per region and condition, with the trial counts behind it."""
    return (
        unit_ceiling.groupby(["region", "condition"])
        .agg(
            n_units=("unit_uuid", "size"),
            median_trials=("n_trials", "median"),
            median_half_correlation=("half_correlation", "median"),
            median_reliability=("reliability", "median"),
            reliability_q25=("reliability", lambda v: float(np.nanpercentile(v, 25))),
            reliability_q75=("reliability", lambda v: float(np.nanpercentile(v, 75))),
        )
        .reset_index()
    )


def _stack_halves(
    unit_ceiling: pd.DataFrame,
    region: str,
    unit_ids: Sequence[str],
    conditions: Sequence[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(condition x time, units)`` matrices for the two independent half-averages."""
    block = unit_ceiling[unit_ceiling["region"] == str(region).strip().lower()]
    lookup = {
        (str(unit), str(condition)): index
        for index, (unit, condition) in enumerate(zip(block["unit_uuid"], block["condition"]))
    }
    half_a_all = list(block["half_a"])
    half_b_all = list(block["half_b"])
    panels_a, panels_b = [], []
    for condition in conditions:
        indices = [lookup.get((str(unit), str(condition))) for unit in unit_ids]
        if any(index is None for index in indices):
            return None
        panels_a.append(np.stack([half_a_all[i] for i in indices], axis=-1))
        panels_b.append(np.stack([half_b_all[i] for i in indices], axis=-1))
    return np.concatenate(panels_a, axis=0), np.concatenate(panels_b, axis=0)


def population_ceiling_in_pc_space(
    unit_ceiling: pd.DataFrame,
    unit_order_by_region: Mapping[str, Sequence[str]],
    *,
    n_components: int = 42,
    conditions: Sequence[str] = CONDITION_ORDER,
    noise_control: bool = True,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Ceiling in the PC space the model is actually fit in.

    Per-unit reliability is not what the mRNN is scored on -- region PC trajectories are,
    and averaging a few hundred units cancels most of the independent single-unit noise.
    So the two ceilings are very different numbers, and only this one bounds the model.

    **The basis is fit on half A alone** and applied to both halves. Fitting it on the
    full average would let each half be scored on directions partly derived from its own
    noise, which inflates the estimate; the ``noise_control`` column measures exactly
    that, by running the identical procedure on two independent Gaussian half-matrices of
    the same shape. A trustworthy ceiling has a control near zero.
    """
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, object]] = []
    for region, unit_ids in unit_order_by_region.items():
        stacked = _stack_halves(unit_ceiling, region, unit_ids, conditions)
        if stacked is None:
            continue
        half_a, half_b = stacked
        rank = min(int(n_components), min(half_a.shape) - 1)

        mean_a = half_a.mean(axis=0)
        _, _, basis = np.linalg.svd(half_a - mean_a, full_matrices=False)
        basis = basis[:rank]
        scores_a = (half_a - mean_a) @ basis.T
        scores_b = (half_b - mean_a) @ basis.T
        correlations = _pearson(scores_a.T, scores_b.T)

        control = np.full(rank, np.nan)
        if noise_control:
            noise_a = rng.normal(size=half_a.shape)
            noise_b = rng.normal(size=half_a.shape)
            noise_mean = noise_a.mean(axis=0)
            _, _, noise_basis = np.linalg.svd(noise_a - noise_mean, full_matrices=False)
            noise_basis = noise_basis[:rank]
            control = _pearson(
                ((noise_a - noise_mean) @ noise_basis.T).T,
                ((noise_b - noise_mean) @ noise_basis.T).T,
            )

        for component in range(rank):
            rows.append(
                {
                    "region": region,
                    "component": int(component),
                    "half_correlation": float(correlations[component]),
                    "reliability": _spearman_brown(float(correlations[component])),
                    "noise_control": float(control[component]),
                }
            )
    return pd.DataFrame(rows)


def ceiling_corrected_r2(observed_r2: float, reliability: float) -> float:
    """Model $R^2$ expressed as a fraction of what the data can determine.

    A model that reproduces exactly the reproducible part of the signal scores 1.0 here.
    Above 1.0 means it is reproducing sampling noise, which is a diagnosis and not an
    achievement.
    """
    if not np.isfinite(reliability) or reliability <= 0:
        return np.nan
    return float(observed_r2 / reliability)


def resolve_output_dir(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    output_subdir: str = "ephys/psth/fixation_noise_ceiling",
) -> Path:
    """Analysis-output directory for the noise-ceiling products."""
    path = build_analysis_output_dir(load_config(cfg_path), output_subdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "CONDITION_ORDER",
    "CONDITION_SPECS",
    "NoiseCeilingSettings",
    "ceiling_corrected_r2",
    "compute_unit_noise_ceiling",
    "population_ceiling_in_pc_space",
    "region_condition_ceiling",
    "resolve_output_dir",
    "split_half_traces",
]


def population_ceiling_by_cell(
    unit_ceiling: pd.DataFrame,
    *,
    pca_by_region: Mapping[str, Mapping[str, object]],
    normalization_scale: float | None,
    conditions: Sequence[str] = CONDITION_ORDER,
    trace_to_target_scale: float = 100.0,
    noise_control: bool = True,
    random_seed: int = 0,
) -> pd.DataFrame:
    """The ceiling the model's R^2 is actually bounded by, per region x condition.

    ``trace_to_target_scale`` converts the stored half-traces into the units of the
    training export: the ceiling's per-trial pipeline leaves traces in spikes per 10 ms
    bin, the target export is in Hz, so the factor is 100. It is a units convention, not a
    data difference -- the two agree to correlation 1.000 once applied -- and because every
    statistic here is a correlation it does not change the ceiling; it only makes the
    projected traces reproduce the training target exactly, which is the check that the
    stored halves are the model's own data.

    :func:`population_ceiling_in_pc_space` estimates reliability per component in a basis
    fitted on half A, then the chapter averaged those 42 numbers without weighting and
    pooled the conditions. Three mismatches with the R^2 it was used to divide:

    * the model is scored in the PCA basis fitted on the **full** average, not on half A;
    * R^2 sums squared error over components, so it is **variance-weighted** -- dominated
      by the leading components, whose reliability is ~0.999 -- while an unweighted mean
      over 42 components is pulled down by the tail at 0.91-0.95. Dividing a top-heavy R^2
      by a tail-heavy mean is how R^2 / ceiling came out above 1;
    * conditions differ five-fold in trial count, so their reliabilities differ, and a
      pooled ceiling over-rates the fit on the best-measured condition and under-rates the
      others.

    This function projects the two independent half-averages onto the model's own basis,
    centres per component over the pooled (condition x time) trajectory exactly as the R^2
    does, and takes the split-half correlation of the **whole trajectory** -- covariances
    and variances summed over components, which weights each by its variance as R^2 does --
    for every region x condition cell and for each region pooled. Spearman-Brown then gives
    the reliability of the full average, which is the R^2 a perfect model could reach.

    ``normalization_scale`` is the scalar the training targets were divided by; it is
    needed only so the stored PCA mean is subtracted in matching units. ``noise_control``
    runs the identical projection on Gaussian half-matrices and should sit near zero.
    """
    rng = np.random.default_rng(int(random_seed))
    scale = float(normalization_scale) if normalization_scale else 1.0
    rows: list[dict[str, object]] = []

    def pooled_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
        if mask is not None:
            a, b = a[mask], b[mask]
        denominator = float(np.sqrt((a**2).sum() * (b**2).sum()))
        return float((a * b).sum() / denominator) if denominator > 0 else float("nan")

    for region, meta in pca_by_region.items():
        units = [str(u) for u in meta["source_features"]]
        stacked = _stack_halves(unit_ceiling, region, units, conditions)
        if stacked is None:
            continue
        half_a, half_b = stacked
        mean = np.asarray(meta["mean"], dtype=float)
        components = np.asarray(meta["components"], dtype=float)
        scores_a = (half_a * float(trace_to_target_scale) / scale - mean) @ components.T
        scores_b = (half_b * float(trace_to_target_scale) / scale - mean) @ components.T
        # Centred per component over the pooled trajectory, as the R^2 denominator is.
        centred_a = scores_a - scores_a.mean(axis=0)
        centred_b = scores_b - scores_b.mean(axis=0)
        n_time = scores_a.shape[0] // len(conditions)

        per_component = np.array([
            float(np.corrcoef(centred_a[:, j], centred_b[:, j])[0, 1]) for j in range(scores_a.shape[1])
        ])
        weight = (centred_a**2).sum(axis=0) + (centred_b**2).sum(axis=0)
        sb_components = np.array([_spearman_brown(float(r)) for r in per_component])
        control = float("nan")
        if noise_control:
            noise_a = rng.normal(size=half_a.shape) @ components.T
            noise_b = rng.normal(size=half_b.shape) @ components.T
            control = pooled_corr(noise_a - noise_a.mean(axis=0), noise_b - noise_b.mean(axis=0))

        pooled = pooled_corr(centred_a, centred_b)
        rows.append({
            "region": region, "condition": "all",
            "half_correlation": pooled, "reliability": _spearman_brown(pooled),
            "unweighted_component_mean": float(np.mean(sb_components)),
            "variance_weighted_component_mean": float((sb_components * weight).sum() / weight.sum()),
            "noise_control": control, "n_components": int(scores_a.shape[1]),
        })
        for index, condition in enumerate(conditions):
            mask = np.zeros(scores_a.shape[0], dtype=bool)
            mask[index * n_time:(index + 1) * n_time] = True
            r = pooled_corr(centred_a, centred_b, mask)
            rows.append({
                "region": region, "condition": str(condition),
                "half_correlation": r, "reliability": _spearman_brown(r),
                "unweighted_component_mean": np.nan, "variance_weighted_component_mean": np.nan,
                "noise_control": np.nan, "n_components": int(scores_a.shape[1]),
            })
    return pd.DataFrame(rows)
