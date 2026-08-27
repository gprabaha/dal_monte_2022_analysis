"""Score the temporal specificity of average fixation PSTHs.

``fixation_peakiness`` scores a trace with a single family of statistics derived
from ``scipy.signal.find_peaks`` prominences: the largest prominence ``p1``, the
largest prominence outside an exclusion window ``p2``, and the scaled ratio
``p1 / (1 + lambda * p2 / p1)``. That captures "one tall peak that has no rival"
but conflates two distinct things the data actually show: a few units with a
genuinely narrow, isolated transient, and many units with a broad but sustained
elevation. It also says nothing about how ragged a trace is, even though most
units fluctuate with several comparable peaks.

This module scores the same average traces along explicitly separated axes:

``concentration``
    How much of the excess (above-baseline) response mass sits in a small
    fraction of the window. ``mass_width_frac_50``, ``effective_width_ms`` and
    ``lifetime_sparseness`` all measure this, and all are width-like rather than
    amplitude-like, so a low-rate unit with a crisp transient scores as highly
    concentrated.

``single-peak dominance``
    Whether the concentration is carried by *one* excursion. ``peak_dominance``
    generalizes the existing ``p1 / (p1 + p2)`` to all detected peaks
    (``p1 / sum(p)``), and ``n_prominent_peaks`` counts peaks whose prominence
    clears a fraction of the largest.

``sustainedness``
    How long the trace stays high: ``fwhm_frac`` (fraction of the window at or
    above half the excess peak) and ``sustained_frac`` (fraction above a fixed
    excess threshold). These are deliberately *anti*-correlated with the
    concentration axis, so a broad plateau is scored as sustained rather than
    being called "not peaky".

``fluctuation``
    How ragged the trace is independent of its envelope: ``roughness`` is the
    total variation normalized by the peak-to-trough range, i.e. roughly the
    number of monotone excursions, and ``autocorr_width_ms`` is the half-width
    of the trace autocorrelation.

``amplitude``
    ``modulation_index`` and ``peak_z`` say how strong the modulation is at all,
    so a flat unit can be excluded before its shape statistics are interpreted.

Two composites are provided. ``temporal_specificity_index`` rewards a single
concentrated peak and penalizes raggedness; ``sustainedness_index`` rewards a
broad, smooth elevation. They are intentionally not negatives of each other -- a
unit can be low on both (flat) or high on neither (ragged multi-peak).

All scores are computed on the same date-level average PSTH store the peakiness
analysis reads, so unit keys line up across the two tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
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


TEMPORAL_SPECIFICITY_CONDITIONS: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)

#: Trace-shape metrics, in the order they are reported. ``higher_is_peaky`` says
#: which tail of the distribution corresponds to high temporal specificity;
#: ``None`` means the metric is an amplitude or descriptive quantity rather than
#: a shape axis.
METRIC_SPECS: tuple[dict[str, object], ...] = (
    {
        "name": "mass_width_frac_50",
        "label": "50% mass width (frac. of window)",
        "axis": "concentration",
        "higher_is_peaky": False,
    },
    {
        "name": "effective_width_ms",
        "label": "Effective width (ms)",
        "axis": "concentration",
        "higher_is_peaky": False,
    },
    {
        "name": "lifetime_sparseness",
        "label": "Lifetime sparseness",
        "axis": "concentration",
        "higher_is_peaky": True,
    },
    {
        "name": "peak_dominance",
        "label": "Peak dominance (p1 / Σp)",
        "axis": "single_peak",
        "higher_is_peaky": True,
    },
    {
        "name": "n_prominent_peaks",
        "label": "Prominent peaks (n)",
        "axis": "single_peak",
        "higher_is_peaky": False,
    },
    {
        "name": "fwhm_frac",
        "label": "FWHM (frac. of window)",
        "axis": "sustainedness",
        "higher_is_peaky": False,
    },
    {
        "name": "sustained_frac",
        "label": "Sustained fraction (>25% excess)",
        "axis": "sustainedness",
        "higher_is_peaky": False,
    },
    {
        "name": "roughness",
        "label": "Roughness (total variation / range)",
        "axis": "fluctuation",
        "higher_is_peaky": False,
    },
    {
        "name": "autocorr_width_ms",
        "label": "Autocorrelation half-width (ms)",
        "axis": "fluctuation",
        "higher_is_peaky": False,
    },
    {
        "name": "modulation_index",
        "label": "Modulation index",
        "axis": "amplitude",
        "higher_is_peaky": None,
    },
    {
        "name": "peak_z",
        "label": "Peak z (vs. window baseline)",
        "axis": "amplitude",
        "higher_is_peaky": None,
    },
    {
        "name": "temporal_specificity_index",
        "label": "Temporal specificity index",
        "axis": "composite",
        "higher_is_peaky": True,
    },
    {
        "name": "sustainedness_index",
        "label": "Sustainedness index",
        "axis": "composite",
        "higher_is_peaky": False,
    },
)

METRIC_NAMES: tuple[str, ...] = tuple(str(spec["name"]) for spec in METRIC_SPECS)
METRIC_LABELS: dict[str, str] = {str(s["name"]): str(s["label"]) for s in METRIC_SPECS}
METRIC_AXES: dict[str, str] = {str(s["name"]): str(s["axis"]) for s in METRIC_SPECS}

_DESCRIPTIVE_NAMES: tuple[str, ...] = (
    "mean_fr_hz",
    "baseline_fr_hz",
    "peak_fr_hz",
    "peak_latency_ms",
    "excess_mass_hz_s",
    "n_detected_peaks",
    "window_duration_ms",
    "bin_step_ms",
)

_UNIT_UUID_PREFIX = "unit_uuid__"


@dataclass
class FixationTemporalSpecificitySettings:
    """Configuration for temporal-specificity scoring of average fixation PSTHs."""

    cfg_path: str
    average_input_subdir: str = "ephys/psth/fixation_psth_averages"
    average_input_filename: str = "fixations_psth_10ms.pkl"
    output_subdir: str = "ephys/psth/fixation_temporal_specificity"
    condition_output_filename: str = "unit_condition_temporal_specificity.csv"
    unit_output_filename: str = "unit_temporal_specificity.csv"
    region_summary_filename: str = "region_temporal_specificity_summary.csv"
    trace_output_filename: str = "unit_condition_traces.pkl"
    output_pickle_filename: str = "results.pkl"
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    condition_order: tuple[str, ...] = field(
        default_factory=lambda: TEMPORAL_SPECIFICITY_CONDITIONS,
    )
    #: Analysis window in ms relative to fixation onset. The stored averages span
    #: [-1000, 1000] ms; scoring shape over the full span dilutes every width
    #: statistic with a second of pre-fixation baseline, so restrict by default
    #: to the interval the selectivity windows already cover.
    analysis_window_ms: tuple[float, float] = (-500.0, 500.0)
    #: Baseline is the given lower quantile of the in-window trace. A quantile is
    #: used rather than a pre-fixation epoch because the analysis window is
    #: centred on fixation onset and many units are already modulated before it.
    baseline_quantile: float = 0.10
    #: Minimum separation between detected peaks, matching ``fixation_peakiness``.
    peak_distance_ms: float = 30.0
    #: A peak counts as "prominent" when its prominence is at least this fraction
    #: of the largest prominence in the trace.
    prominent_peak_fraction: float = 0.25
    #: Excess-rate threshold (fraction of the excess peak) for ``sustained_frac``.
    sustained_threshold_fraction: float = 0.25
    #: Fraction of total excess mass used for ``mass_width_frac_50``.
    mass_fraction: float = 0.50
    #: Modulation gate. Shape statistics on a flat trace are noise, so units that
    #: clear neither criterion are flagged ``is_modulated = False``. ``peak_z`` is
    #: the primary gate because ``modulation_index`` saturates for high-baseline
    #: units -- a 34 Hz unit with a clean 3 Hz transient has a tiny modulation
    #: index but an unambiguous peak.
    min_peak_z: float = 3.0
    min_modulation_index: float = 0.05
    min_mean_fr_hz: float = 0.5
    min_trials_per_condition: int = 1
    epsilon: float = 1.0e-12
    bin_size_ms_fallback: float = 10.0
    region_order: Optional[Sequence[str]] = None
    #: Persist the (windowed) traces alongside the scores so notebooks can plot
    #: exemplars without re-scanning the per-date average store.
    store_traces: bool = True


# --------------------------------------------------------------------------- #
# Metric definitions                                                           #
# --------------------------------------------------------------------------- #


def _nan_metrics() -> dict[str, float]:
    out: dict[str, float] = {name: np.nan for name in METRIC_NAMES}
    out.update({name: np.nan for name in _DESCRIPTIVE_NAMES})
    return out


def lifetime_sparseness(values: np.ndarray) -> float:
    """Treves-Rolls sparseness of a non-negative trace.

    ``1`` when all the mass is in one bin, ``0`` when the trace is flat. Unlike a
    peak-prominence score this is amplitude-invariant, so it separates a crisp
    transient from a plateau even for a low-rate unit.

    Pass the *excess* (baseline-subtracted) trace. On raw firing rate the measure
    is dominated by the DC offset -- a 20 Hz unit with a 2 Hz transient and a
    2 Hz unit with a 2 Hz transient are the same shape but score an order of
    magnitude apart, which is the opposite of what a shape statistic should do.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan
    x = np.clip(x, 0.0, None)
    denom = float(np.mean(x * x))
    if denom <= 0.0:
        return np.nan
    numer = float(np.mean(x)) ** 2
    return float((1.0 - numer / denom) / (1.0 - 1.0 / n))


def mass_width_fraction(excess: np.ndarray, *, fraction: float) -> float:
    """Smallest fraction of bins holding ``fraction`` of the excess mass.

    Bins are ranked by excess rate and accumulated until the target share of the
    total excess is reached; the answer is the number of bins used divided by the
    window length. It is a width, not a shape fit, so it stays meaningful for
    multi-peak traces where a Gaussian-width estimate would not.
    """
    x = np.asarray(excess, dtype=float).reshape(-1)
    x = np.clip(x[np.isfinite(x)], 0.0, None)
    if x.size == 0:
        return np.nan
    total = float(x.sum())
    if total <= 0.0:
        return np.nan
    ordered = np.sort(x)[::-1]
    cumulative = np.cumsum(ordered)
    target = float(fraction) * total
    n_bins = int(np.searchsorted(cumulative, target, side="left")) + 1
    return float(min(n_bins, x.size)) / float(x.size)


def effective_width_ms(excess: np.ndarray, *, bin_step_ms: float) -> float:
    """Equivalent rectangular width of the excess response.

    ``sum(excess) / max(excess)`` -- the width of the rectangle with the same
    area and height as the response. Insensitive to how the mass is split across
    peaks, which makes it a useful cross-check on ``mass_width_frac_50``.
    """
    x = np.asarray(excess, dtype=float).reshape(-1)
    x = np.clip(x[np.isfinite(x)], 0.0, None)
    if x.size == 0:
        return np.nan
    peak = float(x.max())
    if peak <= 0.0:
        return np.nan
    return float(x.sum() / peak) * float(bin_step_ms)


def threshold_fraction(excess: np.ndarray, *, fraction: float) -> float:
    """Fraction of the window where excess rate is at or above ``fraction`` of its peak."""
    x = np.asarray(excess, dtype=float).reshape(-1)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.nan
    x = np.clip(x[finite], 0.0, None)
    peak = float(x.max())
    if peak <= 0.0:
        return np.nan
    return float(np.count_nonzero(x >= float(fraction) * peak)) / float(x.size)


def roughness(values: np.ndarray) -> float:
    """Total variation normalized by peak-to-trough range.

    A pure single up-down excursion gives ``1``; every additional reversal of
    comparable size adds roughly one. This isolates raggedness from the envelope,
    which the prominence-ratio score cannot do -- a trace with one tall peak and
    twenty small wiggles can still have a high ``p1 / (p1 + p2)``.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    span = float(x.max() - x.min())
    if span <= 0.0:
        return np.nan
    return float(np.abs(np.diff(x)).sum() / (2.0 * span))


def autocorr_half_width_ms(values: np.ndarray, *, bin_step_ms: float) -> float:
    """Lag at which the mean-subtracted autocorrelation first falls below 0.5.

    A slow, smooth modulation keeps correlation over long lags; a jittery trace
    decorrelates within a few bins. Reported in ms so it is comparable across
    binnings.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    x = x - float(x.mean())
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.nan
    full = np.correlate(x, x, mode="full")[x.size - 1 :] / denom
    below = np.flatnonzero(full < 0.5)
    if below.size == 0:
        return float(x.size) * float(bin_step_ms)
    return float(below[0]) * float(bin_step_ms)


def _peak_statistics(
    excess: np.ndarray,
    *,
    distance_bins: int,
    prominent_fraction: float,
) -> tuple[int, int, float]:
    """Return ``(n_detected_peaks, n_prominent_peaks, dominance)``.

    ``dominance`` is the largest prominence divided by the sum of all detected
    prominences -- the all-peak generalization of the existing ``p1 / (p1 + p2)``.
    """
    x = np.asarray(excess, dtype=float).reshape(-1)
    if x.size < 3:
        return 0, 0, np.nan
    # ``prominence=0.0`` is required, not cosmetic: without a prominence bound
    # ``find_peaks`` does not populate ``props["prominences"]`` at all.
    peaks, props = find_peaks(x, distance=max(1, int(distance_bins)), prominence=0.0)
    prominences = np.asarray(props.get("prominences", []), dtype=float).reshape(-1)
    if peaks.size == 0 or prominences.size == 0:
        return 0, 0, np.nan
    prominences = prominences[np.isfinite(prominences)]
    if prominences.size == 0:
        return int(peaks.size), 0, np.nan
    total = float(prominences.sum())
    largest = float(prominences.max())
    dominance = float(largest / total) if total > 0.0 else np.nan
    n_prominent = int(np.count_nonzero(prominences >= float(prominent_fraction) * largest))
    return int(peaks.size), n_prominent, dominance


def score_trace(
    trace_hz: np.ndarray,
    centers_s: np.ndarray,
    settings: FixationTemporalSpecificitySettings,
) -> dict[str, float]:
    """Score one average firing-rate trace on every temporal-specificity metric."""
    values = np.asarray(trace_hz, dtype=float).reshape(-1)
    centers = np.asarray(centers_s, dtype=float).reshape(-1)
    if values.size != centers.size:
        raise ValueError(
            "Trace length must match bin centers when scoring temporal specificity. "
            f"n_values={values.size}, n_centers={centers.size}"
        )

    lo_ms, hi_ms = (float(settings.analysis_window_ms[0]), float(settings.analysis_window_ms[1]))
    centers_ms = centers * 1000.0
    in_window = (centers_ms >= lo_ms) & (centers_ms <= hi_ms) & np.isfinite(values)
    if np.count_nonzero(in_window) < 4:
        return _nan_metrics()

    windowed = values[in_window]
    window_centers_ms = centers_ms[in_window]
    bin_step_ms = _resolve_bin_step_ms(centers, settings)
    window_duration_ms = float(window_centers_ms[-1] - window_centers_ms[0]) + bin_step_ms

    eps = float(settings.epsilon)
    mean_fr_hz = float(np.mean(windowed))
    baseline = float(np.quantile(windowed, float(settings.baseline_quantile)))
    peak_idx = int(np.argmax(windowed))
    peak_fr_hz = float(windowed[peak_idx])
    excess = np.clip(windowed - baseline, 0.0, None)
    excess_peak = float(excess.max())

    # Amplitude first: everything below is shape, and shape on a flat trace is
    # meaningless, so callers filter on these before reading the rest.
    modulation_index = float((peak_fr_hz - baseline) / (peak_fr_hz + baseline + eps))
    residual = windowed - np.median(windowed)
    # Robust spread from the lower half of the trace: a tall transient must not
    # inflate the noise estimate it is being compared against.
    lower = windowed[windowed <= np.median(windowed)]
    spread = float(np.std(lower, ddof=1)) if lower.size > 1 else float(np.std(residual, ddof=1))
    peak_z = float((peak_fr_hz - baseline) / (spread + eps)) if spread > 0.0 else np.nan

    n_detected, n_prominent, peak_dominance = _peak_statistics(
        excess,
        distance_bins=max(1, int(round(float(settings.peak_distance_ms) / max(bin_step_ms, eps)))),
        prominent_fraction=float(settings.prominent_peak_fraction),
    )

    out: dict[str, float] = {
        "mean_fr_hz": mean_fr_hz,
        "baseline_fr_hz": baseline,
        "peak_fr_hz": peak_fr_hz,
        "peak_latency_ms": float(window_centers_ms[peak_idx]),
        "excess_mass_hz_s": float(excess.sum() * bin_step_ms / 1000.0),
        "n_detected_peaks": float(n_detected),
        "window_duration_ms": window_duration_ms,
        "bin_step_ms": bin_step_ms,
        "mass_width_frac_50": mass_width_fraction(excess, fraction=float(settings.mass_fraction)),
        "effective_width_ms": effective_width_ms(excess, bin_step_ms=bin_step_ms),
        "lifetime_sparseness": lifetime_sparseness(excess),
        "peak_dominance": peak_dominance,
        "n_prominent_peaks": float(n_prominent),
        "fwhm_frac": threshold_fraction(excess, fraction=0.5),
        "sustained_frac": threshold_fraction(
            excess,
            fraction=float(settings.sustained_threshold_fraction),
        ),
        "roughness": roughness(windowed),
        "autocorr_width_ms": autocorr_half_width_ms(windowed, bin_step_ms=bin_step_ms),
        "modulation_index": modulation_index,
        "peak_z": peak_z,
    }

    if excess_peak <= 0.0:
        out["temporal_specificity_index"] = np.nan
        out["sustainedness_index"] = np.nan
        return out

    # Composites. Each factor is already on [0, 1] (roughness is squashed), so the
    # products stay interpretable and neither composite can be dominated by an
    # unbounded term.
    narrowness = 1.0 - _safe(out["mass_width_frac_50"], default=np.nan)
    dominance = _safe(out["peak_dominance"], default=np.nan)
    smoothness = 1.0 / (1.0 + max(_safe(out["roughness"], default=np.nan) - 1.0, 0.0))
    out["temporal_specificity_index"] = float(narrowness * dominance * smoothness)
    out["sustainedness_index"] = float(
        _safe(out["sustained_frac"], default=np.nan) * smoothness
    )
    return out


def _safe(value: float, *, default: float) -> float:
    out = float(value) if value is not None else default
    return out if np.isfinite(out) else default


# --------------------------------------------------------------------------- #
# Loading average traces                                                       #
# --------------------------------------------------------------------------- #


def _norm_token(value: object) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def _date_token(value: object) -> str:
    token = _as_optional_str(value)
    if token is None:
        return ""
    if len(token) == 7 and token.isdigit():
        return token.zfill(8)
    return token


def _resolve_bin_step_ms(
    centers_s: np.ndarray,
    settings: FixationTemporalSpecificitySettings,
) -> float:
    centers = np.asarray(centers_s, dtype=float).reshape(-1)
    if centers.size > 1:
        diffs = np.diff(centers)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size > 0:
            out = float(np.median(diffs) * 1000.0)
            if np.isfinite(out) and out > 0.0:
                return out
    return float(settings.bin_size_ms_fallback)


def _extract_average_partitions(obj) -> list[tuple[str, pd.DataFrame, dict]]:
    if isinstance(obj, dict):
        meta = obj.get("meta", {}) or {}
        meta_dict = meta if isinstance(meta, dict) else {}
        out: list[tuple[str, pd.DataFrame, dict]] = []
        for partition, df_key, meta_key in (
            ("split", "averages_split_by_interactive_state", "split_meta"),
            ("unsplit", "averages_unsplit_by_interactive_state", "unsplit_meta"),
        ):
            df = obj.get(df_key)
            if not isinstance(df, pd.DataFrame):
                continue
            merged_meta = dict(meta_dict)
            partition_meta = meta_dict.get(meta_key, {})
            if isinstance(partition_meta, dict):
                merged_meta.update(partition_meta)
            merged_meta["selected_partition"] = partition
            out.append((partition, df, merged_meta))
        if out:
            return out

        df = obj.get("averages")
        if isinstance(df, pd.DataFrame):
            partition = "split" if bool(meta_dict.get("split_by_interactive_state")) else "unsplit"
            merged_meta = dict(meta_dict)
            merged_meta["selected_partition"] = partition
            return [(partition, df, merged_meta)]

    if isinstance(obj, pd.DataFrame):
        return [("unsplit", obj, {})]
    return []


def _average_row_condition(
    row: pd.Series,
    *,
    partition: str,
    settings: FixationTemporalSpecificitySettings,
) -> Optional[str]:
    category = _norm_token(row.get("fixation_category"))
    face = _norm_token(settings.face_label)
    obj = _norm_token(settings.object_label)

    if partition == "unsplit":
        return "object" if category == obj else None
    if partition != "split" or category != face:
        return None

    is_interactive = row.get("is_interactive")
    if is_interactive is not None and not pd.isna(is_interactive):
        interactive = bool(_as_bool(is_interactive, settings.interactive_label))
    else:
        interactive = _norm_token(row.get("interactive_state")) == _norm_token(
            settings.interactive_label
        )
    return "face_interactive" if interactive else "face_non_interactive"


def _resolve_bin_duration_s(
    meta: dict,
    centers_s: np.ndarray,
    settings: FixationTemporalSpecificitySettings,
) -> float:
    for key in ("target_bin_size_s", "output_bin_size_s", "bin_size_s"):
        value = meta.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except Exception:
            continue
        if np.isfinite(out) and out > 0.0:
            return out
    if centers_s.size > 1:
        out = float(np.median(np.diff(centers_s)))
        if np.isfinite(out) and out > 0.0:
            return out
    return float(settings.bin_size_ms_fallback) / 1000.0


def _average_values_are_rate(meta: dict) -> bool:
    value_kind = str(meta.get("psth_value_kind", "")).strip().lower()
    return bool(
        value_kind == "firing_rate_hz"
        or (
            meta.get("convert_to_firing_rate_before_average") is True
            and value_kind != "counts"
        )
    )


def _coerce_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def load_condition_traces(
    settings: FixationTemporalSpecificitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load one trial-weighted average trace per (unit, fixation condition)."""
    cfg = load_config(settings.cfg_path)
    rows = scan_analysis_date_paths(
        cfg,
        settings.average_input_subdir,
        filename=ensure_filename(settings.average_input_filename, ".pkl"),
        dates=dates,
    )
    records: list[dict] = []
    for row in rows:
        path = Path(row["path"])
        obj = load_pickle_path(path)
        for partition, avg_df, meta in _extract_average_partitions(obj):
            if not isinstance(avg_df, pd.DataFrame) or avg_df.empty:
                continue
            if "psth_mean" not in avg_df.columns:
                continue
            centers = resolve_bin_centers_from_meta(meta)
            if centers is None:
                raise ValueError(f"Unable to resolve average PSTH bin centers: {path}")
            centers_s = np.asarray(centers, dtype=float).reshape(-1)
            bin_duration_s = _resolve_bin_duration_s(meta, centers_s, settings)
            is_rate = _average_values_are_rate(meta)
            for _, avg_row in avg_df.iterrows():
                condition = _average_row_condition(
                    avg_row, partition=partition, settings=settings
                )
                if condition is None:
                    continue
                n_trials = _coerce_float(avg_row.get("n_trials"))
                if np.isfinite(n_trials) and n_trials < float(settings.min_trials_per_condition):
                    continue
                date = _date_token(avg_row.get("date")) or _date_token(row.get("date"))
                unit_uuid = _as_optional_str(avg_row.get("unit_uuid"))
                if not date or unit_uuid is None:
                    continue
                trace = np.asarray(avg_row.get("psth_mean"), dtype=float).reshape(-1)
                if trace.size != centers_s.size:
                    raise ValueError(
                        "Average PSTH row length does not match bin centers: "
                        f"path={path}, n_values={trace.size}, n_centers={centers_s.size}"
                    )
                if not is_rate:
                    trace = trace / bin_duration_s
                sem = avg_row.get("psth_sem")
                sem_arr = np.asarray(sem, dtype=float).reshape(-1) if sem is not None else None
                if sem_arr is not None and sem_arr.size == centers_s.size and not is_rate:
                    sem_arr = sem_arr / bin_duration_s
                records.append(
                    {
                        "unit_key": f"{date}|{unit_uuid}",
                        "date": date,
                        "unit_uuid": unit_uuid,
                        "region": _as_optional_str(avg_row.get("region")) or "unknown",
                        "spike_channel": _as_optional_str(avg_row.get("spike_channel")),
                        "recorded_agent": _as_optional_str(avg_row.get("recorded_agent")),
                        "recorded_monkey": _as_optional_str(avg_row.get("recorded_monkey")),
                        "area": _as_optional_str(avg_row.get("area")),
                        "condition": condition,
                        "n_trials": n_trials,
                        "bin_centers_s_rel": centers_s,
                        "trace_hz": trace,
                        "sem_hz": sem_arr,
                    }
                )
    trace_df = pd.DataFrame(records)
    return _aggregate_duplicate_traces(trace_df)


def _first_non_null(series: pd.Series) -> Optional[str]:
    for value in series:
        token = _as_optional_str(value)
        if token is not None:
            return token
    return None


def _aggregate_duplicate_traces(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated (unit, condition) rows with a trial-count weighted mean."""
    if trace_df.empty:
        return trace_df

    rows: list[dict] = []
    for _, group in trace_df.groupby(["unit_key", "condition"], dropna=False, sort=False):
        first = group.iloc[0]
        centers_ref = np.asarray(first["bin_centers_s_rel"], dtype=float).reshape(-1)
        traces: list[np.ndarray] = []
        weights: list[float] = []
        for _, row in group.iterrows():
            centers = np.asarray(row["bin_centers_s_rel"], dtype=float).reshape(-1)
            if centers.shape != centers_ref.shape or not np.allclose(centers, centers_ref):
                raise ValueError(
                    "Mismatched bin centers within the same unit-condition traces: "
                    f"unit_key={first['unit_key']}, condition={first['condition']}"
                )
            traces.append(np.asarray(row["trace_hz"], dtype=float).reshape(-1))
            n_trials = _coerce_float(row.get("n_trials"))
            weights.append(n_trials if np.isfinite(n_trials) and n_trials > 0.0 else 1.0)
        stacked = np.vstack(traces)
        trace_hz = np.average(stacked, axis=0, weights=np.asarray(weights, dtype=float))
        n_trials_values = pd.to_numeric(group["n_trials"], errors="coerce")
        rows.append(
            {
                "unit_key": str(first["unit_key"]),
                "date": str(first["date"]),
                "unit_uuid": str(first["unit_uuid"]),
                "region": _first_non_null(group["region"]) or "unknown",
                "spike_channel": _first_non_null(group["spike_channel"]),
                "recorded_agent": _first_non_null(group["recorded_agent"]),
                "recorded_monkey": _first_non_null(group["recorded_monkey"]),
                "area": _first_non_null(group["area"]),
                "condition": str(first["condition"]),
                "n_trials": float(n_trials_values.dropna().sum())
                if n_trials_values.notna().any()
                else np.nan,
                "n_source_rows": int(len(group)),
                "bin_centers_s_rel": centers_ref,
                "trace_hz": trace_hz,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #

_META_COLUMNS = (
    "unit_key",
    "date",
    "unit_uuid",
    "region",
    "spike_channel",
    "recorded_agent",
    "recorded_monkey",
    "area",
)


def build_condition_table(
    trace_df: pd.DataFrame,
    settings: FixationTemporalSpecificitySettings,
) -> pd.DataFrame:
    """Score every (unit, condition) trace."""
    columns = (
        list(_META_COLUMNS)
        + ["condition", "n_trials", "n_source_rows"]
        + list(_DESCRIPTIVE_NAMES)
        + list(METRIC_NAMES)
        + ["is_modulated"]
    )
    if trace_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for _, row in trace_df.iterrows():
        scores = score_trace(
            np.asarray(row["trace_hz"], dtype=float),
            np.asarray(row["bin_centers_s_rel"], dtype=float),
            settings,
        )
        out = {column: row.get(column) for column in _META_COLUMNS}
        out["condition"] = str(row["condition"])
        out["n_trials"] = _coerce_float(row.get("n_trials"))
        out["n_source_rows"] = int(row.get("n_source_rows", 1))
        out.update(scores)
        peak_z = _coerce_float(scores.get("peak_z"))
        modulation_index = _coerce_float(scores.get("modulation_index"))
        mean_fr = _coerce_float(scores.get("mean_fr_hz"))
        out["is_modulated"] = bool(
            np.isfinite(mean_fr)
            and mean_fr >= float(settings.min_mean_fr_hz)
            and (
                (np.isfinite(peak_z) and peak_z >= float(settings.min_peak_z))
                or (
                    np.isfinite(modulation_index)
                    and modulation_index >= float(settings.min_modulation_index)
                )
            )
        )
        rows.append(out)
    return pd.DataFrame(rows).loc[:, columns]


def build_unit_table(
    condition_df: pd.DataFrame,
    settings: FixationTemporalSpecificitySettings,
) -> pd.DataFrame:
    """Collapse condition scores to one row per unit.

    Each metric is summarized two ways: ``<metric>`` is the trial-count weighted
    mean across the unit's conditions (the unit's typical temporal profile), and
    ``<metric>__<condition>`` keeps the per-condition value so condition-specific
    questions stay answerable from the same table. Which condition carries the
    strongest modulation is recorded as ``best_condition``.
    """
    condition_columns = [
        f"{metric}__{condition}"
        for metric in METRIC_NAMES
        for condition in settings.condition_order
    ]
    columns = (
        list(_META_COLUMNS)
        + [
            "n_conditions_observed",
            "all_conditions_present",
            "n_trials_total",
            "mean_fr_hz",
            "peak_fr_hz",
            "modulation_index_max",
            "best_condition",
            "any_condition_modulated",
        ]
        + list(METRIC_NAMES)
        + condition_columns
    )
    if condition_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for _, group in condition_df.groupby("unit_key", dropna=False, sort=False):
        first = group.iloc[0]
        out = {column: first.get(column) for column in _META_COLUMNS}
        observed = [str(value) for value in group["condition"].astype(str)]
        out["n_conditions_observed"] = int(len(set(observed)))
        out["all_conditions_present"] = bool(
            set(str(c) for c in settings.condition_order).issubset(set(observed))
        )

        weights = pd.to_numeric(group["n_trials"], errors="coerce").to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        out["n_trials_total"] = float(
            pd.to_numeric(group["n_trials"], errors="coerce").dropna().sum()
        )
        out["mean_fr_hz"] = _weighted_nanmean(
            pd.to_numeric(group["mean_fr_hz"], errors="coerce").to_numpy(dtype=float), weights
        )
        out["peak_fr_hz"] = float(
            pd.to_numeric(group["peak_fr_hz"], errors="coerce").max()
        )
        modulation = pd.to_numeric(group["modulation_index"], errors="coerce")
        out["modulation_index_max"] = (
            float(modulation.max()) if modulation.notna().any() else np.nan
        )
        out["any_condition_modulated"] = bool(group["is_modulated"].astype(bool).any())
        out["best_condition"] = (
            str(group.loc[modulation.idxmax(), "condition"])
            if modulation.notna().any()
            else None
        )

        for metric in METRIC_NAMES:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            out[metric] = _weighted_nanmean(values, weights)
            for condition in settings.condition_order:
                match = group.loc[group["condition"].astype(str) == str(condition)]
                out[f"{metric}__{condition}"] = (
                    _coerce_float(match.iloc[0][metric]) if not match.empty else np.nan
                )
        rows.append(out)
    return pd.DataFrame(rows).loc[:, columns]


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def build_region_summary(
    unit_df: pd.DataFrame,
    settings: FixationTemporalSpecificitySettings,
) -> pd.DataFrame:
    """Distribution summary of every metric, per region."""
    columns = [
        "region",
        "metric",
        "axis",
        "n_units",
        "mean",
        "std",
        "median",
        "q10",
        "q25",
        "q75",
        "q90",
        "min",
        "max",
    ]
    if unit_df.empty:
        return pd.DataFrame(columns=columns)

    regions = _ordered_regions(unit_df, settings)
    rows: list[dict] = []
    for region in regions:
        region_df = unit_df.loc[unit_df["region"].astype(str) == region]
        for metric in METRIC_NAMES:
            values = pd.to_numeric(region_df[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                rows.append(
                    {
                        "region": region,
                        "metric": metric,
                        "axis": METRIC_AXES[metric],
                        "n_units": 0,
                        **{key: np.nan for key in columns[4:]},
                    }
                )
                continue
            rows.append(
                {
                    "region": region,
                    "metric": metric,
                    "axis": METRIC_AXES[metric],
                    "n_units": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q10": float(np.quantile(values, 0.10)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q90": float(np.quantile(values, 0.90)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows).loc[:, columns]


def _ordered_regions(
    unit_df: pd.DataFrame,
    settings: FixationTemporalSpecificitySettings,
) -> list[str]:
    observed = [str(v) for v in unit_df["region"].astype(str).dropna().unique()]
    if settings.region_order is None:
        return observed
    ordered = [str(r) for r in settings.region_order if str(r) in observed]
    return ordered + [r for r in observed if r not in ordered]


def unit_uuid_lookup_variants(value: object) -> set[str]:
    """Both the bare and ``unit_uuid__``-prefixed spellings of a unit id."""
    token = _as_optional_str(value)
    if token is None:
        return set()
    out = {token}
    if token.startswith(_UNIT_UUID_PREFIX):
        suffix = token[len(_UNIT_UUID_PREFIX) :].strip()
        if suffix:
            out.add(suffix)
    else:
        out.add(f"{_UNIT_UUID_PREFIX}{token}")
    return out


def run_fixation_temporal_specificity_analysis(
    settings: FixationTemporalSpecificitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Score every unit's average fixation PSTHs and persist the tables."""
    trace_df = load_condition_traces(settings, dates=dates)
    if regions is not None and not trace_df.empty:
        allowed = {str(region) for region in regions}
        trace_df = trace_df.loc[trace_df["region"].astype(str).isin(allowed)].copy()

    condition_df = build_condition_table(trace_df, settings)
    unit_df = build_unit_table(condition_df, settings)
    region_summary_df = build_region_summary(unit_df, settings)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    condition_df.to_csv(
        out_root / ensure_filename(settings.condition_output_filename, ".csv"), index=False
    )
    unit_df.to_csv(out_root / ensure_filename(settings.unit_output_filename, ".csv"), index=False)
    region_summary_df.to_csv(
        out_root / ensure_filename(settings.region_summary_filename, ".csv"), index=False
    )
    if settings.store_traces:
        save_pickle_path(
            trace_df, out_root / ensure_filename(settings.trace_output_filename, ".pkl")
        )

    result: dict[str, object] = {
        "meta": {
            "average_input_subdir": str(settings.average_input_subdir),
            "average_input_filename": ensure_filename(settings.average_input_filename, ".pkl"),
            "analysis_window_ms": [float(v) for v in settings.analysis_window_ms],
            "baseline_quantile": float(settings.baseline_quantile),
            "peak_distance_ms": float(settings.peak_distance_ms),
            "prominent_peak_fraction": float(settings.prominent_peak_fraction),
            "sustained_threshold_fraction": float(settings.sustained_threshold_fraction),
            "mass_fraction": float(settings.mass_fraction),
            "min_peak_z": float(settings.min_peak_z),
            "min_modulation_index": float(settings.min_modulation_index),
            "min_mean_fr_hz": float(settings.min_mean_fr_hz),
            "condition_order": list(settings.condition_order),
            "metric_names": list(METRIC_NAMES),
            "metric_labels": dict(METRIC_LABELS),
            "metric_axes": dict(METRIC_AXES),
            "n_units": int(len(unit_df)),
            "n_condition_rows": int(len(condition_df)),
        },
        "condition_specificity": condition_df,
        "unit_specificity": unit_df,
        "region_summary": region_summary_df,
    }
    if settings.store_traces:
        result["condition_traces"] = trace_df

    save_pickle_path(result, out_root / ensure_filename(settings.output_pickle_filename, ".pkl"))
    return result
