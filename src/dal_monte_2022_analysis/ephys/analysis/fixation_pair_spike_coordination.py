"""Spike coordination between simultaneously recorded neural pairs, per fixation condition.

The question this module answers is whether two neurons recorded at the same
time fire in a coordinated way *beyond what their individual fixation-locked
rate profiles already predict*, and whether that coordination differs between
interactive-face, non-interactive-face and object fixations, within a region
and across regions.

Design notes that matter for interpreting the output
----------------------------------------------------

*Per-fixation, never per-average.*  Coordination is a trial-by-trial quantity.
Cross-correlating condition-averaged PSTHs measures shared rate structure, not
coordination, so every cross-correlation here is computed on one fixation's two
1 ms spike trains and only then averaged.

*Linear cross-correlation, and everything inside one window.*  The analysis
window is a fixed ``[-500, 500] ms`` around fixation onset, and every number
here -- observed and null alike -- is computed from spikes inside it.  The
correlation is linear (zero-padded transform), so at lag ``L`` only the
``N - |L|`` genuinely overlapping bins contribute.  A wrapping (unpadded)
transform would instead pair a spike at -500 ms with one at +500 ms and report
it as a coincidence at lag ``L``; those events are a second apart and that is
not a physical measurement, however convenient the arithmetic.

Linear correlation does taper: the number of contributing bins falls as ``|L|``
grows, so a raw correlation shrinks with lag for a purely mechanical reason.
That needs no correction here, because **both nulls carry the identical
taper** and it cancels in the excess and in every z-score.  :func:`overlap_bins`
is stored with the output for figures that want raw magnitudes on a comparable
scale across lags.

Widening the window is not an option: 95% of ``+/-5 s`` surrounds contain at
least one other analysed fixation (median 5), so data outside the window is not
neutral baseline and a shifted-window null would be a comparison against a
different behavioural condition rather than a null.

*Two nulls, because they answer different questions.*

``trial_shuffle``
    Pairs unit A's train on fixation *i* with unit B's train on fixation
    *j != i*, within the same condition.  This keeps both units' fixation-locked
    rate profiles and destroys only the trial-by-trial covariation, so an excess
    over this null means the two cells co-fluctuate from fixation to fixation.
    It does not distinguish fast synchrony from slow shared drive.

``circular_shift``
    Rotates unit B's train within the same fixation by a random offset, then
    correlates linearly.  This keeps each fixation's own spike count and slow
    envelope but destroys the fine temporal alignment, so an excess over this
    null means coordination at a timescale finer than the shift, over and above
    slow co-modulation.  The wrap introduced by the rotation is acceptable
    *here* precisely because destroying temporal structure is what a null is
    for -- which is the same reason it is not acceptable in the observed
    statistic.

A sharp zero-lag peak that survives *both* nulls is not evidence of a synaptic
connection -- for two randomly sampled cells that prior is near zero.  It is
much more likely a common input shared within a fixation: movement, arousal, or
a reference/ground artifact on the recording system.  Because that artifact
tends to be day- and array-specific rather than pair-specific,
:func:`build_zero_lag_diagnostics` reports the zero-lag excess per date and per
region pair so a contaminated day is visible instead of silently averaged in.

*Count-matched nulls.*  There are ``F * (F - 1)`` possible cross-fixation
pairings but only ``F`` real ones, so a null estimated from all of them would
have a far smaller standard error than the observed statistic and would inflate
every z-score.  Each null draw here is therefore a *derangement* of the fixation
index: exactly ``F`` pairings, each fixation used once on each side.  The null
distribution is then the distribution of the same statistic computed on the same
number of terms, which is what a z-score against it requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path


#: Analysis conditions, in reporting order.  Identical to the condition set the
#: single-unit, population-PCA and mRNN analyses use, so a pair result can be
#: joined against a unit result without a translation table.
CONDITION_ORDER: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)

#: Region reporting order, matching ``ephys.plotting.thesis_common``.
REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")

#: Columns that identify one unit across every analysis in the repository.
UNIT_KEY_COLUMNS: tuple[str, ...] = (
    "date",
    "unit_uuid",
    "region",
    "spike_channel",
    "recorded_agent",
)

DEFAULT_OUTPUT_SUBDIR = "ephys/psth/fixation_pair_spike_coordination"
DEFAULT_SELECTIVITY_SUBDIR = "ephys/psth/fixation_psth_selectivity"
DEFAULT_SELECTIVITY_UNIT_FILENAME = "unit_selectivity__three_condition_core.csv"

#: Tolerance for the exact linear-algebra identities the null construction
#: relies on.  Everything checked is a float64 FFT round trip, so this only has
#: to absorb round-off.
IDENTITY_TOLERANCE = 1e-9


@dataclass
class PairSpikeCoordinationSettings:
    """Configuration for one pair spike-coordination build."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations_spike_train_1ms.pkl"
    #: Unsmoothed counts by design.  Smoothing before cross-correlation blurs
    #: exactly the fine timing the analysis is trying to measure; smooth the
    #: saved traces afterwards if a figure needs it.
    signal_input_column: str = "spike_train_counts"
    signal_window_ms: tuple[float, float] = (-500.0, 500.0)
    #: Lags retained in the saved traces.  The circular correlation is defined
    #: over the full period; this only controls what is written to disk.
    store_max_lag_ms: float = 250.0
    conditions: tuple[str, ...] = CONDITION_ORDER
    #: Minimum fixations in a condition for that condition to be computed.  A
    #: derangement needs at least two fixations; below ~8 the null SD is too
    #: unstable for a usable z-score.
    min_fixations_per_condition: int = 8
    #: Minimum spikes a unit must fire across a condition's fixations to be
    #: included.  An all-but-silent unit produces an all-zero null SD.
    min_spikes_per_unit_condition: int = 10
    n_trial_shuffle_draws: int = 50
    n_circular_shift_draws: int = 50
    #: Smallest circular shift used by the shift null, in ms.  Shifts below the
    #: coordination timescale of interest would leave the effect intact and
    #: make the null too conservative.
    min_circular_shift_ms: float = 50.0
    include_within_region: bool = True
    include_cross_region: bool = True
    #: Also recompute every pair on a common fixation count across conditions.
    #: Interactive-face fixations outnumber non-interactive-face ones about five
    #: to one, so this is the direct control on trial-count-driven differences.
    trial_match_conditions: bool = True
    random_seed: int = 42
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR
    output_filename: str = "pair_coordination.pkl"
    selectivity_subdir: str = DEFAULT_SELECTIVITY_SUBDIR
    selectivity_unit_filename: str = DEFAULT_SELECTIVITY_UNIT_FILENAME
    #: FDR-corrected selectivity, per the project decision to report corrected
    #: calls.  Stored as a column, never used as a filter here, so the notebook
    #: can slice either way without a recompute.
    selectivity_column: str = "is_selective_unit_corrected"
    use_parallel: bool = True
    max_procs: Optional[int] = None
    dates: Optional[Sequence[str]] = None
    sessions: Optional[Sequence[str]] = None
    test_single: bool = False


def build_pair_spike_coordination_settings_from_config(
    *,
    dataset_cfg_path: str,
    coordination_cfg_path: Optional[str] = None,
) -> PairSpikeCoordinationSettings:
    """Build settings from the dataset config plus an optional task config."""
    cfg: dict = {}
    if coordination_cfg_path is not None:
        cfg = load_config(coordination_cfg_path)

    def window(key: str, default: tuple[float, float]) -> tuple[float, float]:
        raw = cfg.get(key)
        if raw is None:
            return default
        values = [float(value) for value in raw]
        if len(values) != 2 or not (values[1] > values[0]):
            raise ValueError(f"{key} must be [start_ms, stop_ms] with stop > start.")
        return (values[0], values[1])

    return PairSpikeCoordinationSettings(
        cfg_path=dataset_cfg_path,
        trial_input_modality=cfg.get("trial_input_modality", "psth"),
        trial_input_filename=cfg.get("trial_input_filename", "fixations_spike_train_1ms.pkl"),
        signal_input_column=cfg.get("signal_input_column", "spike_train_counts"),
        signal_window_ms=window("signal_window_ms", (-500.0, 500.0)),
        store_max_lag_ms=float(cfg.get("store_max_lag_ms", 250.0)),
        conditions=tuple(cfg.get("conditions", CONDITION_ORDER)),
        min_fixations_per_condition=int(cfg.get("min_fixations_per_condition", 8)),
        min_spikes_per_unit_condition=int(cfg.get("min_spikes_per_unit_condition", 10)),
        n_trial_shuffle_draws=int(cfg.get("n_trial_shuffle_draws", 50)),
        n_circular_shift_draws=int(cfg.get("n_circular_shift_draws", 50)),
        min_circular_shift_ms=float(cfg.get("min_circular_shift_ms", 50.0)),
        include_within_region=bool(cfg.get("include_within_region", True)),
        include_cross_region=bool(cfg.get("include_cross_region", True)),
        trial_match_conditions=bool(cfg.get("trial_match_conditions", True)),
        random_seed=int(cfg.get("random_seed", 42)),
        output_subdir=cfg.get("output_subdir", DEFAULT_OUTPUT_SUBDIR),
        output_filename=cfg.get("output_filename", "pair_coordination.pkl"),
        selectivity_subdir=cfg.get("selectivity_subdir", DEFAULT_SELECTIVITY_SUBDIR),
        selectivity_unit_filename=cfg.get(
            "selectivity_unit_filename", DEFAULT_SELECTIVITY_UNIT_FILENAME
        ),
        selectivity_column=cfg.get("selectivity_column", "is_selective_unit_corrected"),
        use_parallel=bool(cfg.get("use_parallel", True)),
        max_procs=cfg.get("max_procs"),
    )


# ---------------------------------------------------------------------------
# Condition assignment and session loading
# ---------------------------------------------------------------------------


def assign_condition(fixation_category: object, interactive_state: object) -> Optional[str]:
    """Map one fixation's category and interactive state onto an analysis condition.

    Object fixations are pooled across interactive state, matching the
    ``unsplit`` partition the PSTH averages and the mRNN bridge use for objects.
    """
    category = str(fixation_category).strip().lower()
    if category == "object":
        return "object"
    if category != "face":
        return None
    state = str(interactive_state).strip().lower()
    if state == "interactive":
        return "face_interactive"
    if state == "non_interactive":
        return "face_non_interactive"
    return None


@dataclass
class SessionSpikeTrains:
    """One session's fixation-aligned spike trains, grouped by condition."""

    date: str
    session: str
    #: Per-unit identity tuples in row order of every ``trains`` array.
    unit_keys: tuple[tuple[str, ...], ...]
    unit_table: pd.DataFrame
    #: ``condition -> (n_units, n_fixations, n_bins)`` unsmoothed 1 ms counts.
    trains: dict[str, np.ndarray]
    bin_size_ms: float
    window_ms: tuple[float, float]

    @property
    def n_units(self) -> int:
        return len(self.unit_keys)


def _window_mask(bin_centers_s: np.ndarray, window_ms: tuple[float, float]) -> np.ndarray:
    centers = np.asarray(bin_centers_s, dtype=float).reshape(-1)
    tol = 1e-12
    start_s = float(window_ms[0]) / 1000.0
    stop_s = float(window_ms[1]) / 1000.0
    return (centers >= start_s - tol) & (centers <= stop_s + tol)


def _resolve_spike_train_bin_centers(meta: Mapping) -> np.ndarray:
    for key in (
        "spike_train_bin_centers_s_rel",
        "bin_centers_s_rel",
    ):
        value = meta.get(key)
        if value is not None:
            return np.asarray(value, dtype=float).reshape(-1)
    raise ValueError("Unable to resolve spike-train bin centers from trial file metadata.")


def load_session_spike_trains(
    path: str | Path,
    settings: PairSpikeCoordinationSettings,
) -> Optional[SessionSpikeTrains]:
    """Load one session's 1 ms trains, windowed and grouped by condition.

    Returns ``None`` when the session has no condition with enough fixations to
    support a count-matched null.
    """
    obj = load_pickle_path(path)
    meta = obj["meta"]
    trials = obj["trials"]
    if not isinstance(trials, pd.DataFrame) or trials.empty:
        return None
    column = settings.signal_input_column
    if column not in trials.columns:
        raise ValueError(
            f"Trial file {path} has no column {column!r}; available: {list(trials.columns)}"
        )

    centers = _resolve_spike_train_bin_centers(meta)
    mask = _window_mask(centers, settings.signal_window_ms)
    if not mask.any():
        raise ValueError(f"signal_window_ms selects no bins for {path}.")
    bin_size_ms = float(np.median(np.diff(centers)) * 1000.0)

    frame = trials.copy()
    frame["condition"] = [
        assign_condition(category, state)
        for category, state in zip(frame["fixation_category"], frame["interactive_state"])
    ]
    frame = frame.loc[frame["condition"].isin(settings.conditions)]
    if frame.empty:
        return None

    for column_name in UNIT_KEY_COLUMNS:
        if column_name in frame.columns:
            frame[column_name] = frame[column_name].astype(str)
    frame["fixation_key"] = (
        frame["fixation_agent"].astype(str) + "|" + frame["fixation_start_idx"].astype(str)
    )

    unit_table = (
        frame.loc[:, list(UNIT_KEY_COLUMNS)]
        .drop_duplicates()
        .sort_values(["region", "unit_uuid"])
        .reset_index(drop=True)
    )
    unit_keys = tuple(tuple(row) for row in unit_table.loc[:, list(UNIT_KEY_COLUMNS)].values)
    unit_index = {key: index for index, key in enumerate(unit_keys)}

    trains: dict[str, np.ndarray] = {}
    for condition in settings.conditions:
        subset = frame.loc[frame["condition"] == condition]
        if subset.empty:
            continue
        fixation_keys = sorted(subset["fixation_key"].unique())
        if len(fixation_keys) < max(2, settings.min_fixations_per_condition):
            continue
        fixation_index = {key: index for index, key in enumerate(fixation_keys)}
        array = np.zeros(
            (len(unit_keys), len(fixation_keys), int(mask.sum())),
            dtype=np.float32,
        )
        for row in subset.itertuples(index=False):
            key = tuple(str(getattr(row, name)) for name in UNIT_KEY_COLUMNS)
            unit_slot = unit_index.get(key)
            if unit_slot is None:
                continue
            trace = np.asarray(getattr(row, column), dtype=np.float32)
            array[unit_slot, fixation_index[row.fixation_key], :] = trace[mask]
        trains[condition] = array

    if not trains:
        return None

    return SessionSpikeTrains(
        date=str(meta.get("date", "")),
        session=str(meta.get("session", "")),
        unit_keys=unit_keys,
        unit_table=unit_table,
        trains=trains,
        bin_size_ms=bin_size_ms,
        window_ms=tuple(float(value) for value in settings.signal_window_ms),
    )


# ---------------------------------------------------------------------------
# Circular cross-correlation and the two nulls
# ---------------------------------------------------------------------------
#
# Everything below works on the real FFT of each fixation's spike train.  Three
# identities make the whole computation cheap enough to run over every pair:
#
# 1. The mean cross-correlation over fixations is the inverse transform of the
#    mean cross-spectrum, because the inverse transform is linear.  So one
#    inverse FFT per pair, not one per fixation.
#
# 2. With the fixation axis contracted, the cross-spectrum for *every* pair at
#    one frequency is a single matrix product ``S @ S.conj().T``.  One batched
#    matmul therefore produces all pairs at once, and a null draw is the same
#    matmul with one operand permuted or phase-rotated.
#
# 3. Circularly shifting a train by ``s`` multiplies its spectrum by a phase
#    ramp, so the shift null needs no extra transforms either.
#
# :func:`verify_null_identities` checks all three against brute force.


def _next_fast_length(target: int) -> int:
    """Smallest power of two at least ``target``, for the zero-padded transform."""
    return 1 << int(np.ceil(np.log2(max(int(target), 1))))


def _lag_axis(n_bins: int, bin_size_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Lags of a linear correlation of two length-``n_bins`` signals.

    Spans ``-(n_bins - 1) .. (n_bins - 1)``.  A positive lag means the first
    unit fired that many bins *after* the second.
    """
    lags_samples = np.arange(-(int(n_bins) - 1), int(n_bins), dtype=np.int64)
    return lags_samples, lags_samples.astype(float) * float(bin_size_ms)


def overlap_bins(n_bins: int) -> np.ndarray:
    """Number of bin pairs contributing at each lag of a linear correlation.

    Falls from ``n_bins`` at lag 0 to 1 at the extremes.  This taper is why a
    raw linear correlation shrinks with ``|lag|`` for a purely mechanical
    reason -- but both nulls carry the identical taper, so it cancels in the
    excess and in every z-score built from it.  The vector is stored with the
    output so a figure can divide it out if it wants raw magnitudes on a
    comparable scale across lags.
    """
    lags = np.arange(-(int(n_bins) - 1), int(n_bins))
    return (int(n_bins) - np.abs(lags)).astype(float)


def _random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random cyclic permutation of ``range(n)`` (Sattolo's algorithm).

    A cyclic permutation has no fixed point, so no fixation is ever paired with
    itself, and it uses each fixation exactly once on each side.  That keeps the
    null statistic built from exactly ``n`` terms, matching the observed one.
    """
    if int(n) < 2:
        raise ValueError("A derangement needs at least two elements.")
    order = np.arange(int(n))
    for index in range(int(n) - 1, 0, -1):
        swap = int(rng.integers(0, index))
        order[index], order[swap] = order[swap], order[index]
    return order


def _cross_spectra_all_pairs(
    spectra_a: np.ndarray,
    spectra_b: np.ndarray,
) -> np.ndarray:
    """Mean cross-spectrum for every unit pair.

    ``spectra_*`` are ``(n_units, n_fixations, n_freqs)``.  Returns
    ``(n_units, n_units, n_freqs)`` where entry ``[a, b]`` is the fixation-mean
    of ``spectra_a[a, f] * conj(spectra_b[b, f])``.
    """
    n_fixations = spectra_a.shape[1]
    # (n_freqs, n_units, n_fixations) @ (n_freqs, n_fixations, n_units)
    left = np.moveaxis(spectra_a, 2, 0)
    right = np.moveaxis(spectra_b.conj(), 2, 0).transpose(0, 2, 1)
    product = np.matmul(left, right) / float(n_fixations)
    return np.moveaxis(product, 0, 2)


def _to_traces(
    cross_spectra: np.ndarray,
    n_bins: int,
    n_fft: int,
    *,
    quantum: Optional[float] = None,
) -> np.ndarray:
    """Inverse-transform a padded cross-spectrum into linear-correlation traces.

    The transform is taken over ``n_fft >= 2 * n_bins - 1`` points, so the
    result is the **linear** correlation: at lag ``L`` only the ``n_bins - |L|``
    genuinely overlapping bins contribute, and no sample is ever paired with one
    a full window away.  An unpadded transform would instead wrap, pairing a
    spike at -500 ms with one at +500 ms and reporting it at lag ``L``, which is
    not a physical coincidence.

    With integer spike counts every fixation's correlation is an integer
    coincidence count, so the fixation mean is an exact multiple of
    ``1 / n_fixations``.  Snapping to that grid removes FFT round-off; without
    it a lag where every null draw is identically zero acquires a spurious
    standard deviation of order 1e-17, and dividing round-off by round-off
    manufactures z-scores of 1e17.
    """
    full = np.fft.irfft(cross_spectra, n=int(n_fft), axis=-1)
    positive = full[..., : int(n_bins)]                 # lags 0 .. n_bins-1
    negative = full[..., -(int(n_bins) - 1) :]          # lags -(n_bins-1) .. -1
    traces = np.concatenate([negative, positive], axis=-1)
    if quantum is not None and quantum > 0:
        traces = np.round(traces / float(quantum)) * float(quantum)
    return traces


def compute_condition_coordination(
    trains: np.ndarray,
    *,
    settings: PairSpikeCoordinationSettings,
    bin_size_ms: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Observed and null cross-correlations for every unit pair in one condition.

    ``trains`` is ``(n_units, n_fixations, n_bins)`` of unsmoothed 1 ms counts
    covering the analysis window and nothing else.  Everything -- the observed
    correlation and both nulls -- is computed from spikes inside that window.

    Returns arrays keyed by name, each ``(n_units, n_units, n_lags)`` except
    ``lags_ms``, ``overlap_bins`` and the per-unit spike counts.  Both nulls are
    summarised by their mean and standard deviation across draws; individual
    draws are not retained because nothing downstream needs them.
    """
    n_units, n_fixations, n_bins = trains.shape
    n_fft = _next_fast_length(2 * n_bins - 1)
    trains = trains.astype(np.float64)
    spectra = np.fft.rfft(trains, n=n_fft, axis=-1)
    # Only integer-count input has an exact quantum; smoothed input does not.
    integer_input = bool(np.array_equal(trains, np.round(trains)))
    quantum = (1.0 / float(n_fixations)) if integer_input else None

    observed = _to_traces(
        _cross_spectra_all_pairs(spectra, spectra), n_bins, n_fft, quantum=quantum
    )

    # --- trial-shuffle null -------------------------------------------------
    # Each draw deranges the fixation index on the second unit, so unit A's
    # train on fixation i meets unit B's train on some other fixation.  Rate
    # profiles survive; trial-by-trial covariation does not.  The derangement
    # acts on the fixation axis only, so the already-computed spectra are
    # reused and each draw costs one inverse transform.
    shuffle_sum = np.zeros_like(observed)
    shuffle_sumsq = np.zeros_like(observed)
    n_shuffle = int(settings.n_trial_shuffle_draws)
    for _ in range(n_shuffle):
        order = _random_derangement(n_fixations, rng)
        draw = _to_traces(
            _cross_spectra_all_pairs(spectra, spectra[:, order, :]),
            n_bins,
            n_fft,
            quantum=quantum,
        )
        shuffle_sum += draw
        shuffle_sumsq += np.square(draw)

    # --- circular-shift null ------------------------------------------------
    # Each draw rotates the second unit's train *within its own analysis
    # window*, then correlates linearly.  Per-fixation spike counts and the slow
    # envelope survive; fine temporal alignment does not.
    #
    # The wrap belongs here and only here.  Rotating a train is a legitimate way
    # to destroy alignment in a null, whereas letting the *observed*
    # correlation wrap would pair a spike at -500 ms with one at +500 ms and
    # report it as a coincidence -- which is why the observed path is linear.
    # Because the rotation happens before zero-padding, it is not a phase ramp
    # on the padded spectrum, so each draw re-transforms the rolled trains.
    min_shift = max(1, int(round(float(settings.min_circular_shift_ms) / float(bin_size_ms))))
    if 2 * min_shift >= n_bins:
        raise ValueError(
            "min_circular_shift_ms leaves no admissible shifts for the analysis window."
        )
    shift_sum = np.zeros_like(observed)
    shift_sumsq = np.zeros_like(observed)
    n_shift = int(settings.n_circular_shift_draws)
    time_index = np.arange(n_bins)
    fixation_index = np.arange(n_fixations)[:, None]
    for _ in range(n_shift):
        shifts = rng.integers(min_shift, n_bins - min_shift, size=n_fixations)
        # One gather instead of a Python loop over fixations: entry (f, t) of
        # the rolled train is the original at (t - shift_f) modulo the window.
        gather = (time_index[None, :] - shifts[:, None]) % n_bins
        rolled = trains[:, fixation_index, gather]
        rolled_spectra = np.fft.rfft(rolled, n=n_fft, axis=-1)
        draw = _to_traces(
            _cross_spectra_all_pairs(spectra, rolled_spectra),
            n_bins,
            n_fft,
            quantum=quantum,
        )
        shift_sum += draw
        shift_sumsq += np.square(draw)

    def moments(total: np.ndarray, total_sq: np.ndarray, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
        mean = total / float(n_draws)
        variance = np.maximum(total_sq / float(n_draws) - np.square(mean), 0.0)
        if n_draws > 1:
            variance *= float(n_draws) / float(n_draws - 1)
        return mean, np.sqrt(variance)

    shuffle_mean, shuffle_sd = moments(shuffle_sum, shuffle_sumsq, n_shuffle)
    shift_mean, shift_sd = moments(shift_sum, shift_sumsq, n_shift)

    _, lags_ms = _lag_axis(n_bins, bin_size_ms)
    return {
        "lags_ms": lags_ms,
        "overlap_bins": overlap_bins(n_bins),
        "observed": observed,
        "trial_shuffle_mean": shuffle_mean,
        "trial_shuffle_sd": shuffle_sd,
        "circular_shift_mean": shift_mean,
        "circular_shift_sd": shift_sd,
        "spike_counts": trains.sum(axis=(1, 2)),
        "n_fixations": np.int64(n_fixations),
    }


def verify_null_identities(
    *,
    n_bins: int = 64,
    n_fixations: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    """Check the fast path against brute-force linear correlation.

    Cheap enough to run at the top of a notebook, which is the point: the
    speed-ups are only worth having if they are provably the same number.  The
    reference here is an explicit sum over genuinely overlapping bins only --
    no wraparound -- which is what the observed statistic has to be.
    """
    rng = np.random.default_rng(seed)
    x = (rng.random((n_fixations, n_bins)) < 0.1).astype(float)
    y = (rng.random((n_fixations, n_bins)) < 0.1).astype(float)
    n_fft = _next_fast_length(2 * n_bins - 1)
    lags = np.arange(-(n_bins - 1), n_bins)

    def brute_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """sum_t a[t] * b[t - lag], over overlapping bins only."""
        out = np.empty(lags.size)
        for index, lag in enumerate(lags):
            total = 0.0
            for t in range(n_bins):
                source = t - lag
                if 0 <= source < n_bins:
                    total += a[t] * b[source]
            out[index] = total
        return out

    def fast(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(a, n=n_fft, axis=-1) * np.fft.rfft(b, n=n_fft, axis=-1).conj()
        return _to_traces(spectrum.mean(axis=0), n_bins, n_fft)

    rows: list[dict] = []

    direct = np.mean([brute_linear(x[i], y[i]) for i in range(n_fixations)], axis=0)
    rows.append(
        {
            "identity": "mean over fixations == inverse transform of mean cross-spectrum",
            "max_abs_error": float(np.max(np.abs(direct - fast(x, y)))),
        }
    )

    order = _random_derangement(n_fixations, rng)
    direct = np.mean([brute_linear(x[i], y[order[i]]) for i in range(n_fixations)], axis=0)
    rows.append(
        {
            "identity": "trial-shuffle draw == same transform with a deranged operand",
            "max_abs_error": float(np.max(np.abs(direct - fast(x, y[order])))),
        }
    )

    shifts = rng.integers(1, n_bins - 1, size=n_fixations)
    rolled = np.stack([np.roll(y[i], int(shifts[i])) for i in range(n_fixations)])
    direct = np.mean([brute_linear(x[i], rolled[i]) for i in range(n_fixations)], axis=0)
    rows.append(
        {
            "identity": "circular-shift draw == linear correlation of the rolled operand",
            "max_abs_error": float(np.max(np.abs(direct - fast(x, rolled)))),
        }
    )

    # The observed path must not wrap: an unpadded transform is a different,
    # non-physical statistic, and this records how different.
    wrapped = np.fft.fftshift(
        np.fft.irfft(
            (
                np.fft.rfft(x, n=n_bins, axis=-1)
                * np.fft.rfft(y, n=n_bins, axis=-1).conj()
            ).mean(axis=0),
            n=n_bins,
        )
    )
    linear = fast(x, y)
    centre = {int(lag): value for lag, value in zip(lags, linear)}
    wrapped_lags = np.fft.fftshift((np.fft.fftfreq(n_bins) * n_bins).astype(int))
    gap = max(
        abs(float(value) - centre[int(lag)]) for lag, value in zip(wrapped_lags, wrapped)
    )
    rows.append(
        {
            "identity": "unpadded (wrapping) transform differs from the linear one",
            "max_abs_error": float(gap),
        }
    )

    frame = pd.DataFrame(rows)
    frame["passes"] = frame["max_abs_error"] < IDENTITY_TOLERANCE
    # The last row documents a difference rather than an identity: it is
    # expected to be non-zero, and a zero there would mean the padding was lost.
    frame.loc[frame.index[-1], "passes"] = bool(
        frame.loc[frame.index[-1], "max_abs_error"] > IDENTITY_TOLERANCE
    )
    return frame


# ---------------------------------------------------------------------------
# Per-session pair tables
# ---------------------------------------------------------------------------
#
# Two standardised quantities are stored, and they answer different questions.
#
# ``z``       (observed - null mean) / null SD, where the null SD is the spread
#             of the *fixation-averaged* statistic across draws.  This is the
#             significance of the excess for that pair.  It grows with the
#             square root of the fixation count, so it is the right quantity for
#             "is this pair coordinated at all" and the wrong one for comparing
#             conditions that differ in how many fixations they contain.
#
# ``effect``  z / sqrt(n_fixations), which is the excess expressed in units of
#             the single-fixation null SD.  Its expectation does not depend on
#             the fixation count, so this is the quantity to compare across
#             conditions.  It matters here: interactive-face fixations outnumber
#             non-interactive-face fixations roughly five to one, so ranking
#             conditions by z would rank them mostly by trial count.
#
# ``trial_match_conditions`` additionally recomputes everything on a common
# fixation count per session, as a direct control on the same confound.

#: Half-widths (ms) over which the z traces are averaged into scalar summaries.
SUMMARY_WINDOWS_MS: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 100.0)

#: Lag range searched for the coordination peak.
DEFAULT_PEAK_SEARCH_MS = 100.0

NULL_NAMES: tuple[str, ...] = ("trial_shuffle", "circular_shift")


def _summarize_pair_traces(
    lags_ms: np.ndarray,
    observed: np.ndarray,
    null_mean: np.ndarray,
    null_sd: np.ndarray,
    *,
    n_fixations: int,
    prefix: str,
    peak_search_ms: float = DEFAULT_PEAK_SEARCH_MS,
) -> dict[str, object]:
    """Scalar summaries of one pair's excess over one null.

    Note on ``*_peak_z``: it is a maximum over every lag in the search window,
    so it is inflated under the null by construction --
    :func:`verify_null_sensitivity` shows an uncoupled pair reaching a peak z
    near 3.  Use it to read off *where* coordination sits, and use the windowed
    ``*_mean_z_pm*ms`` columns, which are not maxima, to test whether it is
    there at all.
    """
    excess = observed - null_mean
    # A lag where every null draw landed on the same value has no spread to
    # standardise against.  If the observed value matches it too, there is
    # genuinely no excess and z is 0; if it does not, the excess is real but
    # unstandardisable from this many draws, so it is left undefined rather
    # than reported as an enormous z.
    degenerate = null_sd <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z_trace = np.where(degenerate, np.nan, excess / np.where(degenerate, 1.0, null_sd))
    z_trace = np.where(degenerate & np.isclose(excess, 0.0, atol=1e-12), 0.0, z_trace)
    effect_trace = z_trace / np.sqrt(float(max(n_fixations, 1)))

    out: dict[str, object] = {}
    zero_index = int(np.argmin(np.abs(lags_ms)))
    out[f"{prefix}_zero_lag_z"] = float(z_trace[zero_index])
    out[f"{prefix}_zero_lag_effect"] = float(effect_trace[zero_index])

    search = np.abs(lags_ms) <= float(peak_search_ms)
    if search.any() and np.isfinite(z_trace[search]).any():
        local = np.where(search, z_trace, -np.inf)
        peak_index = int(np.nanargmax(local))
        out[f"{prefix}_peak_z"] = float(z_trace[peak_index])
        out[f"{prefix}_peak_effect"] = float(effect_trace[peak_index])
        out[f"{prefix}_peak_lag_ms"] = float(lags_ms[peak_index])
    else:
        out[f"{prefix}_peak_z"] = np.nan
        out[f"{prefix}_peak_effect"] = np.nan
        out[f"{prefix}_peak_lag_ms"] = np.nan

    for half_width in SUMMARY_WINDOWS_MS:
        window = np.abs(lags_ms) <= float(half_width)
        label = f"{prefix}_mean_z_pm{int(half_width)}ms"
        effect_label = f"{prefix}_mean_effect_pm{int(half_width)}ms"
        if window.any():
            out[label] = float(np.nanmean(z_trace[window]))
            out[effect_label] = float(np.nanmean(effect_trace[window]))
        else:
            out[label] = np.nan
            out[effect_label] = np.nan
    return out


def _crop_lags(lags_ms: np.ndarray, max_lag_ms: float) -> np.ndarray:
    return np.abs(lags_ms) <= float(max_lag_ms) + 1e-9


def build_session_pair_table(
    session: SessionSpikeTrains,
    settings: PairSpikeCoordinationSettings,
    *,
    selective_unit_keys: Optional[set[tuple[str, ...]]] = None,
) -> Optional[dict]:
    """Compute every simultaneously recorded pair's coordination for one session.

    Pairs are unordered but stored with a fixed orientation: ``unit_1`` is the
    earlier entry in the session's unit table, and a **positive lag means
    ``unit_1`` fired after ``unit_2``**.
    """
    rng = np.random.default_rng(
        abs(hash((settings.random_seed, session.date, session.session))) % (2**32)
    )
    selective_unit_keys = selective_unit_keys or set()

    n_units = session.n_units
    if n_units < 2:
        return None

    regions = [str(key[UNIT_KEY_COLUMNS.index("region")]).lower() for key in session.unit_keys]
    is_selective = [key in selective_unit_keys for key in session.unit_keys]

    matched_n: Optional[int] = None
    if settings.trial_match_conditions:
        counts = [array.shape[1] for array in session.trains.values()]
        if len(counts) == len(settings.conditions) and counts:
            candidate = int(min(counts))
            if candidate >= max(2, settings.min_fixations_per_condition):
                matched_n = candidate

    records: list[dict] = []
    lags_stored: Optional[np.ndarray] = None
    overlap_stored: Optional[np.ndarray] = None

    for condition in settings.conditions:
        trains = session.trains.get(condition)
        if trains is None:
            continue

        result = compute_condition_coordination(
            trains, settings=settings, bin_size_ms=session.bin_size_ms, rng=rng
        )
        lags_ms = result["lags_ms"]
        keep = _crop_lags(lags_ms, settings.store_max_lag_ms)
        if lags_stored is None:
            lags_stored = lags_ms[keep].astype(np.float32)
            overlap_stored = result["overlap_bins"][keep].astype(np.float32)

        matched_result = None
        if matched_n is not None and matched_n <= trains.shape[1]:
            # When this condition already has the common count, the subsample is
            # the whole set; computing it anyway keeps the matched columns
            # present for every condition instead of NaN for the smallest one.
            chosen = rng.choice(trains.shape[1], size=matched_n, replace=False)
            matched_result = compute_condition_coordination(
                trains[:, np.sort(chosen), :],
                settings=settings,
                bin_size_ms=session.bin_size_ms,
                rng=rng,
            )

        spike_counts = result["spike_counts"]
        n_fixations = int(result["n_fixations"])

        for first in range(n_units):
            if spike_counts[first] < settings.min_spikes_per_unit_condition:
                continue
            for second in range(first + 1, n_units):
                if spike_counts[second] < settings.min_spikes_per_unit_condition:
                    continue
                same_region = regions[first] == regions[second]
                if same_region and not settings.include_within_region:
                    continue
                if not same_region and not settings.include_cross_region:
                    continue

                record: dict[str, object] = {
                    "date": session.date,
                    "session": session.session,
                    "condition": condition,
                    "region_1": regions[first],
                    "region_2": regions[second],
                    "same_region": bool(same_region),
                    "region_pair": (
                        regions[first]
                        if same_region
                        else "-".join(sorted((regions[first], regions[second])))
                    ),
                    "n_fixations": n_fixations,
                    "n_spikes_1": float(spike_counts[first]),
                    "n_spikes_2": float(spike_counts[second]),
                    "both_selective": bool(is_selective[first] and is_selective[second]),
                    "any_selective": bool(is_selective[first] or is_selective[second]),
                }
                for slot, unit_index in (("1", first), ("2", second)):
                    for column, value in zip(UNIT_KEY_COLUMNS, session.unit_keys[unit_index]):
                        if column == "date":
                            continue
                        record[f"{column}_{slot}"] = value
                record["pair_key"] = "|".join(
                    [
                        session.date,
                        session.session,
                        str(record["unit_uuid_1"]),
                        str(record["unit_uuid_2"]),
                    ]
                )

                observed = result["observed"][first, second]
                record["observed"] = observed[keep].astype(np.float32)
                for null_name in NULL_NAMES:
                    null_mean = result[f"{null_name}_mean"][first, second]
                    null_sd = result[f"{null_name}_sd"][first, second]
                    record[f"{null_name}_mean"] = null_mean[keep].astype(np.float32)
                    record[f"{null_name}_sd"] = null_sd[keep].astype(np.float32)
                    record.update(
                        _summarize_pair_traces(
                            lags_ms,
                            observed,
                            null_mean,
                            null_sd,
                            n_fixations=n_fixations,
                            prefix=null_name,
                        )
                    )
                    if matched_result is not None:
                        record.update(
                            {
                                f"{key}_matched": value
                                for key, value in _summarize_pair_traces(
                                    matched_result["lags_ms"],
                                    matched_result["observed"][first, second],
                                    matched_result[f"{null_name}_mean"][first, second],
                                    matched_result[f"{null_name}_sd"][first, second],
                                    n_fixations=int(matched_result["n_fixations"]),
                                    prefix=null_name,
                                ).items()
                            }
                        )
                if matched_result is not None:
                    record["n_fixations_matched"] = int(matched_result["n_fixations"])
                records.append(record)

    if not records:
        return None

    return {
        "overlap_bins": overlap_stored,
        "meta": {
            "date": session.date,
            "session": session.session,
            "conditions": tuple(settings.conditions),
            "signal_input_column": settings.signal_input_column,
            "signal_window_ms": session.window_ms,
            "bin_size_ms": session.bin_size_ms,
            "store_max_lag_ms": settings.store_max_lag_ms,
            "n_trial_shuffle_draws": settings.n_trial_shuffle_draws,
            "n_circular_shift_draws": settings.n_circular_shift_draws,
            "min_circular_shift_ms": settings.min_circular_shift_ms,
            # Linear: at every lag only genuinely overlapping bins contribute,
            # so no spike is ever paired with one a full window away.  The wrap
            # appears only inside the circular-shift null, where destroying
            # alignment is the intent.
            "correlation_kind": "linear",
            "null_kinds": {
                "trial_shuffle": "derangement of the fixation index",
                "circular_shift": "rotation within the analysis window",
            },
            "lag_sign_convention": "positive lag = unit_1 fires after unit_2",
            "selectivity_column": settings.selectivity_column,
            "trial_matched_n": matched_n,
        },
        "lags_ms": lags_stored,
        "pairs": pd.DataFrame.from_records(records),
    }


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


def load_selective_unit_keys(
    settings: PairSpikeCoordinationSettings,
) -> set[tuple[str, ...]]:
    """Unit identities called selective by the single-unit analysis.

    Loaded from the ``three_condition_core`` family so the pair analysis and the
    single-unit analysis are talking about the same units and the same contrast
    set.  The FDR-corrected column is the project default.
    """
    cfg = load_config(settings.cfg_path)
    path = build_analysis_output_dir(cfg, settings.selectivity_subdir) / settings.selectivity_unit_filename
    if not path.exists():
        return set()
    frame = pd.read_csv(path, dtype=str)
    column = settings.selectivity_column
    if column not in frame.columns:
        raise ValueError(
            f"Selectivity file {path} has no column {column!r}; available: {list(frame.columns)}"
        )
    flags = frame[column].astype(str).str.strip().str.lower().isin({"true", "1"})
    selected = frame.loc[flags]
    return {
        tuple(str(value) for value in row)
        for row in selected.loc[:, list(UNIT_KEY_COLUMNS)].values
    }


def iter_session_trial_paths(settings: PairSpikeCoordinationSettings) -> list[dict]:
    """Locate every session's 1 ms spike-train trial file."""
    cfg = load_config(settings.cfg_path)
    root = Path(cfg["processed_data_root"])
    filename = settings.trial_input_filename
    rows: list[dict] = []
    for path in sorted(root.glob(f"date=*/session=*/{settings.trial_input_modality}/{filename}")):
        parts = path.parts
        date = next(part.split("=", 1)[1] for part in parts if part.startswith("date="))
        session = next(part.split("=", 1)[1] for part in parts if part.startswith("session="))
        if settings.dates is not None and date not in {str(v) for v in settings.dates}:
            continue
        if settings.sessions is not None and session not in {str(v) for v in settings.sessions}:
            continue
        rows.append({"date": date, "session": session, "path": path})
    return rows


def session_output_path(settings: PairSpikeCoordinationSettings, date: str, session: str) -> Path:
    cfg = load_config(settings.cfg_path)
    root = build_analysis_output_dir(cfg, settings.output_subdir)
    return root / f"date={date}" / f"session={session}" / settings.output_filename


def process_session(
    settings: PairSpikeCoordinationSettings,
    row: Mapping,
    *,
    selective_unit_keys: Optional[set[tuple[str, ...]]] = None,
    overwrite: bool = True,
) -> Optional[Path]:
    """Compute and save one session's pair coordination table."""
    out_path = session_output_path(settings, str(row["date"]), str(row["session"]))
    if out_path.exists() and not overwrite:
        return out_path
    session = load_session_spike_trains(row["path"], settings)
    if session is None:
        return None
    if not session.date:
        session.date = str(row["date"])
    if not session.session:
        session.session = str(row["session"])
    payload = build_session_pair_table(
        session, settings, selective_unit_keys=selective_unit_keys
    )
    if payload is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, out_path)
    return out_path


def _session_worker(args) -> Optional[str]:
    settings, row, selective_unit_keys, overwrite = args
    path = process_session(
        settings, row, selective_unit_keys=selective_unit_keys, overwrite=overwrite
    )
    return None if path is None else str(path)


def run_pair_spike_coordination_build(
    settings: PairSpikeCoordinationSettings,
    *,
    overwrite: bool = True,
) -> list[Path]:
    """Run the pair coordination build over every selected session."""
    rows = iter_session_trial_paths(settings)
    if settings.test_single:
        rows = rows[:1]
    if not rows:
        return []
    selective_unit_keys = load_selective_unit_keys(settings)

    written: list[Path] = []
    if settings.use_parallel and len(rows) > 1:
        import multiprocessing as mp

        procs = settings.max_procs or max(1, min(len(rows), mp.cpu_count()))
        payloads = [(settings, row, selective_unit_keys, overwrite) for row in rows]
        with mp.Pool(processes=int(procs)) as pool:
            for result in pool.imap_unordered(_session_worker, payloads):
                if result is not None:
                    written.append(Path(result))
    else:
        for row in rows:
            path = process_session(
                settings, row, selective_unit_keys=selective_unit_keys, overwrite=overwrite
            )
            if path is not None:
                written.append(path)
    return sorted(written)


# ---------------------------------------------------------------------------
# Aggregation across sessions
# ---------------------------------------------------------------------------


#: Trace columns saved per pair-condition.
TRACE_COLUMNS: tuple[str, ...] = (
    "observed",
    "trial_shuffle_mean",
    "trial_shuffle_sd",
    "circular_shift_mean",
    "circular_shift_sd",
)


def load_pair_coordination(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    output_filename: str = "pair_coordination.pkl",
    with_traces: bool = False,
    dates: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Concatenate every session's pair table.

    ``with_traces=False`` (the default) drops the per-lag arrays, which is what
    every scalar summary and statistical test needs and keeps the table small
    enough to hold comfortably in a notebook.
    """
    cfg = load_config(cfg_path)
    root = build_analysis_output_dir(cfg, output_subdir)
    paths = sorted(root.glob(f"date=*/session=*/{output_filename}"))
    if dates is not None:
        wanted = {str(value) for value in dates}
        paths = [p for p in paths if p.parent.parent.name.split("=", 1)[1] in wanted]
    if not paths:
        raise FileNotFoundError(f"No pair coordination outputs under {root}.")

    frames: list[pd.DataFrame] = []
    lags: Optional[np.ndarray] = None
    for path in paths:
        payload = pd.read_pickle(path)
        frame = payload["pairs"]
        if lags is None:
            lags = np.asarray(payload["lags_ms"], dtype=float)
        if not with_traces:
            frame = frame.drop(columns=[c for c in TRACE_COLUMNS if c in frame.columns])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["condition"] = pd.Categorical(
        combined["condition"], categories=list(CONDITION_ORDER), ordered=True
    )
    return combined, (np.asarray([]) if lags is None else lags)


def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(n_boot), values.size), replace=True).mean(axis=1)
    return (
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )


def summarize_coordination(
    pairs: pd.DataFrame,
    *,
    metric: str = "trial_shuffle_mean_effect_pm10ms",
    z_column: str = "trial_shuffle_mean_z_pm10ms",
    group_columns: Sequence[str] = ("scope", "region_pair", "condition"),
    z_threshold: float = 1.96,
) -> pd.DataFrame:
    """Group-level coordination summary with bootstrap confidence intervals.

    ``metric`` should be an ``_effect`` column when conditions are being
    compared, because the ``_z`` columns scale with the fixation count.
    ``z_column`` is reported alongside purely to answer "what fraction of pairs
    are individually above null".
    """
    frame = pairs.copy()
    if "scope" not in frame.columns:
        frame["scope"] = np.where(frame["same_region"], "within_region", "cross_region")

    rows: list[dict] = []
    for keys, group in frame.groupby(list(group_columns), observed=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        low, high = _bootstrap_ci(finite)
        row = dict(zip(group_columns, [str(k) for k in keys]))
        row.update(
            {
                "n_pairs": int(len(group)),
                "n_pairs_finite": int(finite.size),
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "sem": float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                if finite.size > 1
                else np.nan,
                "median": float(np.median(finite)) if finite.size else np.nan,
                "ci_low": low,
                "ci_high": high,
                "median_n_fixations": float(np.median(group["n_fixations"])),
            }
        )
        if z_column in group.columns:
            z_values = group[z_column].to_numpy(dtype=float)
            z_finite = z_values[np.isfinite(z_values)]
            row["frac_pairs_above_null"] = (
                float(np.mean(z_finite > z_threshold)) if z_finite.size else np.nan
            )
            row["mean_z"] = float(np.mean(z_finite)) if z_finite.size else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    sort_columns = [c for c in group_columns if c in result.columns]
    return result.sort_values(sort_columns).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_against_null(
    pairs: pd.DataFrame,
    *,
    metric: str = "trial_shuffle_mean_effect_pm10ms",
    group_columns: Sequence[str] = ("scope", "condition"),
) -> pd.DataFrame:
    """Is coordination above the null at all, per group?

    A one-sample Wilcoxon signed-rank test of the per-pair excess against zero.
    Zero is the null's own expectation, so this asks whether the *population* of
    pairs sits above its null, without assuming the per-pair excess is normal.
    """
    from scipy.stats import wilcoxon

    frame = pairs.copy()
    if "scope" not in frame.columns:
        frame["scope"] = np.where(frame["same_region"], "within_region", "cross_region")

    rows: list[dict] = []
    for keys, group in frame.groupby(list(group_columns), observed=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        row = dict(zip(group_columns, [str(k) for k in keys]))
        row["n_pairs"] = int(values.size)
        row["mean_excess"] = float(np.mean(values)) if values.size else np.nan
        row["median_excess"] = float(np.median(values)) if values.size else np.nan
        if values.size >= 10 and np.any(values != 0):
            statistic, p_value = wilcoxon(values, alternative="greater")
            row["statistic"] = float(statistic)
            row["p_value"] = float(p_value)
        else:
            row["statistic"] = np.nan
            row["p_value"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_columns)).reset_index(drop=True)


def compare_conditions(
    pairs: pd.DataFrame,
    *,
    metric: str = "trial_shuffle_mean_effect_pm10ms",
    group_columns: Sequence[str] = ("scope",),
    conditions: Sequence[str] = CONDITION_ORDER,
    pvalue_correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Compare conditions **within the same pair**, then test across pairs.

    Every pair contributes all three conditions, so the comparison is paired:
    the same two neurons, the same electrodes, the same session, differing only
    in which fixations were used.  That removes pair identity, firing rate and
    recording quality as explanations in one step, which an unpaired comparison
    across separate pair populations could not do.

    Uses a Wilcoxon signed-rank test on the within-pair difference, with
    Benjamini-Hochberg correction across the condition contrasts reported.
    """
    from itertools import combinations

    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    frame = pairs.copy()
    if "scope" not in frame.columns:
        frame["scope"] = np.where(frame["same_region"], "within_region", "cross_region")

    index_columns = ["pair_key"] + [c for c in group_columns if c != "condition"]
    wide = frame.pivot_table(
        index=index_columns, columns="condition", values=metric, observed=True
    )

    rows: list[dict] = []
    for keys, group in wide.groupby([c for c in group_columns if c != "condition"], observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for first, second in combinations(conditions, 2):
            if first not in group.columns or second not in group.columns:
                continue
            paired = group.loc[:, [first, second]].dropna()
            row = dict(zip([c for c in group_columns if c != "condition"], [str(k) for k in keys]))
            row.update(
                {
                    "condition_a": first,
                    "condition_b": second,
                    "n_pairs": int(len(paired)),
                    "mean_a": float(paired[first].mean()) if len(paired) else np.nan,
                    "mean_b": float(paired[second].mean()) if len(paired) else np.nan,
                }
            )
            row["mean_difference"] = row["mean_a"] - row["mean_b"]
            differences = (paired[first] - paired[second]).to_numpy(dtype=float)
            if differences.size >= 10 and np.any(differences != 0):
                statistic, p_value = wilcoxon(differences, alternative="two-sided")
                row["statistic"] = float(statistic)
                row["p_value"] = float(p_value)
                # Matched-pairs rank-biserial correlation: an effect size that
                # does not inherit the sample size the way the statistic does.
                positive = float(np.sum(differences > 0))
                negative = float(np.sum(differences < 0))
                total = positive + negative
                row["effect_size_rank_biserial"] = (
                    (positive - negative) / total if total else np.nan
                )
            else:
                row["statistic"] = np.nan
                row["p_value"] = np.nan
                row["effect_size_rank_biserial"] = np.nan
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    testable = result["p_value"].notna()
    result["p_value_corrected"] = np.nan
    result["significant"] = False
    if testable.any():
        reject, corrected, _, _ = multipletests(
            result.loc[testable, "p_value"].to_numpy(dtype=float), method=pvalue_correction
        )
        result.loc[testable, "p_value_corrected"] = corrected
        result.loc[testable, "significant"] = reject
    result["pvalue_correction"] = pvalue_correction
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Zero-lag artifact diagnostics
# ---------------------------------------------------------------------------


def build_zero_lag_diagnostics(
    pairs: pd.DataFrame,
    *,
    null_name: str = "circular_shift",
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Per-date, per-scope zero-lag excess, for spotting a contaminated session.

    Two randomly sampled neurons are essentially never monosynaptically
    connected, so a sharp zero-lag peak across *many* pairs at once is not a
    biological pairwise interaction.  It is common input within the fixation --
    movement, arousal, or a shared reference/ground artifact on the recording
    system.  The signature that distinguishes an artifact from a real effect is
    that it is a property of the *day and array*, not of the pair: on a
    contaminated day nearly every simultaneously recorded pair shows it,
    including pairs whose units share nothing else.

    The returned table reports, per date, the fraction of pairs with a zero-lag
    z above ``z_threshold`` and how sharply the zero-lag bin stands out from its
    immediate neighbours, so a day carrying the artifact is visible as an
    outlier instead of being averaged into the population result.
    """
    frame = pairs.copy()
    if "scope" not in frame.columns:
        frame["scope"] = np.where(frame["same_region"], "within_region", "cross_region")

    z_column = f"{null_name}_zero_lag_z"
    neighbour_column = f"{null_name}_mean_z_pm25ms"
    rows: list[dict] = []
    for (date, scope), group in frame.groupby(["date", "scope"], observed=True):
        z_values = group[z_column].to_numpy(dtype=float)
        z_values = z_values[np.isfinite(z_values)]
        row = {
            "date": str(date),
            "scope": str(scope),
            "n_pairs": int(len(group)),
            "mean_zero_lag_z": float(np.mean(z_values)) if z_values.size else np.nan,
            "median_zero_lag_z": float(np.median(z_values)) if z_values.size else np.nan,
            "frac_pairs_zero_lag_above": (
                float(np.mean(z_values > z_threshold)) if z_values.size else np.nan
            ),
        }
        if neighbour_column in group.columns:
            # How much the exact zero-lag bin exceeds the surrounding +/-25 ms.
            # A biological interaction is not confined to a single 1 ms bin
            # across an entire day's worth of unrelated pairs; an artifact is.
            excess = group[z_column].to_numpy(dtype=float) - group[
                neighbour_column
            ].to_numpy(dtype=float)
            excess = excess[np.isfinite(excess)]
            row["mean_zero_lag_sharpness"] = float(np.mean(excess)) if excess.size else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Flag dates whose zero-lag prevalence is a robust outlier against the rest.
    for scope, group in result.groupby("scope"):
        values = group["frac_pairs_zero_lag_above"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < 4:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        scale = mad * 1.4826 if mad > 0 else np.nan
        result.loc[group.index, "zero_lag_outlier_score"] = (
            (group["frac_pairs_zero_lag_above"] - median) / scale
            if np.isfinite(scale) and scale > 0
            else np.nan
        )
    result["suspected_zero_lag_artifact"] = result.get(
        "zero_lag_outlier_score", pd.Series(np.nan, index=result.index)
    ).gt(3.5)
    return result.sort_values(["scope", "date"]).reset_index(drop=True)


def build_pair_inventory(pairs: pd.DataFrame) -> pd.DataFrame:
    """Counts of pairs entering each comparison, by scope, condition and selectivity."""
    frame = pairs.copy()
    if "scope" not in frame.columns:
        frame["scope"] = np.where(frame["same_region"], "within_region", "cross_region")
    rows: list[dict] = []
    for (scope, condition), group in frame.groupby(["scope", "condition"], observed=True):
        rows.append(
            {
                "scope": str(scope),
                "condition": str(condition),
                "n_pairs": int(len(group)),
                "n_pairs_both_selective": int(group["both_selective"].sum()),
                "n_dates": int(group["date"].nunique()),
                "n_sessions": int(group.groupby(["date", "session"]).ngroups),
                "median_n_fixations": float(np.median(group["n_fixations"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "condition"]).reset_index(drop=True)


def verify_null_sensitivity(
    *,
    n_fixations: int = 120,
    n_bins: int = 1000,
    base_rate: float = 0.01,
    seed: int = 0,
    n_draws: int = 60,
) -> pd.DataFrame:
    """Check that each null responds only to the structure it is meant to detect.

    Four synthetic pairs are built on the same fixation-locked rate profile, and
    each is passed through the real computation:

    ``independent``
        No coupling at all.  Both nulls should sit at zero.  If they do not, the
        nulls are mis-specified and every downstream z is inflated.

    ``shared_rate``
        The two units share a per-fixation gain -- some fixations drive both
        cells harder -- but their spikes are otherwise independent within a
        fixation.  The trial-shuffle null should detect this, because it is
        exactly the trial-by-trial covariation that null destroys.  The
        circular-shift null should *not*, because rotating within a fixation
        preserves that fixation's gain.

    ``synchronous``
        A fraction of spikes are copied to the partner at a fixed lag.  Both
        nulls should detect it.

    ``common_zero_lag``
        Spikes injected into both units in the same 1 ms bin, the signature of a
        shared artifact rather than a pairwise interaction.  Both nulls detect
        it, and it appears as a single-bin spike -- which is what
        :func:`build_zero_lag_diagnostics` looks for.

    Returns one row per scenario with the excess each null reports.
    """
    rng = np.random.default_rng(seed)
    settings = PairSpikeCoordinationSettings(
        cfg_path="",
        n_trial_shuffle_draws=int(n_draws),
        n_circular_shift_draws=int(n_draws),
    )

    time_axis = np.arange(n_bins)
    # A fixation-locked rate bump, shared by both units in every scenario, so
    # that shared *rate* structure alone can never be mistaken for coordination.
    profile = base_rate * (1.0 + 1.5 * np.exp(-0.5 * ((time_axis - n_bins / 2) / 60.0) ** 2))

    def draw_spikes(rate: np.ndarray) -> np.ndarray:
        return (rng.random(rate.shape) < rate).astype(np.float64)

    scenarios: dict[str, np.ndarray] = {}

    rates = np.broadcast_to(profile, (n_fixations, n_bins))
    scenarios["independent"] = np.stack([draw_spikes(rates), draw_spikes(rates)])

    gain = rng.lognormal(mean=0.0, sigma=0.6, size=(n_fixations, 1))
    shared = np.clip(rates * gain, 0.0, 1.0)
    scenarios["shared_rate"] = np.stack([draw_spikes(shared), draw_spikes(shared)])

    lag = 4
    first = draw_spikes(rates)
    second = draw_spikes(rates)
    copy_mask = rng.random(first.shape) < 0.25
    second = np.clip(second + np.roll(first * copy_mask, lag, axis=1), 0.0, 1.0)
    scenarios["synchronous"] = np.stack([first, second])

    first = draw_spikes(rates)
    second = draw_spikes(rates)
    common = (rng.random(first.shape) < 0.004).astype(np.float64)
    scenarios["common_zero_lag"] = np.stack(
        [np.clip(first + common, 0.0, 1.0), np.clip(second + common, 0.0, 1.0)]
    )

    rows: list[dict] = []
    for name, trains in scenarios.items():
        result = compute_condition_coordination(
            trains, settings=settings, bin_size_ms=1.0, rng=np.random.default_rng(seed + 1)
        )
        row: dict[str, object] = {"scenario": name}
        for null_name in NULL_NAMES:
            summary = _summarize_pair_traces(
                result["lags_ms"],
                result["observed"][0, 1],
                result[f"{null_name}_mean"][0, 1],
                result[f"{null_name}_sd"][0, 1],
                n_fixations=n_fixations,
                prefix=null_name,
            )
            row[f"{null_name}_peak_z"] = summary[f"{null_name}_peak_z"]
            row[f"{null_name}_peak_lag_ms"] = summary[f"{null_name}_peak_lag_ms"]
            row[f"{null_name}_mean_z_pm10ms"] = summary[f"{null_name}_mean_z_pm10ms"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Group-mean lag traces
# ---------------------------------------------------------------------------


def _pair_z_trace(
    observed: np.ndarray,
    null_mean: np.ndarray,
    null_sd: np.ndarray,
) -> np.ndarray:
    """Per-lag z trace for one pair, with degenerate lags left undefined."""
    observed = np.asarray(observed, dtype=float)
    null_mean = np.asarray(null_mean, dtype=float)
    null_sd = np.asarray(null_sd, dtype=float)
    excess = observed - null_mean
    degenerate = null_sd <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(degenerate, np.nan, excess / np.where(degenerate, 1.0, null_sd))
    return np.where(degenerate & np.isclose(excess, 0.0, atol=1e-12), 0.0, z)


def build_group_traces(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    output_filename: str = "pair_coordination.pkl",
    group_columns: Sequence[str] = ("scope", "condition"),
    selective_only: bool = False,
) -> dict:
    """Group-mean lag traces, accumulated one session at a time.

    Carries the **raw** observed correlation and each null's level alongside the
    standardised excess, because those are different questions: the raw traces
    show what the correlation and its nulls actually look like, and the z traces
    show how far apart they are.  A figure that only ever plots z hides whether
    the nulls sit where they should.

    Holding every pair's traces in memory would cost several gigabytes and no
    figure needs them individually, so this streams the per-session files and
    keeps running sums only.

    Returns ``{"lags_ms", "overlap_bins", "traces"}`` where each row of
    ``traces`` carries, per group, the mean and standard error across pairs of:
    ``observed``, ``trial_shuffle_null``, ``circular_shift_null``, and the two
    excess traces ``z_trial_shuffle`` / ``z_circular_shift``.
    """
    cfg = load_config(cfg_path)
    root = build_analysis_output_dir(cfg, output_subdir)
    paths = sorted(root.glob(f"date=*/session=*/{output_filename}"))
    if not paths:
        raise FileNotFoundError(f"No pair coordination outputs under {root}.")

    channels = ("observed",) + tuple(f"{name}_null" for name in NULL_NAMES) + tuple(
        f"z_{name}" for name in NULL_NAMES
    )
    lags: Optional[np.ndarray] = None
    overlap: Optional[np.ndarray] = None
    sums: dict[tuple, dict[str, dict[str, np.ndarray]]] = {}

    for path in paths:
        payload = pd.read_pickle(path)
        frame = payload["pairs"]
        if lags is None:
            lags = np.asarray(payload["lags_ms"], dtype=float)
            overlap = np.asarray(payload.get("overlap_bins", []), dtype=float)
        if "scope" not in frame.columns:
            frame = frame.assign(
                scope=np.where(frame["same_region"], "within_region", "cross_region")
            )
        if selective_only:
            frame = frame.loc[frame["both_selective"]]
        if frame.empty:
            continue

        for keys, group in frame.groupby(list(group_columns), observed=True, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            keys = tuple(str(value) for value in keys)
            stacks: dict[str, np.ndarray] = {
                "observed": np.vstack([np.asarray(v, dtype=float) for v in group["observed"]])
            }
            for null_name in NULL_NAMES:
                stacks[f"{null_name}_null"] = np.vstack(
                    [np.asarray(v, dtype=float) for v in group[f"{null_name}_mean"]]
                )
                stacks[f"z_{null_name}"] = np.vstack(
                    [
                        _pair_z_trace(obs, mean, sd)
                        for obs, mean, sd in zip(
                            group["observed"],
                            group[f"{null_name}_mean"],
                            group[f"{null_name}_sd"],
                        )
                    ]
                )
            accumulator = sums.setdefault(keys, {})
            for channel in channels:
                stacked = stacks[channel]
                valid = np.isfinite(stacked)
                filled = np.where(valid, stacked, 0.0)
                slot = accumulator.get(channel)
                if slot is None:
                    accumulator[channel] = {
                        "sum": filled.sum(axis=0),
                        "sumsq": np.square(filled).sum(axis=0),
                        "count": valid.sum(axis=0).astype(float),
                    }
                else:
                    slot["sum"] += filled.sum(axis=0)
                    slot["sumsq"] += np.square(filled).sum(axis=0)
                    slot["count"] += valid.sum(axis=0)

    rows: list[dict] = []
    for keys, accumulator in sums.items():
        row = dict(zip(group_columns, keys))
        n_max = 0
        for channel, slot in accumulator.items():
            count = slot["count"]
            safe = np.maximum(count, 1.0)
            mean = slot["sum"] / safe
            variance = np.maximum(slot["sumsq"] / safe - np.square(mean), 0.0)
            row[f"{channel}_mean"] = np.where(count > 0, mean, np.nan).astype(np.float32)
            row[f"{channel}_sem"] = np.where(
                count > 1, np.sqrt(variance / safe), np.nan
            ).astype(np.float32)
            n_max = max(n_max, int(count.max()) if count.size else 0)
        row["n_pairs"] = n_max
        rows.append(row)

    return {
        "lags_ms": np.asarray(lags, dtype=float),
        "overlap_bins": np.asarray(overlap, dtype=float) if overlap is not None else np.asarray([]),
        "traces": pd.DataFrame(rows),
    }


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------
#
# The notebook drives the whole pipeline through these, so that "what is built"
# is a question with a checkable answer rather than something to remember.
# Submission is never implicit: :func:`ensure_pair_coordination_built` reports
# what is missing and only submits when asked, because the array job costs real
# cluster time and a notebook cell is easy to re-run by accident.

DEFAULT_SBATCH_PATH = "hpc/ephys/run_fixation_pair_spike_coordination.sbatch"


def build_status(settings: PairSpikeCoordinationSettings) -> pd.DataFrame:
    """One row per session: whether its pair-coordination output exists.

    Compares the sessions that *have* 1 ms spike trains against the outputs
    already written, so a partially finished array job is visible as such.
    """
    rows: list[dict] = []
    for row in iter_session_trial_paths(settings):
        out_path = session_output_path(settings, str(row["date"]), str(row["session"]))
        rows.append(
            {
                "date": str(row["date"]),
                "session": str(row["session"]),
                "output_path": out_path,
                "exists": out_path.exists(),
                "size_mb": (out_path.stat().st_size / 1e6) if out_path.exists() else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "session"]).reset_index(drop=True)


def summarize_build_status(status: pd.DataFrame) -> pd.DataFrame:
    """Per-date completion counts, for spotting a failed array task."""
    if status.empty:
        return status
    grouped = status.groupby("date", observed=True).agg(
        n_sessions=("session", "count"),
        n_built=("exists", "sum"),
        total_mb=("size_mb", "sum"),
    )
    grouped["complete"] = grouped["n_built"] == grouped["n_sessions"]
    return grouped.reset_index()


def ensure_pair_coordination_built(
    settings: PairSpikeCoordinationSettings,
    *,
    submit: bool = False,
    wait: bool = True,
    sbatch_path: str | Path = DEFAULT_SBATCH_PATH,
    repo_root: Optional[Path] = None,
    poll_secs: int = 60,
    extra_args: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Report build completeness and, if asked, submit the array job and wait.

    With ``submit=False`` (the default) this only inspects and reports: nothing
    is queued, so the notebook can be re-run freely.  With ``submit=True`` it
    submits one array task per date that is not yet complete and, when ``wait``
    is set, blocks until the array finishes before returning the refreshed
    status.

    Returns the per-date status after any submission completes.
    """
    status = build_status(settings)
    per_date = summarize_build_status(status)
    if per_date.empty:
        raise FileNotFoundError(
            "No sessions with 1 ms spike trains were found; check the dataset config."
        )

    incomplete = per_date.loc[~per_date["complete"]]
    print(
        f"dates: {len(per_date)} total, {int(per_date['complete'].sum())} complete, "
        f"{len(incomplete)} incomplete"
    )
    print(
        f"sessions: {int(status['exists'].sum())} of {len(status)} built "
        f"({status['size_mb'].sum() / 1000:.2f} GB on disk)"
    )
    if incomplete.empty:
        print("nothing to build.")
        return per_date
    print(f"incomplete dates: {list(incomplete['date'])}")

    if not submit:
        all_dates = list(per_date["date"])
        indices = sorted(all_dates.index(date) for date in incomplete["date"])
        spec = _compact_array_spec(indices)
        print(
            "\nsubmit=False, so nothing was queued. To build these, run:\n"
            f"    sbatch --array={spec} {sbatch_path}\n"
            "or call this again with submit=True."
        )
        return per_date

    from dal_monte_2022_analysis.runtime.hpc import (
        submit_sbatch_array_job,
        track_job_completion,
    )

    all_dates = list(per_date["date"])
    indices = sorted(all_dates.index(date) for date in incomplete["date"])
    job_id = submit_sbatch_array_job(
        Path(sbatch_path),
        array_spec=_compact_array_spec(indices),
        extra_args=extra_args,
        repo_root=repo_root,
    )
    if wait:
        track_job_completion(job_id, poll_secs=int(poll_secs))
        refreshed = summarize_build_status(build_status(settings))
        still_missing = refreshed.loc[~refreshed["complete"]]
        if len(still_missing):
            print(
                f"WARNING: {len(still_missing)} date(s) still incomplete after the array "
                f"finished: {list(still_missing['date'])}. Check hpc/ephys/logs/."
            )
        else:
            print("all dates built.")
        return refreshed
    return per_date


def _compact_array_spec(indices: Sequence[int]) -> str:
    """Collapse task indices into SLURM's comma/range array syntax."""
    values = sorted({int(value) for value in indices})
    if not values:
        return ""
    parts: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def run_summary_build(
    settings: PairSpikeCoordinationSettings,
    *,
    metric: str = "trial_shuffle_mean_effect_pm10ms",
) -> dict[str, pd.DataFrame]:
    """Build every summary table the notebook reads, and write them to disk.

    Reporting is **per region** for within-region pairs and **per region pair**
    for cross-region pairs, with the pooled scope-level tables kept alongside.
    Pooling first would let one region with many pairs carry a conclusion that
    does not hold in the others, so the resolved tables are the primary ones and
    the pooled ones are the summary.
    """
    cfg = load_config(settings.cfg_path)
    out_dir = build_analysis_output_dir(cfg, settings.output_subdir) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, _ = load_pair_coordination(
        settings.cfg_path,
        output_subdir=settings.output_subdir,
        output_filename=settings.output_filename,
    )
    selective = pairs.loc[pairs["both_selective"]]

    outputs = {
        "pair_inventory": build_pair_inventory(pairs),
        # Per region / region pair -- the primary tables.
        "coordination_summary_by_region": summarize_coordination(
            pairs, metric=metric, group_columns=("scope", "region_pair", "condition")
        ),
        "coordination_vs_null_by_region": test_against_null(
            pairs, metric=metric, group_columns=("scope", "region_pair", "condition")
        ),
        "condition_comparisons_by_region": compare_conditions(
            pairs, metric=metric, group_columns=("scope", "region_pair")
        ),
        "condition_comparisons_by_region_selective": compare_conditions(
            selective, metric=metric, group_columns=("scope", "region_pair")
        ),
        # Pooled across regions -- the summary.
        "coordination_summary": summarize_coordination(
            pairs, metric=metric, group_columns=("scope", "condition")
        ),
        "coordination_vs_null": test_against_null(pairs, metric=metric),
        "condition_comparisons": compare_conditions(pairs, metric=metric),
        "condition_comparisons_selective": compare_conditions(selective, metric=metric),
        "zero_lag_diagnostics": build_zero_lag_diagnostics(pairs),
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    for stem, kwargs in {
        "group_traces_by_scope": {"group_columns": ("scope", "condition")},
        "group_traces_by_region": {
            "group_columns": ("scope", "region_pair", "condition")
        },
        "group_traces_selective": {
            "group_columns": ("scope", "condition"),
            "selective_only": True,
        },
        "group_traces_by_region_selective": {
            "group_columns": ("scope", "region_pair", "condition"),
            "selective_only": True,
        },
    }.items():
        pd.to_pickle(
            build_group_traces(
                settings.cfg_path,
                output_subdir=settings.output_subdir,
                output_filename=settings.output_filename,
                **kwargs,
            ),
            out_dir / f"{stem}.pkl",
        )
    print(f"summary tables written to {out_dir}")
    return outputs
