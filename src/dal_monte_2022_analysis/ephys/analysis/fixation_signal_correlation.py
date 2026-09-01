"""Cross-correlation of condition-averaged firing-rate timelines between unit pairs.

This asks a different question from the pairwise spike-coordination analysis,
and the difference is the whole point.

That analysis cross-correlates two units' **per-fixation spike trains** and asks
whether they fire together on the same fixation more than chance -- *noise*
correlation, trial-by-trial covariation, measured against a null that scrambles
which fixation goes with which.

This one cross-correlates two units' **condition-averaged rate timelines** and
asks whether their mean responses to a fixation have the same shape, and at what
lag -- *signal* correlation, shared tuning.  Averaging over fixations removes the
trial-by-trial covariation entirely, so nothing measured here is noise
correlation.  Two units can have identical mean profiles and be independent
trial to trial, or the reverse.

Why the null has to be a different unit
---------------------------------------

Every unit here is fixation-locked, so every mean timeline has structure around
fixation onset.  Correlating any two of them therefore gives a large value
before any shared tuning is involved, and a null that only scrambles time --
a circular shift, say -- would confirm that trivial structure rather than test
the interesting claim.

The null used instead is a **cross-session pairing**: unit A is correlated
against a unit from the same region recorded on a *different session*.  That
partner is fixation-locked in the same way, has a real unit's response shape, and
has been through the same pipeline, but shares no session, no array and no
behaviour with unit A.  An excess over it means the two units' responses
resemble each other more than two arbitrary units of that region do.

Restricting to selective units
------------------------------

A unit with no reliable fixation response has a mean timeline that is mostly
estimation noise, and correlating noise with noise contributes only variance.
The default here is therefore the FDR-corrected selective set, which is also
what makes the question well posed: "do units that respond to fixation condition
share response shape" presumes units that respond.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_population_pc_subspace import (
    CONDITION_ROW_SPECS,
    load_pair_selective_units,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_bridge import (
    load_combined_fixation_psth,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir

CONDITION_ORDER: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")

DEFAULT_OUTPUT_SUBDIR = "ephys/psth/fixation_signal_correlation"

#: Unit identity, matching every other analysis in the repository.
UNIT_KEY_COLUMNS: tuple[str, ...] = (
    "date",
    "unit_uuid",
    "region",
    "spike_channel",
    "recorded_agent",
)


@dataclass
class SignalCorrelationSettings:
    """Configuration for the signal-correlation analysis."""

    cfg_path: str = "configs/dataset.yaml"
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR
    conditions: tuple[str, ...] = CONDITION_ORDER
    #: Restrict to units the single-unit analysis calls selective.  A unit with
    #: no reliable fixation response contributes an estimate of noise.
    selective_only: bool = True
    #: Lags retained, in ms.  The timelines are 100 bins of 10 ms, so lags run
    #: to +/-990 ms in principle, but the overlap taper makes anything past a
    #: few hundred ms uninterpretable.
    max_lag_ms: float = 250.0
    #: Draws of the cross-session null per pair.
    n_null_draws: int = 20
    #: Only pairs recorded in the same session, so this is comparable with the
    #: spike-coordination analysis pair for pair.
    simultaneous_only: bool = True
    min_pairs_per_group: int = 100
    #: Lags searched for the peak of the null-corrected correlation.
    #:
    #: Kept narrow on purpose.  The correlation at each lag is estimated from
    #: only the overlapping bins -- 100 at zero lag, fewer as the lag grows --
    #: so its variance rises with ``|lag|`` and a maximum taken over a wide
    #: window mostly finds the noisiest lag rather than the best alignment.  At
    #: +/-150 ms the per-pair peak sat at 0.35 for every condition with a median
    #: lag near the search edge, which is what that failure looks like.
    #:
    #: The bias does not vanish at 100 ms, it is just smaller.  It is also
    #: identical across conditions -- same lag count, same pair count -- so
    #: comparisons between conditions remain valid even though the absolute
    #: level is inflated.
    peak_search_ms: float = 100.0
    #: Band averaged on each side of zero for the lead/lag comparison.  It
    #: starts away from zero so the two bands do not both contain the central
    #: peak, which would make them trivially similar.
    lag_band_ms: tuple[float, float] = (20.0, 200.0)
    random_seed: int = 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _normalize_key_value(column: str, value: object) -> str:
    """Normalise one unit-key field for matching across files.

    ``read_csv`` turns a date like ``01312018`` into the integer 1312018, so a
    naive string comparison against the pickle silently drops every date that
    starts with a zero -- which is most of them.  Zero-padding the date and
    lower-casing the region makes the two representations comparable.
    """
    text = str(value).strip()
    if column == "date":
        return text.zfill(8)
    if column == "region":
        return text.lower()
    return text


def load_condition_timelines(
    settings: SignalCorrelationSettings,
) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per unit, one column per condition, each holding a rate timeline.

    Reads the same combined export the population PCA and the mRNN use, so the
    unit set and the timeline are identical to theirs by construction rather
    than by a parallel derivation that could drift.
    """
    loaded = load_combined_fixation_psth(settings.cfg_path)
    frame = loaded.dataframe
    timeline = np.asarray(loaded.timeline_s_rel, dtype=float).reshape(-1)

    for column in UNIT_KEY_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)

    rows: dict[tuple, dict] = {}
    for condition in settings.conditions:
        spec = CONDITION_ROW_SPECS.get(str(condition))
        if spec is None:
            continue
        partition, category, state = spec
        mask = (frame["average_partition"].astype(str) == partition) & (
            frame["fixation_category"].astype(str) == category
        )
        mask &= frame["interactive_state"].isna() if state is None else (
            frame["interactive_state"].astype(str) == state
        )
        for row in frame.loc[mask].itertuples(index=False):
            key = tuple(str(getattr(row, column)) for column in UNIT_KEY_COLUMNS)
            entry = rows.setdefault(key, dict(zip(UNIT_KEY_COLUMNS, key)))
            entry[condition] = np.asarray(row.psth_mean, dtype=float)
            entry[f"{condition}_n_trials"] = int(row.n_trials)
            if hasattr(row, "psth_sem"):
                entry[f"{condition}_sem"] = np.asarray(row.psth_sem, dtype=float)

    units = pd.DataFrame(list(rows.values()))
    # A unit must have all conditions, or it cannot enter a paired comparison.
    units = units.dropna(subset=list(settings.conditions)).reset_index(drop=True)
    units["region"] = units["region"].astype(str).str.lower()

    if settings.selective_only:
        selective, _ = load_pair_selective_units(settings.cfg_path, use_corrected=True)
        keys = {
            tuple(_normalize_key_value(column, value) for column, value in zip(UNIT_KEY_COLUMNS, row))
            for row in selective.loc[:, list(UNIT_KEY_COLUMNS)].values
        }
        keep = [
            tuple(
                _normalize_key_value(column, row[column]) for column in UNIT_KEY_COLUMNS
            )
            in keys
            for _, row in units.iterrows()
        ]
        units = units.loc[keep].reset_index(drop=True)
    return units, timeline


def build_unit_inventory(units: pd.DataFrame) -> pd.DataFrame:
    """Units per region entering the analysis."""
    counts = (
        units.groupby("region", observed=True)
        .agg(n_units=("unit_uuid", "size"), n_dates=("date", "nunique"))
        .reset_index()
    )
    order = {region: index for index, region in enumerate(REGION_ORDER)}
    return counts.sort_values("region", key=lambda s: s.map(order)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trial count, and why it has to be controlled
# ---------------------------------------------------------------------------
#
# A mean timeline estimated from 168 fixations is noisier than one estimated
# from 1169, and a correlation between two noisy estimates is attenuated towards
# zero.  Interactive-face fixations outnumber the others roughly six to one, so
# interactive-face timelines are the cleanest of the three and correlate best
# with anything -- including with each other.  A raw comparison of signal
# correlation across conditions therefore measures trial count as much as shared
# tuning, and the difference is large: uncontrolled, interactive face looks
# twice as high as the other two conditions.
#
# The cross-session null does *not* absorb this.  The null partner shares no
# tuning with unit A, so its correlation sits near zero whatever the trial
# count; reliability inflates the observed correlation between genuinely similar
# units without lifting the null by anything comparable.
#
# Spearman's correction for attenuation is the textbook remedy and is not
# usable here.  It needs the reliability of each timeline, estimated as the
# share of its variance that is not measurement noise -- but these means are
# Gaussian-smoothed before averaging, which suppresses the across-bin variance
# of the mean while leaving the per-bin SEM reflecting unsmoothed trial-to-trial
# spread.  The two are not on the same footing and the estimate comes out
# negative for most units, which would make the correction divide by the square
# root of a negative number.  :func:`estimate_timeline_reliability` computes it
# anyway, as a diagnostic that reports the problem rather than hiding it.
#
# What works instead is stratification.  The ratio of interactive-face to object
# trial counts varies from about 1.3 to 11 across pairs, so the condition
# comparison can be repeated inside strata of that ratio.  If the effect is
# shared tuning it should be flat across strata; if it is trial count it should
# grow with the ratio and vanish where the ratio approaches one.


def estimate_timeline_reliability(
    timeline: np.ndarray, sem: Optional[np.ndarray]
) -> float:
    """Share of a mean timeline's variance that is not estimation noise.

    Reported as a diagnostic, **not** used to correct anything.  These timelines
    are smoothed before averaging while the SEMs are not correspondingly
    reduced, so this comes out negative for most units -- which is the finding,
    not a bug to be clipped away.  Use :func:`stratify_by_trial_ratio` to
    control the trial-count confound instead.
    """
    if sem is None:
        return np.nan
    values = np.asarray(timeline, dtype=float)
    errors = np.asarray(sem, dtype=float)
    if values.size == 0 or errors.size != values.size:
        return np.nan
    total = float(np.var(values, ddof=1))
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float((total - float(np.mean(np.square(errors)))) / total)


def stratify_by_trial_ratio(
    pairs: pd.DataFrame,
    settings: SignalCorrelationSettings,
    *,
    metric: str = "zero_lag_excess",
    reference: str = "object",
    target: str = "face_interactive",
    n_strata: int = 4,
) -> pd.DataFrame:
    """Repeat the condition comparison inside strata of the trial-count ratio.

    This is the control that decides whether a condition difference in signal
    correlation is shared tuning or estimation noise.  Shared tuning predicts a
    difference that is flat across strata; a trial-count artifact predicts one
    that grows with the ratio and disappears as it approaches one.
    """
    frame = pairs.copy()
    ratio = frame[f"{target}_n_trials_1"] / frame[f"{reference}_n_trials_1"]
    frame = frame.loc[np.isfinite(ratio)].copy()
    frame["trial_ratio"] = ratio.loc[frame.index]
    try:
        frame["ratio_stratum"] = pd.qcut(frame["trial_ratio"], int(n_strata), duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rows: list[dict] = []
    for stratum, group in frame.groupby("ratio_stratum", observed=True):
        if len(group) < settings.min_pairs_per_group:
            continue
        row = {
            "ratio_stratum": str(stratum),
            "n_pairs": int(len(group)),
            "median_trial_ratio": float(group["trial_ratio"].median()),
        }
        for condition in settings.conditions:
            values = group[f"{condition}_{metric}"].to_numpy(dtype=float)
            row[condition] = float(np.nanmean(values))
        row[f"{target}_minus_{reference}"] = row[target] - row[reference]
        differences = (
            group[f"{target}_{metric}"].to_numpy(dtype=float)
            - group[f"{reference}_{metric}"].to_numpy(dtype=float)
        )
        differences = differences[np.isfinite(differences)]
        if differences.size >= 10 and np.any(differences != 0):
            from scipy.stats import wilcoxon

            row["p_value"] = float(wilcoxon(differences, alternative="two-sided").pvalue)
            positive = float(np.sum(differences > 0))
            negative = float(np.sum(differences < 0))
            total = positive + negative
            row["effect_size_rank_biserial"] = (
                (positive - negative) / total if total else np.nan
            )
        else:
            row["p_value"] = np.nan
            row["effect_size_rank_biserial"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def normalized_cross_correlation(
    first: np.ndarray, second: np.ndarray, *, max_lag: int
) -> np.ndarray:
    """Pearson correlation between two timelines at each lag in ``+/-max_lag``.

    Each trace is centred and scaled by its own standard deviation before
    correlating, so the result is bounded in ``[-1, 1]`` and is a correlation
    coefficient rather than a dot product that grows with firing rate.  Only the
    overlapping samples enter at each lag, and they are re-centred within that
    overlap, which is what keeps the value a genuine correlation rather than a
    tapered one.

    A positive lag means ``first`` follows ``second``.
    """
    first = np.asarray(first, dtype=float).reshape(-1)
    second = np.asarray(second, dtype=float).reshape(-1)
    n = first.size
    lags = np.arange(-int(max_lag), int(max_lag) + 1)
    out = np.full(lags.size, np.nan)
    for index, lag in enumerate(lags):
        if lag >= 0:
            a, b = first[lag:], second[: n - lag]
        else:
            a, b = first[: n + lag], second[-lag:]
        if a.size < 8:
            continue
        a = a - a.mean()
        b = b - b.mean()
        denominator = float(np.sqrt(a @ a) * np.sqrt(b @ b))
        if denominator > 0:
            out[index] = float(a @ b) / denominator
    return out


def _iter_pairs(units: pd.DataFrame, settings: SignalCorrelationSettings):
    """Yield (index_a, index_b, scope, region_pair) for every pair to measure."""
    if settings.simultaneous_only:
        groups = units.groupby("date", observed=True).groups
    else:
        groups = {"all": units.index}
    for _, index in groups.items():
        index = list(index)
        for first, second in combinations(index, 2):
            region_a = units.at[first, "region"]
            region_b = units.at[second, "region"]
            same = region_a == region_b
            yield (
                first,
                second,
                "within_region" if same else "cross_region",
                region_a if same else "-".join(sorted((region_a, region_b))),
            )


def build_pair_correlations(
    units: pd.DataFrame,
    timeline: np.ndarray,
    settings: SignalCorrelationSettings,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Signal cross-correlation and its cross-session null, for every pair.

    The null partner is drawn from the same region but a *different* date, so it
    is fixation-locked and real but shares nothing with the pair.  Drawing it
    per pair rather than once keeps the null from being dominated by whichever
    unit happened to be chosen.
    """
    bin_size_ms = float(np.median(np.diff(timeline)) * 1000.0)
    max_lag = int(round(float(settings.max_lag_ms) / bin_size_ms))
    lags_ms = np.arange(-max_lag, max_lag + 1) * bin_size_ms
    rng = np.random.default_rng(settings.random_seed)

    by_region_date: dict[tuple[str, str], list[int]] = {}
    for index, row in units.iterrows():
        by_region_date.setdefault((row["region"], row["date"]), []).append(index)
    by_region: dict[str, list[int]] = {}
    for (region, _), indices in by_region_date.items():
        by_region.setdefault(region, []).extend(indices)

    records: list[dict] = []
    for first, second, scope, region_pair in _iter_pairs(units, settings):
        row_a, row_b = units.loc[first], units.loc[second]
        record = {
            "date": row_a["date"],
            "scope": scope,
            "region_pair": region_pair,
            "region_1": row_a["region"],
            "region_2": row_b["region"],
            "unit_uuid_1": row_a["unit_uuid"],
            "unit_uuid_2": row_b["unit_uuid"],
            "pair_key": f"{row_a['date']}|{row_a['unit_uuid']}|{row_b['unit_uuid']}",
        }
        # Null partners for unit B: same region, different date.
        candidates = [
            index
            for index in by_region.get(row_b["region"], [])
            if units.at[index, "date"] != row_a["date"]
        ]
        for condition in settings.conditions:
            observed = normalized_cross_correlation(
                row_a[condition], row_b[condition], max_lag=max_lag
            )
            record[f"{condition}_reliability_1"] = estimate_timeline_reliability(
                row_a[condition], row_a.get(f"{condition}_sem")
            )
            record[f"{condition}_reliability_2"] = estimate_timeline_reliability(
                row_b[condition], row_b.get(f"{condition}_sem")
            )
            record[f"{condition}_n_trials_1"] = row_a.get(f"{condition}_n_trials", np.nan)
            record[f"{condition}_n_trials_2"] = row_b.get(f"{condition}_n_trials", np.nan)
            record[f"{condition}_observed"] = observed.astype(np.float32)
            if candidates:
                draws = rng.choice(
                    candidates, size=min(settings.n_null_draws, len(candidates)),
                    replace=len(candidates) < settings.n_null_draws,
                )
                null = np.nanmean(
                    [
                        normalized_cross_correlation(
                            row_a[condition], units.at[int(other), condition], max_lag=max_lag
                        )
                        for other in draws
                    ],
                    axis=0,
                )
            else:
                null = np.full(observed.shape, np.nan)
            record[f"{condition}_null"] = null.astype(np.float32)
            excess = observed - null
            record[f"{condition}_excess"] = excess.astype(np.float32)
            centre = int(np.argmin(np.abs(lags_ms)))
            record[f"{condition}_zero_lag"] = float(observed[centre])
            record[f"{condition}_zero_lag_excess"] = float(excess[centre])
            # Lag-band measures.  Zero lag is one bin of fifty and a poor
            # summary of a correlation that can peak anywhere: two units whose
            # responses have the same shape but different latencies correlate
            # strongly at a non-zero lag and weakly at zero.  The peak says how
            # similar the shapes are at their best alignment, and the two
            # flanking bands say whether that alignment is symmetric.
            search = np.abs(lags_ms) <= float(settings.peak_search_ms)
            if np.isfinite(excess[search]).any():
                local = np.where(search, excess, -np.inf)
                peak = int(np.nanargmax(local))
                record[f"{condition}_peak_excess"] = float(excess[peak])
                record[f"{condition}_peak_lag_ms"] = float(lags_ms[peak])
                record[f"{condition}_peak_abs_lag_ms"] = float(abs(lags_ms[peak]))
            else:
                record[f"{condition}_peak_excess"] = np.nan
                record[f"{condition}_peak_lag_ms"] = np.nan
                record[f"{condition}_peak_abs_lag_ms"] = np.nan

            low, high = settings.lag_band_ms
            positive = (lags_ms >= low) & (lags_ms <= high)
            negative = (lags_ms <= -low) & (lags_ms >= -high)
            record[f"{condition}_positive_lag_excess"] = (
                float(np.nanmean(excess[positive])) if positive.any() else np.nan
            )
            record[f"{condition}_negative_lag_excess"] = (
                float(np.nanmean(excess[negative])) if negative.any() else np.nan
            )
            record[f"{condition}_lag_asymmetry"] = (
                record[f"{condition}_positive_lag_excess"]
                - record[f"{condition}_negative_lag_excess"]
            )
        records.append(record)

    return pd.DataFrame.from_records(records), lags_ms


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def build_group_traces(
    pairs: pd.DataFrame,
    settings: SignalCorrelationSettings,
    *,
    group_columns: Sequence[str] = ("scope", "region_pair", "condition"),
) -> pd.DataFrame:
    """Mean observed, null and excess traces per region and condition."""
    rows: list[dict] = []
    keys = [c for c in group_columns if c != "condition"]
    for group_keys, group in pairs.groupby(keys, observed=True):
        group_keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        if len(group) < settings.min_pairs_per_group:
            continue
        for condition in settings.conditions:
            row = dict(zip(keys, [str(k) for k in group_keys]))
            row["condition"] = condition
            row["n_pairs"] = int(len(group))
            for channel in ("observed", "null", "excess"):
                stacked = np.vstack(
                    [np.asarray(v, dtype=float) for v in group[f"{condition}_{channel}"]]
                )
                row[f"{channel}_mean"] = np.nanmean(stacked, axis=0).astype(np.float32)
                row[f"{channel}_sem"] = (
                    np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0])
                ).astype(np.float32)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_signal_correlation(
    pairs: pd.DataFrame,
    settings: SignalCorrelationSettings,
    *,
    metric: str = "zero_lag_excess",
) -> pd.DataFrame:
    """Per region and condition summary of one scalar measure."""
    rows: list[dict] = []
    for keys, group in pairs.groupby(["scope", "region_pair"], observed=True):
        if len(group) < settings.min_pairs_per_group:
            continue
        for condition in settings.conditions:
            values = group[f"{condition}_{metric}"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append(
                {
                    "scope": str(keys[0]),
                    "region_pair": str(keys[1]),
                    "condition": condition,
                    "n_pairs": int(values.size),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "sem": float(values.std(ddof=1) / np.sqrt(values.size)),
                    "frac_positive": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(rows)


def compare_conditions(
    pairs: pd.DataFrame,
    settings: SignalCorrelationSettings,
    *,
    metric: str = "zero_lag_excess",
    group_columns: Sequence[str] = ("scope", "region_pair"),
) -> pd.DataFrame:
    """Within-pair condition contrasts on a scalar measure.

    Paired by construction: the same two units contribute all three conditions,
    so unit identity, firing rate and recording quality cannot explain a
    difference.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    rows: list[dict] = []
    for keys, group in pairs.groupby(list(group_columns), observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if len(group) < settings.min_pairs_per_group:
            continue
        for first, second in combinations(settings.conditions, 2):
            a = group[f"{first}_{metric}"].to_numpy(dtype=float)
            b = group[f"{second}_{metric}"].to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b)
            differences = (a - b)[finite]
            row = dict(zip(group_columns, [str(k) for k in keys]))
            row.update(
                {
                    "condition_a": first,
                    "condition_b": second,
                    "n_pairs": int(differences.size),
                    "mean_a": float(a[finite].mean()) if finite.any() else np.nan,
                    "mean_b": float(b[finite].mean()) if finite.any() else np.nan,
                    "mean_difference": float(differences.mean()) if differences.size else np.nan,
                }
            )
            if differences.size >= 10 and np.any(differences != 0):
                row["p_value"] = float(wilcoxon(differences, alternative="two-sided").pvalue)
                positive = float(np.sum(differences > 0))
                negative = float(np.sum(differences < 0))
                total = positive + negative
                row["effect_size_rank_biserial"] = (
                    (positive - negative) / total if total else np.nan
                )
            else:
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
            result.loc[testable, "p_value"].to_numpy(dtype=float), method="fdr_bh"
        )
        result.loc[testable, "p_value_corrected"] = corrected
        result.loc[testable, "significant"] = reject
    return result.reset_index(drop=True)


def join_with_noise_correlation(
    pairs: pd.DataFrame,
    settings: SignalCorrelationSettings,
    *,
    noise_metric: str = "circular_shift_peak_pm2ms",
    signal_metric: str = "peak_excess",
) -> pd.DataFrame:
    """Match each pair to its spike-coordination measurement, where one exists.

    Signal and noise correlation are different quantities and need not track
    each other: two units can share a response profile and be independent trial
    to trial, or covary trial to trial without resembling each other on average.
    Whether they do track each other in this data is a question the two analyses
    can only answer together.
    """
    from dal_monte_2022_analysis.ephys.analysis.fixation_pair_spike_coordination import (
        drop_zero_lag_artifact_dates,
        load_pair_coordination,
    )

    noise, _ = load_pair_coordination(settings.cfg_path)
    noise, _ = drop_zero_lag_artifact_dates(noise)
    noise = noise.copy()
    for column in ("date", "unit_uuid_1", "unit_uuid_2", "condition"):
        noise[column] = noise[column].astype(str)
    noise["match_key"] = (
        noise["date"] + "|" + noise["unit_uuid_1"] + "|" + noise["unit_uuid_2"]
    )
    lookup = noise.set_index(["match_key", "condition"])[noise_metric]

    rows: list[dict] = []
    for _, row in pairs.iterrows():
        for condition in settings.conditions:
            key = (str(row["pair_key"]), condition)
            if key not in lookup.index:
                continue
            value = lookup.loc[key]
            rows.append(
                {
                    "pair_key": row["pair_key"],
                    "scope": row["scope"],
                    "region_pair": row["region_pair"],
                    "condition": condition,
                    "signal": float(row[f"{condition}_{signal_metric}"]),
                    "noise": float(value if np.isscalar(value) else np.asarray(value).ravel()[0]),
                }
            )
    return pd.DataFrame(rows)


def correlate_signal_with_noise(joined: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between signal and noise correlation, per group."""
    from scipy.stats import spearmanr

    rows: list[dict] = []
    for keys, group in joined.groupby(["scope", "region_pair", "condition"], observed=True):
        finite = group.dropna(subset=["signal", "noise"])
        if len(finite) < 50:
            continue
        rho, p_value = spearmanr(finite["signal"], finite["noise"])
        rows.append(
            {
                "scope": str(keys[0]),
                "region_pair": str(keys[1]),
                "condition": str(keys[2]),
                "n_pairs": int(len(finite)),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


#: Lag-band measures reported instead of the zero-lag value alone.
LAG_MEASURES: tuple[str, ...] = (
    "peak_excess",
    "positive_lag_excess",
    "negative_lag_excess",
    "peak_abs_lag_ms",
)


def summarize_lag_measures(
    pairs: pd.DataFrame,
    settings: SignalCorrelationSettings,
    *,
    measures: Sequence[str] = LAG_MEASURES,
    scope: str = "within_region",
) -> pd.DataFrame:
    """Long-format summary of the lag-band measures, one row per group.

    Zero lag is one bin of fifty and a poor summary of a correlation that can
    peak anywhere: two units whose responses share a shape but differ in latency
    correlate strongly at a non-zero lag and weakly at zero.  These summarise
    the peak and the two flanking bands instead.
    """
    subset = pairs.loc[pairs["scope"].astype(str) == scope]
    rows: list[dict] = []
    for region_pair, group in subset.groupby("region_pair", observed=True):
        if len(group) < settings.min_pairs_per_group:
            continue
        for condition in settings.conditions:
            for measure in measures:
                column = f"{condition}_{measure}"
                if column not in group.columns:
                    continue
                values = group[column].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "region_pair": str(region_pair),
                        "condition": condition,
                        "measure": measure,
                        "n_pairs": int(values.size),
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "sem": float(values.std(ddof=1) / np.sqrt(values.size)),
                    }
                )
    return pd.DataFrame(rows)


def compare_lag_measures(
    pairs: pd.DataFrame,
    settings: SignalCorrelationSettings,
    *,
    measures: Sequence[str] = LAG_MEASURES,
    scope: str = "within_region",
) -> pd.DataFrame:
    """Condition contrasts on each lag-band measure, paired within pair."""
    frames: list[pd.DataFrame] = []
    subset = pairs.loc[pairs["scope"].astype(str) == scope]
    for measure in measures:
        result = compare_conditions(
            subset, settings, metric=measure, group_columns=("region_pair",)
        )
        if result.empty:
            continue
        frames.append(result.assign(measure=measure))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
