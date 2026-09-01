"""Spatial decay of pairwise spike coordination across an electrode array.

Why this analysis exists
------------------------

Within-region pairs are recorded on the same electrode array; cross-region pairs
are not.  So the observation that within-region pairs are more coordinated than
cross-region pairs is confounded at the outset: a shared reference, a shared
amplifier or any common noise on an array would produce exactly that, with no
biology involved.  The comparison cannot be interpreted as it stands.

Electrode separation breaks the tie.  A shared reference contaminates every pair
on an array equally, so it predicts coordination **flat** with the distance
between the two electrodes.  Local circuitry -- shared afferents, local
connectivity -- predicts coordination that **decays** with distance.  The two
hypotheses make opposite predictions about the same measurement, on the same
pairs, with no new data required.

This module measures that decay, fits a length constant to it, and asks whether
the length constant depends on what the animal was looking at.

What the separation measure is, and is not
------------------------------------------

Channels are named ``SPKnn`` and are numbered in contiguous blocks per region,
so ``|n1 - n2|`` is used as the separation.  This is a **proxy** for physical
distance, not a calibrated one: it assumes channel numbering runs in spatial
order, which holds for a linear probe or a wired-in-order array but should be
checked against the actual array geometry before the length constants are
quoted in physical units.  A monotone decay is itself evidence that the
numbering tracks something spatial, since an arbitrary permutation of channel
labels would destroy it -- but that is an argument, not a calibration.

Pairs on the *same* channel are excluded from the decay and reported
separately.  They carry a negative zero-lag excess, because a spike sorter
cannot assign two spikes to different units in the same millisecond on one
channel, and that shadowing is a property of the sorter rather than of the
tissue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_pair_spike_coordination import (
    CONDITION_ORDER,
    DEFAULT_OUTPUT_SUBDIR,
    drop_zero_lag_artifact_dates,
    load_pair_coordination,
)

#: The two components of the null-corrected cross-correlation, measured
#: separately because they can decay differently.
PEAK_METRIC = "circular_shift_peak_pm2ms"
SHOULDER_METRIC = "circular_shift_shoulder_20to100ms"
#: Trial-count-matched counterpart of the peak, for the condition comparison.
PEAK_METRIC_MATCHED = PEAK_METRIC + "_matched"

METRIC_LABELS: dict[str, str] = {
    PEAK_METRIC: "Sharp peak (±2 ms)",
    SHOULDER_METRIC: "Broad shoulder (20–100 ms)",
    PEAK_METRIC_MATCHED: "Sharp peak (±2 ms), trial-matched",
}

#: Separation bins, in channels.  Edges are geometric rather than linear because
#: the decay is steepest at short range: linear bins would put almost every
#: informative pair in the first one.
SEPARATION_BINS: tuple[float, ...] = (0.5, 1.5, 3.5, 7.5, 15.5, np.inf)
SEPARATION_LABELS: tuple[str, ...] = ("1", "2–3", "4–7", "8–15", ">15")
#: Representative separation for each bin, used when fitting.
SEPARATION_CENTRES: tuple[float, ...] = (1.0, 2.5, 5.5, 11.5, 24.0)

REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")


@dataclass
class SpatialDecaySettings:
    """Configuration for the spatial-decay analysis."""

    cfg_path: str = "configs/dataset.yaml"
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR
    metric: str = PEAK_METRIC
    #: Bins with fewer pairs than this are dropped from fits and figures.
    min_pairs_per_bin: int = 150
    drop_artifact_dates: bool = True
    selective_only: bool = False
    n_bootstrap: int = 500
    random_seed: int = 0


def parse_channel_number(values: pd.Series) -> pd.Series:
    """Numeric part of a ``SPKnn`` channel label."""
    return pd.to_numeric(
        values.astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )


def load_pairs_with_separation(
    settings: SpatialDecaySettings,
) -> tuple[pd.DataFrame, list[str]]:
    """Load pair coordination and attach electrode separation.

    Returns the pair table with ``scope``, ``channel_separation`` and
    ``separation_bin`` columns, plus the artifact dates removed.
    """
    pairs, _ = load_pair_coordination(
        settings.cfg_path, output_subdir=settings.output_subdir
    )
    dropped: list[str] = []
    if settings.drop_artifact_dates:
        pairs, dropped = drop_zero_lag_artifact_dates(pairs)
    if settings.selective_only:
        pairs = pairs.loc[pairs["both_selective"]]

    pairs = pairs.copy()
    pairs["scope"] = np.where(pairs["same_region"], "within_region", "cross_region")
    first = parse_channel_number(pairs["spike_channel_1"])
    second = parse_channel_number(pairs["spike_channel_2"])
    pairs["channel_separation"] = (first - second).abs()
    pairs["same_channel"] = pairs["channel_separation"] == 0
    pairs["separation_bin"] = pd.cut(
        pairs["channel_separation"],
        bins=list(SEPARATION_BINS),
        labels=list(SEPARATION_LABELS),
    )
    return pairs, dropped


def build_separation_inventory(pairs: pd.DataFrame) -> pd.DataFrame:
    """How many pairs support each region x separation cell."""
    within = pairs.loc[pairs["scope"] == "within_region"]
    rows: list[dict] = []
    for region, group in within.groupby("region_1", observed=True):
        row: dict[str, object] = {
            "region": str(region),
            "n_pairs": int(len(group)),
            "n_same_channel": int(group["same_channel"].sum()),
            "max_separation": float(group["channel_separation"].max()),
        }
        counts = group.loc[~group["same_channel"], "separation_bin"].value_counts()
        for label in SEPARATION_LABELS:
            row[f"n_{label}"] = int(counts.get(label, 0))
        rows.append(row)
    result = pd.DataFrame(rows)
    order = {region: index for index, region in enumerate(REGION_ORDER)}
    return result.sort_values("region", key=lambda s: s.map(order)).reset_index(drop=True)


def build_decay_table(
    pairs: pd.DataFrame,
    settings: SpatialDecaySettings,
    *,
    metric: Optional[str] = None,
    by_condition: bool = False,
) -> pd.DataFrame:
    """Mean coordination per region and separation bin, with bootstrap intervals.

    Same-channel pairs are excluded: their negative zero-lag excess is sorter
    shadowing, not a distance-zero sample of the same decay.
    """
    metric = metric or settings.metric
    within = pairs.loc[
        (pairs["scope"] == "within_region")
        & (~pairs["same_channel"])
        & pairs["separation_bin"].notna()
    ]
    keys = ["region_1", "separation_bin"] + (["condition"] if by_condition else [])
    rng = np.random.default_rng(settings.random_seed)

    rows: list[dict] = []
    for group_keys, group in within.groupby(keys, observed=True):
        group_keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        values = group[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < settings.min_pairs_per_bin:
            continue
        draws = rng.choice(values, size=(settings.n_bootstrap, values.size), replace=True)
        means = draws.mean(axis=1)
        row = {
            "region": str(group_keys[0]),
            "separation_bin": str(group_keys[1]),
            "separation": float(
                SEPARATION_CENTRES[SEPARATION_LABELS.index(str(group_keys[1]))]
            ),
            "n_pairs": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "sem": float(values.std(ddof=1) / np.sqrt(values.size)),
            "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975)),
        }
        if by_condition:
            row["condition"] = str(group_keys[2])
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    order = {region: index for index, region in enumerate(REGION_ORDER)}
    sort_keys = ["region", "separation"] + (["condition"] if by_condition else [])
    return result.sort_values(
        sort_keys, key=lambda s: s.map(order) if s.name == "region" else s
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Decay fitting
# ---------------------------------------------------------------------------


def exponential_decay(
    separation: np.ndarray, amplitude: float, length_constant: float, offset: float
) -> np.ndarray:
    """``amplitude * exp(-separation / length_constant) + offset``.

    The offset matters: it is the level coordination settles to at long range,
    and a shared-reference contribution would live there rather than in the
    amplitude.  Fitting it rather than assuming zero keeps the two separable.
    """
    return amplitude * np.exp(-separation / length_constant) + offset


def fit_decay(
    table: pd.DataFrame,
    *,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Fit an exponential decay to one region's separation profile.

    Returns the amplitude, length constant and long-range offset, each with a
    bootstrap interval taken by resampling the bins with replacement, plus the
    fraction of variance explained.  Fewer than four usable bins leaves the
    length constant unidentifiable and returns NaNs rather than a number that
    looks like a measurement.
    """
    from scipy.optimize import curve_fit

    usable = table.dropna(subset=["separation", "mean"])
    empty = {
        key: np.nan
        for key in (
            "amplitude", "amplitude_low", "amplitude_high",
            "length_constant", "length_constant_low", "length_constant_high",
            "offset", "r_squared", "n_bins",
        )
    }
    if len(usable) < 4:
        empty["n_bins"] = float(len(usable))
        return empty

    x = usable["separation"].to_numpy(dtype=float)
    y = usable["mean"].to_numpy(dtype=float)
    guess = (float(y.max() - y.min()), 5.0, float(y.min()))
    bounds = ([0.0, 0.25, -np.inf], [np.inf, 200.0, np.inf])

    def _fit(values: np.ndarray) -> Optional[tuple[float, float, float]]:
        try:
            popt, _ = curve_fit(
                exponential_decay, x, values, p0=guess, bounds=bounds, maxfev=20000
            )
            return tuple(float(v) for v in popt)
        except Exception:
            return None

    fitted = _fit(y)
    if fitted is None:
        empty["n_bins"] = float(len(usable))
        return empty
    amplitude, length_constant, offset = fitted

    residual = y - exponential_decay(x, amplitude, length_constant, offset)
    total = y - y.mean()
    r_squared = 1.0 - float(residual @ residual) / float(total @ total or np.nan)

    rng = np.random.default_rng(seed)
    sems = usable["sem"].to_numpy(dtype=float)
    amps: list[float] = []
    lengths: list[float] = []
    for _ in range(int(n_bootstrap)):
        jittered = y + rng.normal(0.0, np.where(np.isfinite(sems), sems, 0.0))
        drawn = _fit(jittered)
        if drawn is not None:
            amps.append(drawn[0])
            lengths.append(drawn[1])
    quant = lambda v, q: float(np.quantile(v, q)) if v else np.nan

    return {
        "amplitude": amplitude,
        "amplitude_low": quant(amps, 0.025),
        "amplitude_high": quant(amps, 0.975),
        "length_constant": length_constant,
        "length_constant_low": quant(lengths, 0.025),
        "length_constant_high": quant(lengths, 0.975),
        "offset": offset,
        "r_squared": r_squared,
        "n_bins": float(len(usable)),
    }


def fit_decay_by_group(
    table: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("region",),
    n_bootstrap: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit one decay per group and return the parameters as a table."""
    rows: list[dict] = []
    for keys, group in table.groupby(list(group_columns), observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, [str(k) for k in keys]))
        row.update(fit_decay(group, n_bootstrap=n_bootstrap, seed=seed))
        row["total_pairs"] = int(group["n_pairs"].sum())
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty or "region" not in result.columns:
        return result
    order = {region: index for index, region in enumerate(REGION_ORDER)}
    return result.sort_values(
        ["region"] + [c for c in group_columns if c != "region"],
        key=lambda s: s.map(order) if s.name == "region" else s,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Is the decay fixation-specific?
# ---------------------------------------------------------------------------


def test_decay_flatness(table: pd.DataFrame) -> pd.DataFrame:
    """Is coordination flat with separation, as a shared reference would predict?

    Spearman correlation between separation and the per-bin mean.  Flatness is
    the artifact hypothesis; a strong negative rank correlation rejects it.
    """
    from scipy.stats import spearmanr

    rows: list[dict] = []
    for region, group in table.groupby("region", observed=True):
        if len(group) < 3:
            continue
        rho, p_value = spearmanr(group["separation"], group["mean"])
        near = group.loc[group["separation"] <= 1.5, "mean"]
        far = group.loc[group["separation"] >= 15.0, "mean"]
        rows.append(
            {
                "region": str(region),
                "n_bins": int(len(group)),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "nearest_bin_mean": float(near.iloc[0]) if len(near) else np.nan,
                "farthest_bin_mean": float(far.iloc[0]) if len(far) else np.nan,
                "fold_decay": (
                    float(near.iloc[0] / far.iloc[0])
                    if len(near) and len(far) and far.iloc[0] != 0
                    else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    order = {region: index for index, region in enumerate(REGION_ORDER)}
    return result.sort_values("region", key=lambda s: s.map(order)).reset_index(drop=True)


def test_condition_by_separation(
    pairs: pd.DataFrame,
    *,
    metric: str = PEAK_METRIC_MATCHED,
    conditions: tuple[str, str] = ("face_interactive", "object"),
) -> pd.DataFrame:
    """Does the condition difference itself depend on electrode separation?

    The decay could be fixation-specific in two different ways, and they need
    separating.  Either the *level* of coordination differs by condition at
    every distance, which shifts the curves vertically, or the *reach* differs,
    which changes their shape.  This tests the second: the within-pair condition
    difference is computed inside each separation bin, so a difference that
    grows or shrinks with distance shows up as a trend across bins.

    Uses the trial-count-matched metric by default -- interactive-face fixations
    outnumber the others roughly six to one, and an unmatched comparison mostly
    reflects that.
    """
    from scipy.stats import wilcoxon

    within = pairs.loc[
        (pairs["scope"] == "within_region")
        & (~pairs["same_channel"])
        & pairs["separation_bin"].notna()
    ]
    wide = within.pivot_table(
        index=["pair_key", "region_1", "separation_bin"],
        columns="condition",
        values=metric,
        observed=True,
    ).reset_index()
    first, second = conditions
    if first not in wide.columns or second not in wide.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for keys, group in wide.groupby(["region_1", "separation_bin"], observed=True):
        paired = group.loc[:, [first, second]].dropna()
        if len(paired) < 50:
            continue
        differences = (paired[first] - paired[second]).to_numpy(dtype=float)
        positive = float(np.sum(differences > 0))
        negative = float(np.sum(differences < 0))
        total = positive + negative
        row = {
            "region": str(keys[0]),
            "separation_bin": str(keys[1]),
            "separation": float(
                SEPARATION_CENTRES[SEPARATION_LABELS.index(str(keys[1]))]
            ),
            "n_pairs": int(len(paired)),
            "mean_difference": float(np.mean(differences)),
            "effect_size_rank_biserial": (positive - negative) / total if total else np.nan,
        }
        if np.any(differences != 0):
            row["p_value"] = float(wilcoxon(differences, alternative="two-sided").pvalue)
        else:
            row["p_value"] = np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    from statsmodels.stats.multitest import multipletests

    testable = result["p_value"].notna()
    result["p_value_corrected"] = np.nan
    result["significant"] = False
    if testable.any():
        reject, corrected, _, _ = multipletests(
            result.loc[testable, "p_value"].to_numpy(dtype=float), method="fdr_bh"
        )
        result.loc[testable, "p_value_corrected"] = corrected
        result.loc[testable, "significant"] = reject
    order = {region: index for index, region in enumerate(REGION_ORDER)}
    return result.sort_values(
        ["region", "separation"], key=lambda s: s.map(order) if s.name == "region" else s
    ).reset_index(drop=True)


def build_reference_levels(pairs: pd.DataFrame, *, metric: str = PEAK_METRIC) -> pd.DataFrame:
    """Levels the decay should be read against.

    ``same_channel`` is the sorter-shadowing floor, ``cross_region`` is what
    pairs on different arrays show, and the farthest within-region bin is where
    the decay lands.  If the far within-region level fell to the cross-region
    level the array offset would be zero; that it stays somewhat above is worth
    seeing next to the curves rather than inferred from them.
    """
    rows: list[dict] = []
    same = pairs.loc[pairs["same_channel"] & (pairs["scope"] == "within_region"), metric]
    cross = pairs.loc[pairs["scope"] == "cross_region", metric]
    far = pairs.loc[
        (pairs["scope"] == "within_region")
        & (pairs["separation_bin"].astype(str) == SEPARATION_LABELS[-1]),
        metric,
    ]
    for label, values in (
        ("same channel", same),
        (f">{int(SEPARATION_BINS[-2] - 0.5)} ch apart", far),
        ("cross region", cross),
    ):
        values = values.dropna()
        rows.append(
            {
                "level": label,
                "n_pairs": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)
