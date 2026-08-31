"""Population PC subspace geometry for fixation-conditioned average PSTHs.

The population PCA builder in :mod:`fixation_population_pca` answers "what do
the low-dimensional trajectories look like".  This module answers the follow-up
question the trajectories raise: *how different are the subspaces the three
fixation conditions live in, and is the difference bigger than chance?*

Three things separate it from the existing builder:

- It reads the canonical 10 ms combined average export (the same input the mRNN
  targets and the selectivity analysis use), so unit counts, trial counts and
  the retained-dimension count agree across analyses.
- It fits to full rank rather than a fixed 50-component cap, so the number of
  PCs needed to reach a variance threshold is a measured quantity rather than a
  quantity clipped by configuration.
- It carries an explicit verification suite.  Concatenating conditions along
  time and recovering per-condition projections afterwards is exactly the kind
  of step that fails silently, so every identity it relies on is asserted
  numerically rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.stats.correction import apply_adjusted_pvalues
from dal_monte_2022_analysis.core.stats.tests import safe_paired_ttest
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_bridge import (
    build_mrnn_training_dataframe,
    load_combined_fixation_psth,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")
CONDITION_ORDER: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
#: Column names ``build_mrnn_training_dataframe`` emits, per analysis condition.
CONDITION_TO_TRAINING_COLUMN: dict[str, str] = {
    "face_interactive": "high_interactivity_face",
    "face_non_interactive": "low_interactivity_face",
    "object": "object",
}
#: How each analysis condition is selected from the combined export, as
#: ``(average_partition, fixation_category, interactive_state)``.  Mirrors the
#: mRNN bridge's private specs; kept here so the SEM columns the bridge drops
#: can be pulled for the same rows.
CONDITION_ROW_SPECS: dict[str, tuple[str, str, Optional[str]]] = {
    "face_interactive": ("split", "face", "interactive"),
    "face_non_interactive": ("split", "face", "non_interactive"),
    "object": ("unsplit", "object", None),
}
_UNIT_KEY_COLUMNS: tuple[str, ...] = (
    "date",
    "unit_uuid",
    "region",
    "spike_channel",
    "recorded_agent",
)
CONDITION_PAIRS: tuple[tuple[str, str], ...] = tuple(combinations(CONDITION_ORDER, 2))

DEFAULT_VARIANCE_THRESHOLD = 0.95
DEFAULT_SELECTIVITY_SUBDIR = "ephys/psth/fixation_psth_selectivity"
DEFAULT_SELECTIVITY_UNIT_FILENAME = "unit_selectivity__three_condition_core.csv"
DEFAULT_SELECTIVITY_PAIR_FILENAME = "pair_selectivity__three_condition_core.csv"
DEFAULT_OUTPUT_SUBDIR = "ephys/psth/fixation_population_pc_subspace"

#: Numerical identities below this bound are treated as exact.  Everything the
#: verification suite checks is a linear-algebra identity in float64, so the
#: tolerance only has to absorb accumulated round-off, not model error.
IDENTITY_TOLERANCE = 1e-8


def pair_label(condition_a: str, condition_b: str) -> str:
    """Canonical ``a__vs__b`` label matching the selectivity outputs."""
    return f"{condition_a}__vs__{condition_b}"


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegionPopulation:
    """One region's condition-by-time-by-unit firing-rate tensor.

    ``rates_hz`` is stored condition-major so that concatenating conditions
    along time is a single reshape, and so that the concatenation order is a
    property of the object rather than of whichever function last touched it.
    """

    region: str
    conditions: tuple[str, ...]
    unit_keys: tuple[str, ...]
    bin_centers_s: np.ndarray
    rates_hz: np.ndarray  # (n_conditions, n_time, n_units)
    n_trials: np.ndarray  # (n_conditions, n_units)
    unit_table: pd.DataFrame
    #: Standard error of each average, same shape as ``rates_hz``.  Carried so
    #: the trial-count imbalance between conditions can be corrected for rather
    #: than only noted -- interactive-face fixations outnumber the others about
    #: five to one, and a noisier average looks like a faster-moving population.
    sem_hz: Optional[np.ndarray] = None

    @property
    def n_units(self) -> int:
        return int(self.rates_hz.shape[2])

    @property
    def n_time(self) -> int:
        return int(self.rates_hz.shape[1])

    @property
    def n_conditions(self) -> int:
        return int(self.rates_hz.shape[0])

    def condition_matrix(self, condition: str) -> np.ndarray:
        """Units x time matrix for one condition."""
        index = self.conditions.index(str(condition))
        return np.ascontiguousarray(self.rates_hz[index].T)

    @property
    def condition_matrices(self) -> dict[str, np.ndarray]:
        return {condition: self.condition_matrix(condition) for condition in self.conditions}

    def concatenated_units_by_time(self) -> np.ndarray:
        """Units x (n_conditions * time), concatenated in ``self.conditions`` order."""
        return np.concatenate(
            [self.condition_matrix(condition) for condition in self.conditions],
            axis=1,
        )

    def condition_slices(self) -> dict[str, tuple[int, int]]:
        """Column span each condition occupies in the concatenated matrix."""
        out: dict[str, tuple[int, int]] = {}
        cursor = 0
        for condition in self.conditions:
            out[str(condition)] = (cursor, cursor + self.n_time)
            cursor += self.n_time
        return out

    def take_units(self, indices: Sequence[int]) -> "RegionPopulation":
        """Select units by positional index, allowing repeats.

        Repeats matter: a bootstrap over units has to be able to draw the same
        neuron twice, and de-duplicating the draw would quietly shrink the
        resampled population and bias every variance-based metric downward.
        """
        index = np.asarray(list(indices), dtype=int)
        if index.size == 0:
            raise ValueError(f"No units selected for region {self.region!r}.")
        table = self.unit_table.iloc[index].reset_index(drop=True)
        return RegionPopulation(
            region=self.region,
            conditions=self.conditions,
            unit_keys=tuple(self.unit_keys[i] for i in index),
            bin_centers_s=self.bin_centers_s,
            rates_hz=self.rates_hz[:, :, index],
            n_trials=self.n_trials[:, index],
            unit_table=table,
            sem_hz=None if self.sem_hz is None else self.sem_hz[:, :, index],
        )

    def subset_units(self, unit_keys: Sequence[str]) -> "RegionPopulation":
        """Restrict to a unit subset, preserving the current unit ordering."""
        wanted = {str(key) for key in unit_keys}
        keep = [idx for idx, key in enumerate(self.unit_keys) if str(key) in wanted]
        if not keep:
            raise ValueError(f"No requested units remain in region {self.region!r}.")
        return self.take_units(keep)


def _window_mask(bin_centers_s: np.ndarray, window_ms: Optional[tuple[float, float]]) -> np.ndarray:
    centers = np.asarray(bin_centers_s, dtype=float).reshape(-1)
    if window_ms is None:
        return np.ones_like(centers, dtype=bool)
    start_s = float(window_ms[0]) / 1000.0
    stop_s = float(window_ms[1]) / 1000.0
    return (centers >= start_s) & (centers <= stop_s)


def _build_sem_lookup(
    combined: pd.DataFrame,
    *,
    conditions: Sequence[str],
) -> dict[str, dict[tuple[str, ...], np.ndarray]]:
    """Map condition -> unit identity -> stored SEM trace."""
    out: dict[str, dict[tuple[str, ...], np.ndarray]] = {}
    if "psth_sem" not in combined.columns:
        return out
    for condition in conditions:
        spec = CONDITION_ROW_SPECS.get(str(condition))
        if spec is None:
            continue
        partition, category, state = spec
        mask = (combined["average_partition"].astype(str) == partition) & (
            combined["fixation_category"].astype(str) == category
        )
        if state is None:
            mask &= combined["interactive_state"].isna()
        else:
            mask &= combined["interactive_state"].astype(str) == state
        subset = combined.loc[mask]
        keys = list(
            zip(*(subset[column].astype(str) for column in _UNIT_KEY_COLUMNS))
        )
        out[str(condition)] = {
            key: np.asarray(values, dtype=float).reshape(-1)
            for key, values in zip(keys, subset["psth_sem"])
        }
    return out


def load_region_populations(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    regions: Sequence[str] = REGION_ORDER,
    conditions: Sequence[str] = CONDITION_ORDER,
    window_ms: Optional[tuple[float, float]] = None,
    unit_keys: Optional[Sequence[str]] = None,
) -> dict[str, RegionPopulation]:
    """Assemble per-region condition tensors from the combined 10 ms averages.

    The combined export stores one row per unit-condition; the mRNN bridge
    already reshapes it to one row per unit with an inner-join across the three
    modelled conditions, which is the same "unit must have all conditions"
    requirement the population PCA imposes.  Reusing it keeps the two analyses
    on an identical unit set instead of two independently derived ones.
    """
    loaded = load_combined_fixation_psth(cfg_path)
    frame = build_mrnn_training_dataframe(loaded.dataframe)
    frame = frame.assign(unit_key=frame["date"].astype(str) + "|" + frame["uuid"].astype(str))
    # The bridge keeps psth_mean only, so the SEMs are re-joined here on the
    # same unit identity it merges on.  Keeping the bridge as the sole source of
    # unit selection means the two can never disagree about which units are in.
    frame["_merge_key"] = list(
        zip(
            frame["date"].astype(str),
            frame["uuid"].astype(str),
            frame["source_region"].astype(str),
            frame["spike_channel"].astype(str),
            frame["recorded_agent"].astype(str),
        )
    )
    sem_lookup = _build_sem_lookup(loaded.dataframe, conditions=conditions)

    if unit_keys is not None:
        wanted = {str(key) for key in unit_keys}
        frame = frame.loc[frame["unit_key"].isin(wanted)]

    timeline = np.asarray(loaded.timeline_s_rel, dtype=float).reshape(-1)
    mask = _window_mask(timeline, window_ms)
    bin_centers = timeline[mask]

    conditions = tuple(str(condition) for condition in conditions)
    populations: dict[str, RegionPopulation] = {}
    for region in regions:
        region_frame = frame.loc[frame["region"].astype(str) == str(region)]
        region_frame = region_frame.sort_values(["date", "uuid"]).reset_index(drop=True)
        if region_frame.empty:
            continue

        stacked = []
        trials = []
        sems = []
        for condition in conditions:
            column = CONDITION_TO_TRAINING_COLUMN[condition]
            traces = np.column_stack(
                [np.asarray(values, dtype=float).reshape(-1) for values in region_frame[column]]
            )
            stacked.append(traces[mask, :])
            trials.append(np.asarray(region_frame[f"{column}_n_trials"], dtype=float))
            sem_column = sem_lookup.get(condition)
            if sem_column is not None:
                sem_traces = np.column_stack(
                    [
                        np.asarray(sem_column.get(key, np.full(timeline.size, np.nan)), dtype=float).reshape(-1)
                        for key in region_frame["_merge_key"]
                    ]
                )
                sems.append(sem_traces[mask, :])

        rates = np.stack(stacked, axis=0)
        if not np.isfinite(rates).all():
            raise ValueError(f"Region {region!r} contains non-finite firing rates.")
        sem_array = np.stack(sems, axis=0) if len(sems) == len(conditions) else None
        if sem_array is not None and not np.isfinite(sem_array).all():
            sem_array = None

        populations[str(region)] = RegionPopulation(
            region=str(region),
            conditions=conditions,
            unit_keys=tuple(region_frame["unit_key"].astype(str)),
            bin_centers_s=bin_centers,
            rates_hz=rates,
            n_trials=np.stack(trials, axis=0),
            unit_table=region_frame.loc[
                :, ["unit_key", "date", "uuid", "region", "spike_channel", "recorded_agent"]
            ].copy(),
            sem_hz=sem_array,
        )
    return populations


def build_unit_inventory(populations: Mapping[str, RegionPopulation]) -> pd.DataFrame:
    """One row per region: unit, session and trial counts feeding the PCA."""
    rows: list[dict] = []
    for region, population in populations.items():
        row: dict[str, object] = {
            "region": region,
            "n_units": population.n_units,
            "n_sessions": int(population.unit_table["date"].nunique()),
            "n_time_bins": population.n_time,
            "window_start_ms": float(population.bin_centers_s[0] * 1000.0),
            "window_stop_ms": float(population.bin_centers_s[-1] * 1000.0),
            "fr_matrix_units_by_time": f"{population.n_units} x {population.n_time}",
            "concatenated_matrix_units_by_time": (
                f"{population.n_units} x {population.n_time * population.n_conditions}"
            ),
            "max_pca_rank": int(
                min(population.n_units, population.n_time * population.n_conditions - 1)
            ),
        }
        for index, condition in enumerate(population.conditions):
            row[f"median_trials_{condition}"] = float(np.median(population.n_trials[index]))
            row[f"total_trials_{condition}"] = float(np.sum(population.n_trials[index]))
            row[f"mean_fr_hz_{condition}"] = float(np.mean(population.rates_hz[index]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)


def load_pair_selective_units(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    selectivity_subdir: str = DEFAULT_SELECTIVITY_SUBDIR,
    unit_filename: str = DEFAULT_SELECTIVITY_UNIT_FILENAME,
    pair_filename: str = DEFAULT_SELECTIVITY_PAIR_FILENAME,
    use_corrected: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load unit- and pair-level fixation-pair selectivity calls.

    Returns the unit table restricted to selective units and the full pair
    table, so a notebook can both subset the population and report *which*
    pairs drove each unit's inclusion.
    """
    cfg = load_config(cfg_path)
    root = build_analysis_output_dir(cfg, selectivity_subdir)
    unit_df = pd.read_csv(root / unit_filename)
    pair_df = pd.read_csv(root / pair_filename)

    unit_column = "is_selective_unit_corrected" if use_corrected else "is_selective_unit_raw"
    pair_column = "is_selective_pair_corrected" if use_corrected else "is_selective_pair_raw"
    selective_units = unit_df.loc[unit_df[unit_column].astype(bool)].reset_index(drop=True)
    pair_df = pair_df.assign(is_selective=pair_df[pair_column].astype(bool))
    return selective_units, pair_df


def summarize_selective_units(
    unit_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    *,
    populations: Optional[Mapping[str, RegionPopulation]] = None,
) -> pd.DataFrame:
    """Region-level counts of selective units and of each selective pair."""
    rows: list[dict] = []
    for region, group in unit_df.groupby("region"):
        row: dict[str, object] = {"region": str(region), "n_selective_units": int(len(group))}
        if populations is not None and str(region) in populations:
            total = populations[str(region)].n_units
            row["n_units_total"] = int(total)
            row["fraction_selective"] = float(len(group)) / float(total) if total else np.nan
        region_pairs = pair_df.loc[
            (pair_df["region"].astype(str) == str(region)) & pair_df["is_selective"]
        ]
        for label, count in region_pairs["pair_label"].value_counts().items():
            row[f"n_selective_{label}"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PopulationPCAFit:
    """A full-rank PCA fit of one region, plus per-condition projections.

    ``components`` spans every direction with non-negligible variance, not a
    configured top-k.  Truncation is a decision made downstream by
    :func:`n_components_for_variance`, which keeps "how many dimensions do we
    need" a measurement instead of a setting.
    """

    region: str
    fit_scope: str
    conditions: tuple[str, ...]
    unit_keys: tuple[str, ...]
    bin_centers_s: np.ndarray
    mean: np.ndarray  # (n_units,)
    components: np.ndarray  # (n_pc, n_units)
    singular_values: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    scores_by_condition: dict[str, np.ndarray]  # condition -> (n_pc, n_time)
    concatenated_scores: np.ndarray  # (n_pc, n_conditions * n_time)
    condition_slices: dict[str, tuple[int, int]]
    n_samples_fit: int

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def basis(self, n_components: int) -> np.ndarray:
        """Top-k orthonormal basis as ``(n_units, k)``, columns = directions."""
        k = int(min(max(1, int(n_components)), self.n_components))
        return np.ascontiguousarray(self.components[:k, :].T)

    def project(self, matrix_units_by_time: np.ndarray, *, n_components: Optional[int] = None) -> np.ndarray:
        """Project a units-by-time matrix into PC space as ``(n_pc, n_time)``."""
        k = self.n_components if n_components is None else int(n_components)
        basis = self.basis(k)
        centered = np.asarray(matrix_units_by_time, dtype=float) - self.mean.reshape(-1, 1)
        return basis.T @ centered

    def reconstruct(self, scores_pc_by_time: np.ndarray) -> np.ndarray:
        """Back-project PC scores to units-by-time firing rates."""
        scores = np.asarray(scores_pc_by_time, dtype=float)
        basis = self.basis(scores.shape[0])
        return basis @ scores + self.mean.reshape(-1, 1)


def fit_population_pca(
    population: RegionPopulation,
    *,
    fit_scope: str = "concatenated",
    max_components: Optional[int] = None,
    rank_tolerance: float = 1e-12,
) -> PopulationPCAFit:
    """Fit PCA on time samples of one region, with units as features.

    ``fit_scope`` is either ``"concatenated"`` (all conditions stacked along
    time, the shared basis every cross-condition comparison needs) or a single
    condition name (that condition's own basis).  Either way the returned fit
    projects *all* conditions, so a per-condition fit can immediately be
    evaluated on the conditions it was not fitted to.
    """
    if fit_scope == "concatenated":
        matrix = population.concatenated_units_by_time()
    else:
        if fit_scope not in population.conditions:
            raise ValueError(f"Unknown fit scope {fit_scope!r} for region {population.region!r}.")
        matrix = population.condition_matrix(fit_scope)

    samples = np.asarray(matrix, dtype=float).T  # time samples x units
    n_samples, n_features = samples.shape
    mean = samples.mean(axis=0)
    centered = samples - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    # Directions past the numerical rank carry only round-off; keeping them
    # would put a long tail of ~0 components into every cumulative curve and
    # make "components needed for 95%" depend on float noise.
    keep = singular_values > (rank_tolerance * float(singular_values[0]) * max(centered.shape))
    n_keep = int(np.count_nonzero(keep))
    if max_components is not None:
        n_keep = min(n_keep, int(max_components))
    n_keep = max(1, n_keep)

    components = np.ascontiguousarray(vt[:n_keep, :])
    singular_kept = singular_values[:n_keep]
    denominator = float(max(n_samples - 1, 1))
    explained_variance = (singular_kept**2) / denominator
    total_variance = float(np.sum((singular_values**2) / denominator))
    ratio = explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)

    scores_by_condition: dict[str, np.ndarray] = {}
    for condition in population.conditions:
        condition_centered = population.condition_matrix(condition) - mean.reshape(-1, 1)
        scores_by_condition[str(condition)] = components @ condition_centered

    concatenated_scores = np.concatenate(
        [scores_by_condition[str(condition)] for condition in population.conditions],
        axis=1,
    )

    return PopulationPCAFit(
        region=population.region,
        fit_scope=str(fit_scope),
        conditions=population.conditions,
        unit_keys=population.unit_keys,
        bin_centers_s=population.bin_centers_s,
        mean=mean,
        components=components,
        singular_values=singular_kept,
        explained_variance=explained_variance,
        explained_variance_ratio=ratio,
        cumulative_explained_variance_ratio=np.cumsum(ratio),
        scores_by_condition=scores_by_condition,
        concatenated_scores=concatenated_scores,
        condition_slices=population.condition_slices(),
        n_samples_fit=int(n_samples),
    )


def fit_all_scopes(population: RegionPopulation) -> dict[str, PopulationPCAFit]:
    """Concatenated fit plus one fit per condition, keyed by ``fit_scope``."""
    fits = {"concatenated": fit_population_pca(population, fit_scope="concatenated")}
    for condition in population.conditions:
        fits[str(condition)] = fit_population_pca(population, fit_scope=str(condition))
    return fits


def n_components_for_variance(
    cumulative_ratio: np.ndarray,
    threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> int:
    """Smallest component count whose cumulative variance reaches ``threshold``."""
    cumulative = np.asarray(cumulative_ratio, dtype=float).reshape(-1)
    if cumulative.size == 0:
        return 0
    reached = np.nonzero(cumulative >= float(threshold))[0]
    return int(reached[0] + 1) if reached.size else int(cumulative.size)


def build_dimensionality_table(
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Components needed per region and fit scope, at several thresholds."""
    rows: list[dict] = []
    for region, fits in fits_by_region.items():
        for scope, fit in fits.items():
            row: dict[str, object] = {
                "region": str(region),
                "fit_scope": str(scope),
                "n_units": int(fit.components.shape[1]),
                "n_samples_fit": int(fit.n_samples_fit),
                "n_components_full_rank": int(fit.n_components),
                "participation_ratio": float(participation_ratio(fit.explained_variance)),
            }
            for threshold in thresholds:
                key = f"n_pcs_for_{int(round(float(threshold) * 100))}pct"
                row[key] = n_components_for_variance(
                    fit.cumulative_explained_variance_ratio, threshold
                )
            for index in range(3):
                ratios = fit.explained_variance_ratio
                row[f"explained_variance_ratio_pc{index + 1}"] = (
                    float(ratios[index]) if ratios.size > index else np.nan
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    scope_rank = {"concatenated": 0}
    frame["_scope_rank"] = frame["fit_scope"].map(lambda value: scope_rank.get(value, 1))
    return (
        frame.sort_values(["region", "_scope_rank", "fit_scope"])
        .drop(columns="_scope_rank")
        .reset_index(drop=True)
    )


def resolve_shared_n_components(
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    fit_scope: str = "concatenated",
) -> tuple[int, dict[str, int]]:
    """Shared dimension count: the largest per-region requirement.

    Taking the maximum rather than the mean is what makes the shared count
    honest — it guarantees the retained subspace reaches ``threshold`` in
    *every* region, so no region is analysed in a space that is too small for
    it.  This reproduces the rule the mRNN target builder uses.
    """
    per_region = {
        str(region): n_components_for_variance(
            fits[str(fit_scope)].cumulative_explained_variance_ratio, threshold
        )
        for region, fits in fits_by_region.items()
    }
    return int(max(per_region.values())), per_region


def participation_ratio(explained_variance: np.ndarray) -> float:
    """Effective dimensionality ``(sum l)^2 / sum l^2`` of a variance spectrum."""
    values = np.asarray(explained_variance, dtype=float).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    return float(values.sum() ** 2 / np.sum(values**2))


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def _check_row(name: str, description: str, value: float, tolerance: float) -> dict:
    """A check that passes when ``|value|`` sits within ``tolerance`` of zero."""
    return {
        "check": name,
        "description": description,
        "value": float(value),
        "tolerance": float(tolerance),
        "criterion": "|value| <= tolerance",
        "passed": bool(np.isfinite(value) and abs(float(value)) <= float(tolerance)),
    }


def _margin_row(name: str, description: str, value: float, floor: float) -> dict:
    """A check that passes when ``value`` stays above ``floor``.

    Kept separate from :func:`_check_row` because a margin that must be large
    and a residual that must be small fail in opposite directions, and folding
    both into one absolute-value rule would mark a healthy margin as a failure.
    """
    return {
        "check": name,
        "description": description,
        "value": float(value),
        "tolerance": float(floor),
        "criterion": "value > tolerance",
        "passed": bool(np.isfinite(value) and float(value) > float(floor)),
    }


def verify_pca_identities(
    population: RegionPopulation,
    fit: PopulationPCAFit,
    *,
    n_components: Optional[int] = None,
    tolerance: float = IDENTITY_TOLERANCE,
) -> pd.DataFrame:
    """Assert every linear-algebra identity the projections depend on.

    The one that matters most in practice is the concatenation round-trip: the
    per-condition score blocks must be recoverable from the concatenated
    projection *in the declared condition order*.  An off-by-one-block error
    there would silently relabel every trajectory while leaving the figures
    looking entirely plausible, so it is checked as an exact identity rather
    than eyeballed.
    """
    rows: list[dict] = []
    k = fit.n_components if n_components is None else int(n_components)

    identity = fit.components @ fit.components.T
    rows.append(
        _check_row(
            "components_orthonormal",
            "PCA components form an orthonormal basis (max |CC' - I|).",
            float(np.max(np.abs(identity - np.eye(identity.shape[0])))),
            1e-9,
        )
    )

    concatenated = population.concatenated_units_by_time()
    expected_mean = concatenated.mean(axis=1) if fit.fit_scope == "concatenated" else (
        population.condition_matrix(fit.fit_scope).mean(axis=1)
    )
    rows.append(
        _check_row(
            "fit_mean_is_sample_mean",
            "Stored PCA mean equals the mean over the fitted time samples.",
            float(np.max(np.abs(fit.mean - expected_mean))),
            tolerance,
        )
    )

    slices = population.condition_slices()
    worst_slice = 0.0
    for condition in population.conditions:
        start, stop = slices[str(condition)]
        from_concatenated = fit.concatenated_scores[:, start:stop]
        from_reprojection = fit.project(population.condition_matrix(condition))
        worst_slice = max(worst_slice, float(np.max(np.abs(from_concatenated - from_reprojection))))
    rows.append(
        _check_row(
            "concatenation_slice_roundtrip",
            "Slicing the concatenated projection by condition equals re-projecting "
            "that condition on its own (max abs difference).",
            worst_slice,
            tolerance,
        )
    )

    direct = fit.components @ (concatenated - fit.mean.reshape(-1, 1))
    rows.append(
        _check_row(
            "concatenated_scores_match_manual_projection",
            "Stored concatenated scores equal C (X - mean) computed from scratch.",
            float(np.max(np.abs(direct - fit.concatenated_scores))),
            tolerance,
        )
    )

    worst_reconstruction = 0.0
    for condition in population.conditions:
        original = population.condition_matrix(condition)
        rebuilt = fit.reconstruct(fit.project(original))
        worst_reconstruction = max(
            worst_reconstruction, float(np.max(np.abs(rebuilt - original)))
        )
    rows.append(
        _check_row(
            "full_rank_reconstruction_exact",
            "Back-projecting all retained PCs recovers the original firing rates.",
            worst_reconstruction,
            1e-8,
        )
    )

    worst_variance = 0.0
    for condition in population.conditions:
        centered = population.condition_matrix(condition) - fit.mean.reshape(-1, 1)
        scores = fit.project(population.condition_matrix(condition))
        worst_variance = max(
            worst_variance,
            abs(float(np.sum(centered**2) - np.sum(scores**2))),
        )
    rows.append(
        _check_row(
            "projection_preserves_total_energy",
            "Full-rank projection conserves sum of squares (Parseval identity).",
            worst_variance / max(float(np.sum(concatenated**2)), 1e-12),
            1e-9,
        )
    )

    time_offsets = []
    for condition in population.conditions:
        start, stop = slices[str(condition)]
        block = fit.concatenated_scores[:, start:stop]
        reference = fit.scores_by_condition[str(condition)]
        lag_zero = float(np.sum(block * reference))
        shifted = float(np.sum(block * np.roll(reference, 1, axis=1)))
        time_offsets.append(lag_zero - shifted)
    rows.append(
        _margin_row(
            "time_order_not_shifted",
            "Zero-lag alignment beats a one-bin circular shift for every condition "
            "block, so no block is written back off-by-one in time.",
            float(min(time_offsets)),
            0.0,
        )
    )

    frame = pd.DataFrame(rows)
    frame.insert(0, "region", population.region)
    frame.insert(1, "fit_scope", fit.fit_scope)
    frame.insert(2, "n_components_checked", int(k))
    return frame


def condition_identity_confusion(
    population: RegionPopulation,
    fit: PopulationPCAFit,
    *,
    n_components: int,
) -> pd.DataFrame:
    """Does each condition's PC trajectory back-project to *that* condition?

    Reconstructs every condition from its own truncated PC scores and scores
    the reconstruction against every condition's true firing-rate matrix.  If
    the trajectories were mislabelled anywhere in the concatenate-project-slice
    path, the winning match would be off-diagonal.  Reporting the whole matrix
    rather than a pass/flag also shows *how* separable the conditions are: a
    diagonal that barely wins means overlapping population states, not a bug.
    """
    rows: list[dict] = []
    for source in population.conditions:
        scores = fit.project(population.condition_matrix(source), n_components=n_components)
        rebuilt = fit.reconstruct(scores)
        for target in population.conditions:
            observed = population.condition_matrix(target)
            residual = float(np.sum((observed - rebuilt) ** 2))
            total = float(np.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2))
            rows.append(
                {
                    "region": population.region,
                    "fit_scope": fit.fit_scope,
                    "n_components": int(n_components),
                    "reconstructed_from": str(source),
                    "compared_against": str(target),
                    "r2": 1.0 - residual / total if total > 0 else np.nan,
                    "rmse_hz": float(np.sqrt(residual / observed.size)),
                    "correlation": float(
                        np.corrcoef(rebuilt.reshape(-1), observed.reshape(-1))[0, 1]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    best = (
        frame.loc[frame.groupby("reconstructed_from")["r2"].idxmax()]
        .set_index("reconstructed_from")["compared_against"]
        .to_dict()
    )
    frame["is_best_match"] = [
        best.get(row.reconstructed_from) == row.compared_against for row in frame.itertuples()
    ]
    frame["identity_recovered"] = [
        bool(row.is_best_match and row.reconstructed_from == row.compared_against)
        for row in frame.itertuples()
    ]
    return frame


def verify_against_stored_pca(
    populations: Mapping[str, RegionPopulation],
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    cfg_path: str | Path = "configs/dataset.yaml",
    stored_subdir: str = "ephys/psth/fixation_population_pca",
    stored_filename: str = "results.pkl",
) -> pd.DataFrame:
    """Compare this module's fits with the stored population-PCA results.

    The stored pipeline reads an older average export, so the point of this
    comparison is not to assert equality but to quantify the drift: matched
    unit sets with different underlying trial counts, and the resulting shift
    in how many PCs each region needs.
    """
    cfg = load_config(cfg_path)
    path = build_analysis_output_dir(cfg, stored_subdir) / stored_filename
    if not path.exists():
        return pd.DataFrame()
    stored = pd.read_pickle(path)

    rows: list[dict] = []
    for region, population in populations.items():
        payload = stored.get("regions", {}).get(str(region))
        if payload is None:
            continue
        stored_keys = {str(key) for key in payload["unit_keys"]}
        current_keys = set(population.unit_keys)
        stored_fit = payload["concatenated_fit"]
        stored_cumulative = np.asarray(
            stored_fit["cumulative_explained_variance_ratio"], dtype=float
        )
        current_fit = fits_by_region[str(region)]["concatenated"]

        stored_reached = bool(stored_cumulative[-1] >= DEFAULT_VARIANCE_THRESHOLD)
        rows.append(
            {
                "region": str(region),
                "n_units_stored": int(len(stored_keys)),
                "n_units_current": int(len(current_keys)),
                "n_units_shared": int(len(stored_keys & current_keys)),
                "unit_sets_identical": stored_keys == current_keys,
                "stored_max_components": int(stored_fit["n_components"]),
                "stored_cumulative_at_cap": float(stored_cumulative[-1]),
                "stored_n_pcs_for_95pct": (
                    n_components_for_variance(stored_cumulative) if stored_reached else np.nan
                ),
                "current_n_pcs_for_95pct": n_components_for_variance(
                    current_fit.cumulative_explained_variance_ratio
                ),
                "stored_evr_pc1": float(stored_fit["explained_variance_ratio"][0]),
                "current_evr_pc1": float(current_fit.explained_variance_ratio[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Subspace comparison
# --------------------------------------------------------------------------- #


def condition_covariance(population: RegionPopulation, condition: str) -> np.ndarray:
    """Units-by-units temporal covariance of one condition's trajectory."""
    matrix = population.condition_matrix(condition)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    return (centered @ centered.T) / float(max(matrix.shape[1] - 1, 1))


def variance_captured(covariance: np.ndarray, basis: np.ndarray) -> float:
    """Variance of a covariance captured by an orthonormal ``(n_units, k)`` basis."""
    return float(np.trace(basis.T @ np.asarray(covariance, dtype=float) @ basis))


def alignment_index(
    covariance: np.ndarray,
    basis: np.ndarray,
) -> float:
    """Elsayed-style alignment index: captured variance over the best possible.

    Normalising by the top-k eigenvalues of the *evaluated* condition rather
    than by its total variance is what makes the number comparable across
    conditions and regions: a value of 1 means the foreign basis is as good as
    that condition's own optimal k-dimensional subspace, and a low value means
    genuinely different directions rather than merely a lower-variance
    condition.
    """
    cov = np.asarray(covariance, dtype=float)
    k = int(np.asarray(basis).shape[1])
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    best = float(np.sum(eigenvalues[:k]))
    if best <= 0:
        return float("nan")
    return variance_captured(cov, basis) / best


def analytic_alignment_floor(n_units: int, n_components: int) -> float:
    """Expected alignment index of a random subspace: k / N.

    A uniformly random k-dimensional subspace of R^N captures, in expectation, a
    fraction k/N of the trace of *any* covariance, because the expected
    projection of a fixed unit vector onto it is k/N.  Normalising by the top-k
    eigenvalues leaves that ratio essentially unchanged whenever those
    eigenvalues already account for nearly all the variance, which they do here.

    This is why raw alignment indices must not be compared across regions.  The
    four areas have very different unit counts, so at k = 42 the floor ranges
    from 42/537 = 0.08 in BLA to 42/187 = 0.22 in dmPFC.  dmPFC's higher
    alignment values are partly just its smaller population.
    """
    units = int(n_units)
    if units <= 0:
        return float("nan")
    return float(min(int(n_components), units)) / float(units)


def alignment_above_floor(alignment: float, floor: float) -> float:
    """Rescale an alignment index so 0 is chance and 1 is a perfect match.

    ``(A - k/N) / (1 - k/N)``.  Without this, a region with fewer neurons looks
    more aligned for a purely combinatorial reason, and cross-region statements
    about subspace overlap are not interpretable.
    """
    value = float(alignment)
    base = float(floor)
    if not np.isfinite(value) or not np.isfinite(base) or base >= 1.0:
        return float("nan")
    return (value - base) / (1.0 - base)


def random_subspace_alignment_null(
    covariance: np.ndarray,
    *,
    n_components: int,
    n_samples: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Alignment indices for random orthonormal subspaces of the same rank.

    A k-dimensional subspace of a d-dimensional space captures a non-trivial
    share of variance by chance alone when k is an appreciable fraction of d,
    so a raw alignment index of, say, 0.6 is uninterpretable without this
    baseline.
    """
    cov = np.asarray(covariance, dtype=float)
    rng = np.random.default_rng(int(seed))
    n_units = cov.shape[0]
    k = int(min(max(1, n_components), n_units))
    out = np.empty(int(n_samples), dtype=float)
    for index in range(int(n_samples)):
        basis, _ = np.linalg.qr(rng.standard_normal((n_units, k)))
        out[index] = alignment_index(cov, basis[:, :k])
    return out


def cross_condition_variance_curve(
    population: RegionPopulation,
    fits: Mapping[str, PopulationPCAFit],
    *,
    max_components: int,
) -> pd.DataFrame:
    """Cumulative variance of each condition captured by each condition's PCs.

    The curve, not the single number, is what distinguishes "the subspaces
    differ" from "the subspaces differ only in the tail": two conditions whose
    curves separate over the first few components occupy genuinely different
    leading directions, whereas curves that converge quickly share the same
    dominant structure and differ only in how much they use it.

    Also carries ``fraction_within_retained_pcs``, the normalisation the older
    ``fixation_population_pca`` builder writes to
    ``explained_variance_fraction``.  That column divides by the variance inside
    the retained PCs rather than by the condition's total variance, so it
    approaches 1 for every pairing by construction and is not a
    variance-explained measure; it is reproduced here only so the two outputs
    can be reconciled.
    """
    rows: list[dict] = []
    covariances = {
        condition: condition_covariance(population, condition)
        for condition in population.conditions
    }
    for source in population.conditions:
        fit = fits[str(source)]
        k_max = int(min(max_components, fit.n_components))
        for target in population.conditions:
            covariance = covariances[target]
            total = float(np.trace(covariance))
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
            projected = np.einsum(
                "ij,jk,ki->i",
                fit.components[:k_max, :],
                covariance,
                fit.components[:k_max, :].T,
            )
            cumulative = np.cumsum(projected)
            best = np.cumsum(eigenvalues[:k_max])
            for index in range(k_max):
                rows.append(
                    {
                        "region": population.region,
                        "pc_condition": str(source),
                        "eval_condition": str(target),
                        "is_within_condition": source == target,
                        "n_components": int(index + 1),
                        "projected_variance": float(projected[index]),
                        "cumulative_projected_variance": float(cumulative[index]),
                        "cumulative_variance_explained": (
                            float(cumulative[index] / total) if total > 0 else np.nan
                        ),
                        "alignment_index": (
                            float(cumulative[index] / best[index]) if best[index] > 0 else np.nan
                        ),
                        "fraction_within_retained_pcs": (
                            float(cumulative[index] / cumulative[k_max - 1])
                            if cumulative[k_max - 1] > 0
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def within_condition_alignment_ceiling(
    population: RegionPopulation,
    *,
    n_components: int,
) -> pd.DataFrame:
    """Alignment a condition reaches against an independent half of itself.

    Fits on the odd time bins and evaluates on the even ones.  Cross-condition
    alignment has to be read against this number rather than against 1.0: a
    subspace estimated from finite, noisy data does not perfectly capture even
    its own condition, so the shortfall from 1.0 is partly estimation error and
    only the shortfall *below this ceiling* is evidence of a real difference.

    Because the traces are Gaussian-smoothed at 20 ms over 10 ms bins, adjacent
    bins are not independent and this ceiling is optimistic -- it bounds how
    high alignment could plausibly go, and is not itself a noise estimate.
    """
    rows: list[dict] = []
    for condition in population.conditions:
        matrix = population.condition_matrix(condition)
        odd = matrix[:, 1::2]
        even = matrix[:, 0::2]
        odd_centered = odd - odd.mean(axis=1, keepdims=True)
        even_centered = even - even.mean(axis=1, keepdims=True)
        _, _, vt = np.linalg.svd(odd_centered.T, full_matrices=False)
        k = int(min(n_components, vt.shape[0]))
        basis = np.ascontiguousarray(vt[:k, :].T)
        covariance = (even_centered @ even_centered.T) / float(max(even.shape[1] - 1, 1))
        rows.append(
            {
                "region": population.region,
                "condition": str(condition),
                "n_components": int(k),
                "alignment_ceiling": alignment_index(covariance, basis),
                "variance_explained_ceiling": (
                    variance_captured(covariance, basis) / float(np.trace(covariance))
                    if float(np.trace(covariance)) > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def principal_angle_metrics(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    *,
    degrees: bool = True,
) -> dict[str, float | np.ndarray]:
    """Principal angles between two subspaces and the usual scalar summaries.

    Principal angles are the scale-free counterpart to the variance-based
    measures: they ask whether the *directions* differ, independently of how
    much the population happens to move along them.  Two conditions can share a
    subspace almost perfectly (small angles) while still tracing very different
    trajectories inside it, and the pair of measures separates those cases.
    """
    angles = subspace_angles(np.asarray(basis_a, dtype=float), np.asarray(basis_b, dtype=float))
    angles = np.sort(np.asarray(angles, dtype=float))
    cos2 = float(np.mean(np.cos(angles) ** 2))
    return {
        "principal_angles": np.degrees(angles) if degrees else angles,
        "mean_principal_angle": float(np.degrees(angles).mean() if degrees else angles.mean()),
        "min_principal_angle": float(np.degrees(angles[0]) if degrees else angles[0]),
        "max_principal_angle": float(np.degrees(angles[-1]) if degrees else angles[-1]),
        "mean_cos2": cos2,
        "grassmann_distance": float(np.sqrt(np.sum(angles**2))),
        "chordal_distance": float(np.sqrt(np.sum(np.sin(angles) ** 2))),
        "projection_metric": float(np.sin(angles[-1])),
    }


def trajectory_geometry(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_components: int,
) -> dict[str, float]:
    """Distance, angle and shape statistics between two PC trajectories."""
    k = int(n_components)
    a = np.asarray(scores_a, dtype=float)[:k, :]
    b = np.asarray(scores_b, dtype=float)[:k, :]

    distance = np.linalg.norm(a - b, axis=0)
    centroid_a = a.mean(axis=1)
    centroid_b = b.mean(axis=1)
    spread_a = float(np.mean(np.linalg.norm(a - centroid_a[:, None], axis=0)))
    spread_b = float(np.mean(np.linalg.norm(b - centroid_b[:, None], axis=0)))
    pooled_spread = 0.5 * (spread_a + spread_b)

    norms = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(norms > 1e-12, np.sum(a * b, axis=0) / norms, np.nan)
    angle_deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    # Procrustes removes translation, scale and rotation, so what is left is a
    # pure shape difference: two trajectories that trace the same path in
    # different parts of the space score as similar here but far apart on the
    # centroid distance, which is exactly the distinction we want to report.
    _, _, disparity = procrustes(a.T, b.T)

    # Exact orthogonal split of the mean squared separation into a static and a
    # moving part.  Writing z_c(t) = m_c + d_c(t) with mean_t d_c = 0,
    #     <||z_a - z_b||^2>_t = ||m_a - m_b||^2 + <||d_a - d_b||^2>_t,
    # because the cross term carries a factor mean_t(d_a - d_b) = 0.  The two
    # terms answer different questions -- "do the conditions sit in different
    # places" and "do they move differently" -- and reporting only their sum
    # conflates them.
    deviation_a = a - centroid_a[:, None]
    deviation_b = b - centroid_b[:, None]
    mean_square_distance = float(np.mean(distance**2))
    offset_square = float(np.sum((centroid_a - centroid_b) ** 2))
    dynamics_square = float(np.mean(np.sum((deviation_a - deviation_b) ** 2, axis=0)))
    energy_a = float(np.mean(np.sum(deviation_a**2, axis=0)))
    energy_b = float(np.mean(np.sum(deviation_b**2, axis=0)))
    denominator = float(np.sqrt(energy_a * energy_b))

    return {
        "mean_distance": float(np.mean(distance)),
        "max_distance": float(np.max(distance)),
        "distance_at_fixation_onset": float(distance[distance.size // 2]),
        "centroid_distance": float(np.linalg.norm(centroid_a - centroid_b)),
        "normalized_separation": (
            float(np.mean(distance) / pooled_spread) if pooled_spread > 1e-12 else np.nan
        ),
        "mean_state_angle_deg": float(np.nanmean(angle_deg)),
        "procrustes_disparity": float(disparity),
        # Correlation of the raw score matrices, offsets included -- dominated by
        # where the conditions sit, not by how they move.  Kept because it is
        # the number a naive comparison would report; read
        # ``deviation_correlation`` for the shape question.
        "trajectory_correlation": float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1]),
        "mean_square_distance": mean_square_distance,
        "offset_square_distance": offset_square,
        "dynamics_square_distance": dynamics_square,
        "offset_share_of_separation": (
            offset_square / mean_square_distance if mean_square_distance > 0 else np.nan
        ),
        "dynamics_share_of_separation": (
            dynamics_square / mean_square_distance if mean_square_distance > 0 else np.nan
        ),
        # <d_a, d_b>_t normalised by the two excursion energies: +1 means the two
        # conditions move through their own neighbourhoods in lockstep, 0 means
        # unrelated motion, -1 means mirrored motion.
        "deviation_correlation": (
            float(np.mean(np.sum(deviation_a * deviation_b, axis=0)) / denominator)
            if denominator > 1e-12
            else np.nan
        ),
        "excursion_rms_a": float(np.sqrt(energy_a)),
        "excursion_rms_b": float(np.sqrt(energy_b)),
    }


#: Gaussian smoothing applied by the average builder, in bins.  The 10 ms
#: averages are smoothed at sigma = 20 ms, so sigma = 2 bins.  This is not a
#: free parameter: it is read off the average files' metadata, and it matters
#: because smoothing makes the estimation noise autocorrelated at exactly the
#: timescale a bin-to-bin derivative probes.
DEFAULT_SMOOTHING_SIGMA_BINS = 2.0


def noise_autocorrelation(lag: np.ndarray | float, *, sigma_bins: float) -> np.ndarray:
    """Autocorrelation of white noise after Gaussian smoothing.

    Convolving white noise with a Gaussian of width sigma leaves noise whose
    autocovariance is the kernel's autocorrelation, itself Gaussian with width
    sigma*sqrt(2); normalising gives exp(-lag^2 / (4 sigma^2)).

    Ignoring this and treating the smoothed noise as independent across bins
    overstates the noise in a first difference by more than an order of
    magnitude at sigma = 2 bins, which would drive every corrected speed to
    zero regardless of the data.
    """
    sigma = float(sigma_bins)
    lags = np.asarray(lag, dtype=float)
    if sigma <= 0:
        return (lags == 0).astype(float)
    return np.exp(-(lags**2) / (4.0 * sigma**2))


def noise_centring_fraction(n_time: int, *, sigma_bins: float) -> float:
    """Fraction of a condition's noise energy that survives into its time-mean.

    For independent noise this is 1/T.  Correlated noise averages down more
    slowly, so the centroid keeps more noise and the deviation keeps less; the
    two must use the same number or the offset/dynamics split will not balance.
    """
    size = int(n_time)
    if size <= 1:
        return 1.0
    lags = np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
    return float(noise_autocorrelation(lags, sigma_bins=sigma_bins).mean())


def projected_noise_covariance(
    population: RegionPopulation,
    condition: str,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected noise covariance of one condition's PC scores.

    Each stored average carries a standard error s_c(t,n).  Treating the
    estimation noise as independent across neurons -- they come from different
    channels and mostly different sessions -- the noise covariance in unit space
    at time t is diag(s_c(t,.)^2), and projecting onto an orthonormal basis V
    (units x k) gives V' diag(s^2) V.

    Returns the time-averaged k x k noise covariance and the per-bin noise
    energy trace(V' diag(s^2) V), which is what inflates apparent speed.

    This matters because the conditions are not equally sampled: interactive
    face averages rest on roughly five times as many fixations as the other two,
    so their averages are correspondingly quieter.  Without this correction a
    difference in trial count reads as a difference in population dynamics.
    """
    if population.sem_hz is None:
        raise ValueError(f"Region {population.region!r} has no stored SEMs to correct with.")
    index = population.conditions.index(str(condition))
    variances = np.asarray(population.sem_hz[index], dtype=float) ** 2  # (time, units)
    matrix = np.asarray(basis, dtype=float)  # (units, k)
    squared_loadings = matrix**2  # (units, k)
    per_bin_energy = variances @ squared_loadings.sum(axis=1)  # (time,)
    mean_variance = variances.mean(axis=0)  # (units,)
    covariance = (matrix * mean_variance[:, None]).T @ matrix
    return covariance, per_bin_energy


def condition_dynamics_summary(
    population: RegionPopulation,
    fit: PopulationPCAFit,
    *,
    n_components: int,
    noise_correct: bool = True,
    smoothing_sigma_bins: float = DEFAULT_SMOOTHING_SIGMA_BINS,
) -> pd.DataFrame:
    """Per-condition description of *how* the population state moves.

    Everything is computed on the deviation from each condition's own centroid,
    d_c(t) = z_c(t) - m_c, so none of it can be inflated by the large static
    offsets that dominate the raw trajectories.

    - ``excursion_rms`` = sqrt(<||d_c||^2>_t): the radius of the neighbourhood
      the condition explores.
    - ``rms_speed`` = sqrt(<||z_c(t+1) - z_c(t)||^2>_t) / dt.  A condition can
      have a large excursion and a low speed (a slow drift) or the reverse (fast
      jitter in a small volume), so the two are reported apart.
    - ``dynamics_participation_ratio``: participation ratio of the temporal
      covariance of d_c, i.e. the effective number of dimensions the motion
      uses -- generally far smaller than the retained k.

    All three are inflated by estimation noise, and the conditions are **not**
    equally sampled: interactive-face averages rest on roughly five times as
    many fixations as the other two, so they are intrinsically quieter and would
    otherwise look like a slower, lower-dimensional population for purely
    statistical reasons.  The ``*_corrected`` columns subtract the expected
    noise contribution built from the stored SEMs.  **Read those.**

    ``speed_over_noise`` is the one to check before interpreting speed at all:
    it is the observed step energy divided by the step energy noise alone would
    produce.  Values at or below 1 mean bin-to-bin motion is not resolvable
    above the noise floor, and the corrected speed is then reported as zero
    rather than as a small positive number.
    """
    k = int(n_components)
    n_time = int(population.n_time)
    dt = float(np.mean(np.diff(np.asarray(population.bin_centers_s, dtype=float))))
    basis = fit.basis(k)
    can_correct = bool(noise_correct and population.sem_hz is not None)
    centring = noise_centring_fraction(n_time, sigma_bins=smoothing_sigma_bins)
    adjacent_rho = float(noise_autocorrelation(1.0, sigma_bins=smoothing_sigma_bins))

    rows: list[dict] = []
    for condition in population.conditions:
        scores = np.asarray(fit.scores_by_condition[str(condition)], dtype=float)[:k, :]
        centroid = scores.mean(axis=1, keepdims=True)
        deviation = scores - centroid
        excursion_energy = float(np.mean(np.sum(deviation**2, axis=0)))
        step_energy = float(np.mean(np.sum(np.diff(scores, axis=1) ** 2, axis=0)))
        covariance = (deviation @ deviation.T) / float(max(n_time - 1, 1))

        row: dict[str, object] = {
            "region": population.region,
            "condition": str(condition),
            "n_components": k,
            "median_n_trials": float(
                np.median(population.n_trials[population.conditions.index(str(condition))])
            ),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "excursion_rms": float(np.sqrt(excursion_energy)),
            "rms_speed": float(np.sqrt(step_energy) / dt) if dt > 0 else np.nan,
            "dynamics_participation_ratio": participation_ratio(np.linalg.eigvalsh(covariance)),
        }

        if can_correct:
            noise_covariance, per_bin_energy = projected_noise_covariance(
                population, condition, basis
            )
            mean_noise = float(np.mean(per_bin_energy))
            noise_excursion = mean_noise * (1.0 - centring)
            noise_step = 2.0 * mean_noise * (1.0 - adjacent_rho)
            signal_excursion = max(excursion_energy - noise_excursion, 0.0)
            signal_step = max(step_energy - noise_step, 0.0)
            signal_covariance = covariance - noise_covariance * (1.0 - centring)
            eigenvalues = np.clip(np.linalg.eigvalsh(signal_covariance), 0.0, None)
            row.update(
                {
                    "noise_energy": noise_excursion,
                    "excursion_rms_corrected": float(np.sqrt(signal_excursion)),
                    "signal_fraction_of_excursion": (
                        signal_excursion / excursion_energy if excursion_energy > 0 else np.nan
                    ),
                    "speed_over_noise": (
                        step_energy / noise_step if noise_step > 0 else np.nan
                    ),
                    "rms_speed_corrected": (
                        float(np.sqrt(signal_step) / dt) if dt > 0 else np.nan
                    ),
                    "dynamics_participation_ratio_corrected": participation_ratio(eigenvalues),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def condition_speed_timecourse(
    population: RegionPopulation,
    fit: PopulationPCAFit,
    *,
    n_components: int,
) -> pd.DataFrame:
    """Instantaneous population speed per condition, ||dz/dt|| at each bin."""
    k = int(n_components)
    centers = np.asarray(population.bin_centers_s, dtype=float)
    dt = float(np.mean(np.diff(centers)))
    rows: list[dict] = []
    for condition in population.conditions:
        scores = np.asarray(fit.scores_by_condition[str(condition)], dtype=float)[:k, :]
        speed = np.linalg.norm(np.gradient(scores, dt, axis=1), axis=0)
        for index, center in enumerate(centers):
            rows.append(
                {
                    "region": population.region,
                    "condition": str(condition),
                    "bin_center_s": float(center),
                    "speed": float(speed[index]),
                }
            )
    return pd.DataFrame(rows)


def build_offset_vs_dynamics_table(
    populations: Mapping[str, RegionPopulation],
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    n_components: int,
    noise_correct: bool = True,
    smoothing_sigma_bins: float = DEFAULT_SMOOTHING_SIGMA_BINS,
) -> pd.DataFrame:
    """Per condition pair: how much of the separation is static, how much moves.

    Splits the mean squared separation exactly,

        <||z_a - z_b||^2>_t = ||m_a - m_b||^2 + <||d_a - d_b||^2>_t,

    into an offset term and a dynamics term (the cross term vanishes because
    the deviations are mean-zero by construction).  Reporting only the total
    conflates "the conditions sit in different places" with "the conditions
    move differently", which are separate claims.

    ``deviation_correlation`` is the shape question on its own: the normalised
    inner product <d_a, d_b>_t.  Its numerator is unbiased by estimation noise,
    because noise is independent across conditions; only the normalising
    excursion energies are inflated.  ``deviation_correlation_corrected``
    therefore disattenuates the denominator with the SEM-derived noise energies
    and is the value to read -- the uncorrected one is biased *toward zero*, and
    biased hardest for the conditions with the fewest trials, which would make
    a real shared dynamic look like an absence of one.

    All computations use the shared concatenated basis, so the split is
    expressed in one coordinate system for all three conditions.
    """
    rows: list[dict] = []
    for region, population in populations.items():
        fit = fits_by_region[region]["concatenated"]
        k = int(n_components)
        basis = fit.basis(k)
        n_time = int(population.n_time)
        centring = noise_centring_fraction(n_time, sigma_bins=smoothing_sigma_bins)
        can_correct = bool(noise_correct and population.sem_hz is not None)

        deviations: dict[str, np.ndarray] = {}
        centroids: dict[str, np.ndarray] = {}
        noise_energy: dict[str, float] = {}
        for condition in population.conditions:
            scores = np.asarray(fit.scores_by_condition[str(condition)], dtype=float)[:k, :]
            centroids[condition] = scores.mean(axis=1)
            deviations[condition] = scores - centroids[condition][:, None]
            if can_correct:
                _, per_bin = projected_noise_covariance(population, condition, basis)
                noise_energy[condition] = float(np.mean(per_bin))

        for condition_a, condition_b in combinations(population.conditions, 2):
            deviation_a = deviations[condition_a]
            deviation_b = deviations[condition_b]
            offset_square = float(np.sum((centroids[condition_a] - centroids[condition_b]) ** 2))
            dynamics_square = float(np.mean(np.sum((deviation_a - deviation_b) ** 2, axis=0)))
            total_square = offset_square + dynamics_square
            energy_a = float(np.mean(np.sum(deviation_a**2, axis=0)))
            energy_b = float(np.mean(np.sum(deviation_b**2, axis=0)))
            cross = float(np.mean(np.sum(deviation_a * deviation_b, axis=0)))
            raw_denominator = float(np.sqrt(energy_a * energy_b))

            row: dict[str, object] = {
                "region": region,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "pair_label": pair_label(condition_a, condition_b),
                "n_components": k,
                "mean_square_distance": total_square,
                "offset_square_distance": offset_square,
                "dynamics_square_distance": dynamics_square,
                "offset_share_of_separation": (
                    offset_square / total_square if total_square > 0 else np.nan
                ),
                "dynamics_share_of_separation": (
                    dynamics_square / total_square if total_square > 0 else np.nan
                ),
                "centroid_distance": float(np.sqrt(offset_square)),
                "deviation_correlation": (
                    cross / raw_denominator if raw_denominator > 1e-12 else np.nan
                ),
            }

            if can_correct:
                signal_a = max(energy_a - noise_energy[condition_a] * (1.0 - centring), 0.0)
                signal_b = max(energy_b - noise_energy[condition_b] * (1.0 - centring), 0.0)
                signal_offset = max(
                    offset_square
                    - (noise_energy[condition_a] + noise_energy[condition_b]) * centring,
                    0.0,
                )
                signal_dynamics = max(
                    dynamics_square
                    - (noise_energy[condition_a] + noise_energy[condition_b]) * (1.0 - centring),
                    0.0,
                )
                signal_total = signal_offset + signal_dynamics
                corrected_denominator = float(np.sqrt(signal_a * signal_b))
                row.update(
                    {
                        "offset_share_corrected": (
                            signal_offset / signal_total if signal_total > 0 else np.nan
                        ),
                        "dynamics_share_corrected": (
                            signal_dynamics / signal_total if signal_total > 0 else np.nan
                        ),
                        "deviation_correlation_corrected": (
                            float(np.clip(cross / corrected_denominator, -1.0, 1.0))
                            if corrected_denominator > 1e-12
                            else np.nan
                        ),
                        # Disattenuation rescales observed and null identically,
                        # so a figure can put the corrected value against a
                        # correspondingly rescaled null band without changing
                        # any p-value.
                        "disattenuation_factor": (
                            float(raw_denominator / corrected_denominator)
                            if corrected_denominator > 1e-12
                            else np.nan
                        ),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_deviation_correlation(
    populations: Mapping[str, RegionPopulation],
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    n_components: int,
    min_shift_bins: int = 10,
) -> pd.DataFrame:
    """Circular-shift test for whether two conditions move together in time.

    The null circularly shifts one condition's deviation trajectory relative to
    the other.  This is the right null for this question because it preserves
    each condition's own temporal autocorrelation and its excursion energy, and
    destroys only the *temporal correspondence* between them -- which is exactly
    what a non-zero deviation correlation claims.  A white-noise null would be
    far too permissive: two smooth, slowly varying trajectories correlate
    substantially by accident.

    Shifts smaller than ``min_shift_bins`` are excluded because a smoothed
    trajectory still overlaps itself there, so they are not valid null draws.
    The test is exact rather than sampled: every admissible shift is used.
    """
    rows: list[dict] = []
    for region, population in populations.items():
        fit = fits_by_region[region]["concatenated"]
        k = int(n_components)
        n_time = int(population.n_time)
        deviations = {
            condition: (
                lambda scores: scores - scores.mean(axis=1, keepdims=True)
            )(np.asarray(fit.scores_by_condition[str(condition)], dtype=float)[:k, :])
            for condition in population.conditions
        }
        shifts = [
            shift
            for shift in range(1, n_time)
            if min(shift, n_time - shift) >= int(min_shift_bins)
        ]
        for condition_a, condition_b in combinations(population.conditions, 2):
            deviation_a = deviations[condition_a]
            deviation_b = deviations[condition_b]
            denominator = float(
                np.sqrt(
                    np.mean(np.sum(deviation_a**2, axis=0))
                    * np.mean(np.sum(deviation_b**2, axis=0))
                )
            )
            if denominator <= 1e-12:
                continue
            observed = float(np.mean(np.sum(deviation_a * deviation_b, axis=0)) / denominator)
            null = np.asarray(
                [
                    float(
                        np.mean(np.sum(deviation_a * np.roll(deviation_b, shift, axis=1), axis=0))
                        / denominator
                    )
                    for shift in shifts
                ],
                dtype=float,
            )
            rows.append(
                {
                    "region": region,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "pair_label": pair_label(condition_a, condition_b),
                    "n_components": k,
                    "n_shifts": int(null.size),
                    "observed_deviation_correlation": observed,
                    "shift_null_mean": float(null.mean()),
                    "shift_null_median": float(np.median(null)),
                    "shift_null_sd": float(null.std(ddof=1)) if null.size > 1 else np.nan,
                    # The null is an exact enumeration of every admissible shift,
                    # so its own 2.5th-97.5th percentiles are a 95% interval
                    # without assuming it is symmetric or Gaussian.  A +/-2 SD
                    # band would impose both assumptions on a distribution that
                    # has neither.
                    "shift_null_p2p5": float(np.percentile(null, 2.5)),
                    "shift_null_p97p5": float(np.percentile(null, 97.5)),
                    "shift_null_p95": float(np.percentile(null, 95)),
                    "z_against_shift_null": (
                        float((observed - null.mean()) / null.std(ddof=1))
                        if null.size > 1 and null.std(ddof=1) > 0
                        else np.nan
                    ),
                    "p_value": float(
                        (np.count_nonzero(null >= observed) + 1) / (null.size + 1)
                    ),
                }
            )
    return pd.DataFrame(rows)


COMPONENT_ORDER: tuple[str, ...] = (
    "shared_time_course",
    "condition_offset",
    "condition_by_time",
)
COMPONENT_LABELS: dict[str, str] = {
    "shared_time_course": "Shared time course",
    "condition_offset": "Condition offset",
    "condition_by_time": "Condition x time",
}


def _decomposition_energies(population: RegionPopulation) -> dict[str, np.ndarray]:
    """Raw energy in each part of the three-way split, per neuron."""
    tensor = np.asarray(population.rates_hz, dtype=float)
    n_conditions, n_time, _ = tensor.shape
    centered = tensor - tensor.mean(axis=(0, 1), keepdims=True)
    shared = centered.mean(axis=0)
    offsets = centered.mean(axis=1)
    residual = centered - shared[None, :, :] - offsets[:, None, :]
    return {
        "shared_time_course": n_conditions * np.sum(shared**2, axis=0),
        "condition_offset": n_time * np.sum(offsets**2, axis=0),
        "condition_by_time": np.sum(residual**2, axis=(0, 1)),
    }


def build_variance_share_summary(
    populations: Mapping[str, RegionPopulation],
    *,
    n_bootstrap: int = 400,
    seed: int = 0,
    pvalue_correction: str = "fdr_bh",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share of population variance in each part of the split, with statistics.

    Energies are summed across neurons **before** normalising, so this is the
    variance-weighted share: it describes the population trajectory the PCA
    actually operates on, rather than the average neuron.  Pooling first also
    avoids the bias that per-neuron normalisation introduces once the noise
    correction is applied -- a neuron whose shared and interaction terms are
    both pure noise would have them clipped to zero and its offset share forced
    to 1, which is right for that neuron but distorts an unweighted mean.

    Both normalisations are returned.  ``raw`` is the observed split; in
    ``corrected`` the SEM-derived noise energy is removed from each part first.
    The correction is large and asymmetric, because estimation noise is
    independent across conditions and rough in time and therefore lands almost
    entirely in the interaction term.

    Uncertainty and the tests come from resampling neurons with replacement, so
    they answer whether another sample of neurons from these areas would show
    the same ordering.
    """
    rng = np.random.default_rng(int(seed))
    energies_by_region = {
        region: _decomposition_energies(population) for region, population in populations.items()
    }
    noise_by_region = {
        region: decomposition_noise_energies(population)
        for region, population in populations.items()
    }

    scopes: dict[str, list[str]] = {"all_regions": list(populations)}
    scopes.update({str(region): [str(region)] for region in populations})

    share_rows: list[dict] = []
    contrast_rows: list[dict] = []
    for scope, members in scopes.items():
        raw = {
            name: np.concatenate([energies_by_region[region][name] for region in members])
            for name in COMPONENT_ORDER
        }
        if all(noise_by_region[region] is not None for region in members):
            noise = {
                name: np.concatenate([noise_by_region[region][name] for region in members])
                for name in COMPONENT_ORDER
            }
        else:
            noise = None

        variants = {"raw": raw}
        if noise is not None:
            variants["corrected"] = {
                name: np.clip(raw[name] - noise[name], 0.0, None) for name in COMPONENT_ORDER
            }

        n_units = int(raw[COMPONENT_ORDER[0]].size)
        picks_matrix = rng.integers(0, n_units, size=(int(n_bootstrap), n_units))
        for normalization, energies in variants.items():
            total = sum(energies.values())
            observed = {
                name: float(energies[name].sum() / total.sum()) if total.sum() > 0 else np.nan
                for name in COMPONENT_ORDER
            }
            draws = {name: np.empty(int(n_bootstrap), dtype=float) for name in COMPONENT_ORDER}
            for draw in range(int(n_bootstrap)):
                picks = picks_matrix[draw]
                denominator = float(total[picks].sum())
                for name in COMPONENT_ORDER:
                    draws[name][draw] = (
                        float(energies[name][picks].sum() / denominator)
                        if denominator > 0
                        else np.nan
                    )
            for name in COMPONENT_ORDER:
                share_rows.append(
                    {
                        "scope": scope,
                        "normalization": normalization,
                        "component": name,
                        "share": observed[name],
                        "ci_low": float(np.nanpercentile(draws[name], 2.5)),
                        "ci_high": float(np.nanpercentile(draws[name], 97.5)),
                        "n_units": n_units,
                        "n_bootstrap": int(n_bootstrap),
                    }
                )
            for left, right in combinations(COMPONENT_ORDER, 2):
                difference = draws[left] - draws[right]
                observed_difference = observed[left] - observed[right]
                crossings = (
                    np.count_nonzero(difference <= 0)
                    if observed_difference > 0
                    else np.count_nonzero(difference >= 0)
                )
                contrast_rows.append(
                    {
                        "scope": scope,
                        "normalization": normalization,
                        "component_a": left,
                        "component_b": right,
                        "difference": float(observed_difference),
                        "ci_low": float(np.nanpercentile(difference, 2.5)),
                        "ci_high": float(np.nanpercentile(difference, 97.5)),
                        "p_value": float(
                            min(1.0, 2.0 * (crossings + 1) / (int(n_bootstrap) + 1))
                        ),
                    }
                )

    contrasts = apply_adjusted_pvalues(
        pd.DataFrame(contrast_rows),
        p_col="p_value",
        out_col="p_value_corrected",
        method=pvalue_correction,
        group_cols=["normalization"],
    )
    return pd.DataFrame(share_rows), contrasts


def build_pc_condition_separation(
    populations: Mapping[str, RegionPopulation],
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    n_pcs: int = 3,
    n_bootstrap: int = 400,
    seed: int = 0,
    pvalue_correction: str = "fdr_bh",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How far apart each condition pair sits along each individual component.

    For component *j* and conditions *a*, *b* the effect is the mean absolute
    difference of their score time courses, ``mean_t |Z_a[j,t] - Z_b[j,t]|``.
    The absolute value makes it immune to a component's arbitrary sign;
    averaging over time makes it one number per component and pair.

    Uncertainty comes from resampling **neurons** with replacement while holding
    the component axes fixed: each resampled score is the same weighted sum of
    firing rates, taken over the sampled neurons and renormalised.  Refitting
    the PCA on each resample would be the more obvious choice but is wrong here
    -- components with similar variance rotate into one another between
    resamples, so "PC1" would not name the same axis twice and the interval
    would mix two different quantities.  Holding the axes fixed asks the
    question actually being posed: would this population axis separate these
    conditions as well in another sample of neurons?

    Time bins are not a valid resampling unit either, since a smoothed
    trajectory's bins are heavily autocorrelated.

    Returns per-pair effects with bootstrap intervals, and contrasts between
    pairs within each region and component, corrected across all contrasts.
    """
    rng = np.random.default_rng(int(seed))
    effect_rows: list[dict] = []
    contrast_rows: list[dict] = []

    for region, population in populations.items():
        fit = fits_by_region[region]["concatenated"]
        pairs = list(combinations(population.conditions, 2))
        components = np.asarray(fit.components, dtype=float)[:n_pcs]  # (n_pcs, n_units)
        mean = np.asarray(fit.mean, dtype=float)
        centered = {
            condition: population.condition_matrix(condition) - mean[:, None]
            for condition in population.conditions
        }
        observed = {
            (pair, index): float(
                np.mean(
                    np.abs(
                        fit.scores_by_condition[pair[0]][index]
                        - fit.scores_by_condition[pair[1]][index]
                    )
                )
            )
            for pair in pairs
            for index in range(n_pcs)
        }

        draws = {key: np.empty(int(n_bootstrap), dtype=float) for key in observed}
        n_units = population.n_units
        for draw in range(int(n_bootstrap)):
            picks = rng.integers(0, n_units, size=n_units)
            weights = components[:, picks]
            norms = np.linalg.norm(weights, axis=1, keepdims=True)
            weights = weights / np.where(norms > 1e-12, norms, 1.0)
            scores = {
                condition: weights @ centered[condition][picks, :]
                for condition in population.conditions
            }
            for pair in pairs:
                difference = scores[pair[0]] - scores[pair[1]]
                for index in range(n_pcs):
                    draws[(pair, index)][draw] = float(np.mean(np.abs(difference[index])))

        for pair in pairs:
            for index in range(n_pcs):
                sample = draws[(pair, index)]
                effect_rows.append(
                    {
                        "region": region,
                        "pc_index": index + 1,
                        "condition_a": pair[0],
                        "condition_b": pair[1],
                        "pair_label": pair_label(*pair),
                        "mean_abs_difference": observed[(pair, index)],
                        "ci_low": float(np.percentile(sample, 2.5)),
                        "ci_high": float(np.percentile(sample, 97.5)),
                        "bootstrap_sd": float(np.std(sample, ddof=1)),
                        "n_bootstrap": int(n_bootstrap),
                    }
                )

        for index in range(n_pcs):
            for first, second in combinations(pairs, 2):
                difference = draws[(first, index)] - draws[(second, index)]
                observed_difference = observed[(first, index)] - observed[(second, index)]
                crossings = (
                    np.count_nonzero(difference <= 0)
                    if observed_difference > 0
                    else np.count_nonzero(difference >= 0)
                )
                contrast_rows.append(
                    {
                        "region": region,
                        "pc_index": index + 1,
                        "pair_a": pair_label(*first),
                        "pair_b": pair_label(*second),
                        "difference": observed_difference,
                        "ci_low": float(np.percentile(difference, 2.5)),
                        "ci_high": float(np.percentile(difference, 97.5)),
                        "p_value": float(min(1.0, 2.0 * (crossings + 1) / (int(n_bootstrap) + 1))),
                    }
                )

    contrasts = apply_adjusted_pvalues(
        pd.DataFrame(contrast_rows),
        p_col="p_value",
        out_col="p_value_corrected",
        method=pvalue_correction,
    )
    return pd.DataFrame(effect_rows), contrasts


def bootstrap_subspace_metrics(
    populations: Mapping[str, RegionPopulation],
    *,
    n_components: int,
    n_bootstrap: int = 200,
    subsample_fraction: float = 0.8,
    seed: int = 0,
    pvalue_correction: str = "fdr_bh",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample units to put intervals on the alignment index and principal angle.

    Uses **subsampling without replacement** rather than the usual bootstrap.
    Drawing units with replacement duplicates neurons, and duplicated rows are
    perfectly correlated, which lowers the rank of the covariance and inflates
    every subspace-overlap measure.  Subsampling a fixed fraction keeps every
    unit distinct.

    **What the reported interval is, and is not.**  Both measures depend on the
    number of neurons beyond what the k/N floor correction removes: with fewer
    neurons any two subspaces look closer, so subsamples give systematically
    higher alignment and smaller angles than the full sample.  A subsample
    therefore estimates a slightly different quantity, and a bias-correcting
    interval (reverse percentile) would extrapolate that shift in the wrong
    direction and land beside the observed value rather than around it.

    So the interval reported here is the **spread** of the resample
    distribution, transplanted onto the observed value:
    ``[value - (median - p2.5), value + (p97.5 - median)]``.  It answers "how
    much does this measure move when the sample of neurons changes", not "where
    is the population value".  The raw resample quantiles are returned alongside
    so the shift itself stays visible.

    The **contrasts are unaffected** by any of this: all three condition pairs
    are computed on the same subsample in each draw, so a shift driven by sample
    size is common to them and cancels in the difference.

    Alignment is reported floor-corrected, since the chance level k/N is what
    makes raw values incomparable across regions of different size.
    """
    rng = np.random.default_rng(int(seed))
    effect_rows: list[dict] = []
    contrast_rows: list[dict] = []

    for region, population in populations.items():
        pairs = list(combinations(population.conditions, 2))
        floor = analytic_alignment_floor(population.n_units, n_components)

        def _metrics(target: RegionPopulation) -> dict[tuple[str, str], tuple[float, float]]:
            # The chance floor is k/N of *this* sample, not of the full
            # population: a subsample has fewer neurons and therefore a higher
            # floor, and using the full-population value would make every
            # resampled alignment look inflated.
            sample_floor = analytic_alignment_floor(target.n_units, n_components)
            bases = {
                condition: fit_population_pca(target, fit_scope=condition).basis(n_components)
                for condition in target.conditions
            }
            covariances = {
                condition: condition_covariance(target, condition)
                for condition in target.conditions
            }
            out: dict[tuple[str, str], tuple[float, float]] = {}
            for first, second in pairs:
                forward = alignment_index(covariances[second], bases[first])
                backward = alignment_index(covariances[first], bases[second])
                angle = float(
                    principal_angle_metrics(bases[first], bases[second])["mean_principal_angle"]
                )
                out[(first, second)] = (
                    alignment_above_floor(0.5 * (forward + backward), sample_floor),
                    angle,
                )
            return out

        observed = _metrics(population)
        draws = {pair: {"alignment": [], "angle": []} for pair in pairs}
        n_draw = max(n_components + 1, int(round(float(subsample_fraction) * population.n_units)))
        n_draw = min(n_draw, population.n_units)
        for _ in range(int(n_bootstrap)):
            picks = rng.choice(population.n_units, size=n_draw, replace=False)
            sampled = _metrics(population.take_units(picks))
            for pair in pairs:
                draws[pair]["alignment"].append(sampled[pair][0])
                draws[pair]["angle"].append(sampled[pair][1])

        for pair in pairs:
            for metric, key in (("alignment_above_floor", "alignment"), ("mean_principal_angle", "angle")):
                sample = np.asarray(draws[pair][key], dtype=float)
                value = float(observed[pair][0 if key == "alignment" else 1])
                median = float(np.median(sample))
                low_quantile = float(np.percentile(sample, 2.5))
                high_quantile = float(np.percentile(sample, 97.5))
                # Spread of the resample distribution, centred on the observed
                # value.  See the docstring: subsampling shifts these metrics,
                # so the resamples' own location is not informative about the
                # full-sample value, but their width is informative about how
                # much a change of neurons moves the measure.
                effect_rows.append(
                    {
                        "region": region,
                        "pair_label": pair_label(*pair),
                        "metric": metric,
                        "value": value,
                        "spread_low": value - (median - low_quantile),
                        "spread_high": value + (high_quantile - median),
                        "resample_median": median,
                        "resample_p2p5": low_quantile,
                        "resample_p97p5": high_quantile,
                        "n_bootstrap": int(n_bootstrap),
                        "subsample_fraction": float(subsample_fraction),
                        "chance_floor": floor if metric == "alignment_above_floor" else np.nan,
                    }
                )
        for metric, key in (("alignment_above_floor", "alignment"), ("mean_principal_angle", "angle")):
            for first, second in combinations(pairs, 2):
                difference = np.asarray(draws[first][key], dtype=float) - np.asarray(
                    draws[second][key], dtype=float
                )
                observed_difference = (
                    observed[first][0 if key == "alignment" else 1]
                    - observed[second][0 if key == "alignment" else 1]
                )
                crossings = (
                    np.count_nonzero(difference <= 0)
                    if observed_difference > 0
                    else np.count_nonzero(difference >= 0)
                )
                contrast_rows.append(
                    {
                        "region": region,
                        "metric": metric,
                        "pair_a": pair_label(*first),
                        "pair_b": pair_label(*second),
                        "difference": float(observed_difference),
                        "p_value": float(min(1.0, 2.0 * (crossings + 1) / (difference.size + 1))),
                    }
                )

    contrasts = apply_adjusted_pvalues(
        pd.DataFrame(contrast_rows),
        p_col="p_value",
        out_col="p_value_corrected",
        method=pvalue_correction,
        group_cols=["metric"],
    )
    return pd.DataFrame(effect_rows), contrasts


def verify_subspace_offset_invariance(
    populations: Mapping[str, RegionPopulation],
    *,
    n_components: int,
) -> pd.DataFrame:
    """Show that alignment and principal angles never saw the static offsets.

    A single-condition PCA centres on that condition's own time-average, so its
    basis already describes deviations from its centroid, and
    :func:`condition_covariance` centres the same way.  Alignment indices and
    principal angles are therefore **already** pure dynamics measures: they
    cannot be inflated by the conditions sitting in different places.

    This is easy to assume and easy to get wrong, so it is checked instead:
    every metric is recomputed after explicitly subtracting each condition's
    centroid, and the difference from the uncentred computation must be zero to
    numerical precision.  Only the trajectory-based measures -- which live in
    the shared concatenated basis and are *meant* to see the offsets -- change.
    """
    rows: list[dict] = []
    for region, population in populations.items():
        deviation = marginalize_population(population, mode="remove_condition_offset")
        raw_fits = fit_all_scopes(population)
        deviation_fits = fit_all_scopes(deviation)
        for condition_a, condition_b in combinations(population.conditions, 2):
            raw_alignment = alignment_index(
                condition_covariance(population, condition_b),
                raw_fits[condition_a].basis(n_components),
            )
            deviation_alignment = alignment_index(
                condition_covariance(deviation, condition_b),
                deviation_fits[condition_a].basis(n_components),
            )
            raw_angle = float(
                principal_angle_metrics(
                    raw_fits[condition_a].basis(n_components),
                    raw_fits[condition_b].basis(n_components),
                )["mean_principal_angle"]
            )
            deviation_angle = float(
                principal_angle_metrics(
                    deviation_fits[condition_a].basis(n_components),
                    deviation_fits[condition_b].basis(n_components),
                )["mean_principal_angle"]
            )
            rows.append(
                {
                    "region": region,
                    "pair_label": pair_label(condition_a, condition_b),
                    "alignment_full": raw_alignment,
                    "alignment_centroid_removed": deviation_alignment,
                    "alignment_difference": abs(raw_alignment - deviation_alignment),
                    "mean_angle_full": raw_angle,
                    "mean_angle_centroid_removed": deviation_angle,
                    "angle_difference_deg": abs(raw_angle - deviation_angle),
                }
            )
    frame = pd.DataFrame(rows)
    frame["offset_invariant"] = (frame["alignment_difference"] < 1e-8) & (
        frame["angle_difference_deg"] < 1e-6
    )
    return frame


def decomposition_noise_energies(
    population: RegionPopulation,
    *,
    smoothing_sigma_bins: float = DEFAULT_SMOOTHING_SIGMA_BINS,
) -> Optional[dict[str, np.ndarray]]:
    """Expected noise energy landing in each part of the three-way split.

    The split is an orthogonal projection of each neuron's condition-by-time
    matrix onto three subspaces, so the expected noise in each part is the trace
    of the noise covariance through the corresponding projector.  Writing
    ``g_c`` for a condition's total noise energy and ``h_c`` for the part that
    survives averaging over time,

        shared   = (1/C) * sum_c (g_c - h_c)
        offset   = (1 - 1/C) * sum_c h_c
        residual = (1 - 1/C) * sum_c (g_c - h_c)

    This matters because the noise is **not** shared out evenly.  Estimation
    noise is independent across conditions and rough in time, so almost all of
    it lands in the residual -- exactly the term that would otherwise be read as
    "condition-specific dynamics".  Neurons with low firing rates have
    proportionally more noise, so an uncorrected per-neuron split systematically
    overstates the interaction and can reverse the ordering of the three parts.

    ``h_c`` uses the smoothing kernel's autocorrelation rather than assuming
    independent bins, for the same reason as in
    :func:`condition_dynamics_summary`.
    """
    if population.sem_hz is None:
        return None
    n_conditions = int(population.n_conditions)
    n_time = int(population.n_time)
    lags = np.abs(np.subtract.outer(np.arange(n_time), np.arange(n_time)))
    correlation = noise_autocorrelation(lags, sigma_bins=smoothing_sigma_bins)

    sem = np.asarray(population.sem_hz, dtype=float)  # (conditions, time, units)
    total_energy = np.sum(sem**2, axis=1)  # (conditions, units) = g_c
    time_mean_energy = np.einsum("ctn,tu,cun->cn", sem, correlation, sem) / float(n_time)  # h_c

    fluctuating = total_energy - time_mean_energy
    return {
        "shared_time_course": fluctuating.sum(axis=0) / n_conditions,
        "condition_offset": time_mean_energy.sum(axis=0) * (1.0 - 1.0 / n_conditions),
        "condition_by_time": fluctuating.sum(axis=0) * (1.0 - 1.0 / n_conditions),
    }


def condition_variance_decomposition(population: RegionPopulation) -> pd.DataFrame:
    """Split population variance into condition-independent and condition parts.

    A dPCA-style marginalisation.  Most of a fixation-aligned population's
    variance is usually the shared time course every condition follows; the
    fraction left over is the part any claim about "separated spaces" actually
    rests on, so it is worth reporting next to the geometry.
    """
    tensor = population.rates_hz  # (condition, time, unit)
    grand_mean = tensor.mean(axis=(0, 1), keepdims=True)
    centered = tensor - grand_mean

    time_marginal = centered.mean(axis=0, keepdims=True)  # shared time course
    condition_marginal = centered.mean(axis=1, keepdims=True)  # time-independent offset
    interaction = centered - time_marginal - condition_marginal

    total = float(np.sum(centered**2))
    parts = {
        "condition_independent_time": float(np.sum(np.broadcast_to(time_marginal, centered.shape) ** 2)),
        "condition_main_effect": float(
            np.sum(np.broadcast_to(condition_marginal, centered.shape) ** 2)
        ),
        "condition_by_time_interaction": float(np.sum(interaction**2)),
    }
    rows = [
        {
            "region": population.region,
            "component": name,
            "variance": value,
            "fraction_of_total": value / total if total > 0 else np.nan,
        }
        for name, value in parts.items()
    ]
    rows.append(
        {
            "region": population.region,
            "component": "condition_related_total",
            "variance": parts["condition_main_effect"] + parts["condition_by_time_interaction"],
            "fraction_of_total": (
                (parts["condition_main_effect"] + parts["condition_by_time_interaction"]) / total
                if total > 0
                else np.nan
            ),
        }
    )
    return pd.DataFrame(rows)


MARGINALIZATION_MODES: tuple[str, ...] = (
    "none",
    "remove_shared_time_course",
    "remove_condition_offset",
    "condition_dynamics_only",
)


def single_unit_decomposition(
    population: RegionPopulation,
    unit_index: int,
) -> dict[str, np.ndarray | float]:
    """Split one neuron's three condition curves into the three response parts.

    The population-level decomposition is easiest to read one neuron at a time,
    because for a single neuron it is just arithmetic on three curves:

    1. **Baseline** -- the neuron's overall mean rate across all conditions and
       times.  Subtract it; everything below is a deviation from it.
    2. **Shared time course** -- average the three curves together, point by
       point.  Whatever is left in that average is a response the neuron makes
       whenever the monkey fixates, regardless of what is fixated.
    3. **Condition offset** -- take each curve's own average level, a single
       number per condition.  If those three numbers differ, the neuron fires at
       different rates for different fixation types, steadily across the window.
    4. **Condition-specific wiggle** -- subtract the shared time course and the
       condition's own level from each curve.  What remains is a bump this
       condition has and the others do not.

    Returns the four pieces plus the share of the neuron's variance each carries.
    """
    index = int(unit_index)
    curves = np.asarray(population.rates_hz[:, :, index], dtype=float)  # (conditions, time)
    baseline = float(curves.mean())
    centered = curves - baseline
    shared_time = centered.mean(axis=0)  # (time,)
    offsets = centered.mean(axis=1)  # (conditions,)
    residual = centered - shared_time[None, :] - offsets[:, None]

    n_conditions, n_time = curves.shape
    energies = {
        "shared_time_course": float(n_conditions * np.sum(shared_time**2)),
        "condition_offset": float(n_time * np.sum(offsets**2)),
        "condition_specific_wiggle": float(np.sum(residual**2)),
    }
    total = float(np.sum(centered**2))
    return {
        "unit_index": index,
        "unit_key": population.unit_keys[index],
        "conditions": population.conditions,
        "bin_centers_s": np.asarray(population.bin_centers_s, dtype=float),
        "curves": curves,
        "baseline": baseline,
        "shared_time_course": shared_time,
        "condition_offsets": offsets,
        "residual": residual,
        "energies": energies,
        "shares": {name: (value / total if total > 0 else np.nan) for name, value in energies.items()},
        "total_energy": total,
    }


def select_decomposition_example_unit(population: RegionPopulation) -> int:
    """Pick a neuron in which all three response parts are visible.

    Ranks by the geometric mean of the three normalised component energies, so
    the chosen unit is one where none of the three is negligible.  A unit
    dominated by a single component would illustrate the arithmetic but not the
    point of doing it.
    """
    tensor = np.asarray(population.rates_hz, dtype=float)
    n_conditions, n_time, n_units = tensor.shape
    centered = tensor - tensor.mean(axis=(0, 1), keepdims=True)
    shared = centered.mean(axis=0)  # (time, units)
    offsets = centered.mean(axis=1)  # (conditions, units)
    residual = centered - shared[None, :, :] - offsets[:, None, :]

    energies = np.stack(
        [
            n_conditions * np.sum(shared**2, axis=0),
            n_time * np.sum(offsets**2, axis=0),
            np.sum(residual**2, axis=(0, 1)),
        ],
        axis=0,
    )
    total = energies.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.where(total > 0, energies / total, 0.0)
    score = np.exp(np.mean(np.log(np.clip(shares, 1e-9, None)), axis=0))
    # Weight lightly by absolute size so the example is not a near-silent unit.
    score = score * np.sqrt(np.clip(total, 0.0, None) / max(total.max(), 1e-12))
    return int(np.argmax(score))


def deviation_correlation_profile(
    population: RegionPopulation,
    fit: PopulationPCAFit,
    condition_a: str,
    condition_b: str,
    *,
    n_components: int,
    min_shift_bins: int = 10,
) -> dict:
    """Everything behind one deviation-correlation number, for plotting.

    Returns the two centroid-removed trajectories, the per-bin inner product
    whose time average *is* the correlation numerator, and the full set of
    circular-shift null values.  Exposing the pieces makes the measure
    inspectable rather than something the reader has to take on trust.
    """
    k = int(n_components)
    n_time = int(population.n_time)
    scores_a = np.asarray(fit.scores_by_condition[str(condition_a)], dtype=float)[:k, :]
    scores_b = np.asarray(fit.scores_by_condition[str(condition_b)], dtype=float)[:k, :]
    deviation_a = scores_a - scores_a.mean(axis=1, keepdims=True)
    deviation_b = scores_b - scores_b.mean(axis=1, keepdims=True)
    denominator = float(
        np.sqrt(
            np.mean(np.sum(deviation_a**2, axis=0)) * np.mean(np.sum(deviation_b**2, axis=0))
        )
    )
    per_bin = np.sum(deviation_a * deviation_b, axis=0) / max(denominator, 1e-12)
    shifts = [
        shift for shift in range(1, n_time) if min(shift, n_time - shift) >= int(min_shift_bins)
    ]
    null = np.asarray(
        [
            float(np.mean(np.sum(deviation_a * np.roll(deviation_b, shift, axis=1), axis=0)) / denominator)
            for shift in shifts
        ],
        dtype=float,
    )
    return {
        "region": population.region,
        "condition_a": str(condition_a),
        "condition_b": str(condition_b),
        "n_components": k,
        "bin_centers_s": np.asarray(population.bin_centers_s, dtype=float),
        "deviation_a": deviation_a,
        "deviation_b": deviation_b,
        "per_bin_product": per_bin,
        "observed": float(np.mean(per_bin)),
        "null_values": null,
        "shifts": np.asarray(shifts, dtype=int),
    }


def marginalize_population(
    population: RegionPopulation,
    *,
    mode: str = "remove_shared_time_course",
) -> RegionPopulation:
    """Strip one marginal component out of the condition tensor.

    Roughly half the variance in these populations is a static per-condition
    offset and another fifth is the fixation-locked time course every condition
    shares, which between them dominate any PCA fitted to the raw tensor.  The
    dynamics that distinguish the conditions therefore sit in the residual, and
    are effectively invisible in a plot of the leading raw PCs.  Removing a
    marginal first and refitting is what makes them visible:

    - ``remove_shared_time_course`` subtracts the across-condition mean at each
      time bin, leaving the offset plus condition-specific dynamics.
    - ``remove_condition_offset`` subtracts each condition's own time mean,
      leaving the shared time course plus condition-specific dynamics.
    - ``condition_dynamics_only`` removes both, leaving the condition-by-time
      interaction on its own.

    The returned object is a ``RegionPopulation`` like any other, so every
    metric and figure in this module applies to it unchanged.  Its values are
    residuals rather than firing rates and are no longer non-negative.
    """
    mode = str(mode)
    if mode not in MARGINALIZATION_MODES:
        raise ValueError(f"Unknown marginalization mode {mode!r}.")

    tensor = np.array(population.rates_hz, dtype=float, copy=True)
    if mode == "none":
        return population

    grand_mean = tensor.mean(axis=(0, 1), keepdims=True)
    centered = tensor - grand_mean
    if mode in ("remove_shared_time_course", "condition_dynamics_only"):
        centered = centered - centered.mean(axis=0, keepdims=True)
    if mode in ("remove_condition_offset", "condition_dynamics_only"):
        centered = centered - centered.mean(axis=1, keepdims=True)

    return RegionPopulation(
        region=population.region,
        conditions=population.conditions,
        unit_keys=population.unit_keys,
        bin_centers_s=population.bin_centers_s,
        rates_hz=centered,
        n_trials=population.n_trials,
        unit_table=population.unit_table,
        sem_hz=population.sem_hz,
    )


def shuffle_condition_labels_per_unit(
    population: RegionPopulation,
    *,
    seed: int = 0,
) -> RegionPopulation:
    """Permute condition labels independently for each unit.

    Each unit keeps all three of its real traces, so single-unit temporal
    structure, firing-rate scale and noise are all preserved; only the
    population-level agreement about *which* trace is which condition is
    destroyed.  That makes it the right null for "are these subspaces more
    different than chance", because chance here means "no coordinated
    condition structure", not "no structure at all".
    """
    rng = np.random.default_rng(int(seed))
    tensor = population.rates_hz.copy()
    trials = population.n_trials.copy()
    for unit in range(population.n_units):
        order = rng.permutation(population.n_conditions)
        tensor[:, :, unit] = tensor[order, :, unit]
        trials[:, unit] = trials[order, unit]
    return RegionPopulation(
        region=population.region,
        conditions=population.conditions,
        unit_keys=population.unit_keys,
        bin_centers_s=population.bin_centers_s,
        rates_hz=tensor,
        n_trials=trials,
        unit_table=population.unit_table,
        sem_hz=population.sem_hz,
    )


def build_pairwise_subspace_table(
    population: RegionPopulation,
    fits: Mapping[str, PopulationPCAFit],
    *,
    n_components: int,
    n_null_subspaces: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Every subspace comparison for one region, one row per ordered pair.

    Rows are ordered pairs rather than unordered ones because the
    variance-based measures are asymmetric: how well interactive-face PCs
    explain object activity is a different question from the reverse, and
    collapsing the two hides which condition has the more general subspace.
    """
    rows: list[dict] = []
    covariances = {
        condition: condition_covariance(population, condition)
        for condition in population.conditions
    }
    bases = {
        condition: fits[str(condition)].basis(n_components) for condition in population.conditions
    }
    nulls = {
        condition: random_subspace_alignment_null(
            covariances[condition],
            n_components=n_components,
            n_samples=n_null_subspaces,
            seed=seed + index,
        )
        for index, condition in enumerate(population.conditions)
    }
    shared_fit = fits["concatenated"]

    for source in population.conditions:
        for target in population.conditions:
            covariance = covariances[target]
            basis = bases[source]
            total_variance = float(np.trace(covariance))
            captured = variance_captured(covariance, basis)
            null = nulls[target]
            index_value = alignment_index(covariance, basis)

            floor = analytic_alignment_floor(population.n_units, n_components)
            row: dict[str, object] = {
                "region": population.region,
                "n_units": int(population.n_units),
                "n_components": int(n_components),
                "pc_condition": str(source),
                "eval_condition": str(target),
                "is_within_condition": source == target,
                "variance_captured": captured,
                "variance_total": total_variance,
                "variance_explained_fraction": (
                    captured / total_variance if total_variance > 0 else np.nan
                ),
                "alignment_index": index_value,
                "alignment_null_mean": float(np.mean(null)),
                "alignment_null_p95": float(np.percentile(null, 95)),
                "alignment_floor_analytic": floor,
                # The comparable-across-regions version: 0 is chance, 1 is a
                # perfect match with the condition's own optimal subspace.
                "alignment_above_floor": alignment_above_floor(index_value, floor),
                "alignment_above_null": bool(index_value > np.percentile(null, 95)),
            }
            if source != target:
                row.update(
                    {
                        f"angle_{key}": value
                        for key, value in principal_angle_metrics(
                            bases[source], bases[target]
                        ).items()
                        if key != "principal_angles"
                    }
                )
                row.update(
                    trajectory_geometry(
                        shared_fit.scores_by_condition[str(source)],
                        shared_fit.scores_by_condition[str(target)],
                        n_components=n_components,
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_null_comparison(
    population: RegionPopulation,
    *,
    n_components: int,
    n_shuffles: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Observed subspace similarity against a condition-label shuffle.

    The shuffle permutes condition labels independently within each unit, so
    every neuron keeps all three of its real traces and only the population's
    agreement about which trace is which condition is destroyed.

    Note which way this null points.  It does **not** test whether the
    conditions are separated; a shuffled population has three subspaces that
    are, if anything, *more* different from each other than the real ones,
    because the real conditions share a large common fixation-locked response.
    What it tests is whether the observed subspaces are more *similar* than
    coordinated-structure-free data would give -- i.e. whether the shared
    subspace is real.  Both tail probabilities are reported so neither
    direction has to be inferred from the sign of a difference.
    """
    observed_fits = {
        condition: fit_population_pca(population, fit_scope=condition)
        for condition in population.conditions
    }
    observed_bases = {
        condition: fit.basis(n_components) for condition, fit in observed_fits.items()
    }
    covariances = {
        condition: condition_covariance(population, condition)
        for condition in population.conditions
    }

    null_angles: dict[tuple[str, str], list[float]] = {}
    null_alignments: dict[tuple[str, str], list[float]] = {}
    for shuffle_index in range(int(n_shuffles)):
        shuffled = shuffle_condition_labels_per_unit(population, seed=seed + shuffle_index)
        shuffled_bases = {
            condition: fit_population_pca(shuffled, fit_scope=condition).basis(n_components)
            for condition in shuffled.conditions
        }
        shuffled_covariances = {
            condition: condition_covariance(shuffled, condition)
            for condition in shuffled.conditions
        }
        for condition_a, condition_b in combinations(population.conditions, 2):
            key = (condition_a, condition_b)
            metrics = principal_angle_metrics(
                shuffled_bases[condition_a], shuffled_bases[condition_b]
            )
            null_angles.setdefault(key, []).append(float(metrics["mean_principal_angle"]))
            null_alignments.setdefault(key, []).append(
                float(
                    alignment_index(shuffled_covariances[condition_b], shuffled_bases[condition_a])
                )
            )

    rows: list[dict] = []
    for condition_a, condition_b in combinations(population.conditions, 2):
        key = (condition_a, condition_b)
        metrics = principal_angle_metrics(observed_bases[condition_a], observed_bases[condition_b])
        observed_angle = float(metrics["mean_principal_angle"])
        observed_alignment = float(
            alignment_index(covariances[condition_b], observed_bases[condition_a])
        )
        angle_null = np.asarray(null_angles[key], dtype=float)
        alignment_null = np.asarray(null_alignments[key], dtype=float)
        rows.append(
            {
                "region": population.region,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "pair_label": pair_label(condition_a, condition_b),
                "n_components": int(n_components),
                "n_shuffles": int(n_shuffles),
                "observed_mean_principal_angle": observed_angle,
                "shuffled_mean_principal_angle": float(angle_null.mean()),
                "shuffled_angle_sd": float(angle_null.std(ddof=1)) if angle_null.size > 1 else np.nan,
                "p_angle_larger_than_shuffle": float(
                    (np.count_nonzero(angle_null >= observed_angle) + 1) / (angle_null.size + 1)
                ),
                "p_angle_smaller_than_shuffle": float(
                    (np.count_nonzero(angle_null <= observed_angle) + 1) / (angle_null.size + 1)
                ),
                "observed_alignment_index": observed_alignment,
                "shuffled_alignment_index": float(alignment_null.mean()),
                "shuffled_alignment_sd": (
                    float(alignment_null.std(ddof=1)) if alignment_null.size > 1 else np.nan
                ),
                "p_alignment_higher_than_shuffle": float(
                    (np.count_nonzero(alignment_null >= observed_alignment) + 1)
                    / (alignment_null.size + 1)
                ),
                "p_alignment_lower_than_shuffle": float(
                    (np.count_nonzero(alignment_null <= observed_alignment) + 1)
                    / (alignment_null.size + 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def time_resolved_separation(
    population: RegionPopulation,
    fit: PopulationPCAFit,
    *,
    n_components: int,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Condition separation at each time bin, with unit-bootstrap intervals.

    With trial-averaged input there are no trials left to resample, so the
    interval comes from resampling *units*: it answers "would another sample of
    neurons from this region show the same separation", which is the
    generalisation a population claim actually needs.  It is not a test that
    the separation exceeds noise -- :func:`build_pairwise_null_comparison`
    supplies that.
    """
    rng = np.random.default_rng(int(seed))
    k = int(n_components)
    pairs = list(combinations(population.conditions, 2))
    observed = {
        pair: np.linalg.norm(
            fit.scores_by_condition[pair[0]][:k, :] - fit.scores_by_condition[pair[1]][:k, :],
            axis=0,
        )
        for pair in pairs
    }
    scale = float(
        np.mean(
            [
                np.linalg.norm(fit.scores_by_condition[condition][:k, :], axis=0).mean()
                for condition in population.conditions
            ]
        )
    )

    # One refit per bootstrap draw, reused across all three pairs; refitting per
    # pair would triple the cost for identical numbers.
    samples = {pair: np.empty((int(n_bootstrap), population.n_time), dtype=float) for pair in pairs}
    for index in range(int(n_bootstrap)):
        picks = rng.integers(0, population.n_units, size=population.n_units)
        resampled_fit = fit_population_pca(population.take_units(picks), fit_scope="concatenated")
        k_draw = int(min(k, resampled_fit.n_components))
        for pair in pairs:
            samples[pair][index] = np.linalg.norm(
                resampled_fit.scores_by_condition[pair[0]][:k_draw, :]
                - resampled_fit.scores_by_condition[pair[1]][:k_draw, :],
                axis=0,
            )

    rows: list[dict] = []
    for pair in pairs:
        low, high = np.percentile(samples[pair], [2.5, 97.5], axis=0)
        spread = samples[pair].std(axis=0, ddof=1)
        for bin_index, center in enumerate(population.bin_centers_s):
            rows.append(
                {
                    "region": population.region,
                    "condition_a": pair[0],
                    "condition_b": pair[1],
                    "pair_label": pair_label(*pair),
                    "bin_center_s": float(center),
                    "distance": float(observed[pair][bin_index]),
                    "distance_normalized": (
                        float(observed[pair][bin_index] / scale) if scale > 0 else np.nan
                    ),
                    "ci_low": float(low[bin_index]),
                    "ci_high": float(high[bin_index]),
                    "bootstrap_sd": float(spread[bin_index]),
                }
            )
    return pd.DataFrame(rows)


def split_half_geometry_reliability(
    population: RegionPopulation,
    *,
    n_components: int,
    n_splits: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Does the condition geometry replicate in an independent half of the units?

    Units are split into two disjoint halves and the whole pipeline is rerun in
    each.  Because the two halves span different unit spaces their PC bases are
    not comparable, but the *geometry* they imply is: the relative distances
    among the three conditions, and the relative alignment of their subspaces.
    Agreement across halves is the strongest internal-validity check available
    from condition-averaged data, where no held-out trials exist.
    """
    rng = np.random.default_rng(int(seed))
    pairs = list(combinations(population.conditions, 2))
    rows: list[dict] = []

    for split in range(int(n_splits)):
        order = rng.permutation(population.n_units)
        halves = (order[: population.n_units // 2], order[population.n_units // 2 :])
        summaries = []
        for half in halves:
            half_population = population.take_units(half)
            half_fit = fit_population_pca(half_population, fit_scope="concatenated")
            k = int(min(n_components, half_fit.n_components))
            scores = {
                condition: half_fit.scores_by_condition[condition][:k, :]
                for condition in population.conditions
            }
            # Normalising by the population's own overall excursion makes the
            # two halves comparable despite having different unit counts and so
            # different absolute variance.
            scale = float(np.mean([np.linalg.norm(value, axis=0).mean() for value in scores.values()]))
            summaries.append(
                {
                    pair: float(np.mean(np.linalg.norm(scores[pair[0]] - scores[pair[1]], axis=0)) / scale)
                    if scale > 0
                    else np.nan
                    for pair in pairs
                }
            )
        first = np.asarray([summaries[0][pair] for pair in pairs], dtype=float)
        second = np.asarray([summaries[1][pair] for pair in pairs], dtype=float)
        for pair_index, pair in enumerate(pairs):
            rows.append(
                {
                    "region": population.region,
                    "split": int(split),
                    "pair_label": pair_label(*pair),
                    "half_a_normalized_distance": float(first[pair_index]),
                    "half_b_normalized_distance": float(second[pair_index]),
                }
            )
    return pd.DataFrame(rows)


def summarize_split_half_reliability(reliability: pd.DataFrame) -> pd.DataFrame:
    """Collapse split-half draws into per-region agreement statistics.

    Reports two different rank statistics because they can disagree in an
    informative way.  ``pair_ordering_agreement`` requires all three pairs to
    rank identically in both halves (chance 1/6); it fails whenever two pairs are
    nearly tied, even if the pattern is otherwise stable.
    ``closest_pair_agreement`` asks only whether the two halves pick the same
    *closest* pair (chance 1/3), which is the statement the conclusions actually
    rest on.  A region with low ordering agreement but high closest-pair
    agreement has a stable geometry with one near-tie in it, not an unstable one.
    """
    rows: list[dict] = []
    for region, region_frame in reliability.groupby("region"):
        ordering_agreement: list[float] = []
        closest_agreement: list[float] = []
        closest_choices: list[str] = []
        for _, split_frame in region_frame.groupby("split"):
            ordered = split_frame.sort_values("pair_label")
            labels = ordered["pair_label"].to_numpy()
            first = ordered["half_a_normalized_distance"].to_numpy()
            second = ordered["half_b_normalized_distance"].to_numpy()
            ordering_agreement.append(float(np.array_equal(np.argsort(first), np.argsort(second))))
            closest_first = labels[int(np.argmin(first))]
            closest_second = labels[int(np.argmin(second))]
            closest_agreement.append(float(closest_first == closest_second))
            closest_choices.extend([closest_first, closest_second])
        pooled = np.corrcoef(
            region_frame["half_a_normalized_distance"].to_numpy(),
            region_frame["half_b_normalized_distance"].to_numpy(),
        )[0, 1]
        modal = pd.Series(closest_choices).value_counts()
        row = {
            "region": str(region),
            "n_splits": int(region_frame["split"].nunique()),
            "split_half_distance_correlation": float(pooled),
            "pair_ordering_agreement": float(np.mean(ordering_agreement)),
            "closest_pair_agreement": float(np.mean(closest_agreement)),
            "modal_closest_pair": str(modal.index[0]) if len(modal) else "",
            "modal_closest_pair_frequency": (
                float(modal.iloc[0] / modal.sum()) if len(modal) else np.nan
            ),
        }
        for pair, pair_frame in region_frame.groupby("pair_label"):
            values = np.concatenate(
                [
                    pair_frame["half_a_normalized_distance"].to_numpy(),
                    pair_frame["half_b_normalized_distance"].to_numpy(),
                ]
            )
            row[f"mean_{pair}"] = float(np.nanmean(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)


def bootstrap_metric_over_units(
    population: RegionPopulation,
    metric_fn: Callable[[RegionPopulation], float],
    *,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap a scalar population metric by resampling units."""
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        picks = rng.integers(0, population.n_units, size=population.n_units)
        values[index] = float(metric_fn(population.take_units(picks)))
    finite = values[np.isfinite(values)]
    return {
        "observed": float(metric_fn(population)),
        "bootstrap_mean": float(finite.mean()) if finite.size else np.nan,
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else np.nan,
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else np.nan,
        "n_bootstrap": int(finite.size),
    }


def build_region_subspace_summary(
    populations: Mapping[str, RegionPopulation],
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    n_components: int,
    n_null_subspaces: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Concatenate every region's pairwise subspace table."""
    frames = [
        build_pairwise_subspace_table(
            populations[region],
            fits_by_region[region],
            n_components=n_components,
            n_null_subspaces=n_null_subspaces,
            seed=seed,
        )
        for region in populations
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolve_output_dir(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    scope: str = "all_units",
) -> Path:
    """Analysis-output directory for one unit scope, created on demand."""
    cfg = load_config(cfg_path)
    path = build_analysis_output_dir(cfg, output_subdir) / str(scope)
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "CONDITION_ORDER",
    "CONDITION_PAIRS",
    "DEFAULT_VARIANCE_THRESHOLD",
    "PopulationPCAFit",
    "REGION_ORDER",
    "RegionPopulation",
    "alignment_above_floor",
    "alignment_index",
    "analytic_alignment_floor",
    "bootstrap_metric_over_units",
    "build_dimensionality_table",
    "build_pairwise_null_comparison",
    "build_pairwise_subspace_table",
    "build_region_subspace_summary",
    "build_unit_inventory",
    "condition_covariance",
    "condition_identity_confusion",
    "bootstrap_subspace_metrics",
    "build_pc_condition_separation",
    "build_variance_share_summary",
    "build_offset_vs_dynamics_table",
    "DEFAULT_SMOOTHING_SIGMA_BINS",
    "condition_dynamics_summary",
    "noise_autocorrelation",
    "noise_centring_fraction",
    "projected_noise_covariance",
    "condition_speed_timecourse",
    "condition_variance_decomposition",
    "deviation_correlation_profile",
    "select_decomposition_example_unit",
    "single_unit_decomposition",
    "cross_condition_variance_curve",
    "fit_all_scopes",
    "fit_population_pca",
    "load_pair_selective_units",
    "load_region_populations",
    "MARGINALIZATION_MODES",
    "marginalize_population",
    "n_components_for_variance",
    "pair_label",
    "participation_ratio",
    "principal_angle_metrics",
    "random_subspace_alignment_null",
    "resolve_output_dir",
    "resolve_shared_n_components",
    "shuffle_condition_labels_per_unit",
    "split_half_geometry_reliability",
    "summarize_split_half_reliability",
    "summarize_selective_units",
    "time_resolved_separation",
    "test_deviation_correlation",
    "trajectory_geometry",
    "variance_captured",
    "within_condition_alignment_ceiling",
    "verify_against_stored_pca",
    "verify_pca_identities",
    "verify_subspace_offset_invariance",
]
