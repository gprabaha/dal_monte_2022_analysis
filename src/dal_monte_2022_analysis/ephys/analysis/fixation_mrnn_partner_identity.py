"""The partner-identity acid test: does a region need *this* partner, or any partner?

The ladder (task 01) compares a region fitted alone against a region fitted with the
network attached, and finds the network helps -- but a "pair" network has more parameters
than a "single" network, so part of the gain could be generic extra capacity rather than
anything specific to the partner region. This module builds the matched control: a
**self-pair** duplicates one region into two blocks that both read its *entire* neuron
population (same PCA, bit-identical target, independently trained weights per block), so it
has exactly the architecture and parameter count of a **cross-pair** (two different areas,
each also fitted on its own full population) -- the only thing that differs is whether the
partner's target carries that area's actual information or a second copy of the scored
region's own.

Every region's PCA is fitted on all of its neurons -- there is no unit-splitting here (an
earlier version of this module split each region into two independent halves so a self-pair
would not be a literal duplicate; see the "legacy half-split grid" section below for why that
was abandoned and what it is kept for). Fitting a self-pair as a literal duplicate trades one
confound for another: a block can partly fit its target by mirroring its partner block rather
than by using its own recurrence, so "self beats cross" is weaker evidence against partner
identity mattering than it would be under independent halves, and "cross beats self" is a
correspondingly *stronger* finding. Say so wherever the self-vs-cross contrast is read.

Four regions give ten unique two-block networks: four self-pairs and six cross-pairs. Each
network is scored on both of its blocks, filling a 4x4 matrix of (scored region, partner)
cells. If a region reproduces itself equally well paired with a duplicate of itself as paired
with a real different area, the ladder's "the network helps" result is mostly capacity; if
cross-pairs reliably beat self-pairs *despite* the self-pair's mirroring advantage, the
partner's identity matters.

Everything downstream of target construction is the untouched mRNN pipeline: a self- or
cross-pair's virtual/canonical region names are ordinary "regions" as far as
:class:`FixationMRNNModel`, :func:`train_one_initialization`, :func:`replay_fixation_mrnn_run`
and :func:`score_variant_fit` are concerned, because those only ever iterate ``region_order``
as arbitrary strings. The one place a duplicated region is not already a first-class citizen
is :func:`build_mrnn_training_dataframe`, which hard-casts its ``region`` column to a category
set fixed at the four canonical names; the ``region_name_mapping``/``allowed_regions``
parameters added there and threaded through :func:`build_fixation_mrnn_targets_from_dataframe`
are what let a dataframe with a region duplicated under two virtual labels through unchanged
for everyone else. A cross-pair needs none of this -- its two blocks are just a two-element
subset of the canonical ``region_order``, already supported by
:func:`build_fixation_mrnn_targets_from_dataframe` for the ladder's own pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_bridge import (
    MRNN_REGION_ORDER,
    load_combined_fixation_psth,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_targets import (
    build_fixation_mrnn_targets,
    build_fixation_mrnn_targets_from_dataframe,
    serialize_pca_metadata,
)

#: Suffixes a duplicated region's two blocks carry (``"<region>_1"``, ``"<region>_2"``).
DUPLICATE_TAGS: tuple[str, str] = ("1", "2")

CONDITIONS: tuple[str, ...] = ("face_interactive", "face_non_interactive", "object")

#: The recipe's own PCA/normalisation settings (mirrors ``BASE_SETTINGS`` in
#: ``train_fixation_mrnn_partner_identity_cell.py``): every region's target is built the same
#: way regardless of which architecture uses it, so there is exactly one PCA basis per
#: canonical region and the ceiling can be computed from it directly, without a checkpoint.
TARGET_BUILD_SETTINGS: dict[str, object] = dict(
    input_subdir="ephys/psth/fixation_psth_averages",
    dataframe_filename="fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl",
    timeline_filename="fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl",
    normalize_targets=True,
    normalization_stabilizer=5.0,
    pca_variance_threshold=0.95,
    pca_n_components=42,
    temporal_basis_count=0,
)


def virtual_region_label(region: str, tag: str) -> str:
    return f"{region}_{tag}"


# ======================================================================================
# 1. The grid: four self-pairs, six cross-pairs
# ======================================================================================


@dataclass(frozen=True)
class PartnerIdentityArchitecture:
    """One of the ten unique two-block networks."""

    label: str
    arm: str  # "self" or "cross"
    scored_region: str
    partner_region: str

    def virtual_regions(self) -> tuple[str, str]:
        """``(scored, partner)`` region names as they appear in ``region_order``.

        A self-pair reads as (``<region>_1``, ``<region>_2``) -- both blocks carry the full
        population of the *same* canonical region. A cross-pair reads as the two canonical
        region names directly; no relabelling is needed since its two blocks are just a
        two-element subset of the recorded regions.
        """
        if self.arm == "self":
            return (virtual_region_label(self.scored_region, DUPLICATE_TAGS[0]),
                    virtual_region_label(self.scored_region, DUPLICATE_TAGS[1]))
        return (self.scored_region, self.partner_region)


def partner_identity_architectures(region_order: Sequence[str] = MRNN_REGION_ORDER) -> list[PartnerIdentityArchitecture]:
    """Four self-pairs and six cross-pairs -- the ten unique networks the grid trains."""
    regions = tuple(region_order)
    architectures = [
        PartnerIdentityArchitecture(label=f"self_{region}", arm="self", scored_region=region, partner_region=region)
        for region in regions
    ]
    architectures += [
        PartnerIdentityArchitecture(label=f"cross_{a}_{b}", arm="cross", scored_region=a, partner_region=b)
        for a, b in combinations(regions, 2)
    ]
    return architectures


def scored_cells(architectures: Sequence[PartnerIdentityArchitecture]) -> list[tuple[str, str, str]]:
    """The 4x4 matrix as ``(architecture_label, scored_region, partner_region)`` -- 16 cells from 10 fits.

    A self-pair architecture contributes one diagonal cell; a cross-pair contributes two
    off-diagonal cells (the same trained network, read twice, once for each block).
    """
    cells: list[tuple[str, str, str]] = []
    for arch in architectures:
        cells.append((arch.label, arch.scored_region, arch.partner_region))
        if arch.arm == "cross":
            cells.append((arch.label, arch.partner_region, arch.scored_region))
    return cells


__all__ = [
    "CONDITIONS", "DUPLICATE_TAGS", "PartnerIdentityArchitecture", "TARGET_BUILD_SETTINGS",
    "partner_identity_architectures", "scored_cells", "virtual_region_label",
]


# ======================================================================================
# 2. Full-neuron target construction: duplicate for self-pairs, plain subset for cross-pairs
# ======================================================================================


def duplicate_region_dataframe(
    combined_dataframe: pd.DataFrame, region: str, *, tags: tuple[str, str] = DUPLICATE_TAGS,
) -> pd.DataFrame:
    """Replace one region's rows with two full-population copies, labelled ``<region>_<tag>``.

    Every unit of ``region`` appears twice, once under each virtual label, bit-identical --
    this is what makes a self-pair's two blocks carry exactly the same information a
    cross-pair's two (different) regions would, modulo which region it is. Every other
    region's rows are left untouched, so the pooled normalisation scale
    (:func:`build_fixation_mrnn_targets_from_dataframe`) still sees all four recorded
    regions, just as it does for a cross-pair or the ladder's own pairs.
    """
    out = combined_dataframe.copy()
    out["region"] = out["region"].astype(str)
    region_rows = out.loc[out["region"] == region]
    if region_rows.empty:
        raise ValueError(f"No rows found for region {region!r}.")
    duplicates = []
    for tag in tags:
        copy = region_rows.copy()
        copy["region"] = virtual_region_label(region, tag)
        duplicates.append(copy)
    remainder = out.loc[out["region"] != region]
    return pd.concat([remainder, *duplicates], ignore_index=True)


def duplicate_region_mapping(
    region: str, *, region_order: Sequence[str] = MRNN_REGION_ORDER, tags: tuple[str, str] = DUPLICATE_TAGS,
) -> dict[str, str]:
    """The identity mapping ``build_mrnn_training_dataframe`` needs for a duplicated dataframe."""
    mapping = {str(r): str(r) for r in region_order if r != region}
    mapping.update({virtual_region_label(region, tag): virtual_region_label(region, tag) for tag in tags})
    return mapping


def duplicate_region_allowed(
    region: str, *, region_order: Sequence[str] = MRNN_REGION_ORDER, tags: tuple[str, str] = DUPLICATE_TAGS,
) -> tuple[str, ...]:
    """The widened category set ``build_mrnn_training_dataframe`` needs for a duplicated dataframe."""
    others = tuple(str(r) for r in region_order if r != region)
    return others + tuple(virtual_region_label(region, tag) for tag in tags)


def build_architecture_targets(
    architecture: PartnerIdentityArchitecture,
    combined_dataframe: pd.DataFrame,
    *,
    timeline_s: np.ndarray,
    region_order: Sequence[str] = MRNN_REGION_ORDER,
    normalization_scale_override: float | None = None,
):
    """Full-neuron :class:`FixationMRNNTargets` for one architecture's two blocks.

    Self-pairs train on a duplicated dataframe (:func:`duplicate_region_dataframe`);
    cross-pairs train directly on the canonical dataframe, naming their two regions as a
    subset of ``region_order`` -- the same mechanism the ladder's own pairs already use.

    Pass ``normalization_scale_override`` (see
    :func:`build_fixation_mrnn_targets_from_dataframe`) -- typically
    ``full_population_targets(...).normalization_scale`` -- so a self-pair's duplicated
    dataframe and a cross-pair's plain one land on the same scale rather than each pooling
    its own, which would otherwise differ because a self-pair's scored region contributes
    twice to a freshly-pooled scale and a cross-pair's does not.
    """
    virtual = architecture.virtual_regions()
    build_kwargs = {k: v for k, v in TARGET_BUILD_SETTINGS.items()
                    if k not in ("input_subdir", "dataframe_filename", "timeline_filename")}
    build_kwargs["normalization_scale_override"] = normalization_scale_override
    if architecture.arm == "self":
        region = architecture.scored_region
        dataframe = duplicate_region_dataframe(combined_dataframe, region)
        return build_fixation_mrnn_targets_from_dataframe(
            dataframe, timeline_s=timeline_s, region_order=virtual,
            region_name_mapping=duplicate_region_mapping(region, region_order=region_order),
            allowed_regions=duplicate_region_allowed(region, region_order=region_order),
            **build_kwargs,
        )
    return build_fixation_mrnn_targets_from_dataframe(
        combined_dataframe, timeline_s=timeline_s, region_order=virtual, **build_kwargs,
    )


__all__ += [
    "build_architecture_targets", "duplicate_region_allowed", "duplicate_region_dataframe",
    "duplicate_region_mapping",
]


# ======================================================================================
# 3. Run directories, job commands, and run-state inventory
# ======================================================================================

TRAIN_SCRIPT = "scripts/ephys/modeling/train_fixation_mrnn_partner_identity_cell.py"


def run_dir_for(
    root: str | Path, architecture: PartnerIdentityArchitecture, seed: int, *,
    bottleneck_tag: str | None = None, retry_tag: str | None = None,
) -> Path:
    """The dense grid's layout is ``root/label/seed=``; a bottleneck variant or a
    convergence retry nests under its own tag (``root/<tag>/label/seed=``) so neither ever
    collides with the dense checkpoints or with each other.
    """
    base = Path(root)
    if bottleneck_tag is not None:
        base = base / bottleneck_tag
    if retry_tag is not None:
        base = base / retry_tag
    return base / architecture.label / f"seed={int(seed)}"


def job_commands(
    architectures: Sequence[PartnerIdentityArchitecture],
    seeds: Sequence[int],
    *,
    root: str | Path,
    repo_root: str | Path,
    epochs: int = 100_000,
    conda_env: str = "gaze_processing",
    exclude_run_dirs: Sequence[str | Path] = (),
    within_region_bottleneck_dim: int | None = None,
    recurrent_bottleneck_dim: int | None = None,
    bottleneck_tag: str | None = None,
    lr: float | None = None,
    retry_tag: str | None = None,
    dataset_cfg: str = "configs/dataset.yaml",
) -> tuple[list[str], list[Path]]:
    """One shell command per missing (architecture, seed) cell.

    Every architecture builds its own target directly from the canonical combined PSTH
    export named by ``dataset_cfg`` (no pre-written per-seed dataset, since nothing about
    the target depends on the training seed any more) -- the training script builds that
    cell's target itself rather than through the generic
    ``FixationMRNNRunSettings``-driven loader, so no ``run_config.yaml`` is written here --
    the architecture and seed fully determine the run, and are recorded in
    ``run_dir/architecture.json`` by the training script itself.

    ``within_region_bottleneck_dim``/``recurrent_bottleneck_dim`` default to ``None`` (the
    dense, unconstrained grid); ``lr`` overrides the recipe's learning rate for a targeted
    convergence retry and should be paired with a ``retry_tag`` (e.g. ``"retry_lr1em4"``) so
    the retried cells land in their own subtree instead of overwriting the first attempt.
    """
    repo_root = Path(repo_root)
    claimed = {str(Path(path).resolve()) for path in exclude_run_dirs}
    commands: list[str] = []
    run_dirs: list[Path] = []
    for architecture in architectures:
        for seed in seeds:
            run_dir = run_dir_for(root, architecture, seed, bottleneck_tag=bottleneck_tag, retry_tag=retry_tag)
            run_dirs.append(run_dir)
            if (run_dir / "checkpoint_best.pth").exists() or str(run_dir.resolve()) in claimed:
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                "conda", "run", "-n", conda_env, "python", str(repo_root / TRAIN_SCRIPT),
                "--dataset-cfg", str(dataset_cfg),
                "--scored-region", architecture.scored_region,
                "--partner-region", architecture.partner_region,
                "--arm", architecture.arm,
                "--label", architecture.label,
                "--run-dir", str(run_dir),
                "--seed", str(int(seed)),
                "--epochs", str(int(epochs)),
            ]
            if within_region_bottleneck_dim is not None:
                command += ["--within-region-bottleneck-dim", str(int(within_region_bottleneck_dim))]
            if recurrent_bottleneck_dim is not None:
                command += ["--recurrent-bottleneck-dim", str(int(recurrent_bottleneck_dim))]
            if lr is not None:
                command += ["--lr", str(float(lr))]
            commands.append(" ".join(command))
    return commands, run_dirs


def index_runs(
    root: str | Path, architectures: Sequence[PartnerIdentityArchitecture], seeds: Sequence[int],
    *, bottleneck_tag: str | None = None, retry_tag: str | None = None,
) -> pd.DataFrame:
    """One row per (architecture, seed) cell: whether it has a best checkpoint yet."""
    rows = []
    for architecture in architectures:
        for seed in seeds:
            run_dir = run_dir_for(root, architecture, seed, bottleneck_tag=bottleneck_tag, retry_tag=retry_tag)
            rows.append({
                "label": architecture.label, "arm": architecture.arm, "scored_region": architecture.scored_region,
                "partner_region": architecture.partner_region, "seed": str(int(seed)), "run_dir": str(run_dir),
                "complete": (run_dir / "checkpoint_best.pth").exists(),
            })
    return pd.DataFrame(rows)


__all__ += ["TRAIN_SCRIPT", "index_runs", "job_commands", "run_dir_for"]


# ======================================================================================
# 4. The matched ceiling -- one per canonical region, no checkpoint required
# ======================================================================================


def full_population_targets(
    dataset_cfg_path: str | Path, *, region_order: Sequence[str] = MRNN_REGION_ORDER,
):
    """The canonical 4-region targets the recipe uses: PCA fitted on every neuron of every region.

    Deterministic given the recipe's own settings (:data:`TARGET_BUILD_SETTINGS`), so this
    needs no trained checkpoint -- unlike the half-split grid's matched ceiling, which had to
    read one back to learn that seed's halves. One call here gives every architecture's
    ceiling at once, since a self-pair's two blocks and any cross-pair containing a region
    all share this same PCA basis for it.
    """
    return build_fixation_mrnn_targets(dataset_cfg_path, region_order=tuple(region_order), **TARGET_BUILD_SETTINGS)


def full_population_ceiling_by_cell(
    dataset_cfg_path: str | Path,
    unit_ceiling: pd.DataFrame,
    *,
    region_order: Sequence[str] = MRNN_REGION_ORDER,
    conditions: Sequence[str] = CONDITIONS,
) -> pd.DataFrame:
    """Per-(canonical region, condition) ceiling, in the recipe's own full-population PCA basis."""
    from dal_monte_2022_analysis.ephys.analysis.fixation_psth_noise_ceiling import population_ceiling_by_cell

    targets = full_population_targets(dataset_cfg_path, region_order=region_order)
    serialized = serialize_pca_metadata(targets.pca_by_region)
    return population_ceiling_by_cell(
        unit_ceiling, pca_by_region=serialized, normalization_scale=targets.normalization_scale,
        conditions=list(conditions),
    )


def ceiling_lookup(
    ceiling_by_cell: pd.DataFrame, architectures: Sequence[PartnerIdentityArchitecture],
) -> dict[tuple[str, str], float]:
    """``{(region_as_it_appears_in_a_fit, condition): reliability}``, ready for ``score_variant_fit``.

    A cross-pair's blocks are already canonical region names, so the base table covers them
    directly; a self-pair's two blocks (``<region>_1``, ``<region>_2``) are added by copying
    their canonical region's value, since a duplicate shares that region's PCA basis exactly.
    """
    block = ceiling_by_cell[ceiling_by_cell["condition"] != "all"]
    base = {(str(r), str(c)): float(v) for r, c, v in block[["region", "condition", "reliability"]].itertuples(index=False)}
    lookup = dict(base)
    for arch in architectures:
        if arch.arm != "self":
            continue
        for virtual in arch.virtual_regions():
            for (region, condition), value in base.items():
                if region == arch.scored_region:
                    lookup[(virtual, condition)] = value
    return lookup


__all__ += ["ceiling_lookup", "full_population_ceiling_by_cell", "full_population_targets"]


# ======================================================================================
# 5. Reading the 4x4 matrix and its statistics
# ======================================================================================


def matrix_fit_table(fit: pd.DataFrame, architectures: Sequence[PartnerIdentityArchitecture]) -> pd.DataFrame:
    """``fit`` (one row per region x condition x fit, as ``score_variant_fit`` returns) collapsed onto the 16 scored cells.

    Adds ``scored_region``/``partner_region``/``arm`` by looking the fit's ``label`` up in
    ``architectures``. A cross-pair's network produces both directions :func:`scored_cells`
    enumerates (region A read against region B's network and vice versa) directly under
    their canonical names; a self-pair's two blocks (``<region>_1``, ``<region>_2``) both
    fold onto the one diagonal cell, scoring both of what the network computes for free
    rather than discarding half of it.
    """
    by_label = {a.label: a for a in architectures}
    rows = []
    for label, block in fit.groupby("label"):
        arch = by_label[str(label)]
        if arch.arm == "self":
            cell_regions = [(arch.scored_region, v, arch.scored_region, arch.scored_region)
                            for v in arch.virtual_regions()]
        else:
            scored, partner = arch.virtual_regions()
            cell_regions = [
                (arch.scored_region, scored, arch.scored_region, arch.partner_region),
                (arch.partner_region, partner, arch.partner_region, arch.scored_region),
            ]
        for _, virtual, scored, partner in cell_regions:
            cell = block[block["region"] == virtual].copy()
            cell["scored_region"] = scored
            cell["partner_region"] = partner
            cell["arm"] = arch.arm
            cell["architecture"] = label
            rows.append(cell)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def self_vs_cross_per_seed(matrix_fit: pd.DataFrame, *, value: str = "r2_vs_ceiling") -> pd.DataFrame:
    """Per region and seed: the self-pair score and the mean of the three cross-pair scores, regions/conditions pooled.

    One row per (region, seed), ready for a paired test (self vs. cross, paired by seed).
    """
    pooled = matrix_fit.groupby(["scored_region", "partner_region", "arm", "seed"])[value].mean().reset_index()
    self_scores = pooled[pooled["arm"] == "self"].rename(columns={value: "self"}).drop(columns=["partner_region", "arm"])
    cross_scores = (pooled[pooled["arm"] == "cross"].groupby(["scored_region", "seed"])[value].mean()
                    .reset_index().rename(columns={value: "cross"}))
    return self_scores.merge(cross_scores, on=["scored_region", "seed"], how="inner")


__all__ += ["matrix_fit_table", "self_vs_cross_per_seed"]


# ======================================================================================
# 6. Convergence and adequacy: making sure every cell is actually fitting well
# ======================================================================================


def adequacy_table(matrix_fit: pd.DataFrame, *, tolerance_multiple: float = 2.0) -> pd.DataFrame:
    """Per (architecture, arm): the worst region x condition cell against the ceiling, and whether it clears it.

    Mirrors ``fixation_mrnn_sweep.adequacy_table``'s logic -- adequacy is judged on the
    **worst cell**, not the mean, and the tolerance is the fits' own seed-to-seed spread
    rather than a number chosen by hand -- but there is no single external "dense baseline"
    label here the way there is in the ladder's rung sweep, since every architecture in this
    grid is expected to sit near ceiling on its own terms. The tolerance is instead the
    median seed-spread across all ten architectures.

    Read this **before** §4 onward: a cell that converged (no loss spikes, final near best)
    but still sits well below ceiling on its worst condition is telling you the constraint
    (or here, the architecture) genuinely limits the fit, not that the optimiser struggled --
    the opposite failure a :func:`convergence_table` entry would catch.
    """
    per_seed = matrix_fit.groupby(["architecture", "arm", "seed", "condition"])["r2_vs_ceiling"].mean().reset_index()
    worst = per_seed.groupby(["architecture", "arm", "seed"])["r2_vs_ceiling"].min().reset_index()
    spread = float(worst.groupby("architecture")["r2_vs_ceiling"].std().median())
    tolerance = tolerance_multiple * (spread if np.isfinite(spread) else 0.0)
    summary = (worst.groupby(["architecture", "arm"])["r2_vs_ceiling"]
               .agg(worst_condition="mean", seed_spread="std").reset_index())
    summary["tolerance"] = tolerance
    summary["adequate"] = summary["worst_condition"] >= 1.0 - tolerance
    return summary


#: Learning rates tried, in order, for an architecture that fails the convergence bar at the
#: inherited rate (same ladder ``fixation_mrnn_sweep`` uses, reused here so a retry needs no
#: separate tuning history of its own).
RETRY_LEARNING_RATE_LADDER: tuple[float, ...] = (3e-4, 1e-4, 3e-5)


def retry_tag_for_lr(lr: float) -> str:
    return f"retry_lr{lr:g}".replace(".", "p").replace("-", "m")


def retry_commands_for_nonconverged(
    architectures: Sequence[PartnerIdentityArchitecture],
    convergence: pd.DataFrame,
    seeds: Sequence[int],
    *,
    root: str | Path,
    repo_root: str | Path,
    epochs: int,
    inherited_lr: float = 3e-4,
    ladder: Sequence[float] = RETRY_LEARNING_RATE_LADDER,
    exclude_run_dirs: Sequence[str | Path] = (),
) -> tuple[list[str], list[Path]]:
    """Re-fit architectures that failed the convergence bar, at the next lower learning rate.

    ``convergence`` is a ``fixation_mrnn_sweep.convergence_table`` keyed by ``label``. Only
    architectures with ``n_converged < n_seeds`` are retried; they land under their own
    ``retry_lr<rate>`` subtree so the first (possibly non-converged) attempt is never
    overwritten and both remain inspectable.
    """
    failed_labels = set(convergence.loc[convergence["n_converged"] < convergence["n_seeds"], "label"])
    lower = [rate for rate in ladder if float(rate) < float(inherited_lr)]
    if not lower or not failed_labels:
        return [], []
    rate = lower[0]
    retry_architectures = [a for a in architectures if a.label in failed_labels]
    return job_commands(
        retry_architectures, seeds, root=root, repo_root=repo_root, epochs=epochs,
        exclude_run_dirs=exclude_run_dirs, lr=rate, retry_tag=retry_tag_for_lr(rate),
    )


__all__ += ["RETRY_LEARNING_RATE_LADDER", "adequacy_table", "retry_commands_for_nonconverged", "retry_tag_for_lr"]


# ======================================================================================
# 6b. Convergence speed: does a cross-pair not just fit as well, but get there faster?
# ======================================================================================


def load_histories_by_seed(inventory: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """``{label: {seed: history_frame}}`` for every completed cell with a ``history.csv``.

    Keeps the seed tag explicit, unlike ``fixation_mrnn_sweep.load_histories``'s positional
    per-label list -- that matters here because the self-vs-cross contrasts below are paired
    *by seed*, and a cell can go missing for one architecture without the others, which would
    silently misalign a positional list.
    """
    histories: dict[str, dict[str, pd.DataFrame]] = {}
    for _, run in inventory[inventory["complete"].astype(bool)].iterrows():
        path = Path(run["run_dir"]) / "history.csv"
        if not path.exists():
            continue
        histories.setdefault(str(run["label"]), {})[str(run["seed"])] = pd.read_csv(path)
    return histories


def iterations_to_tolerance(history: pd.DataFrame, *, column: str = "worst_cell_unexplained", tolerance: float = 1.1) -> dict[str, float]:
    """The first iteration at which ``column`` falls within ``tolerance`` of its own run-best.

    Two fits can converge to the same final adequacy while one gets there in a fraction of
    the iterations -- not visible in the final-cell fit alone, only in the trajectory. Scored
    on ``worst_cell_unexplained`` by default, the same per-iteration quantity the minimax
    objective optimises, so "converged" here means the same thing the objective means by it.
    """
    values = history[column].to_numpy(dtype=float)
    iterations = history["iteration"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return {"iterations_to_tolerance": float("nan"), "best_value": float("nan"), "n_iterations": float(len(values))}
    best = float(np.nanmin(values[finite]))
    target = best * float(tolerance) if best > 0 else best
    hit = np.flatnonzero(finite & (values <= target))
    reached = int(iterations[hit[0]]) if hit.size else int(iterations[-1])
    return {"iterations_to_tolerance": float(reached), "best_value": best, "n_iterations": float(len(values))}


def convergence_speed_table(
    histories_by_seed: Mapping[str, Mapping[str, pd.DataFrame]],
    architectures: Sequence[PartnerIdentityArchitecture],
    *,
    column: str = "worst_cell_unexplained",
    tolerance: float = 1.1,
) -> pd.DataFrame:
    """One row per (architecture, seed): how many iterations until the fit stopped improving.

    ``histories_by_seed`` is :func:`load_histories_by_seed`'s output. Pass the result straight
    to ``fixation_mrnn_chapter.paired_contrasts`` (``group="arm"``, ``unit="seed"``) to ask
    whether cross-pairs reach their own adequate fit in fewer iterations than self-pairs do,
    paired by seed -- "fits equally well" and "gets there equally fast" are different claims,
    and only the first is what §4 tests.
    """
    by_label = {a.label: a for a in architectures}
    rows: list[dict[str, object]] = []
    for label, by_seed in histories_by_seed.items():
        arch = by_label.get(str(label))
        if arch is None:
            continue
        for seed, history in by_seed.items():
            scored = iterations_to_tolerance(history, column=column, tolerance=tolerance)
            rows.append({
                "label": label, "arm": arch.arm, "scored_region": arch.scored_region,
                "partner_region": arch.partner_region, "seed": str(seed), **scored,
            })
    return pd.DataFrame(rows)


__all__ += ["convergence_speed_table", "iterations_to_tolerance", "load_histories_by_seed"]


# ======================================================================================
# 7. Legacy half-split grid (kept for the already-trained runs under the original scheme)
# ======================================================================================
#
# Everything below built the earlier design, where every region's units were split into two
# independent, non-overlapping halves so a self-pair would not be a literal duplicate. It is
# kept only because checkpoints already exist under it (``01b_partner_identity``'s original
# tree) and because the deferred rank-bottleneck probe (notebook sections 8-10 of the earlier
# build) reads ``halves.json`` for its region-pair size-asymmetry table; neither is used by
# the current full-neuron grid. See the module docstring for why the design changed.

HALVES: tuple[str, str] = ("a", "b")
RELABELLED_INPUT_ROOT = "ephys/psth/fixation_psth_averages_partner_identity"


def sample_region_halves(unit_ids: Sequence[str], *, seed: int, region_index: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A deterministic, non-overlapping 50/50 split of ``unit_ids``. Legacy half-split grid only."""
    ordered = sorted(str(u) for u in unit_ids)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(region_index)]))
    shuffled = [ordered[i] for i in rng.permutation(len(ordered))]
    cut = -(-len(shuffled) // 2)  # ceil(n / 2)
    return tuple(sorted(shuffled[:cut])), tuple(sorted(shuffled[cut:]))


def relabel_combined_dataframe(
    combined_dataframe: pd.DataFrame,
    *,
    region_order: Sequence[str] = MRNN_REGION_ORDER,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, tuple[str, ...]]]]:
    """A copy of the combined PSTH dataframe with every unit's ``region`` replaced by its virtual half.

    Legacy half-split grid only -- see section 7's header.
    """
    out = combined_dataframe.copy()
    out["region"] = out["region"].astype(str)
    halves: dict[str, dict[str, tuple[str, ...]]] = {}
    for index, region in enumerate(region_order):
        region_mask = out["region"] == region
        unit_ids = out.loc[region_mask, "unit_uuid"].astype(str).unique()
        half_a, half_b = sample_region_halves(unit_ids, seed=seed, region_index=index)
        assert not (set(half_a) & set(half_b)), f"{region}: halves overlap"
        assert set(half_a) | set(half_b) == set(str(u) for u in unit_ids), f"{region}: a unit was dropped"
        halves[region] = {"a": half_a, "b": half_b}
        is_a = region_mask & out["unit_uuid"].astype(str).isin(half_a)
        is_b = region_mask & out["unit_uuid"].astype(str).isin(half_b)
        out.loc[is_a, "region"] = f"{region}_a"
        out.loc[is_b, "region"] = f"{region}_b"
    return out, halves


def write_relabelled_dataset(
    *,
    dataset_cfg_path: str | Path,
    output_root: str | Path,
    seed: int,
    region_order: Sequence[str] = MRNN_REGION_ORDER,
    source_input_subdir: str = "ephys/psth/fixation_psth_averages",
    source_dataframe_filename: str = "fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl",
    source_timeline_filename: str = "fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl",
) -> dict[str, object]:
    """Write one seed's relabelled combined-PSTH pickle (plus its timeline and a halves manifest) to disk.

    Legacy half-split grid only -- see section 7's header.
    """
    import json
    import shutil

    result = load_combined_fixation_psth(
        dataset_cfg_path, input_subdir=source_input_subdir,
        dataframe_filename=source_dataframe_filename, timeline_filename=source_timeline_filename,
    )
    relabelled, halves = relabel_combined_dataframe(result.dataframe, region_order=region_order, seed=seed)

    seed_dir = Path(output_root) / f"seed={int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    dataframe_path = seed_dir / Path(source_dataframe_filename).name
    timeline_path = seed_dir / Path(source_timeline_filename).name
    relabelled.to_pickle(dataframe_path)
    shutil.copyfile(result.timeline_path, timeline_path)
    manifest_path = seed_dir / "halves.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({region: {half: list(ids) for half, ids in sides.items()} for region, sides in halves.items()},
                  handle, indent=1)
    return {
        "seed": int(seed), "seed_dir": seed_dir, "dataframe_path": dataframe_path,
        "timeline_path": timeline_path, "manifest_path": manifest_path, "n_units": {
            region: {half: len(ids) for half, ids in sides.items()} for region, sides in halves.items()
        },
    }


def ceiling_by_cell_for_seed(
    run_dir: str | Path,
    unit_ceiling: pd.DataFrame,
    *,
    region_order: Sequence[str] = MRNN_REGION_ORDER,
    conditions: Sequence[str] = CONDITIONS,
    device: str = "cpu",
) -> pd.DataFrame:
    """The matched per-(virtual region, condition) ceiling for one seed of the legacy half-split grid."""
    from dal_monte_2022_analysis.ephys.analysis.fixation_psth_noise_ceiling import population_ceiling_by_cell
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import replay_fixation_mrnn_run
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_targets import PCAMetadata

    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    checkpoint = replay["checkpoint"]
    pca_by_region: Mapping[str, PCAMetadata] = checkpoint["pca_by_region"]
    normalization_scale = checkpoint.get("normalization_scale")

    frames = []
    for region in region_order:
        for half in HALVES:
            virtual = f"{region}_{half}"
            if virtual not in pca_by_region:
                continue
            meta = pca_by_region[virtual]
            serialized = {"mean": meta.mean, "components": meta.components,
                         "explained_variance_ratio": meta.explained_variance_ratio,
                         "n_components_required": meta.n_components_required,
                         "source_features": list(meta.source_features)} if isinstance(meta, PCAMetadata) else meta
            per_cell = population_ceiling_by_cell(
                unit_ceiling, pca_by_region={region: serialized}, normalization_scale=normalization_scale,
                conditions=list(conditions),
            )
            per_cell["region"] = virtual
            frames.append(per_cell)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def region_pair_asymmetry(
    relabelled_root: str | Path, seeds: Sequence[int], *, region_order: Sequence[str] = MRNN_REGION_ORDER,
) -> pd.DataFrame:
    """Every region pair's mean half size and size ratio, read from each seed's ``halves.json``.

    Legacy half-split grid only -- see section 7's header. Used by the deferred
    rank-bottleneck probe, which asks whether a tight cross-region channel costs more for
    size-mismatched pairs; that question has no counterpart in the full-neuron grid, where
    every self-pair is trivially size-matched (a region against all of itself) rather than
    against an independent half.
    """
    import json
    from itertools import combinations as _combinations

    half_sizes: dict[str, list[float]] = {region: [] for region in region_order}
    for seed in seeds:
        halves = json.loads((Path(relabelled_root) / f"seed={int(seed)}" / "halves.json").read_text())
        for region in region_order:
            sides = halves[region]
            half_sizes[region].append((len(sides["a"]) + len(sides["b"])) / 2)
    mean_half = {region: float(np.mean(sizes)) for region, sizes in half_sizes.items()}

    rows = [{"region_a": region, "region_b": region, "half_size_a": mean_half[region],
            "half_size_b": mean_half[region], "size_ratio": 1.0} for region in region_order]
    for a, b in _combinations(sorted(region_order), 2):
        ratio = max(mean_half[a], mean_half[b]) / min(mean_half[a], mean_half[b])
        rows.append({"region_a": a, "region_b": b, "half_size_a": mean_half[a],
                    "half_size_b": mean_half[b], "size_ratio": ratio})
    return pd.DataFrame(rows).sort_values("size_ratio").reset_index(drop=True)


__all__ += [
    "HALVES", "RELABELLED_INPUT_ROOT", "ceiling_by_cell_for_seed", "region_pair_asymmetry",
    "relabel_combined_dataframe", "sample_region_halves", "write_relabelled_dataset",
]
