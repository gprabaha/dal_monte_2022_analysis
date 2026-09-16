"""The partner-identity acid test: does a region need *this* partner, or any partner?

The ladder (task 01) compares a region fitted alone against a region fitted with the
network attached, and finds the network helps -- but a "pair" network has more parameters
than a "single" network, so part of the gain could be generic extra capacity rather than
anything specific to the partner region. This module builds the matched control: every
region's units are split into two independent, non-overlapping halves ("A" and "B"), each
refit with its own PCA, so a **self-pair** (region A with region B of the *same* area) has
exactly the same architecture and parameter count as a **cross-pair** (region A of one area
with region A of another) -- the only thing that differs is whether the partner's target
carries that area's actual information or an independent draw of the same area's own.

Four regions give ten unique two-block networks: four self-pairs and six cross-pairs. Each
network is scored on both of its blocks, filling a 4x4 matrix of (scored region, partner)
cells. If a region reproduces itself equally well paired with itself as paired with a real
different area, the ladder's "the network helps" result is mostly capacity; if cross-pairs
reliably beat self-pairs, the partner's identity matters.

Everything downstream of target construction is the untouched mRNN pipeline: the halves are
ordinary "regions" as far as :class:`FixationMRNNModel`, :func:`train_one_initialization`,
:func:`replay_fixation_mrnn_run` and :func:`score_variant_fit` are concerned, because those
only ever iterate ``region_order`` as arbitrary strings. The one place a virtual region is
not already a first-class citizen is :func:`build_mrnn_training_dataframe`, which hard-casts
its ``region`` column to a category set fixed at the four canonical names; the
``region_name_mapping``/``allowed_regions`` parameters added there and threaded through
:func:`build_fixation_mrnn_targets_from_dataframe` are what let a relabelled dataframe
through unchanged for everyone else.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_bridge import (
    CombinedFixationPSTHLoadResult,
    MRNN_REGION_ORDER,
    load_combined_fixation_psth,
    resolve_combined_fixation_psth_paths,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_targets import (
    PCAMetadata,
    build_fixation_mrnn_targets_from_dataframe,
    serialize_pca_metadata,
)

HALVES: tuple[str, str] = ("a", "b")


def virtual_region_label(region: str, half: str) -> str:
    return f"{region}_{half}"


def virtual_region_labels(region_order: Sequence[str] = MRNN_REGION_ORDER) -> tuple[str, ...]:
    """Every virtual region a relabelled dataframe carries: two halves per canonical region."""
    return tuple(virtual_region_label(region, half) for region in region_order for half in HALVES)


def region_name_mapping(region_order: Sequence[str] = MRNN_REGION_ORDER) -> dict[str, str]:
    """The identity mapping ``build_mrnn_training_dataframe`` needs to accept virtual labels."""
    return {label: label for label in virtual_region_labels(region_order)}


# ======================================================================================
# 1. The grid: four self-pairs, six cross-pairs
# ======================================================================================


@dataclass(frozen=True)
class PartnerIdentityArchitecture:
    """One of the ten unique two-block networks, before a seed's halves are chosen."""

    label: str
    arm: str  # "self" or "cross"
    scored_region: str
    partner_region: str

    def virtual_regions(self) -> tuple[str, str]:
        """``(scored, partner)`` virtual labels: a self-pair always reads as (A, B)."""
        if self.arm == "self":
            return (virtual_region_label(self.scored_region, "a"), virtual_region_label(self.scored_region, "b"))
        return (virtual_region_label(self.scored_region, "a"), virtual_region_label(self.partner_region, "a"))


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


# ======================================================================================
# 2. Splitting each region's units into two independent halves
# ======================================================================================


def sample_region_halves(unit_ids: Sequence[str], *, seed: int, region_index: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A deterministic, non-overlapping 50/50 split of ``unit_ids``.

    ``region_index`` (this region's position in ``region_order``) enters the seed sequence
    alongside ``seed`` so that every region gets an independent shuffle even though they
    share the same top-level seed -- without it, every region would draw the identical
    permutation pattern relative to its own sorted unit list. Odd counts put the extra unit
    in half A; sizes therefore differ by at most one, and every unit lands in exactly one
    half (verified by the caller).
    """
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

    Returns ``(relabelled_dataframe, halves)`` where ``halves[region] = {"a": unit_ids, "b": unit_ids}``
    records exactly which units went into which half, for provenance and for computing the
    matched noise ceiling later. Rows for regions outside ``region_order`` are left alone;
    every unit of a region *in* ``region_order`` is relabelled into exactly one virtual half,
    so the pooled-normalisation-scale guarantee the ladder relies on (a region's target is
    identical wherever it appears) carries over unit-for-unit to the virtual regions.
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
        out.loc[is_a, "region"] = virtual_region_label(region, "a")
        out.loc[is_b, "region"] = virtual_region_label(region, "b")
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

    ``output_root / f"seed={seed}"`` becomes an ``input_subdir`` a normal
    :class:`FixationMRNNRunSettings` can point at: it contains a dataframe with the same
    schema as the canonical export, just with eight virtual regions in place of the four
    canonical ones, plus the timeline (copied verbatim) and ``halves.json`` for provenance.
    Idempotent: re-running with the same seed reproduces the same split and overwrites the
    same files.
    """
    import json

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
    import shutil

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


__all__ = [
    "HALVES", "PartnerIdentityArchitecture", "partner_identity_architectures", "region_name_mapping",
    "relabel_combined_dataframe", "sample_region_halves", "scored_cells", "virtual_region_label",
    "virtual_region_labels", "write_relabelled_dataset",
]


# ======================================================================================
# 3. Run directories, job commands, and run-state inventory
# ======================================================================================

TRAIN_SCRIPT = "scripts/ephys/modeling/train_fixation_mrnn_partner_identity_cell.py"
RELABELLED_INPUT_ROOT = "ephys/psth/fixation_psth_averages_partner_identity"


def run_dir_for(root: str | Path, architecture: PartnerIdentityArchitecture, seed: int) -> Path:
    return Path(root) / architecture.label / f"seed={int(seed)}"


def job_commands(
    architectures: Sequence[PartnerIdentityArchitecture],
    seeds: Sequence[int],
    *,
    root: str | Path,
    repo_root: str | Path,
    epochs: int = 100_000,
    conda_env: str = "gaze_processing",
    exclude_run_dirs: Sequence[str | Path] = (),
) -> tuple[list[str], list[Path]]:
    """One shell command per missing (architecture, seed) cell; mirrors ``sweep.variant_job_commands``' shape.

    Every cell for a given seed reads that seed's already-written relabelled dataframe
    (``write_relabelled_dataset`` must have been called for every seed first); the training
    script builds that cell's target directly rather than through the generic
    ``FixationMRNNRunSettings``-driven loader, so no ``run_config.yaml`` is written here --
    the architecture and seed fully determine the run, and are recorded in
    ``run_dir/architecture.json`` by the training script itself.
    """
    repo_root = Path(repo_root)
    claimed = {str(Path(path).resolve()) for path in exclude_run_dirs}
    commands: list[str] = []
    run_dirs: list[Path] = []
    for architecture in architectures:
        for seed in seeds:
            run_dir = run_dir_for(root, architecture, seed)
            run_dirs.append(run_dir)
            if (run_dir / "checkpoint_best.pth").exists() or str(run_dir.resolve()) in claimed:
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            input_subdir = f"{RELABELLED_INPUT_ROOT}/seed={int(seed)}"
            commands.append(
                " ".join([
                    "conda", "run", "-n", conda_env, "python", str(repo_root / TRAIN_SCRIPT),
                    "--relabelled-input-subdir", input_subdir,
                    "--scored-region", architecture.scored_region,
                    "--partner-region", architecture.partner_region,
                    "--arm", architecture.arm,
                    "--label", architecture.label,
                    "--run-dir", str(run_dir),
                    "--seed", str(int(seed)),
                    "--epochs", str(int(epochs)),
                ])
            )
    return commands, run_dirs


def index_runs(root: str | Path, architectures: Sequence[PartnerIdentityArchitecture], seeds: Sequence[int]) -> pd.DataFrame:
    """One row per (architecture, seed) cell: whether it has a best checkpoint yet."""
    rows = []
    for architecture in architectures:
        for seed in seeds:
            run_dir = run_dir_for(root, architecture, seed)
            rows.append({
                "label": architecture.label, "arm": architecture.arm, "scored_region": architecture.scored_region,
                "partner_region": architecture.partner_region, "seed": str(int(seed)), "run_dir": str(run_dir),
                "complete": (run_dir / "checkpoint_best.pth").exists(),
            })
    return pd.DataFrame(rows)


__all__ += ["RELABELLED_INPUT_ROOT", "TRAIN_SCRIPT", "index_runs", "job_commands", "run_dir_for"]


# ======================================================================================
# 4. The matched ceiling, one per virtual region per seed
# ======================================================================================


def ceiling_by_cell_for_seed(
    run_dir: str | Path,
    unit_ceiling: pd.DataFrame,
    *,
    region_order: Sequence[str] = MRNN_REGION_ORDER,
    conditions: Sequence[str] = ("face_interactive", "face_non_interactive", "object"),
    device: str = "cpu",
) -> pd.DataFrame:
    """The matched per-(virtual region, condition) ceiling for one seed, read from any one of its checkpoints.

    A virtual region's PCA basis and normalisation scale are identical wherever it
    appears within a seed (the same guarantee the ladder relies on for its four canonical
    regions), so one checkpoint that happens to contain a given virtual region is enough to
    score it -- there is no need to touch every architecture. ``population_ceiling_by_cell``
    is called once per canonical region with the matching half's ``PCAMetadata``, keyed by
    the *canonical* name (the only name ``unit_ceiling`` itself carries), and the returned
    rows are relabelled to the virtual name before being combined.
    """
    from dal_monte_2022_analysis.ephys.analysis.fixation_psth_noise_ceiling import population_ceiling_by_cell
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import replay_fixation_mrnn_run

    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    checkpoint = replay["checkpoint"]
    pca_by_region: Mapping[str, PCAMetadata] = checkpoint["pca_by_region"]
    normalization_scale = checkpoint.get("normalization_scale")

    frames = []
    for region in region_order:
        for half in HALVES:
            virtual = virtual_region_label(region, half)
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


def ceiling_lookup(ceiling_cell: pd.DataFrame) -> dict[tuple[str, str], float]:
    """``{(virtual_region, condition): reliability}`` for the non-pooled rows, ready for ``score_variant_fit``."""
    block = ceiling_cell[ceiling_cell["condition"] != "all"]
    return {(str(r), str(c)): float(v) for r, c, v in
            block[["region", "condition", "reliability"]].itertuples(index=False)}


__all__ += ["ceiling_by_cell_for_seed", "ceiling_lookup"]


# ======================================================================================
# 5. Reading the 4x4 matrix and its statistics
# ======================================================================================


def matrix_fit_table(fit: pd.DataFrame, architectures: Sequence[PartnerIdentityArchitecture]) -> pd.DataFrame:
    """``fit`` (one row per region x condition x fit, as ``score_variant_fit`` returns) collapsed onto the 16 scored cells.

    Adds ``scored_region``/``partner_region``/``arm`` by looking the fit's ``label`` up in
    ``architectures`` and reading *both* blocks a trained network produced -- for a
    cross-pair these are the two directions :func:`scored_cells` enumerates (region A read
    against region B's network and vice versa); for a self-pair they are the two
    independent halves' own reconstructions (A's block and B's block), which the network
    computes for free but which :func:`PartnerIdentityArchitecture.virtual_regions` only
    ever names one of ("A" as scored, "B" as partner). Scoring both halves gives the
    self arm the same "read the whole trained network" treatment the cross arm already
    gets, rather than discarding half of what was already computed.
    """
    by_label = {a.label: a for a in architectures}
    rows = []
    for label, block in fit.groupby("label"):
        arch = by_label[str(label)]
        virtual_scored, virtual_partner = arch.virtual_regions()
        cell_regions = [
            (arch.scored_region, arch.partner_region, virtual_scored),
            (arch.partner_region, arch.scored_region, virtual_partner),
        ]
        for scored, partner, virtual in cell_regions:
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
