"""Cross-run synthesis of the fixation mRNN fits.

Every fitted mRNN lives under ``<analysis_output_root>/ephys/modeling/fixation_mrnn/scratch``
as a run directory carrying ``checkpoint_final.pth`` (with the exact ``settings`` and
``model_spec`` it was trained with), ``history.csv`` and ``manifest.json``. The functions
here read that tree and turn it into tidy tables: what was fitted, how well, how
consistently across random initializations, and what about the fitted dynamics is
invariant.

Nothing here refits. Everything is a re-analysis of the stored checkpoints, so the whole
module runs on CPU in a couple of minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
    extract_fixation_latent_dynamics,
    extract_region_current_vectors,
    load_fixation_mrnn_checkpoint,
    pc_reconstructed_firing_rate_accuracy,
    reconstruction_accuracy,
    replay_fixation_mrnn_run,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir

DEFAULT_MODELING_SUBDIR = "ephys/modeling/fixation_mrnn"
REGION_ORDER: tuple[str, ...] = ("ofc", "bla", "dmpfc", "accg")
CONDITION_ORDER: tuple[str, ...] = ("face_interactive", "face_non_interactive", "object")

#: Prefix -> family label. Order matters: the first matching prefix wins.
RUN_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("smoke_", "smoke test"),
    ("codex_nan_check", "smoke test"),
    ("hidden_unit_sweep_", "hidden-unit sweep"),
    ("loss_sweep_", "loss sweep"),
    ("loss_necessity_", "loss necessity"),
    ("pc_plus_fr_loss_", "loss sweep"),
    ("cca_42pc_tanh_", "architecture comparison"),
    ("bottleneck1d_", "bottleneck ensemble"),
    ("bottleneck3d_", "bottleneck ensemble"),
    ("bottleneck_dim_", "bottleneck sweep"),
    ("interregional_current_", "current geometry"),
    ("ensemble_", "seed ensemble"),
)

#: Settings fields that make two runs incomparable if they differ.
COMPARABILITY_FIELDS: tuple[str, ...] = (
    "epochs",
    "hidden_units",
    "activation",
    "lr",
    "l1_weight_scale",
    "temporal_derivative_loss_scale",
    "temporal_curvature_loss_scale",
    "recurrent_bottleneck_dim",
    "recurrent_connectivity",
    "pca_n_components",
)


def resolve_scratch_root(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    modeling_subdir: str = DEFAULT_MODELING_SUBDIR,
) -> Path:
    """Directory holding every fitted mRNN run."""
    cfg = load_config(cfg_path)
    return build_analysis_output_dir(cfg, modeling_subdir) / "scratch"


def resolve_output_dir(
    cfg_path: str | Path = "configs/dataset.yaml",
    *,
    modeling_subdir: str = DEFAULT_MODELING_SUBDIR,
    scope: str = "synthesis",
) -> Path:
    """Analysis-output directory for the synthesis figures and tables."""
    cfg = load_config(cfg_path)
    path = build_analysis_output_dir(cfg, modeling_subdir) / "plots" / str(scope)
    path.mkdir(parents=True, exist_ok=True)
    return path


def classify_run_family(run_name: str) -> str:
    """Map a run directory name onto the experiment family it belongs to."""
    name = str(run_name)
    for prefix, family in RUN_FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family
    return "other"


def _settings_mapping(checkpoint: Mapping[str, object]) -> dict[str, object]:
    settings = checkpoint.get("settings")
    if settings is None:
        return {}
    return dict(settings) if isinstance(settings, Mapping) else dict(vars(settings))


def _iter_run_dirs(scratch_root: Path) -> Iterable[Path]:
    """Yield every directory holding a final checkpoint, including ensemble members."""
    for path in sorted(scratch_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "checkpoint_final.pth").exists():
            yield path
            continue
        for member in sorted(path.glob("init=*")):
            if (member / "checkpoint_final.pth").exists():
                yield member


def index_fixation_mrnn_runs(scratch_root: str | Path) -> pd.DataFrame:
    """One row per fitted run: family, the settings it was trained with, and final loss.

    The settings come from the checkpoint, not from the current config file, so this is
    the *trained* configuration rather than a re-derivation of it.
    """
    root = Path(scratch_root)
    rows: list[dict[str, object]] = []
    for run_dir in _iter_run_dirs(root):
        checkpoint = torch.load(run_dir / "checkpoint_final.pth", map_location="cpu", weights_only=False)
        settings = _settings_mapping(checkpoint)
        relative = run_dir.relative_to(root)
        top_level = relative.parts[0]
        history_path = run_dir / "history.csv"
        final_loss = np.nan
        best_loss = np.nan
        final_iteration = np.nan
        if history_path.exists():
            history = pd.read_csv(history_path)
            if len(history):
                losses = history["loss"].to_numpy(dtype=float)
                finite = losses[np.isfinite(losses)]
                final_loss = float(losses[-1])
                best_loss = float(finite.min()) if finite.size else np.nan
                final_iteration = (
                    float(history["iteration"].iloc[-1]) if "iteration" in history.columns else float(len(history))
                )
        rows.append(
            {
                "run": str(relative),
                "ensemble": top_level if len(relative.parts) > 1 else "",
                "family": classify_run_family(top_level),
                "run_dir": str(run_dir),
                "target_mode": str(checkpoint.get("target_mode", "")),
                "seed": int(checkpoint.get("seed", -1)),
                "final_loss": final_loss,
                "best_loss": best_loss,
                # Iterations actually completed, read from the loss history rather than
                # from the requested ``epochs`` setting or from the directory name --
                # several run directories are named for an iteration count they never
                # reached, and the checkpoint is the final iterate, not the best one.
                "final_iteration": final_iteration,
                "final_over_best": float(final_loss / best_loss) if best_loss and np.isfinite(best_loss) and best_loss > 0 else np.nan,
                **{field: settings.get(field) for field in COMPARABILITY_FIELDS},
            }
        )
    return pd.DataFrame(rows)


def flag_run_name_iteration_mismatch(inventory: pd.DataFrame) -> pd.DataFrame:
    """Runs whose directory name advertises an iteration count they never reached.

    Several scratch directories carry a ``_100k`` / ``_200k`` suffix from the job that
    was intended rather than the one that ran. Any comparison keyed on the name rather
    than on the stored history inherits that error, so it is worth surfacing explicitly.
    """
    import re

    rows: list[dict[str, object]] = []
    for _, row in inventory.iterrows():
        match = re.search(r"_(\d+)k(?:_|$)", str(row["run"]))
        if match is None:
            continue
        claimed = int(match.group(1)) * 1000
        actual = row["final_iteration"]
        if np.isfinite(actual) and int(actual) != claimed:
            rows.append(
                {
                    "run": row["run"],
                    "name_claims_iterations": claimed,
                    "actual_iterations": int(actual),
                }
            )
    return pd.DataFrame(rows)


def summarize_run_families(inventory: pd.DataFrame) -> pd.DataFrame:
    """Collapse the run index to one row per family, showing the settings spread."""
    rows: list[dict[str, object]] = []
    for family, block in inventory.groupby("family", sort=False):
        row: dict[str, object] = {"family": family, "n_runs": int(len(block))}
        for field in ("epochs", "hidden_units", "activation", "lr", "l1_weight_scale", "recurrent_bottleneck_dim"):
            values = sorted({("None" if pd.isna(v) or v is None else v) for v in block[field]}, key=str)
            row[field] = ", ".join(str(v) for v in values)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_runs", ascending=False).reset_index(drop=True)
    return out


def run_fit_quality(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """Replay one run and score it in PC space and PC-backprojected firing-rate space."""
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    pc = reconstruction_accuracy(replay).assign(space="pc")
    frame = pc
    try:
        fr = pc_reconstructed_firing_rate_accuracy(replay).assign(space="fr")
        frame = pd.concat([pc, fr], ignore_index=True)
    except (KeyError, TypeError, ValueError):
        # Raw-firing-rate targets and runs saved before the PC-backprojection export
        # have no PC basis to project through; PC-space scores still apply.
        pass
    return frame.assign(run_dir=str(run_dir))


def collect_fit_quality(run_dirs: Mapping[str, str | Path], *, device: str = "cpu") -> pd.DataFrame:
    """Score several runs, labelling each with a caller-supplied key."""
    frames = []
    for label, run_dir in run_dirs.items():
        frames.append(run_fit_quality(run_dir, device=device).assign(label=label))
    return pd.concat(frames, ignore_index=True)


def pooled_r2(quality: pd.DataFrame, *, space: str = "pc") -> pd.DataFrame:
    """Mean and median R^2 per label over the region x condition cells."""
    block = quality[quality["space"] == space]
    return (
        block.groupby("label", sort=False)["r2"]
        .agg(mean_r2="mean", median_r2="median", min_r2="min", n_cells="size")
        .reset_index()
    )


# --------------------------------------------------------------------------------------
# Ensembles: training health and cross-initialization consistency
# --------------------------------------------------------------------------------------


def load_ensemble_training_health(ensemble_dir: str | Path) -> pd.DataFrame:
    """Per-seed training health for an ensemble that saved a ``training_health.csv``."""
    path = Path(ensemble_dir) / "ensemble_summaries" / "training_health.csv"
    if not path.exists():
        raise FileNotFoundError(f"no training_health.csv under {ensemble_dir}")
    return pd.read_csv(path)


def load_ensemble_consistency(ensemble_dirs: Mapping[str, str | Path]) -> pd.DataFrame:
    """Cross-initialization consistency scores for one or more ensembles.

    The score is the mean off-diagonal correlation of an initialization x initialization
    similarity matrix, computed on rotation-invariant summaries of the fitted dynamics:
    inter-regional current magnitude, relative source contribution, and the drive RDM.
    A score of 1 would mean every seed found the same dynamics.
    """
    frames = []
    for label, ensemble_dir in ensemble_dirs.items():
        path = Path(ensemble_dir) / "ensemble_summaries" / "consistency_summary.csv"
        frames.append(pd.read_csv(path).assign(ensemble=label))
    return pd.concat(frames, ignore_index=True)


def load_ensemble_model_rank(ensemble_dir: str | Path) -> pd.DataFrame:
    """Fit-quality ranking table saved by an ensemble notebook."""
    return pd.read_csv(Path(ensemble_dir) / "ensemble_summaries" / "model_rank.csv")


# --------------------------------------------------------------------------------------
# What is invariant: condition modulation of inter-regional drive
# --------------------------------------------------------------------------------------


def region_source_contribution(replay: Mapping[str, object]) -> pd.DataFrame:
    """Signed relative contribution of each source region to each target's drive.

    For target region ``r`` under condition ``c`` the drive direction is
    ``W_rec h_t - h_t`` restricted to ``r``'s units. Each source region's current into
    ``r`` is projected onto that unit-normalized direction, and normalized by the sum of
    absolute projections so the four sources form a signed share at every time bin.

    The quantity is invariant to permutation and sign of hidden units, which is what
    makes it comparable across independently initialized fits.
    """
    model = replay["model"]
    regions = tuple(replay["region_order"])
    conditions = tuple(replay["condition_order"])
    slices = model.hidden_region_slices()
    currents = extract_region_current_vectors(replay)
    latent = extract_fixation_latent_dynamics(replay)
    timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)

    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(conditions):
        for target in regions:
            region_slice = slices[target]
            hidden = latent[condition]["hidden_state"][:, region_slice].numpy()
            drive = latent[condition]["recurrent_drive"][:, region_slice].numpy()
            direction = drive - hidden
            norm = np.linalg.norm(direction, axis=-1, keepdims=True)
            unit = np.divide(direction, np.maximum(norm, 1e-8), out=np.zeros_like(direction), where=norm > 1e-8)
            projections = {
                source: np.sum(currents[(source, target)].numpy()[condition_index] * unit, axis=-1)
                for source in regions
            }
            denominator = np.sum([np.abs(v) for v in projections.values()], axis=0)
            denominator = np.where(denominator > 1e-8, denominator, 1.0)
            for source in regions:
                relative = projections[source] / denominator
                for time_index, value in enumerate(projections[source]):
                    rows.append(
                        {
                            "active_condition": condition,
                            "source_region": source,
                            "target_region": target,
                            "time_idx": int(time_index),
                            "time_s": float(timeline[time_index]),
                            "projection": float(value),
                            "relative_projection": float(relative[time_index]),
                        }
                    )
    return pd.DataFrame(rows)


def ensemble_source_contribution(
    run_dirs: Mapping[int, str | Path],
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Replay an ensemble and stack :func:`region_source_contribution` over its seeds."""
    frames = []
    for init_idx, run_dir in run_dirs.items():
        replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
        frames.append(region_source_contribution(replay).assign(init_idx=int(init_idx)))
    return pd.concat(frames, ignore_index=True)


def load_stored_source_contribution(ensemble_dir: str | Path) -> pd.DataFrame:
    """Per-seed source-contribution table saved by the 100-initialization ensemble."""
    path = Path(ensemble_dir) / "ensemble_summaries" / "within_current_projection_by_seed.pkl"
    return pd.read_pickle(path)


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Signed-rank statistic and p-value, tolerating an all-zero difference vector.

    ``scipy.stats.wilcoxon`` raises when every paired difference is exactly zero. That
    is the strongest possible null rather than an error, so it is reported as p = 1.
    """
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if differences.size == 0 or not np.any(differences != 0.0):
        return 0.0, 1.0
    test = stats.wilcoxon(a, b)
    return float(test.statistic), float(test.pvalue)


def _holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    n = values.size
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def condition_contrast_by_source(
    contribution: pd.DataFrame,
    *,
    seed_column: str = "init_idx",
    conditions: Sequence[str] = CONDITION_ORDER,
) -> pd.DataFrame:
    """Test whether each source region's drive share depends on fixation condition.

    Each seed contributes one time-averaged share per source and condition, so the
    comparison is paired across seeds: the same fitted network is asked about all three
    conditions. That pairing is what makes the test meaningful despite the large
    seed-to-seed spread in the absolute share.
    """
    per_seed = (
        contribution.groupby([seed_column, "active_condition", "source_region"])["relative_projection"]
        .mean()
        .reset_index()
    )
    wide = per_seed.pivot_table(
        index=seed_column, columns=["source_region", "active_condition"], values="relative_projection"
    )
    rows: list[dict[str, object]] = []
    pairs = [(conditions[0], conditions[1]), (conditions[0], conditions[2]), (conditions[1], conditions[2])]
    for source in sorted({c[0] for c in wide.columns}):
        for left, right in pairs:
            a = wide[(source, left)].to_numpy(dtype=float)
            b = wide[(source, right)].to_numpy(dtype=float)
            keep = np.isfinite(a) & np.isfinite(b)
            statistic, pvalue = _paired_wilcoxon(a[keep], b[keep])
            rows.append(
                {
                    "source_region": source,
                    "condition_a": left,
                    "condition_b": right,
                    "mean_a": float(np.mean(a[keep])),
                    "mean_b": float(np.mean(b[keep])),
                    "difference": float(np.mean(a[keep]) - np.mean(b[keep])),
                    "n_seeds": int(keep.sum()),
                    "wilcoxon_w": statistic,
                    "p_raw": pvalue,
                }
            )
    out = pd.DataFrame(rows)
    out["p_holm"] = _holm(out["p_raw"].to_numpy())
    return out


def condition_contrast_by_pathway(
    contribution: pd.DataFrame,
    *,
    source_region: str,
    condition_a: str = "face_interactive",
    condition_b: str = "face_non_interactive",
    seed_column: str = "init_idx",
) -> pd.DataFrame:
    """Break one source region's condition contrast down by target region."""
    per_seed = (
        contribution.groupby([seed_column, "active_condition", "source_region", "target_region"])[
            "relative_projection"
        ]
        .mean()
        .reset_index()
    )
    block = per_seed[per_seed["source_region"] == source_region]
    wide = block.pivot_table(index=seed_column, columns=["target_region", "active_condition"], values="relative_projection")
    rows: list[dict[str, object]] = []
    for target in sorted({c[0] for c in wide.columns}):
        a = wide[(target, condition_a)].to_numpy(dtype=float)
        b = wide[(target, condition_b)].to_numpy(dtype=float)
        keep = np.isfinite(a) & np.isfinite(b)
        _, pvalue = _paired_wilcoxon(a[keep], b[keep])
        rows.append(
            {
                "pathway": f"{source_region}→{target}",
                "mean_a": float(np.mean(a[keep])),
                "mean_b": float(np.mean(b[keep])),
                "difference": float(np.mean(a[keep]) - np.mean(b[keep])),
                "n_seeds": int(keep.sum()),
                "p_raw": pvalue,
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = _holm(out["p_raw"].to_numpy())
    return out


# --------------------------------------------------------------------------------------
# What is invariant: the identity of the low-rank inter-regional channels
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelSubspaces:
    """Orthonormal bases for one model's inter-regional communication channels.

    A rank-``r`` inter-regional block is ``left @ right``. ``left`` selects what is read
    out of the source region's state; ``right`` selects where it is written into the
    target. Both live in hidden-unit coordinates, which are permutable and sign-ambiguous
    across seeds, so they are mapped into each region's *PC space* — fixed by the data and
    therefore shared by every fit — before any comparison.
    """

    write: dict[tuple[str, str], np.ndarray]
    read: dict[tuple[str, str], np.ndarray]
    rank: int


def channel_subspaces(run_dir: str | Path, *, device: str = "cpu") -> ChannelSubspaces:
    """Read/write subspaces of every low-rank inter-regional block, in PC coordinates.

    The write subspace into target ``t`` is ``W_out[t] @ right.T``: the PC-space
    directions the channel deposits into. The read subspace from source ``s`` is
    ``pinv(W_out[s]).T @ left``: the PC-space directions the channel is sensitive to,
    obtained by pushing the readout's pseudo-inverse through the left factor.
    """
    model, _ = load_fixation_mrnn_checkpoint(Path(run_dir), device=device)
    if not getattr(model, "_inter_region_left_params", None):
        raise ValueError(f"{run_dir} has no low-rank inter-regional blocks to compare")
    readout = model.readout_weight_matrix().detach().cpu().numpy()
    output_slices = model.output_region_slices()
    hidden_slices = model.hidden_region_slices()

    write: dict[tuple[str, str], np.ndarray] = {}
    read: dict[tuple[str, str], np.ndarray] = {}
    rank = 0
    for (source, target), left_param in model._inter_region_left_params.items():
        left = left_param.detach().cpu().numpy()
        right = model._inter_region_right_params[(source, target)].detach().cpu().numpy()
        rank = int(right.shape[0])
        readout_target = readout[output_slices[target], hidden_slices[target]]
        readout_source = readout[output_slices[source], hidden_slices[source]]
        write[(source, target)] = np.linalg.qr(readout_target @ right.T)[0]
        read[(source, target)] = np.linalg.qr(np.linalg.pinv(readout_source).T @ left)[0]
    return ChannelSubspaces(write=write, read=read, rank=rank)


def subspace_alignment(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Mean cosine of the principal angles between two orthonormal bases."""
    singular_values = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    return float(np.mean(np.clip(singular_values, 0.0, 1.0)))


def random_subspace_alignment_null(
    *,
    ambient_dim: int,
    rank: int,
    n_draws: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Alignment between independent random ``rank``-dimensional subspaces.

    This is the floor the observed cross-seed alignment has to clear: two arbitrary
    3-D subspaces of a 42-D space are not orthogonal, they overlap by about 0.22.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_draws), dtype=float)
    for index in range(int(n_draws)):
        a = np.linalg.qr(rng.normal(size=(ambient_dim, rank)))[0]
        b = np.linalg.qr(rng.normal(size=(ambient_dim, rank)))[0]
        draws[index] = subspace_alignment(a, b)
    return draws


def channel_consistency_across_seeds(
    subspaces_by_seed: Mapping[int, ChannelSubspaces],
    *,
    kinds: Sequence[str] = ("write", "read"),
) -> pd.DataFrame:
    """Cross-seed alignment of each pathway's channel, pathway by pathway."""
    seeds = sorted(subspaces_by_seed)
    rows: list[dict[str, object]] = []
    for kind in kinds:
        pathways = sorted(getattr(subspaces_by_seed[seeds[0]], kind))
        for pathway in pathways:
            values = [
                subspace_alignment(
                    getattr(subspaces_by_seed[a], kind)[pathway],
                    getattr(subspaces_by_seed[b], kind)[pathway],
                )
                for index, a in enumerate(seeds)
                for b in seeds[index + 1 :]
            ]
            rows.append(
                {
                    "kind": kind,
                    "pathway": f"{pathway[0]}→{pathway[1]}",
                    "source_region": pathway[0],
                    "target_region": pathway[1],
                    "mean_alignment": float(np.mean(values)),
                    "sd_alignment": float(np.std(values)),
                    "n_pairs": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def channel_identity_test(
    subspaces_by_seed: Mapping[int, ChannelSubspaces],
    *,
    kinds: Sequence[str] = ("write", "read"),
) -> pd.DataFrame:
    """Is a pathway's channel more like *itself* in another seed than like another pathway?

    Comparing a pathway across seeds against an absolute floor conflates two questions:
    whether any structure is shared at all, and whether the shared structure is
    *pathway-specific*. This test isolates the second. Matched comparisons pair
    ``s→t`` in seed A with ``s→t`` in seed B; mismatched comparisons pair it with a
    different pathway that writes into (or reads from) the same region, so both sets of
    subspaces live in the same PC space and are directly comparable.

    **The p-value is anti-conservative.** Each subspace enters many pairings, so the
    comparisons are not independent and the rank-sum test rejects more often than its
    nominal rate under the null (roughly 1 in 8 rather than 1 in 20 at alpha = 0.05 in
    simulation). That direction is harmless for a *negative* finding -- a test that
    over-rejects and still fails to reject is strong evidence of no effect -- but a
    positive result from this function needs a resampling null over seeds before it can
    be reported. Read ``difference`` alongside the p-value in either case.
    """
    seeds = sorted(subspaces_by_seed)
    rows: list[dict[str, object]] = []
    for kind in kinds:
        pathways = sorted(getattr(subspaces_by_seed[seeds[0]], kind))
        shared_region_index = 1 if kind == "write" else 0
        matched: list[float] = []
        mismatched: list[float] = []
        for a in seeds:
            for b in seeds:
                if a == b:
                    continue
                for left in pathways:
                    for right in pathways:
                        if left[shared_region_index] != right[shared_region_index]:
                            continue
                        value = subspace_alignment(
                            getattr(subspaces_by_seed[a], kind)[left],
                            getattr(subspaces_by_seed[b], kind)[right],
                        )
                        (matched if left == right else mismatched).append(value)
        if matched and mismatched:
            test = stats.mannwhitneyu(matched, mismatched, alternative="greater")
            statistic, pvalue = float(test.statistic), float(test.pvalue)
        else:
            # No mismatched pairing exists when each region is the target of a single
            # pathway; the comparison is undefined rather than null.
            statistic, pvalue = np.nan, np.nan
        rows.append(
            {
                "kind": kind,
                "mean_matched": float(np.mean(matched)) if matched else np.nan,
                "mean_mismatched": float(np.mean(mismatched)) if mismatched else np.nan,
                "difference": float(np.mean(matched) - np.mean(mismatched)) if matched and mismatched else np.nan,
                "n_matched": int(len(matched)),
                "n_mismatched": int(len(mismatched)),
                "mannwhitney_u": statistic,
                "p_one_sided": pvalue,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "CONDITION_ORDER",
    "ChannelSubspaces",
    "DEFAULT_MODELING_SUBDIR",
    "REGION_ORDER",
    "channel_consistency_across_seeds",
    "channel_identity_test",
    "channel_subspaces",
    "classify_run_family",
    "collect_fit_quality",
    "condition_contrast_by_pathway",
    "condition_contrast_by_source",
    "ensemble_source_contribution",
    "flag_run_name_iteration_mismatch",
    "index_fixation_mrnn_runs",
    "load_ensemble_consistency",
    "load_ensemble_model_rank",
    "load_ensemble_training_health",
    "load_stored_source_contribution",
    "pooled_r2",
    "random_subspace_alignment_null",
    "region_source_contribution",
    "resolve_output_dir",
    "resolve_scratch_root",
    "run_fit_quality",
    "subspace_alignment",
    "summarize_run_families",
]
