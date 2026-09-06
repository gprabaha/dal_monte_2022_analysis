"""Audit of what the mRNN scoring code measures, and the invariants that survive it.

Four of the quantities tasks 01-05 report are not the quantities they are named after.
Each function here recomputes one of them both ways -- as the chapter currently scores it,
and as the name implies -- so the difference is a measured number rather than an argument.

===================  =========================================================
``E1`` rank cap      Each region reads ``n_pc`` components out of ``h`` units
                     through an ``(n_pc, h)`` matrix. For ``h < n_pc`` that map
                     has rank ``h``, so the achievable R^2 is capped by the
                     target's own PC spectrum truncated at ``h`` -- a fact of
                     linear algebra that the capacity sweep reads as capacity.
``E2`` self-current  ``_flat_current_magnitudes`` builds its vector from all 16
                     ``(source, target)`` blocks, four of which are each
                     region's own self-drive and are the largest. Only the 12
                     off-diagonal blocks are what a bottleneck constrains.
``E3`` Jacobian      The update is ``phi(W h + b)`` but ``jacobian_eigenvalues``
                     evaluates ``phi'`` at ``W h`` alone. ``b`` carries the
                     condition, and is not small.
``E4`` damage units  ``reconstruction_accuracy`` normalises R^2 by *each
                     condition's own* ``ss_tot``. Interactive face has about
                     half the target variance of either other condition, so an
                     identical absolute perturbation scores twice the damage --
                     the same confound task 02 diagnosed in the objective,
                     reappearing untouched in the metric.
===================  =========================================================

The invariants (``I1``-``I3``) are the complement: quantities compared *within* one model
*across* conditions. The signed permutation applies to both arguments and cancels, so they
need no alignment step and no assumption that the weights are identifiable -- which is
exactly the class of claim an ensemble that fails its identifiability gate can still
support.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_circuit import (
    current_condition_alignment,
    current_drive_alignment,
    load_fixation_mrnn_checkpoint,
    recurrent_blocks,
    region_states,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
    extract_region_current_vectors,
    replay_fixation_mrnn_run,
    replay_fixation_mrnn_run_with_ablations,
)

CONDITION_ORDER: tuple[str, ...] = ("face_interactive", "face_non_interactive", "object")


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def seed_run_dirs(arm_dir: str | Path) -> list[Path]:
    """The ``seed=*`` run directories under one sweep arm, in a stable order."""
    root = Path(arm_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("seed="))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float) - np.mean(a)
    right = np.asarray(b, dtype=float) - np.mean(b)
    denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    return float(np.sum(left * right) / denominator) if denominator > 0 else float("nan")


def _mean_pairwise(vectors: Sequence[np.ndarray]) -> float:
    pairs = [_pearson(vectors[i], vectors[j]) for i, j in combinations(range(len(vectors)), 2)]
    return float(np.nanmean(pairs)) if pairs else float("nan")


def _target_by_region(replay: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {
        region: np.asarray(array, dtype=float)
        for region, array in replay["checkpoint"]["target_by_region"].items()
    }


def condition_bias(replay: Mapping[str, object], condition_index: int) -> np.ndarray:
    """The constant drive ``W_inp u_c + tonic`` a condition contributes.

    This is what distinguishes the three conditions -- the input is a constant one-hot, so
    everything that is not the initial state enters here. Averaged over time rather than
    taken at one bin so the helper stays correct if a time-varying input is ever used.
    """
    model = replay["model"]
    inp = replay["inp"].detach()[condition_index]
    if model.mrnn.inp_constrained:
        weight = model.mrnn.apply_dales_law(
            model.mrnn.W_inp, model.mrnn.W_inp_mask, model.mrnn.W_inp_sign_matrix
        )
    else:
        weight = model.mrnn.W_inp * model.mrnn.W_inp_mask
    return ((weight @ inp.mean(dim=0)) + model.mrnn.tonic_inp).detach().cpu().numpy()


# --------------------------------------------------------------------------------------
# E1: the readout rank cap
# --------------------------------------------------------------------------------------


def readout_rank_cap(
    run_dir: str | Path,
    widths: Sequence[int],
    ceiling_by_region: Mapping[str, float],
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Best R^2 any width could reach, from the target's own PC spectrum.

    A region's output is ``h W_out^T`` with ``h`` of width ``n_units``, so it is confined
    to an ``n_units``-dimensional subspace of the ``n_pc``-dimensional PC space. The
    fraction of target energy inside the best such subspace is the ceiling on fit before
    any dynamics are considered, and for ``n_units < n_pc`` it is well below 1.

    Returned per region and per width, both raw and divided by the measured noise ceiling
    so it can be laid alongside the sweep's own ``r2_vs_ceiling`` scores.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    rows: list[dict[str, object]] = []
    for region, target in _target_by_region(replay).items():
        flat = target.reshape(-1, target.shape[-1])
        energy = np.linalg.svd(flat, compute_uv=False) ** 2
        total = float(energy.sum())
        ceiling = float(ceiling_by_region.get(region, np.nan))
        for width in widths:
            kept = float(energy[: min(int(width), energy.size)].sum())
            rows.append(
                {
                    "region": region,
                    "width": int(width),
                    "n_components": int(target.shape[-1]),
                    "raw_cap": kept / total if total > 0 else np.nan,
                    "ceiling": ceiling,
                    "ceiling_relative_cap": (kept / total) / ceiling if ceiling > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def capacity_against_cap(
    cap: pd.DataFrame,
    measured_by_width: Mapping[int, float],
) -> pd.DataFrame:
    """The capacity sweep rescored as a fraction of what each width could reach.

    ``measured_by_width`` is the sweep's own mean ``r2_vs_ceiling`` per width. The
    ``fraction_of_cap`` column is the part of the curve that is about the network.
    """
    pooled = cap.groupby("width")[["raw_cap", "ceiling_relative_cap"]].mean()
    table = pooled.assign(measured=pd.Series(dict(measured_by_width)))
    table["fraction_of_cap"] = table["measured"] / table["ceiling_relative_cap"]
    table["headroom_lost_to_readout"] = 1.0 - table["ceiling_relative_cap"] / table[
        "ceiling_relative_cap"
    ].max()
    return table.reset_index()


# --------------------------------------------------------------------------------------
# E2: self-drive contamination of the "inter-regional" current measure
# --------------------------------------------------------------------------------------


def _magnitude_vector(replay: Mapping[str, object], *, blocks: str) -> np.ndarray:
    currents = extract_region_current_vectors(replay)
    regions = tuple(replay["region_order"])
    if blocks == "all":
        keys = [(s, t) for s in regions for t in regions if (s, t) in currents]
    elif blocks == "inter":
        keys = [(s, t) for s in regions for t in regions if s != t and (s, t) in currents]
    elif blocks == "self":
        keys = [(s, s) for s in regions if (s, s) in currents]
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown block selection {blocks!r}")
    # A block the architecture removed carries an identically zero current, and two zero
    # vectors agree perfectly. Leaving them in would report an ablated arm as the most
    # reproducible one in the sweep -- which is how `isolate_bla` first scored 0.97.
    parts = [np.linalg.norm(currents[key].numpy(), axis=-1).ravel() for key in keys]
    parts = [part for part in parts if np.any(part != 0.0)]
    return np.concatenate(parts) if parts else np.zeros(0)


def self_drive_energy_share(replay: Mapping[str, object]) -> float:
    """Fraction of total recurrent current energy carried by the four diagonal blocks."""
    currents = extract_region_current_vectors(replay)
    regions = tuple(replay["region_order"])

    def energy(keys: Iterable[tuple[str, str]]) -> float:
        return float(sum((np.linalg.norm(currents[k].numpy(), axis=-1) ** 2).sum() for k in keys))

    own = energy([(r, r) for r in regions if (r, r) in currents])
    cross = energy([(s, t) for s in regions for t in regions if s != t and (s, t) in currents])
    total = own + cross
    return own / total if total > 0 else float("nan")


def current_agreement_decomposition(
    arm_dirs: Mapping[str, str | Path],
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Cross-seed current agreement, split into the blocks a bottleneck does and does not touch.

    The stored measure concatenates all 16 blocks. As the rank constraint tightens, the
    four unconstrained diagonal blocks take a larger share of the vector's energy, so the
    measure drifts towards reporting within-region drive under the name of inter-regional
    communication. Splitting it is the whole point of this function.
    """
    rows: list[dict[str, object]] = []
    for label, arm in arm_dirs.items():
        run_dirs = seed_run_dirs(arm)
        if len(run_dirs) < 2:
            continue
        replays = [replay_fixation_mrnn_run(path, device=device) for path in run_dirs]
        rows.append(
            {
                "label": label,
                "n_seeds": len(replays),
                "self_energy_share": float(np.mean([self_drive_energy_share(r) for r in replays])),
                "agreement_all_blocks": _mean_pairwise(
                    [_magnitude_vector(r, blocks="all") for r in replays]
                ),
                "agreement_inter_only": _mean_pairwise(
                    [_magnitude_vector(r, blocks="inter") for r in replays]
                ),
                "agreement_self_only": _mean_pairwise(
                    [_magnitude_vector(r, blocks="self") for r in replays]
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# E3 / I2: the Jacobian, with the bias the update actually uses
# --------------------------------------------------------------------------------------


def corrected_jacobian_eigenvalues(
    replay: Mapping[str, object],
    state: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Eigenvalues of ``diag(phi'(W h + b)) W``, ordered by decreasing modulus.

    The fix to :func:`fixation_mrnn_circuit.jacobian_eigenvalues`, which evaluates the
    derivative at ``W h`` and so linearizes a system the model does not run. For tanh the
    identity the original docstring intends holds once ``b`` is present: at a fixed point
    ``tanh(W h* + b) = h*``, so ``phi' = 1 - h*^2``.
    """
    model = replay["model"]
    weight = model.recurrent_weight_matrix().detach().cpu().numpy()
    point = np.asarray(state, dtype=float)
    pre = point @ weight.T + np.asarray(bias, dtype=float)
    activation = str(getattr(model.spec, "activation", "tanh")).strip().lower()
    if activation == "tanh":
        derivative = 1.0 - np.tanh(pre) ** 2
    else:
        tensor = torch.as_tensor(pre, dtype=torch.float32, requires_grad=True)
        activated = model.mrnn.activation(tensor)
        derivative = torch.autograd.grad(activated.sum(), tensor)[0].detach().numpy()
    eigenvalues = np.linalg.eigvals(derivative[:, None] * weight)
    return eigenvalues[np.argsort(-np.abs(eigenvalues))]


def slow_point_stability(
    run_dir: str | Path,
    *,
    n_starts: int = 16,
    iterations: int = 600,
    device: str = "cpu",
) -> pd.DataFrame:
    """Local stability at each condition's slowest point, scored both ways.

    The input is a constant one-hot, so each condition is its own autonomous system and
    its fixed-point structure is well defined. Reported with and without the bias term so
    the size of :data:`E3` is visible rather than asserted.
    """
    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_circuit import (
        jacobian_eigenvalues,
        slow_points,
    )

    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    rows: list[dict[str, object]] = []
    for index, condition in enumerate(replay["condition_order"]):
        found = slow_points(replay, index, n_starts=n_starts, iterations=iterations)
        slowest = int(np.argmin(found["speed"]))
        state = found["points"][slowest]
        bias = condition_bias(replay, index)
        as_computed = jacobian_eigenvalues(replay, state)
        corrected = corrected_jacobian_eigenvalues(replay, state, bias)
        rows.append(
            {
                "run_dir": str(run_dir),
                "seed": Path(run_dir).name.replace("seed=", ""),
                "condition": str(condition),
                "slow_point_speed": float(found["speed"][slowest]),
                "state_norm": float(np.linalg.norm(state)),
                "bias_norm": float(np.linalg.norm(bias)),
                "top_modulus_as_computed": float(np.abs(as_computed).max()),
                "top_modulus_corrected": float(np.abs(corrected).max()),
                "n_expanding_as_computed": int((np.abs(as_computed) > 1.0).sum()),
                "n_expanding_corrected": int((np.abs(corrected) > 1.0).sum()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# E4: damage in units that are comparable between conditions
# --------------------------------------------------------------------------------------


def condition_target_energy(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """Per-condition target variance -- the denominator that makes R^2 incomparable.

    A condition with half the variance registers twice the R^2 damage for the same
    absolute error, which is the whole of :data:`E4`.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    conditions = list(replay["condition_order"])
    rows: list[dict[str, object]] = []
    for region, target in _target_by_region(replay).items():
        for index, condition in enumerate(conditions):
            block = target[index]
            rows.append(
                {
                    "region": region,
                    "condition": str(condition),
                    "ss_tot": float(((block - target.mean(axis=(0, 1))) ** 2).sum()),
                }
            )
    return pd.DataFrame(rows)


def _sse_by_region_condition(replay: Mapping[str, object]) -> dict[tuple[str, str], float]:
    conditions = list(replay["condition_order"])
    out: dict[tuple[str, str], float] = {}
    for region, target in _target_by_region(replay).items():
        predicted = replay["output_by_region"][region].detach().cpu().numpy().astype(float)
        for index, condition in enumerate(conditions):
            out[(region, str(condition))] = float(((target[index] - predicted[index]) ** 2).sum())
    return out


def pathway_ablation_in_both_units(
    run_dir: str | Path,
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Post-hoc pathway lesion damage, in R^2 units and in absolute squared error.

    The R^2 column is what the chapter reports; the SSE column is the same perturbation
    without each condition's variance folded into the denominator. Between-condition
    comparisons are only interpretable in the second.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    regions = list(replay["region_order"])
    conditions = [str(c) for c in replay["condition_order"]]
    targets = _target_by_region(replay)
    ss_tot = {
        (region, condition): float(((targets[region][i] - targets[region].mean(axis=(0, 1))) ** 2).sum())
        for region in regions
        for i, condition in enumerate(conditions)
    }
    intact = _sse_by_region_condition(replay)

    rows: list[dict[str, object]] = []
    for source in regions:
        for target_region in regions:
            if source == target_region:
                continue
            lesioned = _sse_by_region_condition(
                replay_fixation_mrnn_run_with_ablations(
                    Path(run_dir), ablations=[(source, target_region)], device=device
                )
            )
            for region in regions:
                for condition in conditions:
                    key = (region, condition)
                    delta = lesioned[key] - intact[key]
                    rows.append(
                        {
                            "seed": Path(run_dir).name.replace("seed=", ""),
                            "pathway": f"{source}→{target_region}",
                            "region": region,
                            "condition": condition,
                            "damage_sse": delta,
                            "damage_r2": delta / ss_tot[key] if ss_tot[key] > 0 else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# I1 / I3: within-model, between-condition invariants
# --------------------------------------------------------------------------------------


def condition_channel_alignment(run_dirs: Sequence[str | Path], *, device: str = "cpu") -> pd.DataFrame:
    """Angle between the current one pathway carries under two conditions, per seed.

    Free of any relabelling ambiguity: the signed permutation applies to both arguments
    and cancels, so no alignment step is needed and the quantity is comparable across fits
    that do not agree about anything else.
    """
    frames = []
    for run_dir in run_dirs:
        replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
        frame = current_condition_alignment(replay)
        frame["seed"] = Path(run_dir).name.replace("seed=", "")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def untrained_channel_alignment(
    run_dir: str | Path,
    *,
    n_draws: int = 6,
    device: str = "cpu",
    seed: int = 0,
) -> pd.DataFrame:
    """The same measure between conditions in networks that were never trained.

    The architectural floor for :func:`condition_channel_alignment`. The untrained models
    are replayed on the fitted run's own inputs and initial states, so the only difference
    from the fitted ensemble is that the recurrent weights never saw the data.
    """
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
        _model_spec_from_checkpoint,
        resolve_device,
    )
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_model import FixationMRNNModel

    resolved = resolve_device(device)
    _, checkpoint = load_fixation_mrnn_checkpoint(Path(run_dir), device=device)
    spec = _model_spec_from_checkpoint(checkpoint, device=resolved)
    inp = torch.as_tensor(checkpoint["input_tensor"], dtype=torch.float32, device=resolved)
    h0 = checkpoint["h0"].to(resolved)

    frames = []
    for draw in range(int(n_draws)):
        torch.manual_seed(int(seed) + 1000 + draw)
        model = FixationMRNNModel(spec).to(resolved).eval()
        with torch.no_grad():
            h_seq = model(inp, h0)["h_seq"]
        replay = {
            "model": model,
            "h_seq": h_seq,
            "inp": inp,
            "h0": h0,
            "region_order": model.region_order,
            "condition_order": list(checkpoint["condition_order"]),
            "checkpoint": checkpoint,
        }
        frame = current_condition_alignment(replay)
        frame["seed"] = f"null{draw}"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def target_condition_similarity(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """Correlation between the *target* PC trajectories of two conditions.

    The data-side comparison for :func:`condition_channel_alignment`. If the fitted
    channels align simply because the trajectories do, the two tables agree; where they
    disagree, the alignment is something the network imposed.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    conditions = [str(c) for c in replay["condition_order"]]
    rows: list[dict[str, object]] = []
    for region, target in _target_by_region(replay).items():
        for i, j in combinations(range(len(conditions)), 2):
            rows.append(
                {
                    "region": region,
                    "condition_a": conditions[i],
                    "condition_b": conditions[j],
                    "correlation": _pearson(target[i].ravel(), target[j].ravel()),
                }
            )
    return pd.DataFrame(rows)


def drive_alignment_by_condition(run_dirs: Sequence[str | Path], *, device: str = "cpu") -> pd.DataFrame:
    """Cosine between incoming inter-regional current and the target's own recurrent drive.

    Also a within-model measure, so also free of the relabelling problem. Positive means
    the rest of the network pushes a region along the direction its internal dynamics were
    already taking it.
    """
    frames = []
    for run_dir in run_dirs:
        replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
        frame = current_drive_alignment(replay)
        frame["seed"] = Path(run_dir).name.replace("seed=", "")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sign_consistency(alignment: pd.DataFrame, *, value: str = "cosine") -> pd.DataFrame:
    """Fraction of pathway x seed cells with a positive value, per condition pair.

    The statistic the invariant claim rests on. A mean cosine can be dragged around by a
    few pathways; the fraction of cells on one side of zero cannot.
    """
    frame = alignment.assign(_positive=alignment[value] > 0)
    return (
        frame.groupby(["condition_a", "condition_b"])
        .agg(
            mean_cosine=(value, "mean"),
            fraction_positive=("_positive", "mean"),
            n_cells=(value, "size"),
        )
        .reset_index()
    )


__all__ = [
    "CONDITION_ORDER",
    "capacity_against_cap",
    "condition_bias",
    "condition_channel_alignment",
    "condition_target_energy",
    "corrected_jacobian_eigenvalues",
    "current_agreement_decomposition",
    "drive_alignment_by_condition",
    "pathway_ablation_in_both_units",
    "readout_rank_cap",
    "seed_run_dirs",
    "self_drive_energy_share",
    "sign_consistency",
    "slow_point_stability",
    "target_condition_similarity",
    "untrained_channel_alignment",
]


# ======================================================================================
# The solution manifold: how to measure agreement, and what is constant across it
# ======================================================================================
#
# Two questions, and they need different tools.
#
# *How different are two fits?* needs a **battery**, not a number. Every similarity
# measure is blind to some group of transformations, and a measure blind to more than the
# model's own symmetry can only overstate agreement. So the measures below are tiered by
# what they are invariant to, strictest first, and each is reported against the same
# untrained floor. Two of them are distances rather than similarities: a correlation near
# zero and a distance near its maximum say the same thing, but only the distance says how
# far apart in units the quantity is actually measured in.
#
# *What is the same in all of them?* needs a **screen**. Candidate properties are computed
# per fit and per condition, and a property counts only if its condition ordering holds in
# every fit of every architecture. That is a much stronger bar than significance across
# seeds of one arm, and it is the bar the solution set imposes.
#
# The screen has one trap, and it is the reason ``target_side_control`` exists. A property
# can be constant across every fit simply because it is a property of the *target* that any
# adequate fit must reproduce. That is a validation of the model, not a finding about it.
# Only a quantity with no counterpart in the target -- or one that survives after the
# target's own value is divided out -- is something the model contributes.


def similarity_transform_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    iterations: int = 1200,
    learning_rate: float = 0.05,
    restarts: int = 1,
    seed: int = 0,
) -> float:
    """:math:`\\min_{C \\in O(n)} \\lVert A - C B C^{\\top}\\rVert_F`, normalised.

    The core of Dynamical Similarity Analysis, implemented directly so the comparison
    carries no extra dependency. Two linear operators describe the same dynamics if one is
    an orthogonal conjugation of the other, and this is the distance to the nearest such
    conjugation. It is strictly stronger than comparing eigenspectra: two operators can
    share a spectrum and still be inequivalent, because the spectrum discards the
    eigenvector geometry.

    :math:`C` is parameterised as :math:`\\exp(S - S^{\\top})`, which is a surjection onto
    the connected component of :math:`O(n)`, so the optimisation is unconstrained.
    Normalised by :math:`\\lVert A\\rVert_F \\lVert B\\rVert_F` raised to one half, so 0 is
    identical and values near 1 are as far apart as two operators of that size can be.

    The optimisation is non-convex over a large group, so the value it reaches is an upper
    bound on the true distance. :func:`operator_distance_resolution` measures how loose that
    bound is by handing the optimiser a pair it should score at exactly zero, and that
    number is the scale below which two operators cannot be told apart.
    """
    left = torch.as_tensor(np.asarray(a, dtype=np.float32))
    right = torch.as_tensor(np.asarray(b, dtype=np.float32))
    scale = float(torch.linalg.matrix_norm(left) * torch.linalg.matrix_norm(right)) ** 0.5
    if scale <= 0:
        return float("nan")

    # Started at the identity, not at random: for two operators that are already equal the
    # answer is then exact rather than whatever a random rotation happens to reach, and for
    # any other pair it is as good a starting point as any. A cosine decay is what makes
    # the last hundred iterations settle instead of jitter -- the same lesson as task 00.
    torch.manual_seed(int(seed))
    best = float(torch.linalg.matrix_norm(left - right))
    for restart in range(int(restarts)):
        skew = torch.zeros(left.shape[0], left.shape[0])
        if restart:
            skew = 0.05 * torch.randn(left.shape[0], left.shape[0])
        skew = skew.requires_grad_(True)
        optimizer = torch.optim.Adam([skew], lr=float(learning_rate))
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(iterations), eta_min=float(learning_rate) * 0.02
        )
        for _ in range(int(iterations)):
            optimizer.zero_grad()
            rotation = torch.matrix_exp(skew - skew.T)
            loss = torch.linalg.matrix_norm(left - rotation @ right @ rotation.T)
            loss.backward()
            optimizer.step()
            schedule.step()
        with torch.no_grad():
            rotation = torch.matrix_exp(skew - skew.T)
            best = min(best, float(torch.linalg.matrix_norm(left - rotation @ right @ rotation.T)))
    return best / scale


def eigenspectrum_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Optimal-transport distance between two eigenvalue clouds in the complex plane.

    The distance counterpart of :func:`fixation_mrnn_circuit.eigenspectrum_agreement`.
    Both spectra have the same number of points, so the transport plan is an exact
    assignment. Normalised by the mean modulus of the two spectra, which makes it
    comparable between architectures of different scale.
    """
    from scipy.optimize import linear_sum_assignment

    left = np.asarray(a).ravel()
    right = np.asarray(b).ravel()
    size = min(left.size, right.size)
    left, right = left[:size], right[:size]
    cost = np.abs(left[:, None] - right[None, :])
    rows, columns = linear_sum_assignment(cost)
    denominator = 0.5 * (np.abs(left).mean() + np.abs(right).mean())
    return float(cost[rows, columns].mean() / denominator) if denominator > 0 else float("nan")


def matched_weight_distance(model_a, model_b) -> float:
    """Weight-space distance after the best signed permutation, normalised.

    The distance counterpart of ``matched_unit_alignment``'s correlation. Reported
    alongside it because a correlation of 0.02 is easy to read as "small but present",
    while a normalised distance of 1.4 makes the size of the disagreement explicit.
    """
    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_circuit import matched_unit_alignment

    match = matched_unit_alignment(model_a, model_b)
    left = model_a.recurrent_weight_matrix().detach().cpu().numpy()
    right = model_b.recurrent_weight_matrix().detach().cpu().numpy()
    signs = match["signs"]
    columns = match["permutation"]
    permuted = right[np.ix_(columns, columns)] * np.outer(signs, signs)
    scale = float(np.linalg.norm(left) * np.linalg.norm(permuted)) ** 0.5
    return float(np.linalg.norm(left - permuted) / scale) if scale > 0 else float("nan")


#: The battery, ordered by how much each measure is blind to. ``kind`` says whether high
#: means agreement or disagreement, which the figures need in order to orient an axis.
AGREEMENT_BATTERY: tuple[tuple[str, str, str], ...] = (
    ("matched weight correlation", "signed permutation", "similarity"),
    ("matched weight distance", "signed permutation", "distance"),
    ("operator conjugation distance", "orthogonal conjugation", "distance"),
    ("eigenspectrum agreement", "any similarity transform", "similarity"),
    ("eigenspectrum distance", "any similarity transform", "distance"),
    ("state Procrustes", "orthogonal transform", "similarity"),
    ("state CKA", "orthogonal transform + scaling", "similarity"),
)


def agreement_battery(
    run_dirs: Sequence[str | Path],
    *,
    device: str = "cpu",
    include_operator_distance: bool = True,
) -> pd.DataFrame:
    """Every measure in :data:`AGREEMENT_BATTERY`, for every pair of fits.

    One row per (pair, measure). Pair with :func:`untrained_agreement_battery` for the
    floor; a measure whose fitted value does not clear its own floor is reporting the
    architecture rather than the data, however agreeable its raw value looks.
    """
    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_circuit import (
        eigenspectrum_agreement,
        linear_cka,
        matched_unit_alignment,
        procrustes_similarity,
        recurrent_eigenspectrum,
    )

    loaded = []
    for run_dir in run_dirs:
        replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
        states = replay["h_seq"].detach().cpu().numpy()
        loaded.append(
            {
                "model": replay["model"],
                "weight": replay["model"].recurrent_weight_matrix().detach().cpu().numpy(),
                "states": states.reshape(-1, states.shape[-1]),
                "spectrum": recurrent_eigenspectrum(replay["model"]),
            }
        )
    return _battery_rows(loaded, include_operator_distance, label="fitted")


def untrained_agreement_battery(
    run_dir: str | Path,
    *,
    n_draws: int = 5,
    device: str = "cpu",
    seed: int = 0,
    include_operator_distance: bool = True,
) -> pd.DataFrame:
    """The battery between untrained models of this architecture -- the floor for all of it."""
    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_circuit import recurrent_eigenspectrum
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
        _model_spec_from_checkpoint,
        resolve_device,
    )
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_model import FixationMRNNModel

    resolved = resolve_device(device)
    _, checkpoint = load_fixation_mrnn_checkpoint(Path(run_dir), device=device)
    spec = _model_spec_from_checkpoint(checkpoint, device=resolved)
    inp = torch.as_tensor(checkpoint["input_tensor"], dtype=torch.float32, device=resolved)
    h0 = checkpoint["h0"].to(resolved)

    loaded = []
    for draw in range(int(n_draws)):
        torch.manual_seed(int(seed) + 1000 + draw)
        model = FixationMRNNModel(spec).to(resolved).eval()
        with torch.no_grad():
            states = model(inp, h0)["h_seq"].detach().cpu().numpy()
        loaded.append(
            {
                "model": model,
                "weight": model.recurrent_weight_matrix().detach().cpu().numpy(),
                "states": states.reshape(-1, states.shape[-1]),
                "spectrum": recurrent_eigenspectrum(model),
            }
        )
    return _battery_rows(loaded, include_operator_distance, label="untrained")


def _battery_rows(loaded: Sequence[Mapping[str, object]], operator: bool, *, label: str) -> pd.DataFrame:
    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_circuit import (
        eigenspectrum_agreement,
        linear_cka,
        matched_unit_alignment,
        procrustes_similarity,
    )

    rows: list[dict[str, object]] = []
    for i, j in combinations(range(len(loaded)), 2):
        left, right = loaded[i], loaded[j]
        match = matched_unit_alignment(left["model"], right["model"])
        values = {
            "matched weight correlation": match["matched_weight_correlation"],
            "matched weight distance": matched_weight_distance(left["model"], right["model"]),
            "eigenspectrum agreement": eigenspectrum_agreement(left["spectrum"], right["spectrum"]),
            "eigenspectrum distance": eigenspectrum_distance(left["spectrum"], right["spectrum"]),
            "state Procrustes": procrustes_similarity(left["states"], right["states"]),
            "state CKA": linear_cka(left["states"], right["states"]),
        }
        if operator:
            values["operator conjugation distance"] = similarity_transform_distance(
                left["weight"], right["weight"]
            )
        for measure, value in values.items():
            rows.append({"pair": f"{i}-{j}", "measure": measure, "value": float(value),
                         "ensemble": label})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# The invariant screen
# --------------------------------------------------------------------------------------


def condition_properties(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """Candidate condition-contrastive properties of one fit, one row per condition.

    Every entry is a within-model, between-condition quantity, so the signed permutation
    cancels and no alignment step is needed. They divide into two kinds and the screen
    keeps them apart:

    *Scale-carrying* -- ``inter_current_magnitude``, ``state_extent``, ``state_speed``.
    These track how large the condition's target is, so a difference between conditions is
    expected of any adequate fit.

    *Scale-free* -- ``drive_alignment`` (a cosine), ``current_dimensionality`` and
    ``inter_fraction`` (ratios). A difference here is not forced by the target's size, and
    only ``drive_alignment`` has no counterpart that can be computed on the target at all.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    regions = list(replay["region_order"])
    conditions = [str(c) for c in replay["condition_order"]]
    currents = extract_region_current_vectors(replay)
    states = replay["h_seq"].detach().cpu().numpy()
    drive = current_drive_alignment(replay).groupby("condition")["cosine_with_intrinsic_drive"].mean()

    rows: list[dict[str, object]] = []
    for index, condition in enumerate(conditions):
        cross = [(s, t) for s in regions for t in regions if s != t and (s, t) in currents]
        inter = float(np.mean([np.linalg.norm(currents[k].numpy()[index], axis=-1).mean() for k in cross]))
        own = float(np.mean([
            np.linalg.norm(currents[(r, r)].numpy()[index], axis=-1).mean()
            for r in regions if (r, r) in currents
        ]))
        ratios = []
        for key in cross:
            block = currents[key].numpy()[index]
            block = block - block.mean(axis=0, keepdims=True)
            energy = np.linalg.svd(block, compute_uv=False) ** 2
            ratios.append(float(energy.sum() ** 2 / (energy**2).sum()) if energy.sum() > 0 else np.nan)
        h = states[index]
        rows.append({
            "seed": Path(run_dir).name.replace("seed=", ""),
            "condition": condition,
            "drive_alignment": float(drive.get(condition, np.nan)),
            "current_dimensionality": float(np.nanmean(ratios)),
            "inter_fraction": inter / (inter + own) if (inter + own) > 0 else np.nan,
            "inter_current_magnitude": inter,
            "state_extent": float(np.linalg.norm(h - h.mean(axis=0), axis=-1).mean()),
            "state_speed": float(np.linalg.norm(np.diff(h, axis=0), axis=-1).mean()),
        })
    return pd.DataFrame(rows)


def invariant_screen(arm_dirs: Mapping[str, str | Path], *, device: str = "cpu") -> pd.DataFrame:
    """:func:`condition_properties` for every fit of every architecture."""
    frames = []
    for label, arm in arm_dirs.items():
        for run_dir in seed_run_dirs(arm):
            frame = condition_properties(run_dir, device=device)
            frame.insert(0, "arm", label)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


#: Properties in the screen, and whether the target can be asked the same question. A
#: property the target also has is a consistency check on the fit; only ``drive_alignment``
#: is defined for a model and undefined for the data.
SCREEN_PROPERTIES: tuple[tuple[str, str, str], ...] = (
    ("drive_alignment", "scale-free", "none — model only"),
    ("current_dimensionality", "scale-free", "target_dimensionality"),
    ("inter_fraction", "scale-free", "none — model only"),
    ("inter_current_magnitude", "scale-carrying", "target_extent"),
    ("state_extent", "scale-carrying", "target_extent"),
    ("state_speed", "scale-carrying", "target_speed"),
)


def screen_consistency(screen: pd.DataFrame, properties: Sequence[str] | None = None) -> pd.DataFrame:
    """For each property, the fraction of fits in which interactive face is the extreme one.

    The screen's verdict. A property whose condition ordering holds in every fit of every
    architecture is a property of the solution *set*, which is a far stronger statement
    than significance across the seeds of one arm.
    """
    names = list(properties or [name for name, _, _ in SCREEN_PROPERTIES])
    rows: list[dict[str, object]] = []
    for name in names:
        wide = screen.pivot_table(index=["arm", "seed"], columns="condition", values=name)
        wide = wide[[c for c in CONDITION_ORDER if c in wide.columns]].dropna()
        if wide.empty:
            continue
        reference = wide["face_non_interactive"] if "face_non_interactive" in wide else np.nan
        rows.append({
            "property": name,
            "n_fits": int(len(wide)),
            "fi_lowest": float((wide.idxmin(axis=1) == "face_interactive").mean()),
            "fi_highest": float((wide.idxmax(axis=1) == "face_interactive").mean()),
            "fi_over_fn_median": float((wide["face_interactive"] / reference).median()),
            "fi_over_fn_min": float((wide["face_interactive"] / reference).min()),
            "fi_over_fn_max": float((wide["face_interactive"] / reference).max()),
        })
    return pd.DataFrame(rows)


def target_side_control(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """The screen's scale-carrying quantities, computed on the target instead of the model.

    The control that separates a finding from an echo. If interactive face is smaller,
    slower and lower-dimensional in the *data*, then a fit which reproduces the data must
    show the same, and showing it is a check that the fit works rather than a discovery
    about routing.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    conditions = [str(c) for c in replay["condition_order"]]
    rows: list[dict[str, object]] = []
    for region, target in _target_by_region(replay).items():
        for index, condition in enumerate(conditions):
            block = target[index]
            centred = block - block.mean(axis=0, keepdims=True)
            energy = np.linalg.svd(centred, compute_uv=False) ** 2
            rows.append({
                "region": region,
                "condition": condition,
                "target_extent": float(np.linalg.norm(centred, axis=1).mean()),
                "target_speed": float(np.linalg.norm(np.diff(block, axis=0), axis=1).mean()),
                "target_dimensionality": float(energy.sum() ** 2 / (energy**2).sum()),
                "target_variance": float((centred**2).sum()),
            })
    return pd.DataFrame(rows)


def readout_state_truncation(run_dir: str | Path, ks: Sequence[int], *, device: str = "cpu") -> pd.DataFrame:
    """Fit retained when a region's state is truncated to its top ``k`` directions.

    Answers whether a wide region is actually using its width. The state is projected onto
    its own leading principal directions and pushed through the model's own readout,
    **bias included** -- omitting it makes the intact model score below zero and the whole
    curve meaningless.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    slices = replay["model"].hidden_region_slices()
    heads = replay["model"].output_heads
    states = replay["h_seq"].detach().cpu().numpy()
    rows: list[dict[str, object]] = []
    for region, target in _target_by_region(replay).items():
        weight = heads[region].weight.detach().cpu().numpy()
        bias = heads[region].bias.detach().cpu().numpy()
        observed = target.reshape(-1, target.shape[-1])
        block = states[:, :, slices[region]].reshape(-1, states[:, :, slices[region]].shape[-1])
        centre = block.mean(axis=0)
        u, s, vt = np.linalg.svd(block - centre, full_matrices=False)
        ss_tot = float(((observed - observed.mean(axis=0)) ** 2).sum())
        for k in ks:
            k = min(int(k), s.size)
            reconstructed = (u[:, :k] * s[:k]) @ vt[:k] + centre
            predicted = reconstructed @ weight.T + bias
            rows.append({
                "seed": Path(run_dir).name.replace("seed=", ""),
                "region": region,
                "k": k,
                "r2": 1.0 - float(((observed - predicted) ** 2).sum()) / ss_tot,
            })
    return pd.DataFrame(rows)


__all__ += [
    "AGREEMENT_BATTERY",
    "SCREEN_PROPERTIES",
    "agreement_battery",
    "condition_properties",
    "eigenspectrum_distance",
    "invariant_screen",
    "matched_weight_distance",
    "readout_state_truncation",
    "screen_consistency",
    "similarity_transform_distance",
    "target_side_control",
    "untrained_agreement_battery",
]


def operator_distance_resolution(
    weight: np.ndarray,
    *,
    n_draws: int = 3,
    seed: int = 0,
    **kwargs,
) -> float:
    """The floor of :func:`similarity_transform_distance` on this size of matrix.

    Hands the optimiser a matrix and a random orthogonal conjugation of itself, which are
    the same operator and should score zero. What it actually returns is the measure's
    resolution: a fitted pair scoring at or below this is indistinguishable from "the same
    dynamics", and one scoring well above it is genuinely further apart. Reporting a
    non-convex distance without this number invites reading optimiser slack as a result.
    """
    generator = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(n_draws)):
        rotation, _ = np.linalg.qr(generator.normal(size=weight.shape))
        values.append(similarity_transform_distance(weight, rotation @ weight @ rotation.T, **kwargs))
    return float(np.mean(values))


__all__ += ["operator_distance_resolution"]


# ======================================================================================
# Lesions on a trained network, scored so the comparisons are interpretable
# ======================================================================================
#
# Three corrections separate this from the lesion analysis the chapter carried before.
#
# *Comparable units.* ``reconstruction_accuracy`` divides by each condition's own variance,
# and interactive face has about half the variance of the other two, so an identical
# perturbation scored twice the damage. Every per-condition comparison here is in absolute
# squared error, and R^2 is reported only against a denominator shared across conditions.
#
# *A matched random control.* Under a sparsity mask, blocks keep different numbers of
# connections by chance. Silencing a pathway removes k live weights; the control removes k
# live weights drawn at random from elsewhere. Without it, "this pathway matters" can mean
# no more than "this pathway kept more weights".
#
# *The statistic is the ranking, not the damage.* Ten networks can disagree about every
# weight and still agree that severing one pair costs more than severing another. That is a
# claim about routing which does not require the routing to be identifiable, and it is the
# strongest form of result still available once the weight-level gate has failed.


def _lesion_groups(regions: Sequence[str]) -> list[tuple[str, str, tuple[tuple[str, str], ...]]]:
    """``(kind, name, blocks)`` for every lesion the battery runs."""
    groups: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
    for source in regions:
        for target in regions:
            if source != target:
                groups.append(("directed", f"{source}→{target}", ((source, target),)))
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            groups.append(("bidirectional", f"{a}↔{b}", ((a, b), (b, a))))
    for region in regions:
        groups.append(("within-region", f"{region} self", ((region, region),)))
    for region in regions:
        blocks = tuple((s, t) for s in regions for t in regions
                       if s != t and region in (s, t))
        groups.append(("isolation", f"{region} isolated", blocks))
    return groups


def _live_entries(replay: Mapping[str, object], blocks: Sequence[tuple[str, str]]) -> int:
    """Non-zero weights inside a set of blocks -- the size a random control has to match."""
    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_circuit import recurrent_blocks

    all_blocks = recurrent_blocks(replay["model"])
    return int(sum(int(np.count_nonzero(all_blocks[key])) for key in blocks if key in all_blocks))


def lesion_battery(
    run_dir: str | Path,
    *,
    device: str = "cpu",
    n_random_controls: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Damage from every directed, bidirectional, within-region and whole-region lesion.

    Reported per region and condition in absolute squared error, alongside the number of
    live weights the lesion removed and a matched random control of the same size. The
    control is what makes the comparison between lesions interpretable under a sparsity
    mask: a pathway that kept more connections will do more damage for that reason alone.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    regions = list(replay["region_order"])
    conditions = [str(c) for c in replay["condition_order"]]
    targets = _target_by_region(replay)
    ss_tot_all = float(sum(((targets[r] - targets[r].mean(axis=(0, 1))) ** 2).sum() for r in regions))
    intact = _sse_by_region_condition(replay)

    rows: list[dict[str, object]] = []
    for kind, name, blocks in _lesion_groups(regions):
        lesioned = _sse_by_region_condition(
            replay_fixation_mrnn_run_with_ablations(
                Path(run_dir), ablations=list(blocks), device=device
            )
        )
        removed = _live_entries(replay, blocks)
        for region in regions:
            for condition in conditions:
                key = (region, condition)
                rows.append({
                    "seed": Path(run_dir).name.replace("seed=", ""),
                    "lesion_kind": kind,
                    "lesion": name,
                    "region": region,
                    "condition": condition,
                    "live_weights_removed": removed,
                    "damage_sse": lesioned[key] - intact[key],
                    # One denominator for every condition, so a condition with a smaller
                    # target does not register a larger effect for that reason alone.
                    "damage_shared_r2": (lesioned[key] - intact[key]) / ss_tot_all,
                })
    frame = pd.DataFrame(rows)

    controls = _random_lesion_controls(
        Path(run_dir), sorted(set(frame["live_weights_removed"])), replay=replay,
        n_draws=n_random_controls, seed=seed, device=device, intact=intact,
        ss_tot_all=ss_tot_all, regions=regions, conditions=conditions,
    )
    return pd.concat([frame, controls], ignore_index=True)


def _random_lesion_controls(
    run_dir: Path,
    sizes: Sequence[int],
    *,
    replay: Mapping[str, object],
    n_draws: int,
    seed: int,
    device: str,
    intact: Mapping[tuple[str, str], float],
    ss_tot_all: float,
    regions: Sequence[str],
    conditions: Sequence[str],
) -> pd.DataFrame:
    """Damage from silencing ``k`` randomly chosen live weights, for each ``k`` observed.

    The null every lesion is read against. Implemented by zeroing entries of the assembled
    recurrent matrix directly rather than by block, because the point is to remove the same
    *number* of connections without respecting the block structure.
    """
    model = replay["model"]
    weight = model.recurrent_weight_matrix().detach().clone()
    live = torch.nonzero(weight != 0, as_tuple=False)
    generator = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    inp, h0 = replay["inp"], replay["h0"]

    for size in sizes:
        if size <= 0 or size > live.shape[0]:
            continue
        for draw in range(int(n_draws)):
            choice = generator.choice(live.shape[0], size=int(size), replace=False)
            masked = weight.clone()
            masked[live[choice, 0], live[choice, 1]] = 0.0
            with torch.no_grad():
                h_seq = _replay_with_weight(model, masked, inp, h0)
                predicted = _outputs_from_states(model, h_seq)
            for region in regions:
                target = np.asarray(replay["checkpoint"]["target_by_region"][region], dtype=float)
                for index, condition in enumerate(conditions):
                    sse = float(((target[index] - predicted[region][index]) ** 2).sum())
                    rows.append({
                        "seed": run_dir.name.replace("seed=", ""),
                        "lesion_kind": "random control",
                        "lesion": f"random {size} ({draw})",
                        "region": region,
                        "condition": condition,
                        "live_weights_removed": int(size),
                        "damage_sse": sse - intact[(region, condition)],
                        "damage_shared_r2": (sse - intact[(region, condition)]) / ss_tot_all,
                    })
    return pd.DataFrame(rows)


def _replay_with_weight(model, weight: torch.Tensor, inp: torch.Tensor, h0: torch.Tensor):
    """Roll the model forward with a substituted recurrent matrix."""
    states = []
    h = h0
    for step in range(inp.shape[1]):
        drive = h @ weight.T
        if model.mrnn.inp_constrained:
            w_inp = model.mrnn.apply_dales_law(
                model.mrnn.W_inp, model.mrnn.W_inp_mask, model.mrnn.W_inp_sign_matrix
            )
        else:
            w_inp = model.mrnn.W_inp * model.mrnn.W_inp_mask
        h = model.mrnn.activation(drive + inp[:, step] @ w_inp.T + model.mrnn.tonic_inp)
        states.append(h)
    return torch.stack(states, dim=1)


def _outputs_from_states(model, h_seq: torch.Tensor) -> dict[str, np.ndarray]:
    slices = model.hidden_region_slices()
    return {
        region: model.output_heads[region](h_seq[..., slices[region]]).detach().cpu().numpy()
        for region in model.region_order
    }


def lesion_ranking_agreement(
    lesions: pd.DataFrame,
    *,
    n_permutations: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Do independently seeded fits agree on which lesions hurt most?

    Kendall's :math:`\\tau` between every pair of seeds' damage rankings, per lesion family
    and per arm, against a null of randomly permuted rankings. This is the question the
    ensemble exists to answer: a ranking can be reproducible where the weights are not, and
    if it is, the chapter can rank pathways without claiming to have recovered a circuit.
    """
    from scipy.stats import kendalltau

    generator = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    grouped = lesions[lesions["lesion_kind"] != "random control"]
    keys = ["arm", "lesion_kind"] if "arm" in grouped.columns else ["lesion_kind"]

    for key, block in grouped.groupby(keys):
        wide = block.groupby(["seed", "lesion"])["damage_sse"].sum().unstack("lesion").dropna(axis=1)
        if wide.shape[0] < 2 or wide.shape[1] < 3:
            continue
        observed = [
            float(kendalltau(wide.iloc[i], wide.iloc[j]).statistic)
            for i, j in combinations(range(wide.shape[0]), 2)
        ]
        null = []
        values = wide.values
        for _ in range(int(n_permutations)):
            i, j = generator.choice(values.shape[0], size=2, replace=False)
            null.append(float(kendalltau(values[i], generator.permutation(values[j])).statistic))
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        entry.update({
            "n_seeds": int(wide.shape[0]),
            "n_lesions": int(wide.shape[1]),
            "tau_mean": float(np.mean(observed)),
            "tau_min": float(np.min(observed)),
            "null_mean": float(np.mean(null)),
            "null_p95": float(np.percentile(null, 95)),
            "clears_null": bool(np.mean(observed) > np.percentile(null, 95)),
        })
        rows.append(entry)
    return pd.DataFrame(rows)


def lesion_condition_contrast(lesions: pd.DataFrame) -> pd.DataFrame:
    """Per-condition lesion damage in absolute units, and whether the seeds agree on it.

    The version of "interactive face is the condition coupling is for" that survives the
    variance confound. ``fraction_of_fits_highest`` is the statistic to read: a mean
    difference between conditions can be carried by one fit, a fraction cannot.
    """
    block = lesions[lesions["lesion_kind"] != "random control"]
    keys = ["arm"] if "arm" in block.columns else []
    rows: list[dict[str, object]] = []
    for key, chunk in (block.groupby(keys) if keys else [((), block)]):
        per_fit = chunk.groupby(["seed", "lesion", "condition"])["damage_sse"].sum()
        wide = per_fit.unstack("condition")
        conditions = [c for c in CONDITION_ORDER if c in wide.columns]
        wide = wide[conditions].dropna()
        if wide.empty:
            continue
        by_seed = chunk.groupby(["seed", "condition"])["damage_sse"].sum().unstack("condition")[conditions]
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for condition in conditions:
            entry[f"{condition}__mean_sse"] = float(wide[condition].mean())
        entry["fraction_of_fits_fi_highest"] = float(
            (by_seed.idxmax(axis=1) == "face_interactive").mean()
        )
        entry["fi_over_fn"] = float(
            (by_seed["face_interactive"] / by_seed["face_non_interactive"]).median()
        )
        entry["n_seeds"] = int(by_seed.shape[0])
        rows.append(entry)
    return pd.DataFrame(rows)


__all__ += [
    "lesion_battery",
    "lesion_condition_contrast",
    "lesion_ranking_agreement",
]


# ======================================================================================
# The final series: the region ladder and the within x cross rank grid
# ======================================================================================
#
# Both are fitted on the minimax objective (every region x condition cell scored by its
# unexplained fraction, the worst cell optimised) and scored on the ceiling-matched R^2, so
# neither can be compared numerically with the ``chapter/`` tree. They answer two questions
# the earlier series could not:
#
# *Does a region need the network to reproduce itself?*  Fit each region alone, in every
# pair, in every triple, and in the full network -- eight points per region -- with the same
# 40-unit readout at every rung, so the rank cap is constant and any gain is dynamical.
#
# *Which is the low-dimensional channel, a region's own recurrence or its input from the
# others?*  A grid over within-region rank and cross-region rank, with the two marginals
# (one side dense) that separate the pure costs, read against total drive rank
# ``r_within + 3 r_cross`` so the three-to-one counting of cross blocks is not mistaken
# for a finding.

from itertools import combinations as _combinations


def ladder_variants(
    region_order: Sequence[str],
    *,
    base_overrides: Mapping[str, object],
) -> list["ModelVariant"]:
    """One ``ModelVariant`` per subset of regions: singles, pairs, triples, and the full set.

    The variant's ``region_order`` override is what makes a subset fit: the target builder
    fits each region's PCA on its own units and pools the normalisation scale over every
    recorded region, so a region's target is identical at every rung.
    """
    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import ModelVariant

    regions = tuple(region_order)
    variants: list[ModelVariant] = []
    for size in range(1, len(regions) + 1):
        for subset in _combinations(regions, size):
            if size == len(regions):
                label, arm = "full", "full network"
            elif size == 1:
                label, arm = f"single_{subset[0]}", "alone"
            elif size == len(regions) - 1:
                missing = next(r for r in regions if r not in subset)
                label, arm = f"triple_no_{missing}", "three regions"
            else:
                label, arm = "pair_" + "_".join(subset), "pair"
            variants.append(ModelVariant(
                label=label, arm=arm,
                overrides={**dict(base_overrides), "region_order": tuple(subset)},
            ))
    return variants


def annotate_ladder(fit: pd.DataFrame, region_order: Sequence[str]) -> pd.DataFrame:
    """Add ``n_partners`` and ``partners`` to a per-(label, seed, region, condition) fit table.

    ``n_partners`` is how many *other* regions the scored region was fitted alongside, which
    is the ladder's x-axis; ``partners`` names them, which is what the partner-specific
    panels read.
    """
    regions = tuple(region_order)

    def members(label: str) -> tuple[str, ...]:
        if label == "full":
            return regions
        if label.startswith("single_"):
            return (label.removeprefix("single_"),)
        if label.startswith("triple_no_"):
            missing = label.removeprefix("triple_no_")
            return tuple(r for r in regions if r != missing)
        if label.startswith("pair_"):
            return tuple(label.removeprefix("pair_").split("_"))
        raise ValueError(f"not a ladder label: {label!r}")

    out = fit.copy()
    out["members"] = out["label"].map(lambda l: members(str(l)))
    out["n_partners"] = out["members"].map(len) - 1
    out["partners"] = [
        tuple(m for m in mem if m != reg) for mem, reg in zip(out["members"], out["region"])
    ]
    return out.drop(columns=["members"])


#: Frequency bands the recovery is reported in. 10 ms bins, so Nyquist is 50 Hz.
RECOVERY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-5 Hz", 0.0, 5.0),
    ("5-10 Hz", 5.0, 10.0),
    ("10-20 Hz", 10.0, 20.0),
    ("20-50 Hz", 20.0, 50.0),
)


def band_resolved_recovery(
    run_dir: str | Path,
    *,
    bin_size_s: float = 0.01,
    bands: Sequence[tuple[str, float, float]] = RECOVERY_BANDS,
    device: str = "cpu",
) -> pd.DataFrame:
    """Fraction of the target's power in each band that the fit fails to reproduce.

    ``residual_fraction`` is the residual's band power over the target's, per region. Total
    R^2 is dominated by slow, high-variance structure that a region reproduces on its own;
    this is where the network's contribution actually shows, which is why the ladder reports
    it beside R^2 rather than instead of it.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    frequencies = None
    rows: list[dict[str, object]] = []
    for region, target in _target_by_region(replay).items():
        predicted = replay["output_by_region"][region].detach().cpu().numpy().astype(float)
        centred = target - target.mean(axis=1, keepdims=True)
        residual = target - predicted
        spectrum_target = (np.abs(np.fft.rfft(centred, axis=1)) ** 2).sum(axis=(0, 2))
        spectrum_residual = (np.abs(np.fft.rfft(residual, axis=1)) ** 2).sum(axis=(0, 2))
        if frequencies is None:
            frequencies = np.fft.rfftfreq(target.shape[1], d=float(bin_size_s))
        for name, low, high in bands:
            keep = (frequencies >= low) & (frequencies < high)
            total = float(spectrum_target[keep].sum())
            rows.append({
                "seed": Path(run_dir).name.replace("seed=", ""),
                "region": region,
                "band": name,
                "target_power": total,
                "residual_fraction": float(spectrum_residual[keep].sum() / total) if total > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def rank_grid_variants(
    ranks: Sequence[int],
    *,
    base_overrides: Mapping[str, object],
    include_marginals: bool = True,
) -> list["ModelVariant"]:
    """Every (within rank, cross rank) cell, plus the two marginals with one side dense.

    Labels are ``w{r_w}_c{r_c}``; ``dense`` stands in for an unconstrained side. The fully
    dense corner is not generated -- it is the ladder's ``full`` arm, read from there.
    """
    from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import ModelVariant

    def cell(r_w, r_c):
        label = f"w{'dense' if r_w is None else r_w}_c{'dense' if r_c is None else r_c}"
        if r_w is None:
            arm = "cross-region marginal"
        elif r_c is None:
            arm = "within-region marginal"
        else:
            arm = "grid"
        return ModelVariant(label=label, arm=arm, overrides={
            **dict(base_overrides),
            "within_region_bottleneck_dim": None if r_w is None else int(r_w),
            "recurrent_bottleneck_dim": None if r_c is None else int(r_c),
        })

    variants = [cell(r_w, r_c) for r_w in ranks for r_c in ranks]
    if include_marginals:
        variants += [cell(None, r_c) for r_c in ranks]
        variants += [cell(r_w, None) for r_w in ranks]
    return variants


def annotate_rank_grid(fit: pd.DataFrame, *, hidden_units: int, n_regions: int = 4) -> pd.DataFrame:
    """Add ``rank_within``, ``rank_cross`` and ``drive_rank`` to a fit table keyed by label.

    ``drive_rank`` is ``r_within + (n_regions - 1) * r_cross``, the bound on the rank of a
    region's pre-activation. Dense sides count as the full width. Fit plotted against this
    single number is the check on whether "cross matters more" is anything beyond counting.
    """
    def parse(label: str) -> tuple[int, int]:
        body = str(label)
        if body == "full":
            return hidden_units, hidden_units
        w, c = body.removeprefix("w").split("_c")
        return (hidden_units if w == "dense" else int(w), hidden_units if c == "dense" else int(c))

    out = fit.copy()
    parsed = out["label"].map(parse)
    out["rank_within"] = parsed.map(lambda t: t[0])
    out["rank_cross"] = parsed.map(lambda t: t[1])
    out["drive_rank"] = out["rank_within"] + (int(n_regions) - 1) * out["rank_cross"]
    return out


def select_bottleneck(adequacy: pd.DataFrame, *, hidden_units: int, bar: float) -> dict[str, object] | None:
    """The tightest grid cell whose worst region x condition cell clears ``bar``.

    Tightest means lowest drive rank, then lowest cross rank -- the cross constraint being
    the one the chapter is about. Marginals are excluded: the selected model constrains both
    sides, and the marginals exist to interpret it, not to be it.
    """
    table = annotate_rank_grid(adequacy, hidden_units=hidden_units)
    table = table[(table["rank_within"] < hidden_units) & (table["rank_cross"] < hidden_units)]
    passing = table[table["worst_condition"] >= float(bar)]
    if passing.empty:
        return None
    pick = passing.sort_values(["drive_rank", "rank_cross", "rank_within"]).iloc[0]
    return {
        "label": str(pick["label"]),
        "within_region_bottleneck_dim": int(pick["rank_within"]),
        "recurrent_bottleneck_dim": int(pick["rank_cross"]),
        "drive_rank": int(pick["drive_rank"]),
        "worst_condition": float(pick["worst_condition"]),
        "selection_rule": f"lowest drive rank with worst cell >= {bar} of ceiling; ties to lower cross rank",
    }


__all__ += [
    "RECOVERY_BANDS",
    "annotate_ladder",
    "annotate_rank_grid",
    "band_resolved_recovery",
    "ladder_variants",
    "rank_grid_variants",
    "select_bottleneck",
]
