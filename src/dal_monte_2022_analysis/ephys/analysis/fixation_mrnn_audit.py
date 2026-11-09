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


def band_resolved_recovery_by_condition(
    run_dir: str | Path,
    *,
    bin_size_s: float = 0.01,
    bands: Sequence[tuple[str, float, float]] = RECOVERY_BANDS,
    device: str = "cpu",
) -> pd.DataFrame:
    """:func:`band_resolved_recovery`, kept split by condition instead of pooled over it.

    ``band_resolved_recovery`` sums the spectrum over the condition axis before dividing,
    so a band recovered well in two conditions and poorly in a third reads as "recovered
    reasonably". This computes the same residual-over-target band power per condition, which
    is what a claim about interactive face specifically, rather than fixation PSTHs in
    general, needs.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    conditions = list(replay["condition_order"])
    frequencies = None
    rows: list[dict[str, object]] = []
    for region, target in _target_by_region(replay).items():
        predicted = replay["output_by_region"][region].detach().cpu().numpy().astype(float)
        for index, condition in enumerate(conditions):
            target_c = target[index]
            residual_c = target_c - predicted[index]
            centred = target_c - target_c.mean(axis=0, keepdims=True)
            spectrum_target = (np.abs(np.fft.rfft(centred, axis=0)) ** 2).sum(axis=1)
            spectrum_residual = (np.abs(np.fft.rfft(residual_c, axis=0)) ** 2).sum(axis=1)
            if frequencies is None:
                frequencies = np.fft.rfftfreq(target_c.shape[0], d=float(bin_size_s))
            for name, low, high in bands:
                keep = (frequencies >= low) & (frequencies < high)
                total = float(spectrum_target[keep].sum())
                rows.append({
                    "seed": Path(run_dir).name.replace("seed=", ""),
                    "region": region,
                    "condition": str(condition),
                    "band": name,
                    "target_power": total,
                    "residual_fraction": float(spectrum_residual[keep].sum() / total) if total > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def self_vs_cross_band_recovery(
    recovery: pd.DataFrame, *, self_rung: int = 0, cross_rung: int = 3,
) -> pd.DataFrame:
    """Per-band residual fraction, region alone ("self") vs. the full network ("cross"), one row per seed and band.

    ``recovery`` is :func:`band_resolved_recovery`'s output concatenated across rungs, with
    an ``n_partners`` column added (as the ladder notebook does). ``self_rung`` (default 0,
    no cross-region connections at all) and ``cross_rung`` (default 3, every other region
    present) pick the two endpoints of the ladder rather than every rung in between, for a
    direct test of whether cross-region connections recover more power in fast bands than
    slow ones. Regions are pooled per seed, so the contrast is pooled over region rather than
    split by it as well -- the per-region, per-rung curve is :func:`plot_band_recovery_by_rung`'s.
    """
    block = recovery[recovery["n_partners"].isin([self_rung, cross_rung])].copy()
    block["arm"] = block["n_partners"].map({self_rung: "self", cross_rung: "cross"})
    return block.groupby(["arm", "seed", "band"])["residual_fraction"].mean().reset_index()


__all__ += ["band_resolved_recovery_by_condition", "self_vs_cross_band_recovery"]


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


# ======================================================================================
# Flow over time, and what the bottleneck does to it
# ======================================================================================
#
# The current from region s into region t at time tau is c_{s->t}(tau) = W[t,s] h_s(tau).
# Summarised as a scalar it is what the rebuild called "current magnitude"; resolved in time
# it is the closest thing the model has to a picture of information flow -- how much each
# source drives each target, when, and for which fixation type. Every quantity here is
# within-model, so it is free of the relabelling ambiguity and comparable across fits, and
# each is computed for the self block too, so a region's total drive can be decomposed into
# its own recurrence and each partner's contribution at every time step.


def time_resolved_flow(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """``|c_{s->t}(tau)|`` for every source, target, condition and time bin, self blocks included."""
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    currents = extract_region_current_vectors(replay)
    timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
    conditions = [str(c) for c in replay["condition_order"]]
    seed = Path(run_dir).name.replace("seed=", "")
    frames = []
    for (source, target), current in currents.items():
        norm = np.linalg.norm(current.numpy(), axis=-1)  # (condition, time)
        for index, condition in enumerate(conditions):
            frames.append(pd.DataFrame({
                "seed": seed, "source": source, "target": target, "condition": condition,
                "time_s": timeline[: norm.shape[1]], "current_norm": norm[index, : timeline.size],
            }))
    return pd.concat(frames, ignore_index=True)


def time_resolved_alignment(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """Cosine between the incoming current from ``source`` and the target's own recurrent drive, in time.

    The time-resolved form of the drive-alignment invariant: positive when a partner pushes
    a region along the direction its own recurrence was already taking it, negative when it
    opposes. Resolved in time so the question "when does the network push with or against a
    region" can be asked per fixation type.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    currents = extract_region_current_vectors(replay)
    timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
    conditions = [str(c) for c in replay["condition_order"]]
    regions = list(replay["region_order"])
    seed = Path(run_dir).name.replace("seed=", "")
    frames = []
    for target in regions:
        own = currents[(target, target)].numpy()
        for source in regions:
            if source == target:
                continue
            incoming = currents[(source, target)].numpy()
            numerator = (incoming * own).sum(axis=-1)
            denominator = np.linalg.norm(incoming, axis=-1) * np.linalg.norm(own, axis=-1)
            cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
            for index, condition in enumerate(conditions):
                frames.append(pd.DataFrame({
                    "seed": seed, "source": source, "target": target, "condition": condition,
                    "time_s": timeline[: cosine.shape[1]], "cosine": cosine[index, : timeline.size],
                }))
    return pd.concat(frames, ignore_index=True)


def flow_decomposition(flow: pd.DataFrame) -> pd.DataFrame:
    """Per (target, condition, time): self drive, cross drive, and the cross fraction of total drive energy.

    ``cross_fraction`` is ``|cross|^2 / (|self|^2 + |cross|^2)`` where the cross norm pools the
    three partners. This is the time course of how much of what a region receives comes from
    the rest of the network rather than from itself.
    """
    keys = [k for k in ("label", "seed") if k in flow.columns] + ["target", "condition", "time_s"]
    own = flow[flow["source"] == flow["target"]].set_index(keys)["current_norm"].rename("self_norm")
    cross = (flow[flow["source"] != flow["target"]]
             .assign(sq=lambda d: d["current_norm"] ** 2)
             .groupby(keys)["sq"].sum().pipe(np.sqrt).rename("cross_norm"))
    out = pd.concat([own, cross], axis=1).reset_index()
    total = out["self_norm"] ** 2 + out["cross_norm"] ** 2
    out["cross_fraction"] = np.where(total > 0, out["cross_norm"] ** 2 / total, np.nan)
    return out


def region_state_properties(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """Per (region, condition): how many dimensions the drive and the state actually occupy.

    ``drive_pr`` is the participation ratio of a region's pre-activation over time -- bounded
    by ``r_within + 3 r_cross`` under the rank constraints -- and ``state_pr`` the same for the
    post-nonlinearity state. The readout needs ~39 linear dimensions of state, so the gap
    between the two is how much the tanh and the trained initial state re-expand a
    low-dimensional drive. ``cross_pr`` and ``self_pr`` split the drive by origin.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    currents = extract_region_current_vectors(replay)
    regions = list(replay["region_order"])
    conditions = [str(c) for c in replay["condition_order"]]
    slices = replay["model"].hidden_region_slices()
    states = replay["h_seq"].detach().cpu().numpy()
    seed = Path(run_dir).name.replace("seed=", "")

    def participation(block: np.ndarray) -> float:
        centred = block - block.mean(axis=0, keepdims=True)
        energy = np.linalg.svd(centred, compute_uv=False) ** 2
        return float(energy.sum() ** 2 / (energy ** 2).sum()) if energy.sum() > 0 else np.nan

    rows: list[dict[str, object]] = []
    for target in regions:
        own = currents[(target, target)].numpy()
        cross = sum(currents[(s, target)].numpy() for s in regions if s != target)
        for index, condition in enumerate(conditions):
            h = states[index][:, slices[target]]
            rows.append({
                "seed": seed, "region": target, "condition": condition,
                "state_pr": participation(h),
                "drive_pr": participation(own[index] + cross[index]),
                "self_pr": participation(own[index]),
                "cross_pr": participation(cross[index]),
                "state_speed": float(np.linalg.norm(np.diff(h, axis=0), axis=-1).mean()),
                "state_extent": float(np.linalg.norm(h - h.mean(axis=0), axis=-1).mean()),
                "cross_energy_fraction": float((cross[index] ** 2).sum() / ((own[index] ** 2).sum() + (cross[index] ** 2).sum())),
            })
    return pd.DataFrame(rows)


def collect_over_arms(arm_dirs: Mapping[str, str | Path], function, *, device: str = "cpu") -> pd.DataFrame:
    """Apply a per-run function to every seed of every arm, tagging rows with the arm label."""
    frames = []
    for label, arm in arm_dirs.items():
        for run_dir in seed_run_dirs(arm):
            frames.append(function(run_dir, device=device).assign(label=label))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ += [
    "collect_over_arms",
    "flow_decomposition",
    "region_state_properties",
    "time_resolved_alignment",
    "time_resolved_flow",
]


def flow_temporal_summary(decomposed: pd.DataFrame) -> pd.DataFrame:
    """Per (label, seed, target, condition): the mean cross share and how much it moves within a trial.

    ``cross_fraction_mean`` is how much of a region's drive comes from the network on average;
    ``cross_fraction_modulation`` is its standard deviation over time within the trial -- the
    scale-free measure of whether the network's contribution is steady or comes in bursts.
    A channel too narrow to carry everything at once shows up here as high modulation: the
    region alternates between driving itself and being driven.
    """
    keys = [k for k in ("label", "seed") if k in decomposed.columns] + ["target", "condition"]
    grouped = decomposed.groupby(keys)
    out = grouped.agg(
        cross_fraction_mean=("cross_fraction", "mean"),
        cross_fraction_modulation=("cross_fraction", "std"),
        cross_norm_mean=("cross_norm", "mean"),
        self_norm_mean=("self_norm", "mean"),
    ).reset_index()
    cv = grouped["cross_norm"].agg(lambda s: float(np.std(s) / np.mean(s)) if np.mean(s) > 0 else np.nan)
    out["cross_norm_cv"] = cv.values
    return out.rename(columns={"target": "region"})


__all__ += ["flow_temporal_summary"]


# --------------------------------------------------------------------------------------
# Matched-drive-rank within vs. cross contrast
# --------------------------------------------------------------------------------------
#
# Holding one side dense while the other is bottlenecked (the grid's own marginals, above)
# cannot be made parameter-matched in the regime that matters. With every region the same
# width h=40, a within-region block costs 2*r_w*h and a cross-region block 2*r_c*h, and
# with 4 regions there are 4 within blocks against 12 ordered cross blocks, so the two
# totals are 320*r_w and 960*r_c on this dataset -- cross costs exactly (n_regions - 1)
# times as much per unit rank as within, for any h. Dense within is 4*h*h = 6400 parameters,
# dense cross is 12*h*h = 19200: matching "cross bottlenecked, within dense" (6400 + 960 c)
# against "within bottlenecked, cross dense" (320 w + 19200) requires 960 c = 320 w + 12800,
# i.e. c >= 14 for any w >= 1 -- the cross side would have to be barely constrained (14/40
# of full rank) to match even a severely bottlenecked within side. There is no matched pair
# in the regime where both sides are actually squeezed.
#
# The functions below instead bottleneck both sides (no dense arm at all) and match cells
# by total **drive rank** -- the same quantity :func:`annotate_rank_grid` already names
# ``drive_rank``, w + (n_regions - 1) * c, the bound on a region's pre-activation rank --
# which is proportional to total recurrent weight count. Two cells at the same drive rank
# spend it differently -- more on within vs. more on cross -- which is the comparison the
# chapter actually wants. Displayed to the reader as "total bottleneck dimension".


def matched_drive_rank_groups(ranks: Sequence[int], *, n_regions: int = 4) -> pd.DataFrame:
    """Every non-dense ``(r_w, r_c)`` grid cell, grouped by total drive rank.

    ``drive_rank`` is ``r_w + (n_regions - 1) * r_c`` (the same bound :func:`annotate_rank_grid`
    computes); cells sharing it spend the same total number of recurrent weights differently.
    ``cross_drive_fraction`` is the share spent on the cross-region side -- 0 is all within, 1
    is all cross -- distinct from :func:`flow_temporal_summary`'s ``cross_fraction``, which is
    a measured drive share rather than a parameter share. Drive ranks reached by only one cell
    are dropped: there is nothing to compare them against.
    """
    rows = [
        {
            "label": f"w{r_w}_c{r_c}",
            "rank_within": int(r_w),
            "rank_cross": int(r_c),
            "drive_rank": int(r_w) + (int(n_regions) - 1) * int(r_c),
        }
        for r_w in ranks for r_c in ranks
    ]
    table = pd.DataFrame(rows)
    table["cross_drive_fraction"] = (int(n_regions) - 1) * table["rank_cross"] / table["drive_rank"]
    counts = table.groupby("drive_rank")["label"].transform("nunique")
    return table[counts >= 2].sort_values(["drive_rank", "cross_drive_fraction"]).reset_index(drop=True)


def matched_drive_rank_extremes(groups: pd.DataFrame) -> pd.DataFrame:
    """Within each matched drive rank, the most within-heavy cell paired with the most cross-heavy.

    The largest within-vs-cross contrast the drive rank allows. A drive rank reached by more
    than two cells has interior points too (worth plotting, e.g. the three-way matches at
    drive rank 13 and 16 on the current grid) but they are not part of the paired contrast,
    which needs exactly two arms per drive rank.
    """
    rows = []
    for drive_rank, block in groups.groupby("drive_rank"):
        low = block.loc[block["cross_drive_fraction"].idxmin()]
        high = block.loc[block["cross_drive_fraction"].idxmax()]
        if low["label"] == high["label"]:
            continue
        rows.append({
            "drive_rank": int(drive_rank),
            "within_heavy_label": str(low["label"]),
            "within_heavy_rank_within": int(low["rank_within"]),
            "within_heavy_rank_cross": int(low["rank_cross"]),
            "cross_heavy_label": str(high["label"]),
            "cross_heavy_rank_within": int(high["rank_within"]),
            "cross_heavy_rank_cross": int(high["rank_cross"]),
        })
    return pd.DataFrame(rows).sort_values("drive_rank").reset_index(drop=True)


def matched_drive_rank_fit_long(fit: pd.DataFrame, extremes: pd.DataFrame) -> pd.DataFrame:
    """Per-seed, per-region fit at every matched drive rank, tagged by arm -- the violins' data.

    ``fit`` is :func:`fixation_mrnn_sweep.score_variant_fit`'s output. Region is kept rather
    than averaged out, so each (drive rank, condition, arm) violin is drawn from seeds x
    regions points rather than five; every paired significance test downstream still pairs by
    seed only, averaging over region first, so the richer violin is a display choice and does
    not add degrees of freedom to a test. One condition row is added beyond the three fixation
    types: ``all_conditions`` (mean over the three per seed and region), the
    averaged-over-fixation-type view of overall network performance this chapter reports
    beside the per-condition breakdown. (An earlier version also added ``worst_cell``, the
    minimax objective's own region x condition minimum; dropped here since the question this
    task asks -- within vs. cross, and which fixation type it costs -- does not need it.)
    """
    labels = sorted(set(extremes["within_heavy_label"]) | set(extremes["cross_heavy_label"]))
    block = fit[fit["label"].isin(labels)][["label", "seed", "region", "condition", "r2_vs_ceiling"]].copy()

    all_conditions = (
        block.groupby(["label", "seed", "region"])["r2_vs_ceiling"].mean().reset_index()
        .assign(condition="all_conditions")
    )
    long = pd.concat([block, all_conditions], ignore_index=True)

    frames = []
    for _, arm in extremes.iterrows():
        for label, tag in ((arm["within_heavy_label"], "within-heavy"), (arm["cross_heavy_label"], "cross-heavy")):
            tagged = long[long["label"] == label].copy()
            tagged["drive_rank"] = int(arm["drive_rank"])
            tagged["arm"] = tag
            frames.append(tagged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def matched_drive_rank_fit_contrast(long: pd.DataFrame) -> pd.DataFrame:
    """Paired within-heavy vs. cross-heavy fit at each matched drive rank, one test per rank.

    ``long`` is :func:`matched_drive_rank_fit_long`'s output, averaged over region first (one
    value per label x seed x condition), because every grid cell shares the same seed list
    (:func:`fixation_mrnn_protocol.protocol_seeds`) and the comparison is paired by seed, not
    region. ``cross_advantage`` is positive when cross-heavy fits better. This is the
    nine-separate-tests view -- useful as a robustness check on whether the pooled result
    (:func:`matched_drive_rank_fit_pooled`) depends on total capacity, not as the primary
    result: nine Holm-corrected tests of five seeds each have much less power than one test
    of forty-five.
    """
    from dal_monte_2022_analysis.core.stats.correction import adjust_pvalues
    from dal_monte_2022_analysis.core.stats.proportions import significance_stars
    from dal_monte_2022_analysis.core.stats.tests import paired_ttest

    per_seed = long.groupby(["drive_rank", "arm", "condition", "seed"])["r2_vs_ceiling"].mean().reset_index()
    rows = []
    for (drive_rank, condition), block in per_seed.groupby(["drive_rank", "condition"]):
        wide = block.pivot(index="seed", columns="arm", values="r2_vs_ceiling").dropna()
        if len(wide) < 2 or "within-heavy" not in wide or "cross-heavy" not in wide:
            continue
        t_stat, p_value = paired_ttest(wide["cross-heavy"].to_numpy(), wide["within-heavy"].to_numpy())
        rows.append({
            "drive_rank": int(drive_rank),
            "condition": str(condition),
            "n_seeds": int(len(wide)),
            "within_heavy_mean": float(wide["within-heavy"].mean()),
            "cross_heavy_mean": float(wide["cross-heavy"].mean()),
            "cross_advantage": float((wide["cross-heavy"] - wide["within-heavy"]).mean()),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = np.nan
    for _, idx in table.groupby("condition").groups.items():
        table.loc[idx, "p_holm"] = adjust_pvalues(table.loc[idx, "p_value"].to_numpy(), "holm")
    table["stars"] = table["p_holm"].map(significance_stars)
    table["significant"] = table["stars"] != "n.s."
    return table


def matched_drive_rank_band_long(band_by_condition: pd.DataFrame, extremes: pd.DataFrame) -> pd.DataFrame:
    """Per-seed, per-region band recovery at every matched drive rank, tagged by arm.

    ``band_by_condition`` is :func:`band_resolved_recovery_by_condition`'s output concatenated
    across cells with a ``label`` column attached, as the ladder notebook does for
    :func:`band_resolved_recovery`. No ``all_conditions`` row: recovery is read per fixation
    type specifically, following :func:`band_resolved_recovery_by_condition`'s own reasoning.
    """
    labels = sorted(set(extremes["within_heavy_label"]) | set(extremes["cross_heavy_label"]))
    block = band_by_condition[band_by_condition["label"].isin(labels)][
        ["label", "seed", "region", "condition", "band", "residual_fraction"]
    ].copy()
    frames = []
    for _, arm in extremes.iterrows():
        for label, tag in ((arm["within_heavy_label"], "within-heavy"), (arm["cross_heavy_label"], "cross-heavy")):
            tagged = block[block["label"] == label].copy()
            tagged["drive_rank"] = int(arm["drive_rank"])
            tagged["arm"] = tag
            frames.append(tagged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ += [
    "matched_drive_rank_band_long",
    "matched_drive_rank_extremes",
    "matched_drive_rank_fit_contrast",
    "matched_drive_rank_fit_long",
    "matched_drive_rank_groups",
]


# --------------------------------------------------------------------------------------
# Pooled across total bottleneck dimension -- aggregate power, region and band breakdowns
# --------------------------------------------------------------------------------------
#
# matched_drive_rank_fit_contrast's nine separate paired tests, one per total bottleneck
# dimension and Holm-corrected against each other, is the right question when asking whether
# any single matched pair clears significance on its own -- but splitting the grid this way
# before testing throws away power for no benefit to the actual question, which is whether
# within and cross differ at all: the same seeds recur at every dimension, so (total
# bottleneck dimension, seed) is itself a valid paired replicate, within-heavy and
# cross-heavy at one dimension being the same seed, just a different split of the same
# total. Every function below pairs on (drive_rank, seed) explicitly (a ``pivot`` indexed by
# both, never by seed alone) precisely so a within-heavy value is never compared to a
# cross-heavy value from a *different* drive rank -- pooling must not break the parameter
# match it exists to test. ``cross_advantage`` is signed so that positive always means
# cross-heavy performed better, whichever way the underlying metric points (fit: higher is
# better, so cross-heavy minus within-heavy; band residual: lower is better, so within-heavy
# minus cross-heavy) -- the sign flip a reader has to track between Sections has otherwise
# cost more confusion than it is worth.


def matched_drive_rank_fit_pooled(long: pd.DataFrame) -> pd.DataFrame:
    """Paired within-heavy vs. cross-heavy fit, pooling over every total bottleneck dimension.

    ``long`` is :func:`matched_drive_rank_fit_long`'s output. Region is averaged out first;
    the replicate is (drive_rank, seed), so n is (number of matched dimensions) x (number of
    seeds) -- 9 x 5 = 45 on the current grid, for ``all_conditions`` and each fixation type.
    Reports a paired Cohen's :math:`d_z` (mean difference over its own standard deviation)
    alongside the t-test, because a tiny but near-universally-signed effect and a tiny,
    inconsistent one both have small p-values, and the sign-test columns are the audit for
    the second case: a tight p-value from the t-test with a sign test near 0.5 would mean the
    effect is driven by a few large seeds rather than most of them agreeing.
    """
    from scipy import stats as scipy_stats

    from dal_monte_2022_analysis.core.stats.correction import adjust_pvalues
    from dal_monte_2022_analysis.core.stats.proportions import significance_stars
    from dal_monte_2022_analysis.core.stats.tests import paired_ttest

    per_seed = long.groupby(["drive_rank", "arm", "condition", "seed"])["r2_vs_ceiling"].mean().reset_index()
    rows = []
    for condition, block in per_seed.groupby("condition"):
        wide = block.pivot_table(index=["drive_rank", "seed"], columns="arm", values="r2_vs_ceiling").dropna()
        if len(wide) < 2 or "within-heavy" not in wide or "cross-heavy" not in wide:
            continue
        diff = (wide["cross-heavy"] - wide["within-heavy"]).to_numpy()
        t_stat, p_value = paired_ttest(wide["cross-heavy"].to_numpy(), wide["within-heavy"].to_numpy())
        n_nonzero = int((diff != 0).sum())
        n_positive = int((diff > 0).sum())
        sign_p = float(scipy_stats.binomtest(n_positive, n_nonzero, 0.5).pvalue) if n_nonzero else float("nan")
        rows.append({
            "condition": str(condition),
            "n_pairs": int(len(wide)),
            "within_heavy_mean": float(wide["within-heavy"].mean()),
            "cross_heavy_mean": float(wide["cross-heavy"].mean()),
            "cross_advantage": float(diff.mean()),
            "se": float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else np.nan,
            "cohens_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "n_positive": n_positive,
            "n_pairs_nonzero": n_nonzero,
            "sign_test_p": sign_p,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = adjust_pvalues(table["p_value"].to_numpy(), "holm")
    table["stars"] = table["p_holm"].map(significance_stars)
    table["significant"] = table["stars"] != "n.s."
    return table


def matched_drive_rank_fit_by_region(long: pd.DataFrame) -> pd.DataFrame:
    """Paired within-heavy vs. cross-heavy fit per region, pooling over every total bottleneck dimension.

    The replicate is again (drive_rank, seed); ``p_holm`` corrects within each condition
    panel across its four regions, matching the one-correction-per-panel convention used
    throughout this chapter.
    """
    from dal_monte_2022_analysis.core.stats.correction import adjust_pvalues
    from dal_monte_2022_analysis.core.stats.proportions import significance_stars
    from dal_monte_2022_analysis.core.stats.tests import paired_ttest

    rows = []
    for (region, condition), sub in long.groupby(["region", "condition"]):
        wide = sub.pivot_table(index=["drive_rank", "seed"], columns="arm", values="r2_vs_ceiling",
                               aggfunc="mean").dropna()
        if len(wide) < 2 or "within-heavy" not in wide or "cross-heavy" not in wide:
            continue
        diff = (wide["cross-heavy"] - wide["within-heavy"]).to_numpy()
        t_stat, p_value = paired_ttest(wide["cross-heavy"].to_numpy(), wide["within-heavy"].to_numpy())
        rows.append({
            "region": str(region),
            "condition": str(condition),
            "n_pairs": int(len(wide)),
            "within_heavy_mean": float(wide["within-heavy"].mean()),
            "cross_heavy_mean": float(wide["cross-heavy"].mean()),
            "cross_advantage": float(diff.mean()),
            "se": float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else np.nan,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = np.nan
    for _, idx in table.groupby("condition").groups.items():
        table.loc[idx, "p_holm"] = adjust_pvalues(table.loc[idx, "p_value"].to_numpy(), "holm")
    table["stars"] = table["p_holm"].map(significance_stars)
    table["significant"] = table["stars"] != "n.s."
    return table


def matched_drive_rank_band_pooled(long: pd.DataFrame) -> pd.DataFrame:
    """Paired within-heavy vs. cross-heavy band recovery per band and condition, pooling over every total bottleneck dimension.

    ``long`` is :func:`matched_drive_rank_band_long`'s output. ``cross_advantage`` is
    ``within_heavy_mean - cross_heavy_mean`` here, not the other way around as in the fit
    tables: residual fraction is a *lower-is-better* metric, so cross-heavy doing better
    means its residual is the smaller one. Keeping the sign of ``cross_advantage`` meaning
    "cross-heavy ahead by this much" in every table in this module, regardless of which way
    the underlying metric points, is what the fit tables already do; this is the one table
    where getting that right requires swapping the subtraction order. ``p_holm`` corrects
    within each condition panel across its four bands.
    """
    from dal_monte_2022_analysis.core.stats.correction import adjust_pvalues
    from dal_monte_2022_analysis.core.stats.proportions import significance_stars
    from dal_monte_2022_analysis.core.stats.tests import paired_ttest

    per_seed = (
        long.groupby(["drive_rank", "arm", "condition", "band", "seed"])["residual_fraction"]
        .mean().reset_index()
    )
    rows = []
    for (condition, band), block in per_seed.groupby(["condition", "band"]):
        wide = block.pivot_table(index=["drive_rank", "seed"], columns="arm", values="residual_fraction").dropna()
        if len(wide) < 2 or "within-heavy" not in wide or "cross-heavy" not in wide:
            continue
        diff = (wide["within-heavy"] - wide["cross-heavy"]).to_numpy()
        t_stat, p_value = paired_ttest(wide["within-heavy"].to_numpy(), wide["cross-heavy"].to_numpy())
        rows.append({
            "condition": str(condition),
            "band": str(band),
            "n_pairs": int(len(wide)),
            "within_heavy_mean": float(wide["within-heavy"].mean()),
            "cross_heavy_mean": float(wide["cross-heavy"].mean()),
            "cross_advantage": float(diff.mean()),
            "se": float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else np.nan,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = np.nan
    for _, idx in table.groupby("condition").groups.items():
        table.loc[idx, "p_holm"] = adjust_pvalues(table.loc[idx, "p_value"].to_numpy(), "holm")
    table["stars"] = table["p_holm"].map(significance_stars)
    table["significant"] = table["stars"] != "n.s."
    return table


# --------------------------------------------------------------------------------------
# Does the within-vs-cross effect itself differ between fixation types?
# --------------------------------------------------------------------------------------
#
# The functions above test, within each fixation type separately, whether cross-heavy beats
# within-heavy. None of them say whether *that effect's size* differs between fixation types
# -- a region-pooled version of "is one type of bottlenecking costlier to one fixation type
# than another". The replicate is still (drive_rank, seed): the same replicate produced the
# gap for every fixation type, so the comparison between two fixation types' gaps is a
# paired test too, not Welch's.


def matched_drive_rank_condition_contrast(
    long: pd.DataFrame, *, conditions: Sequence[str] = ("face_interactive", "face_non_interactive", "object"),
) -> pd.DataFrame:
    """Every pairwise comparison of the cross-vs-within gap between fixation types, region pooled.

    For each (drive_rank, seed), the gap is cross-heavy minus within-heavy fit, averaged over
    region. Two fixation types' gaps at the same (drive_rank, seed) came from the same
    network fits, so the comparison between them is paired; ``p_holm`` corrects within the
    one panel this makes, across the ``len(conditions) choose 2`` pairs.
    """
    from itertools import combinations

    from dal_monte_2022_analysis.core.stats.correction import adjust_pvalues
    from dal_monte_2022_analysis.core.stats.proportions import significance_stars
    from dal_monte_2022_analysis.core.stats.tests import paired_ttest

    per_seed = long.groupby(["drive_rank", "arm", "condition", "seed"])["r2_vs_ceiling"].mean().reset_index()
    gaps: dict[str, pd.Series] = {}
    for condition in conditions:
        wide = (per_seed[per_seed["condition"] == condition]
               .pivot_table(index=["drive_rank", "seed"], columns="arm", values="r2_vs_ceiling").dropna())
        if "within-heavy" in wide and "cross-heavy" in wide:
            gaps[condition] = wide["cross-heavy"] - wide["within-heavy"]

    rows = []
    for a, b in combinations([c for c in conditions if c in gaps], 2):
        gap_a, gap_b = gaps[a].align(gaps[b], join="inner")
        if len(gap_a) < 2:
            continue
        t_stat, p_value = paired_ttest(gap_a.to_numpy(), gap_b.to_numpy())
        rows.append({
            "condition_a": a,
            "condition_b": b,
            "n_pairs": int(len(gap_a)),
            "gap_a_mean": float(gap_a.mean()),
            "gap_b_mean": float(gap_b.mean()),
            "gap_difference": float((gap_a - gap_b).mean()),
            "se": float((gap_a - gap_b).std(ddof=1) / np.sqrt(len(gap_a))) if len(gap_a) > 1 else np.nan,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = adjust_pvalues(table["p_value"].to_numpy(), "holm")
    table["stars"] = table["p_holm"].map(significance_stars)
    table["significant"] = table["stars"] != "n.s."
    return table


def matched_drive_rank_condition_contrast_by_region(
    long: pd.DataFrame, *, conditions: Sequence[str] = ("face_interactive", "face_non_interactive", "object"),
) -> pd.DataFrame:
    """:func:`matched_drive_rank_condition_contrast`, run separately within each region.

    ``p_holm`` corrects within each region's own panel across its pairwise comparisons, not
    across regions -- the four regions are different panels, each asking its own version of
    "does one fixation type's gap differ from another's here".
    """
    from itertools import combinations

    from dal_monte_2022_analysis.core.stats.correction import adjust_pvalues
    from dal_monte_2022_analysis.core.stats.proportions import significance_stars
    from dal_monte_2022_analysis.core.stats.tests import paired_ttest

    tables = []
    for region, region_block in long.groupby("region"):
        per_seed = (region_block.groupby(["drive_rank", "arm", "condition", "seed"])["r2_vs_ceiling"]
                   .mean().reset_index())
        gaps: dict[str, pd.Series] = {}
        for condition in conditions:
            wide = (per_seed[per_seed["condition"] == condition]
                   .pivot_table(index=["drive_rank", "seed"], columns="arm", values="r2_vs_ceiling").dropna())
            if "within-heavy" in wide and "cross-heavy" in wide:
                gaps[condition] = wide["cross-heavy"] - wide["within-heavy"]

        rows = []
        for a, b in combinations([c for c in conditions if c in gaps], 2):
            gap_a, gap_b = gaps[a].align(gaps[b], join="inner")
            if len(gap_a) < 2:
                continue
            t_stat, p_value = paired_ttest(gap_a.to_numpy(), gap_b.to_numpy())
            rows.append({
                "region": str(region),
                "condition_a": a,
                "condition_b": b,
                "n_pairs": int(len(gap_a)),
                "gap_a_mean": float(gap_a.mean()),
                "gap_b_mean": float(gap_b.mean()),
                "gap_difference": float((gap_a - gap_b).mean()),
                "se": float((gap_a - gap_b).std(ddof=1) / np.sqrt(len(gap_a))) if len(gap_a) > 1 else np.nan,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            })
        if rows:
            sub = pd.DataFrame(rows)
            sub["p_holm"] = adjust_pvalues(sub["p_value"].to_numpy(), "holm")
            tables.append(sub)
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if not table.empty:
        table["stars"] = table["p_holm"].map(significance_stars)
        table["significant"] = table["stars"] != "n.s."
    return table


def matched_drive_rank_band_condition_contrast(
    long: pd.DataFrame, *, conditions: Sequence[str] = ("face_interactive", "face_non_interactive", "object"),
) -> pd.DataFrame:
    """:func:`matched_drive_rank_condition_contrast`'s question, asked per frequency band.

    ``long`` is :func:`matched_drive_rank_band_long`'s output. For each (drive_rank, seed,
    band), the gap is within-heavy minus cross-heavy residual fraction, averaged over
    region -- ``within_heavy minus cross_heavy`` rather than the other way around, matching
    :func:`matched_drive_rank_band_pooled`'s sign convention so a positive gap still means
    cross-heavy did better. Two fixation types' gaps at the same (drive_rank, seed, band)
    came from the same network fits, so the comparison between them is paired, not Welch's.
    ``p_holm`` corrects within each band's own panel, across its pairwise comparisons, not
    across bands -- each band is asking its own version of "does one fixation type's
    cross-heavy advantage differ from another's here".
    """
    from itertools import combinations

    from dal_monte_2022_analysis.core.stats.correction import adjust_pvalues
    from dal_monte_2022_analysis.core.stats.proportions import significance_stars
    from dal_monte_2022_analysis.core.stats.tests import paired_ttest

    tables = []
    for band, band_block in long.groupby("band"):
        per_seed = (band_block.groupby(["drive_rank", "arm", "condition", "seed"])["residual_fraction"]
                   .mean().reset_index())
        gaps: dict[str, pd.Series] = {}
        for condition in conditions:
            wide = (per_seed[per_seed["condition"] == condition]
                   .pivot_table(index=["drive_rank", "seed"], columns="arm", values="residual_fraction").dropna())
            if "within-heavy" in wide and "cross-heavy" in wide:
                gaps[condition] = wide["within-heavy"] - wide["cross-heavy"]

        rows = []
        for a, b in combinations([c for c in conditions if c in gaps], 2):
            gap_a, gap_b = gaps[a].align(gaps[b], join="inner")
            if len(gap_a) < 2:
                continue
            t_stat, p_value = paired_ttest(gap_a.to_numpy(), gap_b.to_numpy())
            rows.append({
                "band": str(band),
                "condition_a": a,
                "condition_b": b,
                "n_pairs": int(len(gap_a)),
                "gap_a_mean": float(gap_a.mean()),
                "gap_b_mean": float(gap_b.mean()),
                "gap_difference": float((gap_a - gap_b).mean()),
                "se": float((gap_a - gap_b).std(ddof=1) / np.sqrt(len(gap_a))) if len(gap_a) > 1 else np.nan,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            })
        if rows:
            sub = pd.DataFrame(rows)
            sub["p_holm"] = adjust_pvalues(sub["p_value"].to_numpy(), "holm")
            tables.append(sub)
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if not table.empty:
        table["stars"] = table["p_holm"].map(significance_stars)
        table["significant"] = table["stars"] != "n.s."
    return table


__all__ += [
    "matched_drive_rank_band_condition_contrast",
    "matched_drive_rank_band_pooled",
    "matched_drive_rank_condition_contrast",
    "matched_drive_rank_condition_contrast_by_region",
    "matched_drive_rank_fit_by_region",
    "matched_drive_rank_fit_pooled",
]
