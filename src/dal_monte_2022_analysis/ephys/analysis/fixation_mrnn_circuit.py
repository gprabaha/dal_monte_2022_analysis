"""Circuit-level readouts of a fitted mRNN, and what can honestly be compared across fits.

Two questions run through this module, and they need different machinery.

**Is the solution unique?** A fitted network is only evidence about the brain if an
independent fit of the same model to the same data recovers the same circuit. Comparing
two fits directly is meaningless, because the model has an exact symmetry: for a tanh
network with a linear readout, relabelling hidden units by a permutation :math:`P` and
flipping their signs with a diagonal :math:`S \\in \\{\\pm1\\}` leaves every output
identical, since :math:`\\tanh` is odd:

.. math::
    W_{rec} \\mapsto PS\\,W_{rec}\\,(PS)^{\\top}, \\quad
    W_{in} \\mapsto PS\\,W_{in}, \\quad W_{out} \\mapsto W_{out}(PS)^{\\top}.

This is the *only* exact symmetry -- the nonlinearity blocks general rotations -- so the
signed permutation group is exactly the licence a comparison has. Anything invariant to it
is fair game; anything invariant to *more* than it (CKA, RSA) is permissive and will
overstate agreement, which is why those are reported alongside stricter measures rather
than instead of them.

**What does the circuit do?** The current from one region to another lives in the target
region's unit space, which is seed-arbitrary, so its direction cannot be compared across
fits as it stands. Two routes out, both used here: angles computed *within* one model are
invariant automatically, because the same signed permutation applies to both arguments and
cancels; and pushing the current through the readout puts it in the observed PC space,
which every fit shares by construction. Splitting the current into the part the readout
can see and the part it cannot is then a measurement rather than an assumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
    load_fixation_mrnn_checkpoint,
    replay_fixation_mrnn_run,
)

# --------------------------------------------------------------------------------------
# Pulling the pieces out of a replay
# --------------------------------------------------------------------------------------


def region_slices(model) -> dict[str, slice]:
    return model.hidden_region_slices()


def recurrent_blocks(model) -> dict[tuple[str, str], np.ndarray]:
    """``block[(source, target)]`` of shape ``(n_target, n_source)``.

    The assembled matrix is written ``weight[target, source]`` and the forward pass
    computes :math:`Wh`, so this orientation is the one that multiplies a source state to
    give a contribution to the target -- not the transpose.
    """
    weight = model.recurrent_weight_matrix().detach().cpu().numpy()
    slices = region_slices(model)
    return {
        (source, target): weight[slices[target], slices[source]]
        for source in model.region_order
        for target in model.region_order
    }


def region_states(replay: Mapping[str, object]) -> dict[str, np.ndarray]:
    """``h[region]`` of shape ``(condition, time, n_units)``."""
    model = replay["model"]
    h_seq = replay["h_seq"].detach().cpu().numpy()
    slices = region_slices(model)
    return {region: h_seq[..., slices[region]] for region in model.region_order}


def readout_matrices(model) -> dict[str, np.ndarray]:
    """``W_out[region]`` of shape ``(n_pc, n_units)``."""
    return {
        region: head.weight.detach().cpu().numpy()
        for region, head in model.output_heads.items()
    }


# --------------------------------------------------------------------------------------
# Inter-regional currents
# --------------------------------------------------------------------------------------


def inter_regional_currents(replay: Mapping[str, object]) -> dict[tuple[str, str], np.ndarray]:
    """:math:`c_{s\\to t}(\\tau) = W[t,s]\\,h_s(\\tau)`, shape ``(condition, time, n_target)``.

    This is the quantity the earlier analyses summarised as a scalar magnitude. The
    magnitude is invariant to relabelling but throws the direction away, and the direction
    is where the interesting questions live -- whether two fixation types use the same
    channel, whether the current pushes along or against the target's own dynamics.
    """
    model = replay["model"]
    blocks = recurrent_blocks(model)
    states = region_states(replay)
    return {
        (source, target): states[source] @ blocks[(source, target)].T
        for source in model.region_order
        for target in model.region_order
        if source != target
    }


def readout_projected_currents(
    replay: Mapping[str, object],
) -> dict[tuple[str, str], np.ndarray]:
    """Currents expressed in the observed PC space, shape ``(condition, time, n_pc)``.

    The readout is the one map out of the model whose coordinates every fit shares, so
    this is the form in which a current's *direction* can be compared across seeds. It
    keeps only what the readout can see; :func:`current_potency` measures how much that
    leaves behind.
    """
    model = replay["model"]
    readouts = readout_matrices(model)
    return {
        pair: current @ readouts[pair[1]].T
        for pair, current in inter_regional_currents(replay).items()
    }


def _participation_ratio(values: np.ndarray) -> float:
    """Effective number of dimensions carrying a set of singular values."""
    energy = np.asarray(values, dtype=float) ** 2
    total = float(energy.sum())
    if total <= 0:
        return float("nan")
    return float(total**2 / float((energy**2).sum()))


def current_potency(replay: Mapping[str, object]) -> pd.DataFrame:
    """Per pathway and condition: how large the current is, how much of it the readout sees.

    The **potent fraction** is the share of the current's energy lying in the row space of
    that region's readout. Its complement is inter-regional signalling that never reaches
    the observed PCs -- a communication subspace, measured directly rather than inferred
    from correlations between recorded populations.

    **Dimensionality** is the participation ratio of the current's singular values over
    condition x time. It says how many directions the pathway actually uses, which is the
    empirical counterpart of the rank the bottleneck imposes: if a dense model already
    uses three dimensions, a rank-3 constraint is a description rather than a restriction.
    """
    model = replay["model"]
    readouts = readout_matrices(model)
    conditions = list(replay["condition_order"])
    rows: list[dict[str, object]] = []
    for (source, target), current in inter_regional_currents(replay).items():
        # An orthonormal basis for the readout's row space; the projector onto it is the
        # part of the target's state space the observation can distinguish.
        basis = np.linalg.svd(readouts[target], full_matrices=False)[2]
        for index, condition in enumerate(conditions):
            block = current[index]
            total = float((block**2).sum())
            potent = float(((block @ basis.T) ** 2).sum())
            singular = np.linalg.svd(block - block.mean(axis=0, keepdims=True), compute_uv=False)
            rows.append({
                "source": source,
                "target": target,
                "pathway": f"{source}→{target}",
                "condition": condition,
                "magnitude": float(np.linalg.norm(block, axis=-1).mean()),
                "potent_fraction": potent / total if total > 0 else float("nan"),
                "dimensionality": _participation_ratio(singular),
                # Reported so a saturated potent fraction is legible rather than
                # mysterious: when the readout has at least as many independent rows as
                # the region has units, its row space *is* the whole state space and
                # nothing can be output-null. The split only becomes a measurement when
                # the region is wider than the number of components read out of it.
                "readout_rank": int(np.linalg.matrix_rank(readouts[target])),
                "target_units": int(readouts[target].shape[1]),
            })
    return pd.DataFrame(rows)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(a.ravel(), b.ravel()) / denominator)


def current_condition_alignment(replay: Mapping[str, object]) -> pd.DataFrame:
    """Angle between the same pathway's current under two conditions.

    Computed *within* one model, so it is invariant to relabelling without any alignment
    step: the signed permutation applies to both arguments and cancels. A cosine near 1
    means the two fixation types drive the target region through the same channel and
    differ only in how hard; a low cosine means they use genuinely different routes.
    """
    conditions = list(replay["condition_order"])
    rows: list[dict[str, object]] = []
    for (source, target), current in inter_regional_currents(replay).items():
        for i, condition_a in enumerate(conditions):
            for j, condition_b in enumerate(conditions):
                if j <= i:
                    continue
                rows.append({
                    "pathway": f"{source}→{target}",
                    "source": source,
                    "target": target,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    # Time-averaged over matched bins, so this is the angle between the
                    # two pathways-in-use rather than between two arbitrary snapshots.
                    "cosine": float(np.mean([
                        _cosine(current[i, t], current[j, t]) for t in range(current.shape[1])
                    ])),
                })
    return pd.DataFrame(rows)


def current_drive_alignment(replay: Mapping[str, object]) -> pd.DataFrame:
    """Angle between incoming inter-regional current and the target's own recurrent drive.

    Positive means the input from ``source`` pushes the target along the direction its
    internal dynamics were already going; negative means it opposes them. Also invariant
    within a model, for the same reason as :func:`current_condition_alignment`.
    """
    model = replay["model"]
    blocks = recurrent_blocks(model)
    states = region_states(replay)
    conditions = list(replay["condition_order"])
    rows: list[dict[str, object]] = []
    for target in model.region_order:
        intrinsic = states[target] @ blocks[(target, target)].T
        for source in model.region_order:
            if source == target:
                continue
            current = states[source] @ blocks[(source, target)].T
            for index, condition in enumerate(conditions):
                rows.append({
                    "pathway": f"{source}→{target}",
                    "source": source,
                    "target": target,
                    "condition": condition,
                    "cosine_with_intrinsic_drive": float(np.mean([
                        _cosine(current[index, t], intrinsic[index, t])
                        for t in range(current.shape[1])
                    ])),
                })
    return pd.DataFrame(rows)


def source_share(replay: Mapping[str, object]) -> pd.DataFrame:
    """Share of the total recurrent input to each region contributed by each source, in time.

    Magnitudes are not comparable between models, but shares are: they are ratios of
    quantities in the same space, so the normalisation cancels.
    """
    model = replay["model"]
    blocks = recurrent_blocks(model)
    states = region_states(replay)
    conditions = list(replay["condition_order"])
    timeline = np.asarray(replay["checkpoint"].get("timeline_s", []), dtype=float)
    rows: list[dict[str, object]] = []
    for target in model.region_order:
        contributions = {
            source: np.linalg.norm(states[source] @ blocks[(source, target)].T, axis=-1)
            for source in model.region_order
        }
        total = sum(contributions.values())
        for source, magnitude in contributions.items():
            for index, condition in enumerate(conditions):
                for step in range(magnitude.shape[1]):
                    rows.append({
                        "target": target,
                        "source": source,
                        "condition": condition,
                        "time_s": float(timeline[step]) if step < len(timeline) else float(step),
                        "share": float(magnitude[index, step] / total[index, step])
                        if total[index, step] > 0 else float("nan"),
                    })
    return pd.DataFrame(rows)


def pathway_ablation_effect(
    run_dir: str | Path,
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Damage to each region's reconstruction when one pathway is silenced, per condition.

    This is what having a generative model buys that the data cannot: the pathway is cut
    *after* fitting, so nothing re-adapts, and the question is what that route was
    carrying. Scored per condition, because a pathway that matters only for one fixation
    type is exactly the result this chapter is looking for and a pooled score would hide
    it.

    Note the difference from task 03. There, each variant was **refitted** without the
    connection, which asks whether the data can be reproduced at all without it. Here the
    fitted model is left alone, which asks what the connection was doing in the solution
    the model found. The two answers can differ, and the pair is more informative than
    either alone.
    """
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
        reconstruction_accuracy,
        replay_fixation_mrnn_run_with_ablations,
    )

    intact = reconstruction_accuracy(replay_fixation_mrnn_run(Path(run_dir), device=device))
    baseline = intact.set_index(["region", "condition"])["r2"]
    model, _ = load_fixation_mrnn_checkpoint(Path(run_dir), device=device)

    rows: list[dict[str, object]] = []
    for source in model.region_order:
        for target in model.region_order:
            if source == target:
                continue
            lesioned = reconstruction_accuracy(
                replay_fixation_mrnn_run_with_ablations(
                    Path(run_dir), ablations=[(source, target)], device=device
                )
            )
            for _, row in lesioned.iterrows():
                key = (row["region"], row["condition"])
                rows.append({
                    "pathway": f"{source}→{target}",
                    "source": source,
                    "target": target,
                    "region": row["region"],
                    "condition": row["condition"],
                    "r2_intact": float(baseline.get(key, np.nan)),
                    "r2_lesioned": float(row["r2"]),
                    "damage": float(baseline.get(key, np.nan)) - float(row["r2"]),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Invariants: quantities two independent fits can legitimately be compared on
# --------------------------------------------------------------------------------------


def recurrent_eigenspectrum(model) -> np.ndarray:
    """Eigenvalues of the assembled recurrent matrix, ordered by decreasing modulus.

    Invariant under *any* similarity transform and so, in particular, under the signed
    permutation the model is free to choose. It is the cheapest honest comparison
    available: it needs no alignment step, no fitting, and no choice of hyperparameter.
    """
    weight = model.recurrent_weight_matrix().detach().cpu().numpy()
    eigenvalues = np.linalg.eigvals(weight)
    return eigenvalues[np.argsort(-np.abs(eigenvalues))]


def eigenspectrum_agreement(a: np.ndarray, b: np.ndarray, *, n_modes: int = 20) -> float:
    """Agreement of two spectra on a 0-1 scale, from their leading moduli.

    The moduli are compared rather than the complex values because eigenvalues of a real
    matrix come in conjugate pairs whose ordering within a pair is arbitrary. Scaled by
    the mean modulus so the number does not depend on the overall gain.
    """
    left = np.sort(np.abs(a))[::-1][:n_modes]
    right = np.sort(np.abs(b))[::-1][:n_modes]
    size = min(len(left), len(right))
    if size == 0:
        return float("nan")
    left, right = left[:size], right[:size]
    scale = float(np.mean(np.concatenate([left, right])))
    if scale <= 0:
        return float("nan")
    return float(1.0 - np.mean(np.abs(left - right)) / scale)


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """Linear CKA between two state matrices, each ``(samples, units)``.

    Invariant to rotation and isotropic scaling, which is *more* than the model's own
    symmetry group allows. That makes it a permissive measure: it can only overstate how
    much two fits agree, so a low CKA is decisive while a high one is not.
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(left.T @ right, "fro") ** 2)
    scale = float(np.linalg.norm(left.T @ left, "fro") * np.linalg.norm(right.T @ right, "fro"))
    return float(cross / scale) if scale > 0 else float("nan")


def procrustes_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Best orthogonal alignment of two state matrices, as variance explained in 0-1.

    Stricter than CKA: one shared rotation has to work for the whole trajectory, rather
    than the alignment being implicit and per-sample.
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    left = left / max(float(np.linalg.norm(left)), 1e-12)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(singular.sum() ** 2)


def matched_unit_alignment(model_a, model_b) -> dict[str, object]:
    """Align two fits by the signed permutation the model is actually free to choose.

    Units are matched on their input *and* output weight profiles -- the only description
    of a unit that does not itself depend on the labelling -- by solving the assignment
    problem on the absolute correlation between profiles, with the sign taken from the
    correlation. The recurrent matrices are then compared entrywise after applying the
    match.

    This is the strictest comparison available. If the spectra agree but this does not,
    the two fits implement similar dynamics through genuinely different circuits, which is
    a different and weaker claim than circuit-level recovery.
    """
    from scipy.optimize import linear_sum_assignment

    def profile(model) -> np.ndarray:
        weight = model.recurrent_weight_matrix().detach().cpu().numpy()
        readout = np.zeros((weight.shape[0], 0))
        heads = readout_matrices(model)
        slices = region_slices(model)
        columns = []
        for region in model.region_order:
            block = np.zeros((weight.shape[0], heads[region].shape[0]))
            block[slices[region]] = heads[region].T
            columns.append(block)
        readout = np.concatenate(columns, axis=1)
        # A unit is described by what it sends, what it receives, and how it is read out.
        return np.concatenate([weight, weight.T, readout], axis=1)

    left = profile(model_a)
    right = profile(model_b)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    left = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-12, None)
    right = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-12, None)
    similarity = left @ right.T
    rows, columns = linear_sum_assignment(-np.abs(similarity))
    signs = np.sign(similarity[rows, columns])
    signs[signs == 0] = 1.0

    weight_a = model_a.recurrent_weight_matrix().detach().cpu().numpy()
    weight_b = model_b.recurrent_weight_matrix().detach().cpu().numpy()
    permuted = weight_b[np.ix_(columns, columns)] * np.outer(signs, signs)
    correlation = float(np.corrcoef(weight_a.ravel(), permuted.ravel())[0, 1])
    return {
        "permutation": columns,
        "signs": signs,
        "matched_profile_similarity": float(np.abs(similarity[rows, columns]).mean()),
        "matched_weight_correlation": correlation,
    }


# --------------------------------------------------------------------------------------
# Local dynamics: the flow, not the trajectory
# --------------------------------------------------------------------------------------


def _activation_name(model) -> str:
    return str(getattr(model.spec, "activation", "tanh")).strip().lower()


def slow_points(
    replay: Mapping[str, object],
    condition_index: int,
    *,
    n_starts: int = 32,
    iterations: int = 800,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """States where the update is close to stationary, for one condition.

    The input to this model is a constant one-hot per condition, so each condition defines
    its own **autonomous** system: the conditions differ only in where they start and in a
    fixed offset. Every difference between fixation types is therefore a difference in
    dynamics rather than in drive, and the fixed points of each condition's system are
    well defined. Minimising :math:`q(h)=\\tfrac12\\lVert\\phi(Wh+b)-h\\rVert^2` from points
    on the trajectory finds the ones the trajectory actually visits.
    """
    model = replay["model"]
    device = next(model.parameters()).device
    weight = model.recurrent_weight_matrix().detach()
    h_seq = replay["h_seq"].detach()[condition_index]
    inp = replay["inp"].detach()[condition_index]

    if model.mrnn.inp_constrained:
        w_inp = model.mrnn.apply_dales_law(
            model.mrnn.W_inp, model.mrnn.W_inp_mask, model.mrnn.W_inp_sign_matrix
        )
    else:
        w_inp = model.mrnn.W_inp * model.mrnn.W_inp_mask
    # Constant across time for these inputs, but averaged rather than assumed so the
    # function stays correct if a time-varying input is ever used.
    # Detached: only the state is being optimised here, and leaving the model's
    # parameters in the graph would make the second iteration backward through it twice.
    bias = ((w_inp @ inp.mean(dim=0)) + model.mrnn.tonic_inp).detach()

    generator = np.random.default_rng(int(seed))
    starts = generator.choice(h_seq.shape[0], size=min(n_starts, h_seq.shape[0]), replace=False)
    state = h_seq[starts].clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([state], lr=float(learning_rate))
    for _ in range(int(iterations)):
        optimizer.zero_grad()
        nxt = model.mrnn.activation(state @ weight.T + bias)
        loss = 0.5 * ((nxt - state) ** 2).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        nxt = model.mrnn.activation(state @ weight.T + bias)
        speed = torch.linalg.norm(nxt - state, dim=-1)
    return {
        "points": state.detach().cpu().numpy(),
        "speed": speed.cpu().numpy(),
        "start_times": np.asarray(starts),
    }


def jacobian_eigenvalues(replay: Mapping[str, object], state: np.ndarray) -> np.ndarray:
    """Eigenvalues of the linearized update at ``state``, ordered by decreasing modulus.

    For :math:`h_{t+1}=\\phi(Wh_t+b)` the Jacobian is
    :math:`J=\\mathrm{diag}(\\phi'(u))\\,W`. Its eigenvalues are the local time constants
    and rotation frequencies of the flow, and they are invariant under signed permutation
    because :math:`J` transforms by conjugation. Modulus above 1 is locally expanding;
    complex pairs mean local oscillation, so their argument converts directly to a
    frequency in hertz given the bin size.
    """
    model = replay["model"]
    weight = model.recurrent_weight_matrix().detach().cpu().numpy()
    point = torch.as_tensor(np.asarray(state, dtype=np.float32))
    if _activation_name(model) == "tanh":
        # phi' = 1 - tanh(u)^2, and at the point itself tanh(u) is the next state.
        derivative = 1.0 - np.tanh(point.numpy() @ weight.T) ** 2
    else:  # softplus and anything else: differentiate numerically through the module
        pre = torch.as_tensor(point.numpy() @ weight.T, dtype=torch.float32, requires_grad=True)
        activated = model.mrnn.activation(pre)
        derivative = torch.autograd.grad(activated.sum(), pre)[0].detach().numpy()
    jacobian = derivative[:, None] * weight
    eigenvalues = np.linalg.eigvals(jacobian)
    return eigenvalues[np.argsort(-np.abs(eigenvalues))]


def jacobian_frequencies(eigenvalues: np.ndarray, *, bin_size_s: float) -> np.ndarray:
    """Oscillation frequency in Hz of each complex mode of a discrete-time Jacobian."""
    return np.abs(np.angle(np.asarray(eigenvalues))) / (2.0 * np.pi * float(bin_size_s))


# --------------------------------------------------------------------------------------
# Ensembles and nulls
# --------------------------------------------------------------------------------------


def untrained_reference_spectra(
    run_dir: str | Path,
    *,
    n_draws: int = 8,
    device: str = "cpu",
    seed: int = 0,
) -> list[np.ndarray]:
    """Spectra of freshly initialized models with this run's architecture.

    The architectural null. A block-structured matrix with a set spectral radius has a
    constrained spectrum before it is trained at all, so some agreement between fitted
    models is forced by the architecture rather than earned from the data. This measures
    how much, and it costs nothing: no fitting is involved.
    """
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
        _model_spec_from_checkpoint,
        resolve_device,
    )
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_model import FixationMRNNModel

    resolved = resolve_device(device)
    _, checkpoint = load_fixation_mrnn_checkpoint(run_dir, device=device)
    # The spec is rebuilt from the checkpoint rather than taken from it: what is stored is
    # a plain dict of fields, and the constructor needs the dataclass.
    spec = _model_spec_from_checkpoint(checkpoint, device=resolved)
    spectra: list[np.ndarray] = []
    for draw in range(int(n_draws)):
        torch.manual_seed(int(seed) + draw)
        spectra.append(recurrent_eigenspectrum(FixationMRNNModel(spec).to(resolved)))
    return spectra


def matched_weight_null(
    run_dir: str | Path,
    *,
    n_draws: int = 4,
    device: str = "cpu",
    seed: int = 0,
) -> float:
    """Matched-weight correlation between *untrained* models of the same architecture.

    The alignment step searches over :math:`n!\,2^n` signed permutations for the best
    match, so it will find some apparent agreement between any two matrices. This measures
    how much, and it is the number a fitted ensemble has to beat before "the circuits
    agree" means anything.
    """
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
        _model_spec_from_checkpoint,
        resolve_device,
    )
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_model import FixationMRNNModel

    resolved = resolve_device(device)
    _, checkpoint = load_fixation_mrnn_checkpoint(run_dir, device=device)
    spec = _model_spec_from_checkpoint(checkpoint, device=resolved)
    models = []
    for draw in range(int(n_draws)):
        torch.manual_seed(int(seed) + 1000 + draw)
        models.append(FixationMRNNModel(spec).to(resolved))
    values = [
        matched_unit_alignment(models[i], models[j])["matched_weight_correlation"]
        for i in range(len(models))
        for j in range(i + 1, len(models))
    ]
    return float(np.mean(values)) if values else float("nan")


def architectural_null_invariants(
    run_dir: str | Path,
    *,
    n_draws: int = 5,
    device: str = "cpu",
    seed: int = 0,
    include_matched_weights: bool = True,
) -> pd.DataFrame:
    """The same agreement measures between *untrained* models of this architecture.

    This is the floor every fitted number has to clear. A block-structured matrix rescaled
    to a set spectral radius already has a constrained spectrum, and the alignment step in
    the matched-weight measure searches a factorially large group and will find something
    in any pair of matrices -- so some agreement exists before any data is seen. The
    untrained models are replayed on the fitted run's own inputs and initial states, so the
    only difference from the fitted ensemble is that the weights were never trained.

    Returns the same columns as :func:`ensemble_invariants`, so the two can be concatenated
    and plotted against each other.
    """
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
        _model_spec_from_checkpoint,
        resolve_device,
    )
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_model import FixationMRNNModel

    resolved = resolve_device(device)
    _, checkpoint = load_fixation_mrnn_checkpoint(run_dir, device=device)
    spec = _model_spec_from_checkpoint(checkpoint, device=resolved)
    inp = torch.as_tensor(checkpoint["input_tensor"], dtype=torch.float32, device=resolved)
    h0 = checkpoint["h0"].to(resolved)

    draws = []
    for draw in range(int(n_draws)):
        torch.manual_seed(int(seed) + 1000 + draw)
        model = FixationMRNNModel(spec).to(resolved).eval()
        with torch.no_grad():
            h_seq = model(inp, h0)["h_seq"].detach().cpu().numpy()
        draws.append({
            "model": model,
            "states": h_seq.reshape(-1, h_seq.shape[-1]),
            "spectrum": recurrent_eigenspectrum(model),
        })

    rows: list[dict[str, object]] = []
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            left, right = draws[i], draws[j]
            values = {
                "eigenspectrum": eigenspectrum_agreement(left["spectrum"], right["spectrum"]),
                "state Procrustes": procrustes_similarity(left["states"], right["states"]),
                "state CKA": linear_cka(left["states"], right["states"]),
            }
            if include_matched_weights:
                values["matched weight correlation"] = matched_unit_alignment(
                    left["model"], right["model"]
                )["matched_weight_correlation"]
            for measure, value in values.items():
                rows.append({"pair": f"null {i}-{j}", "measure": measure,
                             "agreement": float(value), "ensemble": "architectural null"})
    return pd.DataFrame(rows)


def agreement_against_null(fitted: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    """Fitted agreement next to its floor, per measure.

    ``margin`` is what training bought. A measure whose margin is near zero is reporting
    the architecture, not the data, however high its raw value looks -- which is the whole
    reason the null is computed rather than assumed to be zero.
    """
    left = fitted.groupby("measure")["agreement"].agg(fitted_mean="mean", fitted_sd="std")
    right = null.groupby("measure")["agreement"].agg(null_mean="mean", null_sd="std")
    table = left.join(right, how="outer")
    table["margin"] = table["fitted_mean"] - table["null_mean"]
    # Scaled by the null's own spread, so "above the floor" is judged against how much the
    # floor itself varies rather than against a number chosen by hand.
    table["margin_in_null_sd"] = table["margin"] / table["null_sd"].replace(0.0, np.nan)
    order = [name for name, _ in INVARIANT_MEASURES if name in table.index]
    return table.loc[order + [i for i in table.index if i not in order]].reset_index()


#: Everything an ensemble is scored on, and what each one is invariant to. Ordered from
#: strictest to most permissive, which is also the order in which they should be read: a
#: measure only means something if the stricter ones above it have been reported too.
INVARIANT_MEASURES: tuple[tuple[str, str], ...] = (
    ("matched weight correlation", "signed permutation (exact model symmetry)"),
    ("eigenspectrum", "any similarity transform"),
    ("state Procrustes", "orthogonal transform"),
    ("state CKA", "orthogonal transform and isotropic scaling"),
)


def ensemble_invariants(
    run_dirs: Sequence[str | Path],
    *,
    device: str = "cpu",
    include_matched_weights: bool = True,
) -> pd.DataFrame:
    """Pairwise agreement between fits, one row per (pair, measure).

    Every measure here is invariant to at least the signed permutation the model is free
    to choose, so a low value means the fits genuinely differ rather than that their units
    are numbered differently. What the numbers mean still depends on a null: see
    :func:`untrained_reference_spectra` for the architectural floor, and the
    ``target_surrogate`` training option for the stronger one.
    """
    loaded = []
    for run_dir in run_dirs:
        replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
        h_seq = replay["h_seq"].detach().cpu().numpy()
        loaded.append({
            "run_dir": str(run_dir),
            "model": replay["model"],
            "states": h_seq.reshape(-1, h_seq.shape[-1]),
            "spectrum": recurrent_eigenspectrum(replay["model"]),
        })

    rows: list[dict[str, object]] = []
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            left, right = loaded[i], loaded[j]
            pair = f"{i}-{j}"
            values = {
                "eigenspectrum": eigenspectrum_agreement(left["spectrum"], right["spectrum"]),
                "state Procrustes": procrustes_similarity(left["states"], right["states"]),
                "state CKA": linear_cka(left["states"], right["states"]),
            }
            if include_matched_weights:
                values["matched weight correlation"] = matched_unit_alignment(
                    left["model"], right["model"]
                )["matched_weight_correlation"]
            for measure, value in values.items():
                rows.append({"pair": pair, "measure": measure,
                             "agreement": float(value), "ensemble": "fitted"})
    return pd.DataFrame(rows)
