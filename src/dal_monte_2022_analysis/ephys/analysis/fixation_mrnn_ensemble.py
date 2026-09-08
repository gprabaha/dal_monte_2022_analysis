"""One architecture fitted many times: what the fits agree on, and how the fixation types differ.

The final series settles on one bottlenecked model (task 02's ``selected_bottleneck.yaml``)
and refits it ten times. This module holds the analyses that only make sense on such an
ensemble, alongside the ten-seed dense ensemble task 01 produced:

* **cost against the dense network** -- per fixation type, the drop in ceiling-relative
  :math:`R^2` from dense to constrained, with a bootstrap over seeds. This is the direct
  test of "does the bottleneck hurt one fixation type more", which the grid could only
  show as a mean.
* **fixation-type similarity** -- for every property the model has per condition (target,
  state, drives, cross share, lesion profile, fixed points), how similar each *pair* of
  conditions is within one fit. The question the chapter asks is whether the two face
  conditions are closer to each other than either is to object, and whether the network
  inherits that from the data or imposes it.
* **dynamics** -- fixed points of each condition's autonomous system, their stability, how
  far the trajectory gets to them, the flow field around the trajectory, and the local
  linearisation along the trajectory, whole network and per region.
* **lesions** -- the bidirectional pair lesions as a matrix (which pairs matter, for
  which fixation type, in which target region), and what a lesion does to the lesioned
  network's own dynamics rather than only to its fit.
* **consistency** -- for every per-condition quantity, whether the condition ordering
  holds across seeds and how far the conditions sit apart relative to the seed spread.

Everything is computed *within* one fit and compared *across* conditions, so the
non-identifiability the audit established (weights, spectra and state geometry all at
their untrained floors) does not touch it: a signed permutation applies to every term of
a within-model contrast and cancels.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_audit import (
    _lesion_groups,
    _target_by_region,
    condition_bias,
    corrected_jacobian_eigenvalues,
    eigenspectrum_distance,
    seed_run_dirs,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
    extract_region_current_vectors,
    replay_fixation_mrnn_run,
    replay_fixation_mrnn_run_with_ablations,
)

CONDITION_ORDER: tuple[str, ...] = ("face_interactive", "face_non_interactive", "object")
CONDITION_PAIRS: tuple[tuple[str, str], ...] = (
    ("face_interactive", "face_non_interactive"),
    ("face_interactive", "object"),
    ("face_non_interactive", "object"),
)
PAIR_LABELS: dict[tuple[str, str], str] = {
    ("face_interactive", "face_non_interactive"): "FI–FN",
    ("face_interactive", "object"): "FI–OBJ",
    ("face_non_interactive", "object"): "FN–OBJ",
}

__all__: list[str] = ["CONDITION_PAIRS", "PAIR_LABELS"]


def seed_of(run_dir: str | Path) -> str:
    return Path(run_dir).name.replace("seed=", "")


def collect(arm_dirs: Mapping[str, str | Path], function, *, device: str = "cpu", **kwargs) -> pd.DataFrame:
    """Apply a per-run function to every seed of every arm, tagging rows with ``arm``."""
    frames = []
    for arm, path in arm_dirs.items():
        for run_dir in seed_run_dirs(path):
            frames.append(function(run_dir, device=device, **kwargs).assign(arm=arm))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ += ["collect", "seed_of"]


# ======================================================================================
# 1. Cost against the dense network
# ======================================================================================


def fit_cost_by_condition(
    constrained: pd.DataFrame,
    dense: pd.DataFrame,
    *,
    n_bootstrap: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per fixation type: fit in each arm, the cost of the constraint, and a bootstrap CI.

    Regions are pooled per seed (mean over the four regions). The two ensembles are not
    paired -- seeds are independent draws -- so the CI resamples seeds within each arm.
    Two readings of the cost are given: the drop in ceiling-relative :math:`R^2`, and the
    ratio of unexplained variance (constrained over dense), which is scale-free and so the
    fairer comparison between a fixation type with half the variance and the other two.
    """
    generator = np.random.default_rng(int(seed))
    per_c = constrained.groupby(["seed", "condition"])[["r2_vs_ceiling", "r2"]].mean().unstack("condition")
    per_d = dense.groupby(["seed", "condition"])[["r2_vs_ceiling", "r2"]].mean().unstack("condition")
    conditions = [c for c in CONDITION_ORDER if c in per_c["r2_vs_ceiling"].columns]
    rows: list[dict[str, object]] = []
    draws: dict[str, np.ndarray] = {}
    for condition in conditions:
        c_fit = per_c["r2_vs_ceiling"][condition].to_numpy(dtype=float)
        d_fit = per_d["r2_vs_ceiling"][condition].to_numpy(dtype=float)
        c_un = 1.0 - per_c["r2"][condition].to_numpy(dtype=float)
        d_un = 1.0 - per_d["r2"][condition].to_numpy(dtype=float)
        boot_cost = np.empty(n_bootstrap)
        boot_ratio = np.empty(n_bootstrap)
        for k in range(n_bootstrap):
            ic = generator.integers(0, len(c_fit), len(c_fit))
            idx = generator.integers(0, len(d_fit), len(d_fit))
            boot_cost[k] = d_fit[idx].mean() - c_fit[ic].mean()
            boot_ratio[k] = c_un[ic].mean() / d_un[idx].mean()
        draws[condition] = boot_cost
        rows.append({
            "condition": condition,
            "dense_mean": float(d_fit.mean()), "dense_sd": float(d_fit.std(ddof=1)),
            "constrained_mean": float(c_fit.mean()), "constrained_sd": float(c_fit.std(ddof=1)),
            "cost": float(d_fit.mean() - c_fit.mean()),
            "cost_ci_low": float(np.percentile(boot_cost, 2.5)),
            "cost_ci_high": float(np.percentile(boot_cost, 97.5)),
            "unexplained_ratio": float(c_un.mean() / d_un.mean()),
            "unexplained_ratio_ci_low": float(np.percentile(boot_ratio, 2.5)),
            "unexplained_ratio_ci_high": float(np.percentile(boot_ratio, 97.5)),
        })
    table = pd.DataFrame(rows)
    # The differential test: is interactive face's cost larger than each other condition's?
    if "face_interactive" in draws:
        for other in conditions:
            if other == "face_interactive":
                continue
            diff = draws["face_interactive"] - draws[other]
            table.loc[table["condition"] == other, "fi_extra_cost"] = float(
                table.loc[table["condition"] == "face_interactive", "cost"].iloc[0]
                - table.loc[table["condition"] == other, "cost"].iloc[0])
            table.loc[table["condition"] == other, "fi_extra_cost_ci_low"] = float(np.percentile(diff, 2.5))
            table.loc[table["condition"] == other, "fi_extra_cost_ci_high"] = float(np.percentile(diff, 97.5))
            table.loc[table["condition"] == other, "p_fi_costs_more"] = float((diff > 0).mean())
    return table


def fit_cost_by_cell(constrained: pd.DataFrame, dense: pd.DataFrame) -> pd.DataFrame:
    """The same cost per region x condition cell, mean over seeds in each arm."""
    c = constrained.groupby(["region", "condition"])["r2_vs_ceiling"].agg(["mean", "std"]).add_prefix("constrained_")
    d = dense.groupby(["region", "condition"])["r2_vs_ceiling"].agg(["mean", "std"]).add_prefix("dense_")
    out = pd.concat([c, d], axis=1).reset_index()
    out["cost"] = out["dense_mean"] - out["constrained_mean"]
    return out


def reconstruction_traces(
    run_dirs: Sequence[str | Path],
    *,
    region: str,
    indices: Sequence[int] = (0, 1, 2),
    device: str = "cpu",
) -> pd.DataFrame:
    """Observed and predicted PC traces of one region for every seed, for an overlay gallery."""
    frames = []
    for run_dir in run_dirs:
        replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
        timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
        target = _target_by_region(replay)[region]
        predicted = replay["output_by_region"][region].detach().cpu().numpy()
        for c, condition in enumerate(replay["condition_order"]):
            for index in indices:
                frames.append(pd.DataFrame({
                    "seed": seed_of(run_dir), "region": region, "condition": str(condition), "index": int(index),
                    "time_s": timeline[: target.shape[1]],
                    "observed": target[c, :, index], "predicted": predicted[c, : target.shape[1], index],
                }))
    return pd.concat(frames, ignore_index=True)


__all__ += ["fit_cost_by_cell", "fit_cost_by_condition", "reconstruction_traces"]


# ======================================================================================
# 2. Fixation-type similarity
# ======================================================================================


def _pair_correlations(array: np.ndarray) -> dict[tuple[int, int], float]:
    """Pearson correlation between two conditions' blocks of a ``(condition, time, dim)`` array.

    Each condition's block is centred over its own time axis per dimension, so the
    correlation measures whether the two conditions trace the same *shape* of temporal
    modulation. Offsets between conditions are measured separately by
    :func:`_pair_distances`. (Centring over the pooled axis instead would make three
    conditions anti-correlated by construction, since their offsets sum to zero.)
    """
    centred = array - array.mean(axis=1, keepdims=True)
    out: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(array.shape[0]), 2):
        a, b = centred[i].ravel(), centred[j].ravel()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        out[(i, j)] = float(a @ b / denominator) if denominator > 0 else np.nan
    return out


def _pair_distances(array: np.ndarray) -> dict[tuple[int, int], float]:
    """RMS distance between two conditions' blocks, offsets included, over the pooled spread.

    Normalised by the RMS of the array about its pooled mean, so 1 means the two conditions
    are as far apart as the whole array's typical excursion; comparable across seeds, arms
    and quantities of different scale.
    """
    scale = float(np.sqrt(((array - array.mean(axis=(0, 1), keepdims=True)) ** 2).mean(axis=(0, 1)).sum()))
    out: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(array.shape[0]), 2):
        rms = float(np.sqrt(((array[i] - array[j]) ** 2).mean(axis=0).sum()))
        out[(i, j)] = rms / scale if scale > 0 else np.nan
    return out


def _similarity_rows(array: np.ndarray, conditions: Sequence[str], *, prop: str, scope: str, seed: str,
                     kind: str = "similarity") -> list[dict[str, object]]:
    """Both readings of one property: shape correlation (``prop``) and normalised distance (``prop_distance``)."""
    rows = []
    for (i, j), value in _pair_correlations(array).items():
        rows.append({"seed": seed, "property": prop, "scope": scope, "kind": kind,
                     "condition_a": conditions[i], "condition_b": conditions[j],
                     "pair": PAIR_LABELS.get((conditions[i], conditions[j]), f"{conditions[i]}–{conditions[j]}"),
                     "value": value})
    for (i, j), value in _pair_distances(array).items():
        rows.append({"seed": seed, "property": f"{prop}_distance", "scope": scope, "kind": "distance",
                     "condition_a": conditions[i], "condition_b": conditions[j],
                     "pair": PAIR_LABELS.get((conditions[i], conditions[j]), f"{conditions[i]}–{conditions[j]}"),
                     "value": value})
    return rows


def condition_similarity(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """How similar each pair of fixation types is, for every time-course property of one fit.

    One row per (property, scope, pair). Properties:

    ``target``       the data itself (PC trajectories) -- the reference every model row is
                     read against;
    ``state``        the hidden state, per region and whole network;
    ``self_drive``   :math:`W_{rr}h_r(t)`, per region;
    ``cross_drive``  :math:`\\sum_{s\\ne r}W_{rs}h_s(t)`, per region;
    ``cross_share``  the time course of the network's share of a region's drive, all four
                     regions stacked (scope ``network``).

    All are correlations after pooled centring, so a value near 1 means the two fixation
    types trace the same shape and differ only in scale or offset.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    seed = seed_of(run_dir)
    regions = list(replay["region_order"])
    conditions = [str(c) for c in replay["condition_order"]]
    slices = replay["model"].hidden_region_slices()
    states = replay["h_seq"].detach().cpu().numpy()
    currents = extract_region_current_vectors(replay)
    targets = _target_by_region(replay)

    rows: list[dict[str, object]] = []
    for region in regions:
        rows += _similarity_rows(targets[region], conditions, prop="target", scope=region, seed=seed)
        rows += _similarity_rows(states[:, :, slices[region]], conditions, prop="state", scope=region, seed=seed)
        own = currents[(region, region)].numpy()
        cross = sum(currents[(s, region)].numpy() for s in regions if s != region)
        rows += _similarity_rows(own, conditions, prop="self_drive", scope=region, seed=seed)
        rows += _similarity_rows(cross, conditions, prop="cross_drive", scope=region, seed=seed)
    rows += _similarity_rows(np.concatenate([targets[r] for r in regions], axis=-1), conditions,
                             prop="target", scope="network", seed=seed)
    rows += _similarity_rows(states, conditions, prop="state", scope="network", seed=seed)
    share = []
    for region in regions:
        own = np.linalg.norm(currents[(region, region)].numpy(), axis=-1) ** 2
        cross = sum(np.linalg.norm(currents[(s, region)].numpy(), axis=-1) ** 2 for s in regions if s != region)
        share.append(cross / (own + cross))
    rows += _similarity_rows(np.stack(share, axis=-1), conditions, prop="cross_share", scope="network", seed=seed)
    return pd.DataFrame(rows)


def lesion_profile_similarity(lesions: pd.DataFrame) -> pd.DataFrame:
    """How similar two fixation types' lesion-damage profiles are, per fit.

    The profile is the vector of damage over every lesion in the battery (random controls
    excluded), in absolute squared error summed over target regions. Two conditions with
    correlated profiles are hurt by the same connections.
    """
    block = lesions[lesions["lesion_kind"] != "random control"]
    keys = ["arm", "seed"] if "arm" in block.columns else ["seed"]
    rows: list[dict[str, object]] = []
    for key, chunk in block.groupby(keys):
        wide = chunk.groupby(["lesion", "condition"])["damage_sse"].sum().unstack("condition")
        conditions = [c for c in CONDITION_ORDER if c in wide.columns]
        array = np.stack([wide[c].to_numpy(dtype=float)[:, None] for c in conditions])  # (cond, lesion, 1)
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for row in _similarity_rows(array, conditions, prop="lesion_profile", scope="network", seed=str(entry["seed"])):
            if row["kind"] == "similarity":
                rows.append({**entry, **row})
    return pd.DataFrame(rows)


def similarity_summary(similarity: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, property, scope): mean value per pair, and how often the two faces are the closest pair.

    ``fi_fn_closest`` is the fraction of fits in which FI–FN is the most similar pair (or
    the smallest distance, for ``kind == 'distance'``). That fraction, not the mean, is the
    statistic to read: a mean can be carried by one fit.
    """
    keys = [k for k in ("arm", "property", "scope") if k in similarity.columns]
    rows: list[dict[str, object]] = []
    for key, block in similarity.groupby(keys):
        wide = block.pivot_table(index="seed", columns="pair", values="value")
        pairs = [PAIR_LABELS[p] for p in CONDITION_PAIRS if PAIR_LABELS[p] in wide.columns]
        wide = wide[pairs].dropna()
        if wide.empty:
            continue
        kind = str(block["kind"].iloc[0])
        closest = wide.idxmin(axis=1) if kind == "distance" else wide.idxmax(axis=1)
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        entry.update({"kind": kind, "n_fits": int(len(wide))})
        for pair in pairs:
            entry[f"{pair}"] = float(wide[pair].mean())
            entry[f"{pair} sd"] = float(wide[pair].std(ddof=1)) if len(wide) > 1 else np.nan
        entry["fi_fn_closest"] = float((closest == "FI–FN").mean())
        if kind == "distance":
            entry["fi_fn_closer_than_fi_obj"] = float((wide["FI–FN"] < wide["FI–OBJ"]).mean())
            entry["fi_fn_closer_than_fn_obj"] = float((wide["FI–FN"] < wide["FN–OBJ"]).mean())
        else:
            entry["fi_fn_closer_than_fi_obj"] = float((wide["FI–FN"] > wide["FI–OBJ"]).mean())
            entry["fi_fn_closer_than_fn_obj"] = float((wide["FI–FN"] > wide["FN–OBJ"]).mean())
        rows.append(entry)
    return pd.DataFrame(rows)


__all__ += ["condition_similarity", "lesion_profile_similarity", "similarity_summary"]


# ======================================================================================
# 3. Dynamics: fixed points, flow fields, local linearisation
# ======================================================================================


def _condition_map(replay: Mapping[str, object], condition_index: int):
    """The autonomous update ``h -> phi(W h + b_c)`` of one condition, as a torch callable."""
    model = replay["model"]
    weight = model.recurrent_weight_matrix().detach()
    bias = torch.as_tensor(condition_bias(replay, condition_index), dtype=weight.dtype, device=weight.device)
    activation = model.mrnn.activation
    return (lambda h: activation(h @ weight.T + bias)), weight, bias


def find_fixed_points(
    replay: Mapping[str, object],
    condition_index: int,
    *,
    n_starts: int = 48,
    iterations: int = 2000,
    learning_rate: float = 0.05,
    noise_scale: float = 0.5,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Minimise ``q(h) = ½‖φ(Wh+b_c) − h‖²`` from many starts; return every end point with its speed.

    Half the starts are states the trajectory visits, the other half those states plus
    Gaussian noise scaled to the state's spread, so that fixed points off the trajectory
    can be found as well as the ones it approaches. The learning rate decays on a cosine
    so the end points are converged rather than still moving.
    """
    step, weight, bias = _condition_map(replay, condition_index)
    h_seq = replay["h_seq"].detach()[condition_index]
    generator = np.random.default_rng(int(seed))
    picks = generator.choice(h_seq.shape[0], size=int(n_starts), replace=True)
    starts = h_seq[picks].clone()
    half = int(n_starts) // 2
    spread = float(h_seq.std())
    noise = torch.as_tensor(generator.normal(size=tuple(starts[half:].shape)), dtype=starts.dtype, device=starts.device)
    starts[half:] = starts[half:] + noise_scale * spread * noise
    state = starts.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([state], lr=float(learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(iterations), eta_min=learning_rate * 1e-3)
    for _ in range(int(iterations)):
        optimizer.zero_grad()
        residual = step(state) - state
        loss = 0.5 * (residual ** 2).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
    points = state.detach().cpu().numpy().astype(float)
    weight_np = weight.detach().cpu().numpy().astype(float)
    bias_np = bias.detach().cpu().numpy().astype(float)
    points, speed = _newton_polish(points, weight_np, bias_np)
    return {"points": points, "speed": speed, "from_trajectory": np.arange(n_starts) < half}


def _newton_polish(points: np.ndarray, weight: np.ndarray, bias: np.ndarray, *, steps: int = 30, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Damped Newton iterations on ``g(h) = tanh(W h + b) − h`` from each Adam end point.

    Adam gets close; Newton finishes. A step is accepted only if it lowers the residual,
    halving its length up to eight times before giving up on that point, so a start far
    from any fixed point keeps its Adam end point and its (larger) speed.
    """
    out = points.copy()
    speeds = np.empty(len(points))
    for k, h in enumerate(out):
        residual = np.tanh(h @ weight.T + bias) - h
        norm = float(np.linalg.norm(residual))
        for _ in range(int(steps)):
            if norm < tol:
                break
            derivative = 1.0 - np.tanh(h @ weight.T + bias) ** 2
            jac = derivative[:, None] * weight - np.eye(len(h))
            try:
                delta = np.linalg.solve(jac, -residual)
            except np.linalg.LinAlgError:
                break
            scale, accepted = 1.0, False
            for _ in range(8):
                candidate = h + scale * delta
                candidate_residual = np.tanh(candidate @ weight.T + bias) - candidate
                candidate_norm = float(np.linalg.norm(candidate_residual))
                if candidate_norm < norm:
                    h, residual, norm, accepted = candidate, candidate_residual, candidate_norm, True
                    break
                scale *= 0.5
            if not accepted:
                break
        out[k] = h
        speeds[k] = norm
    return out, speeds


def _cluster_points(points: np.ndarray, speed: np.ndarray, *, speed_tol: float, merge_tol: float):
    """Greedy merge of converged end points into distinct fixed points, slowest first."""
    order = np.argsort(speed)
    centres: list[np.ndarray] = []
    members: list[list[int]] = []
    for index in order:
        if speed[index] > speed_tol:
            continue
        for k, centre in enumerate(centres):
            if np.linalg.norm(points[index] - centre) < merge_tol:
                members[k].append(int(index))
                break
        else:
            centres.append(points[index])
            members.append([int(index)])
    return centres, members


def condition_fixed_points(
    run_dir: str | Path,
    *,
    n_starts: int = 48,
    iterations: int = 2000,
    speed_tol: float = 1e-6,
    slow_tol: float = 1e-2,
    merge_tol: float = 0.1,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed points of every condition's autonomous system in one fit, and what the trajectory does with them.

    Returns ``(points, summary)``. ``points`` has one row per distinct fixed point with its
    stability (Jacobian with the bias included), its distance from the trajectory's end and
    from the trajectory's nearest visit, and the share of starts that fell into it.
    ``summary`` has one row per condition: how many fixed points were found, whether the
    trajectory ends at one (distance from ``h(T)`` to the nearest, relative to the state's
    own spread), and the stability of that nearest one. Distances are in state units
    normalised by the pooled state extent, so they compare across seeds and arms.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    seed = seed_of(run_dir)
    model = replay["model"]
    regions = list(replay["region_order"])
    slices = model.hidden_region_slices()
    conditions = [str(c) for c in replay["condition_order"]]
    states = replay["h_seq"].detach().cpu().numpy()
    extent = float(np.linalg.norm(states - states.mean(axis=(0, 1), keepdims=True), axis=-1).mean())
    targets = _target_by_region(replay)
    target_extent = {r: float(np.linalg.norm(t - t.mean(axis=(0, 1), keepdims=True), axis=-1).mean())
                     for r, t in targets.items()}

    point_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for index, condition in enumerate(conditions):
        found = find_fixed_points(replay, index, n_starts=n_starts, iterations=iterations)
        centres, members = _cluster_points(found["points"], found["speed"], speed_tol=speed_tol, merge_tol=merge_tol)
        bias = condition_bias(replay, index)
        trajectory = states[index]
        end = trajectory[-1]
        nearest_to_end = None
        for k, (centre, group) in enumerate(zip(centres, members)):
            eigenvalues = corrected_jacobian_eigenvalues(replay, centre, bias)
            top = eigenvalues[0]
            with torch.no_grad():
                readout_gap = float(np.mean([
                    np.linalg.norm(model.output_heads[r](torch.as_tensor(centre[slices[r]], dtype=torch.float32)
                                                          .to(next(model.parameters()).device)).cpu().numpy()
                                   - targets[r][index, -1]) / target_extent[r]
                    for r in regions]))
            distance_end = float(np.linalg.norm(end - centre) / extent)
            distance_min = float(np.linalg.norm(trajectory - centre, axis=-1).min() / extent)
            row = {
                "seed": seed, "condition": condition, "fixed_point": k,
                "speed": float(found["speed"][group].min()),
                "basin_share": float(len(group) / len(found["speed"])),
                "found_from_trajectory": float(found["from_trajectory"][group].mean()),
                "top_modulus": float(np.abs(top)),
                "n_expanding": int((np.abs(eigenvalues) > 1.0).sum()),
                "stable": bool(np.abs(top) < 1.0),
                "dominant_period_bins": float(2 * np.pi / abs(np.angle(top))) if abs(np.angle(top)) > 1e-6 else np.inf,
                "distance_to_trajectory_end": distance_end,
                "distance_to_trajectory_min": distance_min,
                "readout_gap_at_end": readout_gap,
                "state_norm": float(np.linalg.norm(centre)),
            }
            point_rows.append(row)
            if nearest_to_end is None or distance_end < nearest_to_end["distance_to_trajectory_end"]:
                nearest_to_end = {**row, "centre": centre}
        n_unconverged = int((found["speed"] > speed_tol).sum())
        slow_centres, _ = _cluster_points(found["points"], found["speed"], speed_tol=slow_tol, merge_tol=merge_tol)
        summary = {
            "seed": seed, "condition": condition,
            "n_fixed_points": len(centres), "n_slow_points": len(slow_centres), "n_unconverged_starts": n_unconverged,
            "slowest_speed": float(found["speed"].min()),
            "trajectory_end_speed": float(np.linalg.norm(trajectory[-1] - trajectory[-2])),
            "trajectory_extent": float(np.linalg.norm(trajectory - trajectory.mean(axis=0), axis=-1).mean() / extent),
        }
        if nearest_to_end is not None:
            for key in ("distance_to_trajectory_end", "distance_to_trajectory_min", "top_modulus", "n_expanding",
                        "stable", "readout_gap_at_end", "basin_share", "dominant_period_bins"):
                summary[f"nearest_{key}"] = nearest_to_end[key]
            summary["_centre"] = nearest_to_end["centre"]
        summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    return pd.DataFrame(point_rows), summary_frame


def fixed_point_condition_distances(summary: pd.DataFrame, run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """Distances between the conditions' trajectory-end fixed points, and between their local spectra.

    Both are within-model quantities. The state distance is normalised by the pooled state
    extent; the spectrum distance is :func:`eigenspectrum_distance` between the Jacobians
    at the two fixed points (each with its own condition's bias). Emitted in the
    similarity schema with ``kind == 'distance'`` so :func:`similarity_summary` can rank
    the pairs.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    states = replay["h_seq"].detach().cpu().numpy()
    extent = float(np.linalg.norm(states - states.mean(axis=(0, 1), keepdims=True), axis=-1).mean())
    conditions = [str(c) for c in replay["condition_order"]]
    centres = {row["condition"]: (row.get("_centre") if isinstance(row.get("_centre"), np.ndarray) else None)
               for _, row in summary.iterrows()}
    spectra = {}
    for index, condition in enumerate(conditions):
        centre = centres.get(condition)
        if centre is not None:
            spectra[condition] = corrected_jacobian_eigenvalues(replay, centre, condition_bias(replay, index))
    rows: list[dict[str, object]] = []
    seed = seed_of(run_dir)
    for a, b in CONDITION_PAIRS:
        if centres.get(a) is None or centres.get(b) is None:
            continue
        base = {"seed": seed, "condition_a": a, "condition_b": b, "pair": PAIR_LABELS[(a, b)], "scope": "network", "kind": "distance"}
        rows.append({**base, "property": "fixed_point_distance", "value": float(np.linalg.norm(centres[a] - centres[b]) / extent)})
        rows.append({**base, "property": "fixed_point_spectrum_distance", "value": float(eigenspectrum_distance(spectra[a], spectra[b]))})
        # And the trajectory end points themselves, for comparison with the fixed points.
        ia, ib = conditions.index(a), conditions.index(b)
        rows.append({**base, "property": "trajectory_end_distance", "value": float(np.linalg.norm(states[ia, -1] - states[ib, -1]) / extent)})
    return pd.DataFrame(rows)


def ensemble_fixed_points(arm_dirs: Mapping[str, str | Path], *, device: str = "cpu", **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """:func:`condition_fixed_points` over every seed of every arm: ``(points, summary, distances)``."""
    points, summaries, distances = [], [], []
    for arm, path in arm_dirs.items():
        for run_dir in seed_run_dirs(path):
            p, s = condition_fixed_points(run_dir, device=device, **kwargs)
            distances.append(fixed_point_condition_distances(s, run_dir, device=device).assign(arm=arm))
            points.append(p.assign(arm=arm))
            summaries.append(s.drop(columns=["_centre"], errors="ignore").assign(arm=arm))
    cat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return cat(points), cat(summaries), cat(distances)


def state_plane(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-two principal axes of a ``(condition, time, dim)`` state array: ``(origin, basis, explained)``."""
    flat = states.reshape(-1, states.shape[-1])
    origin = flat.mean(axis=0)
    _, singular, vt = np.linalg.svd(flat - origin, full_matrices=False)
    explained = singular ** 2 / (singular ** 2).sum()
    return origin, vt[:2].T, explained[:2]


def flow_fields(
    run_dir: str | Path,
    *,
    scope: str = "network",
    n_grid: int = 19,
    margin: float = 1.35,
    fixed_points: pd.DataFrame | None = None,
    device: str = "cpu",
    n_starts: int = 32,
    iterations: int = 1500,
) -> dict[str, object]:
    """The flow of each condition's map in the plane of its top-two state axes, with the trajectory and fixed points.

    ``scope == 'network'``: the full 160-unit map on the plane spanned by the pooled
    states' first two principal axes. ``scope == <region>``: that region's 40 units under
    its own recurrence, with the drive it receives from the other three regions **clamped
    at its time average for that condition** -- the flow the region would show if the rest
    of the network held still. That is an approximation, stated on the figure: a region is
    not an autonomous system, and its "fixed points" here are conditional on that input.

    Each arrow is ``F(h) − h`` for the grid point ``h`` in the plane, projected back onto
    the plane; the background is its magnitude, the speed of the flow.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    model = replay["model"]
    regions = list(replay["region_order"])
    slices = model.hidden_region_slices()
    conditions = [str(c) for c in replay["condition_order"]]
    states = replay["h_seq"].detach().cpu().numpy()
    weight = model.recurrent_weight_matrix().detach().cpu().numpy()
    currents = extract_region_current_vectors(replay)

    if scope == "network":
        sub = states
        block = weight
        def bias_for(index):
            return condition_bias(replay, index)
    else:
        sl = slices[scope]
        sub = states[:, :, sl]
        block = weight[sl, sl]
        def bias_for(index):
            clamped = sum(currents[(s, scope)].numpy()[index] for s in regions if s != scope).mean(axis=0)
            return condition_bias(replay, index)[sl] + clamped

    origin, basis, explained = state_plane(sub)
    projected = (sub - origin) @ basis  # (condition, time, 2)
    half = margin * np.abs(projected).max(axis=(0, 1))
    xs = np.linspace(-half[0], half[0], n_grid)
    ys = np.linspace(-half[1], half[1], n_grid)
    gx, gy = np.meshgrid(xs, ys)
    grid = origin + gx[..., None] * basis[:, 0] + gy[..., None] * basis[:, 1]

    out: dict[str, object] = {"seed": seed_of(run_dir), "scope": scope, "explained": explained,
                              "xs": xs, "ys": ys, "conditions": {}}
    for index, condition in enumerate(conditions):
        bias = bias_for(index)
        nxt = np.tanh(grid @ block.T + bias)
        velocity = nxt - grid
        u = velocity @ basis[:, 0]
        v = velocity @ basis[:, 1]
        entry = {"trajectory": projected[index], "u": u, "v": v, "speed": np.linalg.norm(velocity, axis=-1)}
        # Fixed points in the plane: the network's from the table if given, otherwise found here.
        if scope == "network":
            if fixed_points is not None and "_centre" in fixed_points.columns:
                block_fp = fixed_points[fixed_points["condition"] == condition]
                centres = [np.asarray(c) for c in block_fp["_centre"] if isinstance(c, np.ndarray)]
                stable = list(block_fp["nearest_stable"]) if "nearest_stable" in block_fp else [True] * len(centres)
            else:
                found = find_fixed_points(replay, index, n_starts=n_starts, iterations=iterations)
                centres, members = _cluster_points(found["points"], found["speed"], speed_tol=1e-6, merge_tol=0.1)
                stable = [bool(np.abs(corrected_jacobian_eigenvalues(replay, c, bias)[0]) < 1.0) for c in centres]
        else:
            # The clamped region map's own fixed points, by the same minimisation.
            found = _numpy_fixed_points(block, bias, sub[index], n_starts=n_starts, iterations=iterations)
            centres, members = _cluster_points(found["points"], found["speed"], speed_tol=1e-6, merge_tol=0.1)
            stable = []
            for c in centres:
                derivative = 1.0 - np.tanh(c @ block.T + bias) ** 2
                stable.append(bool(np.abs(np.linalg.eigvals(derivative[:, None] * block)).max() < 1.0))
        entry["fixed_points"] = np.asarray([(c - origin) @ basis for c in centres]).reshape(-1, 2)
        entry["fixed_point_stable"] = np.asarray(stable, dtype=bool)
        out["conditions"][condition] = entry
    return out


def _numpy_fixed_points(block: np.ndarray, bias: np.ndarray, trajectory: np.ndarray, *, n_starts: int, iterations: int,
                        learning_rate: float = 0.05, noise_scale: float = 0.5, seed: int = 0) -> dict[str, np.ndarray]:
    """Twin of :func:`find_fixed_points` for a small map ``h -> tanh(block h + bias)`` given as arrays."""
    generator = np.random.default_rng(int(seed))
    picks = generator.choice(trajectory.shape[0], size=int(n_starts), replace=True)
    starts = trajectory[picks].copy()
    half = int(n_starts) // 2
    starts[half:] += noise_scale * float(trajectory.std()) * generator.normal(size=starts[half:].shape)
    state = torch.as_tensor(starts, dtype=torch.float32).requires_grad_(True)
    weight_t = torch.as_tensor(np.asarray(block), dtype=torch.float32)
    bias_t = torch.as_tensor(np.asarray(bias), dtype=torch.float32)
    optimizer = torch.optim.Adam([state], lr=float(learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(iterations), eta_min=learning_rate * 1e-3)
    for _ in range(int(iterations)):
        optimizer.zero_grad()
        residual = torch.tanh(state @ weight_t.T + bias_t) - state
        loss = 0.5 * (residual ** 2).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
    points, speed = _newton_polish(state.detach().numpy().astype(float), np.asarray(block, dtype=float), np.asarray(bias, dtype=float))
    return {"points": points, "speed": speed, "from_trajectory": np.arange(n_starts) < half}


def jacobian_along_trajectory(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """Local linearisation at every state the trajectory visits, whole network and per region block.

    At ``h(t)`` the Jacobian of the update is ``diag(1 − h(t+1)²) W`` exactly, because
    ``h(t+1) = tanh(W h(t) + b)``. Reported per (scope, condition, time): the largest
    eigenvalue modulus (above 1 locally expanding), how many modes expand, the dominant
    mode's rotation frequency in hertz, and the slowest contracting timescale. The region
    rows use the block ``diag(1 − h_r(t+1)²) W_rr``: the region's own local dynamics with
    its input held fixed.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    model = replay["model"]
    slices = model.hidden_region_slices()
    conditions = [str(c) for c in replay["condition_order"]]
    timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
    bin_s = float(np.median(np.diff(timeline))) if timeline.size > 1 else 0.01
    states = replay["h_seq"].detach().cpu().numpy()
    weight = model.recurrent_weight_matrix().detach().cpu().numpy()
    scopes = {"network": slice(None), **slices}
    rows: list[dict[str, object]] = []
    seed = seed_of(run_dir)
    for index, condition in enumerate(conditions):
        for t in range(states.shape[1] - 1):
            derivative = 1.0 - states[index, t + 1] ** 2
            for scope, sl in scopes.items():
                jac = derivative[sl][:, None] * weight[sl, sl]
                eigenvalues = np.linalg.eigvals(jac)
                top = eigenvalues[np.argmax(np.abs(eigenvalues))]
                modulus = float(np.abs(top))
                rows.append({
                    "seed": seed, "scope": scope, "condition": condition, "time_s": float(timeline[min(t, timeline.size - 1)]),
                    "top_modulus": modulus,
                    "n_expanding": int((np.abs(eigenvalues) > 1.0).sum()),
                    "dominant_frequency_hz": float(abs(np.angle(top)) / (2 * np.pi * bin_s)),
                    "slowest_timescale_s": float(-bin_s / np.log(modulus)) if 0 < modulus < 1 else np.inf,
                    "gain_mean": float(derivative[sl].mean()),
                })
    return pd.DataFrame(rows)


__all__ += [
    "condition_fixed_points", "ensemble_fixed_points", "find_fixed_points", "fixed_point_condition_distances",
    "flow_fields", "jacobian_along_trajectory", "state_plane",
]


# ======================================================================================
# 4. Lesions: pairs, and the lesioned network's own dynamics
# ======================================================================================


def pair_lesion_table(lesions: pd.DataFrame) -> pd.DataFrame:
    """Bidirectional pair lesions as a long table: damage per (arm, seed, pair, condition, target region).

    Adds ``damage_share``: the pair's share of the summed damage over all six pairs within
    that (seed, condition), so pairs can be ranked without the arm's overall scale.
    """
    block = lesions[lesions["lesion_kind"] == "bidirectional"].copy()
    keys = [k for k in ("arm", "seed", "condition") if k in block.columns]
    total = block.groupby(keys)["damage_sse"].transform("sum")
    block["damage_share"] = np.where(total > 0, block["damage_sse"] / total, np.nan)
    block = block.rename(columns={"lesion": "pair"})
    return block[keys + ["pair", "region", "live_weights_removed", "damage_sse", "damage_shared_r2", "damage_share"]]


def lesion_ranking_by_condition(
    lesions: pd.DataFrame,
    *,
    kind: str = "bidirectional",
    n_permutations: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Kendall's τ between seeds' lesion rankings, separately per fixation type.

    The per-condition twin of ``audit.lesion_ranking_agreement``: whether the seeds agree
    on which connections matter *for interactive face*, and whether that ranking is the
    same one they agree on for the other two.
    """
    from scipy.stats import kendalltau

    generator = np.random.default_rng(int(seed))
    block = lesions[lesions["lesion_kind"] == kind]
    keys = ["arm", "condition"] if "arm" in block.columns else ["condition"]
    rows: list[dict[str, object]] = []
    for key, chunk in block.groupby(keys):
        wide = chunk.groupby(["seed", "lesion"])["damage_sse"].sum().unstack("lesion").dropna(axis=1)
        if wide.shape[0] < 2 or wide.shape[1] < 3:
            continue
        observed = [float(kendalltau(wide.iloc[i], wide.iloc[j]).statistic)
                    for i, j in combinations(range(wide.shape[0]), 2)]
        values = wide.values
        null = []
        for _ in range(int(n_permutations)):
            i, j = generator.choice(values.shape[0], size=2, replace=False)
            null.append(float(kendalltau(values[i], generator.permutation(values[j])).statistic))
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        consensus = wide.mean(axis=0).sort_values(ascending=False)
        entry.update({
            "lesion_kind": kind, "n_seeds": int(wide.shape[0]), "n_lesions": int(wide.shape[1]),
            "tau_mean": float(np.mean(observed)), "tau_min": float(np.min(observed)),
            "null_p95": float(np.percentile(null, 95)),
            "clears_null": bool(np.mean(observed) > np.percentile(null, 95)),
            "top_lesion": str(consensus.index[0]),
            "top_lesion_is_top_in": float((wide.idxmax(axis=1) == consensus.index[0]).mean()),
        })
        rows.append(entry)
    return pd.DataFrame(rows)


def cross_condition_lesion_ranking(lesions: pd.DataFrame, *, kind: str = "bidirectional") -> pd.DataFrame:
    """Within each seed, Kendall's τ between the lesion rankings of two fixation types.

    Whether the connections that matter for one fixation type are the ones that matter for
    another -- a within-fit comparison, emitted in the similarity schema.
    """
    from scipy.stats import kendalltau

    block = lesions[lesions["lesion_kind"] == kind]
    keys = ["arm", "seed"] if "arm" in block.columns else ["seed"]
    rows: list[dict[str, object]] = []
    for key, chunk in block.groupby(keys):
        wide = chunk.groupby(["lesion", "condition"])["damage_sse"].sum().unstack("condition")
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for a, b in CONDITION_PAIRS:
            if a in wide and b in wide:
                rows.append({**entry, "seed": str(entry["seed"]), "property": f"lesion_ranking_{kind}", "scope": "network",
                             "kind": "similarity", "condition_a": a, "condition_b": b, "pair": PAIR_LABELS[(a, b)],
                             "value": float(kendalltau(wide[a], wide[b]).statistic)})
    return pd.DataFrame(rows)


def lesion_dynamics(run_dir: str | Path, *, device: str = "cpu") -> pd.DataFrame:
    """What each lesion does to the lesioned network's own dynamics, not only to its fit.

    For the intact model and every lesion in the battery: the recurrent matrix's spectral
    radius and number of modes outside the unit circle (a property of the weights, no
    condition), and per condition the lesioned trajectory's state extent, speed and
    participation ratio, plus the local linearisation at the lesioned trajectory's end
    state (largest modulus, expanding modes, dominant frequency). ``delta_*`` columns are
    lesioned minus intact.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    regions = list(replay["region_order"])
    conditions = [str(c) for c in replay["condition_order"]]
    timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
    bin_s = float(np.median(np.diff(timeline))) if timeline.size > 1 else 0.01
    seed = seed_of(run_dir)

    def participation(block: np.ndarray) -> float:
        centred = block - block.mean(axis=0, keepdims=True)
        energy = np.linalg.svd(centred, compute_uv=False) ** 2
        return float(energy.sum() ** 2 / (energy ** 2).sum()) if energy.sum() > 0 else np.nan

    def describe(rep: Mapping[str, object], kind: str, name: str) -> list[dict[str, object]]:
        weight = rep["model"].recurrent_weight_matrix().detach().cpu().numpy()
        w_eigs = np.linalg.eigvals(weight)
        states = rep["h_seq"].detach().cpu().numpy()
        out = []
        for index, condition in enumerate(conditions):
            h = states[index]
            derivative = 1.0 - h[-1] ** 2
            eigenvalues = np.linalg.eigvals(derivative[:, None] * weight)
            top = eigenvalues[np.argmax(np.abs(eigenvalues))]
            out.append({
                "seed": seed, "lesion_kind": kind, "lesion": name, "condition": condition,
                "weight_spectral_radius": float(np.abs(w_eigs).max()),
                "weight_n_outside_unit": int((np.abs(w_eigs) > 1.0).sum()),
                "state_extent": float(np.linalg.norm(h - h.mean(axis=0), axis=-1).mean()),
                "state_speed": float(np.linalg.norm(np.diff(h, axis=0), axis=-1).mean()),
                "state_pr": participation(h),
                "end_top_modulus": float(np.abs(top)),
                "end_n_expanding": int((np.abs(eigenvalues) > 1.0).sum()),
                "end_dominant_frequency_hz": float(abs(np.angle(top)) / (2 * np.pi * bin_s)),
            })
        return out

    rows = describe(replay, "intact", "intact")
    for kind, name, blocks in _lesion_groups(regions):
        rows += describe(replay_fixation_mrnn_run_with_ablations(Path(run_dir), ablations=list(blocks), device=device), kind, name)
    frame = pd.DataFrame(rows)
    metrics = ["weight_spectral_radius", "weight_n_outside_unit", "state_extent", "state_speed", "state_pr",
               "end_top_modulus", "end_n_expanding", "end_dominant_frequency_hz"]
    intact = frame[frame["lesion_kind"] == "intact"].set_index("condition")[metrics]
    for metric in metrics:
        frame[f"delta_{metric}"] = frame[metric] - frame["condition"].map(intact[metric])
    return frame


__all__ += ["cross_condition_lesion_ranking", "lesion_dynamics", "lesion_ranking_by_condition", "pair_lesion_table"]


# ======================================================================================
# 5. Consistency across seeds
# ======================================================================================


def ordering_consistency(
    long: pd.DataFrame,
    *,
    value: str = "value",
    keys: Sequence[str] = ("arm", "property"),
) -> pd.DataFrame:
    """For every per-condition quantity: does the condition ordering hold across seeds, and how cleanly?

    ``long`` has one row per (keys..., seed, condition). Reports the fraction of seeds in
    which interactive face is the lowest and the highest, the fraction in which the two
    faces are adjacent in the ordering (object at an end), and ``separation``: the mean
    between-condition range divided by the mean within-condition seed sd. A separation
    well above 1 with an ordering fraction of 1 is a property of the solution set; a
    fraction of 0.6 is a property of some fits.
    """
    keys = [k for k in keys if k in long.columns]
    rows: list[dict[str, object]] = []
    for key, block in long.groupby(keys):
        wide = block.pivot_table(index="seed", columns="condition", values=value)
        conditions = [c for c in CONDITION_ORDER if c in wide.columns]
        wide = wide[conditions].dropna()
        if wide.empty or len(conditions) < 3:
            continue
        order = wide.apply(lambda r: tuple(r.sort_values().index), axis=1)
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        entry.update({
            "n_fits": int(len(wide)),
            "fi_lowest": float((wide.idxmin(axis=1) == "face_interactive").mean()),
            "fi_highest": float((wide.idxmax(axis=1) == "face_interactive").mean()),
            "object_at_end": float(order.map(lambda o: o[0] == "object" or o[-1] == "object").mean()),
            "modal_order": " < ".join(order.mode().iloc[0]),
            "modal_order_fraction": float((order == order.mode().iloc[0]).mean()),
            "separation": float((wide.max(axis=1) - wide.min(axis=1)).mean() / wide.std(ddof=1).mean())
            if len(wide) > 1 else np.nan,
        })
        for condition in conditions:
            entry[f"{condition}_mean"] = float(wide[condition].mean())
        rows.append(entry)
    return pd.DataFrame(rows)


def to_long(frame: pd.DataFrame, columns: Sequence[str], *, keys: Sequence[str] = ("arm", "seed", "condition")) -> pd.DataFrame:
    """Melt named per-condition columns of a wide table into the ``(property, value)`` schema."""
    keys = [k for k in keys if k in frame.columns]
    present = [c for c in columns if c in frame.columns]
    out = frame.groupby(keys)[present].mean().reset_index().melt(id_vars=keys, var_name="property", value_name="value")
    return out


__all__ += ["ordering_consistency", "to_long"]
