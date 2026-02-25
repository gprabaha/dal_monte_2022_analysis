"""Fit Poisson hidden semi-Markov models to joint face-fixation observations."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.data.behavioral_data import FixationBinaryVectorsData
from dal_monte_2022_analysis.behav.preprocessing.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


EPS = 1e-12
OBSERVATION_LABELS = {
    0: "00_none",
    1: "10_m1_only",
    2: "01_m2_only",
    3: "11_both",
}


@dataclass
class FaceFixationHSMMSettings:
    """Configuration for fitting a face-fixation Poisson-HSMM."""

    cfg_path: str
    input_modality: str = "fixation_binary_vectors"
    fixation_label: str = "face"
    output_subdir: str = "face_fixation_hsmm"
    grouping: str = "session"  # one of: session, day, pair, global
    n_hidden_states: int = 2
    max_duration: int = 300
    n_iter: int = 50
    tol: float = 1e-3
    n_init: int = 3
    seed: int = 13
    show_progress: bool = True
    show_inner_progress: bool = False
    inner_progress_every: int = 5
    parallelize_across_iterations: bool = True
    max_parallel_workers: int = 8
    allow_self_transitions: bool = False
    transition_pseudocount: float = 1.0
    emission_pseudocount: float = 1.0
    group_summary_filename: str = "face_fixation_hsmm_group_summary.csv"
    session_summary_filename: str = "face_fixation_hsmm_session_summary.csv"
    segments_filename: str = "face_fixation_hsmm_segments.csv"
    fits_filename: str = "face_fixation_hsmm_fits.pkl"
    dates: Optional[Sequence[str]] = None
    sessions: Optional[Sequence[str]] = None
    test_single: bool = False


@dataclass
class SessionSequence:
    """One session's joint m1/m2 face-fixation observation sequence."""

    date: str
    session: str
    monkey_name_m1: Optional[str]
    monkey_name_m2: Optional[str]
    observations: np.ndarray


@dataclass
class HSMMParameters:
    """Poisson-HSMM parameters."""

    initial_probs: np.ndarray  # shape: (K,)
    transition_probs: np.ndarray  # shape: (K, K)
    duration_lambdas: np.ndarray  # shape: (K,)
    emission_probs: np.ndarray  # shape: (K, M)


@dataclass
class DecodedSequence:
    """Decoded state sequence and segment table for one observation sequence."""

    states: np.ndarray
    segments: list[dict]
    score: float


@dataclass
class GroupFitResult:
    """Model fit output for one group."""

    group_id: str
    group_key: tuple
    params: HSMMParameters
    decoded: list[DecodedSequence]
    final_score: float
    converged: bool
    n_iterations: int


def _load_pickle(path):
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_fixation_vector(
    obj,
    fixation_label: str,
) -> Optional[np.ndarray]:
    """Extract a fixation vector from supported object layouts."""
    if isinstance(obj, FixationBinaryVectorsData):
        vectors = obj.vectors
    elif isinstance(obj, dict) and "vectors" in obj:
        vectors = obj["vectors"]
    elif isinstance(obj, dict):
        vectors = obj
    else:
        return None

    if not vectors or fixation_label not in vectors:
        return None

    vec = np.asarray(vectors[fixation_label])
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    return vec


def _extract_monkey_name(obj) -> Optional[str]:
    """Extract monkey name from supported object layouts."""
    if isinstance(obj, FixationBinaryVectorsData):
        return obj.context.monkey_name
    if isinstance(obj, dict):
        context = obj.get("context")
        if context is not None:
            if hasattr(context, "monkey_name"):
                return getattr(context, "monkey_name")
            if isinstance(context, dict) and "monkey_name" in context:
                return context.get("monkey_name")
        if "monkey_name" in obj:
            return obj.get("monkey_name")
    return None


def _normalize_optional_filter(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    """Normalize optional filters to sets."""
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(v) for v in values}


def _index_agent_paths(cfg: dict, modality: str) -> tuple[dict, dict]:
    """Index m1/m2 paths by (date, session)."""
    index_df = index_processed_dataset(cfg, modality)
    rows = index_df.to_dict(orient="records")

    m1_paths: dict[tuple[str, str], object] = {}
    m2_paths: dict[tuple[str, str], object] = {}
    for row in rows:
        key = (row["date"], row["session"])
        if row.get("agent") == "m1":
            m1_paths[key] = row["path"]
        elif row.get("agent") == "m2":
            m2_paths[key] = row["path"]

    return m1_paths, m2_paths


def _to_bool(vec: np.ndarray) -> np.ndarray:
    """Convert vector to 1D bool array."""
    out = np.asarray(vec).astype(bool, copy=False)
    if out.ndim != 1:
        out = out.reshape(-1)
    return out


def _build_joint_observations(m1_bool: np.ndarray, m2_bool: np.ndarray) -> np.ndarray:
    """Map (m1, m2) binary face-fix pairs to symbols 0..3."""
    return m1_bool.astype(np.uint8) + (m2_bool.astype(np.uint8) << 1)


def _build_session_sequences(
    cfg: dict,
    settings: FaceFixationHSMMSettings,
) -> list[SessionSequence]:
    """Load per-session joint observation sequences from fixation binary vectors."""
    m1_paths, m2_paths = _index_agent_paths(cfg, settings.input_modality)
    shared_keys = sorted(set(m1_paths).intersection(m2_paths))
    if not shared_keys:
        raise RuntimeError(
            "No m1/m2 overlap found for input modality "
            f"'{settings.input_modality}'."
        )

    date_filter = _normalize_optional_filter(settings.dates)
    session_filter = _normalize_optional_filter(settings.sessions)

    sequences: list[SessionSequence] = []
    for date, session in shared_keys:
        if date_filter is not None and date not in date_filter:
            continue
        if session_filter is not None and session not in session_filter:
            continue

        m1_obj = _load_pickle(m1_paths[(date, session)])
        m2_obj = _load_pickle(m2_paths[(date, session)])
        m1_vec = _extract_fixation_vector(m1_obj, settings.fixation_label)
        m2_vec = _extract_fixation_vector(m2_obj, settings.fixation_label)
        if m1_vec is None or m2_vec is None:
            continue

        m1_bool = _to_bool(m1_vec)
        m2_bool = _to_bool(m2_vec)
        if m1_bool.size == 0 or m2_bool.size == 0:
            continue

        if m1_bool.size != m2_bool.size:
            n = int(min(m1_bool.size, m2_bool.size))
            if n <= 0:
                continue
            m1_bool = m1_bool[:n]
            m2_bool = m2_bool[:n]

        observations = _build_joint_observations(m1_bool, m2_bool).astype(np.int64)
        sequences.append(
            SessionSequence(
                date=date,
                session=session,
                monkey_name_m1=_extract_monkey_name(m1_obj),
                monkey_name_m2=_extract_monkey_name(m2_obj),
                observations=observations,
            )
        )

    if not sequences:
        raise RuntimeError(
            "No valid sequences were loaded after filtering and vector checks."
        )

    if settings.test_single:
        return [sequences[0]]
    return sequences


def _group_key_for_sequence(seq: SessionSequence, grouping: str) -> tuple:
    """Build grouping key for one session sequence."""
    if grouping == "session":
        return ("session", seq.date, seq.session)
    if grouping == "day":
        return ("day", seq.date)
    if grouping == "pair":
        return (
            "pair",
            seq.monkey_name_m1 or "unknown_m1",
            seq.monkey_name_m2 or "unknown_m2",
        )
    if grouping == "global":
        return ("global", "all_sessions")
    raise ValueError(
        f"Unsupported grouping '{grouping}'. Expected one of: session, day, pair, global."
    )


def _format_group_id(group_key: tuple) -> str:
    """Render a stable string ID from a grouping key."""
    if group_key[0] == "session":
        _, date, session = group_key
        return f"date={date}|session={session}"
    if group_key[0] == "day":
        _, date = group_key
        return f"date={date}"
    if group_key[0] == "pair":
        _, m1_name, m2_name = group_key
        return f"pair={m1_name}__{m2_name}"
    return "all_sessions"


def _build_groups(
    sequences: list[SessionSequence],
    grouping: str,
) -> list[tuple[tuple, list[SessionSequence]]]:
    """Group session sequences for model fitting."""
    groups: dict[tuple, list[SessionSequence]] = {}
    for seq in sequences:
        key = _group_key_for_sequence(seq, grouping)
        groups.setdefault(key, []).append(seq)
    return [(key, groups[key]) for key in sorted(groups)]


def _normalize_prob_vector(values: np.ndarray) -> np.ndarray:
    """Normalize vector to sum to one with stability handling."""
    vals = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(vals.sum())
    if total <= 0:
        return np.full(vals.shape, 1.0 / float(vals.size), dtype=float)
    return vals / total


def _normalize_prob_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of a matrix to sum to one."""
    mat = np.clip(np.asarray(matrix, dtype=float), 0.0, None)
    out = np.zeros_like(mat, dtype=float)
    for row_idx in range(mat.shape[0]):
        row = mat[row_idx]
        total = float(row.sum())
        if total <= 0:
            out[row_idx] = np.full(row.shape, 1.0 / float(row.size), dtype=float)
        else:
            out[row_idx] = row / total
    return out


def _duration_log_probs(duration_lambdas: np.ndarray, max_duration: int) -> np.ndarray:
    """Compute log duration PMFs (Poisson, truncated at max_duration)."""
    d = np.arange(1, max_duration + 1, dtype=float)
    n_states = int(duration_lambdas.size)
    log_probs = np.empty((n_states, max_duration), dtype=float)
    for state in range(n_states):
        lam = float(max(duration_lambdas[state], 1e-3))
        pmf = poisson.pmf(d, lam)
        tail_mass = max(0.0, 1.0 - float(poisson.cdf(max_duration, lam)))
        pmf[-1] += tail_mass
        pmf = np.clip(pmf, EPS, None)
        pmf = pmf / pmf.sum()
        log_probs[state] = np.log(pmf)
    return log_probs


def _initialize_parameters(
    sequences: list[SessionSequence],
    n_states: int,
    n_symbols: int,
    max_duration: int,
    allow_self_transitions: bool,
    rng: np.random.Generator,
) -> HSMMParameters:
    """Initialize HSMM parameters."""
    if n_states < 2:
        raise ValueError("n_hidden_states must be >= 2.")

    lengths = np.asarray([seq.observations.size for seq in sequences], dtype=float)
    mean_length = float(np.mean(lengths)) if lengths.size else float(max_duration)
    lambda_base = min(max_duration, max(5.0, mean_length / 10.0))
    lambdas = np.clip(
        lambda_base * (0.5 + rng.random(n_states)),
        1.0,
        float(max_duration),
    )
    if n_states == 2:
        lambdas[0] = max(2.0, min(float(max_duration), lambda_base * 0.6))
        lambdas[1] = max(lambdas[0] + 1.0, min(float(max_duration), lambda_base * 1.6))

    all_obs = np.concatenate([seq.observations for seq in sequences], axis=0)
    obs_hist = np.bincount(all_obs, minlength=n_symbols).astype(float)
    base_obs = _normalize_prob_vector(obs_hist + 1.0)

    if n_states == 2 and n_symbols == 4:
        template = np.array(
            [
                [0.80, 0.08, 0.08, 0.04],
                [0.08, 0.30, 0.30, 0.32],
            ],
            dtype=float,
        )
        random_emissions = np.vstack(
            [rng.dirichlet(np.ones(n_symbols)) for _ in range(n_states)]
        )
        emissions = 0.65 * template + 0.20 * random_emissions + 0.15 * base_obs[None, :]
    else:
        emissions = np.vstack(
            [rng.dirichlet(1.0 + base_obs * 5.0) for _ in range(n_states)]
        )
    emissions = _normalize_prob_rows(emissions)

    initial = np.full(n_states, 1.0 / float(n_states), dtype=float)
    transitions = np.full((n_states, n_states), 1.0, dtype=float)
    if not allow_self_transitions:
        np.fill_diagonal(transitions, 0.0)
    transitions = _normalize_prob_rows(transitions)

    return HSMMParameters(
        initial_probs=initial,
        transition_probs=transitions,
        duration_lambdas=lambdas,
        emission_probs=emissions,
    )


def _viterbi_decode_hsmm(
    observations: np.ndarray,
    params: HSMMParameters,
    *,
    max_duration: int,
    allow_self_transitions: bool,
) -> Optional[DecodedSequence]:
    """Decode a sequence with explicit-duration Viterbi for a Poisson-HSMM."""
    obs = np.asarray(observations, dtype=np.int64)
    if obs.ndim != 1 or obs.size == 0:
        return None

    n_states, n_symbols = params.emission_probs.shape
    if obs.min() < 0 or obs.max() >= n_symbols:
        return None

    t_max = int(obs.size)
    d_max = int(max(1, min(max_duration, t_max)))

    log_initial = np.log(np.clip(params.initial_probs, EPS, None))
    log_trans = np.log(np.clip(params.transition_probs, EPS, None))
    log_duration = _duration_log_probs(params.duration_lambdas, d_max)
    log_emissions = np.log(np.clip(params.emission_probs, EPS, None))

    obs_log = log_emissions[:, obs]  # (K, T)
    prefix = np.zeros((n_states, t_max + 1), dtype=float)
    prefix[:, 1:] = np.cumsum(obs_log, axis=1)

    delta = np.full((t_max, n_states), -np.inf, dtype=float)
    psi_prev_state = np.full((t_max, n_states), -1, dtype=np.int64)
    psi_duration = np.ones((t_max, n_states), dtype=np.int64)

    for t_idx in range(t_max):
        max_d = min(d_max, t_idx + 1)
        for state in range(n_states):
            best_score = -np.inf
            best_prev = -1
            best_d = 1

            for duration in range(1, max_d + 1):
                seg_start = t_idx + 1 - duration
                seg_log_em = prefix[state, t_idx + 1] - prefix[state, seg_start]
                seg_score = seg_log_em + log_duration[state, duration - 1]

                if seg_start == 0:
                    score = log_initial[state] + seg_score
                    prev_state = -1
                else:
                    prev_scores = delta[seg_start - 1, :] + log_trans[:, state]
                    if not allow_self_transitions:
                        prev_scores[state] = -np.inf
                    prev_state = int(np.argmax(prev_scores))
                    score = float(prev_scores[prev_state] + seg_score)

                if score > best_score:
                    best_score = score
                    best_prev = prev_state
                    best_d = duration

            delta[t_idx, state] = best_score
            psi_prev_state[t_idx, state] = best_prev
            psi_duration[t_idx, state] = best_d

    final_state = int(np.argmax(delta[t_max - 1, :]))
    final_score = float(delta[t_max - 1, final_state])
    if not np.isfinite(final_score):
        return None

    states = np.full(t_max, -1, dtype=np.int64)
    segments_rev: list[dict] = []
    t_idx = t_max - 1
    state = final_state
    seg_idx = 0
    while t_idx >= 0:
        duration = int(psi_duration[t_idx, state])
        seg_start = t_idx + 1 - duration
        if seg_start < 0:
            return None
        states[seg_start : t_idx + 1] = state
        segments_rev.append(
            {
                "segment_index": seg_idx,
                "state": int(state),
                "start": int(seg_start),
                "stop": int(t_idx),
                "duration": int(duration),
            }
        )
        prev_state = int(psi_prev_state[t_idx, state])
        t_idx = seg_start - 1
        state = prev_state
        seg_idx += 1
        if t_idx >= 0 and state < 0:
            return None

    if np.any(states < 0):
        return None

    segments = list(reversed(segments_rev))
    for idx, segment in enumerate(segments):
        segment["segment_index"] = idx

    return DecodedSequence(states=states, segments=segments, score=final_score)


def _reestimate_parameters(
    params: HSMMParameters,
    sequences: list[SessionSequence],
    decoded: list[DecodedSequence],
    *,
    transition_pseudocount: float,
    emission_pseudocount: float,
    max_duration: int,
    allow_self_transitions: bool,
) -> HSMMParameters:
    """Re-estimate HSMM parameters from Viterbi-decoded sequences."""
    n_states, n_symbols = params.emission_probs.shape

    init_counts = np.zeros(n_states, dtype=float)
    transition_counts = np.zeros((n_states, n_states), dtype=float)
    emission_counts = np.zeros((n_states, n_symbols), dtype=float)
    duration_sum = np.zeros(n_states, dtype=float)
    duration_count = np.zeros(n_states, dtype=float)

    for seq, dec in zip(sequences, decoded):
        if dec.states.size == 0:
            continue

        init_state = int(dec.states[0])
        init_counts[init_state] += 1.0

        obs = seq.observations
        st = dec.states
        for state in range(n_states):
            mask = st == state
            if not np.any(mask):
                continue
            state_obs = obs[mask]
            emission_counts[state] += np.bincount(state_obs, minlength=n_symbols)

        for segment in dec.segments:
            state = int(segment["state"])
            duration_sum[state] += float(segment["duration"])
            duration_count[state] += 1.0

        for left, right in zip(dec.segments[:-1], dec.segments[1:]):
            i = int(left["state"])
            j = int(right["state"])
            transition_counts[i, j] += 1.0

    initial = _normalize_prob_vector(init_counts + transition_pseudocount)

    transitions = transition_counts + transition_pseudocount
    if not allow_self_transitions:
        np.fill_diagonal(transitions, 0.0)
    for state in range(n_states):
        row = np.clip(transitions[state], 0.0, None)
        if not allow_self_transitions:
            row[state] = 0.0
        row_sum = float(row.sum())
        if row_sum <= 0:
            if allow_self_transitions:
                row = np.full(n_states, 1.0 / float(n_states), dtype=float)
            else:
                row = np.ones(n_states, dtype=float)
                row[state] = 0.0
                row = row / row.sum()
        else:
            row = row / row_sum
        transitions[state] = row

    emissions = emission_counts + emission_pseudocount
    emissions = _normalize_prob_rows(emissions)

    lambdas = params.duration_lambdas.copy()
    for state in range(n_states):
        if duration_count[state] > 0:
            lambdas[state] = duration_sum[state] / duration_count[state]
    lambdas = np.clip(lambdas, 1.0, float(max_duration))

    return HSMMParameters(
        initial_probs=initial,
        transition_probs=transitions,
        duration_lambdas=lambdas,
        emission_probs=emissions,
    )


def _fit_single_hsmm_init(
    sequences: list[SessionSequence],
    settings: FaceFixationHSMMSettings,
    *,
    seed: int,
    init_idx: int,
    n_init: int,
    progress_label: Optional[str] = None,
    show_inner_progress: bool = False,
) -> Optional[GroupFitResult]:
    """Fit one HSMM initialization and return its decoded result."""
    n_states = int(settings.n_hidden_states)
    n_symbols = len(OBSERVATION_LABELS)
    rng = np.random.default_rng(int(seed))
    params = _initialize_parameters(
        sequences=sequences,
        n_states=n_states,
        n_symbols=n_symbols,
        max_duration=settings.max_duration,
        allow_self_transitions=settings.allow_self_transitions,
        rng=rng,
    )

    prev_score: Optional[float] = None
    converged = False
    n_iterations = 0

    iter_values = range(1, int(settings.n_iter) + 1)
    iter_progress = None
    if show_inner_progress:
        iter_desc = (
            f"{progress_label} init {init_idx + 1}/{n_init}"
            if progress_label
            else f"HSMM init {init_idx + 1}/{n_init}"
        )
        iter_progress = tqdm(iter_values, desc=iter_desc, unit="iter", leave=False)
        iter_iter = iter_progress
    else:
        iter_iter = iter_values

    for iter_idx in iter_iter:
        decoded: list[DecodedSequence] = []
        total_score = 0.0
        failed = False
        for seq in sequences:
            dec = _viterbi_decode_hsmm(
                seq.observations,
                params,
                max_duration=settings.max_duration,
                allow_self_transitions=settings.allow_self_transitions,
            )
            if dec is None:
                failed = True
                break
            decoded.append(dec)
            total_score += float(dec.score)

        if failed or not decoded:
            break

        if (
            iter_progress is not None
            and (
                iter_idx == 1
                or iter_idx == int(settings.n_iter)
                or iter_idx % max(1, int(settings.inner_progress_every)) == 0
            )
        ):
            if prev_score is None:
                iter_progress.set_postfix(score=f"{total_score:.2f}", refresh=False)
            else:
                iter_progress.set_postfix(
                    score=f"{total_score:.2f}",
                    delta=f"{(total_score - prev_score):.4f}",
                    refresh=False,
                )

        new_params = _reestimate_parameters(
            params=params,
            sequences=sequences,
            decoded=decoded,
            transition_pseudocount=settings.transition_pseudocount,
            emission_pseudocount=settings.emission_pseudocount,
            max_duration=settings.max_duration,
            allow_self_transitions=settings.allow_self_transitions,
        )

        n_iterations = iter_idx
        if prev_score is not None and abs(total_score - prev_score) <= settings.tol:
            params = new_params
            converged = True
            break

        prev_score = total_score
        params = new_params

    if iter_progress is not None:
        iter_progress.close()

    decoded_final: list[DecodedSequence] = []
    final_score = 0.0
    failed_final = False
    for seq in sequences:
        dec = _viterbi_decode_hsmm(
            seq.observations,
            params,
            max_duration=settings.max_duration,
            allow_self_transitions=settings.allow_self_transitions,
        )
        if dec is None:
            failed_final = True
            break
        decoded_final.append(dec)
        final_score += float(dec.score)

    if failed_final or not decoded_final:
        return None

    return GroupFitResult(
        group_id="",
        group_key=(),
        params=params,
        decoded=decoded_final,
        final_score=final_score,
        converged=converged,
        n_iterations=n_iterations,
    )


def _fit_hsmm_init_worker(
    task: tuple[
        int,
        list[SessionSequence],
        FaceFixationHSMMSettings,
        int,
        int,
    ],
) -> tuple[int, Optional[GroupFitResult]]:
    """Worker wrapper for one HSMM initialization fit."""
    init_idx, sequences, settings, seed, n_init = task
    result = _fit_single_hsmm_init(
        sequences,
        settings,
        seed=seed,
        init_idx=init_idx,
        n_init=n_init,
        show_inner_progress=False,
    )
    return init_idx, result


def _fit_poisson_hsmm_viterbi(
    sequences: list[SessionSequence],
    settings: FaceFixationHSMMSettings,
    *,
    rng: np.random.Generator,
    progress_label: Optional[str] = None,
) -> GroupFitResult:
    """Fit a Poisson-HSMM with Viterbi re-estimation to grouped sequences."""
    n_init = max(1, int(settings.n_init))
    best_result: Optional[GroupFitResult] = None
    init_seeds = [
        int(rng.integers(0, np.iinfo(np.int32).max))
        for _ in range(n_init)
    ]

    use_parallel = bool(settings.parallelize_across_iterations) and n_init > 1
    if use_parallel:
        n_proc = get_n_processes(max_procs=max(1, int(settings.max_parallel_workers)))
        n_proc = min(n_proc, n_init)
        if n_proc > 1:
            tasks = [
                (init_idx, sequences, settings, init_seeds[init_idx], n_init)
                for init_idx in range(n_init)
            ]
            ordered_results: list[Optional[GroupFitResult]] = [None] * n_init
            with Pool(processes=n_proc) as pool:
                iterator = pool.imap_unordered(_fit_hsmm_init_worker, tasks)
                if settings.show_inner_progress:
                    desc = (
                        f"{progress_label} inits ({n_proc} workers)"
                        if progress_label
                        else f"HSMM inits ({n_proc} workers)"
                    )
                    iterator = tqdm(iterator, total=n_init, desc=desc, unit="init", leave=False)
                for init_idx, result in iterator:
                    ordered_results[init_idx] = result

            for result in ordered_results:
                if result is None:
                    continue
                if best_result is None or result.final_score > best_result.final_score:
                    best_result = result
            if best_result is not None:
                return best_result

        use_parallel = False

    init_iter = range(n_init)
    if settings.show_inner_progress and n_init > 1:
        init_desc = f"{progress_label} inits" if progress_label else "HSMM inits"
        init_iter = tqdm(init_iter, desc=init_desc, unit="init", leave=False)
    for init_idx in init_iter:
        candidate = _fit_single_hsmm_init(
            sequences,
            settings,
            seed=init_seeds[init_idx],
            init_idx=init_idx,
            n_init=n_init,
            progress_label=progress_label,
            show_inner_progress=settings.show_inner_progress,
        )
        if candidate is None:
            continue
        if best_result is None or candidate.final_score > best_result.final_score:
            best_result = candidate

    if best_result is None:
        raise RuntimeError("HSMM fitting failed for at least one group.")
    return best_result


def _segment_observation_counts(
    observations: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    """Count observation symbols inside one segment."""
    seg = observations[start : stop + 1]
    return np.bincount(seg, minlength=len(OBSERVATION_LABELS))


def _flatten_group_results(
    settings: FaceFixationHSMMSettings,
    grouped_sequences: list[tuple[tuple, list[SessionSequence]]],
    group_results: list[GroupFitResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Build summary tables and serialized fit payload."""
    group_rows: list[dict] = []
    session_rows: list[dict] = []
    segment_rows: list[dict] = []
    serialized_fits: list[dict] = []

    for (group_key, seqs), result in zip(grouped_sequences, group_results):
        result.group_key = group_key
        result.group_id = _format_group_id(group_key)
        params = result.params
        n_states = int(params.initial_probs.size)

        total_samples = int(sum(seq.observations.size for seq in seqs))
        total_segments = int(sum(len(dec.segments) for dec in result.decoded))

        group_row = {
            "group_id": result.group_id,
            "grouping": settings.grouping,
            "n_sessions": int(len(seqs)),
            "total_samples": total_samples,
            "total_segments": total_segments,
            "viterbi_score": float(result.final_score),
            "converged": bool(result.converged),
            "n_iterations": int(result.n_iterations),
        }
        for state in range(n_states):
            group_row[f"initial_prob_state_{state}"] = float(params.initial_probs[state])
            group_row[f"duration_lambda_state_{state}"] = float(
                params.duration_lambdas[state]
            )
            for obs_idx, obs_label in OBSERVATION_LABELS.items():
                group_row[f"emission_state_{state}_{obs_label}"] = float(
                    params.emission_probs[state, obs_idx]
                )
            for next_state in range(n_states):
                group_row[f"transition_{state}_to_{next_state}"] = float(
                    params.transition_probs[state, next_state]
                )
        group_rows.append(group_row)

        serialized_fit = {
            "group_id": result.group_id,
            "group_key": group_key,
            "grouping": settings.grouping,
            "params": {
                "initial_probs": params.initial_probs.copy(),
                "transition_probs": params.transition_probs.copy(),
                "duration_lambdas": params.duration_lambdas.copy(),
                "emission_probs": params.emission_probs.copy(),
                "observation_labels": OBSERVATION_LABELS.copy(),
            },
            "viterbi_score": float(result.final_score),
            "converged": bool(result.converged),
            "n_iterations": int(result.n_iterations),
            "sequences": [],
        }

        for seq, dec in zip(seqs, result.decoded):
            n_samples = int(seq.observations.size)
            n_segments = int(len(dec.segments))
            session_row = {
                "group_id": result.group_id,
                "date": seq.date,
                "session": seq.session,
                "monkey_name_m1": seq.monkey_name_m1,
                "monkey_name_m2": seq.monkey_name_m2,
                "n_samples": n_samples,
                "n_segments": n_segments,
                "viterbi_score": float(dec.score),
            }
            for state in range(n_states):
                session_row[f"state_{state}_fraction"] = float(
                    np.mean(dec.states == state)
                )
            session_rows.append(session_row)

            for segment in dec.segments:
                start = int(segment["start"])
                stop = int(segment["stop"])
                counts = _segment_observation_counts(seq.observations, start, stop)
                segment_row = {
                    "group_id": result.group_id,
                    "date": seq.date,
                    "session": seq.session,
                    "segment_index": int(segment["segment_index"]),
                    "state": int(segment["state"]),
                    "start": start,
                    "stop": stop,
                    "duration": int(segment["duration"]),
                }
                for obs_idx, obs_label in OBSERVATION_LABELS.items():
                    segment_row[f"n_{obs_label}"] = int(counts[obs_idx])
                segment_rows.append(segment_row)

            serialized_fit["sequences"].append(
                {
                    "date": seq.date,
                    "session": seq.session,
                    "monkey_name_m1": seq.monkey_name_m1,
                    "monkey_name_m2": seq.monkey_name_m2,
                    "observations": seq.observations.copy(),
                    "states": dec.states.copy(),
                    "segments": [dict(seg) for seg in dec.segments],
                    "viterbi_score": float(dec.score),
                }
            )

        serialized_fits.append(serialized_fit)

    group_df = pd.DataFrame.from_records(group_rows)
    session_df = pd.DataFrame.from_records(session_rows)
    segments_df = pd.DataFrame.from_records(segment_rows)
    return group_df, session_df, segments_df, serialized_fits


def run_face_fixation_hsmm_analysis(
    settings: FaceFixationHSMMSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit grouped face-fixation HSMMs and persist summaries/artifacts."""
    if settings.n_hidden_states != 2:
        raise ValueError(
            "This workflow currently expects n_hidden_states=2 for the requested model."
        )

    cfg = load_dataset_config(settings.cfg_path)
    sequences = _build_session_sequences(cfg, settings)
    grouped_sequences = _build_groups(sequences, settings.grouping)
    if settings.test_single and grouped_sequences:
        grouped_sequences = [grouped_sequences[0]]

    seed_rng = np.random.default_rng(int(settings.seed))
    group_seeds = [
        int(seed_rng.integers(0, np.iinfo(np.int32).max))
        for _ in range(len(grouped_sequences))
    ]

    group_results: list[GroupFitResult] = []
    iterator = enumerate(grouped_sequences)
    if settings.show_progress:
        iterator = tqdm(
            iterator,
            total=len(grouped_sequences),
            desc="Fitting HSMM",
            unit="group",
        )
    for idx, (group_key, group_seqs) in iterator:
        rng = np.random.default_rng(int(group_seeds[idx]))
        group_result = _fit_poisson_hsmm_viterbi(
            group_seqs,
            settings,
            rng=rng,
            progress_label=_format_group_id(group_key),
        )
        group_result.group_key = group_key
        group_result.group_id = _format_group_id(group_key)
        group_results.append(group_result)

    group_df, session_df, segments_df, serialized_fits = _flatten_group_results(
        settings,
        grouped_sequences,
        group_results,
    )

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_df.to_csv(out_dir / settings.group_summary_filename, index=False)
    session_df.to_csv(out_dir / settings.session_summary_filename, index=False)
    segments_df.to_csv(out_dir / settings.segments_filename, index=False)
    with open(out_dir / settings.fits_filename, "wb") as f:
        pickle.dump(serialized_fits, f)

    return group_df, session_df, segments_df
