"""Compute fixation binary-vector cross-correlations within and across sessions."""

from __future__ import annotations

from multiprocessing import Pool
import hashlib
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.behav.analysis_primitives import (
    build_interactive_mask as _build_interactive_mask,
    extract_fixation_vector as _extract_fixation_vector,
    extract_monkey_name as _extract_monkey_name,
)
from dal_monte_2022_analysis.core.signal.cross_correlation import (
    assert_lag_axis_match as assert_lag_axis_match_shared,
    fft_cross_correlation,
    normalize_cross_correlation_sqrt_bin_count,
    summarize_cross_correlation,
)
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.runtime.io.processed_data import (
    index_agent_paths as _index_agent_paths,
    index_shared_paths as _index_shared_paths,
)
from dal_monte_2022_analysis.runtime.execution.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import (
    build_analysis_output_dir,
    build_fix_cross_correlation_output_filename,
    normalize_fix_cross_correlation_time_scope,
)


@dataclass
class FixCrossCorrelationSettings:
    """Configuration for fixation cross-correlation analysis."""

    cfg_path: str
    input_modality: str = "fixation_binary_vectors"
    fixation_label: str = "face"
    output_subdir: str = "crosscorr_outputs"
    within_filename: Optional[str] = None
    cross_filename: Optional[str] = None
    lags_filename: Optional[str] = None
    max_lag: Optional[int] = 60000
    time_scope: str = "whole"
    interactive_modality: str = "interactive_periods"
    interactive_state_label: str = "interactive"
    cross_pairs_max: Optional[int] = None
    cross_pairs_seed: int = 13
    cross_exclude_same_session: bool = True
    cross_exclude_same_date: bool = False
    parallelize_across_crosscorr_pairs: bool = False
    shuffle_output_filename: Optional[str] = None
    shuffle_pairs_subdir: str = "within_session_shuffle_pair_results"
    shuffle_n_shuffles: int = 1000
    shuffle_stringent: bool = True
    shuffle_seed: int = 13
    shuffle_parallelize_within_pair: bool = True
    shuffle_log_every: int = 100
    test_single: bool = False


def _load_fixation_vector(
    path,
    fixation_label: str,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Load a fixation vector and monkey name from a pickle path."""
    obj = load_pickle_path(path)
    return _extract_fixation_vector(obj, fixation_label), _extract_monkey_name(obj)


def _load_interactive_periods(path) -> Optional[pd.DataFrame]:
    """Load interactive periods from pickle (DataFrame expected)."""
    obj = load_pickle_path(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    return None


def _apply_time_scope_filter(
    vec_bool: np.ndarray,
    *,
    time_scope: str,
    interactive_periods_path,
    interactive_state_label: Optional[str],
) -> np.ndarray:
    """Filter fixation vector to whole/interactive/non-interactive scope."""
    scope = normalize_fix_cross_correlation_time_scope(time_scope)
    vec = np.asarray(vec_bool, dtype=bool)

    if scope == "whole":
        return vec

    if interactive_periods_path is None:
        raise RuntimeError(
            "Missing interactive-period path while running cross-correlation "
            f"time_scope='{scope}'."
        )
    periods_df = _load_interactive_periods(interactive_periods_path)
    if periods_df is None:
        raise RuntimeError(
            "Interactive-period file is missing/invalid while running cross-correlation "
            f"time_scope='{scope}': {interactive_periods_path}"
        )

    interactive_mask = _build_interactive_mask(
        periods_df,
        n_samples=int(vec.size),
        state_label=interactive_state_label,
    )
    if scope == "interactive":
        keep_mask = interactive_mask
    else:
        keep_mask = ~interactive_mask
    return vec & keep_mask


def _build_cross_pairs(
    settings: FixCrossCorrelationSettings,
    m1_keys: list[tuple[str, str]],
    m2_keys: list[tuple[str, str]],
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Generate cross-session pairs with optional exclusions and subsampling."""
    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for key1 in m1_keys:
        for key2 in m2_keys:
            if settings.cross_exclude_same_session and key1 == key2:
                continue
            if settings.cross_exclude_same_date and key1[0] == key2[0]:
                continue
            pairs.append((key1, key2))

    if settings.cross_pairs_max is not None and len(pairs) > settings.cross_pairs_max:
        rng = random.Random(settings.cross_pairs_seed)
        pairs = rng.sample(pairs, settings.cross_pairs_max)

    return pairs


def _print_test_single_debug(
    *,
    mode: str,
    settings: FixCrossCorrelationSettings,
    m1_id: tuple[str, str],
    m2_id: tuple[str, str],
    m1_len: int,
    m2_len: int,
    m1_fix_count: int,
    m2_fix_count: int,
    m1_fix_bin_count: int,
    m2_fix_bin_count: int,
    lags: np.ndarray,
    corr: np.ndarray,
) -> None:
    """Print selected task info and quick sanity checks for test-single mode."""
    max_allowed = 1.0 + 1e-9
    finite_ok = bool(np.isfinite(corr).all())
    in_range_ok = bool(((corr >= -1e-9) & (corr <= max_allowed)).all()) if corr.size else True
    has_zero_lag = bool(np.any(lags == 0))

    if corr.size:
        min_corr = float(np.min(corr))
        max_corr = float(np.max(corr))
        mean_corr = float(np.mean(corr))
    else:
        min_corr = float("nan")
        max_corr = float("nan")
        mean_corr = float("nan")

    print("\n[test-single] --------------------------------------------------")
    print(f"[test-single] mode: {mode}")
    print(
        "[test-single] selected pair: "
        f"m1(date={m1_id[0]}, session={m1_id[1]}) "
        f"vs m2(date={m2_id[0]}, session={m2_id[1]})"
    )
    print(
        "[test-single] vector sizes/counts: "
        f"m1_len={m1_len}, m2_len={m2_len}, "
        f"m1_fix_events={m1_fix_count}, m2_fix_events={m2_fix_count}, "
        f"m1_fix_bins={m1_fix_bin_count}, m2_fix_bins={m2_fix_bin_count}"
    )
    if lags.size:
        print(
            "[test-single] lag window: "
            f"{int(lags[0])}..{int(lags[-1])} (n_lags={lags.size}, max_lag={settings.max_lag})"
        )
    else:
        print("[test-single] lag window: empty")
    print(
        "[test-single] corr summary: "
        f"min={min_corr:.6f}, max={max_corr:.6f}, mean={mean_corr:.6f}"
    )
    print(
        "[test-single] sanity: "
        f"finite={finite_ok}, within_[0,1]={in_range_ok}, has_zero_lag={has_zero_lag}"
    )
    if corr.size:
        preview = np.array2string(corr[:10], precision=4, separator=", ")
        print(f"[test-single] corr preview (first 10): {preview}")
    print("[test-single] --------------------------------------------------\n")


def _fft_cross_correlation(
    x_bool: np.ndarray,
    y_bool: np.ndarray,
    *,
    max_lag: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full linear cross-correlation via FFT."""
    return fft_cross_correlation(x_bool, y_bool, max_lag=max_lag, round_to_int=True)


def _summarize_corr(
    lags: np.ndarray,
    corr: np.ndarray,
) -> dict:
    """Compute summary stats for one cross-correlation trace."""
    return summarize_cross_correlation(lags, corr)


def _normalize_corr(corr: np.ndarray, x_bin_count: int, y_bin_count: int) -> np.ndarray:
    """Normalize cross-correlation by sqrt(m1_fixation_bins * m2_fixation_bins)."""
    return normalize_cross_correlation_sqrt_bin_count(corr, x_bin_count, y_bin_count)


def _count_fixation_events(vec_bool: np.ndarray) -> int:
    """Count contiguous fixation events (islands of 1s) in a binary vector."""
    vec = np.asarray(vec_bool, dtype=bool)
    if vec.size == 0:
        return 0
    starts = vec & ~np.r_[False, vec[:-1]]
    return int(np.count_nonzero(starts))


def _extract_fixation_durations(vec_bool: np.ndarray) -> list[int]:
    """Extract contiguous fixation (1-run) durations from a binary vector."""
    vec = np.asarray(vec_bool, dtype=bool)
    if vec.size == 0:
        return []
    padded = np.r_[False, vec, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return (stops - starts).astype(int).tolist()


def _generate_uniform_partitions(total: int, n_parts: int, rng: np.random.Generator) -> list[int]:
    """Partition `total` into `n_parts` nonnegative integers."""
    if n_parts <= 0:
        return []
    if n_parts == 1:
        return [int(total)]
    if total <= 0:
        return [0] * n_parts
    probs = np.full(n_parts, 1.0 / n_parts, dtype=np.float64)
    return rng.multinomial(int(total), probs).astype(int).tolist()


def _interleave_segments(
    fix_durs: list[int],
    non_fix_durs: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Interleave non-fixation and fixation segments."""
    fix = list(fix_durs)
    non_fix = list(non_fix_durs)
    rng.shuffle(fix)
    rng.shuffle(non_fix)
    segments: list[tuple[int, int]] = []
    for idx, dur in enumerate(fix):
        segments.append((int(non_fix[idx]), 0))
        segments.append((int(dur), 1))
    segments.append((int(non_fix[-1]), 0))
    return segments


def _construct_shuffled_vector(segments: list[tuple[int, int]], run_length: int) -> np.ndarray:
    """Construct a shuffled binary vector from duration/value segments."""
    vec = np.zeros(int(run_length), dtype=np.uint8)
    idx = 0
    for dur, val in segments:
        if idx >= run_length:
            break
        dur = max(0, int(dur))
        end = min(idx + dur, run_length)
        if val == 1 and end > idx:
            vec[idx:end] = 1
        idx += dur
    return vec


def _generate_single_shuffled_vector(
    fixation_durations: list[int],
    non_fixation_total: int,
    run_length: int,
    stringent: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one shuffled vector preserving fixation load and (optionally) durations."""
    if not fixation_durations:
        return np.zeros(int(run_length), dtype=np.uint8)

    total_fix = int(sum(fixation_durations))
    run_length = int(run_length)
    if total_fix <= 0 or run_length <= 0:
        return np.zeros(run_length, dtype=np.uint8)

    if not stringent:
        vec = np.zeros(run_length, dtype=np.uint8)
        vec[: min(total_fix, run_length)] = 1
        rng.shuffle(vec)
        return vec

    non_fixation_durations = _generate_uniform_partitions(
        int(max(0, non_fixation_total)),
        len(fixation_durations) + 1,
        rng,
    )
    segments = _interleave_segments(fixation_durations, non_fixation_durations, rng)
    return _construct_shuffled_vector(segments, run_length)


def _stable_pair_seed(base_seed: int, key1: tuple[str, str], key2: tuple[str, str]) -> int:
    """Generate a deterministic per-pair seed."""
    token = f"{key1[0]}|{key1[1]}|{key2[0]}|{key2[1]}"
    digest = hashlib.sha1(token.encode("utf-8")).digest()[:8]
    offset = int.from_bytes(digest, byteorder="little", signed=False)
    return int((int(base_seed) + offset) % (2**32 - 1))


def _single_shuffle_corr_worker(
    args: tuple[
        list[int],
        int,
        int,
        list[int],
        int,
        int,
        bool,
        Optional[int],
        int,
    ],
) -> tuple[np.ndarray, np.ndarray]:
    """Worker for one shuffled cross-correlation draw."""
    (
        m1_fix_durs,
        m1_non_fix_total,
        m1_len,
        m2_fix_durs,
        m2_non_fix_total,
        m2_len,
        stringent,
        max_lag,
        seed,
    ) = args

    rng = np.random.default_rng(int(seed))
    m1_shuf = _generate_single_shuffled_vector(
        fixation_durations=m1_fix_durs,
        non_fixation_total=m1_non_fix_total,
        run_length=m1_len,
        stringent=stringent,
        rng=rng,
    ).astype(bool)
    m2_shuf = _generate_single_shuffled_vector(
        fixation_durations=m2_fix_durs,
        non_fixation_total=m2_non_fix_total,
        run_length=m2_len,
        stringent=stringent,
        rng=rng,
    ).astype(bool)

    lags, corr = _fft_cross_correlation(m1_shuf, m2_shuf, max_lag=max_lag)
    corr_norm = _normalize_corr(
        corr,
        x_bin_count=int(np.count_nonzero(m1_shuf)),
        y_bin_count=int(np.count_nonzero(m2_shuf)),
    )
    return lags, corr_norm


def _compute_shuffled_pair_summary(
    *,
    m1_vec: np.ndarray,
    m2_vec: np.ndarray,
    max_lag: Optional[int],
    n_shuffles: int,
    stringent: bool,
    base_seed: int,
    parallelize_within_pair: bool,
    log_prefix: Optional[str] = None,
    log_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute mean/std over shuffled cross-correlation draws for one pair."""
    m1_fix_durs = _extract_fixation_durations(m1_vec)
    m2_fix_durs = _extract_fixation_durations(m2_vec)
    m1_len = int(m1_vec.size)
    m2_len = int(m2_vec.size)
    m1_non_fix = int(max(0, m1_len - int(np.sum(m1_fix_durs))))
    m2_non_fix = int(max(0, m2_len - int(np.sum(m2_fix_durs))))

    if n_shuffles <= 0:
        lags, _ = _fft_cross_correlation(m1_vec, m2_vec, max_lag=max_lag)
        empty = np.full(lags.size, np.nan, dtype=np.float32)
        return lags, empty, empty, 0

    tasks = [
        (
            m1_fix_durs,
            m1_non_fix,
            m1_len,
            m2_fix_durs,
            m2_non_fix,
            m2_len,
            bool(stringent),
            max_lag,
            int((int(base_seed) + i) % (2**32 - 1)),
        )
        for i in range(int(n_shuffles))
    ]

    use_parallel = bool(parallelize_within_pair and n_shuffles > 1)
    lag_axis: Optional[np.ndarray] = None
    sum_corr: Optional[np.ndarray] = None
    sum_sq_corr: Optional[np.ndarray] = None
    n_done = 0

    if log_prefix is not None:
        mode = "parallel" if use_parallel else "serial"
        print(
            f"{log_prefix} shuffle start: n_shuffles={int(n_shuffles)} "
            f"mode={mode} stringent={bool(stringent)}"
        )

    def _maybe_log_progress() -> None:
        if log_prefix is None:
            return
        if n_done == 0:
            return
        step = int(log_every) if int(log_every) > 0 else max(1, int(n_shuffles) // 10)
        if (n_done % step == 0) or (n_done == int(n_shuffles)):
            pct = 100.0 * float(n_done) / float(max(1, int(n_shuffles)))
            print(f"{log_prefix} shuffle progress: {n_done}/{int(n_shuffles)} ({pct:.1f}%)")

    if use_parallel:
        n_proc = get_n_processes(max_procs=min(32, int(n_shuffles)))
        with Pool(processes=n_proc) as pool:
            iterator = pool.imap(_single_shuffle_corr_worker, tasks)
            for lags, corr in iterator:
                if lag_axis is None:
                    lag_axis = lags
                    sum_corr = np.zeros(corr.size, dtype=np.float64)
                    sum_sq_corr = np.zeros(corr.size, dtype=np.float64)
                else:
                    _assert_lags_match(lag_axis, lags)
                sum_corr += corr
                sum_sq_corr += corr * corr
                n_done += 1
                _maybe_log_progress()
    else:
        for task in tasks:
            lags, corr = _single_shuffle_corr_worker(task)
            if lag_axis is None:
                lag_axis = lags
                sum_corr = np.zeros(corr.size, dtype=np.float64)
                sum_sq_corr = np.zeros(corr.size, dtype=np.float64)
            else:
                _assert_lags_match(lag_axis, lags)
            sum_corr += corr
            sum_sq_corr += corr * corr
            n_done += 1
            _maybe_log_progress()

    if lag_axis is None or sum_corr is None or sum_sq_corr is None or n_done == 0:
        lags, _ = _fft_cross_correlation(m1_vec, m2_vec, max_lag=max_lag)
        empty = np.full(lags.size, np.nan, dtype=np.float32)
        if log_prefix is not None:
            print(f"{log_prefix} shuffle done: no valid shuffles completed")
        return lags, empty, empty, 0

    mean = sum_corr / float(n_done)
    if n_done > 1:
        var = (sum_sq_corr - (sum_corr * sum_corr) / float(n_done)) / float(n_done - 1)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
    else:
        std = np.zeros_like(mean)

    if log_prefix is not None:
        print(
            f"{log_prefix} shuffle done: completed={n_done} "
            f"mean[min,max]=({float(np.min(mean)):.6f}, {float(np.max(mean)):.6f}) "
            f"std[min,max]=({float(np.min(std)):.6f}, {float(np.max(std)):.6f})"
        )

    return lag_axis, mean.astype(np.float32), std.astype(np.float32), int(n_done)


def _build_pair_result(
    fixation_label: str,
    max_lag: Optional[int],
    key1: tuple[str, str],
    key2: tuple[str, str],
    m1_path,
    m2_path,
    *,
    time_scope: str = "whole",
    interactive_state_label: Optional[str] = "interactive",
    interactive_periods_path_key1=None,
    interactive_periods_path_key2=None,
) -> Optional[dict]:
    """Build one pair result, including normalized cross-correlation and metadata."""
    m1_vec, m1_name = _load_fixation_vector(m1_path, fixation_label)
    if m1_vec is None or m1_vec.size == 0:
        return None
    m2_vec, m2_name = _load_fixation_vector(m2_path, fixation_label)
    if m2_vec is None or m2_vec.size == 0:
        return None

    m1_vec = _apply_time_scope_filter(
        m1_vec,
        time_scope=time_scope,
        interactive_periods_path=interactive_periods_path_key1,
        interactive_state_label=interactive_state_label,
    )
    m2_vec = _apply_time_scope_filter(
        m2_vec,
        time_scope=time_scope,
        interactive_periods_path=interactive_periods_path_key2,
        interactive_state_label=interactive_state_label,
    )

    lags, corr = _fft_cross_correlation(m1_vec, m2_vec, max_lag=max_lag)
    m1_fix_bin_count = int(np.count_nonzero(m1_vec))
    m2_fix_bin_count = int(np.count_nonzero(m2_vec))
    m1_fix_event_count = _count_fixation_events(m1_vec)
    m2_fix_event_count = _count_fixation_events(m2_vec)
    corr_normalized = _normalize_corr(corr, m1_fix_bin_count, m2_fix_bin_count)

    result = {
        "key1": key1,
        "key2": key2,
        "lags": lags,
        "cross_correlation": corr_normalized,
        "n_samples_m1": int(m1_vec.size),
        "n_samples_m2": int(m2_vec.size),
        "m1_fixation_count": m1_fix_event_count,
        "m2_fixation_count": m2_fix_event_count,
        "m1_fixation_bin_count": m1_fix_bin_count,
        "m2_fixation_bin_count": m2_fix_bin_count,
        "monkey_name_m1": m1_name,
        "monkey_name_m2": m2_name,
    }
    result.update(_summarize_corr(lags, corr_normalized))
    return result


def _build_pair_result_worker(
    args: tuple[
        str,
        Optional[int],
        tuple[str, str],
        tuple[str, str],
        object,
        object,
        str,
        Optional[str],
        object,
        object,
    ],
) -> Optional[dict]:
    """Pool worker wrapper for computing one pair result."""
    (
        fixation_label,
        max_lag,
        key1,
        key2,
        m1_path,
        m2_path,
        time_scope,
        interactive_state_label,
        interactive_periods_path_key1,
        interactive_periods_path_key2,
    ) = args
    return _build_pair_result(
        fixation_label=fixation_label,
        max_lag=max_lag,
        key1=key1,
        key2=key2,
        m1_path=m1_path,
        m2_path=m2_path,
        time_scope=time_scope,
        interactive_state_label=interactive_state_label,
        interactive_periods_path_key1=interactive_periods_path_key1,
        interactive_periods_path_key2=interactive_periods_path_key2,
    )


def _assert_lags_match(expected_lags: np.ndarray, lags: np.ndarray) -> None:
    """Validate that all results share the same lag axis."""
    assert_lag_axis_match_shared(
        expected_lags,
        lags,
        message=(
            "Encountered mismatched lag vectors across pairs. "
            "Use a fixed max_lag (e.g., 60000) so lags are consistent."
        ),
    )


def _within_session_row_from_result(
    settings: FixCrossCorrelationSettings,
    result: dict,
) -> dict:
    """Construct one within-session output row from a pair-result payload."""
    key = result["key1"]
    return {
        "fixation_label": settings.fixation_label,
        "time_scope": normalize_fix_cross_correlation_time_scope(settings.time_scope),
        "date": key[0],
        "session": key[1],
        "n_samples_m1": result["n_samples_m1"],
        "n_samples_m2": result["n_samples_m2"],
        "m1_fixation_count": result["m1_fixation_count"],
        "m2_fixation_count": result["m2_fixation_count"],
        "m1_fixation_bin_count": result["m1_fixation_bin_count"],
        "m2_fixation_bin_count": result["m2_fixation_bin_count"],
        "monkey_name_m1": result["monkey_name_m1"],
        "monkey_name_m2": result["monkey_name_m2"],
        "cross_correlation": result["cross_correlation"],
        "n_lags": result["n_lags"],
        "zero_lag_correlation": result["zero_lag_correlation"],
        "peak_lag": result["peak_lag"],
        "peak_correlation": result["peak_correlation"],
    }


def _within_session_metadata_from_result(result: dict) -> dict:
    """Construct per-session metadata payload from one pair-result payload."""
    return {
        "monkey_name_m1": result["monkey_name_m1"],
        "monkey_name_m2": result["monkey_name_m2"],
        "m1_fixation_count": result["m1_fixation_count"],
        "m2_fixation_count": result["m2_fixation_count"],
        "m1_fixation_bin_count": result["m1_fixation_bin_count"],
        "m2_fixation_bin_count": result["m2_fixation_bin_count"],
    }


def _build_within_session_rows(
    settings: FixCrossCorrelationSettings,
    m1_paths: dict,
    m2_paths: dict,
    interactive_paths: Optional[dict] = None,
) -> tuple[list[dict], dict[tuple[str, str], dict], Optional[np.ndarray]]:
    """Build within-session rows and return per-session metadata + lag axis."""
    rows: list[dict] = []
    metadata_by_key: dict[tuple[str, str], dict] = {}
    lag_axis: Optional[np.ndarray] = None

    shared_keys = sorted(set(m1_paths).intersection(m2_paths))
    if settings.test_single and shared_keys:
        shared_keys = [random.choice(shared_keys)]

    scope = normalize_fix_cross_correlation_time_scope(settings.time_scope)
    tasks = []
    missing_interactive = 0
    for key in shared_keys:
        interactive_path = None
        if scope != "whole":
            if interactive_paths is None:
                raise RuntimeError(
                    "Interactive paths are required for non-whole cross-correlation scopes."
                )
            interactive_path = interactive_paths.get(key)
            if interactive_path is None:
                missing_interactive += 1
                continue

        tasks.append(
            (
                settings.fixation_label,
                settings.max_lag,
                key,
                key,
                m1_paths[key],
                m2_paths[key],
                scope,
                settings.interactive_state_label,
                interactive_path,
                interactive_path,
            )
        )

    if missing_interactive > 0:
        print(
            f"[fix-xcorr] skipped {missing_interactive} within-session keys due to "
            f"missing interactive periods for scope='{scope}'."
        )

    use_parallel = (
        settings.parallelize_across_crosscorr_pairs
        and not settings.test_single
        and len(tasks) > 1
    )

    if use_parallel:
        n_proc = get_n_processes(max_procs=32)
        with Pool(processes=n_proc) as pool:
            iterator = pool.imap(_build_pair_result_worker, tasks)
            for result in tqdm(
                iterator,
                total=len(tasks),
                desc=f"Within-session xcorr ({n_proc} workers)",
                unit="session",
            ):
                if result is None:
                    continue
                lags = result["lags"]
                if lag_axis is None:
                    lag_axis = lags
                else:
                    _assert_lags_match(lag_axis, lags)

                key = result["key1"]
                rows.append(_within_session_row_from_result(settings, result))
                metadata_by_key[key] = _within_session_metadata_from_result(result)
        return rows, metadata_by_key, lag_axis

    for task in tqdm(tasks, desc="Within-session xcorr", unit="session"):
        result = _build_pair_result_worker(task)
        if result is None:
            continue
        lags = result["lags"]
        if lag_axis is None:
            lag_axis = lags
        else:
            _assert_lags_match(lag_axis, lags)

        key = result["key1"]
        if settings.test_single:
            _print_test_single_debug(
                mode="within-session",
                settings=settings,
                m1_id=key,
                m2_id=key,
                m1_len=result["n_samples_m1"],
                m2_len=result["n_samples_m2"],
                m1_fix_count=result["m1_fixation_count"],
                m2_fix_count=result["m2_fixation_count"],
                m1_fix_bin_count=result["m1_fixation_bin_count"],
                m2_fix_bin_count=result["m2_fixation_bin_count"],
                lags=lags,
                corr=result["cross_correlation"],
            )

        rows.append(_within_session_row_from_result(settings, result))
        metadata_by_key[key] = _within_session_metadata_from_result(result)

    return rows, metadata_by_key, lag_axis


def _build_cross_session_control_rows(
    settings: FixCrossCorrelationSettings,
    m1_paths: dict,
    m2_paths: dict,
    within_keys: list[tuple[str, str]],
    metadata_by_key: dict[tuple[str, str], dict],
    lag_axis: Optional[np.ndarray],
    interactive_paths: Optional[dict] = None,
) -> tuple[list[dict], Optional[np.ndarray]]:
    """Build cross-session control rows aggregated per within-session key.

    For each within-session key K, aggregates all cross-session pairs where
    K appears as m1 side OR m2 side, and stores mean/std over those pairs.
    """
    m1_keys = sorted(m1_paths)
    m2_keys = sorted(m2_paths)
    within_key_set = set(within_keys)
    pairs = _build_cross_pairs(settings, m1_keys, m2_keys)

    if settings.test_single and pairs:
        pairs = [random.choice(pairs)]

    scope = normalize_fix_cross_correlation_time_scope(settings.time_scope)
    tasks = []
    missing_interactive = 0
    for key1, key2 in pairs:
        interactive_path_key1 = None
        interactive_path_key2 = None
        if scope != "whole":
            if interactive_paths is None:
                raise RuntimeError(
                    "Interactive paths are required for non-whole cross-correlation scopes."
                )
            interactive_path_key1 = interactive_paths.get(key1)
            interactive_path_key2 = interactive_paths.get(key2)
            if interactive_path_key1 is None or interactive_path_key2 is None:
                missing_interactive += 1
                continue
        tasks.append(
            (
                settings.fixation_label,
                settings.max_lag,
                key1,
                key2,
                m1_paths[key1],
                m2_paths[key2],
                scope,
                settings.interactive_state_label,
                interactive_path_key1,
                interactive_path_key2,
            )
        )

    if missing_interactive > 0:
        print(
            f"[fix-xcorr] skipped {missing_interactive} cross-session pairs due to "
            f"missing interactive periods for scope='{scope}'."
        )

    accum: dict[tuple[str, str], dict] = {}
    accum_m1_source: dict[tuple[str, str], dict] = {}
    accum_m2_source: dict[tuple[str, str], dict] = {}

    def _update_accum(
        target_accum: dict[tuple[str, str], dict],
        key: tuple[str, str],
        corr: np.ndarray,
    ) -> None:
        if key not in target_accum:
            target_accum[key] = {
                "sum": np.zeros(corr.size, dtype=np.float64),
                "sum_sq": np.zeros(corr.size, dtype=np.float64),
                "n_pairs": 0,
            }
        target_accum[key]["sum"] += corr
        target_accum[key]["sum_sq"] += corr * corr
        target_accum[key]["n_pairs"] += 1

    def _stats_from_accum(
        target_accum: dict[tuple[str, str], dict],
        key: tuple[str, str],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        stats = target_accum.get(key)
        if stats is None or stats["n_pairs"] == 0:
            return (
                np.full(lag_axis.size, np.nan, dtype=np.float32),
                np.full(lag_axis.size, np.nan, dtype=np.float32),
                0,
            )

        n_pairs = int(stats["n_pairs"])
        mean = stats["sum"] / float(n_pairs)
        if n_pairs > 1:
            var = (stats["sum_sq"] - (stats["sum"] * stats["sum"]) / float(n_pairs)) / float(
                n_pairs - 1
            )
            var = np.maximum(var, 0.0)
            std = np.sqrt(var)
        else:
            std = np.zeros_like(mean)
        return mean.astype(np.float32), std.astype(np.float32), n_pairs

    use_parallel = (
        settings.parallelize_across_crosscorr_pairs
        and not settings.test_single
        and len(tasks) > 1
    )

    if use_parallel:
        n_proc = get_n_processes(max_procs=32)
        with Pool(processes=n_proc) as pool:
            iterator = pool.imap(_build_pair_result_worker, tasks)
            for result in tqdm(
                iterator,
                total=len(tasks),
                desc=f"Cross-session xcorr ({n_proc} workers)",
                unit="pair",
            ):
                if result is None:
                    continue
                lags = result["lags"]
                if lag_axis is None:
                    lag_axis = lags
                else:
                    _assert_lags_match(lag_axis, lags)

                key1 = result["key1"]
                key2 = result["key2"]
                corr = result["cross_correlation"]
                if key1 in within_key_set:
                    _update_accum(accum, key1, corr)
                    _update_accum(accum_m1_source, key1, corr)
                if key2 in within_key_set and key2 != key1:
                    _update_accum(accum, key2, corr)
                    # key2 contributes as m2 side; flip lag sign so positive means m2 leads.
                    _update_accum(accum_m2_source, key2, corr[::-1])
    else:
        for task in tqdm(tasks, desc="Cross-session xcorr", unit="pair"):
            result = _build_pair_result_worker(task)
            if result is None:
                continue
            lags = result["lags"]
            if lag_axis is None:
                lag_axis = lags
            else:
                _assert_lags_match(lag_axis, lags)

            key1 = result["key1"]
            key2 = result["key2"]
            corr = result["cross_correlation"]

            if settings.test_single:
                _print_test_single_debug(
                    mode="cross-session",
                    settings=settings,
                    m1_id=key1,
                    m2_id=key2,
                    m1_len=result["n_samples_m1"],
                    m2_len=result["n_samples_m2"],
                    m1_fix_count=result["m1_fixation_count"],
                    m2_fix_count=result["m2_fixation_count"],
                    m1_fix_bin_count=result["m1_fixation_bin_count"],
                    m2_fix_bin_count=result["m2_fixation_bin_count"],
                    lags=lags,
                    corr=corr,
                )

            if key1 in within_key_set:
                _update_accum(accum, key1, corr)
                _update_accum(accum_m1_source, key1, corr)
            if key2 in within_key_set and key2 != key1:
                _update_accum(accum, key2, corr)
                # key2 contributes as m2 side; flip lag sign so positive means m2 leads.
                _update_accum(accum_m2_source, key2, corr[::-1])

    rows: list[dict] = []
    if lag_axis is None:
        return rows, lag_axis

    for key in within_keys:
        meta = metadata_by_key.get(key, {})
        mean_corr, std_corr, n_pairs = _stats_from_accum(accum, key)
        mean_corr_m1_source, std_corr_m1_source, n_pairs_m1_source = _stats_from_accum(
            accum_m1_source,
            key,
        )
        mean_corr_m2_source, std_corr_m2_source, n_pairs_m2_source = _stats_from_accum(
            accum_m2_source,
            key,
        )

        rows.append({
            "fixation_label": settings.fixation_label,
            "time_scope": normalize_fix_cross_correlation_time_scope(settings.time_scope),
            "date": key[0],
            "session": key[1],
            "monkey_name_m1": meta.get("monkey_name_m1"),
            "monkey_name_m2": meta.get("monkey_name_m2"),
            "m1_fixation_count": meta.get("m1_fixation_count"),
            "m2_fixation_count": meta.get("m2_fixation_count"),
            "m1_fixation_bin_count": meta.get("m1_fixation_bin_count"),
            "m2_fixation_bin_count": meta.get("m2_fixation_bin_count"),
            "n_pairs": n_pairs,
            "cross_correlation_mean": mean_corr,
            "cross_correlation_std": std_corr,
            "n_pairs_m1_source": n_pairs_m1_source,
            "n_pairs_m2_source": n_pairs_m2_source,
            "cross_correlation_mean_m1_source": mean_corr_m1_source,
            "cross_correlation_std_m1_source": std_corr_m1_source,
            "cross_correlation_mean_m2_source": mean_corr_m2_source,
            "cross_correlation_std_m2_source": std_corr_m2_source,
        })

    return rows, lag_axis


def build_within_session_pair_tasks(
    settings: FixCrossCorrelationSettings,
) -> list[tuple[str, str]]:
    """Return within-session pair keys as (date, session) tuples."""
    cfg = load_config(settings.cfg_path)
    m1_paths, m2_paths = _index_agent_paths(cfg, settings.input_modality)
    shared_keys = sorted(set(m1_paths).intersection(m2_paths))

    scope = normalize_fix_cross_correlation_time_scope(settings.time_scope)
    if scope != "whole":
        interactive_paths = _index_shared_paths(cfg, settings.interactive_modality)
        shared_keys = [key for key in shared_keys if key in interactive_paths]

    if settings.test_single and shared_keys:
        shared_keys = [random.choice(shared_keys)]
    return shared_keys


def _shuffle_pair_output_dir(cfg: dict, settings: FixCrossCorrelationSettings) -> Path:
    """Return output directory for per-pair shuffled results."""
    scope = normalize_fix_cross_correlation_time_scope(settings.time_scope)
    return build_analysis_output_dir(
        cfg,
        f"{settings.output_subdir}/{settings.shuffle_pairs_subdir}/phase={scope}",
    )


def _shuffle_pair_result_path(
    pair_dir: Path,
    key: tuple[str, str],
) -> Path:
    """Return per-pair shuffled result path."""
    date, session = key
    return pair_dir / f"date={date}__session={session}.pkl"


def _build_within_session_shuffle_row(
    settings: FixCrossCorrelationSettings,
    key: tuple[str, str],
    m1_path,
    m2_path,
    *,
    interactive_periods_path=None,
) -> Optional[dict]:
    """Build shuffled-summary row for one within-session pair."""
    m1_vec, m1_name = _load_fixation_vector(m1_path, settings.fixation_label)
    if m1_vec is None or m1_vec.size == 0:
        return None
    m2_vec, m2_name = _load_fixation_vector(m2_path, settings.fixation_label)
    if m2_vec is None or m2_vec.size == 0:
        return None

    m1_vec = _apply_time_scope_filter(
        m1_vec,
        time_scope=settings.time_scope,
        interactive_periods_path=interactive_periods_path,
        interactive_state_label=settings.interactive_state_label,
    )
    m2_vec = _apply_time_scope_filter(
        m2_vec,
        time_scope=settings.time_scope,
        interactive_periods_path=interactive_periods_path,
        interactive_state_label=settings.interactive_state_label,
    )

    m1_fix_event_count = _count_fixation_events(m1_vec)
    m2_fix_event_count = _count_fixation_events(m2_vec)
    m1_fix_bin_count = int(np.count_nonzero(m1_vec))
    m2_fix_bin_count = int(np.count_nonzero(m2_vec))

    pair_seed = _stable_pair_seed(settings.shuffle_seed, key, key)
    log_prefix = f"[shuffle-pair {key[0]}-{key[1]}]"
    lags, mean_corr, std_corr, n_shuffles_done = _compute_shuffled_pair_summary(
        m1_vec=m1_vec,
        m2_vec=m2_vec,
        max_lag=settings.max_lag,
        n_shuffles=settings.shuffle_n_shuffles,
        stringent=settings.shuffle_stringent,
        base_seed=pair_seed,
        parallelize_within_pair=settings.shuffle_parallelize_within_pair,
        log_prefix=log_prefix,
        log_every=settings.shuffle_log_every,
    )

    return {
        "fixation_label": settings.fixation_label,
        "time_scope": normalize_fix_cross_correlation_time_scope(settings.time_scope),
        "date": key[0],
        "session": key[1],
        "monkey_name_m1": m1_name,
        "monkey_name_m2": m2_name,
        "n_samples_m1": int(m1_vec.size),
        "n_samples_m2": int(m2_vec.size),
        "m1_fixation_count": m1_fix_event_count,
        "m2_fixation_count": m2_fix_event_count,
        "m1_fixation_bin_count": m1_fix_bin_count,
        "m2_fixation_bin_count": m2_fix_bin_count,
        "shuffle_stringent": bool(settings.shuffle_stringent),
        "n_shuffles": int(n_shuffles_done),
        "cross_correlation_shuffle_mean": mean_corr,
        "cross_correlation_shuffle_std": std_corr,
        "lags": lags,
    }


def process_and_save_within_session_shuffle_pair(
    settings: FixCrossCorrelationSettings,
    date: str,
    session: str,
) -> Optional[Path]:
    """Compute and save shuffled-summary output for one within-session pair."""
    cfg = load_config(settings.cfg_path)
    m1_paths, m2_paths = _index_agent_paths(cfg, settings.input_modality)
    scope = normalize_fix_cross_correlation_time_scope(settings.time_scope)
    key = (str(date), str(session))
    if key not in m1_paths or key not in m2_paths:
        print(f"[shuffle-worker] missing within-session pair for date={date} session={session}")
        return None

    interactive_path = None
    if scope != "whole":
        interactive_paths = _index_shared_paths(cfg, settings.interactive_modality)
        interactive_path = interactive_paths.get(key)
        if interactive_path is None:
            print(
                f"[shuffle-worker] missing interactive periods for date={date} "
                f"session={session} scope={scope}"
            )
            return None

    print(
        f"[shuffle-worker] start pair date={date} session={session} "
        f"label={settings.fixation_label} n_shuffles={settings.shuffle_n_shuffles} "
        f"max_lag={settings.max_lag} scope={scope}"
    )

    row = _build_within_session_shuffle_row(
        settings=settings,
        key=key,
        m1_path=m1_paths[key],
        m2_path=m2_paths[key],
        interactive_periods_path=interactive_path,
    )
    if row is None:
        print(f"[shuffle-worker] no result for date={date} session={session}")
        return None

    out_dir = _shuffle_pair_output_dir(cfg, settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _shuffle_pair_result_path(out_dir, key)
    with open(out_path, "wb") as f:
        pickle.dump(row, f)
    mean_arr = np.asarray(row["cross_correlation_shuffle_mean"])
    std_arr = np.asarray(row["cross_correlation_shuffle_std"])
    lags_arr = np.asarray(row["lags"])
    preview_n = min(5, int(mean_arr.size))
    lag_preview = np.array2string(lags_arr[:preview_n], separator=", ")
    mean_preview = np.array2string(mean_arr[:preview_n], precision=4, separator=", ")
    std_preview = np.array2string(std_arr[:preview_n], precision=4, separator=", ")
    print(
        f"[shuffle-worker] wrote: {out_path} | "
        f"n_shuffles={int(row['n_shuffles'])} n_lags={int(mean_arr.size)} "
        f"mean[min,max]=({float(np.nanmin(mean_arr)):.6f}, {float(np.nanmax(mean_arr)):.6f}) "
        f"std[min,max]=({float(np.nanmin(std_arr)):.6f}, {float(np.nanmax(std_arr)):.6f}) "
        f"preview lags={lag_preview} mean={mean_preview} std={std_preview}"
    )
    return out_path


def collate_within_session_shuffle_results(
    settings: FixCrossCorrelationSettings,
) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Collate per-pair shuffled outputs into one analysis table."""
    cfg = load_config(settings.cfg_path)
    pair_dir = _shuffle_pair_output_dir(cfg, settings)
    if not pair_dir.exists():
        raise RuntimeError(f"No shuffle pair directory found: {pair_dir}")

    pair_paths = sorted(pair_dir.glob("date=*__session=*.pkl"))
    if not pair_paths:
        raise RuntimeError(f"No per-pair shuffle files found in: {pair_dir}")

    rows: list[dict] = []
    lag_axis: Optional[np.ndarray] = None

    for path in tqdm(pair_paths, desc="Collating shuffled pairs", unit="pair"):
        with open(path, "rb") as f:
            row = pickle.load(f)
        lags = np.asarray(row.get("lags"))
        if lag_axis is None:
            lag_axis = lags
        else:
            _assert_lags_match(lag_axis, lags)

        row = dict(row)
        row.pop("lags", None)
        rows.append(row)

    shuffle_df = pd.DataFrame.from_records(rows)
    if {"date", "session"}.issubset(shuffle_df.columns):
        shuffle_df = shuffle_df.sort_values(["date", "session"]).reset_index(drop=True)

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _resolve_shuffle_output_filename(settings)
    shuffle_df.to_pickle(out_path)

    if lag_axis is not None:
        lags_path = out_dir / _resolve_lags_filename(settings)
        with open(lags_path, "wb") as f:
            pickle.dump(lag_axis, f)

    print(f"[shuffle-collate] wrote: {out_path}")
    if lag_axis is not None:
        print(
            "[shuffle-collate] lags: "
            f"n={int(lag_axis.size)} start={int(lag_axis[0])} stop={int(lag_axis[-1])}"
        )

    print(f"[shuffle-collate] rows: {len(shuffle_df)}")
    if not shuffle_df.empty and "n_shuffles" in shuffle_df.columns:
        n_shuf = pd.to_numeric(shuffle_df["n_shuffles"], errors="coerce")
        if n_shuf.notna().any():
            print(
                "[shuffle-collate] n_shuffles stats: "
                f"min={int(n_shuf.min())} max={int(n_shuf.max())} mean={float(n_shuf.mean()):.2f}"
            )
    if not shuffle_df.empty and "cross_correlation_shuffle_mean" in shuffle_df.columns:
        lens = shuffle_df["cross_correlation_shuffle_mean"].map(
            lambda arr: int(np.asarray(arr).size) if arr is not None else 0
        )
        print(
            "[shuffle-collate] mean-array length: "
            f"min={int(lens.min())} max={int(lens.max())}"
        )
        finite_rows = shuffle_df["cross_correlation_shuffle_mean"].map(
            lambda arr: bool(np.isfinite(np.asarray(arr)).all()) if arr is not None else False
        )
        print(
            "[shuffle-collate] finite mean arrays: "
            f"{int(finite_rows.sum())}/{len(finite_rows)}"
        )
        sample = shuffle_df.iloc[0]
        sample_mean = np.asarray(sample["cross_correlation_shuffle_mean"])
        sample_std = np.asarray(sample["cross_correlation_shuffle_std"])
        n_preview = min(5, int(sample_mean.size))
        sample_mean_preview = np.array2string(
            sample_mean[:n_preview],
            precision=4,
            separator=", ",
        )
        sample_std_preview = np.array2string(
            sample_std[:n_preview],
            precision=4,
            separator=", ",
        )
        print(
            "[shuffle-collate] sample row preview: "
            f"date={sample.get('date')} session={sample.get('session')} "
            f"mean={sample_mean_preview} std={sample_std_preview}"
        )
    return shuffle_df, lag_axis


def _resolve_within_filename(settings: FixCrossCorrelationSettings) -> str:
    """Return within-session output filename."""
    if settings.within_filename:
        return settings.within_filename
    return build_fix_cross_correlation_output_filename(
        settings.fixation_label,
        "within",
        time_scope=settings.time_scope,
    )


def _resolve_cross_filename(settings: FixCrossCorrelationSettings) -> str:
    """Return cross-session output filename."""
    if settings.cross_filename:
        return settings.cross_filename
    return build_fix_cross_correlation_output_filename(
        settings.fixation_label,
        "cross",
        time_scope=settings.time_scope,
    )


def _resolve_shuffle_output_filename(settings: FixCrossCorrelationSettings) -> str:
    """Return shuffled-summary output filename."""
    if settings.shuffle_output_filename:
        return settings.shuffle_output_filename
    return build_fix_cross_correlation_output_filename(
        settings.fixation_label,
        "shuffle",
        time_scope=settings.time_scope,
    )


def _resolve_lags_filename(settings: FixCrossCorrelationSettings) -> str:
    """Return lag-axis output filename."""
    if settings.lags_filename:
        return settings.lags_filename
    return build_fix_cross_correlation_output_filename(
        settings.fixation_label,
        "lags",
        time_scope=settings.time_scope,
    )


def _print_output_sanity_summary(
    within_df: pd.DataFrame,
    cross_df: Optional[pd.DataFrame],
    lag_axis: Optional[np.ndarray],
) -> None:
    """Print compact sanity checks for saved outputs."""
    print("\n[fix-xcorr] Output sanity summary")
    print("[fix-xcorr] -----------------------------------------------")

    print(f"[fix-xcorr] within rows: {len(within_df)}")
    if not within_df.empty:
        if "cross_correlation" in within_df.columns:
            lengths = within_df["cross_correlation"].map(
                lambda arr: int(np.asarray(arr).size) if arr is not None else 0
            )
            print(
                "[fix-xcorr] within corr length: "
                f"min={int(lengths.min())}, max={int(lengths.max())}"
            )
        if "peak_correlation" in within_df.columns:
            peak_vals = pd.to_numeric(within_df["peak_correlation"], errors="coerce")
            if peak_vals.notna().any():
                print(
                    "[fix-xcorr] within peak corr: "
                    f"min={float(peak_vals.min()):.6f}, max={float(peak_vals.max()):.6f}, "
                    f"mean={float(peak_vals.mean()):.6f}"
                )

    if cross_df is None:
        print("[fix-xcorr] cross rows: not computed")
    else:
        print(f"[fix-xcorr] cross rows: {len(cross_df)}")
        if not cross_df.empty and "n_pairs" in cross_df.columns:
            n_pairs_vals = pd.to_numeric(cross_df["n_pairs"], errors="coerce")
            if n_pairs_vals.notna().any():
                print(
                    "[fix-xcorr] cross n_pairs: "
                    f"min={int(n_pairs_vals.min())}, max={int(n_pairs_vals.max())}, "
                    f"mean={float(n_pairs_vals.mean()):.2f}"
                )
        if not cross_df.empty and "cross_correlation_mean" in cross_df.columns:
            mean_lengths = cross_df["cross_correlation_mean"].map(
                lambda arr: int(np.asarray(arr).size) if arr is not None else 0
            )
            print(
                "[fix-xcorr] cross mean-corr length: "
                f"min={int(mean_lengths.min())}, max={int(mean_lengths.max())}"
            )

    if lag_axis is None:
        print("[fix-xcorr] lags: missing")
    else:
        n_lags = int(lag_axis.size)
        if n_lags == 0:
            print("[fix-xcorr] lags: empty")
        else:
            preview_left = np.array2string(lag_axis[:5], separator=", ")
            preview_right = np.array2string(lag_axis[-5:], separator=", ")
            print(
                "[fix-xcorr] lags: "
                f"n={n_lags}, start={int(lag_axis[0])}, stop={int(lag_axis[-1])}"
            )
            print(f"[fix-xcorr] lags preview: head={preview_left} tail={preview_right}")

    print("[fix-xcorr] -----------------------------------------------\n")


def run_fix_cross_correlation_analysis(
    settings: FixCrossCorrelationSettings,
    *,
    compute_cross: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run fixation cross-correlation analysis and persist outputs.

    Output files:
    - within filename (resolved by scope/label): per-session within-session vectors.
    - cross filename (resolved by scope/label): per-session cross-session controls.
    - lags filename (resolved by scope/label): shared lag axis.
    """
    cfg = load_config(settings.cfg_path)
    m1_paths, m2_paths = _index_agent_paths(cfg, settings.input_modality)
    scope = normalize_fix_cross_correlation_time_scope(settings.time_scope)

    if not m1_paths or not m2_paths:
        raise RuntimeError(
            "Missing fixation binary vectors for m1 or m2. "
            f"Found m1={len(m1_paths)} m2={len(m2_paths)}."
        )

    interactive_paths = None
    if scope != "whole":
        interactive_paths = _index_shared_paths(cfg, settings.interactive_modality)
        if not interactive_paths:
            raise RuntimeError(
                "No interactive-period files found for non-whole cross-correlation scope "
                f"'{scope}' in modality '{settings.interactive_modality}'."
            )

    within_rows, metadata_by_key, lag_axis = _build_within_session_rows(
        settings,
        m1_paths,
        m2_paths,
        interactive_paths=interactive_paths,
    )
    within_df = pd.DataFrame.from_records(within_rows)

    cross_df = None
    if compute_cross:
        within_keys = sorted(metadata_by_key)
        cross_rows, lag_axis = _build_cross_session_control_rows(
            settings=settings,
            m1_paths=m1_paths,
            m2_paths=m2_paths,
            within_keys=within_keys,
            metadata_by_key=metadata_by_key,
            lag_axis=lag_axis,
            interactive_paths=interactive_paths,
        )
        cross_df = pd.DataFrame.from_records(cross_rows)

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    within_path = out_dir / _resolve_within_filename(settings)
    within_df.to_pickle(within_path)

    if cross_df is not None:
        cross_path = out_dir / _resolve_cross_filename(settings)
        cross_df.to_pickle(cross_path)

    if lag_axis is not None:
        lags_path = out_dir / _resolve_lags_filename(settings)
        with open(lags_path, "wb") as f:
            pickle.dump(lag_axis, f)

    _print_output_sanity_summary(within_df, cross_df, lag_axis)

    return within_df, cross_df
