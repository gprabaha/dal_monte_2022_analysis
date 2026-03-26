"""Compute fixation probability stats within sessions and across sessions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import random
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.behav.analysis_primitives import (
    clip_period as _clip_period,
    extract_fixation_vector as _extract_fixation_vector,
    extract_monkey_name as _extract_monkey_name,
    filter_interactive_periods as _filter_interactive_periods,
    to_bool as _to_bool,
)
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.runtime.io.processed_data import (
    index_agent_paths as _index_agent_paths,
    index_shared_paths as _index_shared_paths,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


@dataclass
class FixationProbabilitySettings:
    """Configuration for fixation probability analysis."""
    cfg_path: str
    input_modality: str = "fixation_binary_vectors"
    fixation_label: str = "face"
    output_subdir: str = "fixation_probability"
    within_filename: str = "within_session_face_fixation_probability.csv"
    cross_filename: str = "cross_session_face_fixation_probability.csv"
    interactive_modality: str = "interactive_periods"
    interactive_state_label: str = "interactive"
    interactive_periods_filename: str = (
        "within_session_interactive_period_face_fixation_probability.csv"
    )
    interactive_concat_filename: str = (
        "within_session_interactive_concat_face_fixation_probability.csv"
    )
    decimal_precision: int = 50
    cross_pairs_max: Optional[int] = None
    cross_pairs_seed: int = 13
    cross_exclude_same_session: bool = True
    cross_exclude_same_date: bool = False
    test_single: bool = False


def _load_interactive_periods(path) -> Optional[pd.DataFrame]:
    """Load interactive periods from a pickle path."""
    obj = load_pickle_path(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    return None


def _decimal_ratio(numer: int, denom: int, precision: int) -> Optional[Decimal]:
    """Return numer/denom as a Decimal with specified precision."""
    if denom <= 0:
        return None
    with localcontext() as ctx:
        ctx.prec = precision
        return Decimal(numer) / Decimal(denom)


def _decimal_str(value: Optional[Decimal]) -> Optional[str]:
    """Render a Decimal as a fixed-point string (no exponent)."""
    if value is None:
        return None
    return format(value, "f")


def _load_fixation_vector(
    path,
    fixation_label: str,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Load a fixation vector and monkey name from a pickle path."""
    obj = load_pickle_path(path)
    return _extract_fixation_vector(obj, fixation_label), _extract_monkey_name(obj)


def _build_within_session_rows(
    settings: FixationProbabilitySettings,
    m1_paths: dict,
    m2_paths: dict,
) -> list[dict]:
    """Build within-session probability rows."""
    rows: list[dict] = []
    shared_keys = sorted(set(m1_paths).intersection(m2_paths))

    if settings.test_single and shared_keys:
        shared_keys = [shared_keys[0]]

    for key in tqdm(shared_keys, desc="Within-session fixation", unit="session"):
        date, session = key
        m1_vec, m1_name = _load_fixation_vector(m1_paths[key], settings.fixation_label)
        m2_vec, m2_name = _load_fixation_vector(m2_paths[key], settings.fixation_label)
        if m1_vec is None or m2_vec is None:
            continue

        m1_bool = _to_bool(m1_vec)
        m2_bool = _to_bool(m2_vec)
        if m1_bool.size == 0 or m2_bool.size == 0:
            continue
        if m1_bool.size != m2_bool.size:
            continue

        n_samples = int(m1_bool.size)
        m1_count = int(np.count_nonzero(m1_bool))
        m2_count = int(np.count_nonzero(m2_bool))
        joint_count = int(np.count_nonzero(m1_bool & m2_bool))

        p_m1 = _decimal_ratio(m1_count, n_samples, settings.decimal_precision)
        p_m2 = _decimal_ratio(m2_count, n_samples, settings.decimal_precision)
        p_joint = _decimal_ratio(joint_count, n_samples, settings.decimal_precision)
        p_product = None
        if p_m1 is not None and p_m2 is not None:
            with localcontext() as ctx:
                ctx.prec = settings.decimal_precision
                p_product = p_m1 * p_m2

        rows.append({
            "date": date,
            "session": session,
            "n_samples": n_samples,
            "monkey_name_m1": m1_name,
            "monkey_name_m2": m2_name,
            "m1_face_count": m1_count,
            "m2_face_count": m2_count,
            "joint_face_count": joint_count,
            "p_m1_decimal": _decimal_str(p_m1),
            "p_m2_decimal": _decimal_str(p_m2),
            "p_joint_decimal": _decimal_str(p_joint),
            "p_product_decimal": _decimal_str(p_product),
        })

    return rows


def _build_interactive_rows(
    settings: FixationProbabilitySettings,
    m1_paths: dict,
    m2_paths: dict,
    interactive_paths: dict,
) -> tuple[list[dict], list[dict]]:
    """Build per-period and concatenated interactive rows."""
    period_rows: list[dict] = []
    concat_rows: list[dict] = []

    shared_keys = sorted(
        set(m1_paths).intersection(m2_paths).intersection(interactive_paths)
    )

    if settings.test_single and shared_keys:
        shared_keys = [shared_keys[0]]

    for key in tqdm(shared_keys, desc="Interactive-period fixation", unit="session"):
        date, session = key
        m1_vec, m1_name = _load_fixation_vector(m1_paths[key], settings.fixation_label)
        m2_vec, m2_name = _load_fixation_vector(m2_paths[key], settings.fixation_label)
        if m1_vec is None or m2_vec is None:
            continue

        m1_bool = _to_bool(m1_vec)
        m2_bool = _to_bool(m2_vec)
        if m1_bool.size == 0 or m2_bool.size == 0:
            continue
        if m1_bool.size != m2_bool.size:
            continue

        periods_df = _load_interactive_periods(interactive_paths[key])
        periods_df = _filter_interactive_periods(
            periods_df,
            settings.interactive_state_label,
        )
        if periods_df.empty:
            continue

        total_samples = 0
        total_m1 = 0
        total_m2 = 0
        total_joint = 0
        used_periods = 0
        period_index = 0

        for _, period_row in periods_df.iterrows():
            clipped = _clip_period(
                period_row.get("start"),
                period_row.get("stop"),
                m1_bool.size,
            )
            if clipped is None:
                continue
            start, stop = clipped
            m1_seg = m1_bool[start:stop + 1]
            m2_seg = m2_bool[start:stop + 1]
            if m1_seg.size == 0 or m2_seg.size == 0:
                continue

            n_samples = int(m1_seg.size)
            m1_count = int(np.count_nonzero(m1_seg))
            m2_count = int(np.count_nonzero(m2_seg))
            joint_count = int(np.count_nonzero(m1_seg & m2_seg))

            p_m1 = _decimal_ratio(m1_count, n_samples, settings.decimal_precision)
            p_m2 = _decimal_ratio(m2_count, n_samples, settings.decimal_precision)
            p_joint = _decimal_ratio(joint_count, n_samples, settings.decimal_precision)
            p_product = None
            if p_m1 is not None and p_m2 is not None:
                with localcontext() as ctx:
                    ctx.prec = settings.decimal_precision
                    p_product = p_m1 * p_m2

            period_rows.append({
                "date": date,
                "session": session,
                "interactive_period_index": period_index,
                "interactive_start": start,
                "interactive_stop": stop,
                "interactive_state": period_row.get("state"),
                "mean_density": period_row.get("mean_density"),
                "threshold": period_row.get("threshold"),
                "n_samples": n_samples,
                "monkey_name_m1": m1_name,
                "monkey_name_m2": m2_name,
                "m1_face_count": m1_count,
                "m2_face_count": m2_count,
                "joint_face_count": joint_count,
                "p_m1_decimal": _decimal_str(p_m1),
                "p_m2_decimal": _decimal_str(p_m2),
                "p_joint_decimal": _decimal_str(p_joint),
                "p_product_decimal": _decimal_str(p_product),
            })

            period_index += 1
            used_periods += 1
            total_samples += n_samples
            total_m1 += m1_count
            total_m2 += m2_count
            total_joint += joint_count

        if total_samples == 0:
            continue

        p_m1 = _decimal_ratio(total_m1, total_samples, settings.decimal_precision)
        p_m2 = _decimal_ratio(total_m2, total_samples, settings.decimal_precision)
        p_joint = _decimal_ratio(total_joint, total_samples, settings.decimal_precision)
        p_product = None
        if p_m1 is not None and p_m2 is not None:
            with localcontext() as ctx:
                ctx.prec = settings.decimal_precision
                p_product = p_m1 * p_m2

        first_row = periods_df.iloc[0]
        concat_rows.append({
            "date": date,
            "session": session,
            "n_samples": total_samples,
            "n_interactive_periods": used_periods,
            "interactive_state": first_row.get("state"),
            "mean_density": first_row.get("mean_density"),
            "threshold": first_row.get("threshold"),
            "monkey_name_m1": m1_name,
            "monkey_name_m2": m2_name,
            "m1_face_count": total_m1,
            "m2_face_count": total_m2,
            "joint_face_count": total_joint,
            "p_m1_decimal": _decimal_str(p_m1),
            "p_m2_decimal": _decimal_str(p_m2),
            "p_joint_decimal": _decimal_str(p_joint),
            "p_product_decimal": _decimal_str(p_product),
        })

    return period_rows, concat_rows


def _build_cross_pairs(
    settings: FixationProbabilitySettings,
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


def _build_cross_session_rows(
    settings: FixationProbabilitySettings,
    m1_paths: dict,
    m2_paths: dict,
) -> list[dict]:
    """Build cross-session probability rows."""
    rows: list[dict] = []
    m1_keys = sorted(m1_paths)
    m2_keys = sorted(m2_paths)

    pairs = _build_cross_pairs(settings, m1_keys, m2_keys)
    if settings.test_single and pairs:
        pairs = pairs[: min(10, len(pairs))]

    m1_cache: dict[tuple[str, str], np.ndarray] = {}
    m2_cache: dict[tuple[str, str], np.ndarray] = {}
    m1_name_cache: dict[tuple[str, str], Optional[str]] = {}
    m2_name_cache: dict[tuple[str, str], Optional[str]] = {}

    for (date1, session1), (date2, session2) in tqdm(
        pairs,
        desc="Cross-session fixation",
        unit="pair",
    ):
        key1 = (date1, session1)
        key2 = (date2, session2)

        if key1 not in m1_cache:
            m1_vec, m1_name = _load_fixation_vector(
                m1_paths[key1],
                settings.fixation_label,
            )
            if m1_vec is None:
                continue
            m1_cache[key1] = _to_bool(m1_vec)
            m1_name_cache[key1] = m1_name
        if key2 not in m2_cache:
            m2_vec, m2_name = _load_fixation_vector(
                m2_paths[key2],
                settings.fixation_label,
            )
            if m2_vec is None:
                continue
            m2_cache[key2] = _to_bool(m2_vec)
            m2_name_cache[key2] = m2_name

        m1_bool = m1_cache[key1]
        m2_bool = m2_cache[key2]
        if m1_bool.size == 0 or m2_bool.size == 0:
            continue

        n_samples_m1 = int(m1_bool.size)
        n_samples_m2 = int(m2_bool.size)
        n_joint = int(min(n_samples_m1, n_samples_m2))
        if n_joint == 0:
            continue

        m1_slice = m1_bool[:n_joint]
        m2_slice = m2_bool[:n_joint]

        m1_count = int(np.count_nonzero(m1_slice))
        m2_count = int(np.count_nonzero(m2_slice))
        joint_count = int(np.count_nonzero(m1_slice & m2_slice))

        p_m1 = _decimal_ratio(m1_count, n_joint, settings.decimal_precision)
        p_m2 = _decimal_ratio(m2_count, n_joint, settings.decimal_precision)
        p_joint = _decimal_ratio(joint_count, n_joint, settings.decimal_precision)
        p_product = None
        if p_m1 is not None and p_m2 is not None:
            with localcontext() as ctx:
                ctx.prec = settings.decimal_precision
                p_product = p_m1 * p_m2

        rows.append({
            "date_m1": date1,
            "session_m1": session1,
            "date_m2": date2,
            "session_m2": session2,
            "n_samples_m1": n_samples_m1,
            "n_samples_m2": n_samples_m2,
            "n_samples_joint": n_joint,
            "monkey_name_m1": m1_name_cache.get(key1),
            "monkey_name_m2": m2_name_cache.get(key2),
            "m1_face_count_joint": m1_count,
            "m2_face_count_joint": m2_count,
            "joint_face_count": joint_count,
            "p_m1_decimal": _decimal_str(p_m1),
            "p_m2_decimal": _decimal_str(p_m2),
            "p_joint_decimal": _decimal_str(p_joint),
            "p_product_decimal": _decimal_str(p_product),
        })

    return rows


def run_fixation_probability_analysis(
    settings: FixationProbabilitySettings,
    *,
    compute_cross: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Run fixation probability analysis and persist outputs."""
    cfg = load_config(settings.cfg_path)
    m1_paths, m2_paths = _index_agent_paths(cfg, settings.input_modality)

    if not m1_paths or not m2_paths:
        raise RuntimeError(
            "Missing fixation binary vectors for m1 or m2. "
            f"Found m1={len(m1_paths)} m2={len(m2_paths)}."
        )

    within_rows = _build_within_session_rows(settings, m1_paths, m2_paths)
    within_df = pd.DataFrame.from_records(within_rows)

    cross_df = None
    if compute_cross:
        cross_rows = _build_cross_session_rows(settings, m1_paths, m2_paths)
        cross_df = pd.DataFrame.from_records(cross_rows)

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    within_path = out_dir / settings.within_filename
    within_df.to_csv(within_path, index=False)

    if cross_df is not None:
        cross_path = out_dir / settings.cross_filename
        cross_df.to_csv(cross_path, index=False)

    return within_df, cross_df


def run_interactive_fixation_probability_analysis(
    settings: FixationProbabilitySettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run interactive-period fixation probability analysis and persist outputs."""
    cfg = load_config(settings.cfg_path)
    m1_paths, m2_paths = _index_agent_paths(cfg, settings.input_modality)
    try:
        interactive_paths = _index_shared_paths(cfg, settings.interactive_modality)
    except RuntimeError:
        interactive_paths = {}

    if not interactive_paths:
        print(
            "No interactive periods found; skipping interactive-period "
            "fixation probability outputs."
        )
        return pd.DataFrame(), pd.DataFrame()

    period_rows, concat_rows = _build_interactive_rows(
        settings,
        m1_paths,
        m2_paths,
        interactive_paths,
    )

    period_df = pd.DataFrame.from_records(period_rows)
    concat_df = pd.DataFrame.from_records(concat_rows)

    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    period_path = out_dir / settings.interactive_periods_filename
    period_df.to_csv(period_path, index=False)

    concat_path = out_dir / settings.interactive_concat_filename
    concat_df.to_csv(concat_path, index=False)

    return period_df, concat_df
