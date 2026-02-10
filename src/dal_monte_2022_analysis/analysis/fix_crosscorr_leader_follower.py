"""Derive leader-follower summaries from within-session fixation cross-correlations."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir, build_processed_data_path


@dataclass
class FixCrossCorrLeaderFollowerSettings:
    """Configuration for leader-follower summaries from cross-correlation outputs."""

    cfg_path: str
    fixation_label: str = "face"
    output_subdir: str = "fix_cross_correlation"
    within_filename: str = "within_session_face_fix_cross_correlation.pkl"
    lags_filename: Optional[str] = None
    session_output_filename: str = "within_session_face_fix_crosscorr_leader_follower.csv"
    date_summary_filename: str = "date_summary_face_fix_crosscorr_leader_follower.csv"
    pair_summary_filename: str = "pair_summary_face_fix_crosscorr_leader_follower.csv"
    # Backward compatibility for older callers/configs.
    total_summary_filename: Optional[str] = None
    global_summary_filename: str = "global_summary_face_fix_crosscorr_leader_follower.csv"
    fixations_modality: str = "fixations"
    pupil_modality: str = "pupil_size"
    pupil_session_output_filename: str = (
        "within_session_face_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
    )
    pupil_date_summary_filename: str = (
        "date_summary_face_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
    )
    pupil_pair_summary_filename: str = (
        "pair_summary_face_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
    )
    pupil_global_summary_filename: str = (
        "global_summary_face_fix_crosscorr_leader_follower_pupil_during_fixation.csv"
    )
    pupil_roi_keywords: Optional[list[str]] = None
    pupil_test_n_permutations: int = 2000
    pupil_test_seed: int = 13
    pupil_test_alpha: float = 0.05
    pupil_test_max_samples_per_group: int = 5000
    tie_epsilon: float = 0.0


LEADER_DELTA_COL = "leader_minus_follower_fixation_count"
SESSION_REQUIRED_COLUMNS = {
    "date",
    "session",
    "monkey_name_m1",
    "monkey_name_m2",
    "m1_fixation_count",
    "m2_fixation_count",
    "cross_correlation",
}
SESSION_OUTPUT_COLUMNS = [
    "fixation_label",
    "date",
    "session",
    "pair_key",
    "monkey_name_m1",
    "monkey_name_m2",
    "m1_fixation_count",
    "m2_fixation_count",
    "mean_positive_lag_correlation",
    "mean_negative_lag_correlation",
    "lead_score",
    "leader_agent",
    "follower_agent",
    "leader_monkey",
    "follower_monkey",
    "leader_fixation_count",
    "follower_fixation_count",
    LEADER_DELTA_COL,
]
PROPERTY_SUMMARY_METRIC_COLUMNS = [
    "n_sessions",
    "n_pos",
    "n_neg",
    "n_zero",
    "mean_delta",
    "delta_consistency",
]
PUPIL_PROPERTY_BASE_COLUMNS = [
    "n_sessions",
    "n_comp_sessions",
    "n_lead",
    "n_follow",
    "lead_mean",
    "follow_mean",
    "mean_diff",
    "p",
    "sig",
    "higher",
]
PUPIL_PROPERTY_SESSION_COLUMNS = [
    "fixation_label",
    "date",
    "session",
    "pair_key",
    "monkey_name_m1",
    "monkey_name_m2",
    "leader_agent",
    "follower_agent",
    *PUPIL_PROPERTY_BASE_COLUMNS,
]
PUPIL_PROPERTY_SUMMARY_COLUMNS = PUPIL_PROPERTY_BASE_COLUMNS
PUPIL_PROPERTY_GLOBAL_PREFIX_COLUMNS = ["fixation_label", "n_pairs", "n_dates"]
DEFAULT_ROI_KEYWORDS_BY_FIXATION_LABEL = {
    "face": ("face", "eyes_nf", "mouth"),
    "out_of_roi": ("out_of_roi",),
    "object": ("right_nonsocial_object", "left_nonsocial_object"),
}


def _resolve_lags_filename(settings: FixCrossCorrLeaderFollowerSettings) -> str:
    """Return lag-axis filename."""
    if settings.lags_filename:
        return settings.lags_filename
    return f"{settings.fixation_label}_crosscorrelation_lags.pkl"


def _resolve_pair_summary_filename(settings: FixCrossCorrLeaderFollowerSettings) -> str:
    """Return pair-level summary filename with backward compatibility."""
    if settings.pair_summary_filename:
        return settings.pair_summary_filename
    if settings.total_summary_filename:
        return settings.total_summary_filename
    return f"pair_summary_{settings.fixation_label}_fix_crosscorr_leader_follower.csv"


def _load_lags(path: Path) -> np.ndarray:
    """Load lag axis from pickle."""
    with open(path, "rb") as f:
        lags = pickle.load(f)
    lags = np.asarray(lags, dtype=np.int64).reshape(-1)
    if lags.size == 0:
        raise RuntimeError(f"Lag axis is empty: {path}")
    return lags


def _safe_float(value: object) -> float:
    """Convert to float, returning NaN for invalid values."""
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan


def _empty_summary_df(group_cols: list[str]) -> pd.DataFrame:
    """Return an empty summary table with standard output columns."""
    return pd.DataFrame(columns=group_cols + PROPERTY_SUMMARY_METRIC_COLUMNS)


def _assign_consistency_label(
    n_positive: np.ndarray,
    n_negative: np.ndarray,
    n_zero: np.ndarray,
) -> np.ndarray:
    """Classify sign consistency of leader-minus-follower fixation deltas."""
    valid = n_positive + n_negative + n_zero
    labels = np.full(valid.size, "mixed", dtype=object)
    labels[valid == 0.0] = "no_data"
    labels[(valid > 0.0) & (n_positive == valid)] = "all_positive"
    labels[(valid > 0.0) & (n_negative == valid)] = "all_negative"
    labels[(valid > 0.0) & (n_zero == valid)] = "all_zero"
    labels[(valid > 0.0) & (n_positive > 0.0) & (n_negative == 0.0) & (n_zero > 0.0)] = (
        "positive_or_zero"
    )
    labels[(valid > 0.0) & (n_negative > 0.0) & (n_positive == 0.0) & (n_zero > 0.0)] = (
        "negative_or_zero"
    )
    return labels


def _load_pickle(path: Path):
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_pupil_vector(obj) -> np.ndarray:
    """Extract 1D pupil-size vector from supported object layouts."""
    if hasattr(obj, "d"):
        values = getattr(obj, "d")
    elif isinstance(obj, dict) and "d" in obj:
        values = obj["d"]
    else:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr if arr.size else np.asarray([], dtype=np.float64)


def _coerce_location_labels(loc) -> list[str]:
    """Normalize fixation location field into lowercase labels."""
    if loc is None:
        return []
    if isinstance(loc, (list, tuple, set, np.ndarray)):
        labels = [str(val).lower() for val in loc if val is not None]
    else:
        try:
            if pd.isna(loc):
                return []
        except Exception:
            pass
        labels = [str(loc).lower()]
    return labels


def _location_matches_keywords(location_labels: list[str], keywords: tuple[str, ...]) -> bool:
    """Return whether any location label contains any keyword substring."""
    for label in location_labels:
        for keyword in keywords:
            if keyword in label:
                return True
    return False


def _resolve_pupil_roi_keywords(settings: FixCrossCorrLeaderFollowerSettings) -> tuple[str, ...]:
    """Resolve ROI keywords used for pupil-during-fixation property extraction."""
    if settings.pupil_roi_keywords:
        return tuple(str(val).lower() for val in settings.pupil_roi_keywords)
    keywords = DEFAULT_ROI_KEYWORDS_BY_FIXATION_LABEL.get(str(settings.fixation_label).lower())
    if keywords:
        return tuple(str(val).lower() for val in keywords)
    return (str(settings.fixation_label).lower(),)


def _stable_int_seed(base_seed: int, *parts: object) -> int:
    """Build deterministic seed for reproducible subsampling/permutation tests."""
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(token.encode("utf-8")).digest()[:8]
    offset = int.from_bytes(digest, byteorder="little", signed=False)
    return int((int(base_seed) + offset) % (2**32 - 1))


def _subsample_for_test(
    values: np.ndarray,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly subsample values when sample sizes are too large for permutation testing."""
    values = np.asarray(values, dtype=np.float64)
    if max_samples <= 0 or values.size <= max_samples:
        return values
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx]


def _permutation_p_value_mean_diff(
    lead_values: np.ndarray,
    follow_values: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided permutation p-value for mean difference."""
    if n_permutations <= 0:
        return float("nan")

    x = np.asarray(lead_values, dtype=np.float64)
    y = np.asarray(follow_values, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")

    observed = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = int(x.size)
    n_total = int(pooled.size)
    if n_total <= 1:
        return float("nan")

    extreme = 0
    for _ in range(int(n_permutations)):
        perm_idx = rng.permutation(n_total)
        diff = float(np.mean(pooled[perm_idx[:n_x]]) - np.mean(pooled[perm_idx[n_x:]]))
        if abs(diff) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (int(n_permutations) + 1))


def _compare_pupil_samples(
    lead_values: np.ndarray,
    follow_values: np.ndarray,
    *,
    n_permutations: int,
    alpha: float,
    max_samples_per_group: int,
    seed: int,
) -> dict:
    """Compare leader vs follower pupil arrays and report direction + significance."""
    lead_values = np.asarray(lead_values, dtype=np.float64)
    follow_values = np.asarray(follow_values, dtype=np.float64)
    lead_values = lead_values[np.isfinite(lead_values)]
    follow_values = follow_values[np.isfinite(follow_values)]

    n_lead = int(lead_values.size)
    n_follow = int(follow_values.size)

    lead_mean = float(np.mean(lead_values)) if n_lead > 0 else np.nan
    follow_mean = float(np.mean(follow_values)) if n_follow > 0 else np.nan
    mean_diff = float(lead_mean - follow_mean) if n_lead > 0 and n_follow > 0 else np.nan

    if not np.isfinite(mean_diff):
        higher = "no_data"
    elif mean_diff > 0.0:
        higher = "leader"
    elif mean_diff < 0.0:
        higher = "follower"
    else:
        higher = "equal"

    if n_lead == 0 or n_follow == 0:
        p_value = np.nan
    else:
        rng = np.random.default_rng(int(seed))
        lead_for_test = _subsample_for_test(
            lead_values,
            max_samples=int(max_samples_per_group),
            rng=rng,
        )
        follow_for_test = _subsample_for_test(
            follow_values,
            max_samples=int(max_samples_per_group),
            rng=rng,
        )
        p_value = _permutation_p_value_mean_diff(
            lead_for_test,
            follow_for_test,
            n_permutations=int(n_permutations),
            rng=rng,
        )
    is_significant = bool(np.isfinite(p_value) and p_value < float(alpha))

    return {
        "n_lead": n_lead,
        "n_follow": n_follow,
        "lead_mean": lead_mean,
        "follow_mean": follow_mean,
        "mean_diff": mean_diff,
        "p": p_value,
        "sig": is_significant,
        "higher": higher,
    }


def _extract_pupil_during_fixations(
    *,
    cfg: dict,
    date: str,
    session: str,
    agent: str,
    fixations_modality: str,
    pupil_modality: str,
    roi_keywords: tuple[str, ...],
    fix_cache: dict[tuple[str, str, str], Optional[pd.DataFrame]],
    pupil_cache: dict[tuple[str, str, str], np.ndarray],
) -> np.ndarray:
    """Extract pupil samples during ROI-matching fixations for one session/agent."""
    key = (str(date), str(session), str(agent))
    row = {"date": str(date), "session": str(session)}

    if key not in fix_cache:
        fix_path = build_processed_data_path(cfg, row, fixations_modality, agent)
        if fix_path.exists():
            obj = _load_pickle(fix_path)
            fix_cache[key] = obj if isinstance(obj, pd.DataFrame) else None
        else:
            fix_cache[key] = None
    if key not in pupil_cache:
        pupil_path = build_processed_data_path(cfg, row, pupil_modality, agent)
        if pupil_path.exists():
            pupil_cache[key] = _extract_pupil_vector(_load_pickle(pupil_path))
        else:
            pupil_cache[key] = np.asarray([], dtype=np.float64)

    fix_df = fix_cache[key]
    pupil = pupil_cache[key]
    if fix_df is None or fix_df.empty or pupil.size == 0:
        return np.asarray([], dtype=np.float64)
    if "start" not in fix_df.columns or "stop" not in fix_df.columns or "location" not in fix_df.columns:
        return np.asarray([], dtype=np.float64)

    n_samples = int(pupil.size)
    segments: list[np.ndarray] = []
    for _, fix_row in fix_df.iterrows():
        locations = _coerce_location_labels(fix_row.get("location"))
        if not _location_matches_keywords(locations, roi_keywords):
            continue
        start = pd.to_numeric(fix_row.get("start"), errors="coerce")
        stop = pd.to_numeric(fix_row.get("stop"), errors="coerce")
        if pd.isna(start) or pd.isna(stop):
            continue
        start_i = int(start)
        stop_i = int(stop)
        if stop_i < 0 or start_i >= n_samples:
            continue
        start_i = max(0, start_i)
        stop_i = min(n_samples - 1, stop_i)
        if start_i > stop_i:
            continue
        segment = pupil[start_i : stop_i + 1]
        segment = segment[np.isfinite(segment)]
        if segment.size > 0:
            segments.append(segment)

    if not segments:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(segments)


def _build_session_pupil_property_table(
    *,
    cfg: dict,
    settings: FixCrossCorrLeaderFollowerSettings,
    session_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build session-level pupil property table from known leader/follower calls."""
    roi_keywords = _resolve_pupil_roi_keywords(settings)
    fix_cache: dict[tuple[str, str, str], Optional[pd.DataFrame]] = {}
    pupil_cache: dict[tuple[str, str, str], np.ndarray] = {}

    rows: list[dict] = []
    for _, session_row in session_df.iterrows():
        date = str(session_row["date"])
        session = str(session_row["session"])
        leader_agent = str(session_row["leader_agent"])
        follower_agent = str(session_row["follower_agent"])

        m1_vals = _extract_pupil_during_fixations(
            cfg=cfg,
            date=date,
            session=session,
            agent="m1",
            fixations_modality=settings.fixations_modality,
            pupil_modality=settings.pupil_modality,
            roi_keywords=roi_keywords,
            fix_cache=fix_cache,
            pupil_cache=pupil_cache,
        )
        m2_vals = _extract_pupil_during_fixations(
            cfg=cfg,
            date=date,
            session=session,
            agent="m2",
            fixations_modality=settings.fixations_modality,
            pupil_modality=settings.pupil_modality,
            roi_keywords=roi_keywords,
            fix_cache=fix_cache,
            pupil_cache=pupil_cache,
        )

        if leader_agent == "m1" and follower_agent == "m2":
            lead_vals = m1_vals
            follow_vals = m2_vals
        elif leader_agent == "m2" and follower_agent == "m1":
            lead_vals = m2_vals
            follow_vals = m1_vals
        else:
            lead_vals = np.asarray([], dtype=np.float64)
            follow_vals = np.asarray([], dtype=np.float64)

        compare = _compare_pupil_samples(
            lead_vals,
            follow_vals,
            n_permutations=settings.pupil_test_n_permutations,
            alpha=settings.pupil_test_alpha,
            max_samples_per_group=settings.pupil_test_max_samples_per_group,
            seed=_stable_int_seed(
                settings.pupil_test_seed,
                "session",
                settings.fixation_label,
                date,
                session,
            ),
        )
        rows.append(
            {
                "fixation_label": settings.fixation_label,
                "date": date,
                "session": session,
                "pair_key": session_row["pair_key"],
                "monkey_name_m1": session_row["monkey_name_m1"],
                "monkey_name_m2": session_row["monkey_name_m2"],
                "leader_agent": leader_agent,
                "follower_agent": follower_agent,
                "n_sessions": 1,
                "n_comp_sessions": int(compare["n_lead"] > 0 and compare["n_follow"] > 0),
                **compare,
                "_lead_vals": lead_vals,
                "_follow_vals": follow_vals,
            }
        )

    if not rows:
        return pd.DataFrame(columns=PUPIL_PROPERTY_SESSION_COLUMNS)
    out = pd.DataFrame.from_records(rows)
    out = out.sort_values(["pair_key", "date", "session"]).reset_index(drop=True)
    return out


def _summarize_pupil_property_table(
    session_pupil_df: pd.DataFrame,
    *,
    group_cols: list[str],
    sort_cols: list[str],
    seed_token: str,
    settings: FixCrossCorrLeaderFollowerSettings,
) -> pd.DataFrame:
    """Aggregate session-level pupil samples and test leader vs follower by group."""
    if session_pupil_df.empty:
        return pd.DataFrame(columns=group_cols + PUPIL_PROPERTY_SUMMARY_COLUMNS)

    rows: list[dict] = []
    for group_values, group_df in session_pupil_df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_dict = dict(zip(group_cols, group_values))

        lead_segments = [np.asarray(arr, dtype=np.float64) for arr in group_df["_lead_vals"] if arr is not None]
        follow_segments = [
            np.asarray(arr, dtype=np.float64) for arr in group_df["_follow_vals"] if arr is not None
        ]
        lead_values = (
            np.concatenate([arr for arr in lead_segments if arr.size > 0])
            if any(arr.size > 0 for arr in lead_segments)
            else np.asarray([], dtype=np.float64)
        )
        follow_values = (
            np.concatenate([arr for arr in follow_segments if arr.size > 0])
            if any(arr.size > 0 for arr in follow_segments)
            else np.asarray([], dtype=np.float64)
        )

        compare = _compare_pupil_samples(
            lead_values,
            follow_values,
            n_permutations=settings.pupil_test_n_permutations,
            alpha=settings.pupil_test_alpha,
            max_samples_per_group=settings.pupil_test_max_samples_per_group,
            seed=_stable_int_seed(
                settings.pupil_test_seed,
                "summary",
                seed_token,
                settings.fixation_label,
                *group_values,
            ),
        )

        row = {
            **group_dict,
            "n_sessions": int(len(group_df)),
            "n_comp_sessions": int(
                np.sum(
                    (group_df["n_lead"].to_numpy(dtype=np.int64) > 0)
                    & (group_df["n_follow"].to_numpy(dtype=np.int64) > 0)
                )
            ),
            **compare,
        }
        rows.append(row)

    out = pd.DataFrame.from_records(rows)
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out[group_cols + PUPIL_PROPERTY_SUMMARY_COLUMNS]


def _summarize_pupil_property_by_date(
    session_pupil_df: pd.DataFrame,
    settings: FixCrossCorrLeaderFollowerSettings,
) -> pd.DataFrame:
    """Aggregate pupil property by pair/date."""
    group_cols = ["fixation_label", "pair_key", "monkey_name_m1", "monkey_name_m2", "date"]
    if session_pupil_df.empty:
        return pd.DataFrame(columns=group_cols + PUPIL_PROPERTY_SUMMARY_COLUMNS)
    return _summarize_pupil_property_table(
        session_pupil_df,
        group_cols=group_cols,
        sort_cols=["pair_key", "date"],
        seed_token="date",
        settings=settings,
    )


def _summarize_pupil_property_by_pair(
    session_pupil_df: pd.DataFrame,
    settings: FixCrossCorrLeaderFollowerSettings,
) -> pd.DataFrame:
    """Aggregate pupil property by pair across sessions."""
    group_cols = ["fixation_label", "pair_key", "monkey_name_m1", "monkey_name_m2"]
    if session_pupil_df.empty:
        return pd.DataFrame(columns=group_cols + PUPIL_PROPERTY_SUMMARY_COLUMNS)
    return _summarize_pupil_property_table(
        session_pupil_df,
        group_cols=group_cols,
        sort_cols=["pair_key"],
        seed_token="pair",
        settings=settings,
    )


def _summarize_pupil_property_global(
    session_pupil_df: pd.DataFrame,
    settings: FixCrossCorrLeaderFollowerSettings,
) -> pd.DataFrame:
    """Aggregate pupil property globally across sessions/pairs/dates."""
    group_cols = ["fixation_label"]
    if session_pupil_df.empty:
        return pd.DataFrame(columns=PUPIL_PROPERTY_GLOBAL_PREFIX_COLUMNS + PUPIL_PROPERTY_SUMMARY_COLUMNS)

    key_counts = (
        session_pupil_df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_pairs=("pair_key", "nunique"),
            n_dates=("date", "nunique"),
        )
        .reset_index(drop=True)
    )
    prop = _summarize_pupil_property_table(
        session_pupil_df,
        group_cols=group_cols,
        sort_cols=["fixation_label"],
        seed_token="global",
        settings=settings,
    )
    out = key_counts.merge(prop, how="inner", on="fixation_label")
    return out[PUPIL_PROPERTY_GLOBAL_PREFIX_COLUMNS + PUPIL_PROPERTY_SUMMARY_COLUMNS]

def _determine_session_leader_follower(
    within_df: pd.DataFrame,
    *,
    lags: np.ndarray,
    fixation_label: str,
    tie_epsilon: float,
) -> pd.DataFrame:
    """Compute per-session leader/follower calls and fixation count deltas."""
    missing = SESSION_REQUIRED_COLUMNS.difference(within_df.columns)
    if missing:
        raise RuntimeError(
            f"Within-session cross-correlation table is missing required columns: {sorted(missing)}"
        )

    pos_idx = np.flatnonzero(lags > 0)
    neg_idx = np.flatnonzero(lags < 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise RuntimeError("Lag axis must include both positive and negative lags.")

    rows: list[dict] = []
    for _, row in within_df.iterrows():
        corr = np.asarray(row["cross_correlation"], dtype=np.float64).reshape(-1)
        if corr.size != lags.size:
            raise RuntimeError(
                "Cross-correlation length does not match lag axis length "
                f"for date={row['date']} session={row['session']} "
                f"(corr={corr.size}, lags={lags.size})."
            )

        mean_pos = float(np.mean(corr[pos_idx]))
        mean_neg = float(np.mean(corr[neg_idx]))
        lead_score = mean_pos - mean_neg
        m1_fixation_count = _safe_float(row["m1_fixation_count"])
        m2_fixation_count = _safe_float(row["m2_fixation_count"])
        pair_key = f"{row['monkey_name_m1']}__{row['monkey_name_m2']}"

        if lead_score > float(tie_epsilon):
            leader_agent = "m1"
            follower_agent = "m2"
            leader_monkey = row["monkey_name_m1"]
            follower_monkey = row["monkey_name_m2"]
            leader_fixation_count = m1_fixation_count
            follower_fixation_count = m2_fixation_count
        elif lead_score < -float(tie_epsilon):
            leader_agent = "m2"
            follower_agent = "m1"
            leader_monkey = row["monkey_name_m2"]
            follower_monkey = row["monkey_name_m1"]
            leader_fixation_count = m2_fixation_count
            follower_fixation_count = m1_fixation_count
        else:
            leader_agent = "tie"
            follower_agent = "tie"
            leader_monkey = None
            follower_monkey = None
            leader_fixation_count = np.nan
            follower_fixation_count = np.nan

        if np.isfinite(leader_fixation_count) and np.isfinite(follower_fixation_count):
            leader_minus_follower_fixation_count = leader_fixation_count - follower_fixation_count
        else:
            leader_minus_follower_fixation_count = np.nan

        rows.append(
            {
                "fixation_label": fixation_label,
                "date": row["date"],
                "session": row["session"],
                "pair_key": pair_key,
                "monkey_name_m1": row["monkey_name_m1"],
                "monkey_name_m2": row["monkey_name_m2"],
                "m1_fixation_count": m1_fixation_count,
                "m2_fixation_count": m2_fixation_count,
                "mean_positive_lag_correlation": mean_pos,
                "mean_negative_lag_correlation": mean_neg,
                "lead_score": lead_score,
                "leader_agent": leader_agent,
                "follower_agent": follower_agent,
                "leader_monkey": leader_monkey,
                "follower_monkey": follower_monkey,
                "leader_fixation_count": leader_fixation_count,
                "follower_fixation_count": follower_fixation_count,
                LEADER_DELTA_COL: leader_minus_follower_fixation_count,
            }
        )

    if not rows:
        return pd.DataFrame(columns=SESSION_OUTPUT_COLUMNS)
    return pd.DataFrame.from_records(rows, columns=SESSION_OUTPUT_COLUMNS).sort_values(
        ["pair_key", "date", "session"]
    ).reset_index(drop=True)


def _compute_session_leader_rows(
    within_df: pd.DataFrame,
    *,
    lags: np.ndarray,
    fixation_label: str,
    tie_epsilon: float,
) -> pd.DataFrame:
    """Backward-compatible alias for session-level leader/follower determination."""
    return _determine_session_leader_follower(
        within_df=within_df,
        lags=lags,
        fixation_label=fixation_label,
        tie_epsilon=tie_epsilon,
    )


def _summarize_fixation_count_property_by_date(session_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize leader-vs-follower fixation-count deltas by pair and date."""
    if session_df.empty:
        return _empty_summary_df(
            ["fixation_label", "pair_key", "monkey_name_m1", "monkey_name_m2", "date"]
        )
    return _build_fixation_count_property_summary(
        session_df,
        group_cols=["fixation_label", "pair_key", "monkey_name_m1", "monkey_name_m2", "date"],
        sort_cols=["pair_key", "date"],
    )


def _summarize_fixation_count_property_by_pair(session_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize leader-vs-follower fixation-count deltas by pair across sessions."""
    if session_df.empty:
        return _empty_summary_df(["fixation_label", "pair_key", "monkey_name_m1", "monkey_name_m2"])
    return _build_fixation_count_property_summary(
        session_df,
        group_cols=["fixation_label", "pair_key", "monkey_name_m1", "monkey_name_m2"],
        sort_cols=["pair_key"],
    )


def _summarize_fixation_count_property_global(session_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize leader-vs-follower fixation-count deltas globally across all sessions."""
    if session_df.empty:
        return _empty_summary_df(["fixation_label", "n_pairs", "n_dates"])

    key_counts = (
        session_df.groupby(["fixation_label"], dropna=False, as_index=False)
        .agg(
            n_pairs=("pair_key", "nunique"),
            n_dates=("date", "nunique"),
        )
        .reset_index(drop=True)
    )
    property_summary = _build_fixation_count_property_summary(
        session_df,
        group_cols=["fixation_label"],
        sort_cols=["fixation_label"],
    )
    return key_counts.merge(property_summary, how="inner", on="fixation_label")


def _build_fixation_count_property_summary(
    session_df: pd.DataFrame,
    *,
    group_cols: list[str],
    sort_cols: list[str],
) -> pd.DataFrame:
    """Aggregate fixation-count difference properties given known leader/follower labels."""
    summary = (
        session_df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_sessions=("leader_agent", "size"),
            n_pos=(LEADER_DELTA_COL, lambda s: int((s > 0.0).sum())),
            n_neg=(LEADER_DELTA_COL, lambda s: int((s < 0.0).sum())),
            n_zero=(LEADER_DELTA_COL, lambda s: int((s == 0.0).sum())),
            mean_delta=(LEADER_DELTA_COL, "mean"),
        )
    )
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)

    pos = summary["n_pos"].to_numpy(dtype=np.float64)
    neg = summary["n_neg"].to_numpy(dtype=np.float64)
    zero = summary["n_zero"].to_numpy(dtype=np.float64)
    summary["delta_consistency"] = _assign_consistency_label(pos, neg, zero)
    return summary[group_cols + PROPERTY_SUMMARY_METRIC_COLUMNS]


def _summarize_by_date(session_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for date-level fixation-count property summaries."""
    return _summarize_fixation_count_property_by_date(session_df)


def _summarize_by_pair(session_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for pair-level fixation-count property summaries."""
    return _summarize_fixation_count_property_by_pair(session_df)


def _summarize_total(session_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for pair-level fixation-count property summaries."""
    return _summarize_fixation_count_property_by_pair(session_df)


def _summarize_global(session_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for global fixation-count property summaries."""
    return _summarize_fixation_count_property_global(session_df)


def _print_fixation_count_property_summaries(
    *,
    fixation_label: str,
    date_summary_df: pd.DataFrame,
    pair_summary_df: pd.DataFrame,
    global_summary_df: pd.DataFrame,
) -> None:
    """Print fixation-count property summaries based on precomputed leader/follower labels."""
    print("\n[leader-follower] -----------------------------------------------")
    print(f"[leader-follower] fixation_label={fixation_label}")
    print("[leader-follower] Fixation-count properties by date and pair")
    print("[leader-follower] -----------------------------------------------")

    if date_summary_df.empty:
        print("[leader-follower] No date-level rows found.")
    else:
        for pair_key in date_summary_df["pair_key"].drop_duplicates():
            pair_rows = date_summary_df[date_summary_df["pair_key"] == pair_key]
            monkey_name_m1 = pair_rows["monkey_name_m1"].iloc[0]
            monkey_name_m2 = pair_rows["monkey_name_m2"].iloc[0]
            print(f"\nPair: m1={monkey_name_m1} vs m2={monkey_name_m2}")
            table_cols = [
                "date",
                *PROPERTY_SUMMARY_METRIC_COLUMNS,
            ]
            print(pair_rows[table_cols].to_string(index=False))

    print("\n[leader-follower] -----------------------------------------------")
    print("[leader-follower] Fixation-count properties by pair (all sessions)")
    print("[leader-follower] -----------------------------------------------")
    if pair_summary_df.empty:
        print("[leader-follower] No pair-summary rows found.")
    else:
        print(
            pair_summary_df[
                [
                    "monkey_name_m1",
                    "monkey_name_m2",
                    *PROPERTY_SUMMARY_METRIC_COLUMNS,
                ]
            ].to_string(index=False)
        )

    print("\n[leader-follower] -----------------------------------------------")
    print("[leader-follower] Global fixation-count properties")
    print("[leader-follower] -----------------------------------------------")
    if global_summary_df.empty:
        print("[leader-follower] No global-summary rows found.")
    else:
        print(
            global_summary_df[
                [
                    "fixation_label",
                    "n_pairs",
                    "n_dates",
                    *PROPERTY_SUMMARY_METRIC_COLUMNS,
                ]
            ].to_string(index=False)
        )
    print("[leader-follower] -----------------------------------------------\n")


def _print_pupil_property_summaries(
    *,
    fixation_label: str,
    pupil_session_df: pd.DataFrame,
    pupil_date_df: pd.DataFrame,
    pupil_pair_df: pd.DataFrame,
    pupil_global_df: pd.DataFrame,
) -> None:
    """Print pupil-size-during-fixation property summaries from leader/follower labels."""
    print("\n[leader-follower] -----------------------------------------------")
    print(f"[leader-follower] fixation_label={fixation_label}")
    print("[leader-follower] Pupil properties during ROI fixations")
    print("[leader-follower] -----------------------------------------------")
    print(f"[leader-follower] pupil session rows: {len(pupil_session_df)}")

    print("\n[leader-follower] Pupil properties by date and pair")
    if pupil_date_df.empty:
        print("[leader-follower] No pupil date-level rows found.")
    else:
        for pair_key in pupil_date_df["pair_key"].drop_duplicates():
            pair_rows = pupil_date_df[pupil_date_df["pair_key"] == pair_key]
            monkey_name_m1 = pair_rows["monkey_name_m1"].iloc[0]
            monkey_name_m2 = pair_rows["monkey_name_m2"].iloc[0]
            print(f"\nPair: m1={monkey_name_m1} vs m2={monkey_name_m2}")
            print(pair_rows[["date", *PUPIL_PROPERTY_SUMMARY_COLUMNS]].to_string(index=False))

    print("\n[leader-follower] Pupil properties by pair (all sessions)")
    if pupil_pair_df.empty:
        print("[leader-follower] No pupil pair-level rows found.")
    else:
        print(
            pupil_pair_df[
                [
                    "monkey_name_m1",
                    "monkey_name_m2",
                    *PUPIL_PROPERTY_SUMMARY_COLUMNS,
                ]
            ].to_string(index=False)
        )

    print("\n[leader-follower] Pupil properties global")
    if pupil_global_df.empty:
        print("[leader-follower] No pupil global rows found.")
    else:
        print(
            pupil_global_df[
                [
                    "fixation_label",
                    "n_pairs",
                    "n_dates",
                    *PUPIL_PROPERTY_SUMMARY_COLUMNS,
                ]
            ].to_string(index=False)
        )
    print("[leader-follower] -----------------------------------------------\n")


def run_fix_crosscorr_leader_follower_analysis(
    settings: FixCrossCorrLeaderFollowerSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build session-level leader calls and fixation-count property summaries."""
    cfg = load_dataset_config(settings.cfg_path)
    out_dir = build_analysis_output_dir(cfg, settings.output_subdir)
    within_path = out_dir / settings.within_filename
    lags_path = out_dir / _resolve_lags_filename(settings)

    if not within_path.exists():
        raise RuntimeError(f"Missing within-session cross-correlation file: {within_path}")
    if not lags_path.exists():
        raise RuntimeError(f"Missing lag-axis file: {lags_path}")

    within_df = pd.read_pickle(within_path)
    lags = _load_lags(lags_path)
    session_df = _determine_session_leader_follower(
        within_df=within_df,
        lags=lags,
        fixation_label=settings.fixation_label,
        tie_epsilon=settings.tie_epsilon,
    )
    date_summary_df = _summarize_fixation_count_property_by_date(session_df)
    pair_summary_df = _summarize_fixation_count_property_by_pair(session_df)
    global_summary_df = _summarize_fixation_count_property_global(session_df)
    pupil_session_df = _build_session_pupil_property_table(
        cfg=cfg,
        settings=settings,
        session_df=session_df,
    )
    pupil_date_df = _summarize_pupil_property_by_date(pupil_session_df, settings)
    pupil_pair_df = _summarize_pupil_property_by_pair(pupil_session_df, settings)
    pupil_global_df = _summarize_pupil_property_global(pupil_session_df, settings)

    out_dir.mkdir(parents=True, exist_ok=True)
    session_out = out_dir / settings.session_output_filename
    date_out = out_dir / settings.date_summary_filename
    pair_out = out_dir / _resolve_pair_summary_filename(settings)
    global_out = out_dir / settings.global_summary_filename
    pupil_session_out = out_dir / settings.pupil_session_output_filename
    pupil_date_out = out_dir / settings.pupil_date_summary_filename
    pupil_pair_out = out_dir / settings.pupil_pair_summary_filename
    pupil_global_out = out_dir / settings.pupil_global_summary_filename

    session_df.to_csv(session_out, index=False)
    date_summary_df.to_csv(date_out, index=False)
    pair_summary_df.to_csv(pair_out, index=False)
    global_summary_df.to_csv(global_out, index=False)
    pupil_session_df[PUPIL_PROPERTY_SESSION_COLUMNS].to_csv(pupil_session_out, index=False)
    pupil_date_df.to_csv(pupil_date_out, index=False)
    pupil_pair_df.to_csv(pupil_pair_out, index=False)
    pupil_global_df.to_csv(pupil_global_out, index=False)

    print(f"[leader-follower] wrote session-level table: {session_out}")
    print(f"[leader-follower] wrote date-level summary: {date_out}")
    print(f"[leader-follower] wrote pair-level summary: {pair_out}")
    print(f"[leader-follower] wrote global summary: {global_out}")
    print(f"[leader-follower] wrote pupil session-level summary: {pupil_session_out}")
    print(f"[leader-follower] wrote pupil date-level summary: {pupil_date_out}")
    print(f"[leader-follower] wrote pupil pair-level summary: {pupil_pair_out}")
    print(f"[leader-follower] wrote pupil global summary: {pupil_global_out}")
    print(
        "[leader-follower] rows: "
        f"session={len(session_df)} date={len(date_summary_df)} "
        f"pair={len(pair_summary_df)} global={len(global_summary_df)} "
        f"pupil_session={len(pupil_session_df)} pupil_date={len(pupil_date_df)} "
        f"pupil_pair={len(pupil_pair_df)} pupil_global={len(pupil_global_df)}"
    )

    print(
        "[leader-follower] note: summaries printed below report fixation-count properties "
        "given leader/follower labels, not separate leader-call breakdown tables."
    )

    _print_fixation_count_property_summaries(
        fixation_label=settings.fixation_label,
        date_summary_df=date_summary_df,
        pair_summary_df=pair_summary_df,
        global_summary_df=global_summary_df,
    )
    _print_pupil_property_summaries(
        fixation_label=settings.fixation_label,
        pupil_session_df=pupil_session_df,
        pupil_date_df=pupil_date_df,
        pupil_pair_df=pupil_pair_df,
        pupil_global_df=pupil_global_df,
    )

    return session_df, date_summary_df, pair_summary_df
