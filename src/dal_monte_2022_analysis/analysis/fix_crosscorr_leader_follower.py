"""Derive leader-follower summaries from within-session fixation cross-correlations."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


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

    out_dir.mkdir(parents=True, exist_ok=True)
    session_out = out_dir / settings.session_output_filename
    date_out = out_dir / settings.date_summary_filename
    pair_out = out_dir / _resolve_pair_summary_filename(settings)
    global_out = out_dir / settings.global_summary_filename

    session_df.to_csv(session_out, index=False)
    date_summary_df.to_csv(date_out, index=False)
    pair_summary_df.to_csv(pair_out, index=False)
    global_summary_df.to_csv(global_out, index=False)

    print(f"[leader-follower] wrote session-level table: {session_out}")
    print(f"[leader-follower] wrote date-level summary: {date_out}")
    print(f"[leader-follower] wrote pair-level summary: {pair_out}")
    print(f"[leader-follower] wrote global summary: {global_out}")
    print(
        "[leader-follower] rows: "
        f"session={len(session_df)} date={len(date_summary_df)} "
        f"pair={len(pair_summary_df)} global={len(global_summary_df)}"
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

    return session_df, date_summary_df, pair_summary_df
