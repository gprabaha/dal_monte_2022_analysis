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
    total_summary_filename: str = "total_summary_face_fix_crosscorr_leader_follower.csv"
    tie_epsilon: float = 0.0


def _resolve_lags_filename(settings: FixCrossCorrLeaderFollowerSettings) -> str:
    """Return lag-axis filename."""
    if settings.lags_filename:
        return settings.lags_filename
    return f"{settings.fixation_label}_crosscorrelation_lags.pkl"


def _load_lags(path: Path) -> np.ndarray:
    """Load lag axis from pickle."""
    with open(path, "rb") as f:
        lags = pickle.load(f)
    lags = np.asarray(lags, dtype=np.int64).reshape(-1)
    if lags.size == 0:
        raise RuntimeError(f"Lag axis is empty: {path}")
    return lags


def _build_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stable monkey-pair identifiers."""
    out = df.copy()
    out["pair_key"] = out["monkey_name_m1"].astype(str) + "__" + out["monkey_name_m2"].astype(
        str
    )
    return out


def _compute_session_leader_rows(
    within_df: pd.DataFrame,
    *,
    lags: np.ndarray,
    fixation_label: str,
    tie_epsilon: float,
) -> pd.DataFrame:
    """Compute per-session leader/follower calls from lag-signed means."""
    required_cols = {
        "date",
        "session",
        "monkey_name_m1",
        "monkey_name_m2",
        "cross_correlation",
    }
    missing = required_cols.difference(within_df.columns)
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

        if lead_score > float(tie_epsilon):
            leader_agent = "m1"
            follower_agent = "m2"
            leader_monkey = row["monkey_name_m1"]
            follower_monkey = row["monkey_name_m2"]
        elif lead_score < -float(tie_epsilon):
            leader_agent = "m2"
            follower_agent = "m1"
            leader_monkey = row["monkey_name_m2"]
            follower_monkey = row["monkey_name_m1"]
        else:
            leader_agent = "tie"
            follower_agent = "tie"
            leader_monkey = None
            follower_monkey = None

        rows.append(
            {
                "fixation_label": fixation_label,
                "date": row["date"],
                "session": row["session"],
                "monkey_name_m1": row["monkey_name_m1"],
                "monkey_name_m2": row["monkey_name_m2"],
                "mean_positive_lag_correlation": mean_pos,
                "mean_negative_lag_correlation": mean_neg,
                "lead_score": lead_score,
                "leader_agent": leader_agent,
                "follower_agent": follower_agent,
                "leader_monkey": leader_monkey,
                "follower_monkey": follower_monkey,
            }
        )

    session_df = pd.DataFrame.from_records(rows)
    if session_df.empty:
        return session_df

    session_df = _build_pair_columns(session_df)
    session_df = session_df.sort_values(["pair_key", "date", "session"]).reset_index(drop=True)
    return session_df


def _summarize_by_date(session_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate leader/follower counts per pair and date."""
    if session_df.empty:
        return pd.DataFrame(
            columns=[
                "fixation_label",
                "pair_key",
                "monkey_name_m1",
                "monkey_name_m2",
                "date",
                "n_sessions",
                "m1_leader_sessions",
                "m2_leader_sessions",
                "m1_lead_fraction",
            ]
        )

    summary = (
        session_df.groupby(
            ["fixation_label", "pair_key", "monkey_name_m1", "monkey_name_m2", "date"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_sessions=("leader_agent", "size"),
            m1_leader_sessions=("leader_agent", lambda s: int((s == "m1").sum())),
            m2_leader_sessions=("leader_agent", lambda s: int((s == "m2").sum())),
        )
        .sort_values(["pair_key", "date"])
        .reset_index(drop=True)
    )
    m1_counts = summary["m1_leader_sessions"].to_numpy(dtype=np.float64)
    m2_counts = summary["m2_leader_sessions"].to_numpy(dtype=np.float64)
    denom = m1_counts + m2_counts
    summary["m1_lead_fraction"] = np.where(denom > 0.0, m1_counts / denom, np.nan)
    return summary


def _summarize_total(session_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate leader/follower counts per pair across all sessions."""
    if session_df.empty:
        return pd.DataFrame(
            columns=[
                "fixation_label",
                "pair_key",
                "monkey_name_m1",
                "monkey_name_m2",
                "n_sessions",
                "m1_leader_sessions",
                "m2_leader_sessions",
                "m1_lead_fraction",
            ]
        )

    summary = (
        session_df.groupby(
            ["fixation_label", "pair_key", "monkey_name_m1", "monkey_name_m2"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_sessions=("leader_agent", "size"),
            m1_leader_sessions=("leader_agent", lambda s: int((s == "m1").sum())),
            m2_leader_sessions=("leader_agent", lambda s: int((s == "m2").sum())),
        )
        .sort_values(["pair_key"])
        .reset_index(drop=True)
    )
    m1_counts = summary["m1_leader_sessions"].to_numpy(dtype=np.float64)
    m2_counts = summary["m2_leader_sessions"].to_numpy(dtype=np.float64)
    denom = m1_counts + m2_counts
    summary["m1_lead_fraction"] = np.where(denom > 0.0, m1_counts / denom, np.nan)
    return summary


def _print_summaries(
    *,
    fixation_label: str,
    date_summary_df: pd.DataFrame,
    total_summary_df: pd.DataFrame,
) -> None:
    """Print date-level and total pair summaries."""
    print("\n[leader-follower] -----------------------------------------------")
    print(f"[leader-follower] fixation_label={fixation_label}")
    print("[leader-follower] Date-level summary by monkey pair")
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
                "n_sessions",
                "m1_leader_sessions",
                "m2_leader_sessions",
                "m1_lead_fraction",
            ]
            print(pair_rows[table_cols].to_string(index=False))

    print("\n[leader-follower] -----------------------------------------------")
    print("[leader-follower] Total-session summary by monkey pair")
    print("[leader-follower] -----------------------------------------------")
    if total_summary_df.empty:
        print("[leader-follower] No total-summary rows found.")
    else:
        print(
            total_summary_df[
                [
                    "monkey_name_m1",
                    "monkey_name_m2",
                    "n_sessions",
                    "m1_leader_sessions",
                    "m2_leader_sessions",
                    "m1_lead_fraction",
                ]
            ].to_string(index=False)
        )
    print("[leader-follower] -----------------------------------------------\n")


def run_fix_crosscorr_leader_follower_analysis(
    settings: FixCrossCorrLeaderFollowerSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and save leader-follower summaries from existing xcorr outputs."""
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
    session_df = _compute_session_leader_rows(
        within_df=within_df,
        lags=lags,
        fixation_label=settings.fixation_label,
        tie_epsilon=settings.tie_epsilon,
    )
    date_summary_df = _summarize_by_date(session_df)
    total_summary_df = _summarize_total(session_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    session_out = out_dir / settings.session_output_filename
    date_out = out_dir / settings.date_summary_filename
    total_out = out_dir / settings.total_summary_filename

    session_df.to_csv(session_out, index=False)
    date_summary_df.to_csv(date_out, index=False)
    total_summary_df.to_csv(total_out, index=False)

    print(f"[leader-follower] wrote session-level table: {session_out}")
    print(f"[leader-follower] wrote date-level summary: {date_out}")
    print(f"[leader-follower] wrote total-session summary: {total_out}")
    print(f"[leader-follower] rows: session={len(session_df)} date={len(date_summary_df)} total={len(total_summary_df)}")

    _print_summaries(
        fixation_label=settings.fixation_label,
        date_summary_df=date_summary_df,
        total_summary_df=total_summary_df,
    )

    return session_df, date_summary_df, total_summary_df
