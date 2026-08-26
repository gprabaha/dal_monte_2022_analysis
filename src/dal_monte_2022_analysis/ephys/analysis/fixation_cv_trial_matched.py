"""Trial-count-matched coefficient of variation of the mean fixation PSTH.

``fixation_psth_variability`` computes, per unit and fixation category, the
coefficient of variation of the trial-averaged firing-rate trace **across time
bins**. Interactive-face fixations are far more numerous than the other two
categories (median ~850 trials per unit versus ~170 and ~200), and that is a
confound even though every condition contributes the same number of time bins.

The bin count is not what protects the statistic. Each bin of the mean trace is
itself an average over trials, so its sampling error scales as 1/sqrt(N). Writing
the mean trace as signal plus estimation noise, ``m(t) = s(t) + e(t)``, the
variance taken across bins is

    Var_t[m] ~= Var_t[s] + E[sigma^2] / N

so a condition observed with fewer trials inherits extra across-bin variance --
and therefore a higher CV -- from estimation noise alone. Measured on real data,
holding condition and signal fixed and varying only N, the median CV of
interactive-face traces inflates by roughly 10% at N=400, 50% at N=200 and 80% at
N=150 relative to the full ~570-trial estimate.

This module removes that confound directly: for each unit it repeatedly draws a
**matched number of trials** from every condition, recomputes the mean trace and
its CV, and averages across draws. The matched trial count is the smallest trial
count available for that unit across the compared conditions, so all conditions
are estimated from equally noisy averages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
)
from dal_monte_2022_analysis.core.stats import adjust_pvalues, safe_paired_ttest
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    save_pickle_path,
    scan_processed_paths_for_filename,
)
from dal_monte_2022_analysis.utils.filenames import ensure_filename

CV_CONDITIONS: tuple[str, ...] = ("face_interactive", "face_non_interactive", "object")


@dataclass
class FixationCVTrialMatchedSettings:
    """Configuration for the trial-count-matched CV control."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations_psth_10ms.pkl"
    output_subdir: str = "ephys/psth/fixation_cv_trial_matched"
    unit_output_filename: str = "unit_condition_cv_trial_matched.csv"
    stats_output_filename: str = "within_region_cv_trial_matched_stats.csv"
    inflation_output_filename: str = "cv_trial_count_inflation.csv"
    output_pickle_filename: str = "results.pkl"
    conditions: tuple[str, ...] = field(default_factory=lambda: CV_CONDITIONS)
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    smoothing_sigma_ms: float = 20.0
    bin_size_ms_fallback: float = 10.0
    #: Number of independent subsamples averaged per unit and condition.
    n_draws: int = 25
    #: Units whose smallest condition has fewer trials than this are dropped: a
    #: CV estimated from a handful of trials is dominated by noise in every
    #: condition and adds nothing but variance to the paired test.
    min_matched_trials: int = 40
    #: Trial counts used for the inflation curve that documents the confound.
    inflation_trial_counts: tuple[int, ...] = (150, 200, 300, 400)
    inflation_min_trials: int = 500
    inflation_n_draws: int = 8
    random_seed: int = 20240517
    pvalue_correction: str = "fdr_bh"
    alpha: float = 0.05
    use_parallel: bool = True
    max_procs: int = 16


def _norm_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _trial_condition(
    row: pd.Series,
    *,
    settings: FixationCVTrialMatchedSettings,
) -> Optional[str]:
    category = _norm_token(row.get("fixation_category"))
    if category == _norm_token(settings.object_label):
        return "object"
    if category != _norm_token(settings.face_label):
        return None
    is_interactive = row.get("is_interactive")
    if is_interactive is not None and not pd.isna(is_interactive):
        interactive = bool(_as_bool(is_interactive, settings.interactive_label))
    else:
        interactive = _norm_token(row.get("interactive_state")) == _norm_token(
            settings.interactive_label
        )
    return "face_interactive" if interactive else "face_non_interactive"


def coefficient_of_variation_across_bins(
    rates_hz: np.ndarray,
    *,
    sigma_bins: float,
) -> float:
    """CV across time bins of the trial-averaged, smoothed firing-rate trace."""
    mean_trace = np.asarray(rates_hz, dtype=float).mean(axis=0)
    if sigma_bins > 0:
        mean_trace = gaussian_filter1d(mean_trace, sigma_bins, mode="nearest")
    mean_value = float(np.mean(mean_trace))
    if not np.isfinite(mean_value) or mean_value <= 0.0:
        return np.nan
    return float(np.std(mean_trace, ddof=1) / mean_value)


def _subsampled_cv(
    rates_hz: np.ndarray,
    n_trials: int,
    *,
    sigma_bins: float,
    n_draws: int,
    rng: np.random.Generator,
) -> float:
    """Mean CV over ``n_draws`` random subsamples of ``n_trials`` trials."""
    total = int(rates_hz.shape[0])
    if total < n_trials:
        return np.nan
    if total == n_trials:
        return coefficient_of_variation_across_bins(rates_hz, sigma_bins=sigma_bins)
    values = [
        coefficient_of_variation_across_bins(
            rates_hz[rng.choice(total, size=n_trials, replace=False)],
            sigma_bins=sigma_bins,
        )
        for _ in range(int(n_draws))
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def _date_worker(args) -> tuple[list[dict], list[dict]]:
    paths, date, settings = args
    frames: list[pd.DataFrame] = []
    bin_size_ms = float(settings.bin_size_ms_fallback)
    for path in paths:
        blob = load_pickle_path(path)
        if isinstance(blob, dict):
            trials = blob.get("trials")
            meta = blob.get("meta", {}) or {}
            if isinstance(meta, dict) and meta.get("bin_size_ms") is not None:
                bin_size_ms = float(meta["bin_size_ms"])
        else:
            trials = blob
        if isinstance(trials, pd.DataFrame) and not trials.empty:
            frames.append(trials)
    if not frames:
        return ([], [])

    trials = pd.concat(frames, ignore_index=True)
    if "psth_counts" not in trials.columns or "unit_uuid" not in trials.columns:
        return ([], [])
    trials["_condition"] = trials.apply(
        lambda row: _trial_condition(row, settings=settings), axis=1
    )
    trials = trials.loc[trials["_condition"].notna()]
    if trials.empty:
        return ([], [])

    bin_s = bin_size_ms / 1000.0
    sigma_bins = float(settings.smoothing_sigma_ms) / max(bin_size_ms, 1e-9)

    matched_rows: list[dict] = []
    inflation_rows: list[dict] = []
    for unit_uuid, unit_trials in trials.groupby("unit_uuid", sort=False):
        by_condition: dict[str, np.ndarray] = {}
        for condition in settings.conditions:
            subset = unit_trials.loc[unit_trials["_condition"] == condition]
            if subset.empty:
                continue
            by_condition[condition] = np.vstack(subset["psth_counts"].tolist()) / bin_s
        if len(by_condition) < len(settings.conditions):
            continue

        seed = abs(hash((str(date), str(unit_uuid), int(settings.random_seed)))) % (2**32)
        rng = np.random.default_rng(seed)

        matched_n = int(min(rates.shape[0] for rates in by_condition.values()))
        region = _as_optional_str(unit_trials["region"].iloc[0]) or "unknown"
        row: dict = {
            "unit_key": f"{date}|{unit_uuid}",
            "date": str(date),
            "unit_uuid": str(unit_uuid),
            "region": str(region).strip().lower(),
            "matched_n_trials": matched_n,
            "meets_min_matched_trials": bool(matched_n >= int(settings.min_matched_trials)),
        }
        for condition, rates in by_condition.items():
            row[f"{condition}_n_trials"] = int(rates.shape[0])
            row[f"{condition}_cv_full"] = coefficient_of_variation_across_bins(
                rates, sigma_bins=sigma_bins
            )
            row[f"{condition}_cv_matched"] = _subsampled_cv(
                rates,
                matched_n,
                sigma_bins=sigma_bins,
                n_draws=settings.n_draws,
                rng=rng,
            )
        matched_rows.append(row)

        # Inflation curve: same condition, same signal, only trial count varies.
        interactive = by_condition.get("face_interactive")
        if interactive is not None and interactive.shape[0] >= int(settings.inflation_min_trials):
            full_cv = coefficient_of_variation_across_bins(interactive, sigma_bins=sigma_bins)
            for n_trials in settings.inflation_trial_counts:
                if interactive.shape[0] < n_trials:
                    continue
                inflation_rows.append(
                    {
                        "unit_key": f"{date}|{unit_uuid}",
                        "region": str(region).strip().lower(),
                        "n_trials_available": int(interactive.shape[0]),
                        "n_trials_used": int(n_trials),
                        "cv_full": full_cv,
                        "cv_subsampled": _subsampled_cv(
                            interactive,
                            int(n_trials),
                            sigma_bins=sigma_bins,
                            n_draws=settings.inflation_n_draws,
                            rng=rng,
                        ),
                    }
                )
    return (matched_rows, inflation_rows)


def _build_within_region_stats(
    unit_df: pd.DataFrame,
    settings: FixationCVTrialMatchedSettings,
) -> pd.DataFrame:
    """Paired comparison of matched CV between fixation categories, per region."""
    rows: list[dict] = []
    usable = unit_df.loc[unit_df["meets_min_matched_trials"].astype(bool)]
    for region, region_df in usable.groupby("region", sort=False):
        for variant in ("cv_full", "cv_matched"):
            for index, condition_a in enumerate(settings.conditions):
                for condition_b in settings.conditions[index + 1 :]:
                    column_a = f"{condition_a}_{variant}"
                    column_b = f"{condition_b}_{variant}"
                    paired = region_df.loc[:, [column_a, column_b]].dropna()
                    if len(paired) < 3:
                        continue
                    statistic, p_value, n_pairs = safe_paired_ttest(
                        paired[column_a].to_numpy(dtype=float),
                        paired[column_b].to_numpy(dtype=float),
                    )
                    rows.append(
                        {
                            "region": region,
                            "variant": variant,
                            "condition_a": condition_a,
                            "condition_b": condition_b,
                            "condition_pair": f"{condition_a}__vs__{condition_b}",
                            "n_units_paired": int(n_pairs),
                            "median_a": float(paired[column_a].median()),
                            "median_b": float(paired[column_b].median()),
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                        }
                    )
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df
    # Correct within variant so the matched family is not penalised by the
    # uncorrected family it is being compared against.
    stats_df["p_value_adjusted"] = np.nan
    for variant, group in stats_df.groupby("variant"):
        stats_df.loc[group.index, "p_value_adjusted"] = adjust_pvalues(
            group["p_value"].to_numpy(dtype=float), settings.pvalue_correction
        )
    stats_df["pvalue_correction"] = settings.pvalue_correction
    stats_df["alpha"] = float(settings.alpha)
    stats_df["significant_adjusted"] = stats_df["p_value_adjusted"] < float(settings.alpha)
    return stats_df


def run_fixation_cv_trial_matched_analysis(
    settings: FixationCVTrialMatchedSettings,
    *,
    dates: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Recompute condition CVs from trial-count-matched subsamples."""
    cfg = load_config(settings.cfg_path)
    rows = scan_processed_paths_for_filename(
        cfg,
        settings.trial_input_modality,
        filename=ensure_filename(settings.trial_input_filename, ".pkl"),
        dates=dates,
    )
    by_date: dict[str, list[Path]] = {}
    for row in rows:
        by_date.setdefault(str(row["date"]), []).append(Path(row["path"]))

    tasks = [(paths, date, settings) for date, paths in sorted(by_date.items())]
    results = run_tasks(
        _date_worker,
        tasks,
        desc="Trial-matched CV",
        unit="date",
        use_parallel=settings.use_parallel,
        max_procs=settings.max_procs,
    )

    matched_rows = [row for matched, _ in results for row in matched]
    inflation_rows = [row for _, inflation in results for row in inflation]
    unit_df = pd.DataFrame(matched_rows)
    inflation_df = pd.DataFrame(inflation_rows)
    if not inflation_df.empty:
        inflation_df["inflation_ratio"] = (
            inflation_df["cv_subsampled"] / inflation_df["cv_full"]
        )
    stats_df = _build_within_region_stats(unit_df, settings) if not unit_df.empty else pd.DataFrame()

    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    unit_df.to_csv(out_root / ensure_filename(settings.unit_output_filename, ".csv"), index=False)
    stats_df.to_csv(out_root / ensure_filename(settings.stats_output_filename, ".csv"), index=False)
    inflation_df.to_csv(
        out_root / ensure_filename(settings.inflation_output_filename, ".csv"), index=False
    )

    result = {
        "meta": {
            "conditions": list(settings.conditions),
            "n_draws": int(settings.n_draws),
            "min_matched_trials": int(settings.min_matched_trials),
            "smoothing_sigma_ms": float(settings.smoothing_sigma_ms),
            "inflation_trial_counts": list(settings.inflation_trial_counts),
            "n_units": int(len(unit_df)),
        },
        "unit_cv": unit_df,
        "within_region_stats": stats_df,
        "inflation": inflation_df,
    }
    save_pickle_path(result, out_root / ensure_filename(settings.output_pickle_filename, ".pkl"))
    return result
