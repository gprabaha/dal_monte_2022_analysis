"""Compare three-way fixation-response compositions across brain regions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


REL_COLS = (
    "relative_face_interactive",
    "relative_face_non_interactive",
    "relative_object",
)
_ALLOWED_CORRECTIONS = {"none", "bonferroni", "holm", "fdr_bh"}


@dataclass
class FixationThreeWayRegionComparisonSettings:
    """Configuration for region-comparison testing of three-way compositions."""

    cfg_path: str
    input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    condition_summary_filename: str = "condition_window_means.csv"
    output_subdir: str = "ephys/psth/fixation_psth_selectivity_region_comparison"
    pairwise_summary_filename: str = "pairwise_region_comparisons.csv"
    window_summary_filename: str = "window_region_comparisons.csv"
    output_pickle_filename: str = "results.pkl"
    min_units_per_region: int = 5
    min_regions_per_window: int = 2
    n_permutations: int = 1000
    random_seed: int = 42
    pvalue_correction: str = "fdr_bh"
    alpha: float = 0.05
    require_all_conditions_observed: bool = True
    require_meets_min_trials: bool = False
    pseudo_count: float = 1e-6


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return float(value) != 0.0
    if value is None:
        return False
    token = str(value).strip().lower()
    return token in {"1", "true", "t", "yes", "y"}


def _load_condition_summary_df(settings: FixationThreeWayRegionComparisonSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.condition_summary_filename, ".csv")
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Three-way condition summary CSV not found: {in_path}")
    df = pd.read_csv(in_path)
    if df.empty:
        return df

    missing_rel = [col for col in REL_COLS if col not in df.columns]
    if missing_rel:
        raise ValueError(
            f"Missing required relative-composition columns in condition summary: {missing_rel}"
        )
    if "region" not in df.columns:
        df["region"] = "unknown"
    else:
        df["region"] = df["region"].fillna("unknown").astype(str).replace({"": "unknown"})
    if "window_name" not in df.columns:
        raise ValueError("condition summary missing required column 'window_name'.")

    if "unit_key" not in df.columns and {"date", "unit_uuid"}.issubset(df.columns):
        df["unit_key"] = df["date"].astype(str) + "|" + df["unit_uuid"].astype(str)
    return df


def _window_order(df: pd.DataFrame) -> list[str]:
    if "window_start_ms" not in df.columns:
        return sorted(df["window_name"].astype(str).unique().tolist())
    meta = (
        df.loc[:, ["window_name", "window_start_ms", "window_stop_ms"]]
        .dropna(subset=["window_name"])
        .copy()
    )
    if meta.empty:
        return sorted(df["window_name"].astype(str).unique().tolist())
    meta["window_name"] = meta["window_name"].astype(str)
    meta["window_start_ms"] = pd.to_numeric(meta["window_start_ms"], errors="coerce")
    meta["window_stop_ms"] = pd.to_numeric(meta["window_stop_ms"], errors="coerce")
    grouped = (
        meta.groupby("window_name", as_index=False)
        .agg(
            window_start_ms=("window_start_ms", "median"),
            window_stop_ms=("window_stop_ms", "median"),
        )
        .sort_values(["window_start_ms", "window_stop_ms", "window_name"], na_position="last")
    )
    return grouped["window_name"].astype(str).tolist()


def _ilr_transform(compositions: np.ndarray, *, pseudo_count: float) -> np.ndarray:
    comp = np.asarray(compositions, dtype=float)
    if comp.ndim != 2 or comp.shape[1] != 3:
        raise ValueError("Expected composition array with shape (n, 3).")
    eps = max(float(pseudo_count), 1e-12)
    comp = np.clip(comp, eps, None)
    comp = comp / np.sum(comp, axis=1, keepdims=True)

    x1 = comp[:, 0]
    x2 = comp[:, 1]
    x3 = comp[:, 2]
    ilr1 = np.sqrt(0.5) * np.log(x1 / x2)
    ilr2 = np.sqrt(2.0 / 3.0) * np.log(np.sqrt(x1 * x2) / x3)
    return np.column_stack([ilr1, ilr2])


def _pseudo_f_stat(X: np.ndarray, labels: np.ndarray) -> float:
    arr = np.asarray(X, dtype=float)
    lab = np.asarray(labels)
    if arr.ndim != 2 or arr.shape[0] != lab.shape[0]:
        return np.nan

    groups = np.unique(lab)
    n = int(arr.shape[0])
    g = int(groups.size)
    if g < 2 or n <= g:
        return np.nan

    grand = np.mean(arr, axis=0)
    ss_between = 0.0
    ss_within = 0.0
    for group in groups:
        mask = lab == group
        Xi = arr[mask]
        if Xi.size == 0:
            continue
        mean_i = np.mean(Xi, axis=0)
        ss_between += float(Xi.shape[0]) * float(np.sum((mean_i - grand) ** 2))
        ss_within += float(np.sum((Xi - mean_i) ** 2))

    df_between = float(g - 1)
    df_within = float(n - g)
    if df_between <= 0.0 or df_within <= 0.0:
        return np.nan
    if ss_within <= 0.0:
        return np.inf
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return float(ms_between / ms_within) if ms_within > 0.0 else np.inf


def _permutation_pseudo_f_test(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = _pseudo_f_stat(X, labels)
    if not np.isfinite(observed):
        return observed, np.nan
    if int(n_permutations) <= 0:
        return observed, np.nan

    hits = 0
    labels_arr = np.asarray(labels).copy()
    for _ in range(int(n_permutations)):
        permuted = rng.permutation(labels_arr)
        stat = _pseudo_f_stat(X, permuted)
        if np.isfinite(stat) and stat >= observed - 1e-12:
            hits += 1
    p_value = (float(hits) + 1.0) / (float(n_permutations) + 1.0)
    return observed, float(p_value)


def _adjust_pvalues(p_values: Sequence[float], method: str) -> np.ndarray:
    if method not in _ALLOWED_CORRECTIONS:
        raise ValueError(
            f"Unsupported p-value correction '{method}'. "
            f"Expected one of: {sorted(_ALLOWED_CORRECTIONS)}"
        )
    vec = np.asarray(p_values, dtype=float).reshape(-1)
    out = np.full(vec.shape, np.nan, dtype=float)
    finite = np.isfinite(vec)
    if not np.any(finite):
        return out
    vals = vec[finite]
    m = int(vals.size)

    if method == "none":
        out[finite] = vals
        return out

    if method == "bonferroni":
        out[finite] = np.minimum(vals * float(m), 1.0)
        return out

    order = np.argsort(vals)
    ranked = vals[order]

    if method == "holm":
        holm_ranked = (m - np.arange(m, dtype=float)) * ranked
        holm_ranked = np.maximum.accumulate(holm_ranked)
        holm_ranked = np.clip(holm_ranked, 0.0, 1.0)
        adjusted = np.empty(m, dtype=float)
        adjusted[order] = holm_ranked
        out[finite] = adjusted
        return out

    # Benjamini-Hochberg FDR
    bh_ranked = ranked * (float(m) / np.arange(1.0, float(m) + 1.0))
    bh_ranked = np.minimum.accumulate(bh_ranked[::-1])[::-1]
    bh_ranked = np.clip(bh_ranked, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = bh_ranked
    out[finite] = adjusted
    return out


def run_fixation_three_way_region_comparison(
    settings: FixationThreeWayRegionComparisonSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    windows: Optional[Sequence[str]] = None,
) -> dict:
    """Run region-comparison tests for three-way fixation compositions."""
    df = _load_condition_summary_df(settings)
    if df.empty:
        print("[analysis] no three-way condition summary rows found for region comparison")
        return {"pairwise_summary": pd.DataFrame(), "window_summary": pd.DataFrame()}

    if settings.require_all_conditions_observed and "all_conditions_observed" in df.columns:
        df = df.loc[df["all_conditions_observed"].map(_as_bool)].copy()
    if settings.require_meets_min_trials and "meets_min_trials" in df.columns:
        df = df.loc[df["meets_min_trials"].map(_as_bool)].copy()
    if df.empty:
        print("[analysis] no rows remain after required-condition filters")
        return {"pairwise_summary": pd.DataFrame(), "window_summary": pd.DataFrame()}

    rel = df.loc[:, list(REL_COLS)].apply(pd.to_numeric, errors="coerce")
    valid = np.all(np.isfinite(rel.to_numpy(dtype=float)), axis=1)
    df = df.loc[valid].copy()
    if df.empty:
        print("[analysis] no rows with valid relative composition values")
        return {"pairwise_summary": pd.DataFrame(), "window_summary": pd.DataFrame()}

    if regions is not None:
        allowed_regions = {str(region) for region in regions}
        df = df.loc[df["region"].astype(str).isin(allowed_regions)].copy()
    if windows is not None:
        allowed_windows = {str(window) for window in windows}
        df = df.loc[df["window_name"].astype(str).isin(allowed_windows)].copy()
    if df.empty:
        print("[analysis] no rows remain after region/window filtering")
        return {"pairwise_summary": pd.DataFrame(), "window_summary": pd.DataFrame()}

    window_order = _window_order(df)
    rng = np.random.default_rng(int(settings.random_seed))
    pair_rows: list[dict] = []
    window_rows: list[dict] = []

    for window_name in window_order:
        df_win = df.loc[df["window_name"].astype(str) == str(window_name)].copy()
        if df_win.empty:
            continue
        counts = (
            df_win.groupby("region", dropna=False)["unit_key"]
            .nunique()
            .rename("n_units")
            .reset_index()
        )
        counts["region"] = counts["region"].astype(str)
        eligible_regions = sorted(
            counts.loc[counts["n_units"] >= int(settings.min_units_per_region), "region"]
            .astype(str)
            .tolist()
        )
        df_win = df_win.loc[df_win["region"].astype(str).isin(set(eligible_regions))].copy()

        meta_start = np.nan
        meta_stop = np.nan
        if "window_start_ms" in df_win.columns:
            meta_start = float(pd.to_numeric(df_win["window_start_ms"], errors="coerce").median())
        if "window_stop_ms" in df_win.columns:
            meta_stop = float(pd.to_numeric(df_win["window_stop_ms"], errors="coerce").median())

        n_regions = int(len(eligible_regions))
        n_units_total = int(df_win["unit_key"].astype(str).nunique())
        n_pairs = int(max(0, n_regions * (n_regions - 1) // 2))
        global_f = np.nan
        global_p = np.nan

        if n_regions >= int(settings.min_regions_per_window):
            X = _ilr_transform(
                df_win.loc[:, list(REL_COLS)].to_numpy(dtype=float),
                pseudo_count=float(settings.pseudo_count),
            )
            labels = df_win["region"].astype(str).to_numpy()
            global_f, global_p = _permutation_pseudo_f_test(
                X,
                labels,
                n_permutations=int(settings.n_permutations),
                rng=rng,
            )

            raw_pair_indices: list[int] = []
            raw_pair_pvals: list[float] = []
            for region_a, region_b in combinations(eligible_regions, 2):
                mask = df_win["region"].astype(str).isin({region_a, region_b}).to_numpy(dtype=bool)
                sub = df_win.loc[mask].copy()
                Xa = _ilr_transform(
                    sub.loc[sub["region"].astype(str) == region_a, list(REL_COLS)].to_numpy(dtype=float),
                    pseudo_count=float(settings.pseudo_count),
                )
                Xb = _ilr_transform(
                    sub.loc[sub["region"].astype(str) == region_b, list(REL_COLS)].to_numpy(dtype=float),
                    pseudo_count=float(settings.pseudo_count),
                )
                if Xa.size == 0 or Xb.size == 0:
                    continue
                X_pair = np.vstack([Xa, Xb])
                labels_pair = np.asarray(
                    [region_a] * int(Xa.shape[0]) + [region_b] * int(Xb.shape[0]),
                    dtype=object,
                )
                pair_f, pair_p = _permutation_pseudo_f_test(
                    X_pair,
                    labels_pair,
                    n_permutations=int(settings.n_permutations),
                    rng=rng,
                )

                mean_a = np.mean(Xa, axis=0)
                mean_b = np.mean(Xb, axis=0)
                disp_a = float(np.mean(np.linalg.norm(Xa - mean_a, axis=1)))
                disp_b = float(np.mean(np.linalg.norm(Xb - mean_b, axis=1)))
                row = {
                    "window_name": str(window_name),
                    "window_start_ms": meta_start,
                    "window_stop_ms": meta_stop,
                    "region_a": str(region_a),
                    "region_b": str(region_b),
                    "n_units_a": int(Xa.shape[0]),
                    "n_units_b": int(Xb.shape[0]),
                    "pseudo_f": pair_f,
                    "p_value": pair_p,
                    "centroid_distance_ilr": float(np.linalg.norm(mean_a - mean_b)),
                    "mean_dispersion_a_ilr": disp_a,
                    "mean_dispersion_b_ilr": disp_b,
                    "dispersion_difference_ilr": float(disp_a - disp_b),
                    "n_permutations": int(settings.n_permutations),
                }
                raw_pair_indices.append(len(pair_rows))
                raw_pair_pvals.append(pair_p)
                pair_rows.append(row)

            if raw_pair_indices:
                adj = _adjust_pvalues(raw_pair_pvals, settings.pvalue_correction)
                for idx_local, idx_global in enumerate(raw_pair_indices):
                    p_adj = float(adj[idx_local]) if np.isfinite(adj[idx_local]) else np.nan
                    pair_rows[idx_global]["p_value_adjusted"] = p_adj
                    pair_rows[idx_global]["pvalue_correction"] = settings.pvalue_correction
                    pair_rows[idx_global]["alpha"] = float(settings.alpha)
                    pair_rows[idx_global]["significant"] = bool(
                        np.isfinite(p_adj) and p_adj < float(settings.alpha)
                    )

        window_rows.append(
            {
                "window_name": str(window_name),
                "window_start_ms": meta_start,
                "window_stop_ms": meta_stop,
                "n_regions_tested": n_regions,
                "n_units_total": n_units_total,
                "n_pairs_tested": n_pairs,
                "global_pseudo_f": global_f,
                "global_p_value": global_p,
                "global_p_value_adjusted": np.nan,
                "global_significant": False,
                "global_pvalue_correction": settings.pvalue_correction,
                "alpha": float(settings.alpha),
                "n_permutations": int(settings.n_permutations),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    window_df = pd.DataFrame(window_rows)

    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            ["window_start_ms", "window_stop_ms", "window_name", "region_a", "region_b"],
            na_position="last",
        ).reset_index(drop=True)
    if not window_df.empty:
        adj_global = _adjust_pvalues(window_df["global_p_value"].to_numpy(dtype=float), settings.pvalue_correction)
        window_df["global_p_value_adjusted"] = adj_global
        window_df["global_significant"] = (
            pd.Series(adj_global).apply(lambda p: bool(np.isfinite(p) and p < float(settings.alpha))).to_numpy(dtype=bool)
        )
        window_df = window_df.sort_values(
            ["window_start_ms", "window_stop_ms", "window_name"],
            na_position="last",
        ).reset_index(drop=True)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    pair_csv = out_root / ensure_filename(settings.pairwise_summary_filename, ".csv")
    window_csv = out_root / ensure_filename(settings.window_summary_filename, ".csv")
    result_pkl = out_root / ensure_filename(settings.output_pickle_filename, ".pkl")

    pair_df.to_csv(pair_csv, index=False)
    window_df.to_csv(window_csv, index=False)

    result = {
        "meta": {
            "alpha": float(settings.alpha),
            "pvalue_correction": settings.pvalue_correction,
            "n_permutations": int(settings.n_permutations),
            "min_units_per_region": int(settings.min_units_per_region),
            "min_regions_per_window": int(settings.min_regions_per_window),
            "pseudo_count": float(settings.pseudo_count),
        },
        "pairwise_summary": pair_df,
        "window_summary": window_df,
    }
    save_pickle_path(result, result_pkl)
    return result

