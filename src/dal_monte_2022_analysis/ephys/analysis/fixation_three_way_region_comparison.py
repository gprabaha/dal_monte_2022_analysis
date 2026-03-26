"""Compare three-way fixation-response compositions across brain regions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.stats import (
    adjust_pvalues,
    normalize_pvalue_correction,
    permutation_label_statistic_test,
    permutation_mean_difference_test,
)
from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


REL_COLS = (
    "relative_face_interactive",
    "relative_face_non_interactive",
    "relative_object",
)
TRIANGLE_HEIGHT = float(np.sqrt(3.0) / 2.0)
TRIANGLE_VERTICES = np.asarray(
    [
        [0.5, TRIANGLE_HEIGHT],  # interactive face
        [0.0, 0.0],              # non-interactive face
        [1.0, 0.0],              # object
    ],
    dtype=float,
)


@dataclass
class FixationThreeWayRegionComparisonSettings:
    """Configuration for region-comparison testing of three-way compositions."""

    cfg_path: str
    input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    condition_summary_filename: str = "condition_window_means.csv"
    unit_summary_filename: str = "unit_selectivity.csv"
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
    require_selective_units: bool = False
    pseudo_count: float = 1e-6
    alignment_cosine_threshold: float = 0.95


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


def _load_unit_summary_df(settings: FixationThreeWayRegionComparisonSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.unit_summary_filename, ".csv")
    )
    if not in_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(in_path)
    if df.empty or "unit_key" not in df.columns or "is_selective_unit" not in df.columns:
        return pd.DataFrame()
    out = df.loc[:, ["unit_key", "is_selective_unit"]].copy()
    out["unit_key"] = out["unit_key"].astype(str)
    out["is_selective_unit"] = out["is_selective_unit"].map(_as_bool)
    out = out.drop_duplicates(subset=["unit_key"], keep="last")
    return out


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


def _triangle_xy_from_compositions(compositions: np.ndarray) -> np.ndarray:
    comp = np.asarray(compositions, dtype=float)
    if comp.ndim != 2 or comp.shape[1] != 3:
        raise ValueError("Expected composition array with shape (n, 3).")
    rel_int = comp[:, 0]
    rel_obj = comp[:, 2]
    x = rel_obj + 0.5 * rel_int
    y = TRIANGLE_HEIGHT * rel_int
    return np.column_stack([x, y])


def _safe_covariance_2d(points_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(points_xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return np.full((2, 2), np.nan, dtype=float)
    cov = np.cov(arr, rowvar=False)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2):
        return np.full((2, 2), np.nan, dtype=float)
    return cov


def _anisotropy_index(points_xy: np.ndarray) -> float:
    cov = _safe_covariance_2d(points_xy)
    if not np.all(np.isfinite(cov)):
        return np.nan
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.asarray(eigvals, dtype=float))
    if eigvals.size != 2:
        return np.nan
    l1 = float(eigvals[1])
    l2 = float(max(eigvals[0], 0.0))
    denom = l1 + l2
    if not np.isfinite(denom) or denom <= 0.0:
        return np.nan
    return float((l1 - l2) / denom)


def _alignment_scores_to_vertex_axes(points_xy: np.ndarray, centroid_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    centroid = np.asarray(centroid_xy, dtype=float).reshape(2)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.array([], dtype=float)

    axis_vecs = TRIANGLE_VERTICES - centroid[None, :]
    axis_norms = np.linalg.norm(axis_vecs, axis=1)
    valid_axes = axis_norms > 1e-12
    if not np.any(valid_axes):
        return np.array([], dtype=float)
    axis_unit = axis_vecs[valid_axes] / axis_norms[valid_axes, None]

    centered = points - centroid[None, :]
    norms = np.linalg.norm(centered, axis=1)
    valid_points = norms > 1e-12
    if not np.any(valid_points):
        return np.array([], dtype=float)

    unit_points = centered[valid_points] / norms[valid_points, None]
    cosines = unit_points @ axis_unit.T
    max_cos = np.max(cosines, axis=1)
    return np.asarray(max_cos, dtype=float)


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

    unit_summary_df = _load_unit_summary_df(settings)
    if not unit_summary_df.empty and "unit_key" in df.columns:
        df = df.merge(unit_summary_df, on="unit_key", how="left")
    if "is_selective_unit" not in df.columns:
        df["is_selective_unit"] = False
    df["is_selective_unit"] = df["is_selective_unit"].map(_as_bool)
    if settings.require_selective_units:
        df = df.loc[df["is_selective_unit"]].copy()
        if df.empty:
            print("[analysis] no rows remain after selective-unit filtering")
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
    correction = normalize_pvalue_correction(settings.pvalue_correction)
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
        global_dispersion_f = np.nan
        global_dispersion_p = np.nan
        global_alignment_f = np.nan
        global_alignment_p = np.nan

        region_payload: dict[str, dict[str, object]] = {}
        for region in eligible_regions:
            sub = df_win.loc[df_win["region"].astype(str) == str(region)].copy()
            comp = sub.loc[:, list(REL_COLS)].to_numpy(dtype=float)
            X_ilr = _ilr_transform(comp, pseudo_count=float(settings.pseudo_count))
            xy = _triangle_xy_from_compositions(comp)
            if xy.size == 0:
                centroid_xy = np.array([np.nan, np.nan], dtype=float)
                radial = np.array([], dtype=float)
                alignment_scores = np.array([], dtype=float)
            else:
                centroid_xy = np.mean(xy, axis=0)
                radial = np.linalg.norm(xy - centroid_xy[None, :], axis=1)
                alignment_scores = _alignment_scores_to_vertex_axes(xy, centroid_xy)
            region_payload[str(region)] = {
                "X_ilr": X_ilr,
                "xy": xy,
                "centroid_xy": centroid_xy,
                "radial": np.asarray(radial, dtype=float),
                "alignment_scores": np.asarray(alignment_scores, dtype=float),
                "anisotropy": _anisotropy_index(xy),
            }

        if n_regions >= int(settings.min_regions_per_window):
            X_parts: list[np.ndarray] = []
            label_parts: list[np.ndarray] = []
            for region in eligible_regions:
                Xr = np.asarray(region_payload[str(region)]["X_ilr"], dtype=float)
                if Xr.size == 0:
                    continue
                X_parts.append(Xr)
                label_parts.append(np.asarray([str(region)] * int(Xr.shape[0]), dtype=object))
            if X_parts:
                X = np.vstack(X_parts)
                labels = np.concatenate(label_parts, axis=0)
            else:
                X = np.zeros((0, 2), dtype=float)
                labels = np.asarray([], dtype=object)
            global_f, global_p = permutation_label_statistic_test(
                X,
                labels,
                n_permutations=int(settings.n_permutations),
                rng=rng,
                statistic_fn=_pseudo_f_stat,
            )

            disp_vals: list[float] = []
            disp_labels: list[str] = []
            align_vals: list[float] = []
            align_labels: list[str] = []
            for region in eligible_regions:
                payload = region_payload[str(region)]
                radial = np.asarray(payload["radial"], dtype=float)
                radial = radial[np.isfinite(radial)]
                if radial.size > 0:
                    disp_vals.extend(radial.tolist())
                    disp_labels.extend([str(region)] * int(radial.size))

                align = np.asarray(payload["alignment_scores"], dtype=float)
                align = align[np.isfinite(align)]
                if align.size > 0:
                    align_vals.extend(align.tolist())
                    align_labels.extend([str(region)] * int(align.size))

            if len(set(disp_labels)) >= int(settings.min_regions_per_window):
                global_dispersion_f, global_dispersion_p = permutation_label_statistic_test(
                    np.asarray(disp_vals, dtype=float).reshape(-1, 1),
                    np.asarray(disp_labels, dtype=object),
                    n_permutations=int(settings.n_permutations),
                    rng=rng,
                    statistic_fn=_pseudo_f_stat,
                )
            if len(set(align_labels)) >= int(settings.min_regions_per_window):
                global_alignment_f, global_alignment_p = permutation_label_statistic_test(
                    np.asarray(align_vals, dtype=float).reshape(-1, 1),
                    np.asarray(align_labels, dtype=object),
                    n_permutations=int(settings.n_permutations),
                    rng=rng,
                    statistic_fn=_pseudo_f_stat,
                )

            raw_pair_indices: list[int] = []
            raw_pair_pvals: list[float] = []
            raw_pair_dispersion_pvals: list[float] = []
            raw_pair_alignment_pvals: list[float] = []
            for region_a, region_b in combinations(eligible_regions, 2):
                payload_a = region_payload[str(region_a)]
                payload_b = region_payload[str(region_b)]
                Xa = np.asarray(payload_a["X_ilr"], dtype=float)
                Xb = np.asarray(payload_b["X_ilr"], dtype=float)
                if Xa.size == 0 or Xb.size == 0:
                    continue
                X_pair = np.vstack([Xa, Xb])
                labels_pair = np.asarray(
                    [region_a] * int(Xa.shape[0]) + [region_b] * int(Xb.shape[0]),
                    dtype=object,
                )
                pair_f, pair_p = permutation_label_statistic_test(
                    X_pair,
                    labels_pair,
                    n_permutations=int(settings.n_permutations),
                    rng=rng,
                    statistic_fn=_pseudo_f_stat,
                )

                mean_a = np.mean(Xa, axis=0)
                mean_b = np.mean(Xb, axis=0)
                radial_a = np.asarray(payload_a["radial"], dtype=float)
                radial_b = np.asarray(payload_b["radial"], dtype=float)
                radial_a = radial_a[np.isfinite(radial_a)]
                radial_b = radial_b[np.isfinite(radial_b)]
                disp_a = float(np.mean(radial_a)) if radial_a.size > 0 else np.nan
                disp_b = float(np.mean(radial_b)) if radial_b.size > 0 else np.nan
                disp_diff, disp_p = permutation_mean_difference_test(
                    radial_a,
                    radial_b,
                    n_permutations=int(settings.n_permutations),
                    rng=rng,
                )

                align_a = np.asarray(payload_a["alignment_scores"], dtype=float)
                align_b = np.asarray(payload_b["alignment_scores"], dtype=float)
                align_a = align_a[np.isfinite(align_a)]
                align_b = align_b[np.isfinite(align_b)]
                mean_align_a = float(np.mean(align_a)) if align_a.size > 0 else np.nan
                mean_align_b = float(np.mean(align_b)) if align_b.size > 0 else np.nan
                align_diff, align_p = permutation_mean_difference_test(
                    align_a,
                    align_b,
                    n_permutations=int(settings.n_permutations),
                    rng=rng,
                )

                frac_aligned_a = (
                    float(np.mean(align_a >= float(settings.alignment_cosine_threshold)))
                    if align_a.size > 0
                    else np.nan
                )
                frac_aligned_b = (
                    float(np.mean(align_b >= float(settings.alignment_cosine_threshold)))
                    if align_b.size > 0
                    else np.nan
                )

                anisotropy_a = float(payload_a["anisotropy"]) if np.isfinite(payload_a["anisotropy"]) else np.nan
                anisotropy_b = float(payload_b["anisotropy"]) if np.isfinite(payload_b["anisotropy"]) else np.nan
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
                    "dispersion_difference_p_value": disp_p,
                    "mean_alignment_a_to_vertex_axes": mean_align_a,
                    "mean_alignment_b_to_vertex_axes": mean_align_b,
                    "alignment_difference_to_vertex_axes": align_diff,
                    "alignment_difference_p_value": align_p,
                    "fraction_aligned_a_to_vertex_axes": frac_aligned_a,
                    "fraction_aligned_b_to_vertex_axes": frac_aligned_b,
                    "fraction_aligned_difference_to_vertex_axes": float(frac_aligned_a - frac_aligned_b),
                    "alignment_cosine_threshold": float(settings.alignment_cosine_threshold),
                    "anisotropy_index_a": anisotropy_a,
                    "anisotropy_index_b": anisotropy_b,
                    "anisotropy_index_difference": float(anisotropy_a - anisotropy_b),
                    "n_permutations": int(settings.n_permutations),
                }
                raw_pair_indices.append(len(pair_rows))
                raw_pair_pvals.append(pair_p)
                raw_pair_dispersion_pvals.append(disp_p)
                raw_pair_alignment_pvals.append(align_p)
                pair_rows.append(row)

            if raw_pair_indices:
                adj = adjust_pvalues(raw_pair_pvals, correction)
                adj_disp = adjust_pvalues(raw_pair_dispersion_pvals, correction)
                adj_align = adjust_pvalues(raw_pair_alignment_pvals, correction)
                for idx_local, idx_global in enumerate(raw_pair_indices):
                    p_adj = float(adj[idx_local]) if np.isfinite(adj[idx_local]) else np.nan
                    pair_rows[idx_global]["p_value_adjusted"] = p_adj
                    pair_rows[idx_global]["pvalue_correction"] = correction
                    pair_rows[idx_global]["alpha"] = float(settings.alpha)
                    pair_rows[idx_global]["significant"] = bool(
                        np.isfinite(p_adj) and p_adj < float(settings.alpha)
                    )
                    p_adj_disp = float(adj_disp[idx_local]) if np.isfinite(adj_disp[idx_local]) else np.nan
                    pair_rows[idx_global]["dispersion_difference_p_value_adjusted"] = p_adj_disp
                    pair_rows[idx_global]["dispersion_significant"] = bool(
                        np.isfinite(p_adj_disp) and p_adj_disp < float(settings.alpha)
                    )
                    p_adj_align = float(adj_align[idx_local]) if np.isfinite(adj_align[idx_local]) else np.nan
                    pair_rows[idx_global]["alignment_difference_p_value_adjusted"] = p_adj_align
                    pair_rows[idx_global]["alignment_significant"] = bool(
                        np.isfinite(p_adj_align) and p_adj_align < float(settings.alpha)
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
                "global_dispersion_pseudo_f": global_dispersion_f,
                "global_dispersion_p_value": global_dispersion_p,
                "global_dispersion_p_value_adjusted": np.nan,
                "global_dispersion_significant": False,
                "global_alignment_pseudo_f": global_alignment_f,
                "global_alignment_p_value": global_alignment_p,
                "global_alignment_p_value_adjusted": np.nan,
                "global_alignment_significant": False,
                "global_pvalue_correction": correction,
                "alpha": float(settings.alpha),
                "n_permutations": int(settings.n_permutations),
                "alignment_cosine_threshold": float(settings.alignment_cosine_threshold),
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
        adj_global = adjust_pvalues(window_df["global_p_value"].to_numpy(dtype=float), correction)
        window_df["global_p_value_adjusted"] = adj_global
        window_df["global_significant"] = (
            pd.Series(adj_global).apply(lambda p: bool(np.isfinite(p) and p < float(settings.alpha))).to_numpy(dtype=bool)
        )
        if "global_dispersion_p_value" in window_df.columns:
            adj_disp = adjust_pvalues(
                window_df["global_dispersion_p_value"].to_numpy(dtype=float),
                correction,
            )
            window_df["global_dispersion_p_value_adjusted"] = adj_disp
            window_df["global_dispersion_significant"] = (
                pd.Series(adj_disp)
                .apply(lambda p: bool(np.isfinite(p) and p < float(settings.alpha)))
                .to_numpy(dtype=bool)
            )
        if "global_alignment_p_value" in window_df.columns:
            adj_align = adjust_pvalues(
                window_df["global_alignment_p_value"].to_numpy(dtype=float),
                correction,
            )
            window_df["global_alignment_p_value_adjusted"] = adj_align
            window_df["global_alignment_significant"] = (
                pd.Series(adj_align)
                .apply(lambda p: bool(np.isfinite(p) and p < float(settings.alpha)))
                .to_numpy(dtype=bool)
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
            "pvalue_correction": correction,
            "n_permutations": int(settings.n_permutations),
            "min_units_per_region": int(settings.min_units_per_region),
            "min_regions_per_window": int(settings.min_regions_per_window),
            "pseudo_count": float(settings.pseudo_count),
            "alignment_cosine_threshold": float(settings.alignment_cosine_threshold),
            "require_selective_units": bool(settings.require_selective_units),
            "unit_summary_filename": str(settings.unit_summary_filename),
        },
        "pairwise_summary": pair_df,
        "window_summary": window_df,
    }
    save_pickle_path(result, result_pkl)
    return result
