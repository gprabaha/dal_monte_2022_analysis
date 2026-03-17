"""Analyze fixation-related PSTH selectivity for ephys units."""

from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu, ttest_ind

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.ephys.analysis_primitives import (
    as_bool as _as_bool,
    as_optional_str as _as_optional_str,
    ensure_filename as _ensure_filename,
    extract_trials_df_and_meta as _extract_trials_df_and_meta,
    resolve_bin_centers_from_meta as _resolve_bin_centers_from_meta,
)
from dal_monte_2022_analysis.runtime.execution.task_runner import run_tasks
from dal_monte_2022_analysis.runtime.io.processed_data import (
    load_pickle_path,
    scan_processed_paths_for_filename,
    save_pickle_path,
)
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_SELECTIVITY_WINDOWS_MS: dict[str, tuple[float, float]] = {
    "pre_fix": (-500.0, 0.0),
    "peri_fix": (-250.0, 250.0),
    "post_fix": (0.0, 500.0),
}
DEFAULT_SIGNIFICANCE_WINDOWS: tuple[str, ...] = ("pre_fix", "peri_fix", "post_fix")

DEFAULT_CONDITION_PAIRS: tuple[tuple[str, str], ...] = (
    ("face_interactive", "face_non_interactive"),
    ("face_interactive", "object"),
    ("face_non_interactive", "object"),
)

DEFAULT_COMPARISON_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "three_condition_core": tuple(DEFAULT_CONDITION_PAIRS),
    "interactive_state_matched": (
        ("face_interactive", "object_interactive"),
        ("face_non_interactive", "object_non_interactive"),
        ("face_interactive", "face_non_interactive"),
        ("object_interactive", "object_non_interactive"),
    ),
    "face_vs_object_unsplit": (
        ("face", "object"),
    ),
}
DEFAULT_PRIMARY_COMPARISON_GROUP = "three_condition_core"


def _normalize_comparison_groups(
    raw: Optional[dict | list | tuple],
) -> dict[str, tuple[tuple[str, str], ...]]:
    if raw is None:
        return {name: tuple(pairs) for name, pairs in DEFAULT_COMPARISON_GROUPS.items()}

    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, (list, tuple)):
        items = []
        for idx, entry in enumerate(raw):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", f"comparison_{idx}"))
            pairs = entry.get("pairs", [])
            items.append((name, pairs))
    else:
        raise ValueError("comparison_groups must be a dict or list-like object.")

    out: dict[str, tuple[tuple[str, str], ...]] = {}
    for name, pairs in items:
        label = str(name).strip()
        if not label:
            continue
        pair_list: list[tuple[str, str]] = []
        if not isinstance(pairs, (list, tuple)):
            continue
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            cond_a = str(pair[0]).strip()
            cond_b = str(pair[1]).strip()
            if not cond_a or not cond_b or cond_a == cond_b:
                continue
            pair_list.append((cond_a, cond_b))
        if pair_list:
            out[label] = tuple(pair_list)
    if not out:
        raise ValueError("comparison_groups resolved to empty; provide at least one valid pair set.")
    return out


@dataclass
class FixationPSTHSelectivitySettings:
    """Configuration for fixation-pair selective-unit analysis."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations_psth_10ms.pkl"
    output_subdir: str = "ephys/psth/fixation_psth_selectivity"
    window_stats_filename: str = "window_stats.csv"
    pair_summary_filename: str = "pair_selectivity.csv"
    unit_summary_filename: str = "unit_selectivity.csv"
    condition_summary_filename: str = "condition_window_means.csv"
    output_pickle_filename: str = "results.pkl"
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    windows_ms: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_SELECTIVITY_WINDOWS_MS),
    )
    significance_windows: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_SIGNIFICANCE_WINDOWS),
    )
    comparison_groups: dict[str, tuple[tuple[str, str], ...]] = field(
        default_factory=lambda: {name: tuple(pairs) for name, pairs in DEFAULT_COMPARISON_GROUPS.items()},
    )
    primary_comparison_group: str = DEFAULT_PRIMARY_COMPARISON_GROUP
    condition_pairs: tuple[tuple[str, str], ...] = field(
        default_factory=lambda: tuple(DEFAULT_CONDITION_PAIRS),
    )
    smooth_before_window_average: bool = True
    smoothing_sigma_ms: float = 20.0
    alpha: float = 0.05
    test_name: str = "welch_ttest"
    min_trials_per_condition: int = 2
    use_parallel: bool = True
    max_procs: int = 16
    test_single: bool = False
    bin_size_ms_fallback: float = 10.0
    window_pre_s_fallback: float = 1.0
    window_post_s_fallback: float = 1.0


def _fallback_bin_centers(settings: FixationPSTHSelectivitySettings) -> np.ndarray:
    bin_size_s = float(settings.bin_size_ms_fallback) / 1000.0
    pre = float(settings.window_pre_s_fallback)
    post = float(settings.window_post_s_fallback)
    edges = np.arange(-pre, post + bin_size_s * 0.5, bin_size_s, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _normalize_windows(
    windows_raw: dict[str, tuple[float, float]] | list | tuple,
) -> dict[str, tuple[float, float]]:
    if isinstance(windows_raw, dict):
        items = windows_raw.items()
    elif isinstance(windows_raw, (list, tuple)):
        items = []
        for i, entry in enumerate(windows_raw):
            if isinstance(entry, dict):
                name = entry.get("name", f"window_{i}")
                start = entry.get("start_ms")
                stop = entry.get("stop_ms")
                items.append((name, (start, stop)))
            else:
                items.append((f"window_{i}", entry))
    else:
        raise ValueError("windows_ms must be a dict or list-like object.")

    out: dict[str, tuple[float, float]] = {}
    for name, bounds in items:
        if bounds is None or len(bounds) != 2:
            raise ValueError(f"Window '{name}' must define [start_ms, stop_ms].")
        start_ms = float(bounds[0])
        stop_ms = float(bounds[1])
        if stop_ms <= start_ms:
            raise ValueError(f"Window '{name}' has invalid bounds: {bounds}.")
        out[str(name)] = (start_ms, stop_ms)
    if not out:
        raise ValueError("At least one analysis window must be defined.")
    return out


def _normalize_significance_windows(
    windows_raw: Sequence[str] | None,
    *,
    available_windows: Sequence[str],
) -> tuple[str, ...]:
    available = [str(name).strip() for name in available_windows if str(name).strip()]
    if not available:
        raise ValueError("No available windows were provided for selectivity significance filtering.")

    if windows_raw is None:
        requested = list(DEFAULT_SIGNIFICANCE_WINDOWS)
    else:
        requested = [str(name).strip() for name in windows_raw if str(name).strip()]
    if not requested:
        raise ValueError("significance_windows resolved to empty.")

    out: list[str] = []
    for name in requested:
        if name in available and name not in out:
            out.append(name)
    if not out:
        raise ValueError(
            "No configured significance windows matched available windows. "
            f"requested={requested}, available={available}"
        )
    return tuple(out)


def _load_trial_table(
    settings: FixationPSTHSelectivitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = load_config(settings.cfg_path)
    rows = scan_processed_paths_for_filename(
        cfg,
        settings.trial_input_modality,
        filename=_ensure_filename(settings.trial_input_filename, ".pkl"),
        dates=dates,
        sessions=sessions,
        agents=(None,),
    )
    if not rows:
        return pd.DataFrame(), _fallback_bin_centers(settings)

    dfs: list[pd.DataFrame] = []
    bin_centers_ref = None
    n_empty_trials = 0
    n_missing_psth_counts = 0
    for row in rows:
        obj = load_pickle_path(row["path"])
        trial_df, meta = _extract_trials_df_and_meta(obj)
        if trial_df.empty:
            n_empty_trials += 1
            continue
        if "psth_counts" not in trial_df.columns:
            n_missing_psth_counts += 1
            continue

        centers = _resolve_bin_centers_from_meta(meta)
        if centers is not None:
            if bin_centers_ref is None:
                bin_centers_ref = centers
            elif centers.shape != bin_centers_ref.shape or not np.allclose(centers, bin_centers_ref):
                raise ValueError(f"Mismatched PSTH bin centers across trial files; path={row['path']}")

        df = trial_df.copy()
        if "date" not in df.columns:
            df["date"] = str(row["date"])
        if "session" not in df.columns:
            df["session"] = str(row["session"])
        dfs.append(df)

    if not dfs:
        print(
            "[analysis] trial PSTH files were found but usable trial rows were not. "
            f"n_files={len(rows)}, empty_trials={n_empty_trials}, "
            f"missing_psth_counts={n_missing_psth_counts}"
        )
        return pd.DataFrame(), _fallback_bin_centers(settings)

    out_df = pd.concat(dfs, axis=0, ignore_index=True)
    if bin_centers_ref is None:
        bin_centers_ref = _fallback_bin_centers(settings)
    return out_df, np.asarray(bin_centers_ref, dtype=float)


def _resolve_condition_labels_for_row(
    row,
    settings: FixationPSTHSelectivitySettings,
) -> list[str]:
    category = str(getattr(row, "fixation_category", "")).strip()
    if category not in {settings.face_label, settings.object_label}:
        return []

    interactive = False
    if hasattr(row, "is_interactive"):
        interactive = _as_bool(getattr(row, "is_interactive"), settings.interactive_label)
    elif hasattr(row, "interactive_state"):
        interactive = _as_bool(getattr(row, "interactive_state"), settings.interactive_label)

    if category == settings.face_label:
        labels = ["face_interactive" if interactive else "face_non_interactive", "face"]
    else:
        labels = ["object_interactive" if interactive else "object_non_interactive", "object"]
    return labels


def _resolve_condition_for_row(row, settings: FixationPSTHSelectivitySettings) -> Optional[str]:
    labels = _resolve_condition_labels_for_row(row, settings)
    if not labels:
        return None
    primary = labels[0]
    if primary in {"face_interactive", "face_non_interactive", "object"}:
        return primary
    if primary.startswith("object_"):
        return "object"
    return primary


def _resolve_smoothing_sigma_bins(
    settings: FixationPSTHSelectivitySettings,
    *,
    bin_size_s: float,
) -> Optional[float]:
    if not settings.smooth_before_window_average:
        return None
    if not np.isfinite(float(bin_size_s)) or float(bin_size_s) <= 0:
        raise ValueError("Unable to resolve bin size for selectivity smoothing.")
    sigma_ms = float(settings.smoothing_sigma_ms)
    if sigma_ms <= 0:
        raise ValueError("smoothing_sigma_ms must be > 0 when smoothing is enabled.")
    sigma_bins = sigma_ms / (float(bin_size_s) * 1000.0)
    if not np.isfinite(sigma_bins) or sigma_bins <= 0:
        raise ValueError("Resolved smoothing sigma in bins must be > 0.")
    return float(sigma_bins)


def _compute_window_means_by_condition(
    df_unit: pd.DataFrame,
    *,
    settings: FixationPSTHSelectivitySettings,
    bin_centers_s: np.ndarray,
    windows_ms: dict[str, tuple[float, float]],
) -> dict[str, dict[str, np.ndarray]]:
    bin_size_s = float(np.mean(np.diff(bin_centers_s))) if bin_centers_s.size > 1 else None
    if bin_size_s is None or bin_size_s <= 0:
        raise ValueError("Unable to infer positive bin size for fixation selectivity analysis.")
    sigma_bins = _resolve_smoothing_sigma_bins(settings, bin_size_s=bin_size_s)

    window_masks: dict[str, np.ndarray] = {}
    for name, (start_ms, stop_ms) in windows_ms.items():
        start_s = float(start_ms) / 1000.0
        stop_s = float(stop_ms) / 1000.0
        mask = (bin_centers_s >= start_s) & (bin_centers_s < stop_s)
        if not np.any(mask):
            raise ValueError(f"Window '{name}' has no bins covered by current PSTH bin centers.")
        window_masks[name] = mask

    condition_keys = (
        "face_interactive",
        "face_non_interactive",
        "object_interactive",
        "object_non_interactive",
        "face",
        "object",
    )
    condition_data: dict[str, dict[str, list[float]]] = {
        key: {name: [] for name in windows_ms}
        for key in condition_keys
    }

    n_bins = int(bin_centers_s.size)
    for row in df_unit.itertuples(index=False):
        cond_labels = _resolve_condition_labels_for_row(row, settings)
        if not cond_labels:
            continue
        counts = np.asarray(getattr(row, "psth_counts"), dtype=float).reshape(-1)
        if counts.size != n_bins:
            continue
        if sigma_bins is not None:
            counts = gaussian_filter1d(counts, sigma=sigma_bins, mode="nearest")
        rates_hz = counts / float(bin_size_s)
        for cond in cond_labels:
            if cond not in condition_data:
                continue
            for win_name, mask in window_masks.items():
                condition_data[cond][win_name].append(float(np.mean(rates_hz[mask])))

    out: dict[str, dict[str, np.ndarray]] = {}
    for cond, win_map in condition_data.items():
        out[cond] = {}
        for win_name, values in win_map.items():
            out[cond][win_name] = np.asarray(values, dtype=float).reshape(-1)
    return out


def _run_pair_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    settings: FixationPSTHSelectivitySettings,
) -> tuple[float, float]:
    if settings.test_name == "welch_ttest":
        stat, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return float(stat), float(p)
    if settings.test_name == "mannwhitney":
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(stat), float(p)
    raise ValueError(
        f"Unsupported selective_test '{settings.test_name}'. "
        "Expected 'welch_ttest' or 'mannwhitney'."
    )


def _build_condition_summary_rows(
    *,
    unit_key: str,
    date: str,
    unit_uuid: str,
    region: Optional[str],
    spike_channel: Optional[str],
    recorded_agent: Optional[str],
    recorded_monkey: Optional[str],
    area: Optional[str],
    n_sessions: int,
    windows_ms: dict[str, tuple[float, float]],
    condition_map: dict[str, dict[str, np.ndarray]],
    settings: FixationPSTHSelectivitySettings,
) -> list[dict]:
    rows: list[dict] = []
    condition_keys = ("face_interactive", "face_non_interactive", "object")
    for win_name, (start_ms, stop_ms) in windows_ms.items():
        arr_int = condition_map.get("face_interactive", {}).get(win_name, np.array([], dtype=float))
        arr_nonint = condition_map.get("face_non_interactive", {}).get(win_name, np.array([], dtype=float))
        arr_obj = condition_map.get("object", {}).get(win_name, np.array([], dtype=float))
        arr_face = condition_map.get("face", {}).get(win_name, np.array([], dtype=float))
        arr_obj_int = condition_map.get("object_interactive", {}).get(win_name, np.array([], dtype=float))
        arr_obj_nonint = condition_map.get("object_non_interactive", {}).get(win_name, np.array([], dtype=float))

        n_int = int(arr_int.size)
        n_nonint = int(arr_nonint.size)
        n_obj = int(arr_obj.size)
        n_face = int(arr_face.size)
        n_obj_int = int(arr_obj_int.size)
        n_obj_nonint = int(arr_obj_nonint.size)
        mean_int = float(np.mean(arr_int)) if n_int > 0 else np.nan
        mean_nonint = float(np.mean(arr_nonint)) if n_nonint > 0 else np.nan
        mean_obj = float(np.mean(arr_obj)) if n_obj > 0 else np.nan
        mean_face = float(np.mean(arr_face)) if n_face > 0 else np.nan
        mean_obj_int = float(np.mean(arr_obj_int)) if n_obj_int > 0 else np.nan
        mean_obj_nonint = float(np.mean(arr_obj_nonint)) if n_obj_nonint > 0 else np.nan

        means = np.asarray([mean_int, mean_nonint, mean_obj], dtype=float)
        total_mean_hz = float(np.sum(means)) if np.all(np.isfinite(means)) else np.nan

        rel = np.full(3, np.nan, dtype=float)
        dominant_condition = None
        dominant_relative_value = np.nan
        if np.all(np.isfinite(means)) and float(total_mean_hz) > 0.0:
            rel = means / float(total_mean_hz)
            dominant_idx = int(np.argmax(rel))
            dominant_condition = condition_keys[dominant_idx]
            dominant_relative_value = float(rel[dominant_idx])

        meets_min_trials = (
            n_int >= int(settings.min_trials_per_condition)
            and n_nonint >= int(settings.min_trials_per_condition)
            and n_obj >= int(settings.min_trials_per_condition)
        )
        rows.append(
            {
                "unit_key": unit_key,
                "date": date,
                "unit_uuid": unit_uuid,
                "region": region,
                "spike_channel": spike_channel,
                "recorded_agent": recorded_agent,
                "recorded_monkey": recorded_monkey,
                "area": area,
                "n_sessions": n_sessions,
                "window_name": win_name,
                "window_start_ms": float(start_ms),
                "window_stop_ms": float(stop_ms),
                "n_trials_face_interactive": n_int,
                "n_trials_face_non_interactive": n_nonint,
                "n_trials_face": n_face,
                "n_trials_object_interactive": n_obj_int,
                "n_trials_object_non_interactive": n_obj_nonint,
                "n_trials_object": n_obj,
                "n_trials_total": int(n_int + n_nonint + n_obj),
                "mean_fr_face_interactive_hz": mean_int,
                "mean_fr_face_non_interactive_hz": mean_nonint,
                "mean_fr_face_hz": mean_face,
                "mean_fr_object_interactive_hz": mean_obj_int,
                "mean_fr_object_non_interactive_hz": mean_obj_nonint,
                "mean_fr_object_hz": mean_obj,
                "total_mean_fr_hz": total_mean_hz,
                "relative_face_interactive": float(rel[0]) if np.isfinite(rel[0]) else np.nan,
                "relative_face_non_interactive": float(rel[1]) if np.isfinite(rel[1]) else np.nan,
                "relative_object": float(rel[2]) if np.isfinite(rel[2]) else np.nan,
                "all_conditions_observed": bool(n_int > 0 and n_nonint > 0 and n_obj > 0),
                "all_split_conditions_observed": bool(
                    n_int > 0 and n_nonint > 0 and n_obj_int > 0 and n_obj_nonint > 0
                ),
                "face_object_unsplit_observed": bool(n_face > 0 and n_obj > 0),
                "meets_min_trials": bool(meets_min_trials),
                "meets_min_trials_face_object_unsplit": bool(
                    n_face >= int(settings.min_trials_per_condition)
                    and n_obj >= int(settings.min_trials_per_condition)
                ),
                "dominant_condition": dominant_condition,
                "dominant_relative_value": dominant_relative_value,
            }
        )
    return rows


def _evaluate_pairs_for_comparison(
    *,
    comparison_label: str,
    condition_pairs: Sequence[tuple[str, str]],
    windows_ms: dict[str, tuple[float, float]],
    condition_map: dict[str, dict[str, np.ndarray]],
    unit_key: str,
    date: str,
    unit_uuid: str,
    region: Optional[str],
    spike_channel: Optional[str],
    recorded_agent: Optional[str],
    recorded_monkey: Optional[str],
    area: Optional[str],
    n_sessions: int,
    settings: FixationPSTHSelectivitySettings,
) -> tuple[list[dict], list[dict], dict]:
    window_rows: list[dict] = []
    pair_rows: list[dict] = []
    selectivity_window_set = set(str(name) for name in settings.significance_windows)

    for cond_a, cond_b in condition_pairs:
        pair_label = f"{cond_a}__vs__{cond_b}"
        pair_sig_windows_selective: list[str] = []
        pair_sig_windows_all: list[str] = []
        pair_testable_windows_selective = 0
        pair_testable_windows_all = 0
        min_p_value = np.nan
        min_p_value_all = np.nan

        for win_name, (start_ms, stop_ms) in windows_ms.items():
            arr_a = condition_map.get(cond_a, {}).get(win_name, np.array([], dtype=float))
            arr_b = condition_map.get(cond_b, {}).get(win_name, np.array([], dtype=float))
            n_a = int(arr_a.size)
            n_b = int(arr_b.size)
            mean_a = float(np.mean(arr_a)) if n_a > 0 else np.nan
            mean_b = float(np.mean(arr_b)) if n_b > 0 else np.nan
            counts_toward_selectivity = bool(win_name in selectivity_window_set)

            tested = n_a >= int(settings.min_trials_per_condition) and n_b >= int(settings.min_trials_per_condition)
            stat = np.nan
            p_value = np.nan
            significant = False
            if tested:
                pair_testable_windows_all += 1
                if counts_toward_selectivity:
                    pair_testable_windows_selective += 1
                stat, p_value = _run_pair_test(arr_a, arr_b, settings=settings)
                significant = bool(np.isfinite(p_value) and float(p_value) < float(settings.alpha))
                if significant:
                    pair_sig_windows_all.append(win_name)
                    if counts_toward_selectivity:
                        pair_sig_windows_selective.append(win_name)
                if np.isfinite(p_value):
                    min_p_value_all = (
                        float(p_value) if not np.isfinite(min_p_value_all) else min(min_p_value_all, float(p_value))
                    )
                    if counts_toward_selectivity:
                        min_p_value = (
                            float(p_value) if not np.isfinite(min_p_value) else min(min_p_value, float(p_value))
                        )

            window_rows.append(
                {
                    "comparison_label": comparison_label,
                    "unit_key": unit_key,
                    "date": date,
                    "unit_uuid": unit_uuid,
                    "region": region,
                    "spike_channel": spike_channel,
                    "recorded_agent": recorded_agent,
                    "recorded_monkey": recorded_monkey,
                    "area": area,
                    "n_sessions": n_sessions,
                    "pair_label": pair_label,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "window_name": win_name,
                    "window_start_ms": float(start_ms),
                    "window_stop_ms": float(stop_ms),
                    "n_trials_a": n_a,
                    "n_trials_b": n_b,
                    "mean_fr_a_hz": mean_a,
                    "mean_fr_b_hz": mean_b,
                    "statistic": stat,
                    "p_value": p_value,
                    "alpha": float(settings.alpha),
                    "significant": bool(significant),
                    "counts_toward_selectivity": bool(counts_toward_selectivity),
                    "significant_for_selectivity": bool(significant and counts_toward_selectivity),
                    "tested": bool(tested),
                    "test_name": settings.test_name,
                }
            )

        pair_rows.append(
            {
                "comparison_label": comparison_label,
                "unit_key": unit_key,
                "date": date,
                "unit_uuid": unit_uuid,
                "region": region,
                "spike_channel": spike_channel,
                "recorded_agent": recorded_agent,
                "recorded_monkey": recorded_monkey,
                "area": area,
                "n_sessions": n_sessions,
                "pair_label": pair_label,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "is_selective_pair": bool(len(pair_sig_windows_selective) > 0),
                "n_significant_windows": int(len(pair_sig_windows_selective)),
                "n_tested_windows": int(pair_testable_windows_selective),
                "significant_windows": "|".join(pair_sig_windows_selective),
                "min_p_value": float(min_p_value) if np.isfinite(min_p_value) else np.nan,
                "n_significant_windows_all": int(len(pair_sig_windows_all)),
                "n_tested_windows_all": int(pair_testable_windows_all),
                "significant_windows_all": "|".join(pair_sig_windows_all),
                "min_p_value_all": float(min_p_value_all) if np.isfinite(min_p_value_all) else np.nan,
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    n_selective_pairs = int(pair_df["is_selective_pair"].sum()) if not pair_df.empty else 0
    n_tested_pairs = int((pair_df["n_tested_windows"] > 0).sum()) if not pair_df.empty else 0
    selective_labels = (
        pair_df.loc[pair_df["is_selective_pair"], "pair_label"].astype(str).tolist()
        if not pair_df.empty
        else []
    )
    unit_row = {
        "comparison_label": comparison_label,
        "unit_key": unit_key,
        "date": date,
        "unit_uuid": unit_uuid,
        "region": region,
        "spike_channel": spike_channel,
        "recorded_agent": recorded_agent,
        "recorded_monkey": recorded_monkey,
        "area": area,
        "n_sessions": n_sessions,
        "is_selective_unit": bool(n_selective_pairs > 0),
        "n_selective_pairs": n_selective_pairs,
        "n_tested_pairs": n_tested_pairs,
        "selective_pairs": "|".join(selective_labels),
    }
    return window_rows, pair_rows, unit_row


def _unit_worker(args):
    unit_key, df_unit, bin_centers_s, settings = args
    windows_ms = _normalize_windows(settings.windows_ms)
    condition_map = _compute_window_means_by_condition(
        df_unit,
        settings=settings,
        bin_centers_s=bin_centers_s,
        windows_ms=windows_ms,
    )

    row0 = df_unit.iloc[0]
    date = str(row0.get("date"))
    unit_uuid = str(row0.get("unit_uuid"))
    region = _as_optional_str(row0.get("region"))
    spike_channel = _as_optional_str(row0.get("spike_channel"))
    recorded_agent = _as_optional_str(row0.get("recorded_agent"))
    recorded_monkey = _as_optional_str(row0.get("recorded_monkey"))
    area = _as_optional_str(row0.get("area"))
    n_sessions = int(df_unit["session"].nunique()) if "session" in df_unit.columns else 0

    window_rows: list[dict] = []
    pair_rows: list[dict] = []
    condition_rows: list[dict] = _build_condition_summary_rows(
        unit_key=unit_key,
        date=date,
        unit_uuid=unit_uuid,
        region=region,
        spike_channel=spike_channel,
        recorded_agent=recorded_agent,
        recorded_monkey=recorded_monkey,
        area=area,
        n_sessions=n_sessions,
        windows_ms=windows_ms,
        condition_map=condition_map,
        settings=settings,
    )
    unit_rows: list[dict] = []
    comparison_groups = _normalize_comparison_groups(settings.comparison_groups)
    for comparison_label, condition_pairs in comparison_groups.items():
        comp_window_rows, comp_pair_rows, comp_unit_row = _evaluate_pairs_for_comparison(
            comparison_label=comparison_label,
            condition_pairs=condition_pairs,
            windows_ms=windows_ms,
            condition_map=condition_map,
            unit_key=unit_key,
            date=date,
            unit_uuid=unit_uuid,
            region=region,
            spike_channel=spike_channel,
            recorded_agent=recorded_agent,
            recorded_monkey=recorded_monkey,
            area=area,
            n_sessions=n_sessions,
            settings=settings,
        )
        window_rows.extend(comp_window_rows)
        pair_rows.extend(comp_pair_rows)
        unit_rows.append(comp_unit_row)
    return window_rows, pair_rows, unit_rows, condition_rows


def _comparison_label_token(label: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(label).strip()).strip("_").lower()
    return token or "comparison"


def _filename_with_comparison_label(filename: str, comparison_label: str) -> str:
    safe = _comparison_label_token(comparison_label)
    token = _ensure_filename(filename, ".csv")
    stem = token[:-4]
    return f"{stem}__{safe}.csv"


def _print_table_block(title: str, table_df: pd.DataFrame, *, empty_message: str = "(none)") -> None:
    print(f"[analysis] {title}")
    print(f"[analysis] {'-' * len(title)}")
    if table_df.empty:
        print(f"[analysis] {empty_message}")
        print("[analysis]")
        return
    table_text = table_df.to_string(index=False)
    for line in table_text.splitlines():
        print(f"[analysis] {line}")
    print("[analysis]")


def _print_comparison_summary(
    comparison_label: str,
    pair_df: pd.DataFrame,
    window_df: pd.DataFrame,
    *,
    significance_windows: Sequence[str],
) -> None:
    print("[analysis]")
    print(f"[analysis] === Selectivity Summary: {comparison_label} ===")
    print(
        "[analysis] significance windows used for selectivity calls: "
        f"{', '.join(str(name) for name in significance_windows)}"
    )
    if pair_df.empty:
        print("[analysis] no pair-level rows for this comparison")
        print("[analysis]")
        return

    region_values = pair_df.get("region", pd.Series(dtype=str)).fillna("unknown").astype(str)
    pair_tmp = pair_df.copy()
    pair_tmp["region"] = region_values

    region_total = (
        pair_tmp.groupby("region")["unit_key"].nunique().sort_index()
        if "unit_key" in pair_tmp.columns
        else pd.Series(dtype=int)
    )
    region_sig = (
        pair_tmp.loc[pair_tmp["is_selective_pair"]]
        .groupby("region")["unit_key"]
        .nunique()
        .sort_index()
        if "is_selective_pair" in pair_tmp.columns and "unit_key" in pair_tmp.columns
        else pd.Series(dtype=int)
    )
    region_summary_rows: list[dict] = []
    for region in sorted(region_total.index.tolist()):
        total = int(region_total.get(region, 0))
        sig = int(region_sig.get(region, 0))
        frac = (float(sig) / float(total)) if total > 0 else np.nan
        region_summary_rows.append(
            {
                "region": region,
                "selective_units": sig,
                "total_units": total,
                "fraction_selective": frac,
            }
        )
    _print_table_block(
        "Region-Level Selective Units",
        pd.DataFrame(region_summary_rows),
        empty_message="no region-level counts",
    )
    window_tmp = pd.DataFrame()
    if (
        not window_df.empty
        and {"region", "window_name", "unit_key", "significant_for_selectivity", "counts_toward_selectivity"}.issubset(
            window_df.columns
        )
    ):
        window_tmp = window_df.copy()
        window_tmp["region"] = window_tmp["region"].fillna("unknown").astype(str)
        window_tmp = window_tmp.loc[window_tmp["counts_toward_selectivity"]].copy()

    for region in sorted(region_total.index.tolist()):
        total_units_region = int(region_total.get(region, 0))
        print("[analysis]")
        print(f"[analysis] ### Region: {region} (total_units={total_units_region}) ###")

        if {"region", "pair_label", "unit_key", "is_selective_pair"}.issubset(pair_tmp.columns):
            region_pair_df = pair_tmp.loc[pair_tmp["region"].astype(str) == str(region)].copy()
            pair_sig_counts = (
                region_pair_df.loc[region_pair_df["is_selective_pair"]]
                .groupby("pair_label")["unit_key"]
                .nunique()
                .reset_index(name="selective_units")
                .sort_values("pair_label")
                .reset_index(drop=True)
            )
            if not pair_sig_counts.empty:
                pair_sig_counts["total_units"] = int(total_units_region)
                pair_sig_counts["fraction_selective"] = pair_sig_counts["selective_units"].astype(float) / float(
                    total_units_region
                ) if total_units_region > 0 else np.nan
                pair_sig_counts = pair_sig_counts.rename(columns={"pair_label": "pair"})
            _print_table_block(
                f"Region {region}: Selective Units By Pair",
                pair_sig_counts,
                empty_message="no selective pairs",
            )

        if not window_tmp.empty:
            region_window_df = window_tmp.loc[window_tmp["region"].astype(str) == str(region)].copy()
            sig_units_by_window = (
                region_window_df.loc[region_window_df["significant_for_selectivity"]]
                .groupby("window_name")["unit_key"]
                .nunique()
                .reset_index(name="selective_units")
                .sort_values("window_name")
                .reset_index(drop=True)
            )
            if not sig_units_by_window.empty:
                sig_units_by_window["total_units"] = int(total_units_region)
                sig_units_by_window["fraction_selective"] = sig_units_by_window["selective_units"].astype(float) / float(
                    total_units_region
                ) if total_units_region > 0 else np.nan
                sig_units_by_window = sig_units_by_window.rename(columns={"window_name": "window"})
            _print_table_block(
                f"Region {region}: Selective Units By Window",
                sig_units_by_window,
                empty_message="no significant selective units in configured windows",
            )

        if {"region", "pair_label", "unit_key", "is_selective_pair"}.issubset(pair_tmp.columns):
            region_df = pair_tmp.loc[pair_tmp["region"].astype(str) == str(region)]
            pair_to_units: dict[str, set[str]] = {}
            for pair_label in sorted(region_df["pair_label"].dropna().astype(str).unique().tolist()):
                u = set(
                    region_df.loc[
                        (region_df["pair_label"].astype(str) == pair_label)
                        & (region_df["is_selective_pair"]),
                        "unit_key",
                    ].astype(str).tolist()
                )
                pair_to_units[pair_label] = u
            overlap_rows: list[dict] = []
            if len(pair_to_units) >= 2:
                for pair_a, pair_b in combinations(sorted(pair_to_units.keys()), 2):
                    overlap_n = len(pair_to_units[pair_a].intersection(pair_to_units[pair_b]))
                    overlap_rows.append(
                        {
                            "pair_a": pair_a,
                            "pair_b": pair_b,
                            "overlap_units": int(overlap_n),
                            "total_units": int(total_units_region),
                            "fraction_selective": (
                                float(overlap_n) / float(total_units_region)
                                if total_units_region > 0
                                else np.nan
                            ),
                        }
                    )
            overlap_df = pd.DataFrame(overlap_rows)
            if not overlap_df.empty:
                overlap_df = overlap_df.sort_values(["pair_a", "pair_b"]).reset_index(drop=True)
            _print_table_block(
                f"Region {region}: Pair Overlap Across Selective Units",
                overlap_df,
                empty_message="no pair-overlap rows",
            )


def run_fixation_selectivity_analysis(
    settings: FixationPSTHSelectivitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    unit_uuids: Optional[Sequence[str]] = None,
) -> dict:
    """Run fixation-pair selective-unit analysis from trial PSTH outputs."""
    windows_ms = _normalize_windows(settings.windows_ms)
    significance_windows = _normalize_significance_windows(
        settings.significance_windows,
        available_windows=tuple(windows_ms.keys()),
    )
    comparison_groups = _normalize_comparison_groups(settings.comparison_groups)
    settings.significance_windows = significance_windows
    settings.comparison_groups = comparison_groups
    primary_comparison_label = str(settings.primary_comparison_group).strip()
    if primary_comparison_label not in comparison_groups:
        primary_comparison_label = next(iter(comparison_groups.keys()))
    settings.condition_pairs = tuple(comparison_groups[primary_comparison_label])

    trial_df, bin_centers_s = _load_trial_table(settings, dates=dates, sessions=sessions)
    if trial_df.empty or "unit_uuid" not in trial_df.columns:
        print("[analysis] no trial PSTH rows found for fixation selectivity analysis")
        return {
            "window_stats": pd.DataFrame(),
            "pair_summary": pd.DataFrame(),
            "unit_summary": pd.DataFrame(),
            "condition_summary": pd.DataFrame(),
            "comparison_results": {},
        }

    if unit_uuids is not None:
        allowed_units = {str(unit) for unit in unit_uuids}
        trial_df = trial_df.loc[trial_df["unit_uuid"].astype(str).isin(allowed_units)].copy()
    if trial_df.empty:
        print("[analysis] no matching units found after unit filter")
        return {
            "window_stats": pd.DataFrame(),
            "pair_summary": pd.DataFrame(),
            "unit_summary": pd.DataFrame(),
            "condition_summary": pd.DataFrame(),
            "comparison_results": {},
        }

    if "date" not in trial_df.columns:
        trial_df["date"] = "unknown"

    grouped = trial_df.groupby(["date", "unit_uuid"], sort=True, dropna=False)
    unit_tasks = []
    for (date, unit_uuid), df_unit in grouped:
        unit_key = f"{date}|{unit_uuid}"
        unit_tasks.append((unit_key, df_unit.copy(), bin_centers_s, settings))

    if settings.test_single and unit_tasks:
        unit_tasks = [random.choice(unit_tasks)]

    window_rows_all: list[dict] = []
    pair_rows_all: list[dict] = []
    unit_rows_all: list[dict] = []
    condition_rows_all: list[dict] = []

    results = run_tasks(
        _unit_worker,
        unit_tasks,
        desc="Fixation selectivity",
        unit="unit",
        use_parallel=settings.use_parallel,
        max_procs=settings.max_procs,
    )
    for window_rows, pair_rows, unit_rows, condition_rows in results:
        window_rows_all.extend(window_rows)
        pair_rows_all.extend(pair_rows)
        unit_rows_all.extend(unit_rows)
        condition_rows_all.extend(condition_rows)

    window_df = pd.DataFrame(window_rows_all)
    pair_df = pd.DataFrame(pair_rows_all)
    unit_df = pd.DataFrame(unit_rows_all)
    condition_df = pd.DataFrame(condition_rows_all)

    if not window_df.empty:
        window_df = window_df.sort_values(
            ["comparison_label", "date", "region", "unit_uuid", "pair_label", "window_name"]
        ).reset_index(drop=True)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            ["comparison_label", "date", "region", "unit_uuid", "pair_label"]
        ).reset_index(drop=True)
    if not unit_df.empty:
        unit_df = unit_df.sort_values(
            ["comparison_label", "date", "region", "unit_uuid"]
        ).reset_index(drop=True)
    if not condition_df.empty:
        condition_df = condition_df.sort_values(["date", "region", "unit_uuid", "window_name"]).reset_index(drop=True)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    base_window_filename = _ensure_filename(settings.window_stats_filename, ".csv")
    base_pair_filename = _ensure_filename(settings.pair_summary_filename, ".csv")
    base_unit_filename = _ensure_filename(settings.unit_summary_filename, ".csv")

    comparison_results: dict[str, dict] = {}
    primary_window_df = pd.DataFrame()
    primary_pair_df = pd.DataFrame()
    primary_unit_df = pd.DataFrame()

    for comparison_label in comparison_groups.keys():
        comp_window_df = window_df.loc[
            window_df["comparison_label"].astype(str) == str(comparison_label)
        ].copy() if not window_df.empty else pd.DataFrame()
        comp_pair_df = pair_df.loc[
            pair_df["comparison_label"].astype(str) == str(comparison_label)
        ].copy() if not pair_df.empty else pd.DataFrame()
        comp_unit_df = unit_df.loc[
            unit_df["comparison_label"].astype(str) == str(comparison_label)
        ].copy() if not unit_df.empty else pd.DataFrame()

        comp_window_csv = out_root / _filename_with_comparison_label(base_window_filename, comparison_label)
        comp_pair_csv = out_root / _filename_with_comparison_label(base_pair_filename, comparison_label)
        comp_unit_csv = out_root / _filename_with_comparison_label(base_unit_filename, comparison_label)
        comp_window_df.to_csv(comp_window_csv, index=False)
        comp_pair_df.to_csv(comp_pair_csv, index=False)
        comp_unit_df.to_csv(comp_unit_csv, index=False)

        _print_comparison_summary(
            comparison_label,
            comp_pair_df,
            comp_window_df,
            significance_windows=significance_windows,
        )

        comparison_results[str(comparison_label)] = {
            "window_stats": comp_window_df,
            "pair_summary": comp_pair_df,
            "unit_summary": comp_unit_df,
            "window_stats_filename": comp_window_csv.name,
            "pair_summary_filename": comp_pair_csv.name,
            "unit_summary_filename": comp_unit_csv.name,
        }

        if str(comparison_label) == str(primary_comparison_label):
            primary_window_df = comp_window_df
            primary_pair_df = comp_pair_df
            primary_unit_df = comp_unit_df

    window_csv = out_root / base_window_filename
    pair_csv = out_root / base_pair_filename
    unit_csv = out_root / base_unit_filename
    condition_csv = out_root / _ensure_filename(settings.condition_summary_filename, ".csv")
    result_pkl = out_root / _ensure_filename(settings.output_pickle_filename, ".pkl")

    # Keep legacy unsuffixed filenames mapped to the configured primary comparison.
    primary_window_df.to_csv(window_csv, index=False)
    primary_pair_df.to_csv(pair_csv, index=False)
    primary_unit_df.to_csv(unit_csv, index=False)
    condition_df.to_csv(condition_csv, index=False)

    result_obj = {
        "meta": {
            "alpha": float(settings.alpha),
            "test_name": settings.test_name,
            "min_trials_per_condition": int(settings.min_trials_per_condition),
            "windows_ms": windows_ms,
            "condition_pairs": [list(pair) for pair in settings.condition_pairs],
            "comparison_groups": {
                label: [list(pair) for pair in pairs]
                for label, pairs in comparison_groups.items()
            },
            "significance_windows": [str(name) for name in significance_windows],
            "primary_comparison_group": str(primary_comparison_label),
            "smooth_before_window_average": bool(settings.smooth_before_window_average),
            "smoothing_sigma_ms": float(settings.smoothing_sigma_ms),
            "trial_input_modality": settings.trial_input_modality,
            "trial_input_filename": _ensure_filename(settings.trial_input_filename, ".pkl"),
            "n_units": int(len(primary_unit_df)),
            "n_selective_units": int(primary_unit_df["is_selective_unit"].sum()) if not primary_unit_df.empty else 0,
        },
        "window_stats": primary_window_df,
        "pair_summary": primary_pair_df,
        "unit_summary": primary_unit_df,
        "condition_summary": condition_df,
        "comparison_results": comparison_results,
    }
    save_pickle_path(result_obj, result_pkl)
    return result_obj
