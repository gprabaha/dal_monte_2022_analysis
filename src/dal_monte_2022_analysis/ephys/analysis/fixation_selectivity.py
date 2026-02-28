"""Analyze fixation-related PSTH selectivity for ephys units."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_SELECTIVITY_WINDOWS_MS: dict[str, tuple[float, float]] = {
    "pre_fix": (-500.0, 0.0),
    "peri_fix": (-250.0, 250.0),
    "post_fix": (0.0, 500.0),
}

DEFAULT_CONDITION_PAIRS: tuple[tuple[str, str], ...] = (
    ("face_interactive", "face_non_interactive"),
    ("face_interactive", "object"),
    ("face_non_interactive", "object"),
)


@dataclass
class FixationPSTHSelectivitySettings:
    """Configuration for fixation-pair selective-unit analysis."""

    cfg_path: str
    trial_input_modality: str = "psth"
    trial_input_filename: str = "fixations.pkl"
    output_subdir: str = "ephys/psth/fixation_psth_selectivity"
    window_stats_filename: str = "window_stats.csv"
    pair_summary_filename: str = "pair_selectivity.csv"
    unit_summary_filename: str = "unit_selectivity.csv"
    output_pickle_filename: str = "results.pkl"
    interactive_label: str = "interactive"
    face_label: str = "face"
    object_label: str = "object"
    windows_ms: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_SELECTIVITY_WINDOWS_MS),
    )
    condition_pairs: tuple[tuple[str, str], ...] = field(
        default_factory=lambda: tuple(DEFAULT_CONDITION_PAIRS),
    )
    alpha: float = 0.05
    test_name: str = "welch_ttest"
    min_trials_per_condition: int = 2
    use_parallel: bool = True
    max_procs: int = 16
    test_single: bool = False
    bin_size_ms_fallback: float = 10.0
    window_pre_s_fallback: float = 1.0
    window_post_s_fallback: float = 1.0


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _as_optional_str(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _ensure_filename(name: str, suffix: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("Output filename cannot be empty.")
    return text if text.endswith(suffix) else f"{text}{suffix}"


def _iter_trial_files(
    cfg: dict,
    settings: FixationPSTHSelectivitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    root = Path(cfg["processed_data_root"])
    filename = _ensure_filename(settings.trial_input_filename, ".pkl")
    pattern = root / "date=*" / "session=*" / settings.trial_input_modality / filename

    date_filter = None if dates is None else {str(v) for v in dates}
    session_filter = None if sessions is None else {str(v) for v in sessions}

    rows: list[dict] = []
    for path in root.glob(str(pattern.relative_to(root))):
        parts = path.parts
        try:
            date_part = next(part for part in parts if part.startswith("date="))
            session_part = next(part for part in parts if part.startswith("session="))
        except StopIteration:
            continue

        date = date_part.split("=", 1)[1]
        session = session_part.split("=", 1)[1]
        if date_filter is not None and date not in date_filter:
            continue
        if session_filter is not None and session not in session_filter:
            continue
        rows.append({"date": date, "session": session, "path": path})

    rows.sort(key=lambda row: (row["date"], row["session"]))
    return rows


def _extract_trials_df_and_meta(obj) -> tuple[pd.DataFrame, dict]:
    if isinstance(obj, dict) and "trials" in obj:
        df = obj["trials"]
        meta = obj.get("meta", {}) or {}
        return (df if isinstance(df, pd.DataFrame) else pd.DataFrame(), meta)
    if isinstance(obj, pd.DataFrame):
        return obj, {}
    return pd.DataFrame(), {}


def _resolve_bin_centers_from_meta(meta: dict) -> Optional[np.ndarray]:
    centers = meta.get("bin_centers_s_rel")
    if centers is not None:
        arr = np.asarray(centers, dtype=float).reshape(-1)
        if arr.size > 0:
            return arr
    edges = meta.get("bin_edges_s_rel")
    if edges is not None:
        arr = np.asarray(edges, dtype=float).reshape(-1)
        if arr.size > 1:
            return 0.5 * (arr[:-1] + arr[1:])
    return None


def _fallback_bin_centers(settings: FixationPSTHSelectivitySettings) -> np.ndarray:
    bin_size_s = float(settings.bin_size_ms_fallback) / 1000.0
    pre = float(settings.window_pre_s_fallback)
    post = float(settings.window_post_s_fallback)
    edges = np.arange(-pre, post + bin_size_s * 0.5, bin_size_s, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _truthy_interactive(value, interactive_label: str) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return float(value) != 0.0
    token = str(value).strip().lower()
    return token in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        str(interactive_label).strip().lower(),
        "interactive",
    }


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


def _load_trial_table(
    settings: FixationPSTHSelectivitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = load_config(settings.cfg_path)
    rows = _iter_trial_files(cfg, settings, dates=dates, sessions=sessions)
    if not rows:
        return pd.DataFrame(), _fallback_bin_centers(settings)

    dfs: list[pd.DataFrame] = []
    bin_centers_ref = None
    for row in rows:
        obj = _load_pickle(Path(row["path"]))
        trial_df, meta = _extract_trials_df_and_meta(obj)
        if trial_df.empty or "psth_counts" not in trial_df.columns:
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
        return pd.DataFrame(), _fallback_bin_centers(settings)

    out_df = pd.concat(dfs, axis=0, ignore_index=True)
    if bin_centers_ref is None:
        bin_centers_ref = _fallback_bin_centers(settings)
    return out_df, np.asarray(bin_centers_ref, dtype=float)


def _resolve_condition_for_row(row, settings: FixationPSTHSelectivitySettings) -> Optional[str]:
    category = str(getattr(row, "fixation_category", "")).strip()
    if category == settings.object_label:
        return "object"
    if category != settings.face_label:
        return None

    interactive = False
    if hasattr(row, "is_interactive"):
        interactive = _truthy_interactive(getattr(row, "is_interactive"), settings.interactive_label)
    elif hasattr(row, "interactive_state"):
        interactive = _truthy_interactive(getattr(row, "interactive_state"), settings.interactive_label)
    return "face_interactive" if interactive else "face_non_interactive"


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

    window_masks: dict[str, np.ndarray] = {}
    for name, (start_ms, stop_ms) in windows_ms.items():
        start_s = float(start_ms) / 1000.0
        stop_s = float(stop_ms) / 1000.0
        mask = (bin_centers_s >= start_s) & (bin_centers_s < stop_s)
        if not np.any(mask):
            raise ValueError(f"Window '{name}' has no bins covered by current PSTH bin centers.")
        window_masks[name] = mask

    condition_data: dict[str, dict[str, list[float]]] = {
        "face_interactive": {name: [] for name in windows_ms},
        "face_non_interactive": {name: [] for name in windows_ms},
        "object": {name: [] for name in windows_ms},
    }

    n_bins = int(bin_centers_s.size)
    for row in df_unit.itertuples(index=False):
        cond = _resolve_condition_for_row(row, settings)
        if cond is None:
            continue
        counts = np.asarray(getattr(row, "psth_counts"), dtype=float).reshape(-1)
        if counts.size != n_bins:
            continue
        rates_hz = counts / float(bin_size_s)
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

    for cond_a, cond_b in settings.condition_pairs:
        pair_label = f"{cond_a}__vs__{cond_b}"
        pair_sig_windows: list[str] = []
        pair_testable_windows = 0
        min_p_value = np.nan

        for win_name, (start_ms, stop_ms) in windows_ms.items():
            arr_a = condition_map.get(cond_a, {}).get(win_name, np.array([], dtype=float))
            arr_b = condition_map.get(cond_b, {}).get(win_name, np.array([], dtype=float))
            n_a = int(arr_a.size)
            n_b = int(arr_b.size)
            mean_a = float(np.mean(arr_a)) if n_a > 0 else np.nan
            mean_b = float(np.mean(arr_b)) if n_b > 0 else np.nan

            tested = n_a >= int(settings.min_trials_per_condition) and n_b >= int(settings.min_trials_per_condition)
            stat = np.nan
            p_value = np.nan
            significant = False
            if tested:
                pair_testable_windows += 1
                stat, p_value = _run_pair_test(arr_a, arr_b, settings=settings)
                significant = bool(np.isfinite(p_value) and float(p_value) < float(settings.alpha))
                if significant:
                    pair_sig_windows.append(win_name)
                if np.isfinite(p_value):
                    min_p_value = float(p_value) if not np.isfinite(min_p_value) else min(min_p_value, float(p_value))

            window_rows.append(
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
                    "tested": bool(tested),
                    "test_name": settings.test_name,
                }
            )

        pair_rows.append(
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
                "pair_label": pair_label,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "is_selective_pair": bool(len(pair_sig_windows) > 0),
                "n_significant_windows": int(len(pair_sig_windows)),
                "n_tested_windows": int(pair_testable_windows),
                "significant_windows": "|".join(pair_sig_windows),
                "min_p_value": float(min_p_value) if np.isfinite(min_p_value) else np.nan,
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


def run_fixation_selectivity_analysis(
    settings: FixationPSTHSelectivitySettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    unit_uuids: Optional[Sequence[str]] = None,
) -> dict:
    """Run fixation-pair selective-unit analysis from trial PSTH outputs."""
    trial_df, bin_centers_s = _load_trial_table(settings, dates=dates, sessions=sessions)
    if trial_df.empty or "unit_uuid" not in trial_df.columns:
        print("[analysis] no trial PSTH rows found for fixation selectivity analysis")
        return {"window_stats": pd.DataFrame(), "pair_summary": pd.DataFrame(), "unit_summary": pd.DataFrame()}

    if unit_uuids is not None:
        allowed_units = {str(unit) for unit in unit_uuids}
        trial_df = trial_df.loc[trial_df["unit_uuid"].astype(str).isin(allowed_units)].copy()
    if trial_df.empty:
        print("[analysis] no matching units found after unit filter")
        return {"window_stats": pd.DataFrame(), "pair_summary": pd.DataFrame(), "unit_summary": pd.DataFrame()}

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

    if settings.use_parallel and len(unit_tasks) > 1:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        with Pool(processes=n_proc) as pool:
            for window_rows, pair_rows, unit_row in tqdm(
                pool.imap_unordered(_unit_worker, unit_tasks),
                total=len(unit_tasks),
                desc=f"Fixation selectivity ({n_proc} workers)",
                unit="unit",
            ):
                window_rows_all.extend(window_rows)
                pair_rows_all.extend(pair_rows)
                unit_rows_all.append(unit_row)
    else:
        for task in tqdm(unit_tasks, desc="Fixation selectivity", unit="unit"):
            window_rows, pair_rows, unit_row = _unit_worker(task)
            window_rows_all.extend(window_rows)
            pair_rows_all.extend(pair_rows)
            unit_rows_all.append(unit_row)

    window_df = pd.DataFrame(window_rows_all)
    pair_df = pd.DataFrame(pair_rows_all)
    unit_df = pd.DataFrame(unit_rows_all)

    if not window_df.empty:
        window_df = window_df.sort_values(["date", "region", "unit_uuid", "pair_label", "window_name"]).reset_index(drop=True)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(["date", "region", "unit_uuid", "pair_label"]).reset_index(drop=True)
    if not unit_df.empty:
        unit_df = unit_df.sort_values(["date", "region", "unit_uuid"]).reset_index(drop=True)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    window_csv = out_root / _ensure_filename(settings.window_stats_filename, ".csv")
    pair_csv = out_root / _ensure_filename(settings.pair_summary_filename, ".csv")
    unit_csv = out_root / _ensure_filename(settings.unit_summary_filename, ".csv")
    result_pkl = out_root / _ensure_filename(settings.output_pickle_filename, ".pkl")

    window_df.to_csv(window_csv, index=False)
    pair_df.to_csv(pair_csv, index=False)
    unit_df.to_csv(unit_csv, index=False)

    result_obj = {
        "meta": {
            "alpha": float(settings.alpha),
            "test_name": settings.test_name,
            "min_trials_per_condition": int(settings.min_trials_per_condition),
            "windows_ms": _normalize_windows(settings.windows_ms),
            "condition_pairs": list(settings.condition_pairs),
            "trial_input_modality": settings.trial_input_modality,
            "trial_input_filename": _ensure_filename(settings.trial_input_filename, ".pkl"),
            "n_units": int(len(unit_df)),
            "n_selective_units": int(unit_df["is_selective_unit"].sum()) if not unit_df.empty else 0,
        },
        "window_stats": window_df,
        "pair_summary": pair_df,
        "unit_summary": unit_df,
    }
    _save_pickle(result_obj, result_pkl)
    return result_obj
