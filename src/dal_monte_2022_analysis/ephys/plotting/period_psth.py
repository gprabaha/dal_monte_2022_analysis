"""Plot per-unit interactive/non-interactive period PSTHs."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from dal_monte_2022_analysis.behav.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_PERIOD_CONDITION_COLORS = {
    "period_interactive": "#d62728",
    "period_non_interactive": "#1f77b4",
}


@dataclass
class PeriodPSTHUnitPlotSettings:
    """Configuration for per-unit period PSTH plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    trial_input_modality: str = "psth"
    trial_input_filename: str = "interactive_periods.pkl"
    output_subdir: str = "ephys/psth/period_psth_unit_plots"
    output_extension: str = "png"
    output_dpi: Optional[int] = 220
    interactive_label: str = "interactive"
    non_interactive_label: str = "non_interactive"
    use_parallel: bool = True
    parallelize_units: bool = True
    unit_parallel_min_units: int = 2
    max_procs: int = 16
    test_single: bool = False
    max_trials_per_condition: Optional[int] = None
    random_seed: int = 42
    condition_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PERIOD_CONDITION_COLORS))
    smooth_before_average: bool = True
    smoothing_sigma_ms: float = 200.0
    raster_jitter_within_bin: bool = True
    raster_linelength: float = 0.95
    raster_linewidth: float = 1.0
    raster_alpha: float = 1.0
    raster_darkening_factor: float = 0.65
    raster_show_condition_background: bool = False
    raster_max_spikes_per_bin: Optional[int] = None
    panel_raster_height_ratio: float = 1.2
    panel_rate_height_ratio: float = 2.0
    bin_size_ms_fallback: float = 100.0
    window_pre_s: float = 14.0
    window_post_s: float = 14.0


def _safe_optional_str(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _ensure_ext(ext: str) -> str:
    text = str(ext).strip().lower()
    if not text:
        return "png"
    return text[1:] if text.startswith(".") else text


def _iter_trial_rows(
    cfg: dict,
    settings: PeriodPSTHUnitPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    root = Path(cfg["processed_data_root"])
    filename = (
        settings.trial_input_filename
        if str(settings.trial_input_filename).endswith(".pkl")
        else f"{settings.trial_input_filename}.pkl"
    )
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


def _stable_seed(base_seed: int, *parts: str) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**32 - 1)


def _sample_rows(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= int(max_rows):
        return df
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(df), size=int(max_rows), replace=False))
    return df.iloc[picked]


def _row_counts(row, n_bins: int) -> Optional[np.ndarray]:
    counts = np.asarray(getattr(row, "psth_counts"), dtype=float).reshape(-1)
    if counts.size != n_bins:
        return None
    return counts


def _counts_to_spike_times(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    bin_size_s: float,
    *,
    jitter_within_bin: bool,
    rng: np.random.Generator,
    max_spikes_per_bin: Optional[int],
) -> np.ndarray:
    spike_bins = np.clip(np.rint(np.asarray(counts, dtype=float)).astype(int), 0, None)
    if max_spikes_per_bin is not None and int(max_spikes_per_bin) > 0:
        spike_bins = np.minimum(spike_bins, int(max_spikes_per_bin))
    if spike_bins.size != bin_centers.size:
        return np.array([], dtype=float)
    bin_indices = np.repeat(np.arange(spike_bins.size), spike_bins)
    if bin_indices.size == 0:
        return np.array([], dtype=float)
    spike_ts = np.asarray(bin_centers[bin_indices], dtype=float)
    if jitter_within_bin and float(bin_size_s) > 0:
        jitter = (rng.random(spike_ts.size) - 0.5) * float(bin_size_s)
        spike_ts = spike_ts + jitter
    return np.sort(spike_ts)


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


def _fallback_bin_centers(settings: PeriodPSTHUnitPlotSettings) -> np.ndarray:
    bin_size_s = float(settings.bin_size_ms_fallback) / 1000.0
    pre = float(settings.window_pre_s)
    post = float(settings.window_post_s)
    edges = np.arange(-pre, post + bin_size_s * 0.5, bin_size_s, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _load_trials_for_date(
    paths: Sequence[Path],
    *,
    date: str,
    settings: PeriodPSTHUnitPlotSettings,
) -> tuple[pd.DataFrame, np.ndarray]:
    all_rows: list[pd.DataFrame] = []
    bin_centers_ref: Optional[np.ndarray] = None

    for path in paths:
        obj = load_pickle_path(path)
        trials_df, meta = _extract_trials_df_and_meta(obj)
        if trials_df.empty or "psth_counts" not in trials_df.columns:
            continue

        local_centers = _resolve_bin_centers_from_meta(meta)
        if local_centers is not None:
            if bin_centers_ref is None:
                bin_centers_ref = local_centers
            elif (
                local_centers.shape != bin_centers_ref.shape
                or not np.allclose(local_centers, bin_centers_ref)
            ):
                print(f"[plot] skipping {path} due to mismatched PSTH bin centers")
                continue

        df = trials_df.copy()
        if "date" not in df.columns:
            df["date"] = date
        if "session" not in df.columns:
            session_part = next(
                (part for part in path.parts if part.startswith("session=")),
                "session=unknown",
            )
            df["session"] = session_part.split("=", 1)[1]
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame(), _fallback_bin_centers(settings)

    out_df = pd.concat(all_rows, axis=0, ignore_index=True)
    if bin_centers_ref is None:
        bin_centers_ref = _fallback_bin_centers(settings)
    return out_df, bin_centers_ref


def _resolve_figsize_and_dpi(settings: PeriodPSTHUnitPlotSettings) -> tuple[list[float], Optional[int]]:
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        figsize, cfg_dpi = resolve_figsize(plot_cfg)
    else:
        figsize, cfg_dpi = (None, None)

    if figsize is None:
        figsize = [11.0, 8.0]
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    return [float(figsize[0]), float(figsize[1])], dpi


def _condition_masks(
    df: pd.DataFrame,
    interactive_label: str,
    non_interactive_label: str,
) -> dict[str, pd.Series]:
    if "is_interactive" in df.columns:
        interactive_mask = df["is_interactive"].map(
            lambda val: _truthy_interactive(val, interactive_label),
        )
        interactive_mask = interactive_mask.fillna(False).astype(bool)
        non_interactive_mask = ~interactive_mask
        return {
            "period_interactive": interactive_mask,
            "period_non_interactive": non_interactive_mask,
        }

    if "period_state" in df.columns:
        states = df["period_state"].map(lambda val: str(val).strip().lower())
        interactive_mask = states == str(interactive_label).strip().lower()
        non_interactive_mask = states == str(non_interactive_label).strip().lower()
        return {
            "period_interactive": interactive_mask.fillna(False).astype(bool),
            "period_non_interactive": non_interactive_mask.fillna(False).astype(bool),
        }

    false_mask = pd.Series(False, index=df.index, dtype=bool)
    return {
        "period_interactive": false_mask,
        "period_non_interactive": false_mask,
    }


def _resolve_plot_sigma_bins(settings: PeriodPSTHUnitPlotSettings, bin_size_s: float) -> Optional[float]:
    if not settings.smooth_before_average:
        return None
    if float(settings.smoothing_sigma_ms) <= 0:
        raise ValueError("plot smoothing_sigma_ms must be > 0 when smoothing is enabled.")
    sigma_bins = float(settings.smoothing_sigma_ms) / (float(bin_size_s) * 1000.0)
    if sigma_bins <= 0:
        raise ValueError("resolved smoothing sigma in bins must be > 0.")
    return sigma_bins


def _build_unit_condition_payloads(
    df_unit: pd.DataFrame,
    *,
    unit_key: str,
    bin_centers: np.ndarray,
    bin_size_s: float,
    settings: PeriodPSTHUnitPlotSettings,
) -> list[dict]:
    masks = _condition_masks(
        df_unit,
        interactive_label=settings.interactive_label,
        non_interactive_label=settings.non_interactive_label,
    )
    n_bins = int(bin_centers.size)
    payload_order = [
        ("period_interactive", "Interactive"),
        ("period_non_interactive", "Non-Interactive"),
    ]
    sigma_bins = _resolve_plot_sigma_bins(settings, bin_size_s)

    payloads: list[dict] = []
    for cond_key, cond_label in payload_order:
        cond_df = df_unit.loc[masks[cond_key]].copy()
        seed = _stable_seed(settings.random_seed, unit_key, cond_key)
        cond_df = _sample_rows(cond_df, settings.max_trials_per_condition, seed)

        count_rows: list[np.ndarray] = []
        spike_rows: list[np.ndarray] = []
        for trial_i, row in enumerate(cond_df.itertuples(index=False)):
            counts = _row_counts(row, n_bins)
            if counts is None:
                continue
            count_rows.append(counts)
            trial_rng = np.random.default_rng(
                _stable_seed(settings.random_seed, unit_key, cond_key, str(trial_i)),
            )
            spike_rows.append(
                _counts_to_spike_times(
                    counts,
                    bin_centers,
                    bin_size_s,
                    jitter_within_bin=settings.raster_jitter_within_bin,
                    rng=trial_rng,
                    max_spikes_per_bin=settings.raster_max_spikes_per_bin,
                )
            )

        if not count_rows:
            payloads.append(
                {
                    "key": cond_key,
                    "label": cond_label,
                    "color": settings.condition_colors.get(cond_key, "#444444"),
                    "n_trials": 0,
                    "spike_rows": [],
                    "mean_hz": np.zeros(n_bins, dtype=float),
                    "sem_hz": np.zeros(n_bins, dtype=float),
                }
            )
            continue

        mat = np.vstack(count_rows)
        rates_hz = mat / float(bin_size_s)
        if settings.smooth_before_average:
            rates_hz = gaussian_filter1d(rates_hz, sigma=sigma_bins, axis=1, mode="nearest")
        mean_hz = np.mean(rates_hz, axis=0)
        if rates_hz.shape[0] > 1:
            sem_hz = np.std(rates_hz, axis=0, ddof=1) / np.sqrt(float(rates_hz.shape[0]))
        else:
            sem_hz = np.zeros(n_bins, dtype=float)

        payloads.append(
            {
                "key": cond_key,
                "label": cond_label,
                "color": settings.condition_colors.get(cond_key, "#444444"),
                "n_trials": int(rates_hz.shape[0]),
                "spike_rows": spike_rows,
                "mean_hz": mean_hz,
                "sem_hz": sem_hz,
            },
        )

    return payloads


def _safe_unit_filename(unit_uuid: str) -> str:
    safe = str(unit_uuid).strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace(" ", "_")
    return safe if safe else "unknown"


def _safe_region_folder(region: Optional[str]) -> str:
    if region is None:
        return "region=unknown"
    safe = str(region).strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace(" ", "_")
    safe = safe if safe else "unknown"
    return f"region={safe}"


def _darken_color(hex_color: str, factor: float) -> str:
    fac = min(max(float(factor), 0.0), 1.0)
    r, g, b = to_rgb(hex_color)
    return (r * fac, g * fac, b * fac)


def _plot_single_unit(
    *,
    df_unit: pd.DataFrame,
    date: str,
    unit_uuid: str,
    bin_centers: np.ndarray,
    settings: PeriodPSTHUnitPlotSettings,
    out_dir: Path,
    figsize: list[float],
    dpi: Optional[int],
) -> Optional[Path]:
    if bin_centers.size < 2:
        return None
    bin_size_s = float(np.mean(np.diff(bin_centers)))
    if bin_size_s <= 0:
        return None

    unit_key = f"{date}|{unit_uuid}"
    payloads = _build_unit_condition_payloads(
        df_unit,
        unit_key=unit_key,
        bin_centers=bin_centers,
        bin_size_s=bin_size_s,
        settings=settings,
    )
    if not any(payload["n_trials"] > 0 for payload in payloads):
        return None

    fig, (ax_raster, ax_rate) = plt.subplots(
        2,
        1,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                float(settings.panel_raster_height_ratio),
                float(settings.panel_rate_height_ratio),
            ],
            "hspace": 0.08,
        },
    )

    y_cursor = 1
    y_ticks: list[float] = []
    y_labels: list[str] = []
    for payload in payloads:
        n_trials = int(payload["n_trials"])
        if n_trials <= 0:
            continue
        line_offsets = np.arange(y_cursor, y_cursor + n_trials, dtype=float)
        ax_raster.eventplot(
            payload["spike_rows"],
            lineoffsets=line_offsets,
            linelengths=float(settings.raster_linelength),
            linewidths=float(settings.raster_linewidth),
            colors=[_darken_color(payload["color"], settings.raster_darkening_factor)] * n_trials,
            alpha=float(settings.raster_alpha),
            zorder=3,
        )
        if settings.raster_show_condition_background:
            ax_raster.axhspan(
                float(line_offsets[0]) - 0.5,
                float(line_offsets[-1]) + 0.5,
                color=payload["color"],
                alpha=0.07,
                zorder=0,
            )
        mid = 0.5 * (line_offsets[0] + line_offsets[-1])
        y_ticks.append(float(mid))
        y_labels.append(f"{payload['label']} (n={n_trials})")
        y_cursor += n_trials
        ax_raster.axhline(float(y_cursor) - 0.5, color="#cccccc", linewidth=0.7)

    ax_raster.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax_raster.set_ylabel("Trials")
    if y_ticks:
        ax_raster.set_yticks(y_ticks)
        ax_raster.set_yticklabels(y_labels)
        ax_raster.set_ylim(float(y_cursor) - 0.5, 0.5)
    else:
        ax_raster.set_yticks([])

    for payload in payloads:
        if int(payload["n_trials"]) <= 0:
            continue
        mean_hz = np.asarray(payload["mean_hz"], dtype=float)
        sem_hz = np.asarray(payload["sem_hz"], dtype=float)
        ax_rate.plot(bin_centers, mean_hz, color=payload["color"], label=payload["label"])
        ax_rate.fill_between(
            bin_centers,
            mean_hz - sem_hz,
            mean_hz + sem_hz,
            color=payload["color"],
            alpha=0.22,
            linewidth=0.0,
        )

    ax_rate.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax_rate.set_xlabel("Time From Period Center (s)")
    ax_rate.set_ylabel("Firing Rate (Hz)")
    ax_rate.set_xlim(float(bin_centers[0]), float(bin_centers[-1]))
    ax_rate.legend(loc="upper right", frameon=False)

    row0 = df_unit.iloc[0]
    region = _safe_optional_str(row0.get("region")) if isinstance(row0, pd.Series) else None
    channel = _safe_optional_str(row0.get("spike_channel")) if isinstance(row0, pd.Series) else None
    title_bits = [f"Date {date}", f"Unit {unit_uuid}"]
    if region:
        title_bits.append(f"Region {region}")
    if channel:
        title_bits.append(f"Channel {channel}")
    fig.suptitle(" | ".join(title_bits), y=0.99)

    ext = _ensure_ext(settings.output_extension)
    out_path = out_dir / f"date={date}__unit={_safe_unit_filename(unit_uuid)}.{ext}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("white")
    ax_raster.set_facecolor("white")
    ax_rate.set_facecolor("white")
    fig.savefig(
        out_path,
        format=ext,
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)
    return out_path


def _plot_single_unit_worker(args) -> Optional[Path]:
    df_unit, date, unit_uuid, bin_centers, settings, out_dir, figsize, dpi = args
    return _plot_single_unit(
        df_unit=df_unit,
        date=date,
        unit_uuid=unit_uuid,
        bin_centers=bin_centers,
        settings=settings,
        out_dir=out_dir,
        figsize=figsize,
        dpi=dpi,
    )


def _build_unit_plot_tasks_for_date(args):
    settings, date, paths, unit_filter, unit_key_filter = args
    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    df, bin_centers = _load_trials_for_date(paths, date=date, settings=settings)
    if df.empty or "unit_uuid" not in df.columns:
        return []

    unit_ids = sorted({str(val) for val in df["unit_uuid"].dropna().astype(str).tolist()})
    if unit_filter is not None:
        unit_ids = [unit for unit in unit_ids if unit in unit_filter]
    if settings.test_single and unit_ids:
        unit_ids = [random.choice(unit_ids)]
    if not unit_ids:
        return []

    figsize, dpi = _resolve_figsize_and_dpi(settings)
    unit_tasks = []
    for unit_uuid in unit_ids:
        if unit_key_filter is not None and f"{date}|{unit_uuid}" not in unit_key_filter:
            continue
        df_unit = df.loc[df["unit_uuid"].astype(str) == unit_uuid].copy()
        if df_unit.empty:
            continue
        region_series = (
            df_unit["region"].dropna().astype(str).map(lambda text: text.strip())
            if "region" in df_unit.columns
            else pd.Series(dtype=str)
        )
        region = None
        if not region_series.empty:
            region = next((val for val in region_series if val), None)
        unit_out_dir = out_root / _safe_region_folder(region)
        unit_tasks.append(
            (
                df_unit,
                date,
                unit_uuid,
                bin_centers,
                settings,
                unit_out_dir,
                figsize,
                dpi,
            )
        )
    return unit_tasks


def plot_period_psth_units(
    settings: PeriodPSTHUnitPlotSettings,
    *,
    dates: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    unit_uuids: Optional[Sequence[str]] = None,
    unit_keys: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Generate one period-PSTH figure per unit/date."""
    cfg = load_config(settings.cfg_path)
    trial_rows = _iter_trial_rows(cfg, settings, dates=dates, sessions=sessions)
    if not trial_rows:
        print("[plot] no period PSTH trial files found")
        return []

    grouped: dict[str, list[Path]] = {}
    for row in trial_rows:
        grouped.setdefault(str(row["date"]), []).append(Path(row["path"]))

    tasks = sorted(grouped.items(), key=lambda item: item[0])
    unit_filter = None if unit_uuids is None else {str(unit) for unit in unit_uuids}
    unit_key_filter = None if unit_keys is None else {str(key) for key in unit_keys}
    out_paths: list[Path] = []

    all_unit_tasks = []
    for date, paths in tasks:
        all_unit_tasks.extend(
            _build_unit_plot_tasks_for_date((settings, date, paths, unit_filter, unit_key_filter))
        )
    if settings.test_single and all_unit_tasks:
        all_unit_tasks = [random.choice(all_unit_tasks)]
    if not all_unit_tasks:
        return []

    use_global_unit_parallel = (
        settings.use_parallel
        and settings.parallelize_units
        and len(all_unit_tasks) >= int(settings.unit_parallel_min_units)
    )
    if use_global_unit_parallel:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        with Pool(processes=n_proc) as pool:
            for out_path in tqdm(
                pool.imap_unordered(_plot_single_unit_worker, all_unit_tasks),
                total=len(all_unit_tasks),
                desc=f"Plotting period unit PSTHs ({n_proc} workers)",
                unit="unit",
            ):
                if out_path is not None:
                    out_paths.append(out_path)
        return out_paths

    for task in tqdm(all_unit_tasks, desc="Plotting period unit PSTHs", unit="unit"):
        out_path = _plot_single_unit_worker(task)
        if out_path is not None:
            out_paths.append(out_path)
    return out_paths
