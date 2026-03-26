"""Plot one fixation's 1 ms spike trains for two simultaneously recorded regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import (
    apply_plotting_config,
    counts_to_spike_times,
    ensure_ext as _ensure_ext_shared,
    resolve_figsize,
)
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


_FIXATION_KEY_COLUMNS = (
    "fixation_agent",
    "fixation_monkey_name",
    "fixation_category",
    "fixation_location",
    "fixation_start_idx",
    "fixation_stop_idx",
    "interactive_state",
)
_REGION_LABELS = {
    "bla": "BLA",
    "accg": "ACCg",
    "dmpfc": "dmPFC",
    "ofc": "OFC",
}


@dataclass
class FixationSimultaneousSpikeTrainExamplePlotSettings:
    """Configuration for a two-region simultaneous fixation spike-train plot."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_modality: str = "psth"
    input_filename: str = "fixations_spike_train_1ms.pkl"
    output_subdir: str = "ephys/psth/fixation_neural_cross_correlation/simultaneous_spike_train_examples"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 300
    figsize: Optional[Sequence[float]] = None
    time_window_ms: Optional[Sequence[float]] = (-500.0, 500.0)
    date: str = "01312018"
    session: str = "10"
    region_a: Optional[str] = "bla"
    region_b: Optional[str] = "accg"
    fixation_category: Optional[str] = "face"
    interactive_state: Optional[str] = "non_interactive"
    fixation_start_idx: Optional[int] = 236819
    fixation_stop_idx: Optional[int] = 237720
    fixation_rank: int = 1
    min_units_per_region: int = 10
    max_units_per_region: Optional[int] = None
    line_width: float = 0.45
    line_length: float = 0.82
    title_fontsize: float = 6.6
    label_fontsize: float = 6.0
    tick_fontsize: float = 5.4
    panel_wspace: float = 0.28


def _ensure_ext(ext: str, *, fallback: str) -> str:
    return _ensure_ext_shared(ext, fallback=fallback)


def _safe_token(value: object, *, fallback: str) -> str:
    text = str(value).strip()
    if not text:
        return fallback
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    out = out.strip("_")
    return out or fallback


def _region_label(region: object) -> str:
    token = str(region).strip().lower()
    return _REGION_LABELS.get(token, str(region))


def _normalize_optional_token(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _build_session_input_path(
    cfg: dict,
    settings: FixationSimultaneousSpikeTrainExamplePlotSettings,
) -> Path:
    return (
        Path(cfg["processed_data_root"])
        / f"date={settings.date}"
        / f"session={settings.session}"
        / str(settings.input_modality)
        / str(settings.input_filename)
    )


def _resolve_bin_centers_s(meta: dict) -> np.ndarray:
    centers = meta.get("spike_train_bin_centers_s_rel")
    if centers is not None:
        arr = np.asarray(centers, dtype=float).reshape(-1)
        if arr.size > 0:
            return arr

    edges = meta.get("spike_train_bin_edges_s_rel")
    if edges is not None:
        arr = np.asarray(edges, dtype=float).reshape(-1)
        if arr.size >= 2:
            return 0.5 * (arr[:-1] + arr[1:])

    raise ValueError("Unable to resolve spike-train bin centers from session metadata.")


def _resolve_regions(
    df: pd.DataFrame,
    *,
    region_a: Optional[str],
    region_b: Optional[str],
) -> tuple[str, str]:
    if region_a and region_b:
        return str(region_a), str(region_b)
    unit_counts = df.groupby("region")["unit_uuid"].nunique().sort_values(ascending=False)
    if unit_counts.size < 2:
        raise ValueError("Session does not contain at least two recorded regions.")
    top_regions = [str(unit_counts.index[0]), str(unit_counts.index[1])]
    return top_regions[0], top_regions[1]


def build_candidate_fixation_table(
    df: pd.DataFrame,
    *,
    region_a: str,
    region_b: str,
    fixation_category: Optional[str],
    interactive_state: Optional[str],
    min_units_per_region: int,
) -> pd.DataFrame:
    category_filter = _normalize_optional_token(fixation_category)
    interactive_filter = _normalize_optional_token(interactive_state)
    sub = df.loc[df["region"].isin([region_a, region_b])].copy()
    rows: list[dict[str, object]] = []
    for key_values, group_df in sub.groupby(list(_FIXATION_KEY_COLUMNS), dropna=False):
        by_region = group_df.groupby("region")["unit_uuid"].nunique()
        if str(region_a) not in by_region.index or str(region_b) not in by_region.index:
            continue
        n_a = int(by_region[str(region_a)])
        n_b = int(by_region[str(region_b)])
        if n_a < int(min_units_per_region) or n_b < int(min_units_per_region):
            continue
        fixation_category_value = str(key_values[2])
        interactive_value = str(key_values[6])
        if category_filter is not None and fixation_category_value != category_filter:
            continue
        if interactive_filter is not None and interactive_value != interactive_filter:
            continue
        total_spikes = int(
            sum(int(np.asarray(counts, dtype=float).reshape(-1).sum()) for counts in group_df["spike_train_counts"])
        )
        example_counts = np.asarray(group_df.iloc[0]["spike_train_counts"], dtype=float).reshape(-1)
        rows.append(
            {
                "fixation_agent": key_values[0],
                "fixation_monkey_name": key_values[1],
                "fixation_category": fixation_category_value,
                "fixation_location": key_values[3],
                "fixation_start_idx": int(key_values[4]),
                "fixation_stop_idx": int(key_values[5]),
                "interactive_state": interactive_value,
                "region_a": str(region_a),
                "region_b": str(region_b),
                "region_a_units": n_a,
                "region_b_units": n_b,
                "duration_ms": int(example_counts.size),
                "total_spikes": total_spikes,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["total_spikes", "duration_ms", "fixation_start_idx"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    out.insert(0, "fixation_rank", np.arange(1, len(out) + 1, dtype=int))
    return out


def _select_fixation(candidates: pd.DataFrame, settings: FixationSimultaneousSpikeTrainExamplePlotSettings) -> pd.Series:
    if candidates.empty:
        raise ValueError("No fixation candidates found for the requested session/region filters.")
    start_idx = settings.fixation_start_idx
    stop_idx = settings.fixation_stop_idx
    if start_idx is not None and stop_idx is not None:
        selected = candidates.loc[
            (candidates["fixation_start_idx"].astype(int) == int(start_idx))
            & (candidates["fixation_stop_idx"].astype(int) == int(stop_idx))
        ]
        if not selected.empty:
            return selected.iloc[0]
        raise ValueError(
            "Requested fixation_start_idx/fixation_stop_idx was not found in the candidate fixation table."
        )

    rank = max(1, int(settings.fixation_rank))
    selected = candidates.loc[candidates["fixation_rank"].astype(int) == rank]
    if not selected.empty:
        return selected.iloc[0]
    return candidates.iloc[0]


def _filter_selected_fixation_rows(
    df: pd.DataFrame,
    *,
    selected_fixation: pd.Series,
    region_a: str,
    region_b: str,
) -> pd.DataFrame:
    mask = df["region"].isin([str(region_a), str(region_b)])
    mask &= df["fixation_start_idx"].astype(int) == int(selected_fixation["fixation_start_idx"])
    mask &= df["fixation_stop_idx"].astype(int) == int(selected_fixation["fixation_stop_idx"])
    mask &= df["fixation_category"].astype(str) == str(selected_fixation["fixation_category"])
    mask &= df["interactive_state"].astype(str) == str(selected_fixation["interactive_state"])
    if "fixation_agent" in df.columns and pd.notna(selected_fixation.get("fixation_agent")):
        mask &= df["fixation_agent"].astype(str) == str(selected_fixation["fixation_agent"])
    if "fixation_monkey_name" in df.columns and pd.notna(selected_fixation.get("fixation_monkey_name")):
        mask &= df["fixation_monkey_name"].astype(str) == str(selected_fixation["fixation_monkey_name"])
    return df.loc[mask].copy()


def _select_region_units(
    fixation_df: pd.DataFrame,
    *,
    region: str,
    max_units_per_region: Optional[int],
) -> pd.DataFrame:
    region_df = fixation_df.loc[fixation_df["region"].astype(str) == str(region)].copy()
    if region_df.empty:
        return region_df
    region_df["total_spikes"] = region_df["spike_train_counts"].map(
        lambda counts: int(np.asarray(counts, dtype=float).reshape(-1).sum())
    )
    region_df = region_df.sort_values(
        ["total_spikes", "spike_channel", "unit_uuid"],
        ascending=[False, True, True],
    )
    if max_units_per_region is not None and int(max_units_per_region) > 0:
        region_df = region_df.head(int(max_units_per_region))
    return region_df.reset_index(drop=True)


def _build_output_path(
    cfg: dict,
    settings: FixationSimultaneousSpikeTrainExamplePlotSettings,
    *,
    region_a: str,
    region_b: str,
    selected_fixation: pd.Series,
    ext: str,
) -> Path:
    root = build_analysis_output_dir(cfg, settings.output_subdir)
    file_name = (
        f"date={_safe_token(settings.date, fallback='date')}"
        f"__session={_safe_token(settings.session, fallback='session')}"
        f"__regions={_safe_token(region_a, fallback='regionA')}__vs__{_safe_token(region_b, fallback='regionB')}"
        f"__fixation={int(selected_fixation['fixation_start_idx'])}-{int(selected_fixation['fixation_stop_idx'])}"
        f".{ext}"
    )
    return root / file_name


def _resolve_figsize_and_dpi(
    settings: FixationSimultaneousSpikeTrainExamplePlotSettings,
) -> tuple[list[float], Optional[int]]:
    cfg_figsize = None
    cfg_dpi = None
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        cfg_figsize, cfg_dpi = resolve_figsize(plot_cfg)
    figsize = list(settings.figsize) if settings.figsize is not None else None
    if figsize is None:
        figsize = list(cfg_figsize) if cfg_figsize is not None else [2.1, 3.5]
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    return [float(figsize[0]), float(figsize[1])], dpi


def _resolve_time_window_ms(time_axis_ms: np.ndarray, window_ms: Optional[Sequence[float]]) -> tuple[float, float]:
    if window_ms is None or len(window_ms) != 2:
        return float(time_axis_ms[0]), float(time_axis_ms[-1])
    start_ms = float(window_ms[0])
    stop_ms = float(window_ms[1])
    if stop_ms <= start_ms:
        return float(time_axis_ms[0]), float(time_axis_ms[-1])
    return max(start_ms, float(time_axis_ms[0])), min(stop_ms, float(time_axis_ms[-1]))


def _resolve_neuron_ticks(num_units: int) -> np.ndarray:
    if num_units <= 0:
        return np.asarray([], dtype=int)
    if num_units <= 5:
        return np.arange(1, num_units + 1, dtype=int)
    approx_ticks = np.linspace(1, num_units, num=min(5, num_units), dtype=float)
    ticks = np.unique(np.clip(np.round(approx_ticks).astype(int), 1, num_units))
    if ticks[0] != 1:
        ticks = np.insert(ticks, 0, 1)
    if ticks[-1] != num_units:
        ticks = np.append(ticks, int(num_units))
    return ticks


def plot_fixation_simultaneous_spike_train_example(
    settings: FixationSimultaneousSpikeTrainExamplePlotSettings,
) -> dict[str, object]:
    cfg = load_config(settings.cfg_path)
    input_path = _build_session_input_path(cfg, settings)
    session_obj = load_pickle_path(input_path)
    if not isinstance(session_obj, dict):
        raise ValueError(f"Unexpected fixation spike-train session object at {input_path}")
    meta = session_obj.get("meta", {}) if isinstance(session_obj, dict) else {}
    df = session_obj.get("trials")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"No trial rows found in {input_path}")

    region_a, region_b = _resolve_regions(
        df,
        region_a=settings.region_a,
        region_b=settings.region_b,
    )
    candidates = build_candidate_fixation_table(
        df,
        region_a=region_a,
        region_b=region_b,
        fixation_category=settings.fixation_category,
        interactive_state=settings.interactive_state,
        min_units_per_region=settings.min_units_per_region,
    )
    selected_fixation = _select_fixation(candidates, settings)
    fixation_df = _filter_selected_fixation_rows(
        df,
        selected_fixation=selected_fixation,
        region_a=region_a,
        region_b=region_b,
    )
    if fixation_df.empty:
        raise ValueError("Selected fixation produced no trial rows.")

    bin_centers_s = _resolve_bin_centers_s(meta)
    bin_size_s = float(meta.get("spike_train_bin_size_ms", 1.0)) / 1000.0
    time_axis_ms = bin_centers_s * 1000.0
    region_dfs = [
        _select_region_units(fixation_df, region=region_a, max_units_per_region=settings.max_units_per_region),
        _select_region_units(fixation_df, region=region_b, max_units_per_region=settings.max_units_per_region),
    ]
    xlim_start_ms, xlim_stop_ms = _resolve_time_window_ms(time_axis_ms, settings.time_window_ms)

    figsize, dpi = _resolve_figsize_and_dpi(settings)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharex=True, squeeze=False, facecolor="white")
    plot_axes = list(np.ravel(axes))

    for axis, region, region_df in zip(plot_axes, [region_a, region_b], region_dfs):
        spike_times_ms: list[np.ndarray] = []
        for row in region_df.itertuples(index=False):
            counts = np.asarray(row.spike_train_counts, dtype=float).reshape(-1)
            spike_times = counts_to_spike_times(
                counts,
                bin_centers_s,
                bin_size_s,
                jitter_within_bin=False,
                rng=np.random.default_rng(0),
                max_spikes_per_bin=None,
            )
            spike_times_ms_unit = spike_times * 1000.0
            spike_times_ms_unit = spike_times_ms_unit[
                (spike_times_ms_unit >= xlim_start_ms) & (spike_times_ms_unit <= xlim_stop_ms)
            ]
            spike_times_ms.append(spike_times_ms_unit)

        lineoffsets = np.arange(1, len(spike_times_ms) + 1, dtype=float)
        axis.eventplot(
            spike_times_ms,
            orientation="horizontal",
            lineoffsets=lineoffsets,
            linelengths=float(settings.line_length),
            linewidths=float(settings.line_width),
            colors="#111111",
        )
        axis.axvline(0.0, color="#7a7a7a", linestyle="--", linewidth=0.65)
        axis.set_xlim(xlim_start_ms, xlim_stop_ms)
        axis.set_ylim(0.5, max(1.5, len(spike_times_ms) + 0.5))
        neuron_ticks = _resolve_neuron_ticks(len(spike_times_ms))
        axis.set_yticks(neuron_ticks.astype(float))
        axis.set_yticklabels([str(int(tick)) for tick in neuron_ticks])
        axis.set_ylabel("Neuron #", fontsize=float(settings.label_fontsize), labelpad=1.5)
        axis.set_title(
            f"{_region_label(region)} ({len(region_df)}/{int(fixation_df.loc[fixation_df['region'].astype(str) == str(region), 'unit_uuid'].nunique())} units)",
            loc="left",
            fontsize=float(settings.title_fontsize),
            pad=2.0,
        )
        axis.tick_params(axis="x", labelsize=float(settings.tick_fontsize), length=2.0, pad=1.0)
        axis.tick_params(axis="y", labelsize=float(settings.tick_fontsize), length=2.0, pad=1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for axis in plot_axes:
        axis.set_xlabel("Time from fixation onset (ms)", fontsize=float(settings.label_fontsize), labelpad=1.5)
        axis.set_xticks([xlim_start_ms, 0.0, xlim_stop_ms])
        axis.set_xticklabels([f"{int(round(xlim_start_ms))}", "0", f"{int(round(xlim_stop_ms))}"])

    fig.suptitle(
        f"{settings.date} s{settings.session} | {selected_fixation['fixation_category']} | {selected_fixation['interactive_state']}\n"
        f"fix {int(selected_fixation['fixation_start_idx'])}-{int(selected_fixation['fixation_stop_idx'])}",
        fontsize=float(settings.title_fontsize),
        y=0.985,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.82, bottom=0.22, wspace=float(settings.panel_wspace))

    ext = _ensure_ext(settings.output_extension, fallback="pdf")
    output_path = _build_output_path(
        cfg,
        settings,
        region_a=region_a,
        region_b=region_b,
        selected_fixation=selected_fixation,
        ext=ext,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(
        fig,
        output_path,
        ext=ext,
        dpi=dpi,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "region_a": region_a,
        "region_b": region_b,
        "selected_fixation": selected_fixation.to_dict(),
        "candidate_fixations": candidates,
        "region_a_plotted_units": int(len(region_dfs[0])),
        "region_b_plotted_units": int(len(region_dfs[1])),
    }
