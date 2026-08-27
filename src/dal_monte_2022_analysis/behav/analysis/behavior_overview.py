"""Behavior overview summaries, statistics, and figures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from scipy import stats

from dal_monte_2022_analysis.config.load import load_config, resolve_dataset_cfg_path
from dal_monte_2022_analysis.core.behav.analysis_primitives import build_interactive_mask, to_bool
from dal_monte_2022_analysis.core.behav.roi_geometry import normalize_roi_rect_bounds
from dal_monte_2022_analysis.data.loaders.behavioral import index_behavioral_processed_data_from_cfg
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path


SAMPLE_RATE_HZ = 1000.0
HEATMAP_BINS = (180, 140)
HEATMAP_RANGE = [[-2.5, 2.5], [-2.2, 1.8]]
AGENTS = ("m1", "m2")
PHASES = ("interactive", "non_interactive")
PHASE_COLORS = {"interactive": "#b64198", "non_interactive": "#97ca3d"}
PROBABILITY_LABELS = {
    ("m1", "face"): "M1 face",
    ("m2", "face"): "M2 face",
    ("m1", "object"): "M1 object",
}


def behavior_overview_output_dir(cfg_path: str = "configs/project.yaml") -> Path:
    """Return the configured local-data output directory for behavior overview files."""
    cfg = load_config(resolve_dataset_cfg_path(cfg_path))
    return build_analysis_output_dir(cfg, "behavior_overview")


def apply_manuscript_style() -> None:
    """Apply compact, editable matplotlib defaults for manuscript figures."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.bbox": "tight",
            "savefig.transparent": True,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int = 300) -> None:
    """Save a figure as editable PDF and high-resolution PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _disable_clipping(fig)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi)


def _disable_clipping(artist: mpl.artist.Artist) -> None:
    """Disable artist clipping to keep exported PDFs easier to edit in Illustrator."""
    if hasattr(artist, "set_clip_on"):
        artist.set_clip_on(False)
    if hasattr(artist, "set_clip_path"):
        artist.set_clip_path(None)
    if hasattr(artist, "set_clip_box"):
        artist.set_clip_box(None)
    for child in artist.get_children():
        _disable_clipping(child)


def _index_by_key(cfg: dict, modality: str) -> dict[tuple[str, str, str | None], Path]:
    df = index_behavioral_processed_data_from_cfg(cfg, modality)
    return {
        (str(row.date), str(row.session), row.agent if pd.notna(row.agent) else None): Path(row.path)
        for row in df.itertuples(index=False)
    }


def _shared_index_by_key(cfg: dict, modality: str) -> dict[tuple[str, str], Path]:
    df = index_behavioral_processed_data_from_cfg(cfg, modality)
    out = {}
    for row in df.itertuples(index=False):
        if pd.isna(row.agent):
            out[(str(row.date), str(row.session))] = Path(row.path)
    return out


def _duration_stats(values: Iterable[float]) -> dict[str, float]:
    s = pd.Series(list(values), dtype=float).dropna()
    if s.empty:
        return {
            "n": 0,
            "mean_s": np.nan,
            "median_s": np.nan,
            "sd_s": np.nan,
            "min_s": np.nan,
            "q25_s": np.nan,
            "q75_s": np.nan,
            "max_s": np.nan,
        }
    return {
        "n": int(s.size),
        "mean_s": float(s.mean()),
        "median_s": float(s.median()),
        "sd_s": float(s.std(ddof=1)),
        "min_s": float(s.min()),
        "q25_s": float(s.quantile(0.25)),
        "q75_s": float(s.quantile(0.75)),
        "max_s": float(s.max()),
    }


def _interval_mask(events: pd.DataFrame, n_samples: int) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    if events is None or events.empty:
        return mask
    for row in events.itertuples(index=False):
        start = int(max(0, getattr(row, "start")))
        stop = int(min(n_samples - 1, getattr(row, "stop")))
        if start <= stop:
            mask[start : stop + 1] = True
    return mask


def _rect_corners(bounds: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=float)


def _roi_center(rois: dict, name: str) -> np.ndarray | None:
    rect = rois.get(name)
    if rect is None:
        return None
    bounds = normalize_roi_rect_bounds(rect)
    if bounds is None:
        return None
    x_min, x_max, y_min, y_max = bounds
    return np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=float)


def _m1_affine(rois: dict) -> tuple[np.ndarray, np.ndarray] | None:
    src = []
    dst = []
    mapping = {
        "face": (0.0, 0.0),
        "left_nonsocial_object": (-1.0, -1.0),
        "right_nonsocial_object": (1.0, -1.0),
    }
    for roi_name, target in mapping.items():
        center = _roi_center(rois, roi_name)
        if center is None:
            return None
        src.append([center[0], center[1], 1.0])
        dst.append(target)
    mat, *_ = np.linalg.lstsq(np.asarray(src, dtype=float), np.asarray(dst, dtype=float), rcond=None)
    return mat[:2, :], mat[2, :]


def _m2_face_scale(rois: dict) -> tuple[np.ndarray, np.ndarray] | None:
    bounds = normalize_roi_rect_bounds(rois.get("face"))
    if bounds is None:
        return None
    x_min, x_max, y_min, y_max = bounds
    center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=float)
    scale = max(x_max - x_min, y_max - y_min)
    if not np.isfinite(scale) or scale <= 0:
        return None
    return np.eye(2) / scale, -center / scale


def _transform_points(x: np.ndarray, y: np.ndarray, mat: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.column_stack([x, y]) @ mat + offset
    return out[:, 0], out[:, 1]


def _transform_roi(bounds: tuple[float, float, float, float], mat: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return _rect_corners(bounds) @ mat + offset


def _transform_for_agent(agent: str, rois: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if agent == "m1":
        return _m1_affine(rois)
    return _m2_face_scale(rois)


def _update_hist(hist: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return
    h, _, _ = np.histogram2d(
        y[valid],
        x[valid],
        bins=HEATMAP_BINS[::-1],
        range=[HEATMAP_RANGE[1], HEATMAP_RANGE[0]],
    )
    hist += h


def _summarize_periods(period_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = pd.concat(period_frames, ignore_index=True)
    periods["duration_samples"] = periods["stop"].astype(int) - periods["start"].astype(int) + 1
    periods["duration_s"] = periods["duration_samples"] / SAMPLE_RATE_HZ
    duration_summary = pd.DataFrame(
        [{"state": state, **_duration_stats(group["duration_s"])} for state, group in periods.groupby("state", sort=True)]
    )
    return periods, duration_summary


def _probability_rows(session_keys: list[tuple[str, str]], interactive_paths: dict, fixation_paths: dict) -> list[dict]:
    rows = []
    for date, session in session_keys:
        period_df = load_pickle_path(interactive_paths[(date, session)])
        vectors = {}
        names = {}
        n_samples = None
        for agent in AGENTS:
            obj = load_pickle_path(fixation_paths[(date, session, agent)])
            vectors[(agent, "face")] = to_bool(obj.vectors["face"])
            names[agent] = obj.context.monkey_name
            if agent == "m1":
                vectors[(agent, "object")] = to_bool(obj.vectors["object"])
            n_samples = len(vectors[(agent, "face")]) if n_samples is None else min(n_samples, len(vectors[(agent, "face")]))
        for key in list(vectors):
            vectors[key] = vectors[key][:n_samples]
        masks = {
            "whole": np.ones(n_samples, dtype=bool),
            "interactive": build_interactive_mask(period_df, n_samples=n_samples, state_label="interactive"),
            "non_interactive": build_interactive_mask(period_df, n_samples=n_samples, state_label="non_interactive"),
        }
        for phase, mask in masks.items():
            denom = int(np.count_nonzero(mask))
            for agent in AGENTS:
                count = int(np.count_nonzero(vectors[(agent, "face")] & mask))
                rows.append(
                    {
                        "date": date,
                        "session": session,
                        "phase": phase,
                        "agent": agent,
                        "monkey_name": names[agent],
                        "roi": "face",
                        "probability_type": PROBABILITY_LABELS[(agent, "face")],
                        "n_samples": denom,
                        "fixation_samples": count,
                        "probability": count / denom if denom else np.nan,
                    }
                )
            count = int(np.count_nonzero(vectors[("m1", "object")] & mask))
            rows.append(
                {
                    "date": date,
                    "session": session,
                    "phase": phase,
                    "agent": "m1",
                    "monkey_name": names["m1"],
                    "roi": "object",
                    "probability_type": PROBABILITY_LABELS[("m1", "object")],
                    "n_samples": denom,
                    "fixation_samples": count,
                    "probability": count / denom if denom else np.nan,
                }
            )
    return rows


def summarize_probabilities(prob_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize within-session fixation probabilities by phase and probability type."""
    summary = (
        prob_df.groupby(["phase", "agent", "roi", "probability_type"], dropna=False)
        .agg(
            n_sessions=("probability", "count"),
            mean_probability=("probability", "mean"),
            median_probability=("probability", "median"),
            sd_probability=("probability", "std"),
            total_fixation_samples=("fixation_samples", "sum"),
            total_samples=("n_samples", "sum"),
        )
        .reset_index()
    )
    summary["pooled_probability"] = summary["total_fixation_samples"] / summary["total_samples"]
    return summary


def compare_interactive_probabilities(prob_df: pd.DataFrame) -> pd.DataFrame:
    """Paired session-level tests for interactive vs non-interactive probabilities."""
    rows = []
    for prob_type, group in prob_df[prob_df["phase"].isin(PHASES)].groupby("probability_type", sort=False):
        wide = group.pivot_table(index=["date", "session"], columns="phase", values="probability", aggfunc="first")
        wide = wide.dropna(subset=list(PHASES))
        interactive = wide["interactive"].to_numpy(dtype=float)
        non_interactive = wide["non_interactive"].to_numpy(dtype=float)
        diff = interactive - non_interactive
        if len(diff) == 0:
            continue
        try:
            wilcoxon = stats.wilcoxon(interactive, non_interactive, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan
        ttest = stats.ttest_rel(interactive, non_interactive, nan_policy="omit")
        rows.append(
            {
                "probability_type": prob_type,
                "n_sessions": int(len(diff)),
                "interactive_mean": float(np.mean(interactive)),
                "non_interactive_mean": float(np.mean(non_interactive)),
                "mean_difference": float(np.mean(diff)),
                "median_difference": float(np.median(diff)),
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
                "paired_t_statistic": float(ttest.statistic),
                "paired_t_p": float(ttest.pvalue),
            }
        )
    return pd.DataFrame(rows)


def significance_stars(p_value: float) -> str:
    """Convert a p-value to manuscript-style significance stars."""
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def plot_period_counts(period_count_summary: pd.DataFrame, out_dir: Path) -> None:
    """Plot interactive and non-interactive period counts."""
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.bar(
        period_count_summary["state"],
        period_count_summary["n_periods"],
        color=[PHASE_COLORS.get(state, "0.5") for state in period_count_summary["state"]],
        width=0.65,
    )
    ax.set_ylabel("period count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, out_dir, "period_counts")
    plt.close(fig)


def plot_period_duration_boxplot(periods_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot period duration distributions by state."""
    states = list(PHASES)
    data = [periods_df.loc[periods_df["state"] == state, "duration_s"].dropna() for state in states]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.boxplot(data, labels=states, showfliers=False, widths=0.55)
    ax.set_ylabel("duration (s)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, out_dir, "period_duration_boxplot")
    plt.close(fig)


def plot_fixation_probability_comparison(prob_df: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot paired interactive vs non-interactive fixation probability comparisons."""
    plot_df = prob_df[prob_df["phase"].isin(PHASES)].copy()
    labels = list(PROBABILITY_LABELS.values())
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    for phase_idx, phase in enumerate(PHASES):
        vals = [
            plot_df[(plot_df["phase"] == phase) & (plot_df["probability_type"] == label)]["probability"].mean()
            for label in labels
        ]
        sems = [
            plot_df[(plot_df["phase"] == phase) & (plot_df["probability_type"] == label)]["probability"].sem()
            for label in labels
        ]
        offset = (phase_idx - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=PHASE_COLORS[phase],
            label=phase.replace("_", "-"),
            yerr=sems,
            capsize=2,
            linewidth=0.6,
            edgecolor="black",
        )
    for idx, label in enumerate(labels):
        stat_row = stats_df.loc[stats_df["probability_type"] == label]
        if stat_row.empty:
            continue
        y_max = plot_df.loc[plot_df["probability_type"] == label, "probability"].quantile(0.98)
        y = max(float(y_max), 0.02) + 0.015
        ax.plot([idx - width / 2, idx + width / 2], [y, y], color="black", linewidth=0.8)
        ax.text(idx, y + 0.004, significance_stars(float(stat_row.iloc[0]["wilcoxon_p"])), ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("fixation probability")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "fixation_probability_interactive_vs_non_interactive")
    plt.close(fig)


def _averaged_roi_polys(roi_polys: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {roi_name: np.mean(np.stack(polys), axis=0) for roi_name, polys in roi_polys.items() if polys}


def _draw_roi_overlays(ax: plt.Axes, roi_polys: dict[str, np.ndarray]) -> None:
    for roi_name, poly in roi_polys.items():
        color = "#00a6d6" if roi_name == "face" else "#1b9e3f"
        ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=color, linewidth=1.1))
        label_x = float(poly[:, 0].max()) + 0.04
        label_y = float(poly[:, 1].mean())
        ax.text(
            label_x,
            label_y,
            roi_name.replace("_nonsocial_", "_"),
            color=color,
            fontsize=6,
            ha="left",
            va="center",
        )


def _format_heatmap_tick(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.01:
        return f"{value:.1e}"
    if abs(value) < 1:
        return f"{value:.3f}"
    return f"{value:.0f}"


def _add_three_tick_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    image: mpl.image.AxesImage,
    *,
    ticks: list[float],
    labels: list[str] | None = None,
) -> None:
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(labels if labels is not None else [_format_heatmap_tick(tick) for tick in ticks])
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    colorbar.outline.set_linewidth(0.6)


def _normalized_display_heatmap(data: np.ndarray, *, vmax: float) -> np.ndarray:
    if vmax <= 0:
        return np.zeros_like(data, dtype=float)
    return np.clip(np.log1p(data) / np.log1p(vmax), 0.0, 1.0)


def _normalized_difference_heatmap(diff: np.ndarray) -> np.ndarray:
    limit = max(float(np.nanpercentile(np.abs(diff), 99.5)), np.finfo(float).eps)
    return np.clip(diff / limit, -1.0, 1.0)


def _show_heatmap(ax: plt.Axes, data: np.ndarray, *, vmax: float, title: str, cmap: str = "magma") -> mpl.image.AxesImage:
    display_data = _normalized_display_heatmap(data, vmax=vmax)
    image = ax.imshow(
        display_data,
        origin="lower",
        extent=[HEATMAP_RANGE[0][0], HEATMAP_RANGE[0][1], HEATMAP_RANGE[1][0], HEATMAP_RANGE[1][1]],
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("normalized x")
    ax.set_ylabel("normalized y")
    ax.invert_yaxis()
    return image


def _save_single_heatmap(
    heatmap: np.ndarray,
    roi_polys: dict[str, np.ndarray],
    *,
    agent: str,
    phase: str,
    vmax: float,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    image = _show_heatmap(ax, heatmap, vmax=vmax, title=f"{agent.upper()} fixations, {phase.replace('_', '-')}")
    _draw_roi_overlays(ax, roi_polys)
    _add_three_tick_colorbar(
        fig,
        ax,
        image,
        ticks=[0.0, 0.5, 1.0],
        labels=["0", "0.5", "1"],
    )
    fig.tight_layout()
    save_figure(fig, out_dir, f"heatmap_{agent}_fixations_{phase}", dpi=300)
    plt.close(fig)


def _normalized_heatmap(heatmap: np.ndarray) -> np.ndarray:
    total = float(np.sum(heatmap))
    if total <= 0:
        return np.zeros_like(heatmap, dtype=float)
    return heatmap / total


def _heatmap_bin_centers() -> np.ndarray:
    x_edges = np.linspace(HEATMAP_RANGE[0][0], HEATMAP_RANGE[0][1], HEATMAP_BINS[0] + 1)
    y_edges = np.linspace(HEATMAP_RANGE[1][0], HEATMAP_RANGE[1][1], HEATMAP_BINS[1] + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    xx, yy = np.meshgrid(x_centers, y_centers)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _roi_heatmap_fraction(heatmap: np.ndarray, roi_poly: np.ndarray | None) -> float:
    if roi_poly is None or np.sum(heatmap) <= 0:
        return np.nan
    inside = MplPath(roi_poly).contains_points(_heatmap_bin_centers()).reshape(heatmap.shape)
    return float(heatmap[inside].sum() / heatmap.sum())


def _plot_heatmap_comparison(
    heatmaps: dict[str, np.ndarray],
    roi_polys: dict[str, np.ndarray],
    *,
    agent: str,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    vmax = max(
        float(np.nanpercentile(heatmap, 99.5))
        for heatmap in heatmaps.values()
        if heatmap.sum() > 0
    )
    vmax = max(vmax, 1.0)
    for ax, phase in zip(axes[:2], PHASES):
        image = _show_heatmap(ax, heatmaps[phase], vmax=vmax, title=phase.replace("_", "-"))
        _draw_roi_overlays(ax, roi_polys)
        _add_three_tick_colorbar(
            fig,
            ax,
            image,
            ticks=[0.0, 0.5, 1.0],
            labels=["0", "0.5", "1"],
        )

    diff = _normalized_difference_heatmap(heatmaps["interactive"] - heatmaps["non_interactive"])
    diff_image = axes[2].imshow(
        diff,
        origin="lower",
        extent=[HEATMAP_RANGE[0][0], HEATMAP_RANGE[0][1], HEATMAP_RANGE[1][0], HEATMAP_RANGE[1][1]],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    axes[2].set_title("interactive - non-interactive", fontsize=9)
    axes[2].set_xlabel("normalized x")
    axes[2].set_ylabel("normalized y")
    axes[2].invert_yaxis()
    _draw_roi_overlays(axes[2], roi_polys)
    _add_three_tick_colorbar(
        fig,
        axes[2],
        diff_image,
        ticks=[-1.0, 0.0, 1.0],
        labels=["-1", "0", "1"],
    )
    fig.suptitle(f"{agent.upper()} all-session fixation heatmap comparison", fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_dir, f"heatmap_{agent}_fixations_interactive_vs_non_interactive", dpi=300)
    plt.close(fig)


def plot_all_session_heatmaps(
    heatmaps: dict[str, np.ndarray],
    roi_polys: dict[str, np.ndarray],
    *,
    agent: str,
    out_dir: Path,
) -> None:
    """Plot all-session fixation heatmaps and phase comparison maps."""
    vmax = max(
        float(np.nanpercentile(heatmap, 99.5))
        for heatmap in heatmaps.values()
        if heatmap.sum() > 0
    )
    vmax = max(vmax, 1.0)
    for phase in PHASES:
        _save_single_heatmap(heatmaps[phase], roi_polys, agent=agent, phase=phase, vmax=vmax, out_dir=out_dir)
    _plot_heatmap_comparison(heatmaps, roi_polys, agent=agent, out_dir=out_dir)


def summarize_heatmap_face_density(
    heatmaps_by_agent: dict[str, dict[str, np.ndarray]],
    roi_polys_by_agent: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Summarize all-session fixation heat falling within the canonical face ROI."""
    rows = []
    for agent in AGENTS:
        face_poly = roi_polys_by_agent.get(agent, {}).get("face")
        phase_rows = {}
        for phase in PHASES:
            heatmap = heatmaps_by_agent[agent][phase]
            phase_rows[phase] = {
                "agent": agent,
                "phase": phase,
                "total_fixation_heat": float(heatmap.sum()),
                "face_heat_fraction": _roi_heatmap_fraction(heatmap, face_poly),
            }
            rows.append(phase_rows[phase])
        rows.append(
            {
                "agent": agent,
                "phase": "interactive_minus_non_interactive",
                "total_fixation_heat": phase_rows["interactive"]["total_fixation_heat"]
                - phase_rows["non_interactive"]["total_fixation_heat"],
                "face_heat_fraction": phase_rows["interactive"]["face_heat_fraction"]
                - phase_rows["non_interactive"]["face_heat_fraction"],
            }
        )
    return pd.DataFrame(rows)


def build_fixation_heatmaps_by_phase(
    *,
    session_keys: list[tuple[str, str]],
    gaze_paths: dict[tuple[str, str, str | None], Path],
    roi_paths: dict[tuple[str, str, str | None], Path],
    fixation_event_paths: dict[tuple[str, str, str | None], Path],
    interactive_paths: dict[tuple[str, str], Path],
    out_dir: Path,
) -> None:
    """Build all-session fixation-only gaze heatmaps separately for interactive states."""
    heatmaps = {
        agent: {phase: np.zeros(HEATMAP_BINS[::-1], dtype=float) for phase in PHASES}
        for agent in AGENTS
    }
    roi_polys = {agent: defaultdict(list) for agent in AGENTS}
    for idx, (date, session) in enumerate(session_keys, start=1):
        period_df = load_pickle_path(interactive_paths[(date, session)])
        for agent in AGENTS:
            pos = load_pickle_path(gaze_paths[(date, session, agent)])
            rois = load_pickle_path(roi_paths[(date, session, agent)]).rois
            transform = _transform_for_agent(agent, rois)
            if transform is None:
                continue
            mat, offset = transform
            tx, ty = _transform_points(np.asarray(pos.x, dtype=float), np.asarray(pos.y, dtype=float), mat, offset)
            n_samples = len(tx)
            fixation_mask = _interval_mask(load_pickle_path(fixation_event_paths[(date, session, agent)]), n_samples)
            phase_masks = {phase: build_interactive_mask(period_df, n_samples=n_samples, state_label=phase) for phase in PHASES}
            for phase, phase_mask in phase_masks.items():
                mask = fixation_mask & phase_mask
                _update_hist(heatmaps[agent][phase], tx[mask], ty[mask])
            roi_names = ["face"] if agent == "m2" else ["face", "left_nonsocial_object", "right_nonsocial_object"]
            for roi_name in roi_names:
                bounds = normalize_roi_rect_bounds(rois.get(roi_name))
                if bounds is None:
                    continue
                poly = _transform_roi(bounds, mat, offset)
                roi_polys[agent][roi_name].append(poly)
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(session_keys)} sessions for heatmaps")
    averaged_roi_polys = {agent: _averaged_roi_polys(roi_polys[agent]) for agent in AGENTS}
    for agent in AGENTS:
        plot_all_session_heatmaps(heatmaps[agent], averaged_roi_polys[agent], agent=agent, out_dir=out_dir)
    summarize_heatmap_face_density(heatmaps, averaged_roi_polys).to_csv(
        out_dir / "heatmap_face_density_summary.csv",
        index=False,
    )


def build_behavior_overview(cfg_path: str = "configs/project.yaml", *, output_dir: Path | None = None) -> Path:
    """Build behavior overview tables and figures."""
    apply_manuscript_style()
    cfg = load_config(resolve_dataset_cfg_path(cfg_path))
    out_dir = output_dir if output_dir is not None else build_analysis_output_dir(cfg, "behavior_overview")
    out_dir.mkdir(parents=True, exist_ok=True)
    gaze_paths = _index_by_key(cfg, "gaze_position")
    roi_paths = _index_by_key(cfg, "roi_vertices")
    fixation_paths = _index_by_key(cfg, "fixation_binary_vectors")
    fixation_event_paths = _index_by_key(cfg, "fixations")
    interactive_paths = _shared_index_by_key(cfg, "interactive_periods")
    session_keys = sorted(
        key
        for key in interactive_paths
        if all((key[0], key[1], agent) in gaze_paths for agent in AGENTS)
        and all((key[0], key[1], agent) in fixation_paths for agent in AGENTS)
    )
    session_rows = []
    period_frames = []
    pair_by_session = {}
    for date, session in session_keys:
        m1_vec = load_pickle_path(fixation_paths[(date, session, "m1")])
        m2_vec = load_pickle_path(fixation_paths[(date, session, "m2")])
        pair = f"{m1_vec.context.monkey_name}_{m2_vec.context.monkey_name}"
        pair_by_session[(date, session)] = pair
        n_samples = min(len(m1_vec.vectors["face"]), len(m2_vec.vectors["face"]))
        periods = load_pickle_path(interactive_paths[(date, session)]).copy()
        periods["date"] = date
        periods["session"] = session
        periods["monkey_pair"] = pair
        period_frames.append(periods)
        session_rows.append(
            {
                "date": date,
                "session": session,
                "monkey_pair": pair,
                "n_samples": n_samples,
                "duration_min": n_samples / SAMPLE_RATE_HZ / 60.0,
            }
        )
    session_df = pd.DataFrame(session_rows)
    periods_df, duration_summary = _summarize_periods(period_frames)
    session_summary = pd.DataFrame(
        [
            {
                "n_days": session_df["date"].nunique(),
                "n_5min_sessions": len(session_df),
                "mean_session_duration_min": session_df["duration_min"].mean(),
                "median_session_duration_min": session_df["duration_min"].median(),
                "sd_session_duration_min": session_df["duration_min"].std(ddof=1),
            }
        ]
    )
    period_count_summary = periods_df.groupby("state").size().rename("n_periods").reset_index()
    prob_df = pd.DataFrame(_probability_rows(session_keys, interactive_paths, fixation_paths))
    prob_summary = summarize_probabilities(prob_df)
    prob_stats = compare_interactive_probabilities(prob_df)
    session_df.to_csv(out_dir / "session_summary_by_session.csv", index=False)
    session_summary.to_csv(out_dir / "session_summary.csv", index=False)
    periods_df.to_csv(out_dir / "interactive_periods_with_durations.csv", index=False)
    duration_summary.to_csv(out_dir / "period_duration_summary_by_state.csv", index=False)
    period_count_summary.to_csv(out_dir / "period_count_summary_by_state.csv", index=False)
    prob_df.to_csv(out_dir / "fixation_probability_by_session_phase.csv", index=False)
    prob_summary.to_csv(out_dir / "fixation_probability_summary.csv", index=False)
    prob_stats.to_csv(out_dir / "fixation_probability_interactive_vs_non_interactive_stats.csv", index=False)
    plot_period_counts(period_count_summary, out_dir)
    plot_period_duration_boxplot(periods_df, out_dir)
    plot_fixation_probability_comparison(prob_df, prob_stats, out_dir)
    build_fixation_heatmaps_by_phase(
        session_keys=session_keys,
        gaze_paths=gaze_paths,
        roi_paths=roi_paths,
        fixation_event_paths=fixation_event_paths,
        interactive_paths=interactive_paths,
        out_dir=out_dir,
    )
    print(f"Wrote behavior overview outputs to {out_dir.resolve()}")
    return out_dir


__all__ = [
    "behavior_overview_output_dir",
    "build_behavior_overview",
    "build_fixation_heatmaps_by_phase",
    "compare_interactive_probabilities",
    "significance_stars",
    "summarize_probabilities",
]
