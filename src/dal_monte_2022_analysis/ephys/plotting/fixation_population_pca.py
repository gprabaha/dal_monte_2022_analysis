"""Plotting helpers for fixation population PCA outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FormatStrFormatter, LinearLocator, MaxNLocator
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except Exception:  # pragma: no cover - handled explicitly at runtime
    sns = None

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import apply_plotting_config
from dal_monte_2022_analysis.runtime.io.plot_output import (
    normalize_extension,
    save_figure,
)
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


DEFAULT_CONDITION_ORDER: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
DEFAULT_CONDITION_LABELS: dict[str, str] = {
    "face_interactive": "Interactive Face",
    "face_non_interactive": "Non-Interactive Face",
    "object": "Object",
}
DEFAULT_CONDITION_COLORS: dict[str, str] = {
    "face_interactive": "#d62728",
    "face_non_interactive": "#1f77b4",
    "object": "#2ca02c",
}
DEFAULT_REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")
DEFAULT_REGION_LABELS: dict[str, str] = {
    "bla": "BLA",
    "accg": "ACCg",
    "dmpfc": "dmPFC",
    "ofc": "OFC",
}
DEFAULT_REGION_COLORS: dict[str, str] = {
    "bla": "#7f3b08",
    "accg": "#1b9e77",
    "dmpfc": "#7570b3",
    "ofc": "#e7298a",
}


@dataclass
class FixationPopulationPCAPlotSettings:
    """Configuration for fixation population PCA plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_population_pca"
    input_filename: str = "results.pkl"
    output_subdir: str = "ephys/psth/fixation_population_pca/plots"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 300
    conditions: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_CONDITION_ORDER),
    )
    condition_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONDITION_LABELS),
    )
    condition_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONDITION_COLORS),
    )
    condition_pair_labels: dict[str, str] = field(default_factory=dict)
    condition_pair_colors: dict[str, str] = field(default_factory=dict)
    region_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_REGION_LABELS),
    )
    region_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_REGION_COLORS),
    )
    trajectory_n_pcs: int = 3
    trajectory_n_columns: int = 4
    trajectory_letter_width_in: float = 8.5
    trajectory_letter_height_frac: float = 0.2
    trajectory_view_elev: float = 22.0
    trajectory_view_azim: float = -58.0
    trajectory_region_view_azim_offsets: dict[str, float] = field(default_factory=dict)
    trajectory_grid_alpha: float = 0.28
    trajectory_hide_standard_axes: bool = False
    trajectory_axis_anchor: str = "back_corner"
    trajectory_axis_arrow_length_frac: float = 0.10
    trajectory_axis_label_fontsize: float = 6.5
    trajectory_show_length_inset: bool = False
    trajectory_show_total_variance_panel: bool = True
    trajectory_total_variance_n_pcs: Optional[int] = None
    variance_letter_width_in: float = 7.4
    variance_letter_height_frac: float = 0.48
    max_components_display: int = 20
    pairwise_violin_letter_width_in: float = 8.5
    pairwise_violin_letter_height_frac: float = 0.28


def _resolve_output_ext(settings: FixationPopulationPCAPlotSettings) -> str:
    return normalize_extension(settings.output_extension, fallback="pdf")


def _load_population_pca_result(
    settings: FixationPopulationPCAPlotSettings,
) -> tuple[dict, Path]:
    cfg = load_config(settings.cfg_path)
    in_root = build_analysis_output_dir(cfg, settings.input_subdir)
    in_path = in_root / ensure_filename(settings.input_filename, ".pkl")
    if not in_path.exists():
        raise FileNotFoundError(f"Population PCA results not found: {in_path}")
    obj = load_pickle_path(in_path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict payload in PCA results: {in_path}")
    return obj, in_path


def _resolve_region_order(
    result_obj: dict,
    *,
    regions: Optional[Sequence[str]],
) -> list[str]:
    region_payloads = result_obj.get("regions", {}) if isinstance(result_obj, dict) else {}
    if not isinstance(region_payloads, dict):
        region_payloads = {}
    available = [str(key) for key in region_payloads.keys()]
    return _ordered_region_tokens(available=available, requested=regions)


def _normalize_region_token(region: object) -> str:
    token = str(region).strip()
    return token.lower()


def _ordered_region_tokens(
    *,
    available: Sequence[str],
    requested: Optional[Sequence[str]],
) -> list[str]:
    available_map = {str(region).lower(): str(region) for region in available}
    if requested is not None:
        requested_tokens = [str(region).strip().lower() for region in requested]
        ordered = [available_map[token] for token in requested_tokens if token in available_map]
        return ordered

    ordered: list[str] = []
    for token in DEFAULT_REGION_ORDER:
        if token in available_map:
            ordered.append(available_map[token])
    for token in sorted(available_map.keys()):
        if token not in set(DEFAULT_REGION_ORDER):
            ordered.append(available_map[token])
    return ordered


def _region_display_label(region: object) -> str:
    token = _normalize_region_token(region)
    return DEFAULT_REGION_LABELS.get(token, str(region))


def _display_region(region: object, settings: FixationPopulationPCAPlotSettings) -> str:
    token = _normalize_region_token(region)
    if token in settings.region_labels:
        return str(settings.region_labels[token])
    return DEFAULT_REGION_LABELS.get(token, str(region))


def _resolve_condition_colors(settings: FixationPopulationPCAPlotSettings) -> dict[str, str]:
    out = dict(DEFAULT_CONDITION_COLORS)
    for cond, color in settings.condition_colors.items():
        key = str(cond).strip()
        if key:
            out[key] = str(color).strip()
    return out


def _condition_pair_token(condition_a: object, condition_b: object) -> str:
    return f"{str(condition_a)}__vs__{str(condition_b)}"


def _condition_pair_order(settings: FixationPopulationPCAPlotSettings) -> list[str]:
    return [
        _condition_pair_token(condition_a, condition_b)
        for idx, condition_a in enumerate(settings.conditions)
        for condition_b in settings.conditions[idx + 1 :]
    ]


def _blend_colors(color_a: str, color_b: str) -> str:
    rgb = 0.5 * np.asarray(to_rgb(color_a), dtype=float) + 0.5 * np.asarray(to_rgb(color_b), dtype=float)
    rgb = np.clip(rgb, 0.0, 1.0)
    return mpl.colors.to_hex(rgb)


def _condition_pair_display_label(
    pair_token: object,
    settings: FixationPopulationPCAPlotSettings,
) -> str:
    token = str(pair_token).strip()
    if token in settings.condition_pair_labels:
        return str(settings.condition_pair_labels[token])
    parts = token.split("__vs__")
    if len(parts) != 2:
        return token
    return (
        f"{settings.condition_labels.get(parts[0], parts[0])} vs "
        f"{settings.condition_labels.get(parts[1], parts[1])}"
    )


def _condition_pair_axis_label(
    pair_token: object,
    settings: FixationPopulationPCAPlotSettings,
) -> str:
    return _condition_pair_display_label(pair_token, settings).replace(" vs ", "\nvs\n")


def _resolve_condition_pair_colors(
    settings: FixationPopulationPCAPlotSettings,
) -> dict[str, str]:
    out: dict[str, str] = {}
    condition_colors = _resolve_condition_colors(settings)
    for pair_token in _condition_pair_order(settings):
        if pair_token in settings.condition_pair_colors:
            out[pair_token] = str(settings.condition_pair_colors[pair_token]).strip()
            continue
        parts = pair_token.split("__vs__")
        if len(parts) == 2 and parts[0] in condition_colors and parts[1] in condition_colors:
            out[pair_token] = _blend_colors(condition_colors[parts[0]], condition_colors[parts[1]])
        else:
            out[pair_token] = "#777777"
    for pair_token, color in settings.condition_pair_colors.items():
        key = str(pair_token).strip()
        if key:
            out[key] = str(color).strip()
    return out


def _resolve_region_colors(settings: FixationPopulationPCAPlotSettings) -> dict[str, str]:
    out = dict(DEFAULT_REGION_COLORS)
    for region, color in settings.region_colors.items():
        key = str(region).strip()
        if key:
            out[key] = str(color).strip()
    return out


def _as_df(raw: object) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    if isinstance(raw, dict):
        try:
            return pd.DataFrame(raw)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _extract_pairwise_geometry_df(result_obj: dict) -> pd.DataFrame:
    df = _as_df(result_obj.get("pairwise_geometry_timecourses"))
    required = {
        "region",
        "condition_pair",
        "metric_name",
        "metric_label",
        "metric_unit",
        "bin_index",
        "bin_center_s",
        "value",
    }
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    for col in ("region", "condition_pair", "metric_name", "metric_label", "metric_unit"):
        out[col] = out[col].astype(str)
    out["bin_index"] = pd.to_numeric(out["bin_index"], errors="coerce")
    out["bin_center_s"] = pd.to_numeric(out["bin_center_s"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.loc[out["bin_index"].notna()].copy()
    out["bin_index"] = out["bin_index"].astype(int)
    return out.sort_values(["metric_name", "region", "condition_pair", "bin_index"]).reset_index(drop=True)


def _extract_pairwise_geometry_stats_df(result_obj: dict, key: str) -> pd.DataFrame:
    df = _as_df(result_obj.get(key))
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        if col.endswith("_pair") or col in {
            "metric_name",
            "metric_label",
            "metric_unit",
            "region",
            "region_a",
            "region_b",
            "condition_pair",
            "condition_pair_a",
            "condition_pair_b",
            "test_name",
            "pvalue_correction",
        }:
            out[col] = out[col].astype(str)
    for col in ("p_value", "p_value_adjusted", "statistic", "alpha"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "significant_adjusted" in out.columns:
        out["significant_adjusted"] = out["significant_adjusted"].map(bool)
    return out


def _significance_star(p_value_adj: float) -> str:
    if not np.isfinite(p_value_adj):
        return ""
    p = float(p_value_adj)
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def _safe_suffix_token(value: object) -> str:
    token = str(value).strip().lower().replace(" ", "_").replace("/", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "plot"


def _coerce_scores_pc_by_time(
    raw_scores: object,
    *,
    n_time_bins: int,
) -> np.ndarray:
    arr = np.asarray(raw_scores, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.asarray([], dtype=float)
    # Preferred orientation is PCs x time.
    if arr.shape[1] == int(n_time_bins):
        return arr
    # Backward-compat orientation (time x PCs).
    if arr.shape[0] == int(n_time_bins):
        return arr.T
    return np.asarray([], dtype=float)


def _nearest_marker_indices(bin_centers_s: np.ndarray) -> list[int]:
    centers = np.asarray(bin_centers_s, dtype=float).reshape(-1)
    if centers.size == 0:
        return []
    targets = (-0.5, 0.0, 0.5)
    return [int(np.argmin(np.abs(centers - target))) for target in targets]


def _trajectory_marker_specs() -> list[dict[str, object]]:
    return [
        {"label": "-500 ms", "marker": "o", "size": 14.0},
        {"label": "0 ms", "marker": "s", "size": 14.0},
        {"label": "+500 ms", "marker": "^", "size": 18.0},
    ]


def _trajectory_path_length(
    scores_pc_by_time: np.ndarray,
    *,
    n_pcs: int,
) -> float:
    scores = np.asarray(scores_pc_by_time, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 2:
        return float("nan")
    n_use = min(max(1, int(n_pcs)), int(scores.shape[0]))
    if n_use <= 0:
        return float("nan")
    diffs = np.diff(scores[:n_use, :], axis=1)
    step_lengths = np.linalg.norm(diffs, axis=0)
    finite = step_lengths[np.isfinite(step_lengths)]
    if finite.size == 0:
        return float("nan")
    return float(np.sum(finite))


def _short_condition_label(condition: object) -> str:
    token = str(condition).strip().lower()
    if token == "face_interactive":
        return "Int"
    if token == "face_non_interactive":
        return "Non"
    if token == "object":
        return "Obj"
    return str(condition)


def _resolve_total_variance_n_pcs(
    settings: FixationPopulationPCAPlotSettings,
) -> int:
    raw = settings.trajectory_total_variance_n_pcs
    if raw is None:
        raw = settings.trajectory_n_pcs
    try:
        n_pcs = int(raw)
    except Exception:
        n_pcs = int(settings.trajectory_n_pcs)
    return max(1, n_pcs)


def _trajectory_view_azim_for_region(
    region: object,
    settings: FixationPopulationPCAPlotSettings,
) -> float:
    token = _normalize_region_token(region)
    offset = settings.trajectory_region_view_azim_offsets.get(token, 0.0)
    try:
        offset_value = float(offset)
    except Exception:
        offset_value = 0.0
    return float(settings.trajectory_view_azim) + offset_value


def _draw_trajectory_total_variance_panel(
    ax,
    *,
    cond_order: Sequence[str],
    total_variances: dict[str, float],
    color_map: dict[str, str],
    y_max: float,
    show_ylabel: bool,
) -> None:
    x = np.arange(len(cond_order), dtype=float)
    y = np.asarray(
        [float(total_variances.get(str(condition), np.nan)) for condition in cond_order],
        dtype=float,
    )
    colors = [str(color_map.get(str(condition), "#777777")) for condition in cond_order]
    finite = np.isfinite(y)
    if np.any(finite):
        ax.bar(
            x[finite],
            y[finite],
            width=0.52,
            color=[colors[idx] for idx in np.flatnonzero(finite)],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.95,
        )

    upper = float(y_max) if np.isfinite(y_max) and float(y_max) > 0.0 else 1.0
    ax.set_ylim(0.0, upper)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short_condition_label(condition) for condition in cond_order],
        fontsize=4.0,
    )
    ax.tick_params(axis="x", pad=0.25, length=0.0)
    ax.tick_params(axis="y", labelsize=4.1, pad=0.35, length=1.1, labelleft=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title("Frac", fontsize=4.7, pad=0.9)
    if show_ylabel:
        ax.set_ylabel("Frac", fontsize=4.5, labelpad=0.65)
    else:
        ax.set_ylabel("")
    ax.grid(axis="y", linewidth=0.25, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.45)
    ax.spines["bottom"].set_linewidth(0.45)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.set_facecolor((1.0, 1.0, 1.0, 0.92))


def _apply_axis_limits_3d(ax, all_points: list[np.ndarray]) -> None:
    if not all_points:
        return
    stack = np.vstack(all_points)
    mins = np.nanmin(stack, axis=0)
    maxs = np.nanmax(stack, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pads = 0.08 * spans
    ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
    ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
    ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])


def _hide_trajectory_axes(ax) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="major", length=0, pad=-2, labelsize=0)
    ax.tick_params(axis="z", which="major", length=0, pad=-2, labelsize=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((0.0, 0.0, 0.0, 0.0))
        try:
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass


def _draw_trajectory_axis_arrows(
    ax,
    *,
    all_points: list[np.ndarray],
    settings: FixationPopulationPCAPlotSettings,
) -> None:
    if not all_points:
        return
    stack = np.vstack(all_points)
    mins = np.nanmin(stack, axis=0)
    maxs = np.nanmax(stack, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    arrow_len = float(settings.trajectory_axis_arrow_length_frac) * float(np.max(spans))
    if not np.isfinite(arrow_len) or arrow_len <= 0.0:
        return

    anchor_mode = str(settings.trajectory_axis_anchor).strip().lower()
    if anchor_mode == "origin":
        anchor = np.zeros((3,), dtype=float)
    else:
        anchor = mins + 0.05 * spans

    label_fontsize = float(settings.trajectory_axis_label_fontsize)
    for dim_idx, label in enumerate(("PC1", "PC2", "PC3")):
        vec = np.zeros((3,), dtype=float)
        vec[dim_idx] = arrow_len
        ax.quiver(
            float(anchor[0]),
            float(anchor[1]),
            float(anchor[2]),
            float(vec[0]),
            float(vec[1]),
            float(vec[2]),
            color="black",
            linewidth=0.9,
            arrow_length_ratio=0.22,
            pivot="tail",
            normalize=False,
            zorder=4.5,
        )
        tip = anchor + vec
        ax.text(
            float(tip[0]),
            float(tip[1]),
            float(tip[2]),
            str(label),
            fontsize=label_fontsize,
            color="black",
            ha="left",
            va="bottom",
            zorder=5.0,
        )


def _draw_trajectory_length_inset(
    ax,
    *,
    cond_order: Sequence[str],
    path_lengths: dict[str, float],
    color_map: dict[str, str],
) -> None:
    inset = ax.inset_axes([0.62, 0.05, 0.32, 0.24])
    x = np.arange(len(cond_order), dtype=float)
    y = np.asarray(
        [float(path_lengths.get(str(condition), np.nan)) for condition in cond_order],
        dtype=float,
    )
    colors = [str(color_map.get(str(condition), "#777777")) for condition in cond_order]
    finite = y[np.isfinite(y)]
    if finite.size > 0:
        inset.bar(
            x,
            y,
            width=0.72,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.95,
        )
        ymax = float(np.max(finite))
        inset.set_ylim(0.0, ymax * 1.18 if ymax > 0.0 else 1.0)
    else:
        inset.set_ylim(0.0, 1.0)
    inset.set_xticks(x)
    inset.set_xticklabels(
        [_short_condition_label(condition) for condition in cond_order],
        fontsize=4.8,
    )
    inset.tick_params(axis="y", labelsize=4.6, pad=0.6, length=1.5)
    inset.tick_params(axis="x", pad=0.4, length=0.0)
    inset.set_title("Path", fontsize=5.2, pad=1.4)
    inset.grid(axis="y", linewidth=0.25, alpha=0.22)
    for spine in inset.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#333333")
    inset.set_facecolor((1.0, 1.0, 1.0, 0.88))


def _apply_plotting_style(plotting_cfg_path: str) -> None:
    if plotting_cfg_path and Path(plotting_cfg_path).exists():
        cfg = load_config(plotting_cfg_path)
        apply_plotting_config(cfg)


def _extract_cross_condition_explained_variance_df(result_obj: dict) -> pd.DataFrame:
    raw = result_obj.get("cross_condition_explained_variance")
    if isinstance(raw, pd.DataFrame):
        out = raw.copy()
    elif isinstance(raw, dict):
        out = pd.DataFrame(raw)
    else:
        out = pd.DataFrame()
    if out.empty:
        return pd.DataFrame()
    required = {
        "region",
        "fit_condition",
        "eval_condition",
        "n_components",
        "explained_variance_fraction",
    }
    if not required.issubset(out.columns):
        return pd.DataFrame()
    out = out.copy()
    out["region"] = out["region"].astype(str)
    if "fit_scope" in out.columns:
        out["fit_scope"] = out["fit_scope"].astype(str)
    out["fit_condition"] = out["fit_condition"].astype(str)
    out["eval_condition"] = out["eval_condition"].astype(str)
    out["n_components"] = pd.to_numeric(out["n_components"], errors="coerce")
    out["explained_variance_fraction"] = pd.to_numeric(
        out["explained_variance_fraction"],
        errors="coerce",
    )
    if "projected_variance" in out.columns:
        out["projected_variance"] = pd.to_numeric(out["projected_variance"], errors="coerce")
    if "projected_variance_cumulative" in out.columns:
        out["projected_variance_cumulative"] = pd.to_numeric(
            out["projected_variance_cumulative"],
            errors="coerce",
        )
    if "projected_variance_total" in out.columns:
        out["projected_variance_total"] = pd.to_numeric(
            out["projected_variance_total"],
            errors="coerce",
        )
    if "explained_variance_per_pc_fraction" in out.columns:
        out["explained_variance_per_pc_fraction"] = pd.to_numeric(
            out["explained_variance_per_pc_fraction"],
            errors="coerce",
        )
    if "explained_variance_cumulative_fraction" in out.columns:
        out["explained_variance_cumulative_fraction"] = pd.to_numeric(
            out["explained_variance_cumulative_fraction"],
            errors="coerce",
        )
    out = out.loc[out["n_components"].notna()].copy()
    out["n_components"] = out["n_components"].astype(int)

    group_cols = ["region", "fit_condition", "eval_condition"]
    out = out.sort_values(group_cols + ["n_components"]).reset_index(drop=True)

    if "explained_variance_per_pc_fraction" in out.columns:
        per_pc = np.asarray(out["explained_variance_per_pc_fraction"], dtype=float)
    else:
        raw = np.asarray(out["explained_variance_fraction"], dtype=float)
        # Backward-compat: older files stored cumulative values in explained_variance_fraction.
        # Detect monotonic curves and convert cumulative -> per-PC via finite differences.
        is_likely_cumulative = True
        grouped = out.groupby(group_cols, dropna=False, sort=False)
        for _, grp in grouped:
            values = np.asarray(grp["explained_variance_fraction"], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size <= 1:
                continue
            diffs = np.diff(finite)
            if np.any(diffs < -1e-8):
                is_likely_cumulative = False
                break
        if is_likely_cumulative:
            per_pc = np.full(raw.shape, np.nan, dtype=float)
            for _, idx in grouped.indices.items():
                values = np.asarray(out.loc[idx, "explained_variance_fraction"], dtype=float)
                diffs = np.full(values.shape, np.nan, dtype=float)
                prev = 0.0
                for i, value in enumerate(values):
                    if np.isfinite(value):
                        diffs[i] = max(0.0, float(value) - float(prev))
                        prev = float(value)
                per_pc[np.asarray(idx, dtype=int)] = diffs
        else:
            per_pc = raw
    out["explained_variance_per_pc_fraction"] = per_pc
    out["explained_variance_fraction"] = per_pc

    if "explained_variance_cumulative_fraction" not in out.columns:
        out["explained_variance_cumulative_fraction"] = np.nan
    for _, idx in out.groupby(group_cols, dropna=False, sort=False).indices.items():
        values = np.asarray(out.loc[idx, "explained_variance_per_pc_fraction"], dtype=float)
        cumulative = np.full(values.shape, np.nan, dtype=float)
        running = 0.0
        for i, value in enumerate(values):
            if np.isfinite(value):
                running += float(value)
                cumulative[i] = running
        out.loc[idx, "explained_variance_cumulative_fraction"] = cumulative
    return out


def _condition_total_variance_fractions_from_explained_df(
    explained_df: pd.DataFrame,
    *,
    region: object,
    fit_condition: object,
    cond_order: Sequence[str],
    n_pcs: int,
) -> dict[str, float]:
    if explained_df.empty or "projected_variance" not in explained_df.columns:
        return {}
    n_use = max(1, int(n_pcs))
    sub = explained_df.loc[
        (explained_df["region"] == str(region))
        & (explained_df["fit_condition"] == str(fit_condition))
        & (explained_df["n_components"] >= 1)
        & (explained_df["n_components"] <= n_use)
    ].copy()
    if sub.empty:
        return {}
    totals: dict[str, float] = {}
    for condition in cond_order:
        cond_sub = sub.loc[sub["eval_condition"] == str(condition)].copy()
        if cond_sub.empty:
            continue
        values = np.asarray(cond_sub["projected_variance"], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        total = float(np.sum(np.maximum(values, 0.0)))
        if total > 0.0:
            totals[str(condition)] = total
    denom = float(np.sum(list(totals.values()), dtype=float)) if totals else 0.0
    if denom <= 0.0:
        return {}
    return {
        str(condition): float(totals[str(condition)] / denom)
        for condition in cond_order
        if str(condition) in totals
    }


def _add_compound_bar_patch(
    ax,
    *,
    centers: np.ndarray,
    heights: np.ndarray,
    width: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    alpha: float,
    zorder: float = 2.0,
) -> None:
    x = np.asarray(centers, dtype=float).reshape(-1)
    y = np.asarray(heights, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size or float(width) <= 0.0:
        return
    polys: list[np.ndarray] = []
    half_width = 0.5 * float(width)
    for xv, yv in zip(x, y):
        if not np.isfinite(xv) or not np.isfinite(yv):
            continue
        bottom = min(0.0, float(yv))
        top = max(0.0, float(yv))
        if np.isclose(bottom, top):
            continue
        polys.append(
            np.asarray(
                [
                    [float(xv) - half_width, bottom],
                    [float(xv) - half_width, top],
                    [float(xv) + half_width, top],
                    [float(xv) + half_width, bottom],
                ],
                dtype=float,
            )
        )
    if not polys:
        return
    compound = MplPath.make_compound_path_from_polys(np.asarray(polys, dtype=float))
    patch = PathPatch(
        compound,
        facecolor=str(facecolor),
        edgecolor=str(edgecolor),
        linewidth=float(linewidth),
        alpha=float(alpha),
        antialiased=True,
        joinstyle="miter",
        capstyle="butt",
        clip_on=False,
        zorder=float(zorder),
    )
    ax.add_patch(patch)


def _rowwise_upper_limit(values: object, *, floor: float) -> float:
    vec = np.asarray(values, dtype=float).reshape(-1)
    vec = vec[np.isfinite(vec)]
    if vec.size == 0:
        return float(floor)
    vmax = float(np.max(vec))
    if vmax <= 0.0:
        return float(floor)
    pad = max(0.08 * vmax, 0.01)
    return max(float(floor), vmax + pad)


def plot_fixation_population_pca_trajectories(
    settings: FixationPopulationPCAPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    output_filename: str = "population_pca_pc_trajectories",
) -> Optional[dict]:
    """Plot 3D trajectories by region with sidecar fixation-type variance bars."""
    _apply_plotting_style(settings.plotting_cfg_path)
    result_obj, input_path = _load_population_pca_result(settings)
    region_payloads = result_obj.get("regions", {})
    if not isinstance(region_payloads, dict) or not region_payloads:
        print("[plot] no region payloads available in fixation population PCA results")
        return None

    region_order = _resolve_region_order(result_obj, regions=regions)
    if not region_order:
        print("[plot] no regions available after filtering for PCA trajectory plotting")
        return None

    show_variance_panel = bool(settings.trajectory_show_total_variance_panel)
    variance_n_pcs = _resolve_total_variance_n_pcs(settings)
    n_cols = max(1, int(settings.trajectory_n_columns))
    n_rows = int(np.ceil(len(region_order) / float(n_cols)))
    fig_w = float(settings.trajectory_letter_width_in) * (0.96 if show_variance_panel else 1.0)
    base_fig_h = float(8.5 * settings.trajectory_letter_height_frac * max(1, n_rows))
    fig_h = max(base_fig_h, 2.30 if show_variance_panel else 0.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=settings.output_dpi)
    outer_gs = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.02,
        right=0.995,
        top=0.80,
        bottom=0.07,
        wspace=0.11 if show_variance_panel else 0.02,
        hspace=0.18,
    )

    color_map = _resolve_condition_colors(settings)
    cond_order = [cond for cond in settings.conditions if cond in color_map]
    explained_df = _extract_cross_condition_explained_variance_df(result_obj)
    marker_indices_cache: dict[str, list[int]] = {}
    marker_specs = _trajectory_marker_specs()
    region_plot_data: dict[str, dict[str, object]] = {}

    for region in region_order:
        payload = region_payloads.get(region, {})
        raw_scores_map = payload.get("concatenated_condition_scores_pc_by_time")
        if not isinstance(raw_scores_map, dict):
            raw_scores_map = payload.get("concatenated_condition_scores", {})
        if not isinstance(raw_scores_map, dict):
            raw_scores_map = {}

        bin_centers_s = np.asarray(
            payload.get("bin_centers_s_window", np.asarray([], dtype=float)),
            dtype=float,
        ).reshape(-1)
        marker_indices = _nearest_marker_indices(bin_centers_s)
        marker_indices_cache[str(region)] = marker_indices

        condition_scores_pc_by_time: dict[str, np.ndarray] = {}
        for condition in cond_order:
            raw_scores = raw_scores_map.get(condition, np.asarray([], dtype=float))
            scores = _coerce_scores_pc_by_time(raw_scores, n_time_bins=bin_centers_s.size)
            if scores.size == 0:
                continue
            condition_scores_pc_by_time[str(condition)] = scores

        total_variances = _condition_total_variance_fractions_from_explained_df(
            explained_df,
            region=str(region),
            fit_condition="concatenated",
            cond_order=cond_order,
            n_pcs=variance_n_pcs,
        )

        region_plot_data[str(region)] = {
            "bin_centers_s": bin_centers_s,
            "marker_indices": marker_indices,
            "condition_scores_pc_by_time": condition_scores_pc_by_time,
            "total_variances": total_variances,
        }

    for idx, region in enumerate(region_order):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        if show_variance_panel:
            inner_gs = outer_gs[row_idx, col_idx].subgridspec(
                4,
                2,
                width_ratios=(3.32, 0.48),
                height_ratios=(0.7, 1.0, 1.0, 0.7),
                wspace=0.42,
                hspace=0.0,
            )
            ax = fig.add_subplot(inner_gs[:, 0], projection="3d")
            ax_var = fig.add_subplot(inner_gs[1:3, 1])
        else:
            ax = fig.add_subplot(outer_gs[row_idx, col_idx], projection="3d")
            ax_var = None

        plot_data = region_plot_data.get(str(region), {})
        bin_centers_s = np.asarray(
            plot_data.get("bin_centers_s", np.asarray([], dtype=float)),
            dtype=float,
        ).reshape(-1)
        marker_indices = list(plot_data.get("marker_indices", []))
        scores_map = plot_data.get("condition_scores_pc_by_time", {})
        if not isinstance(scores_map, dict):
            scores_map = {}
        total_variances = plot_data.get("total_variances", {})
        if not isinstance(total_variances, dict):
            total_variances = {}

        all_xyz: list[np.ndarray] = []
        path_lengths: dict[str, float] = {}
        for condition in cond_order:
            scores = np.asarray(
                scores_map.get(condition, np.asarray([], dtype=float)),
                dtype=float,
            )
            if scores.size == 0 or scores.shape[0] < int(settings.trajectory_n_pcs):
                continue
            xyz = np.asarray(scores[:3, :], dtype=float).T
            if xyz.shape[0] < 2:
                continue
            all_xyz.append(xyz)
            path_lengths[str(condition)] = _trajectory_path_length(
                scores,
                n_pcs=int(settings.trajectory_n_pcs),
            )

            border_color = str(color_map.get(condition, DEFAULT_CONDITION_COLORS[condition]))
            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                color="black",
                linewidth=1.6,
                alpha=0.95,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=2,
            )
            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                color=border_color,
                linewidth=1.05,
                alpha=0.98,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )

            for marker_idx, marker_spec in zip(marker_indices, marker_specs):
                marker_idx = max(0, min(int(marker_idx), xyz.shape[0] - 1))
                ax.scatter(
                    [float(xyz[marker_idx, 0])],
                    [float(xyz[marker_idx, 1])],
                    [float(xyz[marker_idx, 2])],
                    s=float(marker_spec["size"]),
                    c=["black"],
                    edgecolors="black",
                    linewidths=0.3,
                    marker=str(marker_spec["marker"]),
                    alpha=0.98,
                )

        _apply_axis_limits_3d(ax, all_xyz)
        ax.view_init(
            elev=float(settings.trajectory_view_elev),
            azim=_trajectory_view_azim_for_region(region, settings),
        )
        ax.grid(True, linewidth=0.35, alpha=float(settings.trajectory_grid_alpha))
        if bool(settings.trajectory_hide_standard_axes):
            _hide_trajectory_axes(ax)
            _draw_trajectory_axis_arrows(ax, all_points=all_xyz, settings=settings)
        else:
            ax.set_xlabel("PC1", labelpad=-4, fontsize=7)
            ax.set_ylabel("PC2", labelpad=-4, fontsize=7)
            ax.set_zlabel("PC3", labelpad=-3, fontsize=7)
            ax.xaxis.set_major_locator(LinearLocator(3))
            ax.yaxis.set_major_locator(LinearLocator(3))
            ax.zaxis.set_major_locator(LinearLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", which="major", labelsize=6, pad=0.5)
            ax.tick_params(axis="z", which="major", labelsize=6, pad=0.5)
        if bool(settings.trajectory_show_length_inset) and not show_variance_panel:
            _draw_trajectory_length_inset(
                ax,
                cond_order=cond_order,
                path_lengths=path_lengths,
                color_map=color_map,
            )
        ax.set_title(_region_display_label(region), fontsize=8, pad=2.0)

        if ax_var is not None:
            _draw_trajectory_total_variance_panel(
                ax_var,
                cond_order=cond_order,
                total_variances=total_variances,
                color_map=color_map,
                y_max=_rowwise_upper_limit(list(total_variances.values()), floor=0.02),
                show_ylabel=(col_idx == 0),
            )

    condition_handles = [
        Line2D([0], [0], color=str(color_map[cond]), lw=1.8, label=settings.condition_labels.get(cond, cond))
        for cond in cond_order
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=str(marker_spec["marker"]),
            linestyle="None",
            markersize=max(4.0, float(marker_spec["size"]) ** 0.5),
            markerfacecolor="black",
            markeredgecolor="black",
            label=str(marker_spec["label"]),
        )
        for marker_spec in marker_specs
    ]
    fig.legend(
        handles=condition_handles + marker_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(condition_handles + marker_handles)),
        fontsize=7,
        frameon=False,
    )

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    ext = _resolve_output_ext(settings)
    out_name = ensure_filename(output_filename, f".{ext}")
    out_path = out_root / out_name
    save_figure(
        fig,
        out_path,
        ext=ext,
        dpi=settings.output_dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)

    return {
        "output_path": str(out_path),
        "input_path": str(input_path),
        "regions": list(region_order),
        "conditions": list(cond_order),
        "marker_indices": marker_indices_cache,
        "total_variance_n_pcs": int(variance_n_pcs),
        "total_variances": {
            str(region): dict(region_plot_data.get(str(region), {}).get("total_variances", {}))
            for region in region_order
        },
        "view_elev": float(settings.trajectory_view_elev),
        "view_azim": float(settings.trajectory_view_azim),
    }


def plot_fixation_population_pca_explained_variance_bars(
    settings: FixationPopulationPCAPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    output_filename: str = "population_pca_explained_variance_bars",
) -> Optional[dict]:
    """Plot per-PC explained variance bars across eval fixation types."""
    _apply_plotting_style(settings.plotting_cfg_path)
    result_obj, input_path = _load_population_pca_result(settings)
    explained_df = _extract_cross_condition_explained_variance_df(result_obj)
    if explained_df.empty:
        print("[plot] no cross-condition explained variance rows found")
        return None

    cond_order = [cond for cond in settings.conditions if cond in set(explained_df["fit_condition"].unique())]
    if not cond_order:
        print("[plot] no matching fit conditions for explained variance plotting")
        return None

    region_order = _ordered_region_tokens(
        available=explained_df["region"].astype(str).unique().tolist(),
        requested=regions,
    )
    if not region_order:
        print("[plot] no matching regions for explained variance plotting")
        return None

    max_comp = max(1, int(settings.max_components_display))
    n_rows = len(cond_order)
    n_cols = max(4, len(region_order))
    fig = plt.figure(
        figsize=(
            float(settings.variance_letter_width_in),
            8.5 * float(settings.variance_letter_height_frac),
        ),
        dpi=settings.output_dpi,
    )
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    color_map = _resolve_condition_colors(settings)
    eval_order = [cond for cond in settings.conditions if cond in set(explained_df["eval_condition"].unique())]
    bar_width = 0.84 / max(1, len(eval_order))
    x = np.arange(1, max_comp + 1, dtype=float)
    row_ymax_map: dict[str, float] = {}
    for fit_condition in cond_order:
        row_sub = explained_df.loc[
            (explained_df["fit_condition"] == str(fit_condition))
            & (explained_df["n_components"] >= 1)
            & (explained_df["n_components"] <= max_comp)
        ].copy()
        row_ymax_map[str(fit_condition)] = _rowwise_upper_limit(
            row_sub["explained_variance_fraction"].to_numpy(dtype=float),
            floor=0.05,
        )

    for row_idx, fit_condition in enumerate(cond_order):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(region_order):
                ax.set_axis_off()
                continue
            region = region_order[col_idx]
            sub = explained_df.loc[
                (explained_df["region"] == str(region))
                & (explained_df["fit_condition"] == str(fit_condition))
                & (explained_df["n_components"] >= 1)
                & (explained_df["n_components"] <= max_comp)
            ].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            for eval_idx, eval_condition in enumerate(eval_order):
                eval_sub = sub.loc[sub["eval_condition"] == str(eval_condition)].copy()
                if eval_sub.empty:
                    continue
                eval_sub = eval_sub.sort_values("n_components")
                y = np.full((max_comp,), np.nan, dtype=float)
                comp_idx = np.asarray(eval_sub["n_components"], dtype=int) - 1
                valid = (comp_idx >= 0) & (comp_idx < max_comp)
                if np.any(valid):
                    y[comp_idx[valid]] = np.asarray(
                        eval_sub.loc[valid, "explained_variance_fraction"],
                        dtype=float,
                    )
                offset = (float(eval_idx) - (len(eval_order) - 1.0) / 2.0) * bar_width
                finite = np.isfinite(y)
                if np.any(finite):
                    _add_compound_bar_patch(
                        ax,
                        centers=x[finite] + offset,
                        heights=y[finite],
                        width=bar_width * 0.95,
                        facecolor=str(
                            color_map.get(
                                eval_condition,
                                DEFAULT_CONDITION_COLORS.get(eval_condition, "#444444"),
                            )
                        ),
                        edgecolor="black",
                        linewidth=0.15,
                        alpha=0.95,
                        zorder=2.5 + 0.1 * float(eval_idx),
                    )

            ax.axhline(0.0, color="#222222", linewidth=0.35, alpha=0.7)
            ax.set_xlim(0.3, max_comp + 0.7)
            ax.set_ylim(0.0, float(row_ymax_map.get(str(fit_condition), 0.05)))
            ax.set_xticks([1, 5, 10, 15, 20] if max_comp >= 20 else list(range(1, max_comp + 1)))
            ax.tick_params(axis="both", which="major", labelsize=5, pad=1.0)
            ax.grid(axis="y", linewidth=0.25, alpha=0.3)
            if row_idx == 0:
                ax.set_title(_region_display_label(region), fontsize=6, pad=1.5)
            if row_idx == n_rows - 1:
                ax.set_xlabel("PC", fontsize=6, labelpad=0.8)
            if col_idx == 0:
                fit_label = settings.condition_labels.get(fit_condition, fit_condition)
                ax.set_ylabel(f"{fit_label}\nExplained", fontsize=6, labelpad=1.0)

    handles = [
        Line2D(
            [0],
            [0],
            color=str(color_map.get(cond, DEFAULT_CONDITION_COLORS.get(cond, "#444444"))),
            lw=2.0,
            label=settings.condition_labels.get(cond, cond),
        )
        for cond in eval_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(handles)),
        fontsize=6,
        frameon=False,
    )
    fig.subplots_adjust(left=0.060, right=0.992, top=0.80, bottom=0.13, wspace=0.13, hspace=0.40)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    ext = _resolve_output_ext(settings)
    out_name = ensure_filename(output_filename, f".{ext}")
    out_path = out_root / out_name
    save_figure(
        fig,
        out_path,
        ext=ext,
        dpi=settings.output_dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)

    return {
        "output_path": str(out_path),
        "input_path": str(input_path),
        "regions": list(region_order),
        "fit_conditions": list(cond_order),
        "eval_conditions": list(eval_order),
        "max_components": int(max_comp),
    }


def plot_fixation_population_pca_explained_variance_cumulative(
    settings: FixationPopulationPCAPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    output_filename: str = "population_pca_explained_variance_cumulative",
) -> Optional[dict]:
    """Plot cumulative explained-variance curves across PCs."""
    _apply_plotting_style(settings.plotting_cfg_path)
    result_obj, input_path = _load_population_pca_result(settings)
    explained_df = _extract_cross_condition_explained_variance_df(result_obj)
    if explained_df.empty:
        print("[plot] no cross-condition explained variance rows found")
        return None

    cond_order = [cond for cond in settings.conditions if cond in set(explained_df["fit_condition"].unique())]
    if not cond_order:
        print("[plot] no matching fit conditions for cumulative explained variance plotting")
        return None

    region_order = _ordered_region_tokens(
        available=explained_df["region"].astype(str).unique().tolist(),
        requested=regions,
    )
    if not region_order:
        print("[plot] no matching regions for cumulative explained variance plotting")
        return None

    max_comp = max(1, int(settings.max_components_display))
    n_rows = len(cond_order)
    n_cols = max(4, len(region_order))
    fig = plt.figure(
        figsize=(
            float(settings.variance_letter_width_in),
            8.5 * float(settings.variance_letter_height_frac),
        ),
        dpi=settings.output_dpi,
    )
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    color_map = _resolve_condition_colors(settings)
    eval_order = [cond for cond in settings.conditions if cond in set(explained_df["eval_condition"].unique())]
    x = np.arange(1, max_comp + 1, dtype=float)
    row_ymax_map: dict[str, float] = {}
    for fit_condition in cond_order:
        row_sub = explained_df.loc[
            (explained_df["fit_condition"] == str(fit_condition))
            & (explained_df["n_components"] >= 1)
            & (explained_df["n_components"] <= max_comp)
        ].copy()
        if "explained_variance_cumulative_fraction" in row_sub.columns:
            row_vals = row_sub["explained_variance_cumulative_fraction"].to_numpy(dtype=float)
        else:
            row_vals = row_sub["explained_variance_fraction"].to_numpy(dtype=float)
        row_ymax_map[str(fit_condition)] = _rowwise_upper_limit(
            row_vals,
            floor=0.10,
        )

    for row_idx, fit_condition in enumerate(cond_order):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(region_order):
                ax.set_axis_off()
                continue
            region = region_order[col_idx]
            sub = explained_df.loc[
                (explained_df["region"] == str(region))
                & (explained_df["fit_condition"] == str(fit_condition))
                & (explained_df["n_components"] >= 1)
                & (explained_df["n_components"] <= max_comp)
            ].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            for eval_condition in eval_order:
                eval_sub = sub.loc[sub["eval_condition"] == str(eval_condition)].copy()
                if eval_sub.empty:
                    continue
                eval_sub = eval_sub.sort_values("n_components")
                per_pc = np.full((max_comp,), np.nan, dtype=float)
                comp_idx = np.asarray(eval_sub["n_components"], dtype=int) - 1
                valid = (comp_idx >= 0) & (comp_idx < max_comp)
                if np.any(valid):
                    per_pc[comp_idx[valid]] = np.asarray(
                        eval_sub.loc[valid, "explained_variance_fraction"],
                        dtype=float,
                    )
                cumulative = np.full((max_comp,), np.nan, dtype=float)
                running = 0.0
                for i, value in enumerate(per_pc):
                    if np.isfinite(value):
                        running += float(value)
                        cumulative[i] = running
                finite = np.isfinite(cumulative)
                if np.any(finite):
                    ax.plot(
                        x[finite],
                        cumulative[finite],
                        color=str(color_map.get(eval_condition, DEFAULT_CONDITION_COLORS.get(eval_condition, "#444444"))),
                        linewidth=1.0,
                        marker="o",
                        markersize=1.6,
                        markeredgewidth=0.0,
                        alpha=0.95,
                    )

            ax.axhline(0.0, color="#222222", linewidth=0.35, alpha=0.7)
            ax.set_xlim(0.8, max_comp + 0.2)
            ax.set_ylim(0.0, float(row_ymax_map.get(str(fit_condition), 0.10)))
            ax.set_xticks([1, 5, 10, 15, 20] if max_comp >= 20 else list(range(1, max_comp + 1)))
            ax.tick_params(axis="both", which="major", labelsize=5, pad=1.0)
            ax.grid(axis="y", linewidth=0.25, alpha=0.3)
            if row_idx == 0:
                ax.set_title(_region_display_label(region), fontsize=6, pad=1.5)
            if row_idx == n_rows - 1:
                ax.set_xlabel("PC", fontsize=6, labelpad=0.8)
            if col_idx == 0:
                fit_label = settings.condition_labels.get(fit_condition, fit_condition)
                ax.set_ylabel(f"{fit_label}\nCumulative", fontsize=6, labelpad=1.0)

    handles = [
        Line2D(
            [0],
            [0],
            color=str(color_map.get(cond, DEFAULT_CONDITION_COLORS.get(cond, "#444444"))),
            lw=1.8,
            marker="o",
            markersize=2.4,
            markeredgewidth=0.0,
            label=settings.condition_labels.get(cond, cond),
        )
        for cond in eval_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(handles)),
        fontsize=6,
        frameon=False,
    )
    fig.subplots_adjust(left=0.060, right=0.992, top=0.80, bottom=0.13, wspace=0.13, hspace=0.40)

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    ext = _resolve_output_ext(settings)
    out_name = ensure_filename(output_filename, f".{ext}")
    out_path = out_root / out_name
    save_figure(
        fig,
        out_path,
        ext=ext,
        dpi=settings.output_dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)

    return {
        "output_path": str(out_path),
        "input_path": str(input_path),
        "regions": list(region_order),
        "fit_conditions": list(cond_order),
        "eval_conditions": list(eval_order),
        "max_components": int(max_comp),
    }


def plot_fixation_population_pca_pairwise_geometry_violins(
    settings: FixationPopulationPCAPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    output_filename: str = "population_pca_pairwise_geometry_violin",
) -> list[dict]:
    """Plot four violin figures for pairwise PCA-trajectory geometry."""
    _apply_plotting_style(settings.plotting_cfg_path)
    if sns is None:
        raise ImportError(
            "seaborn is required for fixation population PCA pairwise geometry violin plots. "
            "Install seaborn or use an environment that includes it."
        )

    result_obj, input_path = _load_population_pca_result(settings)
    geometry_df = _extract_pairwise_geometry_df(result_obj)
    within_df = _extract_pairwise_geometry_stats_df(
        result_obj,
        "pairwise_geometry_within_region_stats",
    )
    cross_df = _extract_pairwise_geometry_stats_df(
        result_obj,
        "pairwise_geometry_cross_region_stats",
    )
    if geometry_df.empty:
        print("[plot] no pairwise geometry rows found in fixation population PCA results")
        return []

    region_order = _ordered_region_tokens(
        available=geometry_df["region"].astype(str).unique().tolist(),
        requested=regions,
    )
    if not region_order:
        print("[plot] no matching regions for pairwise geometry violin plotting")
        return []

    pair_order = [
        pair_token
        for pair_token in _condition_pair_order(settings)
        if pair_token in set(geometry_df["condition_pair"].astype(str).unique().tolist())
    ]
    for pair_token in sorted(geometry_df["condition_pair"].astype(str).unique().tolist()):
        if pair_token not in pair_order:
            pair_order.append(pair_token)
    if not pair_order:
        print("[plot] no condition pairs available for pairwise geometry violin plotting")
        return []

    metric_order = [
        metric
        for metric in ("euclidean_distance", "angle_degrees", "angle_radians")
        if metric in set(geometry_df["metric_name"].astype(str).unique().tolist())
    ]
    if not metric_order:
        print("[plot] no supported geometry metrics found for violin plotting")
        return []

    ext = _resolve_output_ext(settings)
    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    pair_colors = _resolve_condition_pair_colors(settings)
    region_colors = _resolve_region_colors(settings)
    figure_height = float(settings.pairwise_violin_letter_height_frac) * 11.0
    violin_width = 0.88
    n_pcs = int(
        result_obj.get("meta", {}).get(
            "geometry_n_pcs_effective_max",
            result_obj.get("meta", {}).get("geometry_n_pcs", settings.trajectory_n_pcs),
        )
    )
    if n_pcs <= 0 and "n_pcs_used" in geometry_df.columns:
        n_pcs_used = pd.to_numeric(geometry_df["n_pcs_used"], errors="coerce").to_numpy(dtype=float)
        n_pcs_used = n_pcs_used[np.isfinite(n_pcs_used)]
        if n_pcs_used.size > 0:
            n_pcs = int(np.max(n_pcs_used))
    outputs: list[dict] = []

    for metric_name in metric_order:
        metric_df = geometry_df.loc[
            (geometry_df["metric_name"].astype(str) == str(metric_name))
            & (geometry_df["region"].astype(str).isin(set(region_order)))
            & (geometry_df["condition_pair"].astype(str).isin(set(pair_order)))
        ].copy()
        if metric_df.empty:
            continue

        metric_label = str(metric_df["metric_label"].astype(str).iloc[0])
        y_vals = pd.to_numeric(metric_df["value"], errors="coerce").to_numpy(dtype=float)
        y_vals = y_vals[np.isfinite(y_vals)]
        y_max = float(np.nanmax(y_vals)) if y_vals.size > 0 else 1.0
        y_min = float(np.nanmin(y_vals)) if y_vals.size > 0 else 0.0
        span = max(y_max - y_min, 1e-6)
        step = 0.10 * span
        bar_h = 0.028 * span
        metric_suffix = _safe_suffix_token(metric_name)

        fig_within, ax_within = plt.subplots(
            1,
            1,
            figsize=(float(settings.pairwise_violin_letter_width_in), float(figure_height)),
            dpi=settings.output_dpi,
        )
        sns.violinplot(
            ax=ax_within,
            data=metric_df,
            x="region",
            y="value",
            hue="condition_pair",
            order=list(region_order),
            hue_order=list(pair_order),
            palette={
                pair_token: pair_colors.get(pair_token, "#777777")
                for pair_token in pair_order
            },
            inner="quart",
            cut=0.0,
            linewidth=0.8,
            width=violin_width,
        )
        for body in [artist for artist in ax_within.collections if isinstance(artist, PolyCollection)]:
            body.set_edgecolor("#222222")
            body.set_linewidth(0.65)
            body.set_alpha(0.72)
            body.set_rasterized(False)
        ax_within.set_xticks(np.arange(len(region_order)))
        ax_within.set_xticklabels(
            [_display_region(region, settings) for region in region_order],
            rotation=0,
            ha="center",
            fontsize=9,
        )
        ax_within.set_ylabel(str(metric_label), fontsize=9)
        ax_within.set_xlabel("")
        ax_within.grid(axis="y", alpha=0.23, linewidth=0.6)
        handles, labels = ax_within.get_legend_handles_labels()
        if handles:
            ax_within.legend(
                handles[: len(pair_order)],
                [_condition_pair_display_label(label, settings) for label in labels[: len(pair_order)]],
                title="Fixation Pair",
                frameon=False,
                ncol=max(1, len(pair_order)),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                fontsize=8,
                title_fontsize=8,
            )
        dodge_step = float(violin_width) / float(max(len(pair_order), 1))
        pos_map_within = {
            (str(region), str(pair_token)): (
                float(ridx) - float(violin_width) / 2.0 + (float(pidx) + 0.5) * dodge_step
            )
            for ridx, region in enumerate(region_order)
            for pidx, pair_token in enumerate(pair_order)
        }
        within_rows = (
            within_df.loc[
                (within_df["metric_name"].astype(str) == str(metric_name))
                & (within_df["region"].astype(str).isin(set(region_order)))
                & (within_df["significant_adjusted"].map(bool))
            ].copy()
            if not within_df.empty
            else pd.DataFrame()
        )
        if not within_rows.empty and "p_value_adjusted" in within_rows.columns:
            within_rows = within_rows.sort_values(["region", "p_value_adjusted"], na_position="last")
        within_counts: dict[str, int] = {}
        max_within_stack = 0
        for row in within_rows.itertuples(index=False):
            region = str(getattr(row, "region", ""))
            pair_a = str(getattr(row, "condition_pair_a", ""))
            pair_b = str(getattr(row, "condition_pair_b", ""))
            if (region, pair_a) not in pos_map_within or (region, pair_b) not in pos_map_within:
                continue
            stars = _significance_star(float(getattr(row, "p_value_adjusted", np.nan)))
            if not stars:
                continue
            level = int(within_counts.get(region, 0))
            y = y_max + 0.16 * span + step * float(level)
            x1 = float(pos_map_within[(region, pair_a)])
            x2 = float(pos_map_within[(region, pair_b)])
            ax_within.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], color="#222222", linewidth=0.8)
            ax_within.text((x1 + x2) / 2.0, y + bar_h + 0.01 * span, stars, ha="center", va="bottom", fontsize=9)
            within_counts[region] = level + 1
            max_within_stack = max(max_within_stack, level + 1)
        within_top_extra = max(0.22 * span, 0.16 * span + step * float(max_within_stack) + 0.18 * span)
        ax_within.set_ylim(y_min - 0.05 * span, y_max + within_top_extra)
        fig_within.suptitle(
            f"{metric_label} Between Condition Trajectories Within Region (top {n_pcs} PCs)",
            fontsize=10,
        )
        fig_within.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.18)
        out_name_within = ensure_filename(
            f"{output_filename}__{metric_suffix}__within_region",
            f".{ext}",
        )
        out_path_within = out_root / out_name_within
        save_figure(fig_within, out_path_within, ext=ext, dpi=settings.output_dpi)
        plt.close(fig_within)
        outputs.append(
            {
                "output_path": str(out_path_within),
                "input_path": str(input_path),
                "kind": "within_region",
                "metric_name": str(metric_name),
                "metric_label": str(metric_label),
                "regions": list(region_order),
                "condition_pairs": list(pair_order),
            }
        )

        fig_cross, ax_cross = plt.subplots(
            1,
            1,
            figsize=(float(settings.pairwise_violin_letter_width_in), float(figure_height)),
            dpi=settings.output_dpi,
        )
        sns.violinplot(
            ax=ax_cross,
            data=metric_df,
            x="condition_pair",
            y="value",
            hue="region",
            order=list(pair_order),
            hue_order=list(region_order),
            palette={
                region: region_colors.get(
                    str(region),
                    region_colors.get(_normalize_region_token(region), "#777777"),
                )
                for region in region_order
            },
            inner="quart",
            cut=0.0,
            linewidth=0.8,
            width=violin_width,
        )
        for body in [artist for artist in ax_cross.collections if isinstance(artist, PolyCollection)]:
            body.set_edgecolor("#222222")
            body.set_linewidth(0.65)
            body.set_alpha(0.70)
            body.set_rasterized(False)
        ax_cross.set_xticks(np.arange(len(pair_order)))
        ax_cross.set_xticklabels(
            [_condition_pair_axis_label(pair_token, settings) for pair_token in pair_order],
            rotation=0,
            ha="center",
            fontsize=8,
        )
        ax_cross.set_ylabel(str(metric_label), fontsize=9)
        ax_cross.set_xlabel("")
        ax_cross.grid(axis="y", alpha=0.23, linewidth=0.6)
        handles, labels = ax_cross.get_legend_handles_labels()
        if handles:
            ax_cross.legend(
                handles[: len(region_order)],
                [_display_region(label, settings) for label in labels[: len(region_order)]],
                title="Region",
                frameon=False,
                ncol=max(1, len(region_order)),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                fontsize=8,
                title_fontsize=8,
            )
        dodge_step = float(violin_width) / float(max(len(region_order), 1))
        pos_map_cross = {
            (str(pair_token), str(region)): (
                float(pidx) - float(violin_width) / 2.0 + (float(ridx) + 0.5) * dodge_step
            )
            for pidx, pair_token in enumerate(pair_order)
            for ridx, region in enumerate(region_order)
        }
        cross_rows = (
            cross_df.loc[
                (cross_df["metric_name"].astype(str) == str(metric_name))
                & (cross_df["condition_pair"].astype(str).isin(set(pair_order)))
                & (cross_df["significant_adjusted"].map(bool))
            ].copy()
            if not cross_df.empty
            else pd.DataFrame()
        )
        if not cross_rows.empty and "p_value_adjusted" in cross_rows.columns:
            cross_rows = cross_rows.sort_values(["condition_pair", "p_value_adjusted"], na_position="last")
        cross_counts: dict[str, int] = {}
        max_cross_stack = 0
        for row in cross_rows.itertuples(index=False):
            pair_token = str(getattr(row, "condition_pair", ""))
            region_a = str(getattr(row, "region_a", ""))
            region_b = str(getattr(row, "region_b", ""))
            if (pair_token, region_a) not in pos_map_cross or (pair_token, region_b) not in pos_map_cross:
                continue
            stars = _significance_star(float(getattr(row, "p_value_adjusted", np.nan)))
            if not stars:
                continue
            level = int(cross_counts.get(pair_token, 0))
            y = y_max + 0.16 * span + step * float(level)
            x1 = float(pos_map_cross[(pair_token, region_a)])
            x2 = float(pos_map_cross[(pair_token, region_b)])
            bar_color = pair_colors.get(pair_token, "#444444")
            ax_cross.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], color=bar_color, linewidth=0.95)
            ax_cross.text((x1 + x2) / 2.0, y + bar_h + 0.01 * span, stars, ha="center", va="bottom", fontsize=9, color=bar_color)
            cross_counts[pair_token] = level + 1
            max_cross_stack = max(max_cross_stack, level + 1)
        cross_top_extra = max(0.22 * span, 0.16 * span + step * float(max_cross_stack) + 0.18 * span)
        ax_cross.set_ylim(y_min - 0.05 * span, y_max + cross_top_extra)
        fig_cross.suptitle(
            f"{metric_label} Between Condition Trajectories Across Regions (top {n_pcs} PCs)",
            fontsize=10,
        )
        fig_cross.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.22)
        out_name_cross = ensure_filename(
            f"{output_filename}__{metric_suffix}__cross_region",
            f".{ext}",
        )
        out_path_cross = out_root / out_name_cross
        save_figure(fig_cross, out_path_cross, ext=ext, dpi=settings.output_dpi)
        plt.close(fig_cross)
        outputs.append(
            {
                "output_path": str(out_path_cross),
                "input_path": str(input_path),
                "kind": "cross_region",
                "metric_name": str(metric_name),
                "metric_label": str(metric_label),
                "regions": list(region_order),
                "condition_pairs": list(pair_order),
            }
        )

    return outputs
