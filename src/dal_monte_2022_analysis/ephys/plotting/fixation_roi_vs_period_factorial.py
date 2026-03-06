"""Plotting helpers for ROI-vs-period factorial analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except Exception:  # pragma: no cover - handled explicitly at runtime
    sns = None

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import apply_plotting_config
from dal_monte_2022_analysis.runtime.io.plot_output import normalize_extension, save_figure
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")
DEFAULT_REGION_LABELS: dict[str, str] = {
    "bla": "BLA",
    "accg": "ACCg",
    "dmpfc": "dmPFC",
    "ofc": "OFC",
}
DEFAULT_AXIS_ORDER: tuple[str, ...] = (
    "face_object",
    "interactive_state",
    "cross_interaction",
)
DEFAULT_AXIS_LABELS: dict[str, str] = {
    "face_object": "Face-Object",
    "interactive_state": "Interactive State",
    "cross_interaction": "Cross Interaction",
}
DEFAULT_AXIS_COLORS: dict[str, str] = {
    "face_object": "#cc4c02",
    "interactive_state": "#2b8cbe",
    "cross_interaction": "#7a0177",
}
_WINDOW_SORT_ORDER: dict[str, int] = {
    "pre_fix": 0,
    "peri_fix": 1,
    "post_fix": 2,
    "avg_pre_peri_post": 3,
}
_ALLOWED_AXIS_COMPARISON_MODES = {"split_by_window", "averaged_across_windows"}


@dataclass
class FixationROIVsPeriodFactorialPlotSettings:
    """Configuration for ROI-vs-period factorial plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_roi_vs_period_factorial"
    input_filename: str = "results.pkl"
    output_subdir: str = "ephys/psth/fixation_roi_vs_period_factorial/plots"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 300
    axis_magnitude_source: str = "cell_means"
    axis_comparison_mode: str = "averaged_across_windows"
    alpha: float = 0.05
    region_order: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_REGION_ORDER),
    )
    region_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_REGION_LABELS),
    )
    axis_order: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_AXIS_ORDER),
    )
    axis_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_AXIS_LABELS),
    )
    axis_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_AXIS_COLORS),
    )
    violin_letter_width_in: float = 8.5
    violin_letter_height_frac: float = 0.28
    graph_letter_width_in: float = 8.5
    graph_letter_height_frac: float = 0.24
    geometry_letter_width_in: float = 8.5
    geometry_letter_height_frac: float = 0.22
    network_node_size: float = 620.0


def _resolve_output_ext(settings: FixationROIVsPeriodFactorialPlotSettings) -> str:
    return normalize_extension(settings.output_extension, fallback="pdf")


def _normalize_mode(mode: object) -> str:
    token = str(mode).strip().lower()
    aliases = {
        "split": "split_by_window",
        "window_split": "split_by_window",
        "split_by_window": "split_by_window",
        "averaged": "averaged_across_windows",
        "average": "averaged_across_windows",
        "averaged_across_windows": "averaged_across_windows",
    }
    resolved = aliases.get(token, token)
    if resolved not in _ALLOWED_AXIS_COMPARISON_MODES:
        return "averaged_across_windows"
    return resolved


def _as_df(raw: object) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    if isinstance(raw, dict):
        return pd.DataFrame(raw)
    return pd.DataFrame()


def _apply_plotting_style(settings: FixationROIVsPeriodFactorialPlotSettings) -> None:
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(cfg)


def _load_result_payload(
    settings: FixationROIVsPeriodFactorialPlotSettings,
) -> tuple[dict, Path]:
    cfg = load_config(settings.cfg_path)
    in_root = build_analysis_output_dir(cfg, settings.input_subdir)
    in_path = in_root / ensure_filename(settings.input_filename, ".pkl")
    if not in_path.exists():
        raise FileNotFoundError(f"ROI-vs-period results not found: {in_path}")
    payload = load_pickle_path(in_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in ROI-vs-period results: {in_path}")
    return payload, in_path


def _normalize_region(region: object) -> str:
    return str(region).strip().lower()


def _display_region(region: object, settings: FixationROIVsPeriodFactorialPlotSettings) -> str:
    token = _normalize_region(region)
    return settings.region_labels.get(token, str(region))


def _display_axis(axis_name: object, settings: FixationROIVsPeriodFactorialPlotSettings) -> str:
    token = str(axis_name).strip()
    return settings.axis_labels.get(token, token)


def _ordered_tokens(
    *,
    available: Sequence[str],
    preferred: Sequence[str],
) -> list[str]:
    available_set = {str(tok).strip().lower() for tok in available if str(tok).strip()}
    out: list[str] = []
    for tok in preferred:
        key = str(tok).strip().lower()
        if key in available_set and key not in out:
            out.append(key)
    for tok in sorted(available_set):
        if tok not in out:
            out.append(tok)
    return out


def _window_order(windows: Sequence[str]) -> list[str]:
    uniq = {str(win).strip() for win in windows if str(win).strip()}
    return sorted(uniq, key=lambda win: (_WINDOW_SORT_ORDER.get(str(win), 100), str(win)))


def _safe_suffix_token(token: object) -> str:
    out = str(token).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return out if out else "unknown"


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


def _extract_axis_magnitude_units(
    payload: dict,
    settings: FixationROIVsPeriodFactorialPlotSettings,
) -> pd.DataFrame:
    unit_axis_df = _as_df(payload.get("unit_axis_values"))
    if unit_axis_df.empty:
        return pd.DataFrame()
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    mode = _normalize_mode(meta.get("axis_comparison_mode", settings.axis_comparison_mode))
    sig_windows = [
        str(win)
        for win in meta.get("significance_windows", ("pre_fix", "peri_fix", "post_fix"))
        if str(win).strip()
    ]

    df = unit_axis_df.copy()
    df["axis_source"] = df["axis_source"].astype(str)
    df = df.loc[df["axis_source"] == str(settings.axis_magnitude_source)].copy()
    if df.empty:
        return pd.DataFrame()
    if "counts_toward_significance" in df.columns:
        df = df.loc[df["counts_toward_significance"].map(bool)].copy()
    elif "window_name" in df.columns:
        df = df.loc[df["window_name"].astype(str).isin(set(sig_windows))].copy()
    if df.empty:
        return pd.DataFrame()

    df["region"] = df["region"].fillna("unknown").astype(str).str.lower()
    df["axis_name"] = df["axis_name"].astype(str)
    df["window_name"] = df["window_name"].astype(str)
    df["value_abs_norm"] = np.abs(pd.to_numeric(df["value_signed"], errors="coerce"))
    df = df.loc[df["value_abs_norm"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    if mode == "split_by_window":
        out = df.copy()
        out["axis_comparison_mode"] = mode
        return out

    # averaged_across_windows
    group_cols = [col for col in ("unit_key", "region", "axis_name", "axis_source") if col in df.columns]
    rows: list[dict] = []
    for key_vals, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = {col: val for col, val in zip(group_cols, key_vals)}
        vals = grp["value_abs_norm"].to_numpy(dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        row.update(
            {
                "axis_comparison_mode": mode,
                "window_name": "avg_pre_peri_post",
                "value_abs_norm": float(np.mean(vals)),
                "n_windows_averaged": int(vals.size),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _extract_within_axis_df(
    payload: dict,
    settings: FixationROIVsPeriodFactorialPlotSettings,
) -> pd.DataFrame:
    df = _as_df(payload.get("region_axis_within_region"))
    if df.empty:
        return pd.DataFrame()
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    mode = _normalize_mode(meta.get("axis_comparison_mode", settings.axis_comparison_mode))
    df = df.copy()
    if "axis_source" in df.columns:
        df = df.loc[df["axis_source"].astype(str) == str(settings.axis_magnitude_source)].copy()
    if "axis_comparison_mode" in df.columns:
        df = df.loc[df["axis_comparison_mode"].astype(str) == str(mode)].copy()
    if "region" in df.columns:
        df["region"] = df["region"].fillna("unknown").astype(str).str.lower()
    if "window_name" in df.columns:
        df["window_name"] = df["window_name"].astype(str)
    if "significant_adjusted" in df.columns:
        df["significant_adjusted"] = df["significant_adjusted"].map(bool)
    if "p_value_adjusted" in df.columns:
        df["p_value_adjusted"] = pd.to_numeric(df["p_value_adjusted"], errors="coerce")
    return df


def _extract_cross_region_axis_df(
    payload: dict,
    settings: FixationROIVsPeriodFactorialPlotSettings,
) -> pd.DataFrame:
    df = _as_df(payload.get("region_axis_pairwise"))
    if df.empty:
        return pd.DataFrame()
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    mode = _normalize_mode(meta.get("axis_comparison_mode", settings.axis_comparison_mode))
    df = df.copy()
    if "axis_source" in df.columns:
        df = df.loc[df["axis_source"].astype(str) == str(settings.axis_magnitude_source)].copy()
    if "axis_comparison_mode" in df.columns:
        df = df.loc[df["axis_comparison_mode"].astype(str) == str(mode)].copy()
    for col in ("region_a", "region_b"):
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.lower()
    if "window_name" in df.columns:
        df["window_name"] = df["window_name"].astype(str)
    if "axis_name" in df.columns:
        df["axis_name"] = df["axis_name"].astype(str)
    if "significant_adjusted" in df.columns:
        df["significant_adjusted"] = df["significant_adjusted"].map(bool)
    if "p_value_adjusted" in df.columns:
        df["p_value_adjusted"] = pd.to_numeric(df["p_value_adjusted"], errors="coerce")
    if "delta_mean_a_minus_b" in df.columns:
        df["delta_mean_a_minus_b"] = pd.to_numeric(df["delta_mean_a_minus_b"], errors="coerce")
    return df


def _build_output_root(settings: FixationROIVsPeriodFactorialPlotSettings) -> Path:
    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def plot_fixation_roi_vs_period_axis_violin(
    settings: FixationROIVsPeriodFactorialPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    window: Optional[str] = None,
    output_filename: str = "roi_vs_period_axis_violin",
) -> list[dict]:
    """Plot region-wise axis magnitude violins with within-region significance stars."""
    _apply_plotting_style(settings)
    payload, _ = _load_result_payload(settings)
    mag_df = _extract_axis_magnitude_units(payload, settings)
    within_df = _extract_within_axis_df(payload, settings)
    if mag_df.empty:
        print("[plot] no axis-magnitude unit rows available for violin plot")
        return []

    region_tokens = _ordered_tokens(
        available=mag_df["region"].astype(str).unique().tolist(),
        preferred=[str(tok).lower() for tok in (regions or settings.region_order)],
    )
    axis_tokens = _ordered_tokens(
        available=mag_df["axis_name"].astype(str).unique().tolist(),
        preferred=[str(tok) for tok in settings.axis_order],
    )
    if not region_tokens or not axis_tokens:
        print("[plot] unable to resolve regions/axes for violin plot")
        return []

    windows_available = mag_df["window_name"].astype(str).unique().tolist()
    if window is not None:
        window_tokens = [str(window)]
    else:
        window_tokens = _window_order(windows_available)
    if not window_tokens:
        print("[plot] no window rows available for violin plot")
        return []

    ext = _resolve_output_ext(settings)
    out_root = _build_output_root(settings)
    outputs: list[dict] = []
    letter_h = 11.0
    fig_h = float(settings.violin_letter_height_frac) * letter_h

    for win_name in window_tokens:
        win_df = mag_df.loc[mag_df["window_name"].astype(str) == str(win_name)].copy()
        if win_df.empty:
            continue
        if sns is None:
            raise ImportError(
                "seaborn is required for ROI-vs-period violin plotting. "
                "Install seaborn or use an environment that includes it."
            )

        fig, axes = plt.subplots(
            1,
            len(region_tokens),
            figsize=(float(settings.violin_letter_width_in), float(fig_h)),
            dpi=settings.output_dpi,
            squeeze=False,
        )
        axes = axes.ravel()

        for ridx, region in enumerate(region_tokens):
            ax = axes[ridx]
            reg_df = win_df.loc[win_df["region"].astype(str) == str(region)].copy()
            plot_df = reg_df.loc[reg_df["axis_name"].astype(str).isin(set(axis_tokens))].copy()
            if not plot_df.empty:
                plot_df = plot_df.loc[:, ["axis_name", "value_abs_norm"]].copy()
                plot_df["axis_name"] = plot_df["axis_name"].astype(str)
                palette = {
                    axis_name: settings.axis_colors.get(str(axis_name), "#777777")
                    for axis_name in axis_tokens
                }
                sns.violinplot(
                    ax=ax,
                    data=plot_df,
                    x="axis_name",
                    y="value_abs_norm",
                    hue="axis_name",
                    order=list(axis_tokens),
                    hue_order=list(axis_tokens),
                    palette=palette,
                    legend=False,
                    width=0.78,
                    inner="quart",
                    cut=0.0,
                    linewidth=0.8,
                )
                bodies = [artist for artist in ax.collections if isinstance(artist, PolyCollection)]
                for body in bodies:
                    body.set_edgecolor("#222222")
                    body.set_linewidth(0.75)
                    body.set_alpha(0.78)
                    body.set_rasterized(False)
            ax.set_xticks(np.arange(len(axis_tokens)))
            ax.set_xticklabels([_display_axis(axis_name, settings) for axis_name in axis_tokens], rotation=26, ha="right", fontsize=8)
            ax.set_title(_display_region(region, settings), fontsize=10)
            ax.grid(axis="y", alpha=0.23, linewidth=0.6)
            if ridx == 0:
                ax.set_ylabel("|Axis Magnitude| (a.u.)", fontsize=9)

            y_vals = reg_df["value_abs_norm"].to_numpy(dtype=float).reshape(-1)
            y_vals = y_vals[np.isfinite(y_vals)]
            if y_vals.size > 0:
                y_concat = y_vals
                y_max = float(np.nanmax(y_concat)) if np.any(np.isfinite(y_concat)) else 1.0
                y_min = float(np.nanmin(y_concat)) if np.any(np.isfinite(y_concat)) else 0.0
            else:
                y_max = 1.0
                y_min = 0.0
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=9)
            span = max(y_max - y_min, 1e-6)
            step = 0.11 * span
            bar_h = 0.03 * span
            base_y = y_max + 0.12 * span

            pos_map = {axis_name: idx for idx, axis_name in enumerate(axis_tokens)}
            within_rows = (
                within_df.loc[
                    (within_df["region"].astype(str) == str(region))
                    & (within_df["window_name"].astype(str) == str(win_name))
                    & (within_df["significant_adjusted"].map(bool))
                ].copy()
                if not within_df.empty
                else pd.DataFrame()
            )
            if not within_rows.empty and "p_value_adjusted" in within_rows.columns:
                within_rows = within_rows.sort_values("p_value_adjusted", na_position="last")
            n_bars = 0
            for row in within_rows.itertuples(index=False):
                axis_a = str(getattr(row, "axis_a", ""))
                axis_b = str(getattr(row, "axis_b", ""))
                if axis_a not in pos_map or axis_b not in pos_map:
                    continue
                p_adj = float(getattr(row, "p_value_adjusted", np.nan))
                stars = _significance_star(p_adj)
                if not stars:
                    continue
                x1 = float(pos_map[axis_a])
                x2 = float(pos_map[axis_b])
                y = base_y + step * float(n_bars)
                ax.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], color="black", linewidth=0.85)
                ax.text((x1 + x2) / 2.0, y + bar_h + 0.01 * span, stars, ha="center", va="bottom", fontsize=10)
                n_bars += 1

            top_extra = max(0.22 * span, step * max(1, n_bars) + 0.2 * span)
            ax.set_ylim(y_min - 0.05 * span, y_max + top_extra)

        mode = str(payload.get("meta", {}).get("axis_comparison_mode", settings.axis_comparison_mode))
        fig.suptitle(
            f"ROI-vs-Period Axis Magnitude by Region ({win_name}; mode={mode}; source={settings.axis_magnitude_source})",
            fontsize=10,
        )
        fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.30, wspace=0.30)

        suffix = f"__window={_safe_suffix_token(win_name)}"
        out_name = ensure_filename(f"{output_filename}{suffix}", f".{ext}")
        out_path = out_root / out_name
        save_figure(fig, out_path, ext=ext, dpi=settings.output_dpi)
        plt.close(fig)
        outputs.append(
            {
                "output_path": str(out_path),
                "window_name": str(win_name),
                "regions": list(region_tokens),
                "axes": list(axis_tokens),
            }
        )
    return outputs


def _square_positions(region_tokens: Sequence[str]) -> dict[str, tuple[float, float]]:
    tokens = [str(tok).lower() for tok in region_tokens]
    if len(tokens) < 4:
        # fallback: place on unit circle
        out: dict[str, tuple[float, float]] = {}
        n = max(len(tokens), 1)
        for idx, token in enumerate(tokens):
            theta = 2.0 * np.pi * float(idx) / float(n)
            out[token] = (float(np.cos(theta)), float(np.sin(theta)))
        return out
    return {
        tokens[0]: (0.0, 1.0),  # top-left
        tokens[1]: (1.0, 1.0),  # top-right
        tokens[2]: (0.0, 0.0),  # bottom-left
        tokens[3]: (1.0, 0.0),  # bottom-right
    }


def plot_fixation_roi_vs_period_cross_region_graph(
    settings: FixationROIVsPeriodFactorialPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    window: Optional[str] = None,
    output_filename: str = "roi_vs_period_cross_region_graph",
) -> list[dict]:
    """Plot significant cross-region comparisons as square-node graphs per axis."""
    _apply_plotting_style(settings)
    payload, _ = _load_result_payload(settings)
    pair_df = _extract_cross_region_axis_df(payload, settings)
    if pair_df.empty:
        print("[plot] no cross-region axis pairwise rows found")
        return []

    axis_tokens = _ordered_tokens(
        available=pair_df["axis_name"].astype(str).unique().tolist(),
        preferred=[str(tok) for tok in settings.axis_order],
    )
    region_available = sorted(
        set(pair_df["region_a"].astype(str).tolist()) | set(pair_df["region_b"].astype(str).tolist()),
    )
    region_tokens = _ordered_tokens(
        available=region_available,
        preferred=[str(tok).lower() for tok in (regions or settings.region_order)],
    )
    if not axis_tokens or not region_tokens:
        print("[plot] unable to resolve axes/regions for cross-region graph")
        return []

    if window is not None:
        window_tokens = [str(window)]
    else:
        window_tokens = _window_order(pair_df["window_name"].astype(str).unique().tolist())
    if not window_tokens:
        print("[plot] no windows available for cross-region graph")
        return []

    ext = _resolve_output_ext(settings)
    out_root = _build_output_root(settings)
    outputs: list[dict] = []
    letter_h = 11.0
    fig_h = float(settings.graph_letter_height_frac) * letter_h
    node_pos = _square_positions(region_tokens)

    for win_name in window_tokens:
        win_df = pair_df.loc[pair_df["window_name"].astype(str) == str(win_name)].copy()
        if win_df.empty:
            continue
        fig, axes = plt.subplots(
            1,
            len(axis_tokens),
            figsize=(float(settings.graph_letter_width_in), float(fig_h)),
            dpi=settings.output_dpi,
            squeeze=False,
        )
        axes = axes.ravel()

        for aidx, axis_name in enumerate(axis_tokens):
            ax = axes[aidx]
            axis_df = win_df.loc[win_df["axis_name"].astype(str) == str(axis_name)].copy()
            axis_df = axis_df.loc[axis_df["significant_adjusted"].map(bool)].copy()

            # draw a light square frame
            if len(region_tokens) >= 4:
                square = [region_tokens[0], region_tokens[1], region_tokens[3], region_tokens[2], region_tokens[0]]
                sx = [node_pos[token][0] for token in square]
                sy = [node_pos[token][1] for token in square]
                ax.plot(sx, sy, color="#d0d0d0", linewidth=0.8, zorder=0)

            mean_diff = (
                pd.to_numeric(axis_df["mean_a"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(axis_df["mean_b"], errors="coerce").to_numpy(dtype=float)
                if (not axis_df.empty and "mean_a" in axis_df.columns and "mean_b" in axis_df.columns)
                else 0.0
            )
            max_abs_diff = (
                float(np.nanmax(np.abs(mean_diff)))
                if isinstance(mean_diff, np.ndarray) and mean_diff.size > 0
                else 0.0
            )
            max_abs_diff = max(max_abs_diff, 1e-9)
            for row in axis_df.itertuples(index=False):
                ra = str(getattr(row, "region_a", "")).lower()
                rb = str(getattr(row, "region_b", "")).lower()
                if ra not in node_pos or rb not in node_pos:
                    continue
                mean_a = float(getattr(row, "mean_a", np.nan))
                mean_b = float(getattr(row, "mean_b", np.nan))
                high = ra if (np.isfinite(mean_a) and np.isfinite(mean_b) and mean_a >= mean_b) else rb
                low = rb if high == ra else ra
                diff = abs(mean_a - mean_b) if (np.isfinite(mean_a) and np.isfinite(mean_b)) else np.nan
                lw = 1.2 + 2.6 * (float(diff) / max_abs_diff if np.isfinite(diff) else 0.2)
                x_high, y_high = node_pos[high]
                x_low, y_low = node_pos[low]
                ax.plot(
                    [x_high, x_low],
                    [y_high, y_low],
                    color="#4d4d4d",
                    linewidth=lw,
                    alpha=0.9,
                    zorder=1,
                )
                tip_x = x_high + 0.75 * (x_low - x_high)
                tip_y = y_high + 0.75 * (y_low - y_high)
                tail_x = x_high + 0.55 * (x_low - x_high)
                tail_y = y_high + 0.55 * (y_low - y_high)
                ax.annotate(
                    "",
                    xy=(tip_x, tip_y),
                    xytext=(tail_x, tail_y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#4d4d4d",
                        "lw": max(1.0, 0.85 * lw),
                        "alpha": 0.95,
                        "mutation_scale": 9.0 + 1.2 * lw,
                        "shrinkA": 0.0,
                        "shrinkB": 0.0,
                    },
                    zorder=2,
                )

            xs = [node_pos[token][0] for token in region_tokens if token in node_pos]
            ys = [node_pos[token][1] for token in region_tokens if token in node_pos]
            ax.scatter(
                xs,
                ys,
                s=float(settings.network_node_size),
                marker="s",
                facecolor="#f7f7f7",
                edgecolor="#333333",
                linewidth=0.9,
                zorder=3,
            )
            for token in region_tokens:
                if token not in node_pos:
                    continue
                x, y = node_pos[token]
                ax.text(x, y, _display_region(token, settings), ha="center", va="center", fontsize=8, zorder=5)

            ax.set_title(_display_axis(axis_name, settings), fontsize=10)
            if axis_df.empty:
                ax.text(0.5, -0.20, "No significant pairs", ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_xlim(-0.25, 1.25)
            ax.set_ylim(-0.25, 1.25)
            ax.set_aspect("equal")
            ax.axis("off")

        mode = str(payload.get("meta", {}).get("axis_comparison_mode", settings.axis_comparison_mode))
        fig.suptitle(
            f"Cross-Region Significant Axis Differences ({win_name}; mode={mode}; source={settings.axis_magnitude_source})",
            fontsize=10,
        )
        fig.subplots_adjust(left=0.03, right=0.99, top=0.82, bottom=0.12, wspace=0.22)

        suffix = f"__window={_safe_suffix_token(win_name)}"
        out_name = ensure_filename(f"{output_filename}{suffix}", f".{ext}")
        out_path = out_root / out_name
        save_figure(fig, out_path, ext=ext, dpi=settings.output_dpi)
        plt.close(fig)
        outputs.append(
            {
                "output_path": str(out_path),
                "window_name": str(win_name),
                "axes": list(axis_tokens),
                "regions": list(region_tokens),
            }
        )
    return outputs


def plot_fixation_roi_vs_period_axis_geometry(
    settings: FixationROIVsPeriodFactorialPlotSettings,
    *,
    output_filename: str = "roi_vs_period_axis_geometry",
) -> dict:
    """Plot a geometric sketch showing how the three ROI-vs-period axes are defined."""
    _apply_plotting_style(settings)
    _load_result_payload(settings)  # validates input exists in current analysis context
    ext = _resolve_output_ext(settings)
    out_root = _build_output_root(settings)
    letter_h = 11.0
    fig_h = float(settings.geometry_letter_height_frac) * letter_h
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(float(settings.geometry_letter_width_in), float(fig_h)),
        dpi=settings.output_dpi,
        squeeze=True,
    )

    cond_pos = {
        "ON": (0.0, 0.0),
        "OI": (0.0, 1.0),
        "FN": (1.0, 0.0),
        "FI": (1.0, 1.0),
    }
    cond_labels = {
        "ON": "object, non-int",
        "OI": "object, int",
        "FN": "face, non-int",
        "FI": "face, int",
    }
    sq = ["ON", "OI", "FI", "FN", "ON"]
    xs = [cond_pos[k][0] for k in sq]
    ys = [cond_pos[k][1] for k in sq]
    ax.plot(xs, ys, color="#bdbdbd", linewidth=1.0, zorder=0)
    for key, (x, y) in cond_pos.items():
        ax.scatter([x], [y], s=42, color="#333333", zorder=3)
        ax.text(x + 0.03, y + 0.03, key, fontsize=8, fontweight="bold")
        ax.text(x + 0.03, y - 0.10, cond_labels[key], fontsize=7)

    # Base axes in one shared geometric space.
    ax.annotate(
        "",
        xy=(0.98, 0.50),
        xytext=(0.02, 0.50),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#cc4c02"),
    )
    ax.annotate(
        "",
        xy=(0.50, 0.98),
        xytext=(0.50, 0.02),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#2b8cbe"),
    )
    # Cross-interaction is a diagonal contrast across condition pairs.
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#7a0177",
        linewidth=2.0,
        linestyle="--",
        zorder=1,
    )
    ax.text(-0.06, -0.04, "+", color="#7a0177", fontsize=11, fontweight="bold")
    ax.text(1.03, 1.01, "+", color="#7a0177", fontsize=11, fontweight="bold")
    ax.text(-0.06, 1.01, "-", color="#7a0177", fontsize=11, fontweight="bold")
    ax.text(1.03, -0.04, "-", color="#7a0177", fontsize=11, fontweight="bold")

    ax.text(0.50, 1.10, "Interactive-State: ((FI + OI) - (FN + ON)) / 2", color="#2b8cbe", ha="center", va="center", fontsize=8)
    ax.text(0.50, -0.28, "Face-Object: ((FI + FN) - (OI + ON)) / 2", color="#cc4c02", ha="center", va="center", fontsize=8)
    ax.text(0.50, -0.40, "Cross-Interaction: (FI - OI) - (FN - ON) = (FI + ON) - (OI + FN)", color="#7a0177", ha="center", va="center", fontsize=8)

    ax.set_xlim(-0.18, 1.22)
    ax.set_ylim(-0.48, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("ROI-vs-Period Axis Geometry (Shared Space)", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.79, bottom=0.06)

    out_name = ensure_filename(output_filename, f".{ext}")
    out_path = out_root / out_name
    save_figure(fig, out_path, ext=ext, dpi=settings.output_dpi)
    plt.close(fig)
    return {"output_path": str(out_path)}
