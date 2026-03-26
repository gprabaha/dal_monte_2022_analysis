"""Plotting helpers for ROI-vs-period factorial analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except Exception:  # pragma: no cover - handled explicitly at runtime
    sns = None
try:
    import pyvista as pv
except Exception:  # pragma: no cover - optional dependency
    pv = None

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import apply_plotting_config
from dal_monte_2022_analysis.runtime.io.plot_output import normalize_extension, save_figure
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir


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
DEFAULT_REGION_OUTLINE_COLORS: dict[str, str] = {
    "bla": "#7f3b08",
    "accg": "#1b9e77",
    "dmpfc": "#7570b3",
    "ofc": "#e7298a",
}
_WINDOW_SORT_ORDER: dict[str, int] = {
    "pre_fix": 0,
    "peri_fix": 1,
    "post_fix": 2,
    "avg_pre_peri_post": 3,
    "max_abs_across_windows": 4,
}
_ALLOWED_AXIS_COMPARISON_MODES = {
    "split_by_window",
    "averaged_across_windows",
    "max_abs_across_windows",
}
_PYVISTA_RUNTIME_READY = False
_VECTOR_EXTS = {".pdf", ".svg", ".eps", ".ps", ".tex"}


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
    axis_comparison_mode: str = "max_abs_across_windows"
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
    region_outline_colors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_REGION_OUTLINE_COLORS),
    )
    violin_letter_width_in: float = 8.5
    violin_letter_height_frac: float = 0.28
    graph_letter_width_in: float = 8.5
    graph_letter_height_frac: float = 0.24
    axis_space_regions_letter_width_in: float = 8.5
    axis_space_regions_letter_height_frac: float = 0.30
    axis_space_overlay_letter_width_in: float = 8.5
    axis_space_overlay_letter_height_frac: float = 0.30
    axis_space_disk_alpha: float = 0.35
    axis_space_disk_layers: int = 14
    axis_space_quantile_low: float = 0.025
    axis_space_quantile_high: float = 0.975
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
        "max": "max_abs_across_windows",
        "max_abs": "max_abs_across_windows",
        "max_abs_across_windows": "max_abs_across_windows",
        "max_across_windows": "max_abs_across_windows",
        "averaged": "averaged_across_windows",
        "average": "averaged_across_windows",
        "averaged_across_windows": "averaged_across_windows",
    }
    resolved = aliases.get(token, token)
    if resolved not in _ALLOWED_AXIS_COMPARISON_MODES:
        return "max_abs_across_windows"
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


def _extract_collapsed_axis_units(
    payload: dict,
    settings: FixationROIVsPeriodFactorialPlotSettings,
    *,
    mode: str,
    require_signed: bool,
) -> pd.DataFrame:
    if str(mode) == "split_by_window":
        return pd.DataFrame()
    df = _as_df(payload.get("unit_axis_collapsed"))
    if df.empty:
        return pd.DataFrame()
    if "axis_source" in df.columns:
        df["axis_source"] = df["axis_source"].astype(str)
        df = df.loc[df["axis_source"] == str(settings.axis_magnitude_source)].copy()
    if "axis_comparison_mode" in df.columns:
        df = df.loc[df["axis_comparison_mode"].astype(str) == str(mode)].copy()
    else:
        df["axis_comparison_mode"] = str(mode)
    if df.empty:
        return pd.DataFrame()
    if "unit_key" not in df.columns:
        df = df.reset_index().rename(columns={"index": "unit_key"})
    df["unit_key"] = df["unit_key"].astype(str)
    if "region" in df.columns:
        df["region"] = df["region"].fillna("unknown").astype(str).str.lower()
    if "axis_name" in df.columns:
        df["axis_name"] = df["axis_name"].astype(str)
    if "window_name" in df.columns:
        df["window_name"] = df["window_name"].astype(str)
    if require_signed:
        if "value_signed" not in df.columns:
            return pd.DataFrame()
        df["value_signed"] = pd.to_numeric(df["value_signed"], errors="coerce")
        return df.loc[df["value_signed"].notna()].copy()
    if "value_abs" in df.columns:
        df["value_abs_norm"] = pd.to_numeric(df["value_abs"], errors="coerce")
    elif "value_signed" in df.columns:
        df["value_abs_norm"] = np.abs(pd.to_numeric(df["value_signed"], errors="coerce"))
    else:
        return pd.DataFrame()
    return df.loc[df["value_abs_norm"].notna()].copy()


def _extract_axis_magnitude_units(
    payload: dict,
    settings: FixationROIVsPeriodFactorialPlotSettings,
) -> pd.DataFrame:
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    mode = _normalize_mode(meta.get("axis_comparison_mode", settings.axis_comparison_mode))
    collapsed_df = _extract_collapsed_axis_units(
        payload,
        settings,
        mode=mode,
        require_signed=False,
    )
    if not collapsed_df.empty:
        return collapsed_df

    unit_axis_df = _as_df(payload.get("unit_axis_values"))
    if unit_axis_df.empty:
        return pd.DataFrame()
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

    # Legacy fallback for averaged_across_windows / max_abs_across_windows payloads.
    group_cols = [col for col in ("unit_key", "region", "axis_name", "axis_source") if col in df.columns]
    df["window_name"] = df["window_name"].astype(str)
    sig_window_rank = {str(name): idx for idx, name in enumerate(sig_windows)}
    rows: list[dict] = []
    for key_vals, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = {col: val for col, val in zip(group_cols, key_vals)}
        vals = grp["value_abs_norm"].to_numpy(dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if mode == "max_abs_across_windows":
            grp = grp.sort_values(
                by=["value_abs_norm", "window_name"],
                ascending=[False, True],
                kind="mergesort",
            ).copy()
            grp["window_rank"] = grp["window_name"].map(
                lambda name: int(sig_window_rank.get(str(name), len(sig_window_rank)))
            )
            grp = grp.sort_values(
                by=["value_abs_norm", "window_rank", "window_name"],
                ascending=[False, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            selected = grp.iloc[0]
            row.update(
                {
                    "axis_comparison_mode": mode,
                    "window_name": "max_abs_across_windows",
                    "value_abs_norm": float(selected["value_abs_norm"]),
                    "selected_window_name": str(selected.get("window_name", "")),
                    "n_windows_used": int(vals.size),
                }
            )
        else:
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


def _extract_axis_signed_units(
    payload: dict,
    settings: FixationROIVsPeriodFactorialPlotSettings,
) -> pd.DataFrame:
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    mode = _normalize_mode(meta.get("axis_comparison_mode", settings.axis_comparison_mode))
    collapsed_df = _extract_collapsed_axis_units(
        payload,
        settings,
        mode=mode,
        require_signed=True,
    )
    if not collapsed_df.empty:
        return collapsed_df

    unit_axis_df = _as_df(payload.get("unit_axis_values"))
    if unit_axis_df.empty:
        return pd.DataFrame()
    sig_windows = [
        str(win)
        for win in meta.get("significance_windows", ("pre_fix", "peri_fix", "post_fix"))
        if str(win).strip()
    ]

    df = unit_axis_df.copy()
    if "axis_source" in df.columns:
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
    df["value_signed"] = pd.to_numeric(df["value_signed"], errors="coerce")
    df = df.loc[df["value_signed"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    if "unit_key" not in df.columns:
        df = df.reset_index().rename(columns={"index": "unit_key"})
    df["unit_key"] = df["unit_key"].astype(str)

    if mode == "split_by_window":
        out = df.copy()
        out["axis_comparison_mode"] = mode
        return out

    sig_window_rank = {str(name): idx for idx, name in enumerate(sig_windows)}
    group_cols = [col for col in ("unit_key", "region", "axis_name", "axis_source") if col in df.columns]
    rows: list[dict] = []
    for key_vals, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = {col: val for col, val in zip(group_cols, key_vals)}
        vals = grp["value_signed"].to_numpy(dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if mode == "max_abs_across_windows":
            grp["value_abs_norm"] = np.abs(grp["value_signed"].to_numpy(dtype=float))
            grp["window_rank"] = grp["window_name"].map(
                lambda name: int(sig_window_rank.get(str(name), len(sig_window_rank)))
            )
            grp = grp.sort_values(
                by=["value_abs_norm", "window_rank", "window_name"],
                ascending=[False, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            selected = grp.iloc[0]
            row.update(
                {
                    "axis_comparison_mode": mode,
                    "window_name": "max_abs_across_windows",
                    "value_signed": float(selected["value_signed"]),
                    "selected_window_name": str(selected.get("window_name", "")),
                    "n_windows_used": int(vals.size),
                }
            )
        else:
            row.update(
                {
                    "axis_comparison_mode": mode,
                    "window_name": "avg_pre_peri_post",
                    "value_signed": float(np.mean(vals)),
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
    """Plot separate within-region and cross-region axis violins."""
    _apply_plotting_style(settings)
    payload, _ = _load_result_payload(settings)
    mag_df = _extract_axis_magnitude_units(payload, settings)
    within_df = _extract_within_axis_df(payload, settings)
    cross_df = _extract_cross_region_axis_df(payload, settings)
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

        plot_df = win_df.loc[
            win_df["region"].astype(str).isin(set(region_tokens))
            & win_df["axis_name"].astype(str).isin(set(axis_tokens))
        ].copy()
        if plot_df.empty:
            continue
        plot_df["region"] = plot_df["region"].astype(str)
        plot_df["axis_name"] = plot_df["axis_name"].astype(str)
        y_vals = pd.to_numeric(plot_df["value_abs_norm"], errors="coerce").to_numpy(dtype=float).reshape(-1)
        y_vals = y_vals[np.isfinite(y_vals)]
        y_max = float(np.nanmax(y_vals)) if y_vals.size > 0 else 1.0
        y_min = float(np.nanmin(y_vals)) if y_vals.size > 0 else 0.0
        span = max(y_max - y_min, 1e-6)
        step = 0.10 * span
        bar_h = 0.028 * span
        mode = str(payload.get("meta", {}).get("axis_comparison_mode", settings.axis_comparison_mode))
        violin_width = 0.88

        fig_within, ax_within = plt.subplots(
            1,
            1,
            figsize=(float(settings.violin_letter_width_in), float(fig_h)),
            dpi=settings.output_dpi,
        )
        within_palette = {
            axis_name: settings.axis_colors.get(str(axis_name), "#777777")
            for axis_name in axis_tokens
        }
        sns.violinplot(
            ax=ax_within,
            data=plot_df,
            x="region",
            y="value_abs_norm",
            hue="axis_name",
            order=list(region_tokens),
            hue_order=list(axis_tokens),
            palette=within_palette,
            inner="quart",
            cut=0.0,
            linewidth=0.8,
            width=violin_width,
        )
        bodies = [artist for artist in ax_within.collections if isinstance(artist, PolyCollection)]
        for body in bodies:
            body.set_edgecolor("#222222")
            body.set_linewidth(0.65)
            body.set_alpha(0.72)
            body.set_rasterized(False)
        ax_within.set_xticks(np.arange(len(region_tokens)))
        ax_within.set_xticklabels(
            [_display_region(region, settings) for region in region_tokens],
            rotation=0,
            ha="center",
            fontsize=9,
        )
        ax_within.set_ylabel("|Axis Magnitude| (a.u.)", fontsize=9)
        ax_within.set_xlabel("")
        ax_within.grid(axis="y", alpha=0.23, linewidth=0.6)
        handles, labels = ax_within.get_legend_handles_labels()
        if handles:
            ax_within.legend(
                handles[: len(axis_tokens)],
                [_display_axis(label, settings) for label in labels[: len(axis_tokens)]],
                title="Axis",
                frameon=False,
                ncol=max(1, len(axis_tokens)),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                fontsize=8,
                title_fontsize=8,
            )
        dodge_step = float(violin_width) / float(max(len(axis_tokens), 1))
        pos_map_within = {
            (str(region), str(axis_name)): (
                float(ridx) - float(violin_width) / 2.0 + (float(aidx) + 0.5) * dodge_step
            )
            for ridx, region in enumerate(region_tokens)
            for aidx, axis_name in enumerate(axis_tokens)
        }
        within_rows = (
            within_df.loc[
                (within_df["window_name"].astype(str) == str(win_name))
                & (within_df["region"].astype(str).isin(set(region_tokens)))
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
            axis_a = str(getattr(row, "axis_a", ""))
            axis_b = str(getattr(row, "axis_b", ""))
            if (region, axis_a) not in pos_map_within or (region, axis_b) not in pos_map_within:
                continue
            stars = _significance_star(float(getattr(row, "p_value_adjusted", np.nan)))
            if not stars:
                continue
            level = int(within_counts.get(region, 0))
            y = y_max + 0.16 * span + step * float(level)
            x1 = float(pos_map_within[(region, axis_a)])
            x2 = float(pos_map_within[(region, axis_b)])
            ax_within.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], color="#222222", linewidth=0.8)
            ax_within.text((x1 + x2) / 2.0, y + bar_h + 0.01 * span, stars, ha="center", va="bottom", fontsize=9)
            within_counts[region] = level + 1
            max_within_stack = max(max_within_stack, level + 1)
        within_top_extra = max(0.22 * span, 0.16 * span + step * float(max_within_stack) + 0.18 * span)
        ax_within.set_ylim(y_min - 0.05 * span, y_max + within_top_extra)
        fig_within.suptitle(
            (
                "ROI-vs-Period Within-Region Axis Comparison "
                f"({win_name}; mode={mode}; source={settings.axis_magnitude_source})"
            ),
            fontsize=10,
        )
        fig_within.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.18)

        suffix = f"__window={_safe_suffix_token(win_name)}"
        out_name_within = ensure_filename(f"{output_filename}__within_region{suffix}", f".{ext}")
        out_path_within = out_root / out_name_within
        save_figure(fig_within, out_path_within, ext=ext, dpi=settings.output_dpi)
        plt.close(fig_within)
        outputs.append(
            {
                "output_path": str(out_path_within),
                "window_name": str(win_name),
                "kind": "within_region",
                "regions": list(region_tokens),
                "axes": list(axis_tokens),
            }
        )

        fig_cross, ax_cross = plt.subplots(
            1,
            1,
            figsize=(float(settings.violin_letter_width_in), float(fig_h)),
            dpi=settings.output_dpi,
        )
        cross_palette = {
            region: settings.region_outline_colors.get(str(region), "#777777")
            for region in region_tokens
        }
        sns.violinplot(
            ax=ax_cross,
            data=plot_df,
            x="axis_name",
            y="value_abs_norm",
            hue="region",
            order=list(axis_tokens),
            hue_order=list(region_tokens),
            palette=cross_palette,
            inner="quart",
            cut=0.0,
            linewidth=0.8,
            width=violin_width,
        )
        bodies = [artist for artist in ax_cross.collections if isinstance(artist, PolyCollection)]
        for body in bodies:
            body.set_edgecolor("#222222")
            body.set_linewidth(0.65)
            body.set_alpha(0.70)
            body.set_rasterized(False)
        ax_cross.set_xticks(np.arange(len(axis_tokens)))
        ax_cross.set_xticklabels(
            [_display_axis(axis_name, settings) for axis_name in axis_tokens],
            rotation=0,
            ha="center",
            fontsize=9,
        )
        ax_cross.set_ylabel("|Axis Magnitude| (a.u.)", fontsize=9)
        ax_cross.set_xlabel("")
        ax_cross.grid(axis="y", alpha=0.23, linewidth=0.6)
        handles, labels = ax_cross.get_legend_handles_labels()
        if handles:
            ax_cross.legend(
                handles[: len(region_tokens)],
                [_display_region(label, settings) for label in labels[: len(region_tokens)]],
                title="Region",
                frameon=False,
                ncol=max(1, len(region_tokens)),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                fontsize=8,
                title_fontsize=8,
            )
        dodge_step = float(violin_width) / float(max(len(region_tokens), 1))
        pos_map_cross = {
            (str(axis_name), str(region)): (
                float(aidx) - float(violin_width) / 2.0 + (float(ridx) + 0.5) * dodge_step
            )
            for aidx, axis_name in enumerate(axis_tokens)
            for ridx, region in enumerate(region_tokens)
        }
        cross_rows = (
            cross_df.loc[
                (cross_df["window_name"].astype(str) == str(win_name))
                & (cross_df["axis_name"].astype(str).isin(set(axis_tokens)))
                & (cross_df["significant_adjusted"].map(bool))
            ].copy()
            if not cross_df.empty
            else pd.DataFrame()
        )
        if not cross_rows.empty and "p_value_adjusted" in cross_rows.columns:
            cross_rows = cross_rows.sort_values(["axis_name", "p_value_adjusted"], na_position="last")
        cross_counts: dict[str, int] = {}
        max_cross_stack = 0
        for row in cross_rows.itertuples(index=False):
            region_a = str(getattr(row, "region_a", ""))
            region_b = str(getattr(row, "region_b", ""))
            axis_name = str(getattr(row, "axis_name", ""))
            if (axis_name, region_a) not in pos_map_cross or (axis_name, region_b) not in pos_map_cross:
                continue
            stars = _significance_star(float(getattr(row, "p_value_adjusted", np.nan)))
            if not stars:
                continue
            level = int(cross_counts.get(axis_name, 0))
            y = y_max + 0.16 * span + step * float(level)
            x1 = float(pos_map_cross[(axis_name, region_a)])
            x2 = float(pos_map_cross[(axis_name, region_b)])
            bar_color = settings.axis_colors.get(axis_name, "#7f0000")
            ax_cross.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], color=bar_color, linewidth=0.95)
            ax_cross.text((x1 + x2) / 2.0, y + bar_h + 0.01 * span, stars, ha="center", va="bottom", fontsize=9, color=bar_color)
            cross_counts[axis_name] = level + 1
            max_cross_stack = max(max_cross_stack, level + 1)
        cross_top_extra = max(0.22 * span, 0.16 * span + step * float(max_cross_stack) + 0.18 * span)
        ax_cross.set_ylim(y_min - 0.05 * span, y_max + cross_top_extra)
        fig_cross.suptitle(
            (
                "ROI-vs-Period Cross-Region Axis Comparison "
                f"({win_name}; mode={mode}; source={settings.axis_magnitude_source})"
            ),
            fontsize=10,
        )
        fig_cross.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.18)

        out_name_cross = ensure_filename(f"{output_filename}__cross_region{suffix}", f".{ext}")
        out_path_cross = out_root / out_name_cross
        save_figure(fig_cross, out_path_cross, ext=ext, dpi=settings.output_dpi)
        plt.close(fig_cross)
        outputs.append(
            {
                "output_path": str(out_path_cross),
                "window_name": str(win_name),
                "kind": "cross_region",
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


def _axis_direction(axis_name: object) -> tuple[float, float]:
    token = str(axis_name).strip().lower()
    if token == "face_object":
        return (1.0, 0.0)
    if token == "interactive_state":
        return (0.0, 1.0)
    if token == "cross_interaction":
        theta = float(np.deg2rad(32.0))
        return (float(np.cos(theta)), float(np.sin(theta)))
    return (0.0, 0.0)


def _build_axis_space_unit_summary(
    signed_df: pd.DataFrame,
    *,
    settings: FixationROIVsPeriodFactorialPlotSettings,
    region_tokens: Sequence[str],
    axis_tokens: Sequence[str],
    window_name: str,
) -> pd.DataFrame:
    win_df = signed_df.loc[signed_df["window_name"].astype(str) == str(window_name)].copy()
    if win_df.empty:
        return pd.DataFrame()
    needed_cols = {"region", "axis_name", "value_signed", "unit_key"}
    if not needed_cols.issubset(set(win_df.columns)):
        return pd.DataFrame()

    sig_cols = [col for col in ("is_significant_axis", "significant_within_unit", "significant_raw") if col in win_df.columns]
    rows: list[dict] = []
    for region in region_tokens:
        reg_df = win_df.loc[win_df["region"].astype(str) == str(region)].copy()
        if reg_df.empty:
            continue
        for unit_key, unit_grp in reg_df.groupby("unit_key", dropna=False):
            rec: dict[str, object] = {
                "window_name": str(window_name),
                "region": str(region),
                "unit_key": str(unit_key),
            }
            for meta_col in ("date", "unit_uuid", "spike_channel", "recorded_agent", "recorded_monkey", "area"):
                if meta_col in unit_grp.columns:
                    rec[meta_col] = unit_grp.iloc[0].get(meta_col)
            sig_axes: list[str] = []
            vals_abs: list[float] = []
            for axis_name in axis_tokens:
                axis_grp = unit_grp.loc[unit_grp["axis_name"].astype(str) == str(axis_name)].copy()
                if axis_grp.empty:
                    rec[str(axis_name)] = np.nan
                    rec[f"{axis_name}_window"] = np.nan
                    rec[f"{axis_name}_significant"] = False
                    continue
                axis_row = axis_grp.iloc[0]
                value = pd.to_numeric(pd.Series([axis_row.get("value_signed", np.nan)]), errors="coerce").iloc[0]
                axis_sig = False
                for sig_col in sig_cols:
                    raw_flag = axis_row.get(sig_col, False)
                    axis_sig = axis_sig or (bool(raw_flag) if pd.notna(raw_flag) else False)
                rec[str(axis_name)] = float(value) if pd.notna(value) else np.nan
                rec[f"{axis_name}_window"] = axis_row.get("selected_window_name", axis_row.get("window_name", np.nan))
                rec[f"{axis_name}_significant"] = bool(axis_sig)
                if axis_sig:
                    sig_axes.append(str(axis_name))
                if pd.notna(value):
                    vals_abs.append(abs(float(value)))
            needed_axes = ("face_object", "interactive_state", "cross_interaction")
            if not all(np.isfinite(float(rec.get(axis_name, np.nan))) for axis_name in needed_axes):
                continue
            rec["mean_axis_magnitude"] = float(np.mean(vals_abs)) if vals_abs else np.nan
            rec["n_significant_axes"] = int(len(sig_axes))
            rec["significant_axes"] = "|".join(sig_axes)
            rec["is_significant_unit"] = bool(len(sig_axes) > 0)
            rows.append(rec)
    return pd.DataFrame(rows)


def _pairwise_kde_projection(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    lim: float,
    n_grid: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(-float(lim), float(lim), int(max(n_grid, 30)))
    xx, yy = np.meshgrid(grid, grid)
    density = _kde_2d_weighted(
        points_xy=np.column_stack([values_a, values_b]),
        xx=xx,
        yy=yy,
        weights=None,
    )
    if np.any(np.isfinite(density)) and float(np.nanmax(density)) > 0.0:
        density = density / float(np.nanmax(density))
    else:
        density = np.zeros_like(xx, dtype=float)
    return xx, yy, density


def _add_origin_axis_arrows(
    ax: Axes,
    *,
    lim: float,
    axis_x: str,
    axis_y: str,
    settings: FixationROIVsPeriodFactorialPlotSettings,
) -> None:
    arrow_lim = 0.92 * float(lim)
    x_color = settings.axis_colors.get(str(axis_x), "#4d4d4d")
    y_color = settings.axis_colors.get(str(axis_y), "#4d4d4d")
    ax.annotate(
        "",
        xy=(arrow_lim, 0.0),
        xytext=(-arrow_lim, 0.0),
        arrowprops=dict(arrowstyle="<->", color=x_color, linewidth=1.0, alpha=0.80, shrinkA=0.0, shrinkB=0.0),
        zorder=0,
    )
    ax.annotate(
        "",
        xy=(0.0, arrow_lim),
        xytext=(0.0, -arrow_lim),
        arrowprops=dict(arrowstyle="<->", color=y_color, linewidth=1.0, alpha=0.80, shrinkA=0.0, shrinkB=0.0),
        zorder=0,
    )


def _add_origin_axis_arrows_3d(
    ax: Axes3D,
    *,
    lim: float,
    settings: FixationROIVsPeriodFactorialPlotSettings,
) -> None:
    axis_specs = (
        ("face_object", (1.0, 0.0, 0.0)),
        ("interactive_state", (0.0, 1.0, 0.0)),
        ("cross_interaction", (0.0, 0.0, 1.0)),
    )
    arrow_lim = 0.92 * float(lim)
    for axis_name, direction in axis_specs:
        color = settings.axis_colors.get(str(axis_name), "#4d4d4d")
        dx = float(direction[0]) * arrow_lim
        dy = float(direction[1]) * arrow_lim
        dz = float(direction[2]) * arrow_lim
        ax.plot(
            [-dx, dx],
            [-dy, dy],
            [-dz, dz],
            color=color,
            linewidth=1.1,
            alpha=0.55,
        )
        ax.quiver(
            0.0,
            0.0,
            0.0,
            dx,
            dy,
            dz,
            color=color,
            linewidth=1.3,
            alpha=0.92,
            arrow_length_ratio=0.08,
        )
        ax.quiver(
            0.0,
            0.0,
            0.0,
            -dx,
            -dy,
            -dz,
            color=color,
            linewidth=1.0,
            alpha=0.70,
            arrow_length_ratio=0.08,
        )


def _plot_fixation_roi_vs_period_axis_space_true_3d(
    settings: FixationROIVsPeriodFactorialPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    window: Optional[str] = None,
    output_filename_regions: str = "roi_vs_period_axis_space_regions",
    output_filename_contours: str = "roi_vs_period_axis_space_contours",
) -> list[dict]:
    _apply_plotting_style(settings)
    payload, _ = _load_result_payload(settings)
    signed_df = _extract_axis_signed_units(payload, settings)
    if signed_df.empty:
        print("[plot] no signed axis unit rows available for axis-space plot")
        return []

    region_tokens = _ordered_tokens(
        available=signed_df["region"].astype(str).unique().tolist(),
        preferred=[str(tok).lower() for tok in (regions or settings.region_order)],
    )
    axis_tokens = _ordered_tokens(
        available=signed_df["axis_name"].astype(str).unique().tolist(),
        preferred=[str(tok) for tok in settings.axis_order],
    )
    required_axes = ("face_object", "interactive_state", "cross_interaction")
    if not region_tokens or any(axis_name not in axis_tokens for axis_name in required_axes):
        print("[plot] axis-space plot requires face_object, interactive_state, and cross_interaction axes")
        return []

    windows_available = signed_df["window_name"].astype(str).unique().tolist()
    if window is not None:
        window_tokens = [str(window)]
    else:
        window_tokens = _window_order(windows_available)
    if not window_tokens:
        print("[plot] no window rows available for axis-space plot")
        return []

    ext = _resolve_output_ext(settings)
    out_root = _build_output_root(settings)
    outputs: list[dict] = []
    letter_h = 11.0
    region_fig_h = max(float(settings.axis_space_regions_letter_height_frac) * letter_h, 3.2)
    q_high = float(np.clip(settings.axis_space_quantile_high, 0.0, 1.0))

    for win_name in window_tokens:
        unit_df = _build_axis_space_unit_summary(
            signed_df,
            settings=settings,
            region_tokens=region_tokens,
            axis_tokens=axis_tokens,
            window_name=str(win_name),
        )
        if unit_df.empty:
            continue
        for axis_name in required_axes:
            unit_df[axis_name] = pd.to_numeric(unit_df[axis_name], errors="coerce")
        unit_df = unit_df.loc[unit_df.loc[:, list(required_axes)].notna().all(axis=1)].copy()
        if unit_df.empty:
            continue

        mode = str(payload.get("meta", {}).get("axis_comparison_mode", settings.axis_comparison_mode))
        abs_vals = np.abs(unit_df.loc[:, list(required_axes)].to_numpy(dtype=float).reshape(-1))
        abs_vals = abs_vals[np.isfinite(abs_vals)]
        lim = float(np.quantile(abs_vals, q_high)) if abs_vals.size > 0 else 1.0
        lim = max(1.15 * float(lim), 0.25)
        levels = np.linspace(0.18, 0.92, 6)
        sig_color = "#c62828"
        nonsig_color = "#7a7a7a"
        scatter_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=nonsig_color,
                markeredgecolor="#2b2b2b",
                alpha=0.42,
                markersize=6.0,
                label="All other units",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=sig_color,
                markeredgecolor="#2b2b2b",
                alpha=0.90,
                markersize=6.0,
                label="Significant units",
            ),
        ]

        fig = plt.figure(
            figsize=(float(settings.axis_space_regions_letter_width_in), float(region_fig_h)),
            dpi=settings.output_dpi,
        )
        for ridx, region in enumerate(region_tokens):
            ax = fig.add_subplot(1, len(region_tokens), ridx + 1, projection="3d")
            reg_df = unit_df.loc[unit_df["region"].astype(str) == str(region)].copy()
            if reg_df.empty:
                ax.text(0.0, 0.0, 0.0, "No data", ha="center", va="center", fontsize=9)
            else:
                x = reg_df["face_object"].to_numpy(dtype=float)
                y = reg_df["interactive_state"].to_numpy(dtype=float)
                z = reg_df["cross_interaction"].to_numpy(dtype=float)
                sig_mask = reg_df["is_significant_unit"].map(bool).to_numpy(dtype=bool).reshape(-1)
                xx, yy, dens = _pairwise_kde_projection(x, y, lim=float(lim))
                if np.any(np.isfinite(dens)) and float(np.nanmax(dens)) > 0.0:
                    ax.contour(
                        xx,
                        yy,
                        dens,
                        levels=levels,
                        zdir="z",
                        offset=-float(lim),
                        cmap=mpl.colormaps["magma"],
                        linewidths=0.90,
                        alpha=0.72,
                    )
                ax.scatter(
                    x[~sig_mask],
                    y[~sig_mask],
                    z[~sig_mask],
                    s=16.0,
                    c=nonsig_color,
                    edgecolors="#2b2b2b",
                    linewidths=0.2,
                    alpha=0.34,
                    depthshade=False,
                )
                if np.any(sig_mask):
                    ax.scatter(
                        x[sig_mask],
                        y[sig_mask],
                        z[sig_mask],
                        s=24.0,
                        c=sig_color,
                        edgecolors="#2b2b2b",
                        linewidths=0.3,
                        alpha=0.92,
                        depthshade=False,
                    )
            _add_origin_axis_arrows_3d(ax, lim=float(lim), settings=settings)
            ax.set_xlim(-float(lim), float(lim))
            ax.set_ylim(-float(lim), float(lim))
            ax.set_zlim(-float(lim), float(lim))
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.view_init(elev=22.0, azim=38.0)
            ax.set_xlabel("Face-Object", fontsize=8, labelpad=2.0)
            ax.set_ylabel("Interactive State", fontsize=8, labelpad=2.0)
            ax.set_zlabel("Cross Interaction", fontsize=8, labelpad=3.0)
            ax.tick_params(labelsize=6, pad=0)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.08))
            ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.08))
            ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.08))
            ax.xaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.28)
            ax.yaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.28)
            ax.zaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.28)
            n_units_region = int(reg_df.shape[0]) if not reg_df.empty else 0
            n_sig_region = int(reg_df["is_significant_unit"].sum()) if not reg_df.empty else 0
            ax.set_title(
                f"{_display_region(region, settings)}\nN={n_units_region}, sig={n_sig_region}",
                fontsize=9,
                pad=8.0,
            )

        fig.suptitle(
            (
                "ROI-vs-Period 3D Unit Distributions "
                f"({win_name}; mode={mode}; source={settings.axis_magnitude_source})"
            ),
            fontsize=10,
        )
        fig.legend(
            handles=scatter_handles,
            loc="upper center",
            frameon=False,
            fontsize=8,
            ncol=2,
            bbox_to_anchor=(0.5, 0.98),
        )
        fig.subplots_adjust(left=0.03, right=0.99, top=0.83, bottom=0.06, wspace=0.18)

        suffix = f"__window={_safe_suffix_token(win_name)}"
        out_name_regions = ensure_filename(f"{output_filename_regions}{suffix}", f".{ext}")
        out_path_regions = out_root / out_name_regions
        save_figure(fig, out_path_regions, ext=ext, dpi=settings.output_dpi)
        plt.close(fig)
        outputs.append(
            {
                "output_path": str(out_path_regions),
                "window_name": str(win_name),
                "kind": "regions",
                "regions": list(region_tokens),
                "axes": list(axis_tokens),
            }
        )
    return outputs


def _fit_radial_density_profile(
    magnitudes: np.ndarray,
    *,
    q_low: float,
    q_high: float,
) -> dict:
    vals = np.asarray(magnitudes, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "mean_mag": 0.0,
            "q_low_mag": 0.0,
            "q_high_mag": 0.0,
            "sigma_in": 0.10,
            "sigma_out": 0.10,
            "support_radius": 0.35,
        }
    mean_mag = float(np.mean(vals))
    lo = float(np.quantile(vals, float(np.clip(q_low, 0.0, 1.0))))
    hi = float(np.quantile(vals, float(np.clip(q_high, 0.0, 1.0))))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    base_sigma = max(std, 0.06 * max(mean_mag, hi, 1.0), 1e-3)
    sigma_in = max((mean_mag - lo) / 1.96 if mean_mag > lo else 0.0, 0.35 * base_sigma, 1e-3)
    sigma_out = max((hi - mean_mag) / 1.96 if hi > mean_mag else 0.0, 0.35 * base_sigma, 1e-3)
    support_radius = max(hi, mean_mag + 3.2 * sigma_out, 1e-3)
    return {
        "mean_mag": float(mean_mag),
        "q_low_mag": float(lo),
        "q_high_mag": float(hi),
        "sigma_in": float(sigma_in),
        "sigma_out": float(sigma_out),
        "support_radius": float(support_radius),
    }


def _radial_density_surface(
    *,
    r: np.ndarray,
    mean_mag: float,
    sigma_in: float,
    sigma_out: float,
) -> np.ndarray:
    rr = np.asarray(r, dtype=float)
    z = np.zeros_like(rr, dtype=float)
    s_in = max(float(sigma_in), 1e-6)
    s_out = max(float(sigma_out), 1e-6)
    mask = rr <= float(mean_mag)
    z[mask] = np.exp(-0.5 * ((float(mean_mag) - rr[mask]) / s_in) ** 2)
    z[~mask] = np.exp(-0.5 * ((rr[~mask] - float(mean_mag)) / s_out) ** 2)
    z = np.clip(z, 0.0, 1.0)
    return z


def _kde_2d_weighted(
    *,
    points_xy: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros_like(xx, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] == 0:
        return np.zeros_like(xx, dtype=float)

    n = int(pts.shape[0])
    if weights is None:
        w = np.full(n, 1.0 / float(n), dtype=float)
    else:
        ww = np.asarray(weights, dtype=float).reshape(-1)
        if ww.size != n:
            ww = np.full(n, 1.0, dtype=float)
        ww[~np.isfinite(ww)] = 0.0
        ww = np.clip(ww, 0.0, None)
        if float(np.sum(ww)) <= 0.0:
            ww = np.full(n, 1.0, dtype=float)
        w = ww / float(np.sum(ww))

    std_x = float(np.std(pts[:, 0], ddof=1)) if n > 1 else 0.0
    std_y = float(np.std(pts[:, 1], ddof=1)) if n > 1 else 0.0
    span_x = float(np.ptp(pts[:, 0])) if n > 1 else abs(float(pts[0, 0]))
    span_y = float(np.ptp(pts[:, 1])) if n > 1 else abs(float(pts[0, 1]))
    scott = float(np.power(max(n, 2), -1.0 / 6.0))
    hx = max(std_x * scott, 0.10 * max(span_x, 1e-3), 1e-3)
    hy = max(std_y * scott, 0.10 * max(span_y, 1e-3), 1e-3)

    gx = xx.reshape(1, -1)
    gy = yy.reshape(1, -1)
    dx = (gx - pts[:, [0]]) / hx
    dy = (gy - pts[:, [1]]) / hy
    exponent = -0.5 * (dx * dx + dy * dy)
    kernel = np.exp(exponent)
    norm = 1.0 / (2.0 * np.pi * hx * hy)
    dens = norm * np.sum(w[:, None] * kernel, axis=0)
    dens = dens.reshape(xx.shape)
    dens = np.clip(dens, 0.0, None)
    return dens


def _density_threshold_for_mass(
    density: np.ndarray,
    *,
    cell_area: float,
    mass_fraction: float = 0.95,
) -> float:
    d = np.asarray(density, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d > 0.0]
    if d.size == 0:
        return 0.0
    frac = float(np.clip(mass_fraction, 0.0, 1.0))
    if frac <= 0.0:
        return float(np.max(d))
    if frac >= 1.0:
        return 0.0
    d_sorted = np.sort(d)[::-1]
    masses = d_sorted * float(cell_area)
    cum = np.cumsum(masses)
    total = float(cum[-1]) if cum.size > 0 else 0.0
    if total <= 0.0:
        return 0.0
    target = frac * total
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), d_sorted.size - 1)
    return float(d_sorted[idx])


def _surface_95_mask(
    *,
    r: np.ndarray,
    q_low_mag: float,
    q_high_mag: float,
) -> np.ndarray:
    lo = max(float(q_low_mag), 0.0)
    hi = max(float(q_high_mag), lo + 1e-9)
    rr = np.asarray(r, dtype=float)
    return (rr >= lo) & (rr <= hi)


def _render_pyvista_density_surface_image(
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    z_fill: np.ndarray,
    z_mask: np.ndarray,
    lim: float,
    z_axis_max: float,
    z_scale: float,
) -> Optional[np.ndarray]:
    if pv is None:
        return None
    global _PYVISTA_RUNTIME_READY
    try:
        if not _PYVISTA_RUNTIME_READY:
            os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
            os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
            os.environ.setdefault("PYVISTA_AUTO_CLOSE", "true")
            pv.OFF_SCREEN = True
            if not os.environ.get("DISPLAY"):
                try:
                    import vtk  # type: ignore

                    vtk.vtkObject.GlobalWarningDisplayOff()
                except Exception:
                    pass
            _PYVISTA_RUNTIME_READY = True

        pv.OFF_SCREEN = True
        z_scale = float(z_scale)
        z_surface = np.where(
            np.asarray(z_mask, dtype=bool),
            np.asarray(z_fill, dtype=float) * float(z_scale),
            np.nan,
        )
        grid = pv.StructuredGrid(
            np.asarray(xx, dtype=float),
            np.asarray(yy, dtype=float),
            np.asarray(z_surface, dtype=float),
        )
        grid["support"] = np.where(np.asarray(z_mask, dtype=bool), 1.0, 0.0).ravel(order="F")
        surface = grid.extract_surface(algorithm="dataset_surface")
        surface = surface.threshold(value=0.5, scalars="support")
        if surface.n_points == 0:
            return None

        scene_scale = max(float(lim), float(z_axis_max), 1.0)
        plotter = pv.Plotter(off_screen=True, window_size=(980, 760))
        plotter.set_background("white")
        plotter.remove_all_lights()
        plotter.add_light(
            pv.Light(
                position=(2.3 * float(lim), -2.5 * float(lim), 2.8),
                focal_point=(0.0, 0.0, 0.35),
                color="white",
                intensity=1.35,
            )
        )
        plotter.add_light(
            pv.Light(
                position=(-1.8 * float(lim), 2.1 * float(lim), 1.6),
                focal_point=(0.0, 0.0, 0.3),
                color="white",
                intensity=0.42,
            )
        )
        plotter.add_light(
            pv.Light(
                position=(0.0, -2.6 * float(lim), 3.1),
                focal_point=(0.0, 0.0, 0.4),
                color="white",
                intensity=0.28,
            )
        )
        bounds = (
            -float(lim),
            float(lim),
            -float(lim),
            float(lim),
            0.0,
            float(max(z_axis_max, 1e-6)),
        )
        plotter.add_mesh(
            surface,
            color="#b6b6b6",
            smooth_shading=True,
            ambient=0.22,
            diffuse=0.74,
            specular=0.55,
            specular_power=22.0,
            opacity=1.0,
            show_edges=False,
            lighting=True,
        )
        # Keep subplot axis scaling consistent regardless of each region's support mask.
        plotter.add_mesh(
            pv.Box(bounds=bounds),
            opacity=0.0,
            lighting=False,
            show_edges=False,
            pickable=False,
        )
        plotter.show_bounds(
            grid=False,
            location="outer",
            ticks="outside",
            xtitle="Face-Object",
            ytitle="Interactive-Noninteractive",
            ztitle="Distribution Magnitude",
            font_size=15,
            minor_ticks=False,
            fmt="%.2f",
        )
        z_focus = 0.32 * float(max(z_axis_max, 1e-6))
        plotter.camera.position = (3.45 * scene_scale, -3.05 * scene_scale, 2.95 * scene_scale)
        plotter.camera.focal_point = (0.0, 0.0, z_focus)
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.camera.zoom(0.86)
        img = plotter.screenshot(return_img=True)
        plotter.close()
        if isinstance(img, np.ndarray) and img.size > 0:
            return img
    except Exception as exc:
        print(f"[plot] pyvista off-screen image render failed: {exc}")
        return None
    return None


def _save_pyvista_region_surface_figure(
    *,
    out_path: Path,
    region_tokens: Sequence[str],
    region_profiles: dict[str, dict],
    density_maps: dict[str, dict[str, np.ndarray]],
    xx: np.ndarray,
    yy: np.ndarray,
    lim: float,
    z_axis_max: float,
    settings: FixationROIVsPeriodFactorialPlotSettings,
) -> bool:
    if pv is None:
        return False
    global _PYVISTA_RUNTIME_READY
    try:
        if not _PYVISTA_RUNTIME_READY:
            os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
            os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
            os.environ.setdefault("PYVISTA_AUTO_CLOSE", "true")
            pv.OFF_SCREEN = True
            if not os.environ.get("DISPLAY"):
                try:
                    import vtk  # type: ignore

                    vtk.vtkObject.GlobalWarningDisplayOff()
                except Exception:
                    pass
            _PYVISTA_RUNTIME_READY = True

        n_panels = max(1, len(region_tokens))
        plotter = pv.Plotter(
            off_screen=True,
            shape=(1, n_panels),
            border=False,
            window_size=(int(600 * n_panels), 640),
        )
        plotter.remove_all_lights()
        scene_scale = max(float(lim), float(z_axis_max), 1.0)
        key_light = pv.Light(
            position=(2.5 * scene_scale, -2.0 * scene_scale, 2.8 * scene_scale),
            focal_point=(0.0, 0.0, 0.5),
            color="white",
            intensity=1.25,
        )
        fill_light = pv.Light(
            position=(-2.2 * scene_scale, 1.9 * scene_scale, 1.8 * scene_scale),
            focal_point=(0.0, 0.0, 0.45),
            color="white",
            intensity=0.45,
        )
        rim_light = pv.Light(
            position=(0.0, -2.8 * scene_scale, 3.2 * scene_scale),
            focal_point=(0.0, 0.0, 0.55),
            color="white",
            intensity=0.32,
        )
        plotter.add_light(key_light)
        plotter.add_light(fill_light)
        plotter.add_light(rim_light)

        for ridx, region in enumerate(region_tokens):
            plotter.subplot(0, ridx)
            plotter.set_background("white")
            profile = region_profiles.get(str(region))
            dmap = density_maps.get(str(region))
            if profile is None or dmap is None:
                plotter.add_text("No data", position="upper_left", font_size=16, color="black")
                continue

            z_scale = 1.70
            z_surface = np.where(dmap["z_mask"], dmap["z_fill"] * float(z_scale), np.nan)
            grid = pv.StructuredGrid(
                np.asarray(xx, dtype=float),
                np.asarray(yy, dtype=float),
                np.asarray(z_surface, dtype=float),
            )
            grid["support"] = np.where(np.asarray(dmap["z_mask"], dtype=bool), 1.0, 0.0).ravel(order="F")
            surface = grid.extract_surface(algorithm="dataset_surface")
            surface = surface.threshold(value=0.5, scalars="support")
            if surface.n_points == 0:
                plotter.add_text("No support", position="upper_left", font_size=16, color="black")
                continue

            bounds = (
                -float(lim),
                float(lim),
                -float(lim),
                float(lim),
                0.0,
                float(max(z_axis_max, 1e-6)),
            )
            plotter.add_mesh(
                surface,
                color="#b6b6b6",
                smooth_shading=True,
                ambient=0.24,
                diffuse=0.73,
                specular=0.60,
                specular_power=24.0,
                opacity=1.0,
                show_edges=False,
                lighting=True,
            )
            # Add a transparent reference box so all subplots keep identical xyz bounds.
            plotter.add_mesh(
                pv.Box(bounds=bounds),
                opacity=0.0,
                lighting=False,
                show_edges=False,
                pickable=False,
            )
            plotter.show_bounds(
                grid=False,
                location="outer",
                ticks="outside",
                xtitle="Face-Object",
                ytitle="Interactive-Noninteractive",
                ztitle="Distribution Magnitude",
                font_size=18,
                color="black",
                n_xlabels=3,
                n_ylabels=3,
                n_zlabels=3,
                fmt="%.2f",
            )
            plotter.add_text(_display_region(region, settings), position="upper_left", font_size=18, color="black")
            plotter.add_text(
                f"n={int(profile.get('n_units', 0))}",
                position=(0.02, 0.90),
                viewport=True,
                font_size=14,
                color="black",
            )
            z_focus = 0.35 * float(max(z_axis_max, 1e-6))
            plotter.camera.position = (3.45 * scene_scale, -3.05 * scene_scale, 3.10 * scene_scale)
            plotter.camera.focal_point = (0.0, 0.0, z_focus)
            plotter.camera.up = (0.0, 0.0, 1.0)
            plotter.camera.zoom(0.84)

        suffix = out_path.suffix.lower()
        if suffix in _VECTOR_EXTS:
            try:
                plotter.save_graphic(str(out_path), raster=False)
            except TypeError:
                plotter.save_graphic(str(out_path))
        else:
            plotter.screenshot(str(out_path), return_img=False)
        plotter.close()
        return True
    except Exception as exc:
        print(f"[plot] pyvista multi-panel surface export failed: {exc}")
        return False


def plot_fixation_roi_vs_period_axis_space(
    settings: FixationROIVsPeriodFactorialPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
    window: Optional[str] = None,
    output_filename_regions: str = "roi_vs_period_axis_space_regions",
    output_filename_contours: str = "roi_vs_period_axis_space_contours",
) -> list[dict]:
    """Plot true 3D unit-coordinate regional distributions with density contours."""
    return _plot_fixation_roi_vs_period_axis_space_true_3d(
        settings,
        regions=regions,
        window=window,
        output_filename_regions=output_filename_regions,
        output_filename_contours=output_filename_contours,
    )
    _apply_plotting_style(settings)
    payload, _ = _load_result_payload(settings)
    signed_df = _extract_axis_signed_units(payload, settings)
    if signed_df.empty:
        print("[plot] no signed axis unit rows available for axis-space plot")
        return []

    region_tokens = _ordered_tokens(
        available=signed_df["region"].astype(str).unique().tolist(),
        preferred=[str(tok).lower() for tok in (regions or settings.region_order)],
    )
    axis_tokens = _ordered_tokens(
        available=signed_df["axis_name"].astype(str).unique().tolist(),
        preferred=[str(tok) for tok in settings.axis_order],
    )
    if not region_tokens or not axis_tokens:
        print("[plot] unable to resolve regions/axes for axis-space plot")
        return []
    if "face_object" not in axis_tokens or "interactive_state" not in axis_tokens:
        print("[plot] axis-space plot requires face_object and interactive_state axes")
        return []

    windows_available = signed_df["window_name"].astype(str).unique().tolist()
    if window is not None:
        window_tokens = [str(window)]
    else:
        window_tokens = _window_order(windows_available)
    if not window_tokens:
        print("[plot] no window rows available for axis-space plot")
        return []

    ext = _resolve_output_ext(settings)
    out_root = _build_output_root(settings)
    outputs: list[dict] = []
    letter_h = 11.0
    region_fig_h = max(float(settings.axis_space_regions_letter_height_frac) * letter_h, 2.8)

    for win_name in window_tokens:
        unit_df = _build_axis_space_unit_summary(
            signed_df,
            settings=settings,
            region_tokens=region_tokens,
            axis_tokens=axis_tokens,
            window_name=str(win_name),
        )
        if unit_df.empty:
            continue

        region_profiles: dict[str, dict] = {}
        global_lim = 0.0
        q_low = float(np.clip(settings.axis_space_quantile_low, 0.0, 1.0))
        q_high = float(np.clip(settings.axis_space_quantile_high, 0.0, 1.0))
        if q_high < q_low:
            q_low, q_high = q_high, q_low
        for region in region_tokens:
            reg_df = unit_df.loc[unit_df["region"].astype(str) == str(region)].copy()
            if reg_df.empty:
                continue
            px = pd.to_numeric(reg_df.get("point_x"), errors="coerce").to_numpy(dtype=float).reshape(-1)
            py = pd.to_numeric(reg_df.get("point_y"), errors="coerce").to_numpy(dtype=float).reshape(-1)
            pu = pd.to_numeric(reg_df.get("mean_axis_magnitude"), errors="coerce").to_numpy(dtype=float).reshape(-1)
            valid = np.isfinite(px) & np.isfinite(py) & np.isfinite(pu)
            px = px[valid]
            py = py[valid]
            pu = pu[valid]
            if px.size == 0:
                continue

            axis_means: dict[str, float] = {}
            for axis_name in axis_tokens:
                col = f"axis_mean__{axis_name}"
                if col in reg_df.columns:
                    vv = pd.to_numeric(reg_df[col], errors="coerce").to_numpy(dtype=float).reshape(-1)
                    vv = vv[np.isfinite(vv)]
                    axis_means[str(axis_name)] = float(np.mean(vv)) if vv.size > 0 else 0.0
                else:
                    axis_means[str(axis_name)] = 0.0

            radii = np.sqrt(px * px + py * py)
            qlo = float(np.quantile(radii, q_low))
            qhi = float(np.quantile(radii, q_high))
            profile = {
                "points_x": px,
                "points_y": py,
                "weights": pu,
                "q_low_mag": qlo,
                "q_high_mag": qhi,
                "axis_means": axis_means,
                "n_units": int(px.size),
            }
            region_profiles[str(region)] = profile

            ux, uy = _axis_direction("cross_interaction")
            cross_val = float(axis_means.get("cross_interaction", 0.0))
            global_lim = max(
                global_lim,
                float(np.quantile(np.abs(px), q_high)),
                float(np.quantile(np.abs(py), q_high)),
                abs(float(axis_means.get("face_object", 0.0))),
                abs(float(axis_means.get("interactive_state", 0.0))),
                abs(cross_val * ux),
                abs(cross_val * uy),
                qhi,
            )
        if not region_profiles:
            continue
        lim = max(1.15 * float(global_lim), 0.25)
        n_grid = 180
        x = np.linspace(-float(lim), float(lim), n_grid)
        y = np.linspace(-float(lim), float(lim), n_grid)
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx * xx + yy * yy)
        cell_area = float((x[1] - x[0]) * (y[1] - y[0])) if n_grid > 1 else 1.0
        mode = str(payload.get("meta", {}).get("axis_comparison_mode", settings.axis_comparison_mode))
        contour_cmap = mpl.colormaps["magma"]
        density_maps: dict[str, dict[str, np.ndarray]] = {}
        for region in region_tokens:
            profile = region_profiles.get(str(region))
            if profile is None:
                continue
            density = _kde_2d_weighted(
                points_xy=np.column_stack([profile["points_x"], profile["points_y"]]),
                xx=xx,
                yy=yy,
                weights=np.asarray(profile["weights"], dtype=float),
            )
            threshold = _density_threshold_for_mass(
                density,
                cell_area=cell_area,
                mass_fraction=0.95,
            )
            z = density.copy()
            z[density < threshold] = np.nan
            radial_mask = _surface_95_mask(
                r=rr,
                q_low_mag=float(profile["q_low_mag"]),
                q_high_mag=float(profile["q_high_mag"]),
            )
            z[~radial_mask] = np.nan
            if np.any(np.isfinite(z)):
                z_norm = z / float(np.nanmax(z))
            else:
                z_norm = np.full_like(z, np.nan, dtype=float)
            z_fill = np.nan_to_num(z_norm, nan=0.0)
            z_mask = np.isfinite(z_norm)
            density_maps[str(region)] = {
                "z_norm": z_norm,
                "z_fill": z_fill,
                "z_mask": z_mask,
            }

        suffix = f"__window={_safe_suffix_token(win_name)}"
        out_name_regions = ensure_filename(f"{output_filename_regions}{suffix}", f".{ext}")
        out_path_regions = out_root / out_name_regions
        surface_z_scale = 1.70
        z_norm_max = 0.0
        for dmap in density_maps.values():
            z_vals = np.asarray(dmap.get("z_fill"), dtype=float)
            if z_vals.size > 0 and np.any(np.isfinite(z_vals)):
                z_norm_max = max(z_norm_max, float(np.nanmax(z_vals)))
        z_axis_max = max(float(surface_z_scale * max(z_norm_max, 1e-3)), 1e-3)
        saved_surface = _save_pyvista_region_surface_figure(
            out_path=out_path_regions,
            region_tokens=region_tokens,
            region_profiles=region_profiles,
            density_maps=density_maps,
            xx=xx,
            yy=yy,
            lim=float(lim),
            z_axis_max=float(z_axis_max),
            settings=settings,
        )
        if not saved_surface:
            fig = plt.figure(
                figsize=(float(settings.axis_space_regions_letter_width_in), float(region_fig_h)),
                dpi=settings.output_dpi,
            )
            axes_surface = fig.subplots(1, len(region_tokens), squeeze=False).ravel()
            any_surface = False
            for ridx, region in enumerate(region_tokens):
                axs = axes_surface[ridx]
                axs.set_title(_display_region(region, settings), fontsize=10)
                profile = region_profiles.get(str(region))
                dmap = density_maps.get(str(region))
                if profile is None or dmap is None:
                    axs.text(0.5, 0.5, "No data", transform=axs.transAxes, ha="center", va="center", fontsize=9)
                    axs.axis("off")
                    continue
                img = _render_pyvista_density_surface_image(
                    xx=xx,
                    yy=yy,
                    z_fill=dmap["z_fill"],
                    z_mask=dmap["z_mask"],
                    lim=float(lim),
                    z_axis_max=float(z_axis_max),
                    z_scale=float(surface_z_scale),
                )
                if isinstance(img, np.ndarray) and img.size > 0:
                    axs.imshow(img)
                    axs.axis("off")
                    any_surface = True
                else:
                    axs.text(0.5, 0.5, "PyVista render failed", transform=axs.transAxes, ha="center", va="center", fontsize=8)
                    axs.axis("off")
                axs.text(
                    0.02,
                    0.95,
                    f"n={int(profile.get('n_units', 0))}",
                    transform=axs.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="#333333",
                )
            if any_surface:
                fig.suptitle(
                    (
                        "ROI-vs-Period 3D Density Surface by Region (PyVista) "
                        f"({win_name}; mode={mode}; source={settings.axis_magnitude_source})"
                    ),
                    fontsize=10,
                )
                fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.05, wspace=0.01)
                save_figure(fig, out_path_regions, ext=ext, dpi=settings.output_dpi)
                saved_surface = True
            plt.close(fig)
        if saved_surface:
            outputs.append(
                {
                    "output_path": str(out_path_regions),
                    "window_name": str(win_name),
                    "kind": "regions",
                    "regions": list(region_tokens),
                    "axes": list(axis_tokens),
                }
            )
        else:
            print(f"[plot] pyvista rendering unavailable for window={win_name}; skipping 3D surface output")

        fig2, axes2 = plt.subplots(
            1,
            len(region_tokens),
            figsize=(float(settings.axis_space_regions_letter_width_in), max(2.6, float(region_fig_h) * 0.72)),
            dpi=settings.output_dpi,
            squeeze=False,
        )
        axes2 = axes2.ravel()
        contour_handle = None
        contour_axis_lim = 4.0
        axis_handles = [
            Line2D(
                [0],
                [0],
                color=settings.axis_colors.get(axis_name, "#4d4d4d"),
                lw=2.0,
                label=_display_axis(axis_name, settings),
            )
            for axis_name in axis_tokens
        ]
        ux_cross, uy_cross = _axis_direction("cross_interaction")
        for ridx, region in enumerate(region_tokens):
            ax2 = axes2[ridx]
            profile = region_profiles.get(str(region))
            dmap = density_maps.get(str(region))
            if profile is None or dmap is None:
                ax2.text(0.5, 0.5, "No data", transform=ax2.transAxes, ha="center", va="center", fontsize=9)
            else:
                z_fill = dmap["z_fill"]
                levels = np.linspace(0.20, 0.95, 9)
                contour_handle = ax2.contourf(
                    xx,
                    yy,
                    z_fill,
                    levels=levels,
                    cmap=contour_cmap,
                    alpha=0.95,
                    antialiased=True,
                    zorder=1,
                )
                ax2.contour(
                    xx,
                    yy,
                    z_fill,
                    levels=levels,
                    colors="#252525",
                    linewidths=0.35,
                    alpha=0.45,
                    zorder=2,
                )
                ax2.text(
                    0.02,
                    0.95,
                    f"n={int(profile.get('n_units', 0))}",
                    transform=ax2.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="#333333",
                )
                inset = ax2.inset_axes([0.66, 0.06, 0.31, 0.31])
                inset.axhline(0.0, color="#888888", linewidth=0.8, zorder=1)
                inset.axvline(0.0, color="#888888", linewidth=0.8, zorder=1)
                axis_means = profile.get("axis_means", {})
                max_mag = 1e-3
                for axis_name in axis_tokens:
                    mag = float(axis_means.get(str(axis_name), 0.0))
                    dx, dy = _axis_direction(axis_name)
                    max_mag = max(max_mag, abs(mag * dx), abs(mag * dy), abs(mag))
                    color = settings.axis_colors.get(str(axis_name), "#4d4d4d")
                    inset.plot(
                        [0.0, mag * dx],
                        [0.0, mag * dy],
                        color=color,
                        linewidth=1.5,
                        alpha=0.95,
                        zorder=2,
                    )
                inset.plot(
                    [-1.2 * max_mag * ux_cross, 1.2 * max_mag * ux_cross],
                    [-1.2 * max_mag * uy_cross, 1.2 * max_mag * uy_cross],
                    linestyle=(0, (2.2, 2.2)),
                    color=settings.axis_colors.get("cross_interaction", "#7a0177"),
                    linewidth=0.7,
                    alpha=0.45,
                    zorder=1,
                )
                inset.set_xlim(-1.2 * max_mag, 1.2 * max_mag)
                inset.set_ylim(-1.2 * max_mag, 1.2 * max_mag)
                inset.set_aspect("equal", adjustable="box")
                inset.set_xticks([])
                inset.set_yticks([])
                inset_lim = 1.2 * max_mag
                scale_candidates = np.asarray([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0], dtype=float)
                valid_scales = scale_candidates[scale_candidates <= (0.45 * inset_lim)]
                scale_len = float(valid_scales[-1]) if valid_scales.size > 0 else 0.01
                x0 = -0.90 * inset_lim
                x1 = x0 + scale_len
                y0 = -0.85 * inset_lim
                inset.plot([x0, x1], [y0, y0], color="#111111", linewidth=1.2, solid_capstyle="butt", zorder=3)
                inset.text(
                    (x0 + x1) / 2.0,
                    y0 - 0.09 * inset_lim,
                    f"{scale_len:g}",
                    ha="center",
                    va="top",
                    fontsize=5.2,
                    color="#111111",
                )
                inset.set_title("Mean Vectors", fontsize=5.5, pad=1.0)

            ax2.set_title(_display_region(region, settings), fontsize=10)
            ax2.set_xlim(-float(contour_axis_lim), float(contour_axis_lim))
            ax2.set_ylim(-float(contour_axis_lim), float(contour_axis_lim))
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_anchor("C")
            ax2.axhline(0.0, color="#1a1a1a", linewidth=1.6, zorder=5)
            ax2.axvline(0.0, color="#1a1a1a", linewidth=1.6, zorder=5)
            ax2.grid(alpha=0.14, linewidth=0.5)
            ax2.tick_params(axis="both", labelsize=6)
            ax2.set_xlabel("Face-Object", fontsize=7)
            if ridx == 0:
                ax2.set_ylabel("Interactive-State", fontsize=7)
            else:
                ax2.set_ylabel("")

        fig2.legend(handles=axis_handles, loc="upper center", frameon=False, fontsize=8, ncol=max(1, len(axis_handles)))
        if contour_handle is not None:
            cax = fig2.add_axes([0.935, 0.22, 0.012, 0.52])
            cbar2 = fig2.colorbar(
                contour_handle,
                cax=cax,
            )
            cbar2.set_label("Relative Density", fontsize=8)
            cbar2.ax.tick_params(labelsize=7)
            cbar2.formatter = FormatStrFormatter("%.2f")
            cbar2.update_ticks()
        fig2.suptitle(
            (
                "ROI-vs-Period 2D Density Contours by Region "
                f"({win_name}; mode={mode}; source={settings.axis_magnitude_source})"
            ),
            fontsize=10,
        )
        fig2.subplots_adjust(left=0.05, right=0.90, top=0.82, bottom=0.16, wspace=0.22)
        out_name_contours = ensure_filename(f"{output_filename_contours}{suffix}", f".{ext}")
        out_path_contours = out_root / out_name_contours
        save_figure(fig2, out_path_contours, ext=ext, dpi=settings.output_dpi)
        plt.close(fig2)
        outputs.append(
            {
                "output_path": str(out_path_contours),
                "window_name": str(win_name),
                "kind": "contours_2d",
                "regions": list(region_tokens),
                "axes": list(axis_tokens),
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
