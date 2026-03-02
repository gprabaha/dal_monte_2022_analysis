"""Plot pupil-vs-fixation-density correlation violins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PolyCollection
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.behav.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


@dataclass
class PupilFixationDensityCorrelationPlotSettings:
    """Configuration for pupil-density correlation violin plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "pupil_fixation_density_correlation"
    correlations_filename: str = "within_session_pupil_vs_face_fixation_density_correlation.csv"
    output_filename: str = "pupil_fixation_density_correlation_violin.pdf"
    y_label: str = "Pearson r (pupil vs face-fix density)"


_SOURCE_ORDER = ("m1", "m2", "joint")
_SOURCE_LABELS = {"m1": "m1 density", "m2": "m2 density", "joint": "joint density"}

_DEFAULT_STYLE = {
    "figsize": [4.2, 2.8],  # roughly half US letter-page width
    "y_max_ticks": 5,
    "significance_alpha": 0.05,
    "violin_width": 0.78,
    "violin_edgecolor": "#1F1F1F",
    "violin_linewidth": 0.8,
    "xtick_label_rotation": 20.0,
    "xtick_label_ha": "right",
    "panel_titles": {"m1": "m1 pupil", "m2": "m2 pupil"},
    "colors": {"m1_family": "#2abcb1", "m2_family": "#4b3b7d"},
    "alphas": {"own": 1.0, "other": 0.30, "joint": 0.65},
    "significance": {
        "line_color": "#111111",
        "line_width": 0.9,
        "text_size": 9,
        "text_pad_frac": 0.012,
        "one_sample_text_size": 9,
        "one_sample_text_pad_frac": 0.018,
        "bar_height_frac": 0.020,
        "top_pad_frac": 0.050,
        "step_frac": 0.080,
    },
    "zero_line": {"color": "#1a1a1a", "linewidth": 0.9, "linestyle": "-"},
    "y_margin_frac_bottom": 0.08,
}


def _resolve_style(plot_cfg: dict) -> dict:
    """Resolve plotting style with config overrides."""
    style = dict(_DEFAULT_STYLE)
    override = plot_cfg.get("pupil_fixation_density_correlation", {})
    if not isinstance(override, dict):
        return style

    for key in (
        "figsize",
        "y_max_ticks",
        "significance_alpha",
        "violin_width",
        "violin_edgecolor",
        "violin_linewidth",
        "xtick_label_rotation",
        "xtick_label_ha",
        "y_margin_frac_bottom",
    ):
        if key in override:
            style[key] = override[key]

    for key in ("panel_titles", "colors", "alphas", "significance", "zero_line"):
        if key in override and isinstance(override[key], dict):
            merged = dict(style[key])
            merged.update(override[key])
            style[key] = merged
    return style


def _stars_for_pvalue(p: float) -> str:
    """Map p-values to significance-star labels."""
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def _alpha_for_source(
    pupil_agent: str,
    density_source: str,
    alpha_cfg: dict,
) -> float:
    """Resolve violin alpha for own/other/joint source convention."""
    if density_source == "joint":
        return float(alpha_cfg.get("joint", 0.65))
    if density_source == pupil_agent:
        return float(alpha_cfg.get("own", 1.0))
    return float(alpha_cfg.get("other", 0.30))


def _finite_array(values: np.ndarray) -> np.ndarray:
    """Return finite values as 1D float array."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _resolve_y_tick_step(
    y_min: float,
    y_max: float,
    style: dict,
) -> float:
    """Choose a regular y-tick step (0.15 or 0.2 by default)."""
    explicit = style.get("y_tick_step")
    if explicit is not None:
        step = float(explicit)
        if step > 0:
            return step
    span = float(max(y_max - y_min, 1e-9))
    return 0.15 if span <= 1.2 else 0.2


def _build_regular_ticks_with_zero(
    y_min: float,
    y_max: float,
    *,
    step: float,
) -> np.ndarray:
    """Build regular y ticks that always include 0."""
    step = float(max(step, 1e-9))
    start = float(np.floor(y_min / step) * step)
    stop = float(np.ceil(y_max / step) * step)
    ticks = np.arange(start, stop + 0.5 * step, step, dtype=float)
    if ticks.size == 0:
        ticks = np.asarray([0.0], dtype=float)
    if not np.any(np.isclose(ticks, 0.0, atol=step * 1e-6)):
        ticks = np.sort(np.append(ticks, 0.0))
    return np.round(ticks, 6)


def _pairwise_significant_comparisons(
    panel_df: pd.DataFrame,
    *,
    value_column: str,
    alpha: float,
) -> list[dict]:
    """Compute pairwise significance and keep comparisons with p < alpha.

    Uses paired t-tests across matched date/session rows when possible.
    Falls back to Welch two-sample t-test if date/session columns are absent.
    """
    tests: list[dict] = []
    use_paired = {"date", "session", "density_source", value_column}.issubset(panel_df.columns)
    pivot = None
    if use_paired:
        pivot = (
            panel_df
            .pivot_table(
                index=["date", "session"],
                columns="density_source",
                values=value_column,
                aggfunc="mean",
            )
            .sort_index()
        )

    pairs = [("m1", "m2"), ("m1", "joint"), ("m2", "joint")]
    for left_source, right_source in pairs:
        if use_paired and pivot is not None:
            if left_source not in pivot.columns or right_source not in pivot.columns:
                continue
            paired = pivot[[left_source, right_source]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(paired) < 2:
                continue
            p_value = float(
                ttest_rel(
                    paired[left_source].to_numpy(dtype=float),
                    paired[right_source].to_numpy(dtype=float),
                    nan_policy="omit",
                ).pvalue
            )
        else:
            left = _finite_array(
                panel_df.loc[panel_df["density_source"] == left_source, value_column].to_numpy(dtype=float)
            )
            right = _finite_array(
                panel_df.loc[panel_df["density_source"] == right_source, value_column].to_numpy(dtype=float)
            )
            if left.size < 2 or right.size < 2:
                continue
            p_value = float(ttest_ind(left, right, equal_var=False, nan_policy="omit").pvalue)

        if not np.isfinite(p_value) or p_value >= float(alpha):
            continue
        stars = _stars_for_pvalue(p_value)
        if not stars:
            continue
        tests.append({
            "left_source": left_source,
            "right_source": right_source,
            "left_pos": float(_SOURCE_ORDER.index(left_source)),
            "right_pos": float(_SOURCE_ORDER.index(right_source)),
            "p_value": p_value,
            "stars": stars,
        })
    return tests


def _one_sample_positive_stars(
    values: np.ndarray,
    *,
    alpha: float,
) -> str:
    """Return significance stars for one-sample test of mean(values) > 0."""
    arr = _finite_array(values)
    if arr.size < 2:
        return ""
    mean_val = float(np.mean(arr))
    if not np.isfinite(mean_val) or mean_val <= 0.0:
        return ""

    t_res = ttest_1samp(arr, popmean=0.0, nan_policy="omit")
    p_two = float(t_res.pvalue)
    t_stat = float(t_res.statistic)
    if not np.isfinite(p_two) or not np.isfinite(t_stat):
        return ""

    if t_stat > 0:
        p_one = 0.5 * p_two
    else:
        p_one = 1.0 - 0.5 * p_two
    if p_one >= float(alpha):
        return ""
    return _stars_for_pvalue(p_one)


def _draw_significance_bars(
    ax,
    *,
    comparisons: list[dict],
    y_start: float,
    y_span: float,
    sig_cfg: dict,
) -> None:
    """Draw significance brackets/stars for significant pairwise tests."""
    if not comparisons:
        return

    y_step = float(sig_cfg.get("step_frac", 0.08)) * y_span
    bar_height = float(sig_cfg.get("bar_height_frac", 0.02)) * y_span
    text_pad = float(sig_cfg.get("text_pad_frac", 0.012)) * y_span
    line_color = str(sig_cfg.get("line_color", "#111111"))
    line_w = float(sig_cfg.get("line_width", 0.9))
    text_size = float(sig_cfg.get("text_size", 9))

    for i, comp in enumerate(comparisons):
        y = y_start + i * y_step
        x1 = float(comp["left_pos"])
        x2 = float(comp["right_pos"])

        line = ax.plot(
            [x1, x1, x2, x2],
            [y, y + bar_height, y + bar_height, y],
            color=line_color,
            linewidth=line_w,
            zorder=5,
            clip_on=False,
            rasterized=False,
        )[0]
        line.set_clip_path(None)
        ax.text(
            0.5 * (x1 + x2),
            y + bar_height + text_pad,
            str(comp["stars"]),
            ha="center",
            va="bottom",
            color=line_color,
            fontsize=text_size,
            zorder=6,
            clip_on=False,
        )


def _draw_one_sample_stars(
    ax,
    *,
    stars_by_source: dict[str, str],
    values_by_source: dict[str, np.ndarray],
    y_span: float,
    sig_cfg: dict,
) -> None:
    """Draw per-violin stars for one-sample significance above zero."""
    if not stars_by_source:
        return
    color = str(sig_cfg.get("line_color", "#111111"))
    text_size = float(sig_cfg.get("one_sample_text_size", sig_cfg.get("text_size", 9)))
    pad = float(sig_cfg.get("one_sample_text_pad_frac", 0.018)) * y_span

    for source in _SOURCE_ORDER:
        stars = str(stars_by_source.get(source, ""))
        if not stars:
            continue
        vals = _finite_array(values_by_source.get(source, np.array([])))
        if vals.size == 0:
            continue
        x = float(_SOURCE_ORDER.index(source))
        y = float(np.max(vals) + pad)
        ax.text(
            x,
            y,
            stars,
            ha="center",
            va="bottom",
            color=color,
            fontsize=text_size,
            zorder=6,
            clip_on=False,
        )


def _plot_panel(
    ax,
    *,
    panel_df: pd.DataFrame,
    pupil_agent: str,
    style: dict,
    y_min: float,
    y_max: float,
    y_span: float,
    comparisons: list[dict],
    violin_cfg: dict,
    one_sample_stars_by_source: dict[str, str],
    value_column: str,
) -> None:
    """Render one pupil-agent violin panel."""
    color_family = style["colors"]["m1_family"] if pupil_agent == "m1" else style["colors"]["m2_family"]
    alpha_cfg = style["alphas"]

    values_by_source: dict[str, np.ndarray] = {}
    sources_present: list[str] = []
    plot_rows: list[dict] = []
    for source in _SOURCE_ORDER:
        vals = _finite_array(
            panel_df.loc[panel_df["density_source"] == source, value_column].to_numpy(dtype=float)
        )
        values_by_source[source] = vals
        if vals.size > 0:
            sources_present.append(source)
            for value in vals:
                plot_rows.append({
                    "density_source": source,
                    "corr_value": float(value),
                })

    if not plot_rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    plot_df = pd.DataFrame(plot_rows)
    order_sources = list(_SOURCE_ORDER)
    palette = {src: color_family for src in _SOURCE_ORDER}
    sns.violinplot(
        ax=ax,
        data=plot_df,
        x="density_source",
        y="corr_value",
        hue="density_source",
        order=order_sources,
        hue_order=order_sources,
        palette=palette,
        legend=False,
        width=float(style["violin_width"]),
        inner=violin_cfg.get("inner", "quart"),
        cut=float(violin_cfg.get("cut", 0)),
        linewidth=float(style["violin_linewidth"]),
    )
    bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    for body, source in zip(bodies, sources_present):
        body.set_facecolor(color_family)
        body.set_edgecolor(str(style["violin_edgecolor"]))
        body.set_linewidth(float(style["violin_linewidth"]))
        body.set_alpha(_alpha_for_source(pupil_agent, source, alpha_cfg))
        body.set_rasterized(False)
        body.set_clip_path(None)

    title = style["panel_titles"].get(pupil_agent, f"{pupil_agent} pupil")
    ax.set_title(str(title))
    ax.set_xlabel("")
    ax.set_xticks(list(range(len(_SOURCE_ORDER))))
    ax.set_xticklabels(
        [_SOURCE_LABELS[s] for s in _SOURCE_ORDER],
        rotation=float(style.get("xtick_label_rotation", 20.0)),
        ha=str(style.get("xtick_label_ha", "right")),
    )
    ax.set_ylim(y_min, y_max)
    ax.grid(False)
    tick_step = _resolve_y_tick_step(y_min, y_max, style)
    ax.set_yticks(_build_regular_ticks_with_zero(y_min, y_max, step=tick_step))

    zero_cfg = style["zero_line"]
    zero_line = ax.axhline(
        0.0,
        color=str(zero_cfg.get("color", "#1a1a1a")),
        linewidth=float(zero_cfg.get("linewidth", 0.9)),
        linestyle=str(zero_cfg.get("linestyle", "-")),
        zorder=1,
        rasterized=False,
    )
    zero_line.set_clip_path(None)

    y_start = (
        np.max([np.max(_finite_array(v)) for v in values_by_source.values() if _finite_array(v).size > 0])
        + float(style["significance"].get("top_pad_frac", 0.05)) * y_span
    )
    _draw_one_sample_stars(
        ax,
        stars_by_source=one_sample_stars_by_source,
        values_by_source=values_by_source,
        y_span=y_span,
        sig_cfg=style["significance"],
    )
    _draw_significance_bars(
        ax,
        comparisons=comparisons,
        y_start=float(y_start),
        y_span=y_span,
        sig_cfg=style["significance"],
    )


def _load_correlation_df(
    cfg: dict,
    settings: PupilFixationDensityCorrelationPlotSettings,
) -> tuple[pd.DataFrame, Path]:
    """Load the correlation table used for plotting."""
    out_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)
    corr_path = out_dir / settings.correlations_filename
    out_path = out_dir / settings.output_filename

    if not corr_path.exists():
        raise FileNotFoundError(f"Missing correlation file: {corr_path}")
    df = pd.read_csv(corr_path)
    return df, out_path


def plot_pupil_fixation_density_correlation_violin(
    settings: PupilFixationDensityCorrelationPlotSettings,
) -> Path:
    """Plot two-panel pupil-density correlation violins and save a PDF."""
    cfg = load_config(settings.cfg_path)
    plot_cfg = load_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)
    style = _resolve_style(plot_cfg)

    corr_df, out_path = _load_correlation_df(cfg, settings)
    if "correlation_r" in corr_df.columns:
        value_column = "correlation_r"
    elif "pearson_r" in corr_df.columns:
        value_column = "pearson_r"
    else:
        raise RuntimeError(
            "Correlation table missing required value columns. Expected "
            "'correlation_r' (preferred) or 'pearson_r' (legacy)."
        )

    required_cols = {"pupil_agent", "density_source", value_column}
    missing = required_cols.difference(corr_df.columns)
    if missing:
        raise RuntimeError(f"Correlation table missing required columns: {sorted(missing)}")

    corr_df = corr_df.copy()
    corr_df["pupil_agent"] = corr_df["pupil_agent"].astype(str).str.lower()
    corr_df["density_source"] = corr_df["density_source"].astype(str).str.lower()
    corr_df[value_column] = pd.to_numeric(corr_df[value_column], errors="coerce")
    corr_method = "pearson"
    if "correlation_method" in corr_df.columns:
        method_tokens = (
            corr_df["correlation_method"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": ""})
        )
        uniq = sorted(set([m for m in method_tokens if m]))
        if uniq:
            corr_method = uniq[0]

    corr_df = corr_df[
        corr_df["pupil_agent"].isin({"m1", "m2"})
        & corr_df["density_source"].isin(set(_SOURCE_ORDER))
    ]
    if corr_df.empty:
        raise RuntimeError("No valid pupil-density correlation rows found for plotting.")

    panel_data = {
        "m1": corr_df[corr_df["pupil_agent"] == "m1"].copy(),
        "m2": corr_df[corr_df["pupil_agent"] == "m2"].copy(),
    }

    finite_values = _finite_array(corr_df[value_column].to_numpy(dtype=float))
    if finite_values.size == 0:
        raise RuntimeError("No finite correlation values found for plotting.")

    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    data_span = float(max(data_max - data_min, 1e-6))
    y_span = data_span if data_span > 1e-6 else 1.0

    alpha = float(style["significance_alpha"])
    comparisons_by_panel = {
        agent: _pairwise_significant_comparisons(
            panel_data[agent],
            value_column=value_column,
            alpha=alpha,
        )
        for agent in ("m1", "m2")
    }
    one_sample_stars_by_panel: dict[str, dict[str, str]] = {}
    for agent in ("m1", "m2"):
        one_sample_stars_by_panel[agent] = {}
        for source in _SOURCE_ORDER:
            vals = panel_data[agent].loc[
                panel_data[agent]["density_source"] == source,
                value_column,
            ].to_numpy(dtype=float)
            one_sample_stars_by_panel[agent][source] = _one_sample_positive_stars(
                vals,
                alpha=alpha,
            )

    sig_cfg = style["significance"]
    max_sig = max(len(comparisons_by_panel["m1"]), len(comparisons_by_panel["m2"]))
    top_span = (
        float(sig_cfg.get("top_pad_frac", 0.05))
        + max(0, max_sig - 1) * float(sig_cfg.get("step_frac", 0.08))
        + float(sig_cfg.get("bar_height_frac", 0.02))
        + float(sig_cfg.get("text_pad_frac", 0.012))
        + 0.02
    ) * y_span
    y_max = data_max + top_span
    y_min = data_min - float(style.get("y_margin_frac_bottom", 0.08)) * y_span

    global_figsize, dpi = resolve_figsize(plot_cfg)
    violin_cfg = plot_cfg.get("violin", {})
    figsize = style.get("figsize")
    if figsize is None:
        figsize = global_figsize
    if figsize is None:
        figsize = [4.2, 2.8]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(float(figsize[0]), float(figsize[1])),
        dpi=dpi,
        sharey=True,
    )

    _plot_panel(
        axes[0],
        panel_df=panel_data["m1"],
        pupil_agent="m1",
        style=style,
        y_min=y_min,
        y_max=y_max,
        y_span=y_span,
        comparisons=comparisons_by_panel["m1"],
        violin_cfg=violin_cfg,
        one_sample_stars_by_source=one_sample_stars_by_panel["m1"],
        value_column=value_column,
    )
    _plot_panel(
        axes[1],
        panel_df=panel_data["m2"],
        pupil_agent="m2",
        style=style,
        y_min=y_min,
        y_max=y_max,
        y_span=y_span,
        comparisons=comparisons_by_panel["m2"],
        violin_cfg=violin_cfg,
        one_sample_stars_by_source=one_sample_stars_by_panel["m2"],
        value_column=value_column,
    )

    y_label = str(settings.y_label)
    if y_label == PupilFixationDensityCorrelationPlotSettings.y_label:
        if corr_method == "spearman":
            y_label = "Spearman rho (pupil vs face-fix density)"
        else:
            y_label = "Pearson r (pupil vs face-fix density)"
    axes[0].set_ylabel(y_label)
    axes[1].set_ylabel("")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, ext="pdf")
    plt.close(fig)
    return out_path
