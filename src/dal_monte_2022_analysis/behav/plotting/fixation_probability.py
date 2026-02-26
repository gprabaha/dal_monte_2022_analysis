"""Plot fixation probability comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PolyCollection
from scipy import stats

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir
from dal_monte_2022_analysis.behav.plotting.common import (
    apply_plotting_config,
    format_p_value,
    resolve_figsize,
)


@dataclass
class FixationProbabilityPlotSettings:
    """Configuration for fixation probability plotting."""
    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "fixation_probability"
    within_filename: str = "within_session_face_fixation_probability.csv"
    cross_filename: str = "cross_session_face_fixation_probability.csv"
    output_filename: str = "face_fixation_probability_violin.pdf"


@dataclass
class InteractiveFixationProbabilityPlotSettings:
    """Configuration for interactive-period fixation probability plotting."""
    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "fixation_probability"
    interactive_periods_filename: str = (
        "within_session_interactive_period_face_fixation_probability.csv"
    )
    interactive_concat_filename: str = (
        "within_session_interactive_concat_face_fixation_probability.csv"
    )
    output_filename: str = "interactive_face_fixation_probability_violin.pdf"


_DEFAULT_FIXATION_PROB_STYLES = {
    "face": {
        "within": {
            "product": "#F7A6A6",
            "joint": "#FF2D2D",
            "body_alpha": 1.0,
            "body_edgecolor": "#1F1F1F",
        },
        "cross": {
            "product": "#F7A6A6",
            "joint": "#FF2D2D",
            "body_alpha": 1.0,
            "body_edgecolor": "#1F1F1F",
            "hatch": "xx",
            "fill": False,
        },
    },
    "out_of_roi": {
        "within": {
            "product": "#AFAFAF",
            "joint": "#333333",
            "body_alpha": 1.0,
            "body_edgecolor": "#1F1F1F",
        },
        "cross": {
            "product": "#AFAFAF",
            "joint": "#333333",
            "body_alpha": 1.0,
            "body_edgecolor": "#1F1F1F",
            "hatch": "xx",
            "fill": False,
        },
    },
}


def _resolve_fixation_probability_figsize(plot_cfg: dict) -> tuple[tuple[float, float], int]:
    """Resolve figure size/DPI with fixation-probability specific override."""
    figsize, dpi = resolve_figsize(plot_cfg)
    fixprob_cfg = plot_cfg.get("fixation_probability", {})
    override = fixprob_cfg.get("figsize")
    if override is not None and len(override) == 2:
        return (float(override[0]), float(override[1])), dpi
    if figsize is not None:
        return (float(figsize[0]), float(figsize[1])), dpi
    return (8.0, 4.8), dpi


def _infer_fixation_kind(settings: FixationProbabilityPlotSettings) -> str:
    """Infer fixation type from filenames to select palette defaults."""
    label = " ".join([
        settings.within_filename,
        settings.cross_filename,
        settings.output_filename,
    ]).lower()
    return "out_of_roi" if "out_of_roi" in label else "face"


def _resolve_panel_style(
    plot_cfg: dict,
    *,
    fixation_kind: str,
    comparison: str,
) -> dict:
    """Resolve panel style with plot config override and sensible defaults."""
    fixation_kind = fixation_kind if fixation_kind in _DEFAULT_FIXATION_PROB_STYLES else "face"
    comparison = comparison if comparison in ("within", "cross") else "within"

    default_style = _DEFAULT_FIXATION_PROB_STYLES[fixation_kind][comparison]
    fixprob_cfg = plot_cfg.get("fixation_probability", {})
    override = (
        fixprob_cfg.get("styles", {})
        .get(fixation_kind, {})
        .get(comparison, {})
    )
    style = dict(default_style)
    if isinstance(override, dict):
        style.update(override)
    return style


def _pair_label(m1, m2) -> str:
    """Build an order-insensitive monkey-pair label."""
    name_1 = str(m1).strip() if pd.notna(m1) and str(m1).strip() else "unknown_m1"
    name_2 = str(m2).strip() if pd.notna(m2) and str(m2).strip() else "unknown_m2"
    a, b = sorted((name_1, name_2))
    return f"{a}-{b}"


def _compute_monkey_pair_means(
    within_df: pd.DataFrame,
    product: np.ndarray,
    joint: np.ndarray,
) -> pd.DataFrame:
    """Compute monkey-pair average product/joint values for within-session overlay."""
    if within_df.empty:
        return pd.DataFrame()
    if product.size != len(within_df) or joint.size != len(within_df):
        return pd.DataFrame()
    if "monkey_name_m1" not in within_df.columns or "monkey_name_m2" not in within_df.columns:
        return pd.DataFrame()

    overlay = pd.DataFrame({
        "pair": [
            _pair_label(m1, m2)
            for m1, m2 in zip(within_df["monkey_name_m1"], within_df["monkey_name_m2"])
        ],
        "product": np.asarray(product, dtype=float),
        "joint": np.asarray(joint, dtype=float),
    })
    overlay = overlay[np.isfinite(overlay["product"]) & np.isfinite(overlay["joint"])]
    if overlay.empty:
        return pd.DataFrame()
    return overlay.groupby("pair", as_index=False)[["product", "joint"]].mean()


def _overlay_monkey_pair_means(ax, pair_means: pd.DataFrame, overlay_cfg: dict) -> None:
    """Overlay within-session monkey-pair means as connected points."""
    if pair_means.empty:
        return
    if not bool(overlay_cfg.get("enabled", True)):
        return

    use_pair_colors = bool(overlay_cfg.get("use_pair_colors", True))
    pair_cmap_name = str(overlay_cfg.get("pair_colormap", "tab20"))
    fallback_color = str(overlay_cfg.get("line_color", "#000000"))
    line_alpha = float(overlay_cfg.get("line_alpha", 1.0))
    line_width = float(overlay_cfg.get("line_width", 1.0))
    marker_size = float(overlay_cfg.get("marker_size", 24))
    marker_linewidth = float(overlay_cfg.get("marker_linewidth", 0.45))
    marker_alpha = float(overlay_cfg.get("marker_alpha", 1.0))
    clip_on = bool(overlay_cfg.get("clip_on", False))

    pair_means_sorted = pair_means.sort_values("pair").reset_index(drop=True)
    pair_cmap = plt.get_cmap(pair_cmap_name) if use_pair_colors else None
    n_pairs = len(pair_means_sorted)

    for idx, row in pair_means_sorted.iterrows():
        color = fallback_color
        if pair_cmap is not None:
            color = pair_cmap(0.0 if n_pairs <= 1 else idx / float(n_pairs - 1))
        y_vals = [float(row["product"]), float(row["joint"])]
        line_artist = ax.plot(
            [0, 1],
            y_vals,
            color=color,
            alpha=line_alpha,
            linewidth=line_width,
            zorder=3,
            linestyle="-",
            solid_capstyle="round",
            clip_on=clip_on,
        )[0]
        line_artist.set_clip_path(None)

        marker_artist = ax.scatter(
            [0, 1],
            y_vals,
            s=marker_size,
            facecolors=[color, color],
            edgecolors=[color, color],
            linewidths=marker_linewidth,
            alpha=marker_alpha,
            zorder=4,
            clip_on=clip_on,
        )
        marker_artist.set_clip_path(None)


def _apply_probability_axis_ticks(ax, *, y_max_ticks: int) -> None:
    """Ensure y-axis has no more than the configured number of major ticks."""
    max_ticks = max(2, int(y_max_ticks))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=max_ticks - 1, min_n_ticks=2)
    )


def _safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Compute numer/denom with zeros handled as NaN."""
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.full_like(numer, np.nan, dtype=float)
    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    return out


def _compute_tests(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute t-test, ranksum, and KS p-values for two samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"ttest": np.nan, "ranksum": np.nan, "ks": np.nan}

    ttest_res = stats.ttest_ind(a, b, equal_var=False)
    ranksum_res = stats.ranksums(a, b)
    ks_res = stats.ks_2samp(a, b)

    return {
        "ttest": ttest_res.pvalue,
        "ranksum": ranksum_res.pvalue,
        "ks": ks_res.pvalue,
    }


def _title_with_pvalues(title: str, pvals: dict) -> str:
    """Build a multiline title with p-values."""
    return (
        f"{title}\n"
        f"t-test p={format_p_value(pvals.get('ttest'))}\n"
        f"ranksum p={format_p_value(pvals.get('ranksum'))}\n"
        f"KS p={format_p_value(pvals.get('ks'))}"
    )


def _plot_violin_pair(
    ax,
    product: np.ndarray,
    joint: np.ndarray,
    *,
    violin_cfg: dict,
    panel_style: Optional[dict] = None,
) -> None:
    """Plot two violins (product vs joint) using seaborn."""
    panel_style = panel_style or {}
    width = float(panel_style.get("width", violin_cfg.get("width", 0.7)))
    body_alpha = float(panel_style.get("body_alpha", violin_cfg.get("body_alpha", 0.8)))
    body_edge = panel_style.get("body_edgecolor", violin_cfg.get("body_edgecolor", "#1f1f1f"))
    body_linewidth = float(
        panel_style.get("body_linewidth", violin_cfg.get("body_linewidth", 0.8))
    )
    inner = violin_cfg.get("inner", "quart")
    cut = float(violin_cfg.get("cut", 0))
    colors = violin_cfg.get("colors", {})
    product_color = panel_style.get("product", colors.get("product", "#6C8EBF"))
    joint_color = panel_style.get("joint", colors.get("joint", "#E07B39"))
    hatch = panel_style.get("hatch")
    fill = bool(panel_style.get("fill", True))

    product = np.asarray(product, dtype=float)
    joint = np.asarray(joint, dtype=float)
    product = product[np.isfinite(product)]
    joint = joint[np.isfinite(joint)]

    labels = ["P(m1)*P(m2)", "P(m1&m2)"]
    data = pd.DataFrame({
        "comparison": [labels[0]] * product.size + [labels[1]] * joint.size,
        "probability": np.concatenate([product, joint]) if product.size or joint.size else [],
    })

    if data.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    sns.violinplot(
        ax=ax,
        data=data,
        x="comparison",
        y="probability",
        hue="comparison",
        order=labels,
        palette=[product_color, joint_color],
        legend=False,
        width=width,
        inner=inner,
        cut=cut,
        linewidth=body_linewidth,
    )

    body_colors = [product_color, joint_color]
    bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    for idx, body in enumerate(bodies):
        this_color = body_colors[idx % len(body_colors)]
        if fill:
            body.set_edgecolor(body_edge)
            body.set_alpha(body_alpha)
        else:
            body.set_facecolor("none")
            body.set_edgecolor(this_color)
            body.set_alpha(1.0)
        if hatch is not None and str(hatch).strip():
            body.set_hatch(str(hatch))

    ax.set_xlabel("")
    ax.set_ylim(bottom=0)


def _load_probability_frames(
    cfg: dict,
    settings: FixationProbabilityPlotSettings,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load within and cross-session probability tables."""
    out_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)
    within_path = out_dir / settings.within_filename
    cross_path = out_dir / settings.cross_filename

    if not within_path.exists():
        raise FileNotFoundError(f"Missing within-session file: {within_path}")
    within_df = pd.read_csv(within_path)

    cross_df = None
    if cross_path.exists():
        cross_df = pd.read_csv(cross_path)

    return within_df, cross_df


def _within_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute product and joint probabilities for within-session data."""
    denom = df["n_samples"].to_numpy(dtype=float)
    p_m1 = _safe_ratio(df["m1_face_count"].to_numpy(), denom)
    p_m2 = _safe_ratio(df["m2_face_count"].to_numpy(), denom)
    p_product = p_m1 * p_m2
    p_joint = _safe_ratio(df["joint_face_count"].to_numpy(), denom)
    return p_product, p_joint


def _cross_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute product and joint probabilities for cross-session data."""
    denom = df["n_samples_joint"].to_numpy(dtype=float)
    p_m1 = _safe_ratio(df["m1_face_count_joint"].to_numpy(), denom)
    p_m2 = _safe_ratio(df["m2_face_count_joint"].to_numpy(), denom)
    p_product = p_m1 * p_m2
    p_joint = _safe_ratio(df["joint_face_count"].to_numpy(), denom)
    return p_product, p_joint


def _subsample_pair(
    product: np.ndarray,
    joint: np.ndarray,
    *,
    target_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample paired arrays to a target size without replacement."""
    if target_size <= 0:
        return np.array([]), np.array([])
    if product.size <= target_size:
        return product, joint
    rng = np.random.default_rng(seed)
    indices = rng.choice(product.size, size=target_size, replace=False)
    return product[indices], joint[indices]


def plot_fixation_probability_violin(
    settings: FixationProbabilityPlotSettings,
) -> Path:
    """Plot within/cross-session fixation probability violins and save PDF."""
    cfg = load_config(settings.cfg_path)
    plot_cfg = load_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)

    within_df, cross_df = _load_probability_frames(cfg, settings)
    within_product, within_joint = _within_arrays(within_df)
    within_pvals = _compute_tests(within_product, within_joint)
    within_pair_means = _compute_monkey_pair_means(within_df, within_product, within_joint)

    cross_product = np.array([])
    cross_joint = np.array([])
    cross_pvals = {"ttest": np.nan, "ranksum": np.nan, "ks": np.nan}
    if cross_df is not None and not cross_df.empty:
        cross_product, cross_joint = _cross_arrays(cross_df)
        target_size = min(within_product.size, within_joint.size)
        seed = int(plot_cfg.get("cross_subsample_seed", 42))
        cross_product, cross_joint = _subsample_pair(
            cross_product,
            cross_joint,
            target_size=target_size,
            seed=seed,
        )
        cross_pvals = _compute_tests(cross_product, cross_joint)

    figsize, dpi = _resolve_fixation_probability_figsize(plot_cfg)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharey=True)

    violin_cfg = plot_cfg.get("violin", {})
    fixprob_cfg = plot_cfg.get("fixation_probability", {})
    fixation_kind = _infer_fixation_kind(settings)
    within_style = _resolve_panel_style(
        plot_cfg,
        fixation_kind=fixation_kind,
        comparison="within",
    )
    cross_style = _resolve_panel_style(
        plot_cfg,
        fixation_kind=fixation_kind,
        comparison="cross",
    )
    overlay_cfg = fixprob_cfg.get("within_pair_overlay", {})
    y_max_ticks = int(fixprob_cfg.get("y_max_ticks", 5))

    _plot_violin_pair(
        axes[0],
        within_product,
        within_joint,
        violin_cfg=violin_cfg,
        panel_style=within_style,
    )
    _overlay_monkey_pair_means(axes[0], within_pair_means, overlay_cfg)
    _apply_probability_axis_ticks(axes[0], y_max_ticks=y_max_ticks)
    axes[0].set_title(_title_with_pvalues("Within session", within_pvals))
    axes[0].set_ylabel("Probability")

    if cross_df is not None and not cross_df.empty:
        _plot_violin_pair(
            axes[1],
            cross_product,
            cross_joint,
            violin_cfg=violin_cfg,
            panel_style=cross_style,
        )
        _apply_probability_axis_ticks(axes[1], y_max_ticks=y_max_ticks)
        axes[1].set_title(_title_with_pvalues("Cross session", cross_pvals))
    else:
        axes[1].set_axis_off()
        axes[1].text(0.5, 0.5, "No cross-session data", ha="center", va="center")

    fig.tight_layout()

    out_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / settings.output_filename
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return out_path


def _load_interactive_probability_frames(
    cfg: dict,
    settings: InteractiveFixationProbabilityPlotSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load interactive-period probability tables."""
    out_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)
    periods_path = out_dir / settings.interactive_periods_filename
    concat_path = out_dir / settings.interactive_concat_filename

    if not periods_path.exists():
        raise FileNotFoundError(f"Missing interactive-period file: {periods_path}")
    if not concat_path.exists():
        raise FileNotFoundError(f"Missing interactive-concat file: {concat_path}")

    periods_df = pd.read_csv(periods_path)
    concat_df = pd.read_csv(concat_path)

    return periods_df, concat_df


def _interactive_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute product and joint probabilities for interactive-period data."""
    denom = df["n_samples"].to_numpy(dtype=float)
    p_m1 = _safe_ratio(df["m1_face_count"].to_numpy(), denom)
    p_m2 = _safe_ratio(df["m2_face_count"].to_numpy(), denom)
    p_product = p_m1 * p_m2
    p_joint = _safe_ratio(df["joint_face_count"].to_numpy(), denom)
    return p_product, p_joint


def plot_interactive_fixation_probability_violin(
    settings: InteractiveFixationProbabilityPlotSettings,
) -> Path:
    """Plot interactive-period fixation probability violins and save PDF."""
    cfg = load_config(settings.cfg_path)
    plot_cfg = load_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)

    periods_df, concat_df = _load_interactive_probability_frames(cfg, settings)
    periods_product, periods_joint = _interactive_arrays(periods_df)
    periods_pvals = _compute_tests(periods_product, periods_joint)

    concat_product, concat_joint = _interactive_arrays(concat_df)
    concat_pvals = _compute_tests(concat_product, concat_joint)

    figsize, dpi = _resolve_fixation_probability_figsize(plot_cfg)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharey=True)

    violin_cfg = plot_cfg.get("violin", {})
    y_max_ticks = int(plot_cfg.get("fixation_probability", {}).get("y_max_ticks", 5))

    _plot_violin_pair(
        axes[0],
        periods_product,
        periods_joint,
        violin_cfg=violin_cfg,
    )
    _apply_probability_axis_ticks(axes[0], y_max_ticks=y_max_ticks)
    axes[0].set_title(
        _title_with_pvalues("Interactive periods (separate)", periods_pvals)
    )
    axes[0].set_ylabel("Probability")

    _plot_violin_pair(
        axes[1],
        concat_product,
        concat_joint,
        violin_cfg=violin_cfg,
    )
    _apply_probability_axis_ticks(axes[1], y_max_ticks=y_max_ticks)
    axes[1].set_title(
        _title_with_pvalues("Interactive periods (concatenated)", concat_pvals)
    )

    fig.tight_layout()

    out_dir = build_analysis_output_dir(cfg, settings.analysis_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / settings.output_filename
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return out_path
