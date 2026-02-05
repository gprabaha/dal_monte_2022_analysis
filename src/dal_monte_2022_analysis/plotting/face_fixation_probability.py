"""Plot face fixation probability comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_plotting_config,
)
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir
from dal_monte_2022_analysis.plotting.common import (
    apply_plotting_config,
    format_p_value,
    resolve_figsize,
)


@dataclass
class FaceFixationProbabilityPlotSettings:
    """Configuration for face fixation probability plotting."""
    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    analysis_subdir: str = "face_fixation_probability"
    within_filename: str = "within_session_face_fixation_probability.csv"
    cross_filename: str = "cross_session_face_fixation_probability.csv"
    output_filename: str = "face_fixation_probability_violin.pdf"


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
    quantile_cfg: dict,
) -> None:
    """Plot two violins (product vs joint) with quantile overlays."""
    width = float(violin_cfg.get("width", 0.7))
    body_alpha = float(violin_cfg.get("body_alpha", 0.8))
    body_edge = violin_cfg.get("body_edgecolor", "#1f1f1f")
    body_linewidth = float(violin_cfg.get("body_linewidth", 0.8))
    colors = violin_cfg.get("colors", {})
    product_color = colors.get("product", "#6C8EBF")
    joint_color = colors.get("joint", "#E07B39")

    quantiles = quantile_cfg.get("values", [0.25, 0.5, 0.75])
    quantile_color = quantile_cfg.get("color", "#1f1f1f")
    quantile_linewidth = float(quantile_cfg.get("linewidth", 1.2))

    positions = [1, 2]
    datasets = [product, joint]
    colors = [product_color, joint_color]

    for pos, data, color in zip(positions, datasets, colors):
        if data.size == 0:
            continue
        parts = ax.violinplot(
            [data],
            positions=[pos],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor(body_edge)
        body.set_alpha(body_alpha)
        body.set_linewidth(body_linewidth)

        qs = np.quantile(data, quantiles)
        for q in qs:
            ax.plot(
                [pos - width * 0.35, pos + width * 0.35],
                [q, q],
                color=quantile_color,
                linewidth=quantile_linewidth,
                solid_capstyle="round",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(["P(m1)*P(m2)", "P(m1&m2)"])
    ax.set_ylim(bottom=0)


def _load_probability_frames(
    cfg: dict,
    settings: FaceFixationProbabilityPlotSettings,
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


def plot_face_fixation_probability_violin(
    settings: FaceFixationProbabilityPlotSettings,
) -> Path:
    """Plot within/cross-session face fixation probability violins and save PDF."""
    cfg = load_dataset_config(settings.cfg_path)
    plot_cfg = load_plotting_config(settings.plotting_cfg_path)
    apply_plotting_config(plot_cfg)

    within_df, cross_df = _load_probability_frames(cfg, settings)
    within_product, within_joint = _within_arrays(within_df)
    within_pvals = _compute_tests(within_product, within_joint)

    cross_product = np.array([])
    cross_joint = np.array([])
    cross_pvals = {"ttest": np.nan, "ranksum": np.nan, "ks": np.nan}
    if cross_df is not None and not cross_df.empty:
        cross_product, cross_joint = _cross_arrays(cross_df)
        cross_pvals = _compute_tests(cross_product, cross_joint)

    figsize, dpi = resolve_figsize(plot_cfg)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharey=True)

    violin_cfg = plot_cfg.get("violin", {})
    quantile_cfg = plot_cfg.get("quantiles", {})

    _plot_violin_pair(
        axes[0],
        within_product,
        within_joint,
        violin_cfg=violin_cfg,
        quantile_cfg=quantile_cfg,
    )
    axes[0].set_title(_title_with_pvalues("Within session", within_pvals))
    axes[0].set_ylabel("Probability")

    if cross_df is not None and not cross_df.empty:
        _plot_violin_pair(
            axes[1],
            cross_product,
            cross_joint,
            violin_cfg=violin_cfg,
            quantile_cfg=quantile_cfg,
        )
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
