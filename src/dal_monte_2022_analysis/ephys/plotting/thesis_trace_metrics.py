"""Appendix figures quantifying the shape of mean firing-rate traces.

Each function returns **one embeddable panel** -- a single figure, laid out as a
row of four regions -- so the appendix can place them individually rather than
cropping a composite. All panels default to the fixation-category-modulated
subpopulation, since a shape statistic on an unmodulated trace describes noise.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from dal_monte_2022_analysis.core.stats import adjust_pvalues
from dal_monte_2022_analysis.ephys.analysis.fixation_temporal_specificity import (
    METRIC_AXES,
    METRIC_LABELS,
    METRIC_NAMES,
)
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    DPP_ABBREV,
    DPP_AXIS_LABEL,
    DPP_COLUMN,
    EXEMPLAR_STYLE_COLORS,
    EXEMPLAR_STYLE_LABELS,
    EXEMPLAR_STYLE_MARKERS,
    INK,
    MUTED_INK,
    NEUTRAL_EDGE,
    NEUTRAL_FILL,
    REGION_ORDER,
    nice_axis,
    ordinal,
    region_label,
)

#: Short axis titles. ``METRIC_LABELS`` is written for tables and is too long to
#: sit above a 1.6-inch panel.
SHORT_METRIC_LABELS: dict[str, str] = {
    "mass_width_frac_50": "50% mass width\n(frac. of window)",
    "effective_width_ms": "Effective width (ms)",
    "lifetime_sparseness": "Lifetime sparseness",
    "peak_dominance": r"Peak dominance $P_1/\Sigma P$",
    "n_prominent_peaks": "Prominent peaks (n)",
    "fwhm_frac": "FWHM (frac. of window)",
    "sustained_frac": "Sustained fraction",
    "roughness": "Roughness (TV / range)",
    "autocorr_width_ms": "Autocorr. half-width (ms)",
    "modulation_index": "Modulation index",
    "peak_z": "Peak z (vs. baseline)",
    "temporal_specificity_index": "Temporal specificity index",
    "sustainedness_index": "Sustainedness index",
    DPP_COLUMN: DPP_AXIS_LABEL,
}

#: What a high value means, for the appendix table.
METRIC_INTERPRETATION: dict[str, str] = {
    "mass_width_frac_50": "excess response spread over many bins (broad)",
    "effective_width_ms": "wide equivalent-rectangle response (broad)",
    "lifetime_sparseness": "excess mass concentrated in few bins (peaked)",
    "peak_dominance": "one peak carries most of the total prominence",
    "n_prominent_peaks": "many comparable peaks (ragged)",
    "fwhm_frac": "long time above half the excess peak (sustained)",
    "sustained_frac": "long time above 25% of the excess peak (sustained)",
    "roughness": "many monotone excursions (fluctuating)",
    "autocorr_width_ms": "slowly varying trace (smooth)",
    "modulation_index": "large peak relative to baseline",
    "peak_z": "peak large relative to trace noise",
    "temporal_specificity_index": "narrow, single-peaked and smooth",
    "sustainedness_index": "broadly elevated and smooth",
    DPP_COLUMN: "one tall peak with no comparable rival",
}


def metric_axis_label(metric: str) -> str:
    """Short display label for a trace metric."""
    return SHORT_METRIC_LABELS.get(str(metric), METRIC_LABELS.get(str(metric), str(metric)))


def build_metric_definition_table(
    metrics: Sequence[str] = METRIC_NAMES,
) -> pd.DataFrame:
    """Metric name, axis it measures, and what a high value means."""
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "axis": METRIC_AXES.get(metric, "composite"),
                "label": METRIC_LABELS.get(metric, metric),
                "high value means": METRIC_INTERPRETATION.get(metric, ""),
            }
            for metric in metrics
        ]
    )


def build_metric_summary_table(
    units: pd.DataFrame,
    *,
    metrics: Sequence[str],
    regions: Sequence[str] = REGION_ORDER,
) -> pd.DataFrame:
    """Median and interquartile range of every metric, per region."""
    rows = []
    for metric in metrics:
        for region in regions:
            values = pd.to_numeric(
                units.loc[units["region"].astype(str) == str(region), metric],
                errors="coerce",
            ).dropna().to_numpy()
            if values.size == 0:
                continue
            rows.append(
                {
                    "metric": metric,
                    "label": metric_axis_label(metric).replace("\n", " "),
                    "region": region_label(region),
                    "n_units": int(values.size),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_distribution_panel(
    units: pd.DataFrame,
    metric: str,
    *,
    exemplars: Optional[pd.DataFrame] = None,
    regions: Sequence[str] = REGION_ORDER,
    bins: int = 26,
    upper_quantile: float = 0.995,
    figure_width_in: float = 7.2,
    figure_height_in: float = 1.85,
    annotate_exemplars: bool = True,
) -> plt.Figure:
    """One metric, four regions, with the example units marked.

    Bin edges are shared across regions so panel shapes are directly
    comparable; the axis is trimmed at ``upper_quantile`` because several
    metrics have a thin right tail that otherwise flattens the body.
    """
    fig, axes = plt.subplots(
        1,
        len(regions),
        figsize=(figure_width_in, figure_height_in),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    all_values = pd.to_numeric(units[metric], errors="coerce").dropna().to_numpy()
    if all_values.size == 0:
        raise ValueError(f"No finite values for metric {metric!r}.")
    x_max = float(np.quantile(all_values, upper_quantile))
    x_min = float(np.min(all_values))
    edges = np.histogram_bin_edges(all_values[all_values <= x_max], bins=bins)

    for ax, region in zip(axes, regions):
        values = pd.to_numeric(
            units.loc[units["region"].astype(str) == str(region), metric], errors="coerce"
        ).dropna().to_numpy()
        ax.hist(values, bins=edges, color=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE, linewidth=0.4)
        if values.size:
            ax.axvline(float(np.median(values)), color=NEUTRAL_EDGE, linewidth=1.2, zorder=4)

        if exemplars is not None:
            region_exemplars = exemplars.loc[exemplars["region"].astype(str) == str(region)]
            style_y = {"phasic": 0.90, "tonic": 0.68}
            for _, unit in region_exemplars.iterrows():
                if metric not in unit or not np.isfinite(float(unit[metric])):
                    continue
                style = str(unit["style"])
                color = EXEMPLAR_STYLE_COLORS.get(style, INK)
                value = float(unit[metric])
                y_frac = style_y.get(style, 0.8)
                ax.axvline(value, color=color, linestyle="--", linewidth=1.1, zorder=6)
                ax.plot(
                    [value],
                    [y_frac],
                    transform=ax.get_xaxis_transform(),
                    marker=EXEMPLAR_STYLE_MARKERS.get(style, "o"),
                    markersize=3.6,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    zorder=8,
                )
                if annotate_exemplars:
                    to_right = value < x_min + 0.55 * (x_max - x_min)
                    ax.annotate(
                        str(unit["uuid"]),
                        xy=(value, y_frac),
                        xycoords=ax.get_xaxis_transform(),
                        xytext=(4 if to_right else -4, 0),
                        textcoords="offset points",
                        ha="left" if to_right else "right",
                        va="center",
                        fontsize=5.4,
                        color=color,
                        zorder=8,
                    )
        ax.set_title(region_label(region), fontsize=8.2, pad=3)
        nice_axis(ax, y_ticks=3)
    axes[0].set_ylabel("Units", fontsize=7.2)
    axes[0].set_xlim(float(edges[0]), x_max)
    fig.supxlabel(metric_axis_label(metric).replace("\n", " "), fontsize=7.8)
    fig.tight_layout()
    return fig


def plot_metric_correlation_panel(
    units: pd.DataFrame,
    *,
    metrics: Sequence[str],
    figure_width_in: float = 4.6,
    figure_height_in: float = 4.2,
    method: str = "spearman",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Rank-correlation matrix showing which metrics are redundant.

    Diverging scale with a neutral midpoint, because correlation is a polarity
    measure: sign carries as much meaning as magnitude.
    """
    frame = units.loc[:, list(metrics)].apply(pd.to_numeric, errors="coerce")
    corr = frame.corr(method=method)

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [metric_axis_label(metric).replace("\n", " ") for metric in metrics]
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.grid(False)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            value = float(corr.iloc[i, j])
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=4.8,
                color="white" if abs(value) > 0.6 else INK,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label(f"{method.capitalize()} " + r"$\rho$", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    ax.set_title("Trace-metric cross-correlation", fontsize=8.5)
    fig.tight_layout()
    return fig, corr


def plot_metric_vs_dpp_panel(
    units: pd.DataFrame,
    metric: str,
    *,
    exemplars: Optional[pd.DataFrame] = None,
    regions: Sequence[str] = REGION_ORDER,
    score_column: str = DPP_COLUMN,
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.0,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Relationship between one trace metric and the DPP score, per region.

    Shows what the single headline score does and does not capture: a metric
    that tracks DPP closely is redundant with it, one that is orthogonal is
    describing a different property of the same trace.
    """
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(
        1,
        len(regions),
        figsize=(figure_width_in, figure_height_in),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    rows = []
    for ax, region in zip(axes, regions):
        region_units = units.loc[units["region"].astype(str) == str(region)]
        x = pd.to_numeric(region_units[score_column], errors="coerce")
        y = pd.to_numeric(region_units[metric], errors="coerce")
        mask = x.notna() & y.notna()
        ax.scatter(
            x[mask],
            y[mask],
            s=5,
            color=NEUTRAL_FILL,
            edgecolor=NEUTRAL_EDGE,
            linewidth=0.2,
            alpha=0.85,
        )
        rho, p_value = (np.nan, np.nan)
        if int(mask.sum()) > 3:
            rho, p_value = scipy_stats.spearmanr(x[mask], y[mask])
        if exemplars is not None:
            region_exemplars = exemplars.loc[exemplars["region"].astype(str) == str(region)]
            for _, unit in region_exemplars.iterrows():
                if metric not in unit:
                    continue
                style = str(unit["style"])
                ax.scatter(
                    [float(unit[score_column])],
                    [float(unit[metric])],
                    s=34,
                    color=EXEMPLAR_STYLE_COLORS.get(style, INK),
                    marker=EXEMPLAR_STYLE_MARKERS.get(style, "o"),
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=6,
                )
        ax.set_title(
            f"{region_label(region)}\n" + r"$\rho$ = " + f"{rho:.2f}",
            fontsize=7.6,
            pad=3,
        )
        nice_axis(ax, y_ticks=4)
        rows.append(
            {
                "region": region_label(region),
                "metric": metric,
                "n_units": int(mask.sum()),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            }
        )
    axes[0].set_ylabel(metric_axis_label(metric), fontsize=7)
    fig.supxlabel(DPP_AXIS_LABEL, fontsize=7.8)
    fig.tight_layout()

    table = pd.DataFrame(rows)
    table["p_adj"] = adjust_pvalues(table["p_value"].to_numpy(), "fdr_bh")
    return fig, table


def build_exemplar_metric_rank_table(
    units: pd.DataFrame,
    exemplars: pd.DataFrame,
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Within-region percentile rank of each example unit on every metric."""
    ranked = units.copy()
    for metric in metrics:
        ranked[f"pct__{metric}"] = ranked.groupby("region")[metric].rank(pct=True)
    merged = exemplars.loc[:, ["style", "region", "uuid"]].merge(
        ranked.loc[:, ["region", "uuid"] + [f"pct__{metric}" for metric in metrics]],
        on=["region", "uuid"],
        how="left",
    )
    merged["region"] = merged["region"].map(region_label)
    merged["style"] = merged["style"].map(lambda s: EXEMPLAR_STYLE_LABELS.get(str(s), str(s)))
    return merged.rename(
        columns={
            f"pct__{metric}": metric_axis_label(metric).replace("\n", " ") for metric in metrics
        }
    ).rename(columns={"style": "example", "region": "Region", "uuid": "Unit"})


def legend_handles_for_exemplars() -> list:
    """Shared legend handles for the example-unit markers."""
    return [
        Line2D([0], [0], color=NEUTRAL_EDGE, linewidth=1.2, label="Region median"),
    ] + [
        Line2D(
            [0],
            [0],
            color=EXEMPLAR_STYLE_COLORS[style],
            linestyle="--",
            linewidth=1.1,
            marker=EXEMPLAR_STYLE_MARKERS[style],
            markersize=3.6,
            label=f"{EXEMPLAR_STYLE_LABELS[style]} example",
        )
        for style in ("phasic", "tonic")
    ]
