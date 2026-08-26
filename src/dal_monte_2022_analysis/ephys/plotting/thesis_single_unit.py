"""Thesis chapter figures for single-unit fixation responses.

Every public function returns a matplotlib figure laid out as **one row of four
region panels**, the layout the chapter uses throughout, and takes already-loaded
tables rather than reading from disk, so notebooks stay a thin display layer.

Figure inventory:

``plot_unit_yield_panel``
    Units recorded per region and the fraction differentiating at least one
    fixation-category pair.
``plot_pair_selectivity_venn_panel``
    Fixed-geometry three-set Venn of which category pairs each unit separates.
``plot_example_unit_panel``
    Raster over mean+-SEM firing rate for one example unit per region, with the
    three analysis-window bars and an optional Dominant-Peak Prominence
    schematic on the first column.
``plot_dpp_distribution_panel``
    Per-region DPP distribution with the example units marked.
``plot_condition_cv_panel``
    Coefficient of variation of the mean firing-rate trace, by fixation category.
``plot_preferred_condition_panel``
    Which fixation category each modulated unit fires most for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch

from dal_monte_2022_analysis.core.stats import (
    significance_stars,
    wilson_score_interval,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_peakiness import (
    DominantPeakDecomposition,
)
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    ANALYSIS_WINDOWS_MS,
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    DPP_ABBREV,
    DPP_AXIS_LABEL,
    DPP_FORMULA,
    EXEMPLAR_STYLE_COLORS,
    EXEMPLAR_STYLE_LABELS,
    EXEMPLAR_STYLE_MARKERS,
    INK,
    MUTED_INK,
    NEUTRAL_EDGE,
    NEUTRAL_FILL,
    PAIR_LABELS,
    PAIR_ORDER,
    REGION_ORDER,
    add_analysis_window_bars,
    add_significance_bracket,
    condition_legend_handles,
    nice_axis,
    ordinal,
    region_label,
    window_legend_handles,
)


@dataclass(frozen=True)
class ExampleUnitPanelSpec:
    """One column of an example-unit figure."""

    region: str
    unit_uuid: str
    date: str
    bin_centers_s: np.ndarray
    payloads: Sequence[Mapping]
    dpp_score: float
    dpp_percentile: float
    #: Only the schematic column needs this.
    decomposition: Optional[DominantPeakDecomposition] = None


# --------------------------------------------------------------------------- #
# 1 - Unit yield and modulated fraction                                        #
# --------------------------------------------------------------------------- #


def plot_unit_yield_panel(
    units: pd.DataFrame,
    *,
    regions: Sequence[str] = REGION_ORDER,
    selective_column: str = "is_selective",
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.3,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Recorded units and the fraction that differentiate fixation categories.

    Both quantities live in one row: the bar height is the modulated *fraction*
    with a Wilson 95% interval, and the recorded total is printed above each bar
    rather than given its own panel, because a second bar chart of raw counts
    would repeat information the annotation already carries.
    """
    rows = []
    for region in regions:
        region_units = units.loc[units["region"].astype(str) == str(region)]
        n_total = int(len(region_units))
        n_selective = int(region_units[selective_column].astype(bool).sum())
        low, high = wilson_score_interval(n_selective, n_total)
        rows.append(
            {
                "region": region,
                "region_label": region_label(region),
                "n_units": n_total,
                "n_selective": n_selective,
                "fraction": (n_selective / n_total) if n_total else np.nan,
                "ci_low": low,
                "ci_high": high,
                "n_sessions": int(region_units["date"].nunique()),
            }
        )
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in))
    x = np.arange(len(table))
    ax.bar(
        x,
        table["fraction"],
        width=0.6,
        color=NEUTRAL_FILL,
        edgecolor=NEUTRAL_EDGE,
        linewidth=0.9,
    )
    ax.errorbar(
        x,
        table["fraction"],
        yerr=[
            table["fraction"] - table["ci_low"],
            table["ci_high"] - table["fraction"],
        ],
        fmt="none",
        ecolor=NEUTRAL_EDGE,
        elinewidth=1.0,
        capsize=2.5,
    )
    for xi, row in zip(x, table.itertuples()):
        ax.text(
            xi,
            row.ci_high + 0.035,
            f"{row.n_selective}/{row.n_units}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(table["region_label"])
    ax.set_ylabel("Fraction of units")
    ax.set_ylim(0, 0.78)
    ax.set_title(
        "Units differentiating at least one fixation-category pair "
        "(FDR corrected; bars 95% Wilson CI)",
        fontsize=8,
    )
    nice_axis(ax)
    fig.tight_layout()
    return fig, table


# --------------------------------------------------------------------------- #
# 2 - Fixation-pair selectivity Venn                                           #
# --------------------------------------------------------------------------- #

# Fixed unit-circle geometry, shared by all four region panels. Area-scaled
# circles (matplotlib-venn) would give each region a different shape and defeat
# the point of putting the regions side by side.
_VENN_CENTERS = ((-0.34, 0.20), (0.34, 0.20), (0.0, -0.38))
_VENN_RADIUS = 0.62
_VENN_LABEL_POSITIONS = {
    "100": (-0.66, 0.42),
    "010": (0.66, 0.42),
    "001": (0.0, -0.72),
    "110": (0.0, 0.50),
    "101": (-0.40, -0.20),
    "011": (0.40, -0.20),
    "111": (0.0, 0.02),
}
#: Pair identity is carried by a shared color key below the row rather than by
#: per-circle text. With four panels side by side the set labels of adjacent
#: regions overlap, and repeating the same three labels four times is noise.
_VENN_FILL_COLORS = ("#4f81b8", "#c0554a", "#6f9e52")


def compute_pair_selectivity_membership(
    pair_selectivity: pd.DataFrame,
    *,
    regions: Sequence[str] = REGION_ORDER,
    pairs: Sequence[str] = PAIR_ORDER,
    selective_column: str = "is_selective_pair_corrected",
) -> pd.DataFrame:
    """Per region, count units in each of the seven Venn subsets.

    ``subset`` is a 3-bit string ordered as ``pairs``; ``'101'`` means the unit
    separates the first and third pair but not the second.
    """
    frame = pair_selectivity.copy()
    frame["region"] = frame["region"].astype(str)
    significant = frame.loc[frame[selective_column].astype(bool)]

    rows = []
    for region in regions:
        region_sig = significant.loc[significant["region"] == region]
        members = {
            pair: set(region_sig.loc[region_sig["pair_label"] == pair, "unit_key"].astype(str))
            for pair in pairs
        }
        union = set().union(*members.values()) if members else set()
        for unit_key in union:
            bits = "".join("1" if unit_key in members[pair] else "0" for pair in pairs)
            rows.append({"region": region, "unit_key": unit_key, "subset": bits})
    membership = pd.DataFrame(rows, columns=["region", "unit_key", "subset"])
    return membership


def plot_pair_selectivity_venn_panel(
    membership: pd.DataFrame,
    *,
    regions: Sequence[str] = REGION_ORDER,
    pairs: Sequence[str] = PAIR_ORDER,
    region_totals: Optional[Mapping[str, int]] = None,
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.15,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Fixed-geometry three-set Venn of fixation-pair selectivity, one per region."""
    counts_rows = []
    for region in regions:
        region_membership = membership.loc[membership["region"].astype(str) == str(region)]
        counts = region_membership["subset"].value_counts().to_dict()
        entry = {"region": region, "region_label": region_label(region)}
        for bits in _VENN_LABEL_POSITIONS:
            entry[bits] = int(counts.get(bits, 0))
        entry["n_selective"] = int(len(region_membership))
        if region_totals is not None:
            entry["n_units"] = int(region_totals.get(region, 0))
        counts_rows.append(entry)
    counts_table = pd.DataFrame(counts_rows)

    fig, axes = plt.subplots(
        1,
        len(regions),
        figsize=(figure_width_in, figure_height_in),
    )
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, counts_table.iterrows()):
        for center, color in zip(_VENN_CENTERS, _VENN_FILL_COLORS):
            ax.add_patch(
                Circle(
                    center,
                    _VENN_RADIUS,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.30,
                    linewidth=1.0,
                    zorder=2,
                )
            )
        for bits, (x, y) in _VENN_LABEL_POSITIONS.items():
            value = int(row[bits])
            ax.text(
                x,
                y,
                str(value),
                ha="center",
                va="center",
                fontsize=7.6,
                color=INK if value else MUTED_INK,
                fontweight="bold" if bits == "111" else "normal",
                zorder=4,
            )
        title = f"{row['region_label']}"
        if "n_units" in counts_table.columns:
            title += f"\n{int(row['n_selective'])}/{int(row['n_units'])} units"
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 0.92)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.legend(
        handles=[
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.45,
                label=PAIR_LABELS[pair].replace("\n", " "),
            )
            for pair, color in zip(pairs, _VENN_FILL_COLORS)
        ],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        fontsize=6.8,
        handlelength=1.3,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig, counts_table


# --------------------------------------------------------------------------- #
# 3 - Example units: raster over PSTH                                          #
# --------------------------------------------------------------------------- #


def _draw_raster(
    ax,
    payloads: Sequence[Mapping],
    *,
    display_window_s: tuple[float, float],
    max_trials_per_condition: int = 70,
    random_seed: int = 42,
) -> int:
    """Spike raster with trials blocked and coloured by fixation category.

    Trials are subsampled to ``max_trials_per_condition``. At the several
    hundred trials these units actually have, every tick overlaps its
    neighbours and the raster prints as a solid block, which shows less than a
    sparse sample of the same data. Returns the number of rows drawn.
    """
    rng = np.random.default_rng(random_seed)
    offset = 0
    boundaries = []
    for payload in payloads:
        spike_rows = list(payload.get("spike_rows", []))
        if len(spike_rows) > max_trials_per_condition:
            keep = np.sort(
                rng.choice(len(spike_rows), size=max_trials_per_condition, replace=False)
            )
            spike_rows = [spike_rows[index] for index in keep]
        color = CONDITION_COLORS.get(str(payload["key"]), str(payload.get("color", "#444444")))
        for spikes in spike_rows:
            spikes = np.asarray(spikes, dtype=float).reshape(-1)
            spikes = spikes[(spikes >= display_window_s[0]) & (spikes <= display_window_s[1])]
            if spikes.size:
                ax.vlines(
                    spikes,
                    offset + 0.08,
                    offset + 0.92,
                    color=color,
                    linewidth=0.42,
                    alpha=0.95,
                )
            offset += 1
        boundaries.append(offset)
    ax.set_xlim(*display_window_s)
    ax.set_ylim(0, max(offset, 1))
    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="#9a9a9a", linewidth=0.45)
    ax.axvline(0.0, color=INK, linestyle="--", linewidth=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    return offset


def _draw_rate(
    ax,
    payloads: Sequence[Mapping],
    *,
    display_window_s: tuple[float, float],
    highlight_condition: Optional[str] = None,
) -> None:
    """Mean +- SEM firing rate for each fixation category."""
    for payload in payloads:
        key = str(payload["key"])
        centers = np.asarray(payload.get("trace_bin_centers"), dtype=float).reshape(-1)
        mean_hz = np.asarray(payload.get("mean_hz"), dtype=float).reshape(-1)
        sem_hz = np.asarray(payload.get("sem_hz"), dtype=float).reshape(-1)
        if centers.size != mean_hz.size or not np.any(np.isfinite(mean_hz)):
            continue
        mask = (centers >= display_window_s[0]) & (centers <= display_window_s[1])
        color = CONDITION_COLORS.get(key, str(payload.get("color", "#444444")))
        faded = highlight_condition is not None and key != highlight_condition
        ax.fill_between(
            centers[mask],
            (mean_hz - sem_hz)[mask],
            (mean_hz + sem_hz)[mask],
            color=color,
            alpha=0.10 if faded else 0.22,
            linewidth=0,
        )
        ax.plot(
            centers[mask],
            mean_hz[mask],
            color=color,
            linewidth=0.85 if faded else 1.5,
            alpha=0.45 if faded else 1.0,
            zorder=3 if faded else 5,
        )
    ax.axvline(0.0, color=INK, linestyle="--", linewidth=0.7, zorder=2)
    ax.set_xlim(*display_window_s)


def _annotate_dpp_schematic(
    ax,
    decomposition: DominantPeakDecomposition,
) -> None:
    """Draw how P1, P2 and the exclusion window produce the DPP score.

    Prominences are computed on the rate-normalized trace, so each normalized
    prominence is multiplied back by the normalization scale before being drawn
    on a Hz axis: the vertical spans then sit exactly where the peaks are while
    the printed numbers stay the ones the score was computed from.
    """
    if decomposition.primary_index is None:
        return

    scale = float(decomposition.normalization_scale)
    centers_s = decomposition.centers_ms / 1000.0
    primary_t = float(centers_s[decomposition.primary_index])
    primary_top = float(decomposition.values_hz[decomposition.primary_index])
    primary_base = float(decomposition.primary_reference_norm * scale)

    exclusion_s = float(decomposition.competition_exclusion_window_ms) / 1000.0
    ax.axvspan(
        primary_t - exclusion_s,
        primary_t + exclusion_s,
        color="#9a9a9a",
        alpha=0.14,
        linewidth=0,
        zorder=1,
    )
    x_low, x_high = ax.get_xlim()

    def _draw_span(t_s, top, base, symbol, arrow_color):
        # Label on the side with room: a peak near the right edge would push its
        # label off the axes, and one near the left edge clips against the
        # y-axis.
        side = -1 if t_s > x_low + 0.72 * (x_high - x_low) else 1
        ax.plot(
            [t_s - 0.135, t_s + 0.135],
            [base, base],
            color=arrow_color,
            linewidth=0.7,
            linestyle=(0, (3, 2)),
            zorder=6,
        )
        ax.add_patch(
            FancyArrowPatch(
                (t_s, base),
                (t_s, top),
                arrowstyle="<->",
                mutation_scale=5,
                linewidth=0.9,
                color=arrow_color,
                shrinkA=0,
                shrinkB=0,
                zorder=7,
            )
        )
        ax.annotate(
            symbol,
            xy=(t_s, 0.5 * (base + top)),
            xytext=(6 if side > 0 else -6, 0),
            textcoords="offset points",
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=7.5,
            color=arrow_color,
            zorder=9,
        )

    _draw_span(primary_t, primary_top, primary_base, "$P_1$", INK)

    lines = [
        f"$P_1$ = {decomposition.primary_prominence:.2f}",
    ]
    if decomposition.secondary_index is not None:
        secondary_t = float(centers_s[decomposition.secondary_index])
        secondary_top = float(decomposition.values_hz[decomposition.secondary_index])
        secondary_base = float(decomposition.secondary_reference_norm * scale)
        _draw_span(secondary_t, secondary_top, secondary_base, "$P_2$", MUTED_INK)
        lines.append(f"$P_2$ = {decomposition.secondary_prominence:.2f}")
    lines.append(f"{DPP_ABBREV} = {decomposition.dominant_peak_prominence:.2f}")

    ax.text(
        0.985,
        0.975,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=INK,
        linespacing=1.35,
        zorder=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cccccc",
              "linewidth": 0.5, "alpha": 0.92},
    )


def plot_example_unit_panel(
    specs: Sequence[ExampleUnitPanelSpec],
    *,
    display_window_s: tuple[float, float] = (-1.0, 1.0),
    schematic_index: Optional[int] = None,
    schematic_condition: str = "face_interactive",
    title: Optional[str] = None,
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.9,
    raster_height_ratio: float = 0.85,
    show_window_legend: bool = False,
    max_raster_trials_per_condition: int = 70,
) -> plt.Figure:
    """One row of example units: raster above mean+-SEM rate, per region.

    ``schematic_index`` selects the column that additionally carries the
    Dominant-Peak Prominence construction; that column is given extra width so
    the annotations do not crowd the trace.
    """
    n_columns = len(specs)
    width_ratios = [1.0] * n_columns
    if schematic_index is not None and 0 <= schematic_index < n_columns:
        width_ratios[schematic_index] = 1.42

    fig = plt.figure(figsize=(figure_width_in, figure_height_in))
    outer = fig.add_gridspec(
        1,
        n_columns,
        width_ratios=width_ratios,
        wspace=0.30,
        left=0.075,
        right=0.995,
        top=0.86 if title else 0.92,
        bottom=0.175,
    )

    for index, spec in enumerate(specs):
        inner = outer[0, index].subgridspec(
            2,
            1,
            height_ratios=[raster_height_ratio, 1.0],
            hspace=0.10,
        )
        ax_raster = fig.add_subplot(inner[0])
        ax_rate = fig.add_subplot(inner[1])

        is_schematic = schematic_index is not None and index == schematic_index
        n_rows = _draw_raster(
            ax_raster,
            spec.payloads,
            display_window_s=display_window_s,
            max_trials_per_condition=max_raster_trials_per_condition,
        )
        _draw_rate(
            ax_rate,
            spec.payloads,
            display_window_s=display_window_s,
            highlight_condition=schematic_condition if is_schematic else None,
        )

        ax_raster.set_title(
            f"{region_label(spec.region)} unit {spec.unit_uuid}\n"
            f"{DPP_ABBREV} = {spec.dpp_score:.2f} ({ordinal(spec.dpp_percentile * 100)} pct)",
            fontsize=7.4,
            pad=3,
        )
        if index == 0:
            ax_raster.set_ylabel(f"Trials\n({n_rows} shown)", fontsize=7, linespacing=1.1)
            ax_rate.set_ylabel("Firing rate (Hz)", fontsize=7.5)

        if is_schematic and spec.decomposition is not None:
            _annotate_dpp_schematic(ax_rate, spec.decomposition)

        # Freeze the ticks chosen for the data range before the window bars
        # extend the axis downward; otherwise matplotlib re-ticks the enlarged
        # range and prints firing-rate labels in the reserved annotation strip.
        nice_axis(ax_rate, y_ticks=4)
        data_low, data_high = ax_rate.get_ylim()
        data_ticks = [
            tick for tick in ax_rate.get_yticks() if data_low <= tick <= data_high
        ]
        add_analysis_window_bars(ax_rate, time_scale=1e-3, label=False)
        ax_rate.set_yticks(data_ticks)
        ax_rate.spines["left"].set_bounds(min(data_ticks), max(data_ticks))
        ax_rate.set_xlabel("Time from fixation onset (s)", fontsize=7.5)
        ax_rate.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_rate.tick_params(length=2.5, pad=1.5)

    handles = condition_legend_handles(short=True)
    if show_window_legend:
        handles = handles + window_legend_handles()
    if schematic_index is not None:
        handles = handles + [
            Patch(
                facecolor="#9a9a9a",
                edgecolor="none",
                alpha=0.30,
                label=r"$P_2$ search excludes $\pm$250 ms around $P_1$",
            )
        ]
    fig.legend(
        handles=handles,
        ncol=min(len(handles), 4),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.115 if len(handles) > 4 else -0.045),
        fontsize=6.6,
        columnspacing=1.1,
        handlelength=1.5,
    )
    if title:
        fig.suptitle(title, fontsize=9, y=0.995)
    return fig


# --------------------------------------------------------------------------- #
# 4 - DPP distribution                                                         #
# --------------------------------------------------------------------------- #


def plot_dpp_distribution_panel(
    units: pd.DataFrame,
    exemplars: pd.DataFrame,
    *,
    score_column: str = "peakiness_score",
    regions: Sequence[str] = REGION_ORDER,
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.5,
    upper_quantile: float = 0.995,
) -> tuple[plt.Figure, pd.DataFrame]:
    """DPP distribution per region with the example units marked.

    Drawn as a histogram rather than a violin: a violin hides where individual
    units sit, and the point of the panel is that the labelled examples come
    from opposite ends of the same distribution. Exemplar labels are placed
    inside the axes, staggered by style, because four panels of out-of-axis
    annotation collide with the titles.
    """
    fig, axes = plt.subplots(
        1,
        len(regions),
        figsize=(figure_width_in, figure_height_in),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    all_values = pd.to_numeric(units[score_column], errors="coerce").dropna().to_numpy()
    # The score has a long thin right tail; letting it set the axis compresses
    # the body of the distribution into a few pixels.
    x_max = float(np.quantile(all_values, upper_quantile))
    edges = np.histogram_bin_edges(all_values[all_values <= x_max], bins=30)

    summary_rows = []
    for ax, region in zip(axes, regions):
        region_units = units.loc[units["region"].astype(str) == str(region)]
        values = pd.to_numeric(region_units[score_column], errors="coerce").dropna().to_numpy()
        ax.hist(
            values,
            bins=edges,
            color=NEUTRAL_FILL,
            edgecolor=NEUTRAL_EDGE,
            linewidth=0.4,
        )
        median = float(np.median(values)) if values.size else np.nan
        ax.axvline(median, color=NEUTRAL_EDGE, linewidth=1.2, zorder=4)

        region_exemplars = exemplars.loc[exemplars["region"].astype(str) == str(region)]
        style_y = {"phasic": 0.90, "tonic": 0.68}
        for _, unit in region_exemplars.iterrows():
            style = str(unit["style"])
            color = EXEMPLAR_STYLE_COLORS.get(style, INK)
            value = float(unit[score_column])
            y_frac = style_y.get(style, 0.80)
            ax.axvline(value, color=color, linestyle="--", linewidth=1.2, zorder=6)
            ax.plot(
                [value],
                [y_frac],
                transform=ax.get_xaxis_transform(),
                marker=EXEMPLAR_STYLE_MARKERS.get(style, "o"),
                markersize=4.0,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=8,
            )
            # Label on whichever side has room, so the two exemplar labels in a
            # narrow panel never overlap each other.
            to_right = value < 0.55 * x_max
            ax.annotate(
                f"{unit['uuid']} ({ordinal(float(unit['dpp_percentile']) * 100)})",
                xy=(value, y_frac),
                xycoords=ax.get_xaxis_transform(),
                xytext=(5 if to_right else -5, 0),
                textcoords="offset points",
                ha="left" if to_right else "right",
                va="center",
                fontsize=5.8,
                color=color,
                zorder=8,
            )
        summary_rows.append(
            {
                "region": region,
                "region_label": region_label(region),
                "n_units": int(values.size),
                "median": median,
                "q25": float(np.quantile(values, 0.25)) if values.size else np.nan,
                "q75": float(np.quantile(values, 0.75)) if values.size else np.nan,
                "n_above_axis_max": int(np.count_nonzero(values > x_max)),
            }
        )
        ax.set_title(region_label(region), fontsize=8.5, pad=4)
        nice_axis(ax, y_ticks=3)
    axes[0].set_ylabel("Units", fontsize=7.5)
    axes[0].set_xlim(float(edges[0]), x_max)

    handles = [
        Line2D([0], [0], color=NEUTRAL_EDGE, linewidth=1.2, label="Region median"),
    ] + [
        Line2D(
            [0],
            [0],
            color=EXEMPLAR_STYLE_COLORS[style],
            linestyle="--",
            linewidth=1.2,
            marker=EXEMPLAR_STYLE_MARKERS[style],
            markersize=4.0,
            label=f"{EXEMPLAR_STYLE_LABELS[style]} example",
        )
        for style in ("phasic", "tonic")
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=6.8,
    )
    fig.supxlabel(DPP_AXIS_LABEL, fontsize=7.8)
    fig.tight_layout()
    return fig, pd.DataFrame(summary_rows)


# --------------------------------------------------------------------------- #
# 5 - Coefficient of variation by fixation category                            #
# --------------------------------------------------------------------------- #


def plot_condition_cv_panel(
    variability: pd.DataFrame,
    stats_table: pd.DataFrame,
    *,
    regions: Sequence[str] = REGION_ORDER,
    conditions: Sequence[str] = CONDITION_ORDER,
    metric_key: str = "cv",
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.7,
    alpha: float = 0.05,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Per-region CV of the mean firing-rate trace, split by fixation category.

    Significance brackets come from the stored within-region paired tests rather
    than being recomputed here, so the figure and the analysis table can never
    disagree.
    """
    fig, axes = plt.subplots(
        1,
        len(regions),
        figsize=(figure_width_in, figure_height_in),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    summary_rows = []
    panel_datasets: list[np.ndarray] = []
    for ax, region in zip(axes, regions):
        region_frame = variability.loc[variability["region"].astype(str) == str(region)]
        datasets = []
        for condition in conditions:
            values = (
                pd.to_numeric(region_frame[f"{condition}_cv"], errors="coerce").dropna().to_numpy()
            )
            datasets.append(values)
            summary_rows.append(
                {
                    "region": region,
                    "region_label": region_label(region),
                    "condition": condition,
                    "n_units": int(values.size),
                    "median_cv": float(np.median(values)) if values.size else np.nan,
                    "mean_cv": float(np.mean(values)) if values.size else np.nan,
                }
            )
        panel_datasets.extend(datasets)
        positions = np.arange(len(conditions)) + 1
        parts = ax.violinplot(
            datasets,
            positions=positions,
            widths=0.78,
            showextrema=False,
            showmedians=False,
        )
        for body, condition in zip(parts["bodies"], conditions):
            body.set_facecolor(CONDITION_COLORS[condition])
            body.set_edgecolor(INK)
            body.set_linewidth(0.7)
            body.set_alpha(0.85)
        for position, values in zip(positions, datasets):
            if not values.size:
                continue
            median = float(np.median(values))
            ax.plot(
                [position - 0.24, position + 0.24],
                [median, median],
                color=INK,
                linewidth=1.4,
                solid_capstyle="butt",
                zorder=5,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [CONDITION_SHORT_LABELS[condition] for condition in conditions],
            rotation=22,
            ha="right",
        )
        ax.set_title(region_label(region), fontsize=8.5)
        nice_axis(ax, y_ticks=4)

    # Brackets are placed after every panel is drawn: with a shared y-axis the
    # last ``set_ylim`` wins, so per-panel placement would push the earlier
    # regions' brackets above the final limit and into their titles.
    y_max = max((float(np.max(v)) for v in panel_datasets if v.size), default=1.0)
    step = y_max * 0.11
    # Only the two comparisons the chapter argues about get a bracket;
    # annotating all three crowds the panel without adding a claim.
    bracket_pairs = [
        ("face_interactive", "face_non_interactive", 1, 2),
        ("face_interactive", "object", 1, 3),
    ]
    for ax, region in zip(axes, regions):
        ax.set_ylim(0, y_max + step * 3.0)
        region_stats = stats_table.loc[
            (stats_table["region"].astype(str) == str(region))
            & (stats_table["metric_key"].astype(str) == metric_key)
        ]
        for level, (cond_a, cond_b, x_a, x_b) in enumerate(bracket_pairs):
            match = region_stats.loc[
                (region_stats["condition_a"] == cond_a) & (region_stats["condition_b"] == cond_b)
            ]
            if match.empty:
                continue
            p_adj = float(match.iloc[0]["p_value_adjusted"])
            add_significance_bracket(
                ax,
                x_a,
                x_b,
                y_max + step * (level * 1.05 + 0.5),
                significance_stars(p_adj, alpha=alpha),
                fontsize=6.6,
            )

    axes[0].set_ylabel("CV of mean firing rate", fontsize=7.5)
    fig.tight_layout()
    return fig, pd.DataFrame(summary_rows)


# --------------------------------------------------------------------------- #
# 6 - Preferred fixation category                                              #
# --------------------------------------------------------------------------- #


def plot_preferred_condition_panel(
    units: pd.DataFrame,
    *,
    regions: Sequence[str] = REGION_ORDER,
    conditions: Sequence[str] = CONDITION_ORDER,
    preference_column: str = "dominant_condition",
    figure_width_in: float = 7.2,
    figure_height_in: float = 2.5,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Fraction of modulated units preferring each fixation category, by region."""
    rows = []
    for region in regions:
        region_units = units.loc[units["region"].astype(str) == str(region)]
        n_total = int(len(region_units))
        for condition in conditions:
            k = int((region_units[preference_column].astype(str) == condition).sum())
            low, high = wilson_score_interval(k, n_total)
            rows.append(
                {
                    "region": region,
                    "region_label": region_label(region),
                    "condition": condition,
                    "k": k,
                    "n": n_total,
                    "fraction": (k / n_total) if n_total else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1,
        len(regions),
        figsize=(figure_width_in, figure_height_in),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        region_table = table.loc[table["region"] == region].set_index("condition").loc[list(conditions)]
        x = np.arange(len(conditions))
        ax.bar(
            x,
            region_table["fraction"],
            width=0.66,
            color=[CONDITION_COLORS[condition] for condition in conditions],
            edgecolor=INK,
            linewidth=0.7,
        )
        ax.errorbar(
            x,
            region_table["fraction"],
            yerr=[
                region_table["fraction"] - region_table["ci_low"],
                region_table["ci_high"] - region_table["fraction"],
            ],
            fmt="none",
            ecolor=INK,
            elinewidth=0.9,
            capsize=2.0,
        )
        ax.axhline(1 / 3, color=MUTED_INK, linestyle="--", linewidth=0.8, zorder=1)
        for xi, row in zip(x, region_table.itertuples()):
            ax.text(xi, 0.018, str(int(row.k)), ha="center", va="bottom", fontsize=6, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_SHORT_LABELS[condition] for condition in conditions],
            rotation=22,
            ha="right",
        )
        ax.set_title(f"{region_label(region)}\nn = {int(region_table['n'].iloc[0])}", fontsize=8.2)
        ax.set_ylim(0, 0.72)
        nice_axis(ax, y_ticks=4)
    axes[0].set_ylabel("Fraction of modulated units", fontsize=7.5)
    axes[-1].text(
        len(conditions) - 0.45,
        1 / 3 + 0.012,
        "chance",
        fontsize=6,
        color=MUTED_INK,
        ha="right",
    )
    fig.tight_layout()
    return fig, table
