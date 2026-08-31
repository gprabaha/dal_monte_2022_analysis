"""Figures for the fixation population PC subspace notebooks.

Every panel here is built to be read next to the others, so region order,
condition palette and typography all come from
:mod:`dal_monte_2022_analysis.ephys.plotting.thesis_common` rather than being
respecified per figure.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from dal_monte_2022_analysis.ephys.analysis.fixation_population_pc_subspace import (
    CONDITION_ORDER,
    PopulationPCAFit,
    RegionPopulation,
    n_components_for_variance,
    principal_angle_metrics,
)
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    REGION_COLORS,
    ordered_regions,
    region_label,
)


THRESHOLD_COLOR = "#c0392b"
NULL_COLOR = "#9aa5b1"


def _regions(frame_or_mapping) -> list[str]:
    if isinstance(frame_or_mapping, pd.DataFrame):
        observed = frame_or_mapping["region"].astype(str).unique()
    else:
        observed = list(frame_or_mapping)
    return ordered_regions(observed)


def _condition_style(condition: str) -> dict:
    return {
        "color": CONDITION_COLORS.get(str(condition), MUTED_INK),
        "label": CONDITION_SHORT_LABELS.get(str(condition), str(condition)),
    }


# --------------------------------------------------------------------------- #
# Dimensionality
# --------------------------------------------------------------------------- #


def plot_cumulative_variance(
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    threshold: float = 0.95,
    shared_n_components: Optional[int] = None,
    max_components_shown: int = 60,
    figsize: tuple[float, float] = (7.4, 3.4),
) -> plt.Figure:
    """Cumulative variance per region, with the shared retained dimension marked.

    The left panel is the one that justifies the retained dimension: the
    horizontal threshold and the vertical shared-dimension line have to
    intersect at or below every region's curve, which is the claim "N PCs reach
    the threshold in all regions" drawn rather than asserted.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    regions = _regions(fits_by_region)

    ax = axes[0]
    for region in regions:
        fit = fits_by_region[region]["concatenated"]
        cumulative = np.asarray(fit.cumulative_explained_variance_ratio, dtype=float)
        x = np.arange(1, cumulative.size + 1)
        keep = x <= max_components_shown
        ax.plot(
            x[keep],
            cumulative[keep],
            color=REGION_COLORS.get(region, INK),
            label=f"{region_label(region)} (n={fit.components.shape[1]})",
        )
        needed = n_components_for_variance(cumulative, threshold)
        ax.plot([needed], [cumulative[needed - 1]], marker="o", ms=4, color=REGION_COLORS.get(region, INK))

    ax.axhline(threshold, color=THRESHOLD_COLOR, lw=0.9, ls="--")
    ax.annotate(
        f"{threshold:.0%}",
        xy=(1, threshold),
        xytext=(3, 3),
        textcoords="offset points",
        ha="left",
        color=THRESHOLD_COLOR,
        fontsize=7,
    )
    if shared_n_components is not None:
        ax.axvline(shared_n_components, color=THRESHOLD_COLOR, lw=0.9, ls=":")
        ax.annotate(
            f"{shared_n_components} PCs",
            xy=(shared_n_components, 0.42),
            xytext=(5, 0),
            textcoords="offset points",
            color=THRESHOLD_COLOR,
            fontsize=7,
            rotation=90,
            va="center",
        )
    ax.set_xlabel("Principal components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_xlim(1, max_components_shown)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=6.5, bbox_to_anchor=(1.0, 0.02))
    ax.set_title("Concatenated-condition fit", loc="left")

    ax = axes[1]
    width = 0.8 / max(len(regions), 1)
    thresholds = (0.80, 0.90, 0.95, 0.99)
    for index, region in enumerate(regions):
        cumulative = fits_by_region[region]["concatenated"].cumulative_explained_variance_ratio
        counts = [n_components_for_variance(cumulative, value) for value in thresholds]
        ax.bar(
            np.arange(len(thresholds)) + index * width - 0.4 + width / 2,
            counts,
            width=width,
            color=REGION_COLORS.get(region, INK),
            label=region_label(region),
        )
        for position, count in zip(np.arange(len(thresholds)) + index * width - 0.4 + width / 2, counts):
            ax.annotate(
                str(count),
                xy=(position, count),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                color=MUTED_INK,
            )
    if shared_n_components is not None:
        ax.axhline(shared_n_components, color=THRESHOLD_COLOR, lw=0.9, ls=":")
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{value:.0%}" for value in thresholds])
    ax.set_xlabel("Variance threshold")
    ax.set_ylabel("PCs required")
    ax.set_title("Dimensions needed per region", loc="left")
    ax.legend(fontsize=6.5, ncol=2)
    return fig


def plot_variance_spectra(
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    max_components_shown: int = 20,
    figsize: tuple[float, float] = (7.4, 2.2),
) -> plt.Figure:
    """Per-region scree, concatenated fit against each single-condition fit."""
    regions = _regions(fits_by_region)
    fig, axes = plt.subplots(
        1, len(regions), figsize=figsize, sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        fits = fits_by_region[region]
        x = np.arange(1, max_components_shown + 1)
        for condition in CONDITION_ORDER:
            if condition not in fits:
                continue
            ratios = fits[condition].explained_variance_ratio[:max_components_shown]
            style = _condition_style(condition)
            ax.plot(x[: ratios.size], ratios, color=style["color"], lw=1.0, label=style["label"])
        shared = fits["concatenated"].explained_variance_ratio[:max_components_shown]
        ax.plot(x[: shared.size], shared, color=INK, lw=1.2, ls="--", label="Concatenated")
        ax.set_yscale("log")
        ax.set_title(region_label(region), loc="left")
        ax.set_xlabel("PC")
    axes[0].set_ylabel("Variance explained")
    axes[-1].legend(fontsize=6, loc="upper right")
    return fig


# --------------------------------------------------------------------------- #
# 3D trajectories
# --------------------------------------------------------------------------- #


def _project_to_screen(points: np.ndarray, elev: float, azim: float) -> np.ndarray:
    """Orthographic 3D -> 2D projection matching matplotlib's ``view_init``."""
    elevation = np.radians(float(elev))
    azimuth = np.radians(float(azim))
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.array(
        [
            -np.sin(elevation) * np.cos(azimuth),
            -np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation),
        ]
    )
    return np.stack([points @ right, points @ up], axis=1)


def _separation_score(screen_by_condition: Mapping[str, np.ndarray]) -> float:
    """How well one 2D view shows both condition separation and trajectory shape.

    Scored as the product of two normalised terms rather than their sum.  A sum
    lets a view win on separation alone, and the winner is then always a
    near-top-down camera that spreads the three conditions apart while
    collapsing every trajectory into an unreadable scribble.  A product
    requires both to hold: the conditions must be apart *and* each trajectory
    must stay unfolded.

    The unfolding term is normalised by each trajectory's own extent, not by
    the global extent, because these conditions sit far apart relative to how
    far each one moves -- against a global scale the term would be negligible
    and would not constrain the view at all.
    """
    conditions = list(screen_by_condition)
    global_scale = float(
        np.ptp(np.concatenate([screen_by_condition[name] for name in conditions], axis=0), axis=0).max()
    )
    if global_scale <= 1e-12:
        return -np.inf

    separation = np.inf
    for first, second in combinations(conditions, 2):
        a = screen_by_condition[first]
        b = screen_by_condition[second]
        gaps = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        separation = min(separation, float(np.min(gaps)) / global_scale)

    unfolding = []
    for name in conditions:
        points = screen_by_condition[name]
        local_scale = float(np.ptp(points, axis=0).max())
        if local_scale <= 1e-12:
            unfolding.append(0.0)
            continue
        gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(gaps, np.inf)
        # Immediate temporal neighbours are supposed to be close; only a
        # trajectory crossing back over a *distant* part of itself is folding.
        for offset in (1, 2, 3):
            index = np.arange(points.shape[0] - offset)
            gaps[index, index + offset] = np.inf
            gaps[index + offset, index] = np.inf
        unfolding.append(float(np.median(np.min(gaps, axis=1))) / local_scale)

    return float(separation) * float(np.mean(unfolding))


def optimize_view_angle(
    scores_by_condition: Mapping[str, np.ndarray],
    *,
    n_components: int = 3,
    elev_grid: Sequence[float] = tuple(range(-30, 55, 5)),
    azim_grid: Sequence[float] = tuple(range(-180, 180, 5)),
) -> tuple[float, float, float]:
    """Search view angles for the one that separates the trajectories best.

    A fixed camera angle suits at most one region: the conditions separate
    along different PC combinations in each, so a view that reads cleanly in
    BLA can collapse two trajectories on top of each other in dmPFC.  Searching
    per region makes the visualisation report the geometry rather than the
    camera.  Axes are unit-normalised first because matplotlib autoscales each
    axis independently, so the on-screen geometry follows the normalised data.
    """
    conditions = list(scores_by_condition)
    stacked = np.concatenate(
        [np.asarray(scores_by_condition[name], dtype=float)[:n_components, :].T for name in conditions],
        axis=0,
    )
    span = np.ptp(stacked, axis=0)
    span[span <= 1e-12] = 1.0
    normalized = {
        name: (np.asarray(scores_by_condition[name], dtype=float)[:n_components, :].T - stacked.mean(axis=0))
        / span
        for name in conditions
    }

    best = (-np.inf, 22.0, -58.0)
    for elev in elev_grid:
        for azim in azim_grid:
            screen = {name: _project_to_screen(values, elev, azim) for name, values in normalized.items()}
            score = _separation_score(screen)
            if score > best[0]:
                best = (score, float(elev), float(azim))
    return best[1], best[2], best[0]


def plot_pc_trajectories_3d(
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    view_angles: Optional[Mapping[str, tuple[float, float]]] = None,
    auto_view: bool = True,
    n_columns: int = 4,
    panel_size: tuple[float, float] = (2.25, 2.2),
    marker_time_s: float = 0.0,
    line_width: float = 1.9,
    start_marker_size: float = 26.0,
    onset_marker_size: float = 78.0,
) -> tuple[plt.Figure, dict[str, tuple[float, float]]]:
    """Top-3 PC trajectories per region, one panel per region.

    Trajectories are drawn in the *concatenated* basis so all three conditions
    live in one shared coordinate system; plotting each condition in its own
    basis would make the axes mean different things per curve and the visible
    distances meaningless.
    """
    regions = _regions(fits_by_region)
    n_rows = int(np.ceil(len(regions) / n_columns))
    # 3D axes plus a figure-level legend confuse constrained layout into
    # collapsing the panels, so the legend strip is reserved by hand instead.
    fig = plt.figure(figsize=(panel_size[0] * n_columns, panel_size[1] * n_rows + 0.35))

    resolved: dict[str, tuple[float, float]] = {}
    for index, region in enumerate(regions):
        fit = fits_by_region[region]["concatenated"]
        ax = fig.add_subplot(n_rows, n_columns, index + 1, projection="3d")
        scores = {
            condition: fit.scores_by_condition[condition]
            for condition in fit.conditions
            if condition in fit.scores_by_condition
        }

        if view_angles is not None and region in view_angles:
            elev, azim = view_angles[region]
        elif auto_view:
            elev, azim, _ = optimize_view_angle(scores)
        else:
            elev, azim = 22.0, -58.0
        resolved[region] = (float(elev), float(azim))

        centers = np.asarray(fit.bin_centers_s, dtype=float)
        marker_index = int(np.argmin(np.abs(centers - float(marker_time_s))))
        for condition, values in scores.items():
            style = _condition_style(condition)
            path = np.asarray(values, dtype=float)[:3, :]
            ax.plot(
                path[0], path[1], path[2],
                color=style["color"], lw=line_width, solid_capstyle="round", label=style["label"],
            )
            ax.scatter(
                *path[:, 0],
                color=style["color"], s=start_marker_size, marker="o",
                depthshade=False, edgecolors="white", linewidths=0.6,
            )
            ax.scatter(
                *path[:, marker_index],
                color=style["color"], s=onset_marker_size, marker="*",
                depthshade=False, edgecolors="white", linewidths=0.5,
            )
        ax.view_init(elev=elev, azim=azim)
        for axis, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), ("PC1", "PC2", "PC3")):
            axis.set_ticklabels([])
            axis.set_label_text(name)
        ax.tick_params(length=0, pad=-3)
        ax.xaxis.labelpad = -12
        ax.yaxis.labelpad = -12
        ax.zaxis.labelpad = -12
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_fontsize(6.5)
            axis.label.set_color(MUTED_INK)
        ax.set_title(f"{region_label(region)}  ({elev:.0f}°, {azim:.0f}°)", loc="left", fontsize=8)
        ax.grid(True, alpha=0.25)

    handles = [
        Line2D([0], [0], color=_condition_style(condition)["color"], lw=2.0, label=_condition_style(condition)["label"])
        for condition in CONDITION_ORDER
    ]
    handles.extend(
        [
            Line2D([0], [0], color=MUTED_INK, lw=0, marker="o", ms=5, label="Window start"),
            Line2D([0], [0], color=MUTED_INK, lw=0, marker="*", ms=9, label="Fixation onset"),
        ]
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.13, wspace=0.09, hspace=0.10)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(handles),
        fontsize=6.5,
        frameon=False,
    )
    return fig, resolved


def plot_pc_plane_projections(
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    planes: Sequence[tuple[int, int]] = ((0, 1), (0, 2), (1, 2)),
    figsize_scale: tuple[float, float] = (1.7, 1.7),
) -> plt.Figure:
    """Flat PC-plane views of the same trajectories, as a camera-free check.

    Any 3D panel depends on a chosen camera, and a camera can be chosen to
    flatter the data.  These planes cannot: each is a fixed pair of PCs, so a
    separation visible here is a property of the projection rather than of the
    viewing angle.  Read them alongside the 3D panels, not instead of them.
    """
    regions = _regions(fits_by_region)
    fig, axes = plt.subplots(
        len(regions),
        len(planes),
        figsize=(figsize_scale[0] * len(planes), figsize_scale[1] * len(regions)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, region in enumerate(regions):
        fit = fits_by_region[region]["concatenated"]
        centers = np.asarray(fit.bin_centers_s, dtype=float)
        onset = int(np.argmin(np.abs(centers)))
        for column, (first, second) in enumerate(planes):
            ax = axes[row, column]
            for condition in fit.conditions:
                style = _condition_style(condition)
                path = np.asarray(fit.scores_by_condition[condition], dtype=float)
                ax.plot(path[first], path[second], color=style["color"], lw=0.9, label=style["label"])
                ax.scatter(path[first, 0], path[second, 0], color=style["color"], s=8, zorder=3)
                ax.scatter(
                    path[first, onset],
                    path[second, onset],
                    color=style["color"],
                    s=20,
                    marker="*",
                    zorder=3,
                )
            ax.set_xlabel(f"PC{first + 1}")
            if column == 0:
                ax.set_ylabel(f"{region_label(region)}\nPC{second + 1}")
            else:
                ax.set_ylabel(f"PC{second + 1}")
            ax.set_aspect("equal", adjustable="datalim")
    axes[0, -1].legend(fontsize=5.8, loc="best")
    return fig


def plot_pc_timecourses(
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    n_pcs: int = 3,
    figsize_scale: tuple[float, float] = (1.85, 1.3),
) -> plt.Figure:
    """PC score against time, region by PC, in the shared concatenated basis.

    The 3D view shows the shape; this grid is what lets a reader check *when*
    the conditions diverge, and confirms that each condition's curve is aligned
    to fixation onset rather than shifted by a block offset.
    """
    regions = _regions(fits_by_region)
    fig, axes = plt.subplots(
        len(regions),
        n_pcs,
        figsize=(figsize_scale[0] * n_pcs, figsize_scale[1] * len(regions)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, region in enumerate(regions):
        fit = fits_by_region[region]["concatenated"]
        centers = np.asarray(fit.bin_centers_s, dtype=float) * 1000.0
        for column in range(n_pcs):
            ax = axes[row, column]
            for condition in fit.conditions:
                style = _condition_style(condition)
                ax.plot(
                    centers,
                    fit.scores_by_condition[condition][column],
                    color=style["color"],
                    lw=1.0,
                    label=style["label"],
                )
            ax.axvline(0.0, color=MUTED_INK, lw=0.7, ls=":")
            if row == 0:
                ax.set_title(
                    f"PC{column + 1}  ({fit.explained_variance_ratio[column]:.1%})",
                    loc="left",
                )
            if column == 0:
                ax.set_ylabel(region_label(region))
            if row == len(regions) - 1:
                ax.set_xlabel("Time from fixation onset (ms)")
    axes[0, -1].legend(fontsize=6, loc="upper right")
    return fig


# --------------------------------------------------------------------------- #
# Subspace comparison
# --------------------------------------------------------------------------- #


def plot_cross_condition_variance_curves(
    curves: pd.DataFrame,
    *,
    value_column: str = "cumulative_variance_explained",
    ceiling: Optional[pd.DataFrame] = None,
    layout: str = "region_rows",
    figsize_scale: tuple[float, float] = (1.95, 1.75),
) -> plt.Figure:
    """Cumulative variance of each condition captured by each condition's PCs.

    One row per region, one column per evaluated condition; the curve drawn in
    the evaluated condition's own colour is the within-condition reference and
    is by construction the highest possible.
    """
    regions = _regions(curves)
    conditions = [c for c in CONDITION_ORDER if c in set(curves["eval_condition"])]
    # "fit_rows" puts the fitted condition on rows and region on columns, so a
    # row reads as "these PCs, applied everywhere" -- the arrangement that makes
    # cross-region comparison of one condition's basis immediate.
    row_keys = conditions if layout == "fit_rows" else regions
    column_keys = regions if layout == "fit_rows" else conditions
    fig, axes = plt.subplots(
        len(row_keys),
        len(column_keys),
        figsize=(figsize_scale[0] * len(column_keys), figsize_scale[1] * len(row_keys)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, row_key in enumerate(row_keys):
        for column, column_key in enumerate(column_keys):
            ax = axes[row, column]
            region = column_key if layout == "fit_rows" else row_key
            if layout == "fit_rows":
                subset = curves[(curves["region"] == region) & (curves["pc_condition"] == row_key)]
                series_column, reference = "eval_condition", row_key
            else:
                subset = curves[(curves["region"] == region) & (curves["eval_condition"] == column_key)]
                series_column, reference = "pc_condition", column_key
            for name, group in subset.groupby(series_column):
                style = _condition_style(name)
                group = group.sort_values("n_components")
                ax.plot(
                    group["n_components"],
                    group[value_column],
                    color=style["color"],
                    lw=1.4 if name == reference else 1.0,
                    ls="-" if name == reference else "--",
                    label=(
                        f"variance of {style['label']}"
                        if layout == "fit_rows"
                        else f"{style['label']} PCs"
                    ),
                )
            if ceiling is not None and layout != "fit_rows":
                match = ceiling[(ceiling["region"] == region) & (ceiling["condition"] == column_key)]
                if len(match):
                    ax.axhline(
                        float(match["variance_explained_ceiling"].iloc[0]),
                        color=NULL_COLOR, lw=0.8, ls=":",
                    )
            if row == 0:
                ax.set_title(
                    region_label(column_key)
                    if layout == "fit_rows"
                    else f"Variance of {_condition_style(column_key)['label']}",
                    loc="left",
                )
            if column == 0:
                ax.set_ylabel(
                    f"{_condition_style(row_key)['label']} PCs"
                    if layout == "fit_rows"
                    else region_label(row_key)
                )
            if row == len(row_keys) - 1:
                ax.set_xlabel("PCs used")
            ax.set_ylim(0, 1.02)
    axes[0, -1].legend(fontsize=5.6, loc="lower right")
    return fig


def plot_alignment_matrix(
    table: pd.DataFrame,
    *,
    value_column: str = "alignment_index",
    null_column: Optional[str] = "alignment_null_mean",
    figsize_scale: tuple[float, float] = (1.85, 1.85),
) -> plt.Figure:
    """Source-PC by evaluated-condition alignment matrix, one panel per region.

    Annotating each cell with the random-subspace baseline underneath the
    observed value keeps the reader from over-reading a mid-range number: a
    k-dimensional subspace captures some variance by chance, and the baseline
    is what says how much.
    """
    regions = _regions(table)
    conditions = [c for c in CONDITION_ORDER if c in set(table["eval_condition"])]
    fig, axes = plt.subplots(
        1, len(regions), figsize=(figsize_scale[0] * len(regions), figsize_scale[1]), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        subset = table[table["region"] == region]
        matrix = np.full((len(conditions), len(conditions)), np.nan)
        for row_index, pc_condition in enumerate(conditions):
            for column_index, eval_condition in enumerate(conditions):
                cell = subset[
                    (subset["pc_condition"] == pc_condition)
                    & (subset["eval_condition"] == eval_condition)
                ]
                if len(cell):
                    matrix[row_index, column_index] = float(cell[value_column].iloc[0])
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
        for row_index in range(len(conditions)):
            for column_index in range(len(conditions)):
                value = matrix[row_index, column_index]
                if not np.isfinite(value):
                    continue
                text = f"{value:.2f}"
                if null_column is not None:
                    cell = subset[
                        (subset["pc_condition"] == conditions[row_index])
                        & (subset["eval_condition"] == conditions[column_index])
                    ]
                    if len(cell) and np.isfinite(cell[null_column].iloc[0]):
                        text += f"\n({cell[null_column].iloc[0]:.2f})"
                ax.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value < 0.6 else "#111111",
                )
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([_condition_style(c)["label"] for c in conditions], rotation=30, ha="right")
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels([_condition_style(c)["label"] for c in conditions])
        ax.set_title(region_label(region), loc="left")
        if ax is axes[0]:
            ax.set_ylabel("PCs from")
        ax.set_xlabel("Variance of")
    fig.colorbar(image, ax=axes, shrink=0.75, label="Alignment index")
    return fig


def plot_principal_angle_spectra(
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    n_components: int,
    figsize_scale: tuple[float, float] = (1.9, 1.85),
) -> plt.Figure:
    """Sorted principal angles for each condition pair, per region.

    A pair whose spectrum hugs 0° shares a subspace; one that rises steeply
    into the tens of degrees within the first few angles occupies genuinely
    different leading directions.  Plotting the whole spectrum rather than the
    mean shows *how many* dimensions the conditions actually differ in.
    """
    regions = _regions(fits_by_region)
    fig, axes = plt.subplots(
        1, len(regions), figsize=(figsize_scale[0] * len(regions), figsize_scale[1]), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    pair_styles = {
        ("face_interactive", "face_non_interactive"): ("#7b3294", "-"),
        ("face_interactive", "object"): ("#1b7837", "--"),
        ("face_non_interactive", "object"): ("#b35806", "-."),
    }
    for ax, region in zip(axes, regions):
        fits = fits_by_region[region]
        for pair in combinations(CONDITION_ORDER, 2):
            if pair[0] not in fits or pair[1] not in fits:
                continue
            metrics = principal_angle_metrics(
                fits[pair[0]].basis(n_components), fits[pair[1]].basis(n_components)
            )
            angles = np.asarray(metrics["principal_angles"], dtype=float)
            color, linestyle = pair_styles.get(pair, (INK, "-"))
            ax.plot(
                np.arange(1, angles.size + 1),
                angles,
                color=color,
                ls=linestyle,
                lw=1.0,
                label=f"{_condition_style(pair[0])['label']} vs {_condition_style(pair[1])['label']}",
            )
        ax.axhline(90.0, color=MUTED_INK, lw=0.7, ls=":")
        ax.set_title(region_label(region), loc="left")
        ax.set_xlabel("Angle index")
    axes[0].set_ylabel("Principal angle (deg)")
    axes[0].set_ylim(0, 95)
    axes[-1].legend(fontsize=5.8, loc="lower right")
    return fig


def plot_time_resolved_separation(
    separation: pd.DataFrame,
    *,
    value_column: str = "distance_normalized",
    figsize_scale: tuple[float, float] = (1.95, 1.75),
) -> plt.Figure:
    """Condition-pair distance through the fixation, with bootstrap bands."""
    regions = _regions(separation)
    fig, axes = plt.subplots(
        1, len(regions), figsize=(figsize_scale[0] * len(regions), figsize_scale[1]), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    pair_styles = {
        "face_interactive__vs__face_non_interactive": ("#7b3294", "-"),
        "face_interactive__vs__object": ("#1b7837", "--"),
        "face_non_interactive__vs__object": ("#b35806", "-."),
    }
    scale_column = "distance" if value_column == "distance" else "distance_normalized"
    for ax, region in zip(axes, regions):
        subset = separation[separation["region"] == region]
        for pair, group in subset.groupby("pair_label"):
            group = group.sort_values("bin_center_s")
            color, linestyle = pair_styles.get(str(pair), (INK, "-"))
            centers = group["bin_center_s"].to_numpy() * 1000.0
            scale = (
                group[scale_column].to_numpy() / group["distance"].to_numpy()
                if scale_column != "distance"
                else np.ones(len(group))
            )
            ax.plot(
                centers,
                group[value_column],
                color=color,
                ls=linestyle,
                lw=1.0,
                label=str(pair).replace("__vs__", " vs ").replace("_", " "),
            )
            ax.fill_between(
                centers,
                group["ci_low"].to_numpy() * scale,
                group["ci_high"].to_numpy() * scale,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.axvline(0.0, color=MUTED_INK, lw=0.7, ls=":")
        ax.set_title(region_label(region), loc="left")
        ax.set_xlabel("Time from fixation onset (ms)")
    axes[0].set_ylabel("Normalized state distance")
    axes[-1].legend(fontsize=5.8, loc="upper left")
    return fig


def plot_decomposition_schematic(
    decomposition: Mapping[str, object],
    shares: Optional[pd.DataFrame] = None,
    *,
    figsize: tuple[float, float] = (8.4, 3.5),
) -> plt.Figure:
    """Walk one neuron's three condition curves through the decomposition.

    Everything here is **raw firing rate in Hz for a single neuron** -- no PCA is
    involved in this step, and no dimensionality choice affects it.

    Panel 1 is the observed data, which is non-negative. Panels 2-4 are the three
    parts it splits into, each expressed as a **deviation from the neuron's
    baseline** (its overall mean across conditions and time), which is why they
    are centred on zero rather than on the firing rate.  Adding the baseline back
    to panels 2, 3 and 4 reproduces panel 1 exactly; the panel titles say so, and
    the notebook checks it numerically.

    Panels 2-4 share a y-axis so their relative sizes can be read directly, which
    is the visual counterpart of the variance shares in panel 5.
    """
    conditions = list(decomposition["conditions"])
    centers = np.asarray(decomposition["bin_centers_s"], dtype=float) * 1000.0
    curves = np.asarray(decomposition["curves"], dtype=float)
    baseline = float(decomposition["baseline"])
    shared = np.asarray(decomposition["shared_time_course"], dtype=float)
    offsets = np.asarray(decomposition["condition_offsets"], dtype=float)
    residual = np.asarray(decomposition["residual"], dtype=float)

    fig, axes = plt.subplots(1, 5, figsize=figsize, constrained_layout=True)

    ax = axes[0]
    for index, condition in enumerate(conditions):
        ax.plot(centers, curves[index], color=_condition_style(condition)["color"], lw=1.0)
    ax.axhline(baseline, color=MUTED_INK, lw=0.8, ls=":")
    ax.annotate(
        f"baseline {baseline:.1f} Hz",
        xy=(centers[-1], baseline),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=MUTED_INK,
    )
    ax.set_title("1. Observed\n(firing rate)", loc="left", fontsize=7.5)
    ax.set_ylabel("Firing rate (Hz)")

    ax = axes[1]
    ax.plot(centers, shared, color=INK, lw=1.2)
    ax.set_title("2. Shared time course\n(mean of three − baseline)", loc="left", fontsize=7.5)
    ax.set_ylabel("Deviation from baseline (Hz)")

    ax = axes[2]
    for index, condition in enumerate(conditions):
        ax.plot(
            centers,
            np.full_like(centers, offsets[index]),
            color=_condition_style(condition)["color"],
            lw=1.4,
        )
    ax.set_title("3. Condition offset\n(own level − baseline)", loc="left", fontsize=7.5)

    ax = axes[3]
    for index, condition in enumerate(conditions):
        ax.plot(centers, residual[index], color=_condition_style(condition)["color"], lw=1.0)
    ax.set_title("4. What is left\n(condition × time)", loc="left", fontsize=7.5)

    # Panels 2-4 are the three parts of one quantity, so they share a scale.
    limit = 1.08 * max(
        float(np.max(np.abs(shared))),
        float(np.max(np.abs(offsets))),
        float(np.max(np.abs(residual))),
    )
    for ax in axes[1:4]:
        ax.set_ylim(-limit, limit)
        ax.axhline(0.0, color=MUTED_INK, lw=0.7)
    for ax in axes[:4]:
        ax.axvline(0.0, color=MUTED_INK, lw=0.6, ls=":")
        ax.set_xlabel("Time (ms)")
    for ax in axes[2:4]:
        ax.set_yticklabels([])

    ax = axes[4]
    names = ("shared_time_course", "condition_offset", "condition_specific_wiggle")
    labels = ("Shared\ntime", "Condition\noffset", "Condition\n× time")
    colors = ("#b0b7be", "#4c72b0", "#dd8452")
    unit_shares = [float(decomposition["shares"][name]) for name in names]
    x = np.arange(len(names))
    ax.bar(x - 0.19, unit_shares, width=0.36, color=colors, alpha=0.45)
    if shares is not None:
        population_shares = [
            float(shares[column].mean())
            for column in (
                "condition_independent_time",
                "condition_main_effect",
                "condition_by_time_interaction",
            )
        ]
        ax.bar(x + 0.19, population_shares, width=0.36, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.6, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of variance")
    ax.set_title("5. Variance shares\n(sum to 1)", loc="left", fontsize=7.5)
    ax.legend(
        handles=[
            Line2D([0], [0], lw=6, color=MUTED_INK, alpha=0.45, label="This neuron"),
            Line2D([0], [0], lw=6, color=MUTED_INK, label="Population mean"),
        ],
        fontsize=5.6,
        loc="upper right",
    )

    handles = [
        Line2D([0], [0], color=_condition_style(c)["color"], lw=1.4, label=_condition_style(c)["label"])
        for c in conditions
    ]
    handles.append(Line2D([0], [0], color="none", label="panel 1  =  baseline + 2 + 3 + 4"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.45, -0.10),
        ncol=4,
        fontsize=6.5,
        frameon=False,
    )
    return fig


def plot_comovement_schematic(
    profiles: Sequence[Mapping[str, object]],
    *,
    n_traces: int = 2,
    figsize_scale: tuple[float, float] = (2.65, 3.6),
) -> plt.Figure:
    """Show what a deviation correlation measures, pair by pair.

    Row 1 overlays the two conditions' centroid-removed trajectories along the
    leading PCs -- the question "do they wiggle together?" is answered by eye
    here before any number is computed.  Row 2 is the per-bin inner product
    whose time average is the correlation: consistently positive means the two
    lean the same way at the same moments.  Row 3 places the observed value
    against the circular-shift null, which preserves each trajectory's own
    smoothness and destroys only their alignment in time.
    """
    n_columns = len(profiles)
    fig, axes = plt.subplots(
        3,
        n_columns,
        figsize=(figsize_scale[0] * n_columns, figsize_scale[1]),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    if axes.shape[0] != 3:
        axes = axes.T

    # Rows 2 and 3 share limits across columns.  These panels exist to be
    # compared, and independently scaled axes would make a near-zero agreement
    # look as substantial as a strong one.
    product_limit = max(
        float(np.max(np.abs(np.asarray(profile["per_bin_product"], dtype=float))))
        for profile in profiles
    )
    null_low = min(
        min(float(np.min(profile["null_values"])), float(profile["observed"]))
        for profile in profiles
    )
    null_high = max(
        max(float(np.max(profile["null_values"])), float(profile["observed"]))
        for profile in profiles
    )
    null_pad = 0.06 * max(null_high - null_low, 1e-6)

    for column, profile in enumerate(profiles):
        centers = np.asarray(profile["bin_centers_s"], dtype=float) * 1000.0
        deviation_a = np.asarray(profile["deviation_a"], dtype=float)
        deviation_b = np.asarray(profile["deviation_b"], dtype=float)
        style_a = _condition_style(str(profile["condition_a"]))
        style_b = _condition_style(str(profile["condition_b"]))

        ax = axes[0, column]
        spacing = 3.4 * float(np.std(deviation_a[:n_traces]))
        for row in range(int(min(n_traces, deviation_a.shape[0]))):
            offset = row * spacing
            ax.plot(centers, deviation_a[row] + offset, color=style_a["color"], lw=1.0)
            ax.plot(centers, deviation_b[row] + offset, color=style_b["color"], lw=1.0, ls="--")
            ax.annotate(
                f"PC{row + 1}",
                xy=(0.012, offset),
                xycoords=("axes fraction", "data"),
                ha="left",
                va="center",
                fontsize=5.6,
                color=MUTED_INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.75},
            )
        ax.axvline(0.0, color=MUTED_INK, lw=0.6, ls=":")
        ax.set_yticks([])
        ax.set_title(
            f"{style_a['label']} vs {style_b['label']}\n{region_label(str(profile['region']))}",
            loc="left",
            fontsize=7.5,
        )
        if column == 0:
            ax.set_ylabel("Deviation from\nown centroid")

        ax = axes[1, column]
        per_bin = np.asarray(profile["per_bin_product"], dtype=float)
        ax.fill_between(centers, 0.0, per_bin, where=per_bin >= 0, color="#2a7f62", alpha=0.55, linewidth=0)
        ax.fill_between(centers, 0.0, per_bin, where=per_bin < 0, color="#a33a3a", alpha=0.55, linewidth=0)
        ax.axhline(0.0, color=MUTED_INK, lw=0.7)
        ax.axhline(float(profile["observed"]), color=INK, lw=1.0, ls="--")
        ax.axvline(0.0, color=MUTED_INK, lw=0.6, ls=":")
        ax.annotate(
            f"mean = {float(profile['observed']):.2f}",
            xy=(0.02, 0.88),
            xycoords="axes fraction",
            fontsize=6,
            color=INK,
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylim(-1.05 * product_limit, 1.05 * product_limit)
        if column == 0:
            ax.set_ylabel("Per-bin agreement")

        ax = axes[2, column]
        null = np.asarray(profile["null_values"], dtype=float)
        ax.hist(
            null,
            bins=np.linspace(null_low - null_pad, null_high + null_pad, 26),
            color=NULL_COLOR,
            edgecolor="none",
        )
        ax.axvline(float(profile["observed"]), color="#c0392b", lw=1.6)
        exceed = int(np.count_nonzero(null >= float(profile["observed"])))
        ax.annotate(
            f"{exceed}/{null.size} shifts exceed\np = {(exceed + 1) / (null.size + 1):.3f}",
            xy=(0.97, 0.9),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=6,
            color=INK,
        )
        ax.set_xlabel("Deviation correlation")
        ax.set_xlim(null_low - null_pad, null_high + null_pad)
        if column == 0:
            ax.set_ylabel("Circular shifts")
    return fig


def plot_variance_decomposition(
    decomposition: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (4.2, 2.4),
) -> plt.Figure:
    """Stacked share of population variance by marginalisation component."""
    components = (
        "condition_independent_time",
        "condition_main_effect",
        "condition_by_time_interaction",
    )
    labels = {
        "condition_independent_time": "Shared time course",
        "condition_main_effect": "Condition offset",
        "condition_by_time_interaction": "Condition x time",
    }
    colors = {
        "condition_independent_time": "#b0b7be",
        "condition_main_effect": "#4c72b0",
        "condition_by_time_interaction": "#dd8452",
    }
    regions = _regions(decomposition)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    bottom = np.zeros(len(regions))
    for component in components:
        values = np.asarray(
            [
                float(
                    decomposition[
                        (decomposition["region"] == region)
                        & (decomposition["component"] == component)
                    ]["fraction_of_total"].iloc[0]
                )
                for region in regions
            ]
        )
        ax.bar(
            np.arange(len(regions)),
            values,
            bottom=bottom,
            color=colors[component],
            label=labels[component],
            width=0.68,
        )
        for index, (value, base) in enumerate(zip(values, bottom)):
            if value > 0.05:
                ax.text(
                    index,
                    base + value / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if component != "condition_independent_time" else INK,
                )
        bottom = bottom + values
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels([region_label(region) for region in regions])
    ax.set_ylabel("Share of population variance")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    return fig


PAIR_PLOT_STYLES: dict[str, tuple[str, str]] = {
    "face_interactive__vs__face_non_interactive": ("#7b3294", "-"),
    "face_interactive__vs__object": ("#1b7837", "--"),
    "face_non_interactive__vs__object": ("#b35806", "-."),
}


def _pair_display(label: str) -> str:
    return str(label).replace("__vs__", " vs ").replace("face_interactive", "Int face").replace(
        "face_non_interactive", "Non-int face"
    ).replace("object", "Object").replace("_", " ")


def plot_offset_vs_dynamics(
    table: pd.DataFrame,
    *,
    shift_test: Optional[pd.DataFrame] = None,
    use_corrected: bool = True,
    figsize: tuple[float, float] = (7.2, 2.6),
) -> plt.Figure:
    """Static offset vs moving part of the separation, and shared dynamics.

    Left: the exact split of mean squared separation into ``||m_a - m_b||^2``
    and ``<||d_a - d_b||^2>``.  Right: the deviation correlation, which asks the
    complementary question -- given that the conditions sit apart, do they
    nonetheless move through their own neighbourhoods in step?  The two panels
    can disagree, and that disagreement is the interesting case.
    """
    offset_column = "offset_share_corrected" if use_corrected else "offset_share_of_separation"
    if offset_column not in table.columns:
        offset_column = "offset_share_of_separation"
    correlation_column = (
        "deviation_correlation_corrected" if use_corrected else "deviation_correlation"
    )
    if correlation_column not in table.columns:
        correlation_column = "deviation_correlation"

    regions = _regions(table)
    pairs = [pair for pair in PAIR_PLOT_STYLES if pair in set(table["pair_label"])]
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    width = 0.8 / max(len(pairs), 1)

    ax = axes[0]
    for index, pair in enumerate(pairs):
        offsets = np.arange(len(regions)) + index * width - 0.4 + width / 2
        shares = [
            float(table[(table["region"] == region) & (table["pair_label"] == pair)][offset_column].iloc[0])
            for region in regions
        ]
        color = PAIR_PLOT_STYLES[pair][0]
        ax.bar(offsets, shares, width=width, color=color, label=_pair_display(pair))
        ax.bar(
            offsets,
            [1.0 - value for value in shares],
            bottom=shares,
            width=width,
            color=color,
            alpha=0.32,
            linewidth=0,
        )
    ax.axhline(0.5, color=MUTED_INK, lw=0.7, ls=":")
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels([region_label(region) for region in regions])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of squared separation")
    ax.set_title("Solid = static offset,  pale = dynamics", loc="left")
    ax.legend(fontsize=5.8, loc="lower right")

    ax = axes[1]
    for index, pair in enumerate(pairs):
        offsets = np.arange(len(regions)) + index * width - 0.4 + width / 2
        values = [
            float(table[(table["region"] == region) & (table["pair_label"] == pair)][correlation_column].iloc[0])
            for region in regions
        ]
        ax.bar(offsets, values, width=width, color=PAIR_PLOT_STYLES[pair][0])
        if shift_test is not None:
            for position, region in zip(offsets, regions):
                cell = shift_test[
                    (shift_test["region"] == region) & (shift_test["pair_label"] == pair)
                ]
                if not len(cell):
                    continue
                # The bars show the disattenuated correlation while the test
                # runs on the raw one.  Disattenuation is a single positive
                # factor applied to observed and null alike, so rescaling the
                # null by the same factor puts both on one axis without
                # changing any p-value.
                source = table[(table["region"] == region) & (table["pair_label"] == pair)]
                factor = 1.0
                if use_corrected and "disattenuation_factor" in source and len(source):
                    candidate = float(source["disattenuation_factor"].iloc[0])
                    if np.isfinite(candidate) and candidate > 0:
                        factor = candidate
                low, high = _null_interval(cell, factor)
                ax.plot(
                    [position, position],
                    [low, high],
                    color=INK,
                    lw=1.0,
                    solid_capstyle="butt",
                )
                if float(cell["p_value"].iloc[0]) < 0.05:
                    ax.annotate(
                        "*",
                        xy=(position, max(values[list(regions).index(region)], high)),
                        xytext=(0, 1),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                        color=INK,
                    )
    ax.axhline(0.0, color=MUTED_INK, lw=0.7)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels([region_label(region) for region in regions])
    ax.set_ylabel("Deviation correlation")
    ax.set_title("Do they move together?  (lines = 95% of shift null, * p < 0.05)", loc="left")
    return fig


def plot_condition_dynamics(
    summary: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.2, 4.4),
) -> plt.Figure:
    """How far, how fast and in how many dimensions each condition moves.

    Raw and noise-corrected values are drawn together because the size of the
    correction is itself the point: interactive-face averages rest on about five
    times as many fixations as the other two, so an uncorrected comparison would
    partly be a comparison of trial counts.
    """
    regions = _regions(summary)
    conditions = [c for c in CONDITION_ORDER if c in set(summary["condition"])]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    width = 0.8 / max(len(conditions), 1)
    positions = np.arange(len(regions))

    def _values(column: str, condition: str) -> list[float]:
        return [
            float(summary[(summary["region"] == region) & (summary["condition"] == condition)][column].iloc[0])
            for region in regions
        ]

    panels = (
        ("excursion_rms", "excursion_rms_corrected", "Excursion RMS", axes[0, 0]),
        ("dynamics_participation_ratio", "dynamics_participation_ratio_corrected", "Dynamics dimensionality", axes[0, 1]),
    )
    for raw_column, corrected_column, title, ax in panels:
        for index, condition in enumerate(conditions):
            offsets = positions + index * width - 0.4 + width / 2
            style = _condition_style(condition)
            ax.bar(offsets, _values(raw_column, condition), width=width, color=style["color"], alpha=0.34, linewidth=0)
            if corrected_column in summary.columns:
                ax.bar(
                    offsets,
                    _values(corrected_column, condition),
                    width=width * 0.55,
                    color=style["color"],
                    label=style["label"] if ax is axes[0, 0] else None,
                )
        ax.set_xticks(positions)
        ax.set_xticklabels([region_label(region) for region in regions])
        ax.set_title(f"{title}  (pale = raw, solid = noise-corrected)", loc="left", fontsize=7.5)
    axes[0, 0].legend(fontsize=6, loc="upper right")

    ax = axes[1, 0]
    if "speed_over_noise" in summary.columns:
        for index, condition in enumerate(conditions):
            offsets = positions + index * width - 0.4 + width / 2
            ax.bar(
                offsets,
                _values("speed_over_noise", condition),
                width=width,
                color=_condition_style(condition)["color"],
            )
        ax.axhline(1.0, color=THRESHOLD_COLOR, lw=0.9, ls="--")
        ax.annotate(
            "at or below 1: bin-to-bin motion not resolvable",
            xy=(0.02, 0.92),
            xycoords="axes fraction",
            fontsize=5.8,
            color=THRESHOLD_COLOR,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([region_label(region) for region in regions])
    ax.set_ylabel("Step energy / noise expectation")
    ax.set_title("Is the motion above the noise floor?", loc="left", fontsize=7.5)

    ax = axes[1, 1]
    observed: list[float] = []
    for index, condition in enumerate(conditions):
        offsets = positions + index * width - 0.4 + width / 2
        values = _values("median_n_trials", condition)
        observed.extend(values)
        _log_bars(ax, offsets, values, width=width, color=_condition_style(condition)["color"])
    _finish_log_axis(ax, observed)
    ax.set_xticks(positions)
    ax.set_xticklabels([region_label(region) for region in regions])
    ax.set_ylabel("Median fixations per unit")
    ax.set_title("The confound being corrected for", loc="left", fontsize=7.5)
    fig.tight_layout()
    return fig


def plot_alignment_vs_comovement(
    subspace_table: pd.DataFrame,
    offset_table: pd.DataFrame,
    *,
    shift_test: Optional[pd.DataFrame] = None,
    figsize: tuple[float, float] = (7.4, 3.7),
) -> plt.Figure:
    """The two dynamics questions plotted against each other, twice.

    Vertical axis in both panels: the deviation correlation -- do the two
    conditions travel their own neighbourhoods **at the same time**?

    Horizontal axis: do they vary along the same directions at all?  Shown twice
    on purpose.  The left panel uses the raw alignment index, and its points
    cluster by region -- but that clustering is mostly an artefact.  A random
    k-dimensional subspace of an N-dimensional space already captures a fraction
    k/N of anything, and the four areas have very different unit counts, so each
    region's alignment values start from a different floor (dashed lines).  The
    right panel subtracts that floor, putting chance at zero and making the
    regions comparable.

    What survives the correction is the point: the pairs are still not ordered
    along the horizontal axis, and are still cleanly ordered along the vertical
    one.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, constrained_layout=True)
    regions = _regions(offset_table)
    correlation_column = (
        "deviation_correlation_corrected"
        if "deviation_correlation_corrected" in offset_table.columns
        else "deviation_correlation"
    )
    markers = {region: marker for region, marker in zip(regions, ("o", "s", "^", "D", "v", "P"))}
    specs = (
        ("alignment_index", "Alignment index (raw)", True),
        ("alignment_above_floor", "Alignment above chance  (A − k/N)/(1 − k/N)", False),
    )

    for ax, (column, xlabel, draw_floor) in zip(axes, specs):
        if draw_floor:
            # BLA aside, the floors sit close together, so labels are staggered
            # vertically or they overprint each other.
            ordered_floors = sorted(
                regions,
                key=lambda name: float(
                    subspace_table.loc[
                        subspace_table["region"] == name, "alignment_floor_analytic"
                    ].iloc[0]
                )
                if len(subspace_table[subspace_table["region"] == name])
                else 0.0,
            )
            for rank, region in enumerate(ordered_floors):
                cell = subspace_table[subspace_table["region"] == region]
                if not len(cell) or "alignment_floor_analytic" not in cell:
                    continue
                floor = float(cell["alignment_floor_analytic"].iloc[0])
                units = int(cell["n_units"].iloc[0]) if "n_units" in cell else 0
                ax.axvline(floor, color=REGION_COLORS.get(region, MUTED_INK), lw=0.9, ls="--", alpha=0.8)
                ax.annotate(
                    f"{region_label(region)} N={units}",
                    xy=(floor, 1.0 - 0.055 * (rank % 3)),
                    xycoords=("data", "axes fraction"),
                    xytext=(2, -2),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=5.2,
                    color=REGION_COLORS.get(region, MUTED_INK),
                )
        else:
            ax.axvline(0.0, color=MUTED_INK, lw=0.9, ls="--")

        for _, row in offset_table.iterrows():
            pair = str(row["pair_label"])
            first, second = pair.split("__vs__")
            cell = subspace_table[
                (subspace_table["region"] == row["region"])
                & (subspace_table["pc_condition"] == first)
                & (subspace_table["eval_condition"] == second)
            ]
            if not len(cell) or column not in cell:
                continue
            significant = False
            if shift_test is not None:
                match = shift_test[
                    (shift_test["region"] == row["region"]) & (shift_test["pair_label"] == pair)
                ]
                significant = bool(len(match) and float(match["p_value"].iloc[0]) < 0.05)
            ax.scatter(
                float(cell[column].iloc[0]),
                float(row[correlation_column]),
                marker=markers.get(str(row["region"]), "o"),
                s=66 if significant else 46,
                color=PAIR_PLOT_STYLES.get(pair, (INK, "-"))[0],
                edgecolors=INK if significant else "none",
                linewidths=1.1,
                zorder=3,
            )
        ax.axhline(0.0, color=MUTED_INK, lw=0.8)
        ax.set_xlabel(xlabel)

    axes[0].set_ylabel("Deviation correlation\n(move at the same time?)")
    axes[0].set_title("Raw — clusters by region", loc="left", fontsize=8)
    axes[1].set_title("Floor-corrected — comparable", loc="left", fontsize=8)

    handles = [
        Line2D([0], [0], marker="o", lw=0, ms=6, color=PAIR_PLOT_STYLES[pair][0], label=_pair_display(pair))
        for pair in PAIR_PLOT_STYLES
    ]
    handles.extend(
        Line2D([0], [0], marker=markers[region], lw=0, ms=5.5, color=MUTED_INK, label=region_label(region))
        for region in regions
    )
    handles.append(
        Line2D([0], [0], marker="o", lw=0, ms=7, color="none", markeredgecolor=INK, label="shift null p < 0.05")
    )
    axes[1].legend(handles=handles, fontsize=5.6, ncol=2, loc="upper left")
    return fig


def plot_subspace_distance_map(
    fits_by_region: Mapping[str, Mapping[str, PopulationPCAFit]],
    *,
    n_components: int,
    figsize: tuple[float, float] = (3.6, 3.2),
) -> plt.Figure:
    """Two-dimensional map of every region-condition subspace.

    Subspaces are embedded by classical MDS on the Grassmann distance between
    them, computed *within* each region -- distances across regions are not
    defined, since the subspaces live in different unit spaces.  Each region is
    therefore laid out on its own and then plotted in a shared frame, so the
    figure compares the *shape* of the three-condition arrangement across
    regions rather than absolute positions.
    """
    regions = _regions(fits_by_region)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for region in regions:
        fits = fits_by_region[region]
        conditions = [c for c in CONDITION_ORDER if c in fits]
        size = len(conditions)
        distances = np.zeros((size, size))
        for i, first in enumerate(conditions):
            for j, second in enumerate(conditions):
                if i >= j:
                    continue
                value = principal_angle_metrics(
                    fits[first].basis(n_components), fits[second].basis(n_components)
                )["grassmann_distance"]
                distances[i, j] = distances[j, i] = float(value)

        squared = distances**2
        centering = np.eye(size) - np.ones((size, size)) / size
        gram = -0.5 * centering @ squared @ centering
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1][:2]
        coordinates = eigenvectors[:, order] * np.sqrt(np.maximum(eigenvalues[order], 0.0))

        color = REGION_COLORS.get(region, INK)
        closed = np.vstack([coordinates, coordinates[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=0.9, alpha=0.7)
        for index, condition in enumerate(conditions):
            ax.scatter(
                coordinates[index, 0],
                coordinates[index, 1],
                color=_condition_style(condition)["color"],
                edgecolors=color,
                linewidths=1.1,
                s=48,
                zorder=3,
            )
        ax.annotate(
            region_label(region),
            xy=coordinates.mean(axis=0),
            fontsize=6.5,
            color=color,
            ha="center",
        )
    handles = [
        Line2D([0], [0], marker="o", lw=0, ms=6, color=_condition_style(condition)["color"], label=_condition_style(condition)["label"])
        for condition in CONDITION_ORDER
    ]
    handles.extend(
        Line2D([0], [0], color=REGION_COLORS.get(region, INK), lw=1.2, label=region_label(region))
        for region in regions
    )
    ax.legend(handles=handles, fontsize=6, ncol=2, loc="best")
    ax.set_xlabel("MDS 1 (Grassmann distance)")
    ax.set_ylabel("MDS 2")
    ax.set_aspect("equal", adjustable="datalim")
    return fig


def plot_null_comparison(
    null_table: pd.DataFrame,
    *,
    ceiling: Optional[pd.DataFrame] = None,
    figsize: tuple[float, float] = (7.0, 2.5),
) -> plt.Figure:
    """Observed subspace similarity against the condition-shuffled null.

    Drawn with the within-condition ceiling as an upper reference so the
    observed bar can be read as a position on the interval between "no
    coordinated condition structure" and "as aligned as a condition is with
    itself", rather than as a bare number.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    regions = _regions(null_table)
    pairs = list(dict.fromkeys(null_table["pair_label"]))
    width = 0.8 / max(len(pairs), 1)
    specs = (
        ("observed_alignment_index", "shuffled_alignment_index", "Alignment index"),
        ("observed_mean_principal_angle", "shuffled_mean_principal_angle", "Mean principal angle (deg)"),
    )
    for ax, (observed_column, null_column, ylabel) in zip(axes, specs):
        for pair_index, pair in enumerate(pairs):
            offsets = np.arange(len(regions)) + pair_index * width - 0.4 + width / 2
            observed = [
                float(
                    null_table[(null_table["region"] == region) & (null_table["pair_label"] == pair)][
                        observed_column
                    ].iloc[0]
                )
                for region in regions
            ]
            shuffled = [
                float(
                    null_table[(null_table["region"] == region) & (null_table["pair_label"] == pair)][
                        null_column
                    ].iloc[0]
                )
                for region in regions
            ]
            ax.bar(
                offsets,
                observed,
                width=width,
                color=["#7b3294", "#1b7837", "#b35806"][pair_index % 3],
                label=str(pair).replace("__vs__", " vs ").replace("_", " ") if ax is axes[0] else None,
            )
            ax.scatter(offsets, shuffled, marker="_", s=60, color=INK, linewidths=1.1, zorder=3)
        if ceiling is not None and observed_column == "observed_alignment_index":
            values = [
                float(ceiling[ceiling["region"] == region]["alignment_ceiling"].mean())
                for region in regions
            ]
            ax.scatter(np.arange(len(regions)), values, marker="v", s=22, color=THRESHOLD_COLOR, zorder=4)
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels([region_label(region) for region in regions])
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=6, loc="upper left")
    axes[0].annotate(
        "— shuffled null    ▾ within-condition ceiling",
        xy=(0.02, 1.02),
        xycoords="axes fraction",
        fontsize=6,
        color=MUTED_INK,
    )
    return fig


def _log_bars(ax, positions, values, *, width, color, label=None, floor: float = 1.0):
    """Bars on a log axis, resting on ``floor`` rather than on zero.

    A bar drawn from 0 on a log axis has its base at negative infinity.  That is
    invisible on screen because the artist is clipped, but this repo's savefig
    patch strips clip paths before PDF export, and a subsequent
    ``bbox_inches="tight"`` then measures the unclipped artist and writes an
    image hundreds of thousands of pixels tall.  Resting the bars on a positive
    floor keeps every artist finite.
    """
    base = float(floor)
    heights = [max(float(value) - base, 0.0) for value in values]
    return ax.bar(positions, heights, width=width, bottom=base, color=color, label=label)


def _finish_log_axis(ax, values: Sequence[float], *, floor: float = 1.0) -> None:
    """Apply a log scale with explicit finite limits."""
    ax.set_yscale("log")
    finite = [float(value) for value in values if np.isfinite(value) and value > 0]
    if finite:
        ax.set_ylim(float(floor), max(finite) * 1.8)


def plot_unit_inventory(
    inventory: pd.DataFrame,
    *,
    conditions: Sequence[str] = CONDITION_ORDER,
    figsize: tuple[float, float] = (7.0, 2.7),
) -> plt.Figure:
    """Units per region and median fixations per unit behind each condition."""
    regions = _regions(inventory)
    # Log-scaled panels plus the repo's savefig patch make constrained layout
    # re-solve an already-tightened figure and collapse the axes; an explicit
    # tight_layout at the end is stable under both.
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    counts = [int(inventory[inventory["region"] == region]["n_units"].iloc[0]) for region in regions]
    sessions = [int(inventory[inventory["region"] == region]["n_sessions"].iloc[0]) for region in regions]
    bars = ax.bar(
        np.arange(len(regions)),
        counts,
        color=[REGION_COLORS.get(region, INK) for region in regions],
        width=0.66,
    )
    for bar, count in zip(bars, counts):
        ax.annotate(
            str(count),
            xy=(bar.get_x() + bar.get_width() / 2, count),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=6,
            color=MUTED_INK,
        )
    ax.margins(y=0.16)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(
        [f"{region_label(region)}\n{count} sess." for region, count in zip(regions, sessions)]
    )
    ax.set_ylabel("Units entering PCA")
    ax.set_title("Population size", loc="left")

    ax = axes[1]
    width = 0.8 / max(len(conditions), 1)
    observed: list[float] = []
    for index, condition in enumerate(conditions):
        column = f"median_trials_{condition}"
        if column not in inventory.columns:
            continue
        values = [float(inventory[inventory["region"] == region][column].iloc[0]) for region in regions]
        observed.extend(values)
        _log_bars(
            ax,
            np.arange(len(regions)) + index * width - 0.4 + width / 2,
            values,
            width=width,
            color=_condition_style(condition)["color"],
            label=_condition_style(condition)["label"],
        )
    _finish_log_axis(ax, observed)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels([region_label(region) for region in regions])
    ax.set_ylabel("Median fixations per unit")
    ax.set_title("Trials behind each average", loc="left")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    return fig


def plot_verification_summary(
    checks: pd.DataFrame,
    *,
    figsize_scale: tuple[float, float] = (6.6, 0.34),
) -> plt.Figure:
    """Pass/fail grid for the numerical identity checks, region by check."""
    regions = _regions(checks)
    names = list(dict.fromkeys(checks["check"]))
    grid = np.zeros((len(names), len(regions)))
    for row, name in enumerate(names):
        for column, region in enumerate(regions):
            cell = checks[(checks["check"] == name) & (checks["region"] == region)]
            grid[row, column] = float(bool(cell["passed"].all())) if len(cell) else np.nan

    fig, ax = plt.subplots(
        figsize=(figsize_scale[0], figsize_scale[1] * len(names) + 0.9), constrained_layout=True
    )
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for row in range(len(names)):
        for column in range(len(regions)):
            ax.text(
                column,
                row,
                "pass" if grid[row, column] == 1 else "FAIL",
                ha="center",
                va="center",
                fontsize=6.5,
                color=INK,
            )
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([region_label(region) for region in regions])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([name.replace("_", " ") for name in names], fontsize=6.5)
    ax.set_title("PCA identity checks", loc="left")
    return fig


__all__ = [
    "optimize_view_angle",
    "plot_alignment_matrix",
    "plot_cross_condition_variance_curves",
    "plot_comovement_panel",
    "plot_decomposition_overview",
    "plot_pc_condition_separation",
    "plot_subspace_metric_summary",
    "plot_cumulative_variance",
    "plot_null_comparison",
    "plot_pc_plane_projections",
    "plot_pc_timecourses",
    "plot_pc_trajectories_3d",
    "plot_principal_angle_spectra",
    "plot_comovement_schematic",
    "plot_condition_dynamics",
    "plot_decomposition_schematic",
    "plot_offset_vs_dynamics",
    "plot_alignment_vs_comovement",
    "plot_subspace_distance_map",
    "plot_time_resolved_separation",
    "plot_unit_inventory",
    "plot_variance_decomposition",
    "plot_variance_spectra",
    "plot_verification_summary",
]


# --------------------------------------------------------------------------- #
# Chapter figures
# --------------------------------------------------------------------------- #

COMPONENT_COLORS: dict[str, str] = {
    "shared_time_course": "#b0b7be",
    "condition_offset": "#4c72b0",
    "condition_by_time": "#dd8452",
}
COMPONENT_SHORT: dict[str, str] = {
    "shared_time_course": "Shared\ntime",
    "condition_offset": "Condition\noffset",
    "condition_by_time": "Condition\n× time",
}


def _null_interval(row: pd.DataFrame, factor: float = 1.0) -> tuple[float, float]:
    """95% interval of a circular-shift null, scaled by ``factor``.

    Prefers the null's own 2.5th/97.5th percentiles.  The shift null is an exact
    enumeration of every admissible shift rather than a sample from a symmetric
    distribution, so a mean +/- 2 SD band would impose a shape it does not have.
    Falls back to the SD band only for tables written before the percentiles were
    stored.
    """
    if "shift_null_p2p5" in row and np.isfinite(float(row["shift_null_p2p5"].iloc[0])):
        return (
            float(row["shift_null_p2p5"].iloc[0]) * factor,
            float(row["shift_null_p97p5"].iloc[0]) * factor,
        )
    centre = float(row["shift_null_mean"].iloc[0]) * factor
    spread = float(row["shift_null_sd"].iloc[0]) * factor
    return centre - 2.0 * spread, centre + 2.0 * spread


def _stars(p_value: float) -> str:
    """Compact significance marker; ``n.s.`` when nothing survives correction."""
    value = float(p_value)
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "n.s."


def _bracket(ax, x_left: float, x_right: float, y: float, height: float, label: str) -> None:
    """Mark a significant contrast with a plain horizontal line and a star.

    Deliberately not a bracket with downward ticks: the ticks add ink that reads
    as data at small panel sizes, and the span alone already says which bars are
    being compared.
    """
    ax.plot([x_left, x_right], [y, y], color=INK, lw=0.8, solid_capstyle="butt", clip_on=False)
    ax.annotate(
        label,
        xy=(0.5 * (x_left + x_right), y),
        xytext=(0, 0.8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=5.6,
        color=INK,
        clip_on=False,
    )


def plot_decomposition_overview(
    example: Mapping[str, object],
    share_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    normalization: str = "corrected",
    figsize: tuple[float, float] = (8.6, 2.35),
) -> plt.Figure:
    """Compact version of the three-way split, ending in population statistics.

    Panels 1-4 are the arithmetic, worked on one neuron so the reader can see
    that the parts add back. Panel 5 leaves that neuron behind: it is the
    variance-weighted share for all neurons pooled across regions, with
    neuron-bootstrap intervals and the pairwise contrasts, which is what the
    claim in the text rests on.
    """
    conditions = list(example["conditions"])
    centers = np.asarray(example["bin_centers_s"], dtype=float) * 1000.0
    curves = np.asarray(example["curves"], dtype=float)
    baseline = float(example["baseline"])
    shared = np.asarray(example["shared_time_course"], dtype=float)
    offsets = np.asarray(example["condition_offsets"], dtype=float)
    residual = np.asarray(example["residual"], dtype=float)

    fig, axes = plt.subplots(
        1, 5, figsize=figsize, constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]}
    )

    ax = axes[0]
    for index, condition in enumerate(conditions):
        ax.plot(centers, curves[index], color=_condition_style(condition)["color"], lw=1.0)
    ax.axhline(baseline, color=MUTED_INK, lw=0.8, ls=":")
    ax.set_title("1  Observed", loc="left", fontsize=7)
    ax.set_ylabel("Firing rate (Hz)", fontsize=6.5)
    ax.annotate(
        f"baseline {baseline:.0f} Hz",
        xy=(centers[-1], baseline),
        xytext=(-2, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=MUTED_INK,
    )

    ax = axes[1]
    ax.plot(centers, shared, color=INK, lw=1.1)
    ax.set_title("2  Shared time", loc="left", fontsize=7)
    ax.set_ylabel("Δ from baseline (Hz)", fontsize=6.5)

    ax = axes[2]
    for index, condition in enumerate(conditions):
        ax.plot(centers, np.full_like(centers, offsets[index]), color=_condition_style(condition)["color"], lw=1.3)
    ax.set_title("3  Condition offset", loc="left", fontsize=7)

    ax = axes[3]
    for index, condition in enumerate(conditions):
        ax.plot(centers, residual[index], color=_condition_style(condition)["color"], lw=0.9)
    ax.set_title("4  Condition × time", loc="left", fontsize=7)

    limit = 1.08 * max(
        float(np.max(np.abs(shared))), float(np.max(np.abs(offsets))), float(np.max(np.abs(residual)))
    )
    for ax in axes[1:4]:
        ax.set_ylim(-limit, limit)
        ax.axhline(0.0, color=MUTED_INK, lw=0.6)
    for ax in axes[:4]:
        ax.axvline(0.0, color=MUTED_INK, lw=0.5, ls=":")
        ax.set_xlabel("Time (ms)", fontsize=6.5)
        ax.tick_params(labelsize=5.8)
    for ax in axes[2:4]:
        ax.set_yticklabels([])
    axes[0].annotate(
        "1  =  baseline + 2 + 3 + 4",
        xy=(0.0, 1.22),
        xycoords="axes fraction",
        fontsize=6.2,
        color=INK,
    )

    # Panel 5: population statistics.
    ax = axes[4]
    pooled = share_summary[
        (share_summary["scope"] == "all_regions") & (share_summary["normalization"] == normalization)
    ].set_index("component")
    regional = share_summary[
        (share_summary["scope"] != "all_regions") & (share_summary["normalization"] == normalization)
    ]
    order = [c for c in COMPONENT_SHORT if c in pooled.index]
    x = np.arange(len(order))
    values = [float(pooled.loc[c, "share"]) for c in order]
    errors = np.array(
        [
            [float(pooled.loc[c, "share"] - pooled.loc[c, "ci_low"]) for c in order],
            [float(pooled.loc[c, "ci_high"] - pooled.loc[c, "share"]) for c in order],
        ]
    )
    ax.bar(x, values, width=0.62, color=[COMPONENT_COLORS[c] for c in order])
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    for index, component in enumerate(order):
        points = regional[regional["component"] == component]["share"].to_numpy()
        ax.scatter(
            np.full(points.size, index) + 0.22,
            points,
            s=9,
            color=INK,
            alpha=0.65,
            zorder=4,
            linewidths=0,
        )

    top = max(values[i] + errors[1][i] for i in range(len(order)))
    step = 0.085
    level = top + 0.03
    pairs_drawn = 0
    for left, right in combinations(order, 2):
        cell = contrasts[
            (contrasts["scope"] == "all_regions")
            & (contrasts["normalization"] == normalization)
            & (contrasts["component_a"] == left)
            & (contrasts["component_b"] == right)
        ]
        if not len(cell):
            continue
        _bracket(
            ax,
            order.index(left),
            order.index(right),
            level + pairs_drawn * step,
            step * 0.28,
            _stars(float(cell["p_value_corrected"].iloc[0])),
        )
        pairs_drawn += 1
    ax.set_xticks(x)
    ax.set_xticklabels([COMPONENT_SHORT[c] for c in order], fontsize=5.8)
    ax.set_ylim(0, level + pairs_drawn * step + 1.6 * step)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Share of variance", fontsize=6.5)
    ax.tick_params(labelsize=5.8)
    n_units = int(pooled["n_units"].iloc[0])
    ax.set_title(
        f"5  All {n_units} neurons pooled ({normalization})\ndots = regions", loc="left", fontsize=7, pad=10
    )

    handles = [
        Line2D([0], [0], color=_condition_style(c)["color"], lw=1.4, label=_condition_style(c)["label"])
        for c in conditions
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.4, -0.14),
        ncol=3,
        fontsize=6.2,
        frameon=False,
    )
    return fig


def plot_pc_condition_separation(
    effects: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    figsize_scale: tuple[float, float] = (2.05, 2.5),
) -> plt.Figure:
    """Which component separates which pair, region by region.

    Bars are the mean absolute difference between two conditions' score time
    courses along one component, with neuron-bootstrap intervals; brackets are
    the corrected contrasts between pairs. Reading a column tells you what that
    component encodes: a component on which one pair towers over the others is
    an axis for that distinction.
    """
    regions = _regions(effects)
    pcs = sorted(effects["pc_index"].unique())
    pairs = [pair for pair in PAIR_PLOT_STYLES if pair in set(effects["pair_label"])]
    fig, axes = plt.subplots(
        1, len(regions), figsize=(figsize_scale[0] * len(regions), figsize_scale[1]), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    width = 0.8 / max(len(pairs), 1)

    for ax, region in zip(axes, regions):
        subset = effects[effects["region"] == region]
        positions: dict[tuple[int, str], float] = {}
        tops: dict[int, float] = {}
        for pair_index, pair in enumerate(pairs):
            offsets = np.arange(len(pcs)) + pair_index * width - 0.4 + width / 2
            values, low, high = [], [], []
            for pc, position in zip(pcs, offsets):
                cell = subset[(subset["pc_index"] == pc) & (subset["pair_label"] == pair)]
                value = float(cell["mean_abs_difference"].iloc[0]) if len(cell) else np.nan
                values.append(value)
                low.append(float(cell["ci_low"].iloc[0]) if len(cell) else np.nan)
                high.append(float(cell["ci_high"].iloc[0]) if len(cell) else np.nan)
                positions[(pc, pair)] = position
                tops[pc] = max(
                    tops.get(pc, 0.0),
                    max(value, float(cell["ci_high"].iloc[0])) if len(cell) else 0.0,
                )
            ax.bar(
                offsets,
                values,
                width=width,
                color=PAIR_PLOT_STYLES[pair][0],
                label=_pair_display(pair) if ax is axes[0] else None,
            )
            ax.vlines(offsets, low, high, color=INK, lw=0.9, zorder=4)

        span = max(tops.values()) if tops else 1.0
        step = 0.085 * span
        for pc in pcs:
            drawn = 0
            for first, second in combinations(pairs, 2):
                cell = contrasts[
                    (contrasts["region"] == region)
                    & (contrasts["pc_index"] == pc)
                    & (contrasts["pair_a"] == first)
                    & (contrasts["pair_b"] == second)
                ]
                if not len(cell) or float(cell["p_value_corrected"].iloc[0]) >= 0.05:
                    continue
                _bracket(
                    ax,
                    positions[(pc, first)],
                    positions[(pc, second)],
                    tops[pc] + 0.04 * span + drawn * step,
                    step * 0.3,
                    _stars(float(cell["p_value_corrected"].iloc[0])),
                )
                drawn += 1
        ax.set_xticks(np.arange(len(pcs)))
        ax.set_xticklabels([f"PC{pc}" for pc in pcs])
        ax.set_title(region_label(region), loc="left")
        ax.set_ylim(0, max(tops.values()) * 1.45 if tops else 1.0)
    axes[0].set_ylabel("Mean |difference| along component")
    handles = [
        Line2D([0], [0], lw=6, color=PAIR_PLOT_STYLES[pair][0], label=_pair_display(pair))
        for pair in pairs
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(handles),
        fontsize=6.2,
        frameon=False,
    )
    return fig


def plot_comovement_panel(
    profiles: Sequence[Mapping[str, object]],
    offset_table: pd.DataFrame,
    shift_test: pd.DataFrame,
    *,
    n_traces: int = 1,
    figsize: tuple[float, float] = (8.2, 3.1),
) -> plt.Figure:
    """The co-movement measure and its result, in one panel.

    Left two columns build the measure on two example pairs: the centroid-removed
    trajectories along the leading component, then the per-bin inner product
    whose time average *is* the correlation.  Right column is the result across
    all regions with each pair's circular-shift null.

    One component is drawn rather than several.  The measure itself runs over all
    retained components, so no number of traces can show it in full; one is
    enough to make the question legible by eye, and more only costs panel height.
    The trajectory panels share a y-axis so the two example pairs are directly
    comparable.

    The null interval is rescaled by the same disattenuation factor as the bars,
    so both sit on one axis; because that factor is a single positive multiplier
    applied to observed and null alike, no p-value changes.  This is what makes a
    large bar with a large null -- dmPFC's face pair, for instance -- legible as
    "big but not beyond chance" rather than as a contradiction.
    """
    n_rows = int(n_traces) + 1
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(n_rows, 3, width_ratios=[1.0, 1.0, 1.35])

    trace_limit = 1.08 * max(
        float(np.max(np.abs(np.asarray(profile[key], dtype=float)[:n_traces])))
        for profile in profiles[:2]
        for key in ("deviation_a", "deviation_b")
    )
    product_limit = 1.05 * max(
        float(np.max(np.abs(np.asarray(profile["per_bin_product"], dtype=float))))
        for profile in profiles[:2]
    )

    for column, profile in enumerate(profiles[:2]):
        centers = np.asarray(profile["bin_centers_s"], dtype=float) * 1000.0
        deviation_a = np.asarray(profile["deviation_a"], dtype=float)
        deviation_b = np.asarray(profile["deviation_b"], dtype=float)
        style_a = _condition_style(str(profile["condition_a"]))
        style_b = _condition_style(str(profile["condition_b"]))

        for row in range(int(min(n_traces, deviation_a.shape[0]))):
            ax = fig.add_subplot(grid[row, column])
            ax.plot(centers, deviation_a[row], color=style_a["color"], lw=1.1)
            ax.plot(centers, deviation_b[row], color=style_b["color"], lw=1.1, ls="--")
            ax.axhline(0.0, color=MUTED_INK, lw=0.6)
            ax.axvline(0.0, color=MUTED_INK, lw=0.6, ls=":")
            ax.set_ylim(-trace_limit, trace_limit)
            ax.set_xticklabels([])
            if column == 0:
                ax.set_ylabel(f"PC{row + 1} deviation\nfrom centroid (a.u.)", fontsize=6.5)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f"{style_a['label']} vs {style_b['label']}", loc="left", fontsize=7.5)

        ax = fig.add_subplot(grid[n_rows - 1, column])
        per_bin = np.asarray(profile["per_bin_product"], dtype=float)
        ax.fill_between(centers, 0.0, per_bin, where=per_bin >= 0, color="#2a7f62", alpha=0.55, linewidth=0)
        ax.fill_between(centers, 0.0, per_bin, where=per_bin < 0, color="#a33a3a", alpha=0.55, linewidth=0)
        ax.axhline(0.0, color=MUTED_INK, lw=0.7)
        ax.axhline(float(profile["observed"]), color=INK, lw=1.0, ls="--")
        ax.axvline(0.0, color=MUTED_INK, lw=0.6, ls=":")
        ax.set_ylim(-product_limit, product_limit)
        ax.set_xlabel("Time from fixation onset (ms)", fontsize=6.5)
        null = np.asarray(profile["null_values"], dtype=float)
        exceed = int(np.count_nonzero(null >= float(profile["observed"])))
        ax.annotate(
            f"mean {float(profile['observed']):.2f}\n{exceed}/{null.size} shifts exceed",
            xy=(0.03, 0.92),
            xycoords="axes fraction",
            va="top",
            fontsize=5.8,
            color=INK,
        )
        if column == 0:
            n_used = int(profile.get("n_components", deviation_a.shape[0]))
            ax.set_ylabel(f"$d_a\\cdot d_b$ (all {n_used} PCs)", fontsize=6.5)
        else:
            ax.set_yticklabels([])

    ax = fig.add_subplot(grid[:, 2])
    regions = _regions(offset_table)
    pairs = [pair for pair in PAIR_PLOT_STYLES if pair in set(offset_table["pair_label"])]
    width = 0.8 / max(len(pairs), 1)
    column_name = (
        "deviation_correlation_corrected"
        if "deviation_correlation_corrected" in offset_table.columns
        else "deviation_correlation"
    )
    for index, pair in enumerate(pairs):
        offsets = np.arange(len(regions)) + index * width - 0.4 + width / 2
        for position, region in zip(offsets, regions):
            source = offset_table[
                (offset_table["region"] == region) & (offset_table["pair_label"] == pair)
            ]
            test = shift_test[
                (shift_test["region"] == region) & (shift_test["pair_label"] == pair)
            ]
            if not len(source):
                continue
            value = float(source[column_name].iloc[0])
            ax.bar(
                position,
                value,
                width=width,
                color=PAIR_PLOT_STYLES[pair][0],
                label=_pair_display(pair) if region == regions[0] else None,
            )
            if not len(test):
                continue
            factor = 1.0
            if "disattenuation_factor" in source:
                candidate = float(source["disattenuation_factor"].iloc[0])
                if np.isfinite(candidate) and candidate > 0:
                    factor = candidate
            low, high = _null_interval(test, factor)
            ax.plot(
                [position, position],
                [low, high],
                color=INK,
                lw=1.1,
                solid_capstyle="butt",
                zorder=4,
            )
            if float(test["p_value"].iloc[0]) < 0.05:
                ax.annotate(
                    "*",
                    xy=(position, max(value, high)),
                    xytext=(0, 1),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=INK,
                )
    ax.axhline(0.0, color=MUTED_INK, lw=0.8)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels([region_label(region) for region in regions])
    ax.set_ylabel("Deviation correlation")
    ax.set_title(
        "Do the conditions move together?\nlines = 95% of shift null, * p < 0.05",
        loc="left",
        fontsize=7.5,
    )
    ax.legend(fontsize=5.6, loc="lower left")
    return fig


def plot_subspace_metric_summary(
    effects: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.2, 2.7),
) -> plt.Figure:
    """Appendix summary: subspace overlap and principal angle, with statistics.

    Bars are the observed values; whiskers are the spread of the neuron
    subsampling distribution transplanted onto them, so they show how much a
    change of neurons moves each measure rather than a bias-corrected confidence
    interval -- both metrics depend on sample size beyond the k/N correction, so
    a subsample does not estimate quite the same quantity (see
    :func:`bootstrap_subspace_metrics`).  Brackets are the FDR-corrected
    contrasts between condition pairs, which are unaffected by that dependence
    because all three pairs share each subsample.

    Alignment is plotted floor-corrected because the chance level k/N differs
    fourfold across these areas and raw values are not comparable between them.
    """
    metrics = [
        ("alignment_above_floor", "Alignment above chance"),
        ("mean_principal_angle", "Mean principal angle (deg)"),
    ]
    regions = _regions(effects)
    pairs = [pair for pair in PAIR_PLOT_STYLES if pair in set(effects["pair_label"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    width = 0.8 / max(len(pairs), 1)

    for ax, (metric, ylabel) in zip(axes, metrics):
        subset = effects[effects["metric"] == metric]
        positions: dict[tuple[str, str], float] = {}
        tops: dict[str, float] = {}
        for index, pair in enumerate(pairs):
            offsets = np.arange(len(regions)) + index * width - 0.4 + width / 2
            values, low, high = [], [], []
            for position, region in zip(offsets, regions):
                cell = subset[(subset["region"] == region) & (subset["pair_label"] == pair)]
                value = float(cell["value"].iloc[0]) if len(cell) else np.nan
                values.append(value)
                low_column = "spread_low" if "spread_low" in cell else "ci_low"
                high_column = "spread_high" if "spread_high" in cell else "ci_high"
                low.append(float(cell[low_column].iloc[0]) if len(cell) else np.nan)
                high.append(float(cell[high_column].iloc[0]) if len(cell) else np.nan)
                positions[(region, pair)] = position
                tops[region] = max(
                    tops.get(region, 0.0),
                    max(value, float(cell[high_column].iloc[0])) if len(cell) else 0.0,
                )
            ax.bar(
                offsets,
                values,
                width=width,
                color=PAIR_PLOT_STYLES[pair][0],
                label=_pair_display(pair) if ax is axes[0] else None,
            )
            # Drawn as an interval rather than as offsets from the bar, so a
            # resampling interval that does not straddle the observed value is
            # shown honestly instead of raising or being clipped.
            ax.vlines(offsets, low, high, color=INK, lw=0.9, zorder=4)
        span = max(tops.values()) if tops else 1.0
        step = 0.075 * span
        for region in regions:
            drawn = 0
            for first, second in combinations(pairs, 2):
                cell = contrasts[
                    (contrasts["metric"] == metric)
                    & (contrasts["region"] == region)
                    & (contrasts["pair_a"] == first)
                    & (contrasts["pair_b"] == second)
                ]
                if not len(cell) or float(cell["p_value_corrected"].iloc[0]) >= 0.05:
                    continue
                _bracket(
                    ax,
                    positions[(region, first)],
                    positions[(region, second)],
                    tops[region] + 0.035 * span + drawn * step,
                    step * 0.3,
                    _stars(float(cell["p_value_corrected"].iloc[0])),
                )
                drawn += 1
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels([region_label(region) for region in regions])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, span * 1.35)
    axes[0].legend(fontsize=5.6, loc="upper left")
    return fig
