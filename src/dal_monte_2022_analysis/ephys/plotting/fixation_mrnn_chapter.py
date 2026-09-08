"""Figures for the mRNN chapter: schematics and the composite result figures.

The chapter's figures are built from the tables ``analysis.fixation_mrnn_chapter`` assembles.
Conventions shared by every panel: groups of five to ten fits are drawn as **box plots**
(median, quartiles, whiskers to the full range) with the individual fits as small dots;
larger groups as violins; the **dense network is the reference** in every cost panel; the
only horizontal lines are reference values the reader needs (zero, chance, the dense
network, the adequacy bar where a selection is made); and **only significant comparisons
are marked**, with stars from the Holm-corrected tests the analysis module computes.
Schematics are drawn with matplotlib patches so they export as editable vector art.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from dal_monte_2022_analysis.ephys.plotting.fixation_mrnn_audit import CORRECTED_COLOR, REPORTED_COLOR, _panel
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    CONDITION_SHORT_LABELS,
    INK,
    MUTED_INK,
    REGION_COLORS,
    REGION_LABELS,
    nice_axis,
)

WITHIN_COLOR = REPORTED_COLOR      # self-recurrence, as in the bottleneck schematic
CROSS_COLOR = CORRECTED_COLOR      # inter-regional pathways
ARM_COLORS = {"dense": "#8a8a8a", "constrained": "#2a6f8f"}
RUNG_LABELS = {0: "alone", 1: "+1 region", 2: "+2 regions", 3: "all four"}
MODEL_REGION_ORDER = ("ofc", "bla", "dmpfc", "accg")
DISPLAY_REGION_ORDER = ("bla", "accg", "dmpfc", "ofc")
NEUTRAL_EDGE_COLOR = "#31485c"

SHORT_CONSTRAINT_LABELS = {
    "no inter-regional connections": "no inter-\nregional",
    "within-region rank 1": "within\nrank 1",
    "inter-regional rank 10": "inter\nrank 10",
    "inter-regional rank 1": "inter\nrank 1",
    "selected: within 1, inter-regional 10": "selected\n(within 1,\ninter 10)",
}
ARCHITECTURE_LABELS = {
    "no inter-regional connections": "no inter-regional\nconnections",
    "within-region rank 1": "within-region\nrecurrence rank 1",
    "inter-regional rank 1": "inter-regional\npathways rank 1",
    "inter-regional rank 10": "inter-regional\npathways rank 10",
    "selected: within 1, inter-regional 10": "selected model\n(within 1, inter 10)",
}


# ======================================================================================
# helpers: colours, boxes, significance marks
# ======================================================================================


def _mix(color: str, other: str, weight: float) -> tuple[float, float, float]:
    """``weight`` of ``other`` blended into ``color``."""
    a = np.asarray(mcolors.to_rgb(color))
    b = np.asarray(mcolors.to_rgb(other))
    return tuple((1 - weight) * a + weight * b)


def condition_shades(condition: str, n: int = 3) -> list[tuple[float, float, float]]:
    """``n`` shades of a fixation type's colour, dark to light."""
    base = CONDITION_COLORS[condition]
    if n == 1:
        return [mcolors.to_rgb(base)]
    return [_mix(base, "black", 0.35)] + [mcolors.to_rgb(base)] + [_mix(base, "white", w) for w in np.linspace(0.45, 0.65, n - 2)]


def draw_boxes(ax, entries: Sequence[Mapping[str, object]], *, width: float = 0.55, points: bool = True, seed: int = 0,
               point_size: float = 5.0) -> None:
    """Box plots for small groups: median, quartiles, whiskers to the full range, every value a dot.

    Each entry is ``{"x", "values", "color", "filled"}``; a filled box is the group's colour,
    an unfilled one is white with a coloured edge (the dense network, wherever it is drawn
    beside a constrained one).
    """
    rng = np.random.default_rng(seed)
    for entry in entries:
        values = np.asarray(entry["values"], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        color = entry["color"]
        filled = bool(entry.get("filled", True))
        ax.boxplot(
            [values], positions=[float(entry["x"])], widths=width, patch_artist=True, showfliers=False, whis=(0, 100),
            medianprops=dict(color="white" if filled else color, lw=1.2),
            whiskerprops=dict(color=color, lw=0.8), capprops=dict(color=color, lw=0.8),
            boxprops=dict(facecolor=color if filled else "white", edgecolor=color, lw=0.9), zorder=2,
        )
        if points:
            jitter = rng.uniform(-width * 0.28, width * 0.28, values.size)
            size = point_size if values.size <= 20 else point_size * 0.55
            ax.scatter(float(entry["x"]) + jitter, values, s=size, color=INK if filled else color, alpha=0.6 if values.size <= 20 else 0.45,
                       lw=0, zorder=3)


def draw_bars(ax, entries: Sequence[Mapping[str, object]], *, width: float = 0.26, points: bool = True, seed: int = 0,
              point_size: float = 6.0) -> None:
    """Bars for small groups: the mean, a capless SEM error bar, and every value as a dot.

    Same entry schema as :func:`draw_boxes`. A filled bar is the group's colour; an unfilled
    one is white with a coloured edge (the dense network beside a constrained one).
    """
    rng = np.random.default_rng(seed)
    for entry in entries:
        values = np.asarray(entry["values"], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        color = entry["color"]
        filled = bool(entry.get("filled", True))
        x = float(entry["x"])
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        ax.bar(x, mean, width=width, facecolor=color if filled else "white", edgecolor=color, linewidth=0.9, zorder=2)
        ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor=INK if filled else color, elinewidth=1.1, capsize=0, zorder=5)
        if points:
            jitter = rng.uniform(-width * 0.3, width * 0.3, values.size)
            size = point_size if values.size <= 20 else point_size * 0.5
            ax.scatter(x + jitter, values, s=size, color=INK if filled else color, alpha=0.65 if values.size <= 20 else 0.4,
                       lw=0, zorder=4)


def draw_violins(ax, frame: pd.DataFrame, *, x: str, y: str, hue: str, order: Sequence[str], hue_order: Sequence[str],
                 palette: Mapping[str, str], width: float = 0.8) -> None:
    """Grouped violins in the project's thesis style: quartile lines inside, cut at the data range, dark edges, full colour.

    Wraps ``seaborn.violinplot`` with the settings the single-unit and population chapters
    use (``inner="quart"``, ``cut=0``, saturation 1) and darkens the body edges the same way.
    """
    sns.violinplot(ax=ax, data=frame, x=x, y=y, hue=hue, order=list(order), hue_order=list(hue_order), palette=dict(palette),
                   dodge=True, inner="quart", cut=0.0, linewidth=0.8, width=width, saturation=1.0, legend=False)
    for body in [artist for artist in ax.collections if isinstance(artist, PolyCollection)]:
        body.set_edgecolor("#222222")
        body.set_linewidth(0.65)
        body.set_alpha(1.0)
    for line in ax.lines:
        line.set_color("#222222")


def mark_contrasts(ax, comparisons: Sequence[tuple[float, float, str]], *, top: float | None = None,
                   fontsize: float = 6.5, pad_frac: float = 0.04, step_frac: float = 0.085, linewidth: float = 1.2) -> None:
    """Bars with stars for the comparisons handed in -- callers pass only the significant ones.

    A comparison is a plain horizontal bar (no end ticks) with its stars above. Bars are
    stacked so overlapping spans sit on different levels. They are drawn in axes-fraction y
    above ``top`` (the highest data value in the panel; the axes' data limit when not
    given), and the y-limits are extended to make room, so the bars never enter the data
    limits themselves and shared axes stay intact.
    """
    import matplotlib.transforms as mtransforms

    comparisons = [c for c in comparisons if c[2]]
    if not comparisons:
        return
    y_low, y_high = ax.get_ylim()
    if top is None:
        top = ax.dataLim.y1 if np.isfinite(ax.dataLim.y1) else y_high
    top = max(float(top), y_low + 1e-12)
    placed: list[tuple[float, float, int]] = []
    levels: list[tuple[float, float, int, str]] = []
    for x1, x2, label in sorted(comparisons, key=lambda c: abs(c[1] - c[0])):
        lo, hi = min(x1, x2), max(x1, x2)
        level = 0
        while any(not (hi < a - 0.05 or lo > b + 0.05) and lv == level for a, b, lv in placed):
            level += 1
        placed.append((lo, hi, level))
        levels.append((lo, hi, level, label))
    max_level = max(lv for _, _, lv, _ in levels)
    room = pad_frac + (max_level + 1) * step_frac + 0.02
    y_high_new = y_low + (top - y_low) / (1.0 - room)
    ax.set_ylim(y_low, y_high_new)
    top_frac = (top - y_low) / (y_high_new - y_low)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for lo, hi, level, label in levels:
        y = top_frac + pad_frac + level * step_frac
        ax.plot([lo, hi], [y, y], color=INK, lw=linewidth, solid_capstyle="butt", transform=trans, clip_on=False, zorder=6)
        ax.text((lo + hi) / 2, y + 0.004, label, ha="center", va="bottom", fontsize=fontsize, color=INK, transform=trans, clip_on=False)


def contrasts_to_marks(table: pd.DataFrame, position: Callable[[pd.Series, str], float]) -> list[tuple[float, float, str]]:
    """Turn a Holm-adjusted contrast table into ``(x_a, x_b, stars)`` for the significant rows."""
    if table is None or table.empty or "significant" not in table:
        return []
    out = []
    for _, row in table[table["significant"]].iterrows():
        out.append((position(row, "a"), position(row, "b"), str(row["stars"])))
    return out


def _region_graph(ax, *, centre=(0.5, 0.5), radius=0.32, node=0.085, members=MODEL_REGION_ORDER,
                  cross=True, self_loops=True, cross_color=CROSS_COLOR, within_color=WITHIN_COLOR,
                  label=True, fontsize=6.5, cross_lw=0.8, faded=()):
    """Four regions on a square, with the pathways between them and their self-loops."""
    cx, cy = centre
    positions = {
        "ofc": (cx - radius, cy + radius), "bla": (cx + radius, cy + radius),
        "dmpfc": (cx - radius, cy - radius), "accg": (cx + radius, cy - radius),
    }
    present = [r for r in MODEL_REGION_ORDER if r in members]
    if cross:
        for a in present:
            for b in present:
                if a == b:
                    continue
                (x0, y0), (x1, y1) = positions[a], positions[b]
                alpha = 0.25 if (a, b) in faded or (b, a) in faded else 0.75
                ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                             color=cross_color, lw=cross_lw, alpha=alpha, shrinkA=node * 105, shrinkB=node * 105,
                                             connectionstyle="arc3,rad=0.14", zorder=2))
    for r in MODEL_REGION_ORDER:
        x, y = positions[r]
        if r not in members:
            ax.add_patch(Circle((x, y), node, facecolor="white", edgecolor="#cfcfcf", lw=0.8, ls=(0, (2, 2)), zorder=3))
            continue
        ax.add_patch(Circle((x, y), node, facecolor="white", edgecolor=REGION_COLORS[r], lw=1.4, zorder=3))
        if self_loops:
            ax.add_patch(FancyArrowPatch((x + node * 0.75, y + node * 0.95), (x + node * 1.05, y + node * 0.15),
                                         arrowstyle="-|>", mutation_scale=5, color=within_color, lw=1.2,
                                         connectionstyle="arc3,rad=1.6", zorder=4))
        if label:
            ax.text(x, y, REGION_LABELS[r], ha="center", va="center", fontsize=fontsize, color=INK, zorder=5)
    return positions


def _box(ax, xy, width, height, text, *, fontsize=6.5, facecolor="white", edgecolor=INK, lw=0.8, color=INK, pad=0.02):
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle=f"round,pad={pad},rounding_size=0.02",
                                facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=2))
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, color=color, zorder=3)


def _arrow(ax, start, end, *, color=INK, lw=0.9, style="-|>", rad=0.0, mutation=8, zorder=2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=mutation, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=zorder))


def _off(ax, xlim=(0, 1), ylim=(0, 1)):
    """Fixed data limits with equal aspect, anchored to the top of the grid cell so titles line up across a row."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.axis("off")


def _demo_traces(seed: int = 3, n: int = 100) -> dict[str, np.ndarray]:
    """Three smooth, condition-coloured illustrative traces for the schematic."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    out = {}
    for k, condition in enumerate(CONDITION_ORDER):
        phases = rng.uniform(0, 2 * np.pi, 4)
        trace = sum(np.sin(2 * np.pi * (2 + j) * t + phases[j]) / (j + 1.5) for j in range(4))
        out[condition] = 0.35 * trace / np.abs(trace).max() + 0.25 * (1 - k)
    return out


# ======================================================================================
# Figure 1 -- what is fitted, with what, how it is scored, and how the ceiling is measured
# ======================================================================================


def _draw_ceiling_panel(ax) -> None:
    """The noise ceiling as a pipeline: halve the trials, average, project, correlate, Spearman-Brown.

    Four columns in data units of a (0, 2.6) x (0, 1.1) panel: trials halved | the two
    half-averages | both projected onto the model's PCA basis | the numbers.
    """
    rng = np.random.default_rng(7)
    half_a, half_b = "#3b5b7a", "#a4b8cc"
    t = np.linspace(0, 1, 60)
    head = dict(fontsize=5.0, va="top", ha="center")
    foot = dict(fontsize=4.6, color=MUTED_INK, ha="center", va="bottom")
    # column 1: one unit's trials, randomly halved
    ax.text(0.22, 1.10, "one unit, one\nfixation type:\nits trials,\nrandomly halved", **head)
    n_trials = 14
    for i in range(n_trials):
        y = 0.76 - i * 0.038
        color = half_a if rng.uniform() < 0.5 else half_b
        ticks = rng.uniform(0.0, 0.38, size=rng.integers(4, 9))
        ax.scatter(0.03 + ticks, np.full(ticks.size, y), s=2.2, color=color, marker="|", lw=0.7)
    ax.text(0.20, 0.20, "half A", color=half_a, fontsize=5.0, ha="right", va="top")
    ax.text(0.24, 0.20, "half B", color=half_b, fontsize=5.0, ha="left", va="top")
    ax.text(0.22, 0.02, "25 random\nhalvings, averaged", **foot)
    # column 2: the two half-averages
    _arrow(ax, (0.47, 0.50), (0.55, 0.50), color=MUTED_INK, lw=0.8, mutation=6)
    base = np.sin(2 * np.pi * 1.3 * t + 0.4) * 0.5 + 0.3 * np.sin(2 * np.pi * 3.1 * t)
    ax.text(0.80, 1.10, "average each\nhalf, unit\nby unit", **head)
    for color, label, offset in ((half_a, "mean of half A", 0.62), (half_b, "mean of half B", 0.30)):
        noise = np.convolve(rng.normal(size=60) * 0.35, np.ones(4) / 4, mode="same")
        ax.plot(0.61 + 0.38 * t, offset + 0.08 * (base + noise), color=color, lw=1.0)
        ax.text(0.80, offset + 0.12, label, fontsize=5.0, color=color, ha="center", va="bottom")
    ax.text(0.80, 0.02, "every unit of\nthe region, every\nfixation type", **foot)
    # column 3: project both onto the model's PCA basis
    _arrow(ax, (1.04, 0.50), (1.12, 0.50), color=MUTED_INK, lw=0.8, mutation=6)
    ax.text(1.40, 1.10, "project both onto\nthe model's PCA\nbasis; centre as\n$R^2$ is centred", **head)
    for j in range(3):
        y = 0.66 - j * 0.19
        pc = np.sin(2 * np.pi * (1.1 + 0.9 * j) * t + j) * (0.9 - 0.2 * j)
        noise = np.convolve(rng.normal(size=60) * 0.25, np.ones(4) / 4, mode="same")
        ax.plot(1.22 + 0.36 * t, y + 0.055 * pc, color=half_a, lw=0.9)
        ax.plot(1.22 + 0.36 * t, y + 0.055 * (pc + noise), color=half_b, lw=0.9, ls=(0, (3, 1.5)))
        ax.text(1.20, y, f"PC{j + 1}", fontsize=4.6, color=MUTED_INK, ha="right", va="center")
    ax.text(1.40, 0.02, "⋮ 42 components", **foot)
    # column 4: the numbers
    _arrow(ax, (1.63, 0.50), (1.71, 0.50), color=MUTED_INK, lw=0.8, mutation=6)
    ax.text(1.76, 1.10, "$r$ = correlation of A with B\nover every component and\nbin of one region × fixation\ntype (variance-weighted,\nas $R^2$ is)",
            fontsize=5.0, va="top")
    ax.text(1.76, 0.64, "ceiling $= 2r\\,/\\,(1 + r)$\n(Spearman–Brown)", fontsize=5.6, va="top")
    ax.text(1.76, 0.44, "the $R^2$ a perfect model of the\nfull average could reach: 0.983–\n0.998 here; Gaussian-halves\ncontrol ≈ 0 (±0.01)",
            fontsize=5.0, va="top")
    ax.text(1.76, 0.01, "every fit is read as $R^2\\,/\\,$ceiling", fontsize=5.4, va="bottom", fontweight="bold")


def plot_chapter_schematic(
    *,
    n_units_by_region: Mapping[str, int] | None = None,
    n_components: int = 42,
    hidden_units: int = 40,
    n_bins: int = 100,
    figsize: tuple[float, float] = (7.4, 4.9),
):
    """Data → target → model, and how the ceiling every fit is read against is measured.

    (a) trial-averaged firing rates of every unit for the three fixation types;
    (b) the target: each region's own PCA, ``n_components`` scores per bin, three
    trajectories per region; (c) the multi-regional RNN: four all-to-all blocks driven by
    a constant one-hot fixation-type input, each read out linearly into its region's PC
    scores; (d) the noise ceiling: trials halved at random, each half averaged and
    projected onto the model's PCA basis, the two halves correlated over the whole
    trajectory of a region × fixation type, and Spearman–Brown corrected.
    """
    units = dict(n_units_by_region or {})
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.35], height_ratios=[1.0, 0.95], wspace=0.12, hspace=0.18)
    ax_data = fig.add_subplot(grid[0, 0])
    ax_target = fig.add_subplot(grid[0, 1])
    ax_model = fig.add_subplot(grid[:, 2])
    ax_ceiling = fig.add_subplot(grid[1, 0:2])

    # ---- (a) data ---------------------------------------------------------------------------
    ax = ax_data
    traces = _demo_traces()
    t = np.linspace(0.08, 0.92, n_bins)
    for k, condition in enumerate(CONDITION_ORDER):
        ax.text(0.05 + 0.45 * k, 0.99, CONDITION_SHORT_LABELS[condition], ha="left", va="top", fontsize=5.4,
                color=CONDITION_COLORS[condition])
    for k, region in enumerate(MODEL_REGION_ORDER):
        y0 = 0.80 - k * 0.2
        ax.add_patch(Rectangle((0.05, y0 - 0.085), 0.9, 0.17, facecolor="#f4f4f4", edgecolor="none", zorder=1))
        rng = np.random.default_rng(k)
        for condition in CONDITION_ORDER:
            jitter = 0.03 * rng.standard_normal(n_bins)
            ax.plot(t, y0 - 0.02 + 0.09 * traces[condition] + np.convolve(jitter, np.ones(6) / 6, mode="same"),
                    color=CONDITION_COLORS[condition], lw=0.8, zorder=2)
        count = f" · {units[region]} units" if region in units else ""
        ax.text(0.06, y0 + 0.08, f"{REGION_LABELS[region]}{count}", fontsize=5.4, color=REGION_COLORS[region], va="top", zorder=4)
    ax.plot([0.5, 0.5], [0.10, 0.90], color=MUTED_INK, lw=0.5, ls=":", zorder=3)
    ax.text(0.5, 0.08, "fixation onset", ha="center", va="top", fontsize=5.4, color=MUTED_INK)
    ax.text(0.05, 0.02, "−500 ms", ha="left", va="top", fontsize=5.2, color=MUTED_INK)
    ax.text(0.95, 0.02, "+500 ms", ha="right", va="top", fontsize=5.2, color=MUTED_INK)
    ax.set_title("trial-averaged firing rate\nof every unit, 10 ms bins", fontsize=7, pad=2)
    _off(ax, ylim=(-0.06, 1.02))
    _panel(ax, "a")

    # ---- (b) target ------------------------------------------------------------------------
    ax = ax_target
    ax.text(0.5, 0.97, "per-region PCA on the pooled\nfixation-type × time average", ha="center", va="top", fontsize=6.2)
    ax.add_patch(Rectangle((0.12, 0.10), 0.76, 0.68, facecolor="white", edgecolor=INK, lw=0.8, zorder=1))
    for j in range(6):
        y0 = 0.70 - j * 0.105
        for condition in CONDITION_ORDER:
            ax.plot(0.16 + 0.68 * np.linspace(0, 1, n_bins), y0 + 0.1 * traces[condition] * (0.9 - 0.1 * j),
                    color=CONDITION_COLORS[condition], lw=0.6, alpha=0.9 - 0.08 * j, zorder=2)
        ax.text(0.13, y0, f"PC{j + 1}", fontsize=4.8, color=MUTED_INK, ha="left", va="center")
    ax.text(0.5, 0.10, "⋮", ha="center", va="center", fontsize=8, color=MUTED_INK)
    ax.text(0.5, 0.03, f"target per region: 3 fixation types × {n_bins} bins × {n_components} PCs",
            ha="center", va="center", fontsize=5.6, color=INK)
    ax.set_title("the target: PC-score trajectories", fontsize=7, pad=2)
    _off(ax, ylim=(-0.04, 1.02))
    _panel(ax, "b")

    # ---- (c) model -------------------------------------------------------------------------
    ax = ax_model
    positions = _region_graph(ax, centre=(0.5, 0.54), radius=0.2, node=0.085, cross_lw=0.9)
    _box(ax, (0.30, 0.93), 0.40, 0.06, "fixation type · one-hot, constant", fontsize=5.8, facecolor="#f4f4f4", edgecolor=MUTED_INK)
    for r in ("ofc", "bla"):
        x, y = positions[r]
        _arrow(ax, (x, 0.925), (x, y + 0.09), color=MUTED_INK, lw=0.7, mutation=6)
    ax.text(0.5, 0.885, "into every region", ha="center", va="center", fontsize=5.0, color=MUTED_INK)
    for r in MODEL_REGION_ORDER:
        x, y = positions[r]
        dx = -0.135 if r in ("ofc", "dmpfc") else 0.135
        _arrow(ax, (x + np.sign(dx) * 0.09, y), (x + dx * 1.35, y), color=REGION_COLORS[r], lw=0.8, mutation=6)
        _box(ax, (x + dx * 1.35 - (0.11 if dx < 0 else 0.0), y - 0.04), 0.11, 0.08, f"{n_components}\nPCs", fontsize=5.0,
             facecolor="white", edgecolor=REGION_COLORS[r], lw=0.8, color=REGION_COLORS[r])
    ax.text(0.5, 0.22, f"{hidden_units} tanh units per region · linear readout per region\n"
                       "initial state trained per fixation type", ha="center", va="top", fontsize=5.6, color=INK)
    ax.text(0.5, 0.11, "$h(t{+}1) = \\tanh\\left(W\\,h(t) + W_{in}u_c + b\\right)$", ha="center", va="top", fontsize=6.6)
    ax.text(0.5, 0.03, "$\\hat y_r(t) = C_r\\,h_r(t) + d_r$", ha="center", va="top", fontsize=6.6)
    ax.text(0.5, -0.05, "self-recurrence $W_{rr}$ (orange)\ninter-regional $W_{rs}$ (blue)\n"
                        "objective: the worst region × fixation-type\nunexplained fraction (soft maximum)", ha="center", va="top", fontsize=5.4, color=MUTED_INK)
    ax.set_title("the model: four recurrent blocks, all-to-all", fontsize=7, pad=2)
    _off(ax, xlim=(-0.02, 1.02), ylim=(-0.26, 1.0))
    _panel(ax, "c")

    # ---- (d) the ceiling ------------------------------------------------------------------
    ax = ax_ceiling
    _draw_ceiling_panel(ax)
    ax.set_title("the noise ceiling every fit is read against", fontsize=7, pad=2, loc="left")
    _off(ax, xlim=(0.0, 2.55), ylim=(0.0, 1.14))
    _panel(ax, "d")
    return fig


def plot_manipulation_schematic(*, hidden_units: int = 40, figsize: tuple[float, float] = (7.4, 2.5)):
    """The three ways the network is constrained or damaged, one panel each."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.45, 1.15, 1.0]})

    ax = axes[0]
    subsets = [("bla",), ("bla", "ofc"), ("bla", "ofc", "accg"), MODEL_REGION_ORDER]
    for k, members in enumerate(subsets):
        _region_graph(ax, centre=(0.13 + 0.245 * k, 0.52), radius=0.075, node=0.036, members=members, label=False, cross_lw=0.6)
        ax.text(0.13 + 0.245 * k, 0.29, RUNG_LABELS[k], ha="center", va="top", fontsize=5.8)
    ax.annotate("", xy=(0.95, 0.20), xytext=(0.03, 0.20), arrowprops=dict(arrowstyle="-|>", color=MUTED_INK, lw=0.8))
    ax.text(0.49, 0.16, "more of the network attached · same width, same readout, same target", ha="center", va="top",
            fontsize=5.4, color=MUTED_INK)
    ax.text(0.5, 0.92, "refit from scratch; score BLA on its own trajectories at every rung", ha="center", va="top", fontsize=5.6)
    ax.set_title("the ladder: does a region need the network?", fontsize=7, pad=2)
    _off(ax, ylim=(0.05, 1.0))
    _panel(ax, "a")

    ax = axes[1]
    n, side, ox, oy = 4, 0.15, 0.05, 0.28
    for i in range(n):
        for j in range(n):
            color = WITHIN_COLOR if i == j else CROSS_COLOR
            ax.add_patch(Rectangle((ox + j * side, oy + (n - 1 - i) * side), side, side, facecolor=color,
                                   alpha=0.85 if i == j else 0.5, edgecolor="white", lw=1.0))
            ax.text(ox + j * side + side / 2, oy + (n - 1 - i) * side + side / 2, "$r_w$" if i == j else "$r_c$",
                    ha="center", va="center", fontsize=6.4, color="white")
    for k, r in enumerate(MODEL_REGION_ORDER):
        ax.text(ox + k * side + side / 2, oy + n * side + 0.01, REGION_LABELS[r], ha="center", va="bottom", fontsize=4.8)
        ax.text(ox - 0.01, oy + (n - 1 - k) * side + side / 2, REGION_LABELS[r], ha="right", va="center", fontsize=4.8)
    px = 0.74
    ax.add_patch(Rectangle((px, 0.58), 0.2, 0.2, facecolor=CROSS_COLOR, alpha=0.5, edgecolor="none"))
    ax.text(px + 0.1, 0.68, "$W_{rs}$", ha="center", va="center", fontsize=6.5, color="white")
    ax.text(px + 0.1, 0.545, "=", ha="center", va="center", fontsize=8)
    ax.add_patch(Rectangle((px + 0.02, 0.30), 0.045, 0.2, facecolor=CROSS_COLOR, alpha=0.9, edgecolor="none"))
    ax.add_patch(Rectangle((px + 0.085, 0.455), 0.2 - 0.085, 0.045, facecolor=CROSS_COLOR, alpha=0.9, edgecolor="none"))
    ax.text(px + 0.0425, 0.28, "$L$", ha="center", va="top", fontsize=6)
    ax.text(px + 0.145, 0.44, "$R$", ha="center", va="top", fontsize=6)
    ax.text(px + 0.1, 0.19, f"{hidden_units}×$r$ · $r$×{hidden_units}", ha="center", va="top", fontsize=5.2, color=MUTED_INK)
    ax.text(0.5, 0.10, "every block a product of rank $r_w$ (own) or $r_c$ (inter-regional)\n"
                       "refit from scratch on the $(r_w, r_c)$ grid and its two marginals", ha="center", va="top", fontsize=5.4)
    ax.set_title("bottlenecks: how narrow can a route be?", fontsize=7, pad=2)
    _off(ax, ylim=(-0.05, 1.0))
    _panel(ax, "b")

    ax = axes[2]
    positions = _region_graph(ax, centre=(0.5, 0.58), radius=0.2, node=0.075, cross_lw=0.8, faded=(("ofc", "bla"),))
    (x0, y0), (x1, y1) = positions["ofc"], positions["bla"]
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    ax.plot([xm - 0.045, xm + 0.045], [ym - 0.045, ym + 0.045], color="#c0392b", lw=1.6, zorder=6)
    ax.plot([xm - 0.045, xm + 0.045], [ym + 0.045, ym - 0.045], color="#c0392b", lw=1.6, zorder=6)
    ax.text(xm, ym + 0.13, "OFC ↔ BLA silenced", ha="center", va="bottom", fontsize=5.4, color="#c0392b")
    ax.text(0.5, 0.22, "silence one block of the trained network:\na pathway, a pair (both directions), a region's\nown block, or every pathway into and out of\n"
                       "one region · matched random-weight control", ha="center", va="top", fontsize=5.4)
    ax.set_title("lesions: which connections carry the fit?", fontsize=7, pad=2)
    _off(ax, ylim=(-0.05, 1.0))
    _panel(ax, "c")
    fig.tight_layout()
    return fig


# ======================================================================================
# Figures 2 and 3 -- the ladder
# ======================================================================================


def plot_ladder_traces(
    traces: pd.DataFrame,
    *,
    region: str,
    rungs: Sequence[int] = (0, 1, 2, 3),
    n_fits_by_rung: Mapping[int, int] | None = None,
    figsize: tuple[float, float] = (7.4, 3.4),
):
    """Target against the rung-averaged reconstruction, three components overlaid per panel, each on its own scale.

    One row per rung, one column per fixation type. Each component is rescaled to its own
    target range (min to max over the window, the same range at every rung), so all three
    fill the panel. The target is the prominent trace -- thick, in three shades of the
    fixation type's colour (dark to light) -- and the reconstruction the thin grey one in
    the same order, so the eye follows the expected trace and reads the fit as how closely
    the grey line tracks it. The number at the right of each panel is the component's
    $R^2$, averaged over the fits at the rung.
    """
    conditions = [c for c in CONDITION_ORDER if c in set(traces["condition"])]
    indices = sorted(traces["index"].unique())
    model_shades = ["#2b2b2b", "#6a6a6a", "#a3a3a3"][: len(indices)]
    lo = traces.groupby(["condition", "index"])["observed"].min()
    hi = traces.groupby(["condition", "index"])["observed"].max()
    fig, axes = plt.subplots(len(rungs), len(conditions), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, rung in enumerate(rungs):
        for j, condition in enumerate(conditions):
            ax = axes[i][j]
            shades = condition_shades(condition, len(indices))
            block = traces[(traces["n_partners"] == rung) & (traces["condition"] == condition)]
            for k, index in enumerate(indices):
                part = block[block["index"] == index].sort_values("time_s")
                if part.empty:
                    continue
                scale = float(hi[(condition, index)] - lo[(condition, index)]) or 1.0
                observed = (part["observed"] - lo[(condition, index)]) / scale
                predicted = (part["predicted"] - lo[(condition, index)]) / scale
                ax.plot(part["time_s"], observed, color=shades[k], lw=1.8, alpha=0.95, zorder=3 + k)
                ax.plot(part["time_s"], predicted, color=model_shades[k], lw=0.8, zorder=6 + k)
                if "r2_mean" in part.columns:
                    ax.text(0.55, 0.98 - 0.15 * k, f"{part['r2_mean'].iloc[0]:.2f}", ha="left", va="top", fontsize=5.6, color=shades[k])
            ax.axvline(0.0, color=MUTED_INK, lw=0.5, ls=":", zorder=1)
            ax.set_xlim(-0.55, 0.72)
            ax.set_ylim(-0.04, 1.06)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_bounds(-0.5, 0.5)
            ax.set_xticks([-0.5, 0.0, 0.5])
            if i == 0:
                ax.set_title(CONDITION_SHORT_LABELS[condition], fontsize=8, color=CONDITION_COLORS[condition])
            if j == 0:
                count = f"\n{n_fits_by_rung[rung]} fits" if n_fits_by_rung and rung in n_fits_by_rung else ""
                short = {0: "alone", 1: "+1", 2: "+2", 3: "all four"}.get(rung, RUNG_LABELS.get(rung, rung))
                ax.set_ylabel(f"{REGION_LABELS.get(region, region)} {short}{count}", fontsize=6.2, labelpad=2)
            if i == len(rungs) - 1:
                ax.set_xlabel("time from fixation (s)")
            ax.tick_params(labelsize=6.5)
    handles = [plt.Line2D([], [], color=c, lw=1.8) for c in condition_shades("face_interactive", len(indices))] + [plt.Line2D([], [], color=model_shades[1], lw=0.8)]
    labels = [f"target, PC{int(k) + 1}" for k in indices] + ["model (greys, same order)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels), fontsize=6, frameon=False,
               title="each component on its own min–max scale; number: component $R^2$ at that rung", title_fontsize=6)
    fig.tight_layout(h_pad=0.3, w_pad=0.5, rect=(0, 0, 1, 0.92))
    return fig


def plot_ladder_summary(
    ladder_fit: pd.DataFrame,
    band_recovery: pd.DataFrame,
    *,
    region: str,
    figsize: tuple[float, float] = (7.4, 2.5),
):
    """The ladder in numbers: (a) one region per fixation type, (b) every region, (c) where in the spectrum the network acts."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    rungs = sorted(ladder_fit["n_partners"].unique())
    rng = np.random.default_rng(1)

    ax = axes[0]
    per = ladder_fit[ladder_fit["region"] == region].groupby(["n_partners", "label", "seed", "condition"])["r2_vs_ceiling"].mean().reset_index()
    for k, condition in enumerate(CONDITION_ORDER):
        block = per[per["condition"] == condition]
        mean = block.groupby("n_partners")["r2_vs_ceiling"].mean()
        ax.scatter(block["n_partners"] + (k - 1) * 0.12 + rng.uniform(-0.03, 0.03, len(block)), block["r2_vs_ceiling"],
                   s=7, color=CONDITION_COLORS[condition], alpha=0.5, lw=0, zorder=2)
        ax.plot(mean.index + (k - 1) * 0.12, mean.values, color=CONDITION_COLORS[condition], lw=1.4, marker="o", ms=3.5,
                zorder=3, label=CONDITION_SHORT_LABELS[condition])
    ax.set_xticks(rungs)
    ax.set_xlabel("other regions present")
    ax.set_ylabel("$R^2$ / noise ceiling")
    ax.set_title(f"{REGION_LABELS.get(region, region)}, per fixation type", fontsize=8)
    ax.legend(loc="lower right", fontsize=6)
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    per_region = ladder_fit.groupby(["region", "n_partners", "label", "seed"])["r2_vs_ceiling"].mean().reset_index()
    for k, r in enumerate([r for r in DISPLAY_REGION_ORDER if r in set(per_region["region"])]):
        block = per_region[per_region["region"] == r]
        mean = block.groupby("n_partners")["r2_vs_ceiling"].mean()
        ax.scatter(block["n_partners"] + (k - 1.5) * 0.1 + rng.uniform(-0.02, 0.02, len(block)), block["r2_vs_ceiling"],
                   s=6, color=REGION_COLORS[r], alpha=0.45, lw=0, zorder=2)
        ax.plot(mean.index + (k - 1.5) * 0.1, mean.values, color=REGION_COLORS[r], lw=1.3, marker="o", ms=3, zorder=3, label=REGION_LABELS[r])
    ax.set_xticks(rungs)
    ax.set_xlabel("other regions present")
    ax.set_title("every region, fixation types pooled", fontsize=8)
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    nice_axis(ax)
    _panel(ax, "b")

    ax = axes[2]
    partners = ladder_fit.groupby(["label", "region"])["n_partners"].first().reset_index()
    rec = band_recovery.merge(partners, on=["label", "region"], how="inner")
    bands = [b for b in ("0-5 Hz", "5-10 Hz", "10-20 Hz", "20-50 Hz") if b in set(rec["band"])]
    shades = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(bands)))
    for shade, band in zip(shades, bands):
        block = rec[rec["band"] == band]
        mean = block.groupby("n_partners")["residual_fraction"].mean()
        ax.plot(mean.index, mean.values, color=shade, lw=1.3, marker="o", ms=3, label=band)
    ax.set_xticks(rungs)
    ax.set_ylim(0, None)
    ax.set_xlabel("other regions present")
    ax.set_ylabel("target power left in the residual")
    ax.set_title("where the network acts", fontsize=8)
    ax.legend(loc="upper right", fontsize=6)
    nice_axis(ax)
    _panel(ax, "c")
    fig.tight_layout()
    return fig


# ======================================================================================
# Grouped cost boxes (Figures 4, 6, 7) and the rank grid (Figure 5)
# ======================================================================================

CONDITION_OFFSETS = {"face_interactive": -0.27, "face_non_interactive": 0.0, "object": 0.27}
CONDITION_TWO_LINE = {"face_interactive": "Int\nface", "face_non_interactive": "Non-int\nface", "object": "Object"}


def condition_legend(ax, conditions: Sequence[str] = CONDITION_ORDER, *, loc: str = "above", fontsize: float = 6.0) -> None:
    """A three-swatch fixation-type legend, above the axes by default so it never covers a bracket."""
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CONDITION_COLORS[c], edgecolor="none") for c in conditions]
    labels = [CONDITION_SHORT_LABELS[c] for c in conditions]
    if loc == "above":
        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(conditions), fontsize=fontsize,
                  handlelength=1.0, columnspacing=1.0, frameon=False)
    else:
        ax.legend(handles, labels, loc=loc, fontsize=fontsize, handlelength=1.0, frameon=False)


def _grouped_condition_boxes(ax, frame: pd.DataFrame, *, value: str, group_col: str, groups: Sequence[str],
                             gap_tests: pd.DataFrame | None = None, gap_group_col: str | None = None,
                             width: float = 0.26, labels: Mapping[str, str] | None = None, ylabel: str = "",
                             legend: bool = True, label_fontsize: float = 6.4, rotation: float = 0.0,
                             ylim: tuple[float, float] | None = None, top: float | None = None, style: str = "bar",
                             filled_by: Mapping[str, bool] | None = None):
    """Three fixation-type marks per group -- bars (mean, SEM, every fit a dot) or boxes -- with the significant contrasts marked.

    ``filled_by`` maps a group to whether its marks are filled (the dense network is drawn
    hollow wherever it sits beside a constrained one). Bars start the y-axis at zero.
    """
    labels = dict(labels or {})
    filled_by = dict(filled_by or {})
    conditions = [c for c in CONDITION_ORDER if c in set(frame["condition"])]
    entries = []
    for g, group in enumerate(groups):
        for condition in conditions:
            block = frame[(frame[group_col] == group) & (frame["condition"] == condition)]
            entries.append({"x": g + CONDITION_OFFSETS[condition], "values": block[value].to_numpy(float),
                            "color": CONDITION_COLORS[condition], "filled": filled_by.get(group, True)})
    if style == "bar":
        draw_bars(ax, entries, width=width)
    else:
        draw_boxes(ax, entries, width=width)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([labels.get(g, g) for g in groups], fontsize=label_fontsize, rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif style == "bar":
        ax.set_ylim(bottom=0.0)
    ax.set_ylabel(ylabel)
    nice_axis(ax)
    if legend:
        condition_legend(ax, conditions)
    if gap_tests is not None and not gap_tests.empty:
        key = gap_group_col or group_col
        index = {g: k for k, g in enumerate(groups)}
        rows = gap_tests[gap_tests[key].isin(groups)]
        marks = contrasts_to_marks(rows, lambda row, side: index[row[key]] + CONDITION_OFFSETS[row[side]])
        data_top = float(frame[frame[group_col].isin(groups)][value].max()) if top is None else top
        mark_contrasts(ax, marks, top=data_top)


def plot_architecture_cost(
    cost_per_seed: pd.DataFrame,
    gap_tests: pd.DataFrame,
    pooled_tests: pd.DataFrame,
    *,
    constraints: Sequence[str] = ("no inter-regional connections", "within-region rank 1", "inter-regional rank 1"),
    figsize: tuple[float, float] = (7.4, 2.8),
):
    """Removing or squeezing one kind of connection, against the dense network.

    (a) cost per fixation type for each constraint (bars: mean and SEM; dots: fits); the
    marks are fixation-type gaps the constraint widens significantly (Welch's t on
    per-seed gaps against the dense fits, Holm within the panel). (b) the same cost pooled
    over fixation types, one value per seed, with Welch's t between constraints.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    _grouped_condition_boxes(ax, cost_per_seed, value="cost", group_col="arm", groups=list(constraints), gap_tests=gap_tests,
                             labels=ARCHITECTURE_LABELS, ylabel="cost vs dense network  ($\\Delta R^2$ / ceiling)")
    _panel(ax, "a")

    ax = axes[1]
    pooled = cost_per_seed.groupby(["arm", "seed"])["cost"].mean().reset_index()
    entries = [{"x": k, "values": pooled.loc[pooled["arm"] == c, "cost"].to_numpy(float), "color": NEUTRAL_EDGE_COLOR}
               for k, c in enumerate(constraints)]
    draw_bars(ax, entries, width=0.55)
    ax.set_xticks(range(len(constraints)))
    ax.set_xticklabels([SHORT_CONSTRAINT_LABELS.get(c, c) for c in constraints], fontsize=6.4)
    ax.set_xlim(-0.6, len(constraints) - 0.4)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("cost, fixation types pooled")
    nice_axis(ax)
    index = {c: k for k, c in enumerate(constraints)}
    mark_contrasts(ax, contrasts_to_marks(pooled_tests, lambda row, side: index[row[side]]), top=float(pooled["cost"].max()))
    _panel(ax, "b")
    fig.tight_layout()
    return fig


def plot_rank_grid_composite(
    grid_fit: pd.DataFrame,
    dense_fit: pd.DataFrame,
    *,
    ranks: Sequence[int],
    hidden_units: int,
    bar: float = 0.98,
    selected: str | None = None,
    figsize: tuple[float, float] = (7.4, 2.6),
):
    """(a) the within × inter-regional rank grid, worst fixation type; (b) the two marginals; (c) every configuration against total drive rank."""
    per = grid_fit.groupby(["label", "rank_within", "rank_cross", "seed", "condition"])["r2_vs_ceiling"].mean().reset_index()
    worst = per.groupby(["label", "rank_within", "rank_cross", "seed"])["r2_vs_ceiling"].min().reset_index()
    config = worst.groupby(["label", "rank_within", "rank_cross"])["r2_vs_ceiling"].agg(["mean", "std"]).reset_index()
    dense_value = float(dense_fit.groupby(["seed", "condition"])["r2_vs_ceiling"].mean().groupby("seed").min().mean())

    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    ax = axes[0]
    matrix = np.full((len(ranks), len(ranks)), np.nan)
    for i, rw in enumerate(ranks):
        for j, rc in enumerate(ranks):
            row = config[(config["rank_within"] == rw) & (config["rank_cross"] == rc)]
            if not row.empty:
                matrix[i, j] = row["mean"].iloc[0]
    ax.imshow(matrix, origin="lower", cmap="cividis", vmin=0.93, vmax=1.0, aspect="auto")
    for i, rw in enumerate(ranks):
        for j, rc in enumerate(ranks):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=5.6,
                        color="white" if matrix[i, j] < 0.972 else INK, fontweight="bold" if matrix[i, j] >= bar else "normal")
            if selected is not None and f"w{rw}_c{rc}" == selected:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor="#c0392b", lw=1.8, zorder=5))
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([str(r) for r in ranks])
    ax.set_xlabel("inter-regional rank $r_c$")
    ax.set_ylabel("within-region rank $r_w$")
    ax.set_title(f"worst fixation type, $R^2$/ceiling (bold ≥ {bar})", fontsize=7.5)
    ax.tick_params(length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel(ax, "a")

    ax = axes[1]
    cross_m = config[config["rank_within"] == hidden_units].sort_values("rank_cross")
    within_m = config[config["rank_cross"] == hidden_units].sort_values("rank_within")
    x = np.arange(len(ranks))
    if not cross_m.empty:
        ax.errorbar(x[: len(cross_m)], cross_m["mean"], yerr=cross_m["std"], color=CROSS_COLOR, marker="o", ms=4, lw=1.4,
                    capsize=2, label="inter-regional rank $r_c$\n(within dense)")
    if not within_m.empty:
        ax.errorbar(x[: len(within_m)], within_m["mean"], yerr=within_m["std"], color=WITHIN_COLOR, marker="o", ms=4, lw=1.4,
                    capsize=2, label="within-region rank $r_w$\n(inter-regional dense)")
    ax.axhline(dense_value, color=INK, lw=1.3, ls="--", label="dense network", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("rank of the bottleneck")
    ax.set_ylabel("worst fixation type, $R^2$ / ceiling")
    ax.legend(loc="center left", fontsize=5.6)
    nice_axis(ax)
    _panel(ax, "b")

    ax = axes[2]
    interior = config[(config["rank_within"] < hidden_units) & (config["rank_cross"] < hidden_units)]
    ax.scatter(interior["rank_within"] + 3 * interior["rank_cross"], interior["mean"], s=16, facecolor="#d5dde5",
               edgecolor=NEUTRAL_EDGE_COLOR, lw=0.6, zorder=2, label="grid configurations")
    ax.scatter(cross_m["rank_within"] + 3 * cross_m["rank_cross"], cross_m["mean"], s=18, color=CROSS_COLOR, zorder=3, label="inter-regional marginal")
    ax.scatter(within_m["rank_within"] + 3 * within_m["rank_cross"], within_m["mean"], s=18, color=WITHIN_COLOR, zorder=3, label="within marginal")
    if selected is not None:
        row = config[config["label"] == selected]
        if not row.empty:
            ax.scatter(row["rank_within"] + 3 * row["rank_cross"], row["mean"], s=60, facecolor="none", edgecolor="#c0392b", lw=1.4, zorder=4)
    ax.axhline(bar, color=INK, lw=0.7, ls="--")
    ax.set_xlabel("total drive rank  $r_w + 3\\,r_c$")
    ax.legend(loc="lower right", fontsize=5.6)
    nice_axis(ax)
    _panel(ax, "c")
    fig.tight_layout()
    return fig


def plot_bottleneck_cost_by_condition(
    cost_per_seed: pd.DataFrame,
    gap_tests: pd.DataFrame,
    *,
    ranks: Sequence[int],
    figsize: tuple[float, float] = (7.4, 2.8),
):
    """Cost per fixation type along each marginal, boxes per rank, significant fixation-type gaps bracketed.

    ``cost_per_seed`` carries ``side`` (``inter-regional`` / ``within-region``), ``rank``,
    ``seed``, ``condition`` and ``cost``; ``gap_tests`` is keyed by ``arm`` =
    ``"<side> rank <r>"``.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    top = float(cost_per_seed["cost"].max())
    ylim = (0.0, top * 1.02)
    for ax, side, letter, xlabel in ((axes[0], "inter-regional", "a", "inter-regional rank $r_c$  (within-region dense)"),
                                     (axes[1], "within-region", "b", "within-region rank $r_w$  (inter-regional dense)")):
        block = cost_per_seed[cost_per_seed["side"] == side].copy()
        block["group"] = block["rank"].astype(int)
        groups = [int(r) for r in ranks if int(r) in set(block["group"])]
        tests = gap_tests[gap_tests["arm"].str.startswith(side)].copy() if gap_tests is not None and not gap_tests.empty else None
        if tests is not None:
            tests["group"] = tests["arm"].str.extract(r"(\d+)$").astype(int)
        _grouped_condition_boxes(ax, block, value="cost", group_col="group", groups=groups, gap_tests=tests, gap_group_col="group",
                                 labels={g: str(g) for g in groups}, ylabel="cost vs dense network  ($\\Delta R^2$ / ceiling)" if letter == "a" else "",
                                 legend=letter == "a", label_fontsize=7, ylim=ylim, top=top)
        ax.set_xlabel(xlabel)
        if letter == "b":
            ax.tick_params(labelleft=False)
        _panel(ax, letter)
    fig.tight_layout()
    return fig


def _draw_ensemble_row(axes, fit_long: pd.DataFrame, cost_per_seed: pd.DataFrame, arm_tests: pd.DataFrame | None,
                       gap_tests: pd.DataFrame | None, *, constrained: str, letters: str = "abc") -> None:
    """The three ensemble panels: shortfall from the ceiling by arm, cost per fixation type, unexplained-variance ratio."""
    conditions = [c for c in CONDITION_ORDER if c in set(fit_long["condition"])]
    ax = axes[0]
    entries = []
    for j, condition in enumerate(conditions):
        for arm, dx, filled in (("dense", -0.2, False), (constrained, 0.2, True)):
            block = fit_long[(fit_long["arm"] == arm) & (fit_long["condition"] == condition)]
            entries.append({"x": j + dx, "values": 1.0 - block["r2_vs_ceiling"].to_numpy(float), "color": CONDITION_COLORS[condition], "filled": filled})
    draw_bars(ax, entries, width=0.36)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_TWO_LINE[c] for c in conditions], fontsize=6.0)
    shortfall = 1.0 - fit_long[fit_long["arm"].isin(["dense", constrained])]["r2_vs_ceiling"]
    ax.set_ylim(min(0.0, float(shortfall.min()) * 1.1), None)
    ax.set_ylabel("shortfall from the ceiling\n(1 − $R^2$ / ceiling, mean over regions)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, lw=0.8), plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor="none")]
    ax.legend(handles, ["dense", "constrained"], loc="lower right", bbox_to_anchor=(1.02, 1.0), ncol=2, fontsize=6, handlelength=1.0,
              columnspacing=0.8, frameon=False)
    nice_axis(ax)
    index = {c: k for k, c in enumerate(conditions)}
    if arm_tests is not None and not arm_tests.empty:
        marks = contrasts_to_marks(arm_tests, lambda row, side: index[row["condition"]] + (-0.2 if row[side] == "dense" else 0.2))
        mark_contrasts(ax, marks, top=float(shortfall.max()))
    _panel(ax, letters[0])

    ax = axes[1]
    block = cost_per_seed[cost_per_seed["arm"] == constrained]
    draw_bars(ax, [{"x": j, "values": block.loc[block["condition"] == c, "cost"].to_numpy(float), "color": CONDITION_COLORS[c]}
                   for j, c in enumerate(conditions)], width=0.6)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_TWO_LINE[c] for c in conditions], fontsize=6.0)
    ax.set_ylabel("cost of the constraint\n($\\Delta R^2$ / ceiling vs dense)")
    nice_axis(ax)
    if gap_tests is not None and not gap_tests.empty:
        rows = gap_tests[gap_tests["arm"] == constrained]
        mark_contrasts(ax, contrasts_to_marks(rows, lambda row, side: index[row[side]]), top=float(block["cost"].max()))
    _panel(ax, letters[1])

    ax = axes[2]
    draw_bars(ax, [{"x": j, "values": block.loc[block["condition"] == c, "unexplained_ratio"].to_numpy(float), "color": CONDITION_COLORS[c]}
                   for j, c in enumerate(conditions)], width=0.6)
    ax.set_ylim(bottom=0.0)
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=1)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_TWO_LINE[c] for c in conditions], fontsize=6.0)
    ax.set_ylabel("unexplained variance,\nconstrained / dense")
    nice_axis(ax)
    _panel(ax, letters[2])


def plot_ensemble_cost(
    fit_long: pd.DataFrame,
    cost_per_seed: pd.DataFrame,
    arm_tests: pd.DataFrame,
    gap_tests: pd.DataFrame,
    *,
    constrained: str,
    figsize: tuple[float, float] = (7.4, 2.7),
):
    """The selected model against the dense network, ten fits each: shortfall by arm, cost per fixation type, unexplained ratio."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    _draw_ensemble_row(axes, fit_long, cost_per_seed, arm_tests, gap_tests, constrained=constrained)
    fig.tight_layout()
    return fig


def plot_ensemble_and_constraints(
    fit_long: pd.DataFrame,
    cost_per_seed: pd.DataFrame,
    arm_tests: pd.DataFrame,
    gap_tests_selected: pd.DataFrame,
    gap_tests_all: pd.DataFrame,
    *,
    constrained: str,
    constraints: Sequence[str],
    figsize: tuple[float, float] = (7.4, 2.7),
):
    """The ensemble panels (a-c) and the every-constraint summary (d) in one row.

    (a) shortfall from the ceiling per fixation type, dense against constrained; (b) the
    cost of the selected constraint per fixation type; (c) the unexplained-variance ratio;
    (d) the cost of every constraint in the chapter per fixation type, on one axis.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1.05, 0.85, 0.85, 1.9]})
    _draw_ensemble_row(axes[:3], fit_long, cost_per_seed, arm_tests, gap_tests_selected, constrained=constrained)
    ax = axes[3]
    groups = [c for c in constraints if c in set(cost_per_seed["arm"])]
    _grouped_condition_boxes(ax, cost_per_seed, value="cost", group_col="arm", groups=groups, gap_tests=gap_tests_all,
                             labels=SHORT_CONSTRAINT_LABELS, ylabel="cost vs dense network\n($\\Delta R^2$ / ceiling)", label_fontsize=5.8)
    _panel(ax, "d")
    fig.tight_layout(w_pad=0.6)
    return fig


def plot_constraint_cost_summary(
    cost_per_seed: pd.DataFrame,
    gap_tests: pd.DataFrame,
    *,
    constraints: Sequence[str],
    figsize: tuple[float, float] = (7.4, 2.8),
):
    """Every constraint in the chapter, costed against the dense network per fixation type, on one axis."""
    fig, ax = plt.subplots(figsize=figsize)
    groups = [c for c in constraints if c in set(cost_per_seed["arm"])]
    _grouped_condition_boxes(ax, cost_per_seed, value="cost", group_col="arm", groups=groups, gap_tests=gap_tests,
                             labels=SHORT_CONSTRAINT_LABELS, ylabel="cost vs dense network  ($\\Delta R^2$ / ceiling)", label_fontsize=6.4)
    fig.tight_layout()
    return fig


# ======================================================================================
# Figure 8 -- pair lesions
# ======================================================================================


def plot_pair_lesion_summary(
    pair_share: pd.DataFrame,
    share_tests: pd.DataFrame,
    damage_per_fit: pd.DataFrame,
    damage_tests: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.4, 2.6),
):
    """(a) each pair's share of pair-lesion damage, (b) damage per lesion family beside its random control, (c) ranking agreement.

    (a) one bar per pair and arm, fixation types pooled (mean, SEM, every fit a dot); the
    dashed line is one sixth, and a star above a bar marks a pair whose share differs from
    one sixth (one-sample t, Holm). (b) per-fit mean damage per family, in units of the
    target's total variance, beside the matched control that silences the same number of
    randomly chosen weights (black line); a star marks a family whose damage differs from
    its control (paired t by fit, Holm). (c) Kendall's tau between fits' lesion rankings per
    family and arm, against the 95th percentile of the permutation null (black line).
    """
    arms = [a for a in ("dense", "constrained") if a in set(pair_share["arm"])]
    pairs = [p for p in ("ofc↔bla", "ofc↔dmpfc", "ofc↔accg", "bla↔dmpfc", "bla↔accg", "dmpfc↔accg") if p in set(pair_share["pair"])]
    offsets = dict(zip(arms, (-0.2, 0.2) if len(arms) == 2 else (0.0,)))
    families = [f for f in ("directed", "bidirectional", "within-region", "isolation") if f in set(damage_per_fit["family"])]
    family_labels = {"directed": "one pathway", "bidirectional": "one pair", "within-region": "own block", "isolation": "isolated region"}
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})

    ax = axes[0]
    entries = []
    for j, pair in enumerate(pairs):
        for arm in arms:
            block = pair_share[(pair_share["arm"] == arm) & (pair_share["pair"] == pair)]
            entries.append({"x": j + offsets[arm], "values": block["share"].to_numpy(float), "color": ARM_COLORS[arm], "filled": arm != "dense"})
    draw_bars(ax, entries, width=0.36)
    ax.axhline(1 / 6, color=INK, lw=1.0, ls="--", zorder=1)
    if share_tests is not None and not share_tests.empty:
        for _, row in share_tests[share_tests["significant"]].iterrows():
            own = pair_share[(pair_share["arm"] == row["arm"]) & (pair_share["pair"] == row["pair"])]["share"].max()
            ax.text(pairs.index(row["pair"]) + offsets[row["arm"]], own + 0.004, row["stars"], ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(["↔".join(REGION_LABELS.get(r, r) for r in pair.split("↔")) for pair in pairs], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("share of pair-lesion damage")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=ARM_COLORS["dense"], lw=0.8),
               plt.Rectangle((0, 0), 1, 1, facecolor=ARM_COLORS["constrained"], edgecolor="none")]
    ax.legend(handles, ["dense", "constrained"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, handlelength=1.0)
    nice_axis(ax)
    _panel(ax, "a")

    ax = axes[1]
    entries = []
    for j, family in enumerate(families):
        for arm in arms:
            block = damage_per_fit[(damage_per_fit["arm"] == arm) & (damage_per_fit["family"] == family)]
            entries.append({"x": j + offsets[arm], "values": block["damage"].to_numpy(float), "color": ARM_COLORS[arm], "filled": arm != "dense"})
            ax.plot([j + offsets[arm] - 0.18, j + offsets[arm] + 0.18], [block["control"].mean()] * 2, color=INK, lw=1.4, zorder=6)
    draw_bars(ax, entries, width=0.36)
    if damage_tests is not None and not damage_tests.empty:
        for _, row in damage_tests[damage_tests["significant"]].iterrows():
            own = damage_per_fit[(damage_per_fit["arm"] == row["arm"]) & (damage_per_fit["family"] == row["family"])]
            ax.text(families.index(row["family"]) + offsets[row["arm"]], float(max(own["damage"].max(), own["control"].max())) + 0.15,
                    row["stars"], ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([family_labels[f] for f in families], fontsize=7, rotation=30, ha="right")
    ax.set_ylim(0, float(max(damage_per_fit["damage"].max(), damage_per_fit["control"].max())) * 1.18)
    ax.set_ylabel("damage (÷ target variance)")
    ax.plot([], [], color=INK, lw=1.4, label="random-weight control")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8)
    nice_axis(ax)
    _panel(ax, "b")

    ax = axes[2]
    kinds = [k for k in families if k in set(ranking["lesion_kind"])]
    for arm in arms:
        block = ranking[ranking["arm"] == arm].set_index("lesion_kind").loc[kinds]
        x = np.arange(len(kinds)) + offsets[arm]
        ax.bar(x, block["tau_mean"], width=0.36, facecolor=ARM_COLORS[arm] if arm != "dense" else "white",
               edgecolor=ARM_COLORS[arm], lw=0.9, zorder=2)
        for xi, null in zip(x, block["null_p95"]):
            ax.plot([xi - 0.18, xi + 0.18], [null] * 2, color=INK, lw=1.4, zorder=4)
    ax.plot([], [], color=INK, lw=1.4, label="permutation null, 95th pct.")
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([family_labels[f] for f in kinds], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("ranking agreement (Kendall's τ)")
    ax.set_ylim(-0.15, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8)
    nice_axis(ax)
    _panel(ax, "c")
    fig.tight_layout()
    return fig


# ======================================================================================
# Figure 8b -- what a lesion does to a region's local dynamics
# ======================================================================================


def plot_lesion_spectra_clouds(
    spectra: pd.DataFrame,
    *,
    region: str,
    lesion: str,
    arm: str,
    figsize: tuple[float, float] = (7.4, 4.6),
):
    """Eigenvalues of one region's local linearisation, intact (grey) and after one lesion (colour), per fixation type.

    Top row: at the fixed point nearest the end of the trajectory. Bottom row: along the
    trajectory (pooled over every fifth state). The unit circle marks the boundary between
    contracting and expanding modes.
    """
    conditions = [c for c in CONDITION_ORDER if c in set(spectra["condition"])]
    fig, axes = plt.subplots(2, len(conditions), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 200)
    for i, (where, label) in enumerate((("fixed_point", "at the nearest fixed point"), ("trajectory", "along the trajectory"))):
        for j, condition in enumerate(conditions):
            ax = axes[i][j]
            ax.plot(np.cos(theta), np.sin(theta), color=MUTED_INK, lw=0.6, ls="--", zorder=1)
            ax.axhline(0, color="#dddddd", lw=0.5, zorder=0)
            ax.axvline(0, color="#dddddd", lw=0.5, zorder=0)
            base = spectra[(spectra["region"] == region) & (spectra["condition"] == condition) & (spectra["where"] == where)]
            intact = base[base["lesion"] == "intact"]
            lesioned = base[base["lesion"] == lesion]
            size = 9 if where == "fixed_point" else 4
            ax.scatter(intact["re"], intact["im"], s=size, color="#9a9a9a", alpha=0.7 if where == "fixed_point" else 0.35, lw=0, zorder=2, label="intact")
            ax.scatter(lesioned["re"], lesioned["im"], s=size, color=CONDITION_COLORS[condition], alpha=0.85 if where == "fixed_point" else 0.4,
                       lw=0, zorder=3, label=lesion)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(CONDITION_SHORT_LABELS[condition], fontsize=8, color=CONDITION_COLORS[condition])
            if j == 0:
                ax.set_ylabel(f"{label}\nimaginary part", fontsize=7)
            if i == 1:
                ax.set_xlabel("real part")
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=5.8, markerscale=1.5)
            nice_axis(ax)
    scope = ("whole network · eigenvalues of the full Jacobian" if region == "network"
             else f"{REGION_LABELS.get(region, region)} · eigenvalues of its own block's linearisation")
    fig.suptitle(f"{scope} · {arm} network · lesion: {lesion}", fontsize=8, y=1.0)
    fig.tight_layout()
    return fig


def plot_lesion_spectra_clouds_compact(
    spectra: pd.DataFrame,
    *,
    lesion: str,
    condition: str = "face_interactive",
    panels: Sequence[tuple[str, str]] = (("dense", "bla"), ("constrained", "network")),
    figsize: tuple[float, float] = (7.4, 2.15),
):
    """Eigenvalues of the local linearisation for one fixation type, intact (grey) and after one lesion (colour), in one row.

    For each ``(arm, scope)`` in ``panels`` two panels: at the fixed point nearest the end of
    the trajectory, and along the trajectory (every fifth state pooled). The unit circle marks
    the boundary between contracting and expanding modes.
    """
    wheres = (("fixed_point", "at the fixed point"), ("trajectory", "along the trajectory"))
    fig, axes = plt.subplots(1, 2 * len(panels), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 200)
    color = CONDITION_COLORS[condition]
    k = 0
    for arm, scope in panels:
        scope_label = "whole network" if scope == "network" else f"{REGION_LABELS.get(scope, scope)} block"
        for where, where_label in wheres:
            ax = axes[0][k]
            ax.plot(np.cos(theta), np.sin(theta), color=MUTED_INK, lw=0.6, ls="--", zorder=1)
            ax.axhline(0, color="#dddddd", lw=0.5, zorder=0)
            ax.axvline(0, color="#dddddd", lw=0.5, zorder=0)
            base = spectra[(spectra["arm"] == arm) & (spectra["region"] == scope) & (spectra["condition"] == condition) & (spectra["where"] == where)]
            intact = base[base["lesion"] == "intact"]
            lesioned = base[base["lesion"] == lesion]
            size = 8 if where == "fixed_point" else 3.5
            ax.scatter(intact["re"], intact["im"], s=size, color="#9a9a9a", alpha=0.7 if where == "fixed_point" else 0.35, lw=0, zorder=2, label="intact")
            ax.scatter(lesioned["re"], lesioned["im"], s=size, color=color, alpha=0.85 if where == "fixed_point" else 0.4, lw=0, zorder=3, label=f"{lesion} silenced")
            ax.set_aspect("equal")
            ax.set_title(f"{arm} · {scope_label}\n{where_label}", fontsize=7)
            ax.set_xlabel("real part", fontsize=6.8)
            if k == 0:
                ax.set_ylabel("imaginary part", fontsize=6.8)
                ax.legend(loc="upper left", fontsize=5.6, markerscale=1.6, handletextpad=0.3)
            ax.tick_params(labelsize=6.2)
            nice_axis(ax)
            _panel(ax, "abcdefgh"[k])
            k += 1
    fig.suptitle(f"{CONDITION_SHORT_LABELS.get(condition, condition)} · eigenvalues of the linearisation, intact and lesioned", fontsize=8, y=1.02)
    fig.tight_layout()
    return fig


def plot_lesion_spectra_distance(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    lesion_kind: str = "bidirectional",
    where: str = "trajectory",
    figsize: tuple[float, float] = (7.4, 2.4),
):
    """Wasserstein distance between the lesioned and intact spectra: one panel per arm, each region's block and the whole network.

    Bars pool every (fit, lesion) of ``lesion_kind`` at one linearisation point (``where``:
    ``"trajectory"`` or ``"fixed_point"``); marks are significant fixation-type contrasts
    within a scope (paired t by fit x lesion, Holm within the panel). The two panels share
    the y-axis.
    """
    arms = [a for a in ("dense", "constrained") if a in set(summary["arm"])]
    scopes = [r for r in DISPLAY_REGION_ORDER if r in set(summary["region"])] + (["network"] if "network" in set(summary["region"]) else [])
    labels = {**REGION_LABELS, "network": "whole\nnetwork"}
    block_all = summary[(summary["where"] == where) & (summary["lesion_kind"] == lesion_kind) & (summary["region"].isin(scopes))]
    top = float(block_all["w2_to_intact"].max())
    where_label = {"trajectory": "along the trajectory", "fixed_point": "at the fixed point"}.get(where, where)
    fig, axes = plt.subplots(1, len(arms), figsize=figsize, squeeze=False)
    for j, arm in enumerate(arms):
        ax = axes[0][j]
        block = block_all[block_all["arm"] == arm]
        panel_tests = None
        if tests is not None and not tests.empty:
            panel_tests = tests[(tests["arm"] == arm) & (tests["where"] == where)]
        _grouped_condition_boxes(ax, block, value="w2_to_intact", group_col="region", groups=scopes, gap_tests=panel_tests,
                                 gap_group_col="region", labels=labels,
                                 ylabel="spectrum change ($W_2$)" if j == 0 else "",
                                 legend=False, label_fontsize=6.6, ylim=(0.0, top * 1.02), top=top)
        if j > 0:
            ax.tick_params(labelleft=False)
        ax.set_title(f"{arm} network · {where_label}", fontsize=8)
        _panel(ax, "ab"[j])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CONDITION_COLORS[c], edgecolor="none") for c in CONDITION_ORDER]
    fig.legend(handles, [CONDITION_SHORT_LABELS[c] for c in CONDITION_ORDER], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=6.5, handlelength=1.0, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def _draw_flow_panel(ax, entry: Mapping[str, object], *, xs: np.ndarray, ys: np.ndarray, condition: str, norm, arrow_stride: int = 1) -> None:
    """One flow-field panel: speed background, unit arrows, the trajectory light to dark, fixed points as stars.

    Arrows are line segments rather than a Quiver because the repo's PDF export strips
    clipping and an unclipped Quiver reports a bounding box thousands of inches wide.
    """
    gx, gy = np.meshgrid(xs, ys)
    ax.pcolormesh(gx, gy, entry["speed"], cmap="Greys", norm=norm, shading="nearest", alpha=0.55, zorder=1)
    s = arrow_stride
    magnitude = np.hypot(entry["u"], entry["v"])
    spacing = 0.8 * float(min(xs[1] - xs[0], ys[1] - ys[0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(magnitude > 0, entry["u"] / magnitude, 0.0)[::s, ::s].ravel()
        uy = np.where(magnitude > 0, entry["v"] / magnitude, 0.0)[::s, ::s].ravel()
    tails = np.stack([gx[::s, ::s].ravel(), gy[::s, ::s].ravel()], axis=-1)
    tips = tails + spacing * np.stack([ux, uy], axis=-1)
    cos_a, sin_a = np.cos(np.radians(28)), np.sin(np.radians(28))
    head = 0.35 * spacing
    left = tips - head * np.stack([ux * cos_a - uy * sin_a, uy * cos_a + ux * sin_a], axis=-1)
    right = tips - head * np.stack([ux * cos_a + uy * sin_a, uy * cos_a - ux * sin_a], axis=-1)
    segments = np.concatenate([np.stack([tails, tips], axis=1), np.stack([tips, left], axis=1), np.stack([tips, right], axis=1)])
    ax.add_collection(LineCollection(segments, colors="#4b4b4b", linewidths=0.5, alpha=0.8, zorder=2))
    trajectory = np.asarray(entry["trajectory"])
    color = CONDITION_COLORS.get(condition, INK)
    shades = plt.get_cmap("viridis")(np.linspace(0.15, 0.95, len(trajectory) - 1))
    for t in range(len(trajectory) - 1):
        ax.plot(trajectory[t: t + 2, 0], trajectory[t: t + 2, 1], color=shades[t], linewidth=1.6, zorder=4)
    ax.scatter(*trajectory[0], s=26, color=color, edgecolor="white", linewidth=0.6, zorder=5, marker="o")
    ax.scatter(*trajectory[-1], s=30, color=color, edgecolor="white", linewidth=0.6, zorder=5, marker="s")
    outside = 0
    for point, stable in zip(entry["fixed_points"], entry["fixed_point_stable"]):
        if not (xs[0] <= point[0] <= xs[-1] and ys[0] <= point[1] <= ys[-1]):
            outside += 1
            continue
        ax.scatter(point[0], point[1], s=70, marker="*", facecolor=color if stable else "white", edgecolor=color, linewidth=1.0, zorder=6)
    if outside:
        ax.text(0.98, 0.02, f"+{outside} fixed point{'s' if outside > 1 else ''} outside the window", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.6, color=INK)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.tick_params(labelsize=6.5)


def plot_flow_fields_two_arms(fields_by_arm: Mapping[str, Mapping[str, object]], *, figsize: tuple[float, float] = (7.4, 5.4)):
    """The flow of each fixation type's map, one row per arm (a: dense, b: constrained), one representative fit each.

    Each row has its own state plane (the two leading principal axes of that fit's pooled
    hidden-state trajectories) and its own speed scale; the three panels of a row share
    both. The row title names the arm and the fit. A legend at the foot explains the
    trajectory (circle: first bin, square: last bin, light to dark in time), the fixed
    points (stars, filled if stable) and the background (speed of the flow, light is slow).
    """
    from matplotlib.colors import LogNorm
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    arms = [a for a in ("dense", "constrained") if a in fields_by_arm]
    conditions = [c for c in CONDITION_ORDER if all(c in fields_by_arm[a]["conditions"] for a in arms)]
    fig, axes = plt.subplots(len(arms), len(conditions), figsize=figsize, squeeze=False)
    for i, arm in enumerate(arms):
        fields = fields_by_arm[arm]
        xs, ys = np.asarray(fields["xs"]), np.asarray(fields["ys"])
        speeds = np.concatenate([np.asarray(fields["conditions"][c]["speed"]).ravel() for c in conditions])
        norm = LogNorm(vmin=max(float(speeds.min()), 1e-3), vmax=float(speeds.max()))
        for j, condition in enumerate(conditions):
            ax = axes[i][j]
            _draw_flow_panel(ax, fields["conditions"][condition], xs=xs, ys=ys, condition=condition, norm=norm)
            ax.set_title(CONDITION_SHORT_LABELS.get(condition, condition), fontsize=7.5, color=CONDITION_COLORS.get(condition, INK), pad=3)
            ax.set_xlabel(f"axis 1 ({fields['explained'][0]:.0%} of state variance)", fontsize=6.6)
            if j > 0:
                ax.tick_params(labelleft=False)
        axes[i][0].set_ylabel(f"axis 2 ({fields['explained'][1]:.0%})", fontsize=6.6)
        _panel(axes[i][0], "ab"[i])
    viridis = plt.get_cmap("viridis")
    handles = [
        (Line2D([], [], color=viridis(0.15), lw=1.8), Line2D([], [], color=viridis(0.55), lw=1.8), Line2D([], [], color=viridis(0.95), lw=1.8)),
        Line2D([], [], marker="o", color=INK, ls="none", ms=4.5),
        Line2D([], [], marker="s", color=INK, ls="none", ms=4.5),
        Line2D([], [], marker="*", mfc="white", mec=INK, ls="none", ms=8),
        Line2D([], [], marker="*", mfc=INK, mec=INK, ls="none", ms=8),
        Line2D([], [], color="#4b4b4b", lw=0.6),
        Patch(facecolor="#dcdcdc", edgecolor="#8a8a8a", lw=0.5),
    ]
    labels = [
        "trajectory, early → late", "first bin (−495 ms, trained initial state)", "last bin (+495 ms)",
        "fixed point, saddle", "fixed point, stable", "arrows: direction of the flow", "background: speed of the flow (light = slow)",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6, frameon=False, handlelength=2.2, columnspacing=1.4,
               handler_map={tuple: HandlerTuple(ndivide=3, pad=0.0)}, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=2.6, rect=(0, 0.075, 1, 1))
    for i, arm in enumerate(arms):
        left, right = axes[i][0].get_position(), axes[i][-1].get_position()
        seed = fields_by_arm[arm].get("seed", "")
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.052, f"{arm} network · representative fit, seed {seed}",
                 ha="center", va="bottom", fontsize=8.5, color=INK)
    return fig


# ======================================================================================
# Figure 9 -- dynamics, per fixation type
# ======================================================================================

DYNAMICS_LABELS = {
    "state_extent": "state extent\n(distance from the mean state)",
    "state_speed": "state speed\n(state units per 10 ms bin)",
    "state_pr": "state dimensionality\n(participation ratio)",
    "trajectory_end_speed": "speed at the end of the window",
    "nearest_distance_to_trajectory_end": "end of window → nearest\nfixed point (state extents)",
    "network_n_expanding": "expanding modes of the local\nlinearisation along the trajectory",
}
DYNAMICS_MAIN: tuple[str, ...] = ("state_extent", "state_speed", "network_n_expanding")


def plot_dynamics_summary(
    long: pd.DataFrame,
    tests: pd.DataFrame | None = None,
    *,
    properties: Sequence[str] = DYNAMICS_MAIN,
    figsize: tuple[float, float] = (7.4, 2.6),
):
    """Per-fixation-type dynamical quantities, grouped by arm so the fixation types sit side by side.

    One panel per property; within each, the dense network's three fixation types on the
    left and the constrained network's on the right, bars (mean, SEM) with every fit a
    dot. ``tests`` holds paired contrasts between fixation types within each arm, keyed by
    ``property`` and ``arm``; significant ones are marked.
    """
    arms = [a for a in ("dense", "constrained") if a in set(long["arm"])]
    fig, axes = plt.subplots(1, len(properties), figsize=figsize, squeeze=False)
    for k, prop in enumerate(properties):
        ax = axes[0][k]
        block = long[long["property"] == prop]
        panel_tests = tests[tests["property"] == prop] if tests is not None and not tests.empty else None
        _grouped_condition_boxes(ax, block, value="value", group_col="arm", groups=arms, gap_tests=panel_tests, gap_group_col="arm",
                                 labels={"dense": "dense", "constrained": "constrained"}, ylabel=DYNAMICS_LABELS.get(prop, prop),
                                 legend=False, label_fontsize=7, style="bar", width=0.26, top=float(block["value"].max()))
        _panel(ax, "abcdefghi"[k])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CONDITION_COLORS[c], edgecolor="none") for c in CONDITION_ORDER]
    fig.legend(handles, [CONDITION_SHORT_LABELS[c] for c in CONDITION_ORDER], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=6.5, handlelength=1.0, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_lesion_extent_violins(
    lesion_extent: pd.DataFrame,
    tests: pd.DataFrame | None = None,
    *,
    figsize: tuple[float, float] = (7.4, 2.7),
):
    """Change in state extent after a lesion, relative to intact: violins per fixation type, pair lesions and isolations, both arms.

    Each violin pools every (fit, lesion) of that kind, in the project's violin style
    (quartile lines inside, cut at the data range). Marks are significant contrasts between
    fixation types (paired t by fit × lesion, Holm within the panel).
    """
    arms = [a for a in ("dense", "constrained") if a in set(lesion_extent["arm"])]
    kinds = [("bidirectional", "pair lesions"), ("isolation", "region isolated")]
    kind_order = [k for k, _ in kinds]
    fig, axes = plt.subplots(1, len(arms), figsize=figsize, sharey=True, squeeze=False)
    for j, arm in enumerate(arms):
        ax = axes[0][j]
        block = lesion_extent[(lesion_extent["arm"] == arm) & (lesion_extent["lesion_kind"].isin(kind_order))]
        draw_violins(ax, block, x="lesion_kind", y="rel_state_extent", hue="condition", order=kind_order, hue_order=CONDITION_ORDER,
                     palette=CONDITION_COLORS, width=0.8)
        ax.axhline(0, color=INK, lw=0.6, zorder=1)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([label for _, label in kinds])
        ax.set_xlim(-0.6, len(kinds) - 0.4)
        ax.set_xlabel("")
        ax.set_title(f"{arm} network", fontsize=8)
        ax.set_ylabel("state extent after the lesion,\nrelative to intact (Δ / intact)" if j == 0 else "")
        nice_axis(ax)
        if tests is not None and not tests.empty:
            rows = tests[tests["arm"] == arm]
            kind_index = {kind: g for g, (kind, _) in enumerate(kinds)}
            mark_contrasts(ax, contrasts_to_marks(rows, lambda row, side: kind_index[row["lesion_kind"]] + CONDITION_OFFSETS[row[side]]),
                           top=float(block["rel_state_extent"].max()))
        _panel(ax, "ab"[j])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CONDITION_COLORS[c], edgecolor="none") for c in CONDITION_ORDER]
    fig.legend(handles, [CONDITION_SHORT_LABELS[c] for c in CONDITION_ORDER], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=6.5, handlelength=1.0, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


__all__ = [
    "ARCHITECTURE_LABELS", "CONDITION_OFFSETS", "DYNAMICS_LABELS", "DYNAMICS_MAIN", "RUNG_LABELS", "SHORT_CONSTRAINT_LABELS",
    "condition_shades", "contrasts_to_marks", "draw_bars", "draw_boxes", "draw_violins", "mark_contrasts",
    "plot_architecture_cost", "plot_bottleneck_cost_by_condition", "plot_chapter_schematic", "plot_constraint_cost_summary",
    "plot_dynamics_summary", "plot_ensemble_and_constraints", "plot_ensemble_cost", "plot_flow_fields_two_arms", "plot_ladder_summary", "plot_ladder_traces", "plot_lesion_extent_violins",
    "plot_lesion_spectra_clouds", "plot_lesion_spectra_clouds_compact", "plot_lesion_spectra_distance", "plot_manipulation_schematic", "plot_pair_lesion_summary",
    "plot_rank_grid_composite",
]
