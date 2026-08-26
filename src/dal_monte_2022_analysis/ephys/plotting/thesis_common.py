"""Shared style, vocabulary and IO for thesis-facing single-unit figures.

Thesis figures have two constraints the exploratory plotting modules do not:
every panel must survive being dropped into Illustrator with its text still
editable and no clipping masks, and the four brain regions must read as one row
of directly comparable panels. This module centralizes both, so the figure
modules only describe *what* to draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dal_monte_2022_analysis.ephys.plotting.common import apply_plotting_config
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure

#: Canonical score name. The stored column is ``peakiness_score``; every
#: thesis-facing label goes through here so a rename never has to be chased
#: through figure code.
DPP_NAME = "Dominant-Peak Prominence"
DPP_ABBREV = "DPP"
DPP_AXIS_LABEL = "Dominant-Peak Prominence (a.u.)"
DPP_COLUMN = "peakiness_score"
DPP_PRIMARY_SYMBOL = "$P_1$"
DPP_SECONDARY_SYMBOL = "$P_2$"
DPP_FORMULA = r"$\mathrm{DPP}=\dfrac{P_1}{1+\lambda\,P_2/P_1}$"

REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")
REGION_LABELS: dict[str, str] = {
    "bla": "BLA",
    "accg": "ACCg",
    "dmpfc": "dmPFC",
    "ofc": "OFC",
}

#: Prominent, colourblind-safe palette used wherever region is the series
#: (currently only the UpSet panel). Distinct from the condition palette,
#: which the two never share a figure with.
REGION_COLORS: dict[str, str] = {
    "bla": "#0072B2",
    "accg": "#E69F00",
    "dmpfc": "#009E73",
    "ofc": "#CC79A7",
}

CONDITION_ORDER: tuple[str, ...] = ("face_interactive", "face_non_interactive", "object")
CONDITION_LABELS: dict[str, str] = {
    "face_interactive": "Interactive face",
    "face_non_interactive": "Non-interactive face",
    "object": "Object",
}
CONDITION_SHORT_LABELS: dict[str, str] = {
    "face_interactive": "Int face",
    "face_non_interactive": "Non-int face",
    "object": "Object",
}
CONDITION_COLORS: dict[str, str] = {
    "face_interactive": "#b64198",
    "face_non_interactive": "#97ca3d",
    "object": "#754c29",
}
#: Secondary encoding so category identity never rests on hue alone in fills.
CONDITION_HATCHES: dict[str, str] = {
    "face_interactive": "",
    "face_non_interactive": "///",
    "object": "...",
}

PAIR_ORDER: tuple[str, ...] = (
    "face_interactive__vs__face_non_interactive",
    "face_interactive__vs__object",
    "face_non_interactive__vs__object",
)
PAIR_LABELS: dict[str, str] = {
    "face_interactive__vs__face_non_interactive": "Int face\nvs non-int face",
    "face_interactive__vs__object": "Int face\nvs object",
    "face_non_interactive__vs__object": "Non-int face\nvs object",
}

#: The three 500 ms analysis windows, drawn as stacked horizontal bars beneath
#: every example trace. Ordered bottom-to-top as they are drawn.
ANALYSIS_WINDOWS_MS: tuple[tuple[str, float, float], ...] = (
    ("Pre", -500.0, 0.0),
    ("Peri", -250.0, 250.0),
    ("Post", 0.0, 500.0),
)
WINDOW_BAR_COLORS: tuple[str, ...] = ("#c7c7c7", "#8f8f8f", "#4f4f4f")

EXEMPLAR_STYLE_ORDER: tuple[str, ...] = ("phasic", "tonic")
EXEMPLAR_STYLE_LABELS: dict[str, str] = {
    "phasic": f"High {DPP_ABBREV}",
    "tonic": f"Low {DPP_ABBREV}",
}
EXEMPLAR_STYLE_COLORS: dict[str, str] = {"phasic": "#c03a2b", "tonic": "#2878b5"}
EXEMPLAR_STYLE_MARKERS: dict[str, str] = {"phasic": "o", "tonic": "s"}

NEUTRAL_FILL = "#c9d3dd"
NEUTRAL_EDGE = "#31485c"
INK = "#222222"
MUTED_INK = "#666666"


@dataclass
class ThesisFigureSettings:
    """Output settings shared by every thesis figure panel."""

    output_dir: Path
    #: Every panel is written once per extension. PDF carries the editable
    #: vector art; PNG is for pasting into slides and for notebook display.
    extensions: Sequence[str] = field(default_factory=lambda: ("pdf", "png"))
    pdf_compression: int = 0
    png_dpi: int = 400
    transparent: bool = True
    #: One row of four region panels is the default thesis layout.
    row_figure_width_in: float = 7.2
    row_figure_height_in: float = 2.1


def apply_thesis_plot_style(plotting_cfg: Optional[Mapping] = None) -> None:
    """Install Illustrator-friendly rcParams and the PDF clip-mask stripper.

    ``apply_plotting_config`` installs a ``savefig`` patch that removes clipping
    masks from PDF output; without it Illustrator opens every panel as a group
    locked behind a mask and the artwork cannot be edited in place.
    """
    apply_plotting_config(dict(plotting_cfg or {}))
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.dpi": 130,
            "path.simplify": False,
        }
    )


def region_label(region: object) -> str:
    """Display label for a region token."""
    return REGION_LABELS.get(str(region).strip().lower(), str(region).upper())


def readable_text_color(
    face_color: str,
    *,
    light: str = "#ffffff",
    dark: str = "#1f1f1f",
) -> str:
    """Pick white or ink for text drawn on top of ``face_color``.

    The condition palette spans a dark magenta, a light yellow-green and a dark
    brown, so a single fixed label colour is unreadable on at least one of them.
    Chooses by WCAG relative luminance rather than by eye.
    """
    red, green, blue = mpl.colors.to_rgb(face_color)

    def _linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * _linear(red) + 0.7152 * _linear(green) + 0.0722 * _linear(blue)
    return dark if luminance > 0.42 else light


def ordinal(value: float) -> str:
    """English ordinal for a percentile, e.g. 22 -> ``22nd``, 71 -> ``71st``."""
    number = int(round(float(value)))
    if 10 <= (number % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def bare_unit_uuid(value: object) -> str:
    """Unit id without the ``unit_uuid__`` prefix."""
    return str(value).replace("unit_uuid__", "")


def ordered_regions(observed: Iterable[str]) -> list[str]:
    """Regions in canonical thesis order, keeping any unexpected extras last."""
    seen = [str(region).strip().lower() for region in observed]
    ordered = [region for region in REGION_ORDER if region in seen]
    return ordered + [region for region in dict.fromkeys(seen) if region not in ordered]


def save_thesis_figure(
    fig,
    settings: ThesisFigureSettings,
    stem: str,
) -> dict[str, Path]:
    """Write one figure as editable PDF and high-resolution PNG."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for extension in settings.extensions:
        ext = str(extension).strip().lower().lstrip(".")
        written[ext] = save_figure(
            fig,
            settings.output_dir / f"{stem}.{ext}",
            ext=ext,
            dpi=settings.png_dpi if ext == "png" else None,
            transparent=settings.transparent,
            pdf_compression=settings.pdf_compression if ext == "pdf" else None,
        )
    return written


def figure_to_png_bytes(fig, *, dpi: int = 200) -> bytes:
    """Rasterize a figure for inline notebook display without keeping it open."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def add_analysis_window_bars(
    ax,
    *,
    windows: Sequence[tuple[str, float, float]] = ANALYSIS_WINDOWS_MS,
    colors: Sequence[str] = WINDOW_BAR_COLORS,
    band_height_frac: float = 0.052,
    band_gap_frac: float = 0.016,
    bottom_pad_frac: float = 0.02,
    label: bool = False,
    label_fontsize: float = 5.8,
    linewidth: float = 3.0,
    time_scale: float = 1.0,
) -> float:
    """Draw the three 500 ms analysis windows as stacked bars below the trace.

    Replaces the dotted background rectangles used previously: those spanned the
    full panel height and were read as data. Bars sit in reserved space under the
    trace, so they annotate the time axis without overlaying anything.

    The y-limit is extended downward to make room and the new lower limit is
    returned. ``time_scale`` converts the window bounds (ms) into axis units --
    pass ``1e-3`` for axes drawn in seconds.
    """
    n_bands = len(windows)
    y_low, y_high = ax.get_ylim()
    span = float(y_high - y_low)
    if span <= 0.0:
        return float(y_low)

    stack_frac = bottom_pad_frac + n_bands * band_height_frac + (n_bands - 1) * band_gap_frac
    # Solve for the new lower limit such that the reserved strip is
    # ``stack_frac`` of the *final* axis height.
    new_low = y_low - span * stack_frac / max(1.0 - stack_frac, 1e-6)
    new_span = float(y_high - new_low)

    for index, (name, start_ms, stop_ms) in enumerate(windows):
        y = new_low + new_span * (bottom_pad_frac + index * (band_height_frac + band_gap_frac))
        color = colors[index % len(colors)]
        ax.plot(
            [start_ms * time_scale, stop_ms * time_scale],
            [y, y],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            clip_on=False,
            zorder=4,
        )
        if label:
            ax.text(
                stop_ms * time_scale,
                y,
                f" {name}",
                ha="left",
                va="center",
                fontsize=label_fontsize,
                color=color,
                clip_on=False,
            )
    ax.set_ylim(new_low, y_high)
    return float(new_low)


def add_significance_bracket(
    ax,
    x_left: float,
    x_right: float,
    y: float,
    text: str,
    *,
    tick_frac: float = 0.018,
    color: str = INK,
    linewidth: float = 0.8,
    fontsize: float = 7.0,
) -> None:
    """Draw a comparison bracket with a star/n.s. label above two categories."""
    y_low, y_high = ax.get_ylim()
    tick = float(y_high - y_low) * tick_frac
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y - tick, y, y, y - tick],
        color=color,
        linewidth=linewidth,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        0.5 * (x_left + x_right),
        y + tick * 0.35,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
        clip_on=False,
    )


def condition_legend_handles(
    conditions: Sequence[str] = CONDITION_ORDER,
    *,
    short: bool = False,
    linewidth: float = 1.6,
) -> list:
    """Line handles for a shared fixation-category legend."""
    from matplotlib.lines import Line2D

    labels = CONDITION_SHORT_LABELS if short else CONDITION_LABELS
    return [
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS[condition],
            linewidth=linewidth,
            label=labels[condition],
        )
        for condition in conditions
    ]


def window_legend_handles(
    windows: Sequence[tuple[str, float, float]] = ANALYSIS_WINDOWS_MS,
    colors: Sequence[str] = WINDOW_BAR_COLORS,
) -> list:
    """Line handles describing the three analysis-window bars."""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=colors[index % len(colors)],
            linewidth=3.0,
            solid_capstyle="butt",
            label=f"{name} ({start_ms:+.0f} to {stop_ms:+.0f} ms)".replace("+0", "0"),
        )
        for index, (name, start_ms, stop_ms) in enumerate(windows)
    ]


def nice_axis(ax, *, y_ticks: int = 4) -> None:
    """Trim an axis to a small number of ticks and recessive spines."""
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=y_ticks, prune=None))
    ax.tick_params(length=2.5, pad=1.5)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
