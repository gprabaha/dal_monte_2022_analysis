"""Plot region-wise fixation peakiness distributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.plot_output import normalize_extension, save_figure
from dal_monte_2022_analysis.utils.filenames import ensure_filename


DEFAULT_REGION_ORDER: tuple[str, ...] = ("bla", "accg", "dmpfc", "ofc")
DEFAULT_REGION_LABELS: dict[str, str] = {
    "bla": "BLA",
    "accg": "ACCg",
    "dmpfc": "dmPFC",
    "ofc": "OFC",
}
DEFAULT_HIGHLIGHT_STYLE_ORDER: tuple[str, ...] = ("phasic", "tonic")
DEFAULT_HIGHLIGHT_STYLE_LABELS: dict[str, str] = {
    "phasic": "Peaky",
    "tonic": "Tonic",
}
DEFAULT_HIGHLIGHT_STYLE_COLORS: dict[str, str] = {
    "phasic": "#c03a2b",
    "tonic": "#2878b5",
}
DEFAULT_HIGHLIGHT_STYLE_MARKERS: dict[str, str] = {
    "phasic": "o",
    "tonic": "s",
}
_UNIT_UUID_PREFIX = "unit_uuid__"


@dataclass
class FixationPeakinessPlotSettings:
    """Configuration for fixation peakiness distribution plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_peakiness"
    unit_peakiness_filename: str = "unit_peakiness.csv"
    output_subdir: str = "ephys/psth/fixation_peakiness/plots"
    output_filename: str = "fixation_peakiness_by_region"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    region_order: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_REGION_ORDER))
    region_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_REGION_LABELS))
    figure_width_in: Optional[float] = 7.8
    figure_height_in: Optional[float] = 4.0
    left_margin: float = 0.08
    right_margin: float = 0.995
    top_margin: float = 0.92
    bottom_margin: float = 0.20
    violin_width: float = 0.86
    show_points: bool = True
    point_color: str = "#7a7a7a"
    point_alpha: float = 0.28
    point_size: float = 10.0
    violin_facecolor: str = "#d2d7df"
    violin_edgecolor: str = "#3b3f47"
    violin_alpha: float = 0.92
    show_highlight_units: bool = True
    highlight_units: Optional[dict] = None
    highlight_style_order: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_HIGHLIGHT_STYLE_ORDER))
    highlight_style_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HIGHLIGHT_STYLE_LABELS))
    highlight_style_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HIGHLIGHT_STYLE_COLORS))
    highlight_style_markers: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HIGHLIGHT_STYLE_MARKERS))
    highlight_marker_size: float = 42.0
    highlight_annotation_fontsize: float = 7.5
    jitter_seed: int = 0
    y_label: str = "Peakiness Score"
    show_suptitle: bool = False


def _normalize_region_token(value: object) -> str:
    return str(value).strip().lower()


def _as_str_dict(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        key_token = str(key).strip()
        value_token = str(value).strip()
        if key_token and value_token:
            out[key_token] = value_token
    return out


def _dedupe(values: Sequence[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        out.append(token)
        seen.add(token)
    return out


def _resolve_figsize_and_dpi(
    settings: FixationPeakinessPlotSettings,
) -> tuple[list[float], Optional[int]]:
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        figsize, cfg_dpi = resolve_figsize(plot_cfg)
    else:
        figsize, cfg_dpi = (None, None)

    if settings.figure_width_in is not None or settings.figure_height_in is not None:
        if figsize is None:
            figsize = [7.8, 4.0]
        width = float(settings.figure_width_in) if settings.figure_width_in is not None else float(figsize[0])
        height = float(settings.figure_height_in) if settings.figure_height_in is not None else float(figsize[1])
        figsize = [width, height]
    elif figsize is None:
        figsize = [7.8, 4.0]
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    return [float(figsize[0]), float(figsize[1])], dpi


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        raise ValueError(f"Unable to read CSV '{path}': {exc}") from exc


def _load_unit_peakiness_df(settings: FixationPeakinessPlotSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.unit_peakiness_filename, ".csv")
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Fixation peakiness unit CSV not found: {in_path}")
    df = _read_csv_or_empty(in_path)
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=["unit_uuid", "region", "peakiness_score", "region_norm"])
    required = {"unit_uuid", "region", "peakiness_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Fixation peakiness unit CSV is missing required columns: "
            + ", ".join(missing)
        )
    out = df.copy()
    out["unit_uuid"] = out["unit_uuid"].astype(str).map(str.strip)
    out["region"] = out["region"].astype(str).map(str.strip)
    out["region_norm"] = out["region"].map(_normalize_region_token)
    out["peakiness_score"] = pd.to_numeric(out["peakiness_score"], errors="coerce")
    return out


def _resolve_regions(
    df: pd.DataFrame,
    settings: FixationPeakinessPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> list[str]:
    if regions is not None:
        requested = _dedupe(regions)
    else:
        requested = _dedupe(settings.region_order)
    available = {
        _normalize_region_token(region): str(region).strip()
        for region in df.get("region", pd.Series(dtype=str)).astype(str).tolist()
        if str(region).strip()
    }
    resolved: list[str] = []
    seen: set[str] = set()
    for region in requested:
        token = _normalize_region_token(region)
        if token in seen:
            continue
        if token in available or regions is None:
            resolved.append(token)
            seen.add(token)
    for token in available:
        if token not in seen:
            resolved.append(token)
            seen.add(token)
    return resolved


def _region_display_label(token: str, settings: FixationPeakinessPlotSettings) -> str:
    if token in settings.region_labels:
        return str(settings.region_labels[token])
    for key, value in settings.region_labels.items():
        if _normalize_region_token(key) == token:
            return str(value)
    return str(token)


def _unit_uuid_lookup_variants(value: object) -> set[str]:
    token = str(value).strip()
    if not token:
        return set()
    out = {token}
    if token.startswith(_UNIT_UUID_PREFIX):
        suffix = token[len(_UNIT_UUID_PREFIX) :].strip()
        if suffix:
            out.add(suffix)
    else:
        out.add(f"{_UNIT_UUID_PREFIX}{token}")
    return out


def _parse_highlight_specs(
    raw: object,
) -> list[dict[str, str]]:
    if not isinstance(raw, dict):
        return []
    rows: list[dict[str, str]] = []
    for region, style_map in raw.items():
        if not isinstance(style_map, dict):
            continue
        for style, spec in style_map.items():
            unit_uuid = None
            if isinstance(spec, dict):
                value = spec.get("unit_uuid")
                if value is not None:
                    token = str(value).strip()
                    if token:
                        unit_uuid = token
            elif spec is not None:
                token = str(spec).strip()
                if token:
                    unit_uuid = token
            if unit_uuid is None:
                continue
            rows.append(
                {
                    "region_token": _normalize_region_token(region),
                    "region_label": str(region).strip(),
                    "style": str(style).strip(),
                    "configured_unit_uuid": unit_uuid,
                }
            )
    return rows


def _resolve_highlight_offsets(style_order: Sequence[str]) -> dict[str, float]:
    tokens = [str(style).strip() for style in style_order if str(style).strip()]
    if not tokens:
        return {}
    if len(tokens) == 1:
        return {tokens[0]: 0.0}
    span = 0.28
    positions = np.linspace(-span / 2.0, span / 2.0, num=len(tokens))
    return {style: float(pos) for style, pos in zip(tokens, positions)}


def _build_highlight_records(
    df: pd.DataFrame,
    settings: FixationPeakinessPlotSettings,
    *,
    region_order: Sequence[str],
) -> list[dict[str, object]]:
    specs = _parse_highlight_specs(settings.highlight_units)
    if not specs:
        return []
    records: list[dict[str, object]] = []
    for spec in specs:
        region_token = str(spec["region_token"])
        configured_unit_uuid = str(spec["configured_unit_uuid"])
        region_df = df.loc[df["region_norm"].astype(str) == region_token].copy()
        lookup = _unit_uuid_lookup_variants(configured_unit_uuid)
        matched = region_df.loc[region_df["unit_uuid"].astype(str).isin(lookup)].copy()
        row = {
            "region_token": region_token,
            "region_label": _region_display_label(region_token, settings),
            "style": str(spec["style"]),
            "style_label": str(settings.highlight_style_labels.get(str(spec["style"]), str(spec["style"]))),
            "configured_unit_uuid": configured_unit_uuid,
            "annotation_label": None,
            "matched": False,
            "matched_unit_uuid": None,
            "peakiness_score": np.nan,
            "best_condition": None,
        }
        row["annotation_label"] = f"{row['style_label']} {configured_unit_uuid}"
        if not matched.empty:
            picked = matched.iloc[0]
            row["matched"] = True
            row["matched_unit_uuid"] = str(picked["unit_uuid"])
            row["peakiness_score"] = float(picked["peakiness_score"])
            if "best_condition" in matched.columns:
                row["best_condition"] = str(picked["best_condition"])
        records.append(row)
    region_index = {token: idx for idx, token in enumerate(region_order)}
    style_offsets = _resolve_highlight_offsets(settings.highlight_style_order)
    for row in records:
        token = str(row["region_token"])
        row["x_index"] = int(region_index.get(token, -1))
        row["x_offset"] = float(style_offsets.get(str(row["style"]), 0.0))
    return records


def plot_fixation_peakiness_by_region(
    settings: FixationPeakinessPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> Optional[dict[str, object]]:
    """Plot one violin distribution of unit peakiness score by region."""
    df = _load_unit_peakiness_df(settings)
    if df.empty:
        print("[plot] no fixation peakiness unit rows found")
        return None

    region_order = _resolve_regions(df, settings, regions=regions)
    if not region_order:
        print("[plot] no fixation peakiness regions available to render")
        return None

    df = df.loc[df["region_norm"].astype(str).isin(set(region_order))].copy()
    df = df.loc[np.isfinite(pd.to_numeric(df["peakiness_score"], errors="coerce"))].copy()
    if df.empty:
        print("[plot] no finite fixation peakiness rows remain after filtering")
        return None

    figsize, dpi = _resolve_figsize_and_dpi(settings)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, squeeze=True)
    rng = np.random.default_rng(int(settings.jitter_seed))

    x_positions = np.arange(len(region_order), dtype=float)
    violin_data: list[np.ndarray] = []
    for token in region_order:
        values = pd.to_numeric(
            df.loc[df["region_norm"].astype(str) == token, "peakiness_score"],
            errors="coerce",
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        violin_data.append(values)

    valid_positions = [x for x, values in zip(x_positions, violin_data) if values.size > 0]
    valid_data = [values for values in violin_data if values.size > 0]
    if valid_data:
        parts = ax.violinplot(
            valid_data,
            positions=valid_positions,
            widths=float(settings.violin_width),
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(str(settings.violin_facecolor))
            body.set_edgecolor(str(settings.violin_edgecolor))
            body.set_alpha(float(settings.violin_alpha))
            body.set_linewidth(0.9)

    for xpos, token, values in zip(x_positions, region_order, violin_data):
        if values.size == 0:
            continue
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.vlines(xpos, q25, q75, color="#202020", linewidth=1.1, zorder=3)
        ax.hlines(median, xpos - 0.14, xpos + 0.14, color="#202020", linewidth=1.4, zorder=3)
        if settings.show_points:
            jitter = rng.uniform(-0.16, 0.16, size=values.size)
            ax.scatter(
                np.full(values.size, xpos, dtype=float) + jitter,
                values,
                s=float(settings.point_size),
                color=str(settings.point_color),
                alpha=float(settings.point_alpha),
                linewidths=0.0,
                zorder=2,
            )

    highlight_records = _build_highlight_records(df, settings, region_order=region_order) if settings.show_highlight_units else []
    legend_handles: list[Line2D] = []
    used_styles: set[str] = set()
    for row in highlight_records:
        if not bool(row["matched"]):
            continue
        x_index = int(row["x_index"])
        if x_index < 0:
            continue
        xpos = float(x_positions[x_index]) + float(row["x_offset"])
        ypos = float(row["peakiness_score"])
        style = str(row["style"])
        color = str(settings.highlight_style_colors.get(style, "#202020"))
        marker = str(settings.highlight_style_markers.get(style, "o"))
        ax.scatter(
            [xpos],
            [ypos],
            s=float(settings.highlight_marker_size),
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        ax.annotate(
            str(row["annotation_label"]),
            xy=(xpos, ypos),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=float(settings.highlight_annotation_fontsize),
            color=color,
            ha="left",
            va="bottom",
        )
        if style not in used_styles:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    markersize=max(5.0, np.sqrt(float(settings.highlight_marker_size))),
                    label=str(row["style_label"]),
                )
            )
            used_styles.add(style)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([_region_display_label(token, settings) for token in region_order], fontsize=9)
    ax.set_ylabel(str(settings.y_label), fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    if settings.show_suptitle:
        fig.suptitle("Fixation Peakiness by Region", fontsize=12)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)

    fig.subplots_adjust(
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=float(settings.top_margin),
        bottom=float(settings.bottom_margin),
    )

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    ext = normalize_extension(settings.output_extension, fallback="pdf")
    stem = Path(str(settings.output_filename).strip()).stem or "fixation_peakiness_by_region"
    out_path = out_root / f"{stem}.{ext}"
    save_figure(fig, out_path, ext=ext, dpi=dpi, facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return {
        "output_path": out_path,
        "regions": region_order,
        "n_units_total": int(len(df)),
        "highlighted_units": highlight_records,
    }
