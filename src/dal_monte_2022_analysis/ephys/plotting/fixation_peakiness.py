"""Plot region-wise fixation peakiness distributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
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
    figure_height_in: Optional[float] = 2.6
    left_margin: float = 0.08
    right_margin: float = 0.995
    top_margin: float = 0.84
    bottom_margin: float = 0.26
    panel_wspace: float = 0.12
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
        panel_index = int(region_index.get(token, -1))
        panel_offset = float(style_offsets.get(str(row["style"]), 0.0))
        row["panel_index"] = panel_index
        row["panel_offset"] = panel_offset
        row["x_index"] = panel_index
        row["x_offset"] = panel_offset
    return records


def _resolve_density_bandwidth(values: np.ndarray, x_span: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    n = int(finite.size)
    if n <= 1:
        return max(0.06 * max(x_span, 1.0), 0.05)
    std = float(np.std(finite, ddof=1))
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    iqr_sigma = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    sigma_candidates = [value for value in (std, iqr_sigma) if np.isfinite(value) and value > 0.0]
    sigma = min(sigma_candidates) if sigma_candidates else 0.0
    if sigma <= 0.0:
        sigma = max(float(np.ptp(finite)) / 1.349, 0.0)
    if sigma <= 0.0:
        sigma = max(0.08 * max(x_span, 1.0), 0.05)
    bandwidth = 0.9 * sigma * (n ** (-1.0 / 5.0))
    return max(float(bandwidth), 0.04 * max(x_span, 1.0), 0.03)


def _estimate_density_curve(
    values: np.ndarray,
    *,
    x_grid: np.ndarray,
) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros_like(x_grid, dtype=float)
    x_span = float(np.max(x_grid) - np.min(x_grid)) if x_grid.size > 1 else 1.0
    bandwidth = _resolve_density_bandwidth(finite, x_span=x_span)
    z = (x_grid[:, None] - finite[None, :]) / bandwidth
    kernel = np.exp(-0.5 * np.square(z)) / (bandwidth * np.sqrt(2.0 * np.pi))
    density = np.mean(kernel, axis=1)
    density = np.asarray(density, dtype=float)
    density[~np.isfinite(density)] = 0.0
    return density


def plot_fixation_peakiness_by_region(
    settings: FixationPeakinessPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> Optional[dict[str, object]]:
    """Plot one one-sided density distribution per region in a shared multi-panel layout."""
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
    fig, axes = plt.subplots(
        1,
        len(region_order),
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    axes_flat = axes.ravel()

    violin_data: list[np.ndarray] = []
    for token in region_order:
        values = pd.to_numeric(
            df.loc[df["region_norm"].astype(str) == token, "peakiness_score"],
            errors="coerce",
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        violin_data.append(values)

    valid_data = [values for values in violin_data if values.size > 0]
    highlight_records = _build_highlight_records(df, settings, region_order=region_order) if settings.show_highlight_units else []
    finite_scores = np.concatenate(valid_data) if valid_data else np.asarray([], dtype=float)
    if finite_scores.size > 0:
        x_min = float(np.min(finite_scores))
        x_max = float(np.max(finite_scores))
    else:
        x_min = 0.0
        x_max = 1.0
    x_min = min(0.0, x_min)
    x_span = max(x_max - x_min, x_max, 1.0)
    x_pad = 0.06 * x_span
    shared_xlim = (x_min - 0.25 * x_pad, x_max + x_pad)
    x_grid = np.linspace(shared_xlim[0], shared_xlim[1], num=400, dtype=float)

    highlight_by_panel: dict[int, list[dict[str, object]]] = {}
    for row in highlight_records:
        panel_index = int(row.get("panel_index", -1))
        if panel_index < 0:
            continue
        highlight_by_panel.setdefault(panel_index, []).append(row)
    density_curves: list[np.ndarray] = []
    max_density_per_panel: list[float] = []
    for values in violin_data:
        density = _estimate_density_curve(values, x_grid=x_grid)
        density_curves.append(density)
        max_density = float(np.max(density)) if density.size > 0 else 0.0
        max_density_per_panel.append(max_density)

    for idx, (ax, token, values, density, max_density) in enumerate(
        zip(axes_flat, region_order, violin_data, density_curves, max_density_per_panel)
    ):
        y_max = max(max_density, 1.0e-6)
        if values.size > 0:
            ax.fill_between(
                x_grid,
                0.0,
                density,
                color=str(settings.violin_facecolor),
                alpha=float(settings.violin_alpha),
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                x_grid,
                density,
                color=str(settings.violin_edgecolor),
                linewidth=1.1,
                zorder=2,
            )
            q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
            ax.hlines(0.0, q25, q75, color="#202020", linewidth=1.0, zorder=3)
            ax.vlines(median, 0.0, 0.18 * y_max, color="#202020", linewidth=1.2, zorder=3)
            if settings.show_points:
                rug_height = 0.08 * y_max
                ax.vlines(
                    values,
                    0.0,
                    rug_height,
                    color=str(settings.point_color),
                    alpha=float(settings.point_alpha),
                    linewidth=max(0.4, float(settings.point_size) / 16.0),
                    zorder=2,
                )

        panel_rows = [row for row in highlight_by_panel.get(idx, []) if bool(row["matched"])]
        panel_rows.sort(
            key=lambda row: settings.highlight_style_order.index(str(row["style"]))
            if str(row["style"]) in settings.highlight_style_order
            else len(settings.highlight_style_order)
        )
        label_y_top = 0.95 * y_max
        label_y_step = 0.17 * y_max
        for row_idx, row in enumerate(panel_rows):
            xpos = float(row["peakiness_score"])
            style = str(row["style"])
            color = str(settings.highlight_style_colors.get(style, "#202020"))
            ax.axvline(
                xpos,
                color=color,
                linewidth=1.4,
                alpha=0.95,
                zorder=4,
            )
            ax.text(
                xpos,
                max(0.10 * y_max, label_y_top - row_idx * label_y_step),
                str(row["annotation_label"]),
                fontsize=float(settings.highlight_annotation_fontsize),
                color=color,
                rotation=90,
                ha="right",
                va="top",
                zorder=5,
            )

        ax.set_title(_region_display_label(token, settings), fontsize=10, pad=6)
        ax.set_xlim(*shared_xlim)
        ax.set_ylim(0.0, 1.05 * y_max)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=7)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=9)
        else:
            ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_visible(False)

    fig.supxlabel(str(settings.y_label), fontsize=10, y=max(0.02, 0.45 * float(settings.bottom_margin)))
    if settings.show_suptitle:
        fig.suptitle("Fixation Peakiness by Region", fontsize=12)

    fig.subplots_adjust(
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=float(settings.top_margin),
        bottom=float(settings.bottom_margin),
        wspace=float(settings.panel_wspace),
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
