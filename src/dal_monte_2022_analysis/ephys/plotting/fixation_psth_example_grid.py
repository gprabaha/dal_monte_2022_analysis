"""Plot a region-by-preference grid of fixation PSTH example units."""

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

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import darken_color as _darken_color_shared
from dal_monte_2022_analysis.ephys.plotting.fixation_psth import (
    DEFAULT_CONDITION_COLORS,
    FixationPSTHUnitPlotSettings,
    _build_unit_condition_payloads as _build_unit_condition_payloads_shared,
    _ensure_ext as _ensure_ext_shared,
    _iter_trial_rows as _iter_trial_rows_shared,
    _load_trials_for_date as _load_trials_for_date_shared,
)
from dal_monte_2022_analysis.runtime.io.plot_output import save_figure
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_EXAMPLE_GRID_REGIONS = ("BLA", "ACCg", "dmPFC", "OFC")
DEFAULT_EXAMPLE_GRID_ROW_PREFERENCES = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
DEFAULT_EXAMPLE_GRID_ROW_LABELS = {
    "face_interactive": "Interactive Face Favoring",
    "face_non_interactive": "Non-Interactive Face Favoring",
    "object": "Object Favoring",
}

_PREFERENCE_ALIASES = {
    "face_interactive": "face_interactive",
    "interactive": "face_interactive",
    "interactive_face": "face_interactive",
    "interactiveface": "face_interactive",
    "faceinteractive": "face_interactive",
    "int_face": "face_interactive",
    "face_non_interactive": "face_non_interactive",
    "non_interactive": "face_non_interactive",
    "noninteractive": "face_non_interactive",
    "non_interactive_face": "face_non_interactive",
    "noninteractive_face": "face_non_interactive",
    "face_noninteractive": "face_non_interactive",
    "object": "object",
}


@dataclass(frozen=True)
class FixationPSTHExampleUnitSpec:
    """Manual selection for one example unit in one grid cell."""

    region: str
    preference: str
    unit_uuid: str
    date: Optional[str] = None


@dataclass
class FixationPSTHExampleGridPlotSettings:
    """Configuration for selective example-unit grid plotting."""

    unit_plot_settings: FixationPSTHUnitPlotSettings
    output_subdir: str = "ephys/psth/fixation_psth_selective_unit_plots"
    output_filename: str = "selective_example_grid_3x4"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    figure_width_in: float = 8.5
    figure_height_in: float = 6.6
    column_regions: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_EXAMPLE_GRID_REGIONS),
    )
    row_preferences: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_EXAMPLE_GRID_ROW_PREFERENCES),
    )
    row_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EXAMPLE_GRID_ROW_LABELS))
    left_margin: float = 0.07
    right_margin: float = 0.995
    top_margin: float = 0.91
    bottom_margin: float = 0.095
    panel_wspace: float = 0.16
    panel_hspace: float = 0.24
    inner_hspace: float = 0.06
    show_global_legend: bool = True
    legend_ncol: int = 3
    pdf_compression: Optional[int] = 0


@dataclass(frozen=True)
class _ResolvedCell:
    spec: FixationPSTHExampleUnitSpec
    date: str
    bin_centers: np.ndarray
    payloads: list[dict]


def _normalize_date_str(val) -> str:
    if val is None:
        return ""
    token = str(val).strip()
    if not token:
        return ""
    if token.endswith(".0"):
        token = token[:-2]
    if token.isdigit():
        return token.zfill(8)
    try:
        intval = int(float(token))
        return str(intval).zfill(8)
    except Exception:
        return token


def _normalize_region_key(region: object) -> str:
    return str(region).strip().lower()


def _normalize_unit_uuid_key(unit_uuid: object) -> str:
    token = str(unit_uuid).strip()
    if token.startswith("unit_uuid__"):
        token = token[len("unit_uuid__"):]
    return token


def normalize_example_preference(preference: object) -> str:
    """Normalize a user preference token to one of three canonical keys."""
    token = str(preference).strip().lower().replace("-", "_").replace(" ", "_")
    resolved = _PREFERENCE_ALIASES.get(token)
    if resolved is None:
        supported = ", ".join(DEFAULT_EXAMPLE_GRID_ROW_PREFERENCES)
        raise ValueError(
            f"Unsupported preference '{preference}'. Expected one of: {supported}.",
        )
    return resolved


def _canonical_regions(regions: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for region in regions:
        label = str(region).strip()
        if not label:
            continue
        key = _normalize_region_key(label)
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def _canonical_preferences(preferences: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pref in preferences:
        canon = normalize_example_preference(pref)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
    return out


def _parse_unit_entry(
    *,
    region: str,
    preference: str,
    entry: object,
) -> Optional[FixationPSTHExampleUnitSpec]:
    if entry is None:
        return None

    if isinstance(entry, str):
        unit_uuid = entry.strip()
        date = None
    elif isinstance(entry, dict):
        unit_uuid = str(entry.get("unit_uuid", "")).strip()
        date_raw = entry.get("date")
        date = _normalize_date_str(date_raw) if date_raw not in (None, "") else None
    else:
        raise ValueError(
            "Each example unit entry must be either a UUID string or "
            "a dict with keys {unit_uuid, date}.",
        )

    if not unit_uuid:
        return None

    return FixationPSTHExampleUnitSpec(
        region=str(region).strip(),
        preference=normalize_example_preference(preference),
        unit_uuid=unit_uuid,
        date=date,
    )


def parse_example_grid_unit_specs(
    cfg: dict,
    *,
    regions: Sequence[str],
    row_preferences: Sequence[str],
    cfg_key: str = "selective_example_grid_units",
) -> list[FixationPSTHExampleUnitSpec]:
    """Parse manual example-unit selections from config."""
    raw = cfg.get(cfg_key, {})
    regions_canon = _canonical_regions(regions)
    prefs_canon = _canonical_preferences(row_preferences)

    if not regions_canon or not prefs_canon:
        return []

    region_lookup = {_normalize_region_key(region): region for region in regions_canon}
    pref_lookup = set(prefs_canon)
    specs: list[FixationPSTHExampleUnitSpec] = []

    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"{cfg_key}[{idx}] must be a dict entry.")
            region_raw = item.get("region")
            preference_raw = item.get("preference")
            if region_raw is None or preference_raw is None:
                raise ValueError(
                    f"{cfg_key}[{idx}] requires keys 'region' and 'preference'.",
                )
            region_key = _normalize_region_key(region_raw)
            if region_key not in region_lookup:
                continue
            preference = normalize_example_preference(preference_raw)
            if preference not in pref_lookup:
                continue
            parsed = _parse_unit_entry(
                region=region_lookup[region_key],
                preference=preference,
                entry=item,
            )
            if parsed is not None:
                specs.append(parsed)
        return specs

    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_key} must be either a dict or a list of dict entries.")

    nested_region_lookup: dict[str, dict] = {}
    for region_key, region_block in raw.items():
        if isinstance(region_block, dict):
            nested_region_lookup[_normalize_region_key(region_key)] = region_block

    for region in regions_canon:
        block = nested_region_lookup.get(_normalize_region_key(region), {})
        pref_entries: dict[str, object] = {}
        if isinstance(block, dict):
            for pref_key, entry in block.items():
                try:
                    pref_entries[normalize_example_preference(pref_key)] = entry
                except ValueError:
                    continue
        for preference in prefs_canon:
            parsed = _parse_unit_entry(
                region=region,
                preference=preference,
                entry=pref_entries.get(preference),
            )
            if parsed is not None:
                specs.append(parsed)

    return specs


def _spec_map(
    unit_specs: Sequence[FixationPSTHExampleUnitSpec],
) -> dict[tuple[str, str], FixationPSTHExampleUnitSpec]:
    out: dict[tuple[str, str], FixationPSTHExampleUnitSpec] = {}
    for spec in unit_specs:
        canon_pref = normalize_example_preference(spec.preference)
        region = str(spec.region).strip()
        unit_uuid = str(spec.unit_uuid).strip()
        if not region or not unit_uuid:
            continue
        date = _normalize_date_str(spec.date) if spec.date not in (None, "") else None
        key = (canon_pref, _normalize_region_key(region))
        normalized = FixationPSTHExampleUnitSpec(
            region=region,
            preference=canon_pref,
            unit_uuid=unit_uuid,
            date=date,
        )
        if key in out:
            prev = out[key]
            raise ValueError(
                "Duplicate example-unit config for "
                f"preference={canon_pref}, region={region}: "
                f"{prev.unit_uuid} and {unit_uuid}.",
            )
        out[key] = normalized
    return out


def _load_trials_by_date(
    unit_settings: FixationPSTHUnitPlotSettings,
    *,
    sessions: Optional[Sequence[str]] = None,
    date_filter: Optional[Sequence[str]] = None,
) -> dict[str, dict[str, object]]:
    cfg = load_config(unit_settings.cfg_path)
    rows = _iter_trial_rows_shared(
        cfg,
        unit_settings,
        dates=date_filter,
        sessions=sessions,
    )
    if not rows:
        return {}

    grouped: dict[str, list[Path]] = {}
    for row in rows:
        date = _normalize_date_str(row.get("date"))
        if not date:
            date = str(row.get("date", "")).strip()
        if not date:
            date = "unknown"
        grouped.setdefault(date, []).append(Path(row["path"]))

    out: dict[str, dict[str, object]] = {}
    for date, paths in sorted(grouped.items(), key=lambda item: item[0]):
        df, bin_centers = _load_trials_for_date_shared(
            paths,
            date=date,
            settings=unit_settings,
        )
        if df.empty or "unit_uuid" not in df.columns:
            continue
        local = df.copy()
        local["unit_uuid"] = local["unit_uuid"].astype(str).map(lambda val: val.strip())
        if "region" in local.columns:
            local["region_norm"] = local["region"].fillna("").astype(str).map(_normalize_region_key)
        else:
            local["region_norm"] = ""
        out[date] = {
            "df": local,
            "bin_centers": np.asarray(bin_centers, dtype=float),
        }
    return out


def _resolve_cell(
    spec: FixationPSTHExampleUnitSpec,
    by_date: dict[str, dict[str, object]],
    unit_settings: FixationPSTHUnitPlotSettings,
) -> tuple[Optional[_ResolvedCell], Optional[str]]:
    region_key = _normalize_region_key(spec.region)
    unit_uuid = str(spec.unit_uuid).strip()
    unit_uuid_key = _normalize_unit_uuid_key(unit_uuid)
    date_hint = _normalize_date_str(spec.date) if spec.date not in (None, "") else None

    candidate_dates = [date_hint] if date_hint else sorted(by_date.keys())
    matches: list[tuple[str, pd.DataFrame, np.ndarray]] = []

    for date in candidate_dates:
        blob = by_date.get(date)
        if blob is None:
            continue
        df = blob["df"]
        uuid_series = df["unit_uuid"].astype(str)
        mask = (
            (uuid_series == unit_uuid)
            | (uuid_series.map(_normalize_unit_uuid_key) == unit_uuid_key)
        )
        if region_key:
            mask = mask & (df["region_norm"].astype(str) == region_key)
        df_unit = df.loc[mask].copy()
        if df_unit.empty:
            continue
        matches.append((date, df_unit, np.asarray(blob["bin_centers"], dtype=float)))

    if not matches:
        if date_hint:
            return (
                None,
                f"{spec.preference} | {spec.region}: unit_uuid={unit_uuid}, date={date_hint} not found.",
            )
        return (
            None,
            f"{spec.preference} | {spec.region}: unit_uuid={unit_uuid} not found (or region mismatch).",
        )

    if len(matches) > 1:
        dates = ", ".join(sorted(match[0] for match in matches))
        return (
            None,
            f"{spec.preference} | {spec.region}: unit_uuid={unit_uuid} matched multiple dates [{dates}]. Set date explicitly.",
        )

    date, df_unit, bin_centers = matches[0]
    if bin_centers.size < 2:
        return (
            None,
            f"{spec.preference} | {spec.region}: invalid bin centers for unit_uuid={unit_uuid}.",
        )
    bin_size_s = float(np.mean(np.diff(bin_centers)))
    if not np.isfinite(bin_size_s) or bin_size_s <= 0:
        return (
            None,
            f"{spec.preference} | {spec.region}: unable to infer positive bin size for unit_uuid={unit_uuid}.",
        )

    payloads = _build_unit_condition_payloads_shared(
        df_unit,
        unit_key=f"{date}|{unit_uuid}",
        bin_centers=bin_centers,
        bin_size_s=bin_size_s,
        settings=unit_settings,
    )
    if not any(int(payload["n_trials"]) > 0 for payload in payloads):
        return (
            None,
            f"{spec.preference} | {spec.region}: no usable trials for unit_uuid={unit_uuid} (date={date}).",
        )

    return (
        _ResolvedCell(
            spec=spec,
            date=date,
            bin_centers=bin_centers,
            payloads=payloads,
        ),
        None,
    )


def _draw_missing_cell(ax_raster, ax_rate, message: str, *, show_x_axis: bool) -> None:
    for ax in (ax_raster, ax_rate):
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.25)
    ax_rate.text(
        0.5,
        0.5,
        str(message),
        transform=ax_rate.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color="#666666",
    )
    if show_x_axis:
        ax_rate.set_xlabel("Time From Fixation Start (s)", fontsize=7)


def _draw_resolved_cell(
    ax_raster,
    ax_rate,
    resolved: _ResolvedCell,
    *,
    unit_settings: FixationPSTHUnitPlotSettings,
    show_raster_y_axis: bool,
    show_rate_ylabel: bool,
    show_x_axis: bool,
) -> None:
    y_cursor = 1
    y_ticks: list[float] = []
    y_labels: list[str] = []
    for payload in resolved.payloads:
        n_trials = int(payload["n_trials"])
        if n_trials <= 0:
            continue
        line_offsets = np.arange(y_cursor, y_cursor + n_trials, dtype=float)
        ax_raster.eventplot(
            payload["spike_rows"],
            lineoffsets=line_offsets,
            linelengths=float(unit_settings.raster_linelength),
            linewidths=float(unit_settings.raster_linewidth),
            colors=[_darken_color_shared(payload["color"], unit_settings.raster_darkening_factor)] * n_trials,
            alpha=float(unit_settings.raster_alpha),
            zorder=3,
        )
        if unit_settings.raster_show_condition_background:
            ax_raster.axhspan(
                float(line_offsets[0]) - 0.5,
                float(line_offsets[-1]) + 0.5,
                color=payload["color"],
                alpha=0.07,
                zorder=0,
            )
        y_ticks.append(float(0.5 * (line_offsets[0] + line_offsets[-1])))
        y_labels.append(f"{payload['label']} (n={n_trials})")
        y_cursor += n_trials
        ax_raster.axhline(float(y_cursor) - 0.5, color="#cccccc", linewidth=0.55)

    ax_raster.axvline(0.0, color="#333333", linestyle="--", linewidth=0.7)
    ax_raster.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_raster.tick_params(axis="y", labelsize=5.4, pad=1.0, length=1.6)
    if y_ticks:
        if show_raster_y_axis:
            ax_raster.set_yticks(y_ticks)
            ax_raster.set_yticklabels(y_labels, fontsize=5.4)
            ax_raster.set_ylabel("Trials", fontsize=7)
        else:
            ax_raster.set_yticks([])
        ax_raster.set_ylim(float(y_cursor) - 0.5, 0.5)
    else:
        ax_raster.set_yticks([])

    for payload in resolved.payloads:
        if int(payload["n_trials"]) <= 0:
            continue
        mean_hz = np.asarray(payload["mean_hz"], dtype=float)
        sem_hz = np.asarray(payload["sem_hz"], dtype=float)
        ax_rate.plot(
            resolved.bin_centers,
            mean_hz,
            color=payload["color"],
            linewidth=0.95,
        )
        ax_rate.fill_between(
            resolved.bin_centers,
            mean_hz - sem_hz,
            mean_hz + sem_hz,
            color=payload["color"],
            alpha=0.20,
            linewidth=0.0,
        )

    ax_rate.axvline(0.0, color="#333333", linestyle="--", linewidth=0.7)
    ax_rate.set_xlim(float(resolved.bin_centers[0]), float(resolved.bin_centers[-1]))
    ax_rate.grid(True, alpha=0.16, linewidth=0.45)
    ax_rate.tick_params(axis="both", labelsize=6.0, pad=1.0, length=1.8)
    if show_rate_ylabel:
        ax_rate.set_ylabel("Hz", fontsize=7)
    else:
        ax_rate.set_ylabel("")
    if show_x_axis:
        ax_rate.set_xlabel("Time From Fixation Start (s)", fontsize=7)
    else:
        ax_rate.set_xlabel("")
        ax_rate.tick_params(axis="x", labelbottom=False)

    unit_token = str(resolved.spec.unit_uuid).strip()
    if len(unit_token) > 14:
        unit_token = f"{unit_token[:14]}..."
    ax_raster.text(
        0.01,
        1.02,
        f"{resolved.date} | {unit_token}",
        transform=ax_raster.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color="#1f1f1f",
    )


def _condition_handles(condition_colors: dict[str, str]) -> list[Line2D]:
    order = [
        ("face_interactive", "Interactive Face"),
        ("face_non_interactive", "Non-Interactive Face"),
        ("object", "Object"),
    ]
    handles: list[Line2D] = []
    for key, label in order:
        color = condition_colors.get(key, DEFAULT_CONDITION_COLORS.get(key, "#444444"))
        handles.append(Line2D([0], [0], color=color, linewidth=1.2, label=label))
    return handles


def plot_fixation_psth_example_grid(
    settings: FixationPSTHExampleGridPlotSettings,
    *,
    unit_specs: Sequence[FixationPSTHExampleUnitSpec],
    sessions: Optional[Sequence[str]] = None,
    allow_missing: bool = False,
) -> dict[str, object]:
    """Render one 3x4 selective example-unit PSTH grid figure."""
    unit_settings = settings.unit_plot_settings
    row_preferences = _canonical_preferences(settings.row_preferences)
    column_regions = _canonical_regions(settings.column_regions)
    if not row_preferences:
        raise ValueError("At least one row preference must be provided.")
    if not column_regions:
        raise ValueError("At least one column region must be provided.")

    by_cell = _spec_map(unit_specs)
    expected_cells = [
        (pref, _normalize_region_key(region), region)
        for pref in row_preferences
        for region in column_regions
    ]
    missing_specs = [
        f"{pref} | {region}"
        for pref, region_key, region in expected_cells
        if (pref, region_key) not in by_cell
    ]
    if missing_specs and not allow_missing:
        raise ValueError(
            "Missing example-unit config entries for:\n- "
            + "\n- ".join(missing_specs),
        )

    date_hints = sorted(
        {
            _normalize_date_str(spec.date)
            for spec in by_cell.values()
            if spec.date not in (None, "")
        }
    )
    all_cells_have_dates = bool(by_cell) and all(spec.date not in (None, "") for spec in by_cell.values())
    date_filter = date_hints if all_cells_have_dates else None

    by_date = _load_trials_by_date(
        unit_settings,
        sessions=sessions,
        date_filter=date_filter,
    )
    if not by_date:
        raise ValueError("No fixation PSTH trial files found for selective example-grid plotting.")

    resolved_cells: dict[tuple[str, str], _ResolvedCell] = {}
    unresolved_specs: list[str] = []
    for pref, region_key, _region in expected_cells:
        spec = by_cell.get((pref, region_key))
        if spec is None:
            continue
        resolved, err = _resolve_cell(spec, by_date, unit_settings)
        if resolved is not None:
            resolved_cells[(pref, region_key)] = resolved
        elif err is not None:
            unresolved_specs.append(err)

    if unresolved_specs and not allow_missing:
        raise ValueError(
            "Failed to resolve configured example units:\n- "
            + "\n- ".join(unresolved_specs),
        )

    dpi = settings.output_dpi if settings.output_dpi is not None else unit_settings.output_dpi
    fig = plt.figure(
        figsize=[float(settings.figure_width_in), float(settings.figure_height_in)],
        dpi=dpi,
    )
    fig.patch.set_facecolor("white")

    outer = fig.add_gridspec(
        nrows=len(row_preferences),
        ncols=len(column_regions),
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=float(settings.top_margin),
        bottom=float(settings.bottom_margin),
        wspace=float(settings.panel_wspace),
        hspace=float(settings.panel_hspace),
    )

    row_anchors: dict[int, tuple] = {}
    for row_i, pref in enumerate(row_preferences):
        for col_i, region in enumerate(column_regions):
            inner = outer[row_i, col_i].subgridspec(
                2,
                1,
                height_ratios=[
                    float(unit_settings.panel_raster_height_ratio),
                    float(unit_settings.panel_rate_height_ratio),
                ],
                hspace=float(settings.inner_hspace),
            )
            ax_raster = fig.add_subplot(inner[0, 0])
            ax_rate = fig.add_subplot(inner[1, 0], sharex=ax_raster)
            ax_raster.set_facecolor("white")
            ax_rate.set_facecolor("white")

            if col_i == 0:
                row_anchors[row_i] = (ax_raster, ax_rate)
            if row_i == 0:
                ax_raster.set_title(str(region), fontsize=9.5, pad=8.0)

            key = (pref, _normalize_region_key(region))
            show_raster_y = col_i == 0
            show_rate_ylabel = col_i == 0
            show_x = row_i == (len(row_preferences) - 1)
            resolved = resolved_cells.get(key)
            if resolved is None:
                placeholder = "Not Set"
                if key in by_cell:
                    placeholder = "Not Found"
                _draw_missing_cell(
                    ax_raster,
                    ax_rate,
                    placeholder,
                    show_x_axis=show_x,
                )
            else:
                _draw_resolved_cell(
                    ax_raster,
                    ax_rate,
                    resolved,
                    unit_settings=unit_settings,
                    show_raster_y_axis=show_raster_y,
                    show_rate_ylabel=show_rate_ylabel,
                    show_x_axis=show_x,
                )

    for row_i, pref in enumerate(row_preferences):
        label = settings.row_labels.get(
            pref,
            DEFAULT_EXAMPLE_GRID_ROW_LABELS.get(pref, pref.replace("_", " ").title()),
        )
        anchor = row_anchors.get(row_i)
        if anchor is None:
            continue
        ax_raster, ax_rate = anchor
        y_mid = 0.5 * (ax_rate.get_position().y0 + ax_raster.get_position().y1)
        fig.text(
            max(0.002, float(settings.left_margin) * 0.14),
            y_mid,
            str(label),
            rotation=90,
            ha="left",
            va="center",
            fontsize=8.2,
            color="#111111",
        )

    if settings.show_global_legend:
        fig.legend(
            handles=_condition_handles(unit_settings.condition_colors),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=max(1, int(settings.legend_ncol)),
            frameon=False,
            fontsize=7.4,
            handlelength=2.3,
            columnspacing=1.4,
        )

    cfg = load_config(unit_settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    ext = _ensure_ext_shared(settings.output_extension)
    output_filename = str(settings.output_filename).strip() or "selective_example_grid_3x4"
    out_path = out_root / f"{output_filename}.{ext}"
    save_figure(
        fig,
        out_path,
        ext=ext,
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        pdf_compression=(settings.pdf_compression if ext == "pdf" else None),
    )
    plt.close(fig)

    return {
        "output_path": out_path,
        "expected_cells": len(expected_cells),
        "resolved_cells": len(resolved_cells),
        "missing_specs": missing_specs,
        "unresolved_specs": unresolved_specs,
        "row_preferences": row_preferences,
        "column_regions": column_regions,
    }
