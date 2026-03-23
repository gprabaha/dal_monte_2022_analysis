"""Plot region-wise fixation PSTH variability violins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

try:
    import seaborn as sns
except Exception:  # pragma: no cover - handled explicitly at runtime
    sns = None

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.common import apply_plotting_config, resolve_figsize
from dal_monte_2022_analysis.runtime.io.plot_output import normalize_extension, save_figure
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_REGION_ORDER: tuple[str, ...] = ("BLA", "ACCg", "dmPFC", "OFC")
DEFAULT_CONDITION_ORDER: tuple[str, ...] = (
    "face_interactive",
    "face_non_interactive",
    "object",
)
DEFAULT_CONDITION_LABELS: dict[str, str] = {
    "face_interactive": "Int Face",
    "face_non_interactive": "Non-Int Face",
    "object": "Object",
}
DEFAULT_CONDITION_COLORS: dict[str, str] = {
    "face_interactive": "#b64198",
    "face_non_interactive": "#97ca3d",
    "object": "#754c29",
}


@dataclass
class FixationPSTHVariabilityPlotSettings:
    """Configuration for fixation PSTH variability violin plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_psth_variability"
    unit_summary_filename: str = "unit_condition_variability.csv"
    within_region_stats_filename: str = "within_region_condition_variability_stats.csv"
    output_subdir: str = "ephys/psth/fixation_psth_variability/plots"
    output_filename: str = "fixation_psth_variability_violin"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    region_order: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_REGION_ORDER))
    condition_order: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_CONDITION_ORDER))
    condition_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONDITION_LABELS))
    condition_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONDITION_COLORS))
    y_label: str = "SD of Mean FR (Hz)"
    figure_width_in: float = 8.6
    figure_height_in: float = 3.2
    left_margin: float = 0.06
    right_margin: float = 0.995
    top_margin: float = 0.83
    bottom_margin: float = 0.22
    wspace: float = 0.28
    min_units_per_region: int = 1


def _normalize_region_token(value: object) -> str:
    return str(value).strip().lower()


def _variability_column(condition: str) -> str:
    return f"{str(condition).strip()}_variability"


def _empty_plot_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["region", "unit_key", "condition", "fr_variability", "region_norm"])


def _empty_within_region_stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "region",
            "condition_a",
            "condition_b",
            "p_value_adjusted",
            "significant_adjusted",
            "region_norm",
        ]
    )


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        raise ValueError(f"Unable to read CSV '{path}': {exc}") from exc


def _resolve_regions(
    settings: FixationPSTHVariabilityPlotSettings,
    summary_df: pd.DataFrame,
    *,
    regions: Optional[Sequence[str]] = None,
) -> list[str]:
    if regions is not None:
        requested = [str(region).strip() for region in regions if str(region).strip()]
    else:
        requested = [str(region).strip() for region in settings.region_order if str(region).strip()]
    available = {
        _normalize_region_token(region): str(region).strip()
        for region in summary_df.get("region", pd.Series(dtype=str)).astype(str).tolist()
        if str(region).strip()
    }
    resolved: list[str] = []
    seen: set[str] = set()
    for region in requested:
        norm = _normalize_region_token(region)
        if norm in seen:
            continue
        if norm in available or regions is None:
            resolved.append(region)
            seen.add(norm)
    for norm, original in available.items():
        if norm not in seen:
            resolved.append(original)
            seen.add(norm)
    return resolved


def _significance_star(p_value_adj: float) -> str:
    if not np.isfinite(p_value_adj):
        return ""
    p = float(p_value_adj)
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def _coerce_summary_to_long(
    raw_df: pd.DataFrame,
    settings: FixationPSTHVariabilityPlotSettings,
) -> pd.DataFrame:
    if raw_df.empty and len(raw_df.columns) == 0:
        return _empty_plot_summary_df()

    long_required = {"region", "unit_key", "condition", "fr_variability"}
    if long_required.issubset(raw_df.columns):
        df = raw_df.loc[:, ["region", "unit_key", "condition", "fr_variability"]].copy()
    else:
        ordered_conditions = list(dict.fromkeys([*settings.condition_order, *DEFAULT_CONDITION_ORDER]))
        variability_columns = {
            _variability_column(condition): str(condition)
            for condition in ordered_conditions
        }
        present_variability_columns = [
            column for column in variability_columns if column in raw_df.columns
        ]
        if "unit_key" not in raw_df.columns and {"date", "unit_uuid"}.issubset(raw_df.columns):
            raw_df = raw_df.copy()
            raw_df["unit_key"] = (
                raw_df["date"].astype(str).map(str.strip)
                + "|"
                + raw_df["unit_uuid"].astype(str).map(str.strip)
            )
        required_base = {"region", "unit_key"}
        missing_base = sorted(required_base - set(raw_df.columns))
        if present_variability_columns and not missing_base:
            df = raw_df.melt(
                id_vars=[column for column in raw_df.columns if column not in set(present_variability_columns)],
                value_vars=present_variability_columns,
                var_name="condition_column",
                value_name="fr_variability",
            )
            df["condition"] = df["condition_column"].map(variability_columns)
            df = df.loc[:, ["region", "unit_key", "condition", "fr_variability"]].copy()
        else:
            expected_wide = ", ".join(sorted(variability_columns))
            raise ValueError(
                "Fixation PSTH variability summary CSV missing required columns. "
                "Expected either long-format columns {region, unit_key, condition, fr_variability} "
                f"or wide-format columns including region, unit_key, and one or more of: {expected_wide}"
            )

    if df.empty:
        return _empty_plot_summary_df()
    df["region"] = df["region"].astype(str).map(str.strip)
    df["unit_key"] = df["unit_key"].astype(str).map(str.strip)
    df["condition"] = df["condition"].astype(str).map(str.strip)
    df["fr_variability"] = pd.to_numeric(df["fr_variability"], errors="coerce")
    df["region_norm"] = df["region"].map(_normalize_region_token)
    return df.loc[:, ["region", "unit_key", "condition", "fr_variability", "region_norm"]]


def _load_unit_summary_df(settings: FixationPSTHVariabilityPlotSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.unit_summary_filename, ".csv")
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Fixation PSTH variability unit summary not found: {in_path}")
    raw_df = _read_csv_or_empty(in_path)
    return _coerce_summary_to_long(raw_df, settings)


def _load_within_region_stats_df(settings: FixationPSTHVariabilityPlotSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.within_region_stats_filename, ".csv")
    )
    if not in_path.exists():
        return _empty_within_region_stats_df()
    raw_df = _read_csv_or_empty(in_path)
    if raw_df.empty and len(raw_df.columns) == 0:
        return _empty_within_region_stats_df()
    required = {"region", "condition_a", "condition_b", "p_value_adjusted"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(
            "Fixation PSTH variability stats CSV missing required columns: "
            + ", ".join(missing)
        )
    df = raw_df.copy()
    df["region"] = df["region"].astype(str).map(str.strip)
    df["region_norm"] = df["region"].map(_normalize_region_token)
    df["condition_a"] = df["condition_a"].astype(str).map(str.strip)
    df["condition_b"] = df["condition_b"].astype(str).map(str.strip)
    df["p_value_adjusted"] = pd.to_numeric(df["p_value_adjusted"], errors="coerce")
    if "significant_adjusted" in df.columns:
        df["significant_adjusted"] = df["significant_adjusted"].map(bool)
    else:
        df["significant_adjusted"] = df["p_value_adjusted"].to_numpy(dtype=float) < 0.05
    return df


def _resolve_plot_cfg(
    settings: FixationPSTHVariabilityPlotSettings,
) -> tuple[Optional[dict], Optional[int]]:
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        _, cfg_dpi = resolve_figsize(plot_cfg)
        return plot_cfg, cfg_dpi
    return None, None


def plot_fixation_psth_variability_violins(
    settings: FixationPSTHVariabilityPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Render a 1xN region violin figure for condition-specific variability."""

    if sns is None:
        raise ImportError(
            "seaborn is required for fixation PSTH variability violin plotting. "
            "Install seaborn or use an environment that includes it."
        )

    summary_df = _load_unit_summary_df(settings)
    stats_df = _load_within_region_stats_df(settings)
    if summary_df.empty:
        print("[plot] no fixation PSTH variability rows available for plotting")
        return None

    region_order = _resolve_regions(settings, summary_df, regions=regions)
    condition_order = [
        condition
        for condition in settings.condition_order
        if condition in set(summary_df["condition"].astype(str).tolist())
    ]
    if not region_order or not condition_order:
        print("[plot] unable to resolve regions or conditions for fixation PSTH variability plot")
        return None

    _, cfg_dpi = _resolve_plot_cfg(settings)
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    ext = normalize_extension(settings.output_extension, fallback="pdf")
    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / ensure_filename(settings.output_filename, f".{ext}")

    fig, axes = plt.subplots(
        1,
        len(region_order),
        figsize=(float(settings.figure_width_in), float(settings.figure_height_in)),
        dpi=dpi,
        sharey=False,
        squeeze=False,
    )
    axes_flat = list(axes[0])
    violin_width = 0.88
    region_summaries: list[dict] = []

    for ax, region_label in zip(axes_flat, region_order):
        region_norm = _normalize_region_token(region_label)
        plot_df = summary_df.loc[
            (summary_df["region_norm"] == region_norm)
            & (summary_df["condition"].astype(str).isin(set(condition_order)))
        ].copy()
        plot_df["condition"] = plot_df["condition"].astype(str)
        plot_df["fr_variability"] = pd.to_numeric(plot_df["fr_variability"], errors="coerce")
        plot_df = plot_df.loc[np.isfinite(plot_df["fr_variability"].to_numpy(dtype=float))].copy()
        n_units = int(plot_df["unit_key"].astype(str).nunique()) if not plot_df.empty else 0
        region_summaries.append({"region": str(region_label), "n_units": n_units})

        if plot_df.empty or n_units < int(settings.min_units_per_region):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=9.0)
            ax.set_title(f"{region_label}\nN=0", fontsize=10.0, pad=4.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        sns.violinplot(
            ax=ax,
            data=plot_df,
            x="condition",
            y="fr_variability",
            hue="condition",
            order=list(condition_order),
            hue_order=list(condition_order),
            palette={
                condition: settings.condition_colors.get(condition, "#777777")
                for condition in condition_order
            },
            dodge=False,
            legend=False,
            inner="quart",
            cut=0.0,
            linewidth=0.8,
            width=violin_width,
            saturation=1.0,
        )
        for body in [artist for artist in ax.collections if isinstance(artist, PolyCollection)]:
            body.set_edgecolor("#222222")
            body.set_linewidth(0.65)
            body.set_alpha(1.0)
            body.set_clip_on(False)
            body.set_clip_path(None)
            body.set_rasterized(False)
        for line_artist in [artist for artist in ax.collections if isinstance(artist, LineCollection)]:
            line_artist.set_clip_on(False)
            line_artist.set_clip_path(None)
            line_artist.set_rasterized(False)
        for line_artist in ax.lines:
            line_artist.set_clip_on(False)
            line_artist.set_clip_path(None)
            line_artist.set_rasterized(False)

        ax.set_xticks(np.arange(len(condition_order)))
        ax.set_xticklabels(
            [settings.condition_labels.get(condition, str(condition)) for condition in condition_order],
            rotation=24,
            ha="right",
            rotation_mode="anchor",
            fontsize=8.5,
        )
        ax.set_xlabel("")
        ax.set_ylabel(str(settings.y_label) if ax is axes_flat[0] else "", fontsize=9)
        ax.grid(axis="y", alpha=0.23, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{region_label}\nN={n_units}", fontsize=10.0, pad=4.0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        y_vals = plot_df["fr_variability"].to_numpy(dtype=float)
        y_max = float(np.nanmax(y_vals)) if y_vals.size > 0 else 1.0
        y_min = float(np.nanmin(y_vals)) if y_vals.size > 0 else 0.0
        span = y_max - y_min
        if not np.isfinite(span) or span <= 0.0:
            span = max(abs(y_max), 1.0)
        step = 0.10 * span
        bar_h = 0.028 * span
        pos_map = {str(condition): float(idx) for idx, condition in enumerate(condition_order)}

        region_stats = (
            stats_df.loc[
                (stats_df["region_norm"] == region_norm)
                & (stats_df["condition_a"].astype(str).isin(set(condition_order)))
                & (stats_df["condition_b"].astype(str).isin(set(condition_order)))
                & (stats_df["significant_adjusted"].map(bool))
            ].copy()
            if not stats_df.empty
            else pd.DataFrame()
        )
        if not region_stats.empty:
            region_stats = region_stats.sort_values(["p_value_adjusted", "condition_a", "condition_b"], na_position="last")
        max_stack = 0
        for level, row in enumerate(region_stats.itertuples(index=False)):
            condition_a = str(getattr(row, "condition_a", ""))
            condition_b = str(getattr(row, "condition_b", ""))
            if condition_a not in pos_map or condition_b not in pos_map:
                continue
            stars = _significance_star(float(getattr(row, "p_value_adjusted", np.nan)))
            if not stars:
                continue
            x1 = float(pos_map[condition_a])
            x2 = float(pos_map[condition_b])
            y = y_max + 0.16 * span + step * float(level)
            ax.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], color="#222222", linewidth=0.8)
            ax.text((x1 + x2) / 2.0, y + bar_h + 0.01 * span, stars, ha="center", va="bottom", fontsize=9)
            max_stack = max(max_stack, level + 1)

        top_extra = max(0.20 * span, 0.16 * span + step * float(max_stack) + 0.18 * span)
        ax.set_ylim(y_min - 0.06 * span, y_max + top_extra)

    fig.subplots_adjust(
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=float(settings.top_margin),
        bottom=float(settings.bottom_margin),
        wspace=float(settings.wspace),
    )
    fig.patch.set_facecolor("white")
    save_figure(fig, out_path, ext=ext, dpi=dpi, facecolor="white")
    plt.close(fig)

    return {
        "output_path": str(out_path),
        "region_order": list(region_order),
        "condition_order": list(condition_order),
        "region_summaries": region_summaries,
    }
