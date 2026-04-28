"""Plot fixation peakiness comparisons across conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.core.stats import (
    apply_adjusted_pvalues,
    normalize_pvalue_correction,
    safe_paired_ttest,
)
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
class FixationPeakinessConditionComparisonPlotSettings:
    """Configuration for fixation peakiness condition-comparison plotting."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_peakiness"
    unit_peakiness_filename: str = "unit_peakiness.csv"
    output_subdir: str = "ephys/psth/fixation_peakiness/plots"
    output_filename: str = "fixation_peakiness_condition_comparison"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    condition_order: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_CONDITION_ORDER))
    condition_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONDITION_LABELS))
    condition_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONDITION_COLORS))
    region_order: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_REGION_ORDER))
    region_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_REGION_LABELS))
    figure_width_in: Optional[float] = 7.8
    figure_height_in: Optional[float] = 2.8
    left_margin: float = 0.08
    right_margin: float = 0.995
    top_margin: float = 0.86
    bottom_margin: float = 0.22
    panel_wspace: float = 0.22
    violin_width: float = 0.82
    violin_alpha: float = 0.72
    violin_edgecolor: str = "#2f3136"
    show_violin_points: bool = False
    violin_point_color: str = "#6f6f6f"
    violin_point_alpha: float = 0.22
    violin_point_size: float = 9.0
    density_alpha: float = 0.24
    density_linewidth: float = 1.6
    density_grid_n: int = 400
    score_label: str = "Peakiness Score"
    density_label: str = "Density"
    violin_by_region_output_filename: str = "fixation_peakiness_condition_comparison_by_region_violin"
    density_by_region_output_filename: str = "fixation_peakiness_condition_comparison_by_region_density"
    stats_output_filename: str = "fixation_peakiness_condition_comparison_by_region_stats.csv"
    pvalue_correction: str = "fdr_bh"
    alpha: float = 0.05
    min_paired_units_per_region: int = 2
    show_suptitle: bool = False


def _normalize_region_token(value: object) -> str:
    return str(value).strip().lower()


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


def _condition_column(condition: object) -> str:
    return f"{str(condition).strip()}_peakiness_score"


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        raise ValueError(f"Unable to read CSV '{path}': {exc}") from exc


def _resolve_figsize_and_dpi(
    settings: FixationPeakinessConditionComparisonPlotSettings,
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


def _available_conditions(df: pd.DataFrame) -> list[str]:
    suffix = "_peakiness_score"
    out: list[str] = []
    seen: set[str] = set()
    for column in df.columns:
        token = str(column).strip()
        if token == "peakiness_score" or not token.endswith(suffix):
            continue
        condition = token[: -len(suffix)].strip()
        if not condition or condition in seen:
            continue
        out.append(condition)
        seen.add(condition)
    return out


def _load_unit_peakiness_df(
    settings: FixationPeakinessConditionComparisonPlotSettings,
) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.input_subdir)
        / ensure_filename(settings.unit_peakiness_filename, ".csv")
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Fixation peakiness unit CSV not found: {in_path}")

    df = _read_csv_or_empty(in_path)
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=["unit_uuid", "region", "region_norm"])

    required = {"unit_uuid", "region"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Fixation peakiness unit CSV is missing required columns: "
            + ", ".join(missing)
        )

    conditions = _available_conditions(df)
    if not conditions:
        raise ValueError(
            "Fixation peakiness unit CSV is missing condition-specific peakiness columns. "
            "Expected one or more columns ending in '_peakiness_score'."
        )

    out = df.copy()
    out["unit_uuid"] = out["unit_uuid"].astype(str).map(str.strip)
    out["region"] = out["region"].astype(str).map(str.strip)
    out["region_norm"] = out["region"].map(_normalize_region_token)
    for condition in conditions:
        column = _condition_column(condition)
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _resolve_conditions(
    df: pd.DataFrame,
    settings: FixationPeakinessConditionComparisonPlotSettings,
) -> list[str]:
    requested = _dedupe(settings.condition_order)
    available = _available_conditions(df)
    available_set = set(available)

    resolved: list[str] = []
    seen: set[str] = set()
    for condition in requested:
        if condition in available_set and condition not in seen:
            resolved.append(condition)
            seen.add(condition)
    for condition in available:
        if condition not in seen:
            resolved.append(condition)
            seen.add(condition)
    return resolved


def _resolve_regions(
    df: pd.DataFrame,
    settings: FixationPeakinessConditionComparisonPlotSettings,
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


def _region_display_label(
    token: str,
    settings: FixationPeakinessConditionComparisonPlotSettings,
) -> str:
    if token in settings.region_labels:
        return str(settings.region_labels[token])
    for key, value in settings.region_labels.items():
        if _normalize_region_token(key) == token:
            return str(value)
    return str(token)


def _condition_display_label(
    token: str,
    settings: FixationPeakinessConditionComparisonPlotSettings,
) -> str:
    if token in settings.condition_labels:
        return str(settings.condition_labels[token])
    for key, value in settings.condition_labels.items():
        if str(key).strip() == token:
            return str(value)
    return str(token)


def _build_long_df(
    wide_df: pd.DataFrame,
    settings: FixationPeakinessConditionComparisonPlotSettings,
    *,
    condition_order: Sequence[str],
) -> pd.DataFrame:
    column_map = {
        _condition_column(condition): str(condition)
        for condition in condition_order
        if _condition_column(condition) in wide_df.columns
    }
    if not column_map:
        return pd.DataFrame(
            columns=[
                "unit_uuid",
                "region",
                "region_norm",
                "region_label",
                "condition",
                "condition_label",
                "peakiness_score",
            ]
        )
    out = wide_df.melt(
        id_vars=["unit_uuid", "region", "region_norm"],
        value_vars=list(column_map.keys()),
        var_name="condition_column",
        value_name="peakiness_score",
    ).copy()
    out["condition"] = out["condition_column"].map(column_map)
    out["condition_label"] = out["condition"].map(
        lambda token: _condition_display_label(str(token), settings)
    )
    out["region_label"] = out["region_norm"].map(
        lambda token: _region_display_label(str(token), settings)
    )
    out["peakiness_score"] = pd.to_numeric(out["peakiness_score"], errors="coerce")
    return out.loc[
        :,
        [
            "unit_uuid",
            "region",
            "region_norm",
            "region_label",
            "condition",
            "condition_label",
            "peakiness_score",
        ],
    ]


def _summarize_conditions(
    long_df: pd.DataFrame,
    *,
    condition_order: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for condition in condition_order:
        values = pd.to_numeric(
            long_df.loc[long_df["condition"].astype(str) == str(condition), "peakiness_score"],
            errors="coerce",
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            out[str(condition)] = {
                "n_units": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "q25": np.nan,
                "q75": np.nan,
            }
            continue
        out[str(condition)] = {
            "n_units": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
    return out


def _build_region_mean_summary(
    wide_df: pd.DataFrame,
    settings: FixationPeakinessConditionComparisonPlotSettings,
    *,
    region_order: Sequence[str],
    condition_order: Sequence[str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for token in region_order:
        region_df = wide_df.loc[wide_df["region_norm"].astype(str) == str(token)].copy()
        row: dict[str, object] = {
            "region": _region_display_label(str(token), settings),
            "region_token": str(token),
            "n_units": int(region_df["unit_uuid"].astype(str).nunique()),
        }
        for condition in condition_order:
            column = _condition_column(condition)
            values = pd.to_numeric(region_df.get(column, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{condition}_mean"] = float(np.mean(values)) if values.size > 0 else np.nan
            row[f"{condition}_median"] = float(np.median(values)) if values.size > 0 else np.nan
            row[f"{condition}_n_units"] = int(values.size)
        out.append(row)
    return out


def _empty_stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "region",
            "region_label",
            "n_units_region_total",
            "condition_a",
            "condition_b",
            "condition_pair",
            "test_name",
            "n_units_paired",
            "mean_a",
            "mean_b",
            "mean_difference_a_minus_b",
            "median_a",
            "median_b",
            "statistic",
            "p_value",
            "p_value_adjusted",
            "pvalue_correction",
            "alpha",
            "significant_adjusted",
        ]
    )


def _build_within_region_stats(
    wide_df: pd.DataFrame,
    settings: FixationPeakinessConditionComparisonPlotSettings,
    *,
    region_order: Sequence[str],
    condition_order: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    min_pairs = int(max(settings.min_paired_units_per_region, 2))

    for token in region_order:
        region_df = wide_df.loc[wide_df["region_norm"].astype(str) == str(token)].copy()
        if region_df.empty:
            continue
        n_units_region = int(region_df["unit_uuid"].astype(str).nunique())
        available_conditions = [
            condition
            for condition in condition_order
            if _condition_column(condition) in region_df.columns
            and np.isfinite(
                pd.to_numeric(region_df[_condition_column(condition)], errors="coerce").to_numpy(dtype=float)
            ).any()
        ]
        for condition_a, condition_b in combinations(available_conditions, 2):
            arr_a = pd.to_numeric(
                region_df[_condition_column(condition_a)],
                errors="coerce",
            ).to_numpy(dtype=float)
            arr_b = pd.to_numeric(
                region_df[_condition_column(condition_b)],
                errors="coerce",
            ).to_numpy(dtype=float)
            stat, p_value, n_paired = safe_paired_ttest(arr_a, arr_b)
            if int(n_paired) < min_pairs:
                continue
            valid = np.isfinite(arr_a) & np.isfinite(arr_b)
            paired_a = arr_a[valid]
            paired_b = arr_b[valid]
            rows.append(
                {
                    "region": str(token),
                    "region_label": _region_display_label(str(token), settings),
                    "n_units_region_total": int(n_units_region),
                    "condition_a": str(condition_a),
                    "condition_b": str(condition_b),
                    "condition_pair": f"{condition_a}__vs__{condition_b}",
                    "test_name": "paired_ttest",
                    "n_units_paired": int(n_paired),
                    "mean_a": float(np.mean(paired_a)) if paired_a.size > 0 else np.nan,
                    "mean_b": float(np.mean(paired_b)) if paired_b.size > 0 else np.nan,
                    "mean_difference_a_minus_b": float(np.mean(paired_a - paired_b)) if paired_a.size > 0 else np.nan,
                    "median_a": float(np.median(paired_a)) if paired_a.size > 0 else np.nan,
                    "median_b": float(np.median(paired_b)) if paired_b.size > 0 else np.nan,
                    "statistic": float(stat) if np.isfinite(stat) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )

    if not rows:
        return _empty_stats_df()

    out_df = pd.DataFrame(rows)
    correction = normalize_pvalue_correction(settings.pvalue_correction)
    out_df = apply_adjusted_pvalues(
        out_df,
        p_col="p_value",
        out_col="p_value_adjusted",
        method=correction,
        group_cols=("region",),
    )
    out_df["pvalue_correction"] = str(correction)
    out_df["alpha"] = float(settings.alpha)
    out_df["significant_adjusted"] = (
        pd.to_numeric(out_df["p_value_adjusted"], errors="coerce").to_numpy(dtype=float)
        < float(settings.alpha)
    )
    return out_df.sort_values(["region", "condition_a", "condition_b"]).reset_index(drop=True)


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


def _estimate_density_curve(values: np.ndarray, *, x_grid: np.ndarray) -> np.ndarray:
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


def _save_plot_path(
    settings: FixationPeakinessConditionComparisonPlotSettings,
    *,
    output_root: Path,
    filename: str,
) -> Path:
    ext = normalize_extension(settings.output_extension, fallback="pdf")
    stem = Path(str(filename).strip()).stem or "fixation_peakiness_condition_comparison"
    return output_root / f"{stem}.{ext}"


def _jitter_values(
    n_values: int,
    *,
    center: float,
    width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_values <= 0:
        return np.asarray([], dtype=float)
    if n_values == 1:
        return np.asarray([center], dtype=float)
    return center + rng.uniform(-width, width, size=n_values)


def _plot_pooled_violin(
    settings: FixationPeakinessConditionComparisonPlotSettings,
    *,
    long_df: pd.DataFrame,
    condition_order: Sequence[str],
    out_path: Path,
    dpi: Optional[int],
) -> None:
    figsize, _ = _resolve_figsize_and_dpi(settings)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    rng = np.random.default_rng(0)

    positions = np.arange(len(condition_order), dtype=float)
    finite_values = pd.to_numeric(long_df["peakiness_score"], errors="coerce").to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size > 0:
        y_min = min(0.0, float(np.min(finite_values)))
        y_max = float(np.max(finite_values))
        y_span = max(y_max - y_min, y_max, 1.0)
        ax.set_ylim(y_min - 0.04 * y_span, y_max + 0.12 * y_span)

    nonempty_data: list[np.ndarray] = []
    nonempty_positions: list[float] = []
    nonempty_conditions: list[str] = []
    for pos, condition in zip(positions, condition_order):
        values = pd.to_numeric(
            long_df.loc[long_df["condition"].astype(str) == str(condition), "peakiness_score"],
            errors="coerce",
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        nonempty_data.append(values)
        nonempty_positions.append(float(pos))
        nonempty_conditions.append(str(condition))

    if nonempty_data:
        violin_parts = ax.violinplot(
            nonempty_data,
            positions=nonempty_positions,
            widths=float(settings.violin_width),
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, condition in zip(violin_parts["bodies"], nonempty_conditions):
            color = str(settings.condition_colors.get(condition, "#7a7a7a"))
            body.set_facecolor(color)
            body.set_edgecolor(str(settings.violin_edgecolor))
            body.set_alpha(float(settings.violin_alpha))
            body.set_linewidth(0.8)

    for pos, condition in zip(positions, condition_order):
        values = pd.to_numeric(
            long_df.loc[long_df["condition"].astype(str) == str(condition), "peakiness_score"],
            errors="coerce",
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.vlines(pos, q25, q75, color="#202020", linewidth=1.0, zorder=3)
        ax.scatter([pos], [median], s=18.0, color="#202020", zorder=4)
        if settings.show_violin_points:
            x_jitter = _jitter_values(
                int(values.size),
                center=float(pos),
                width=min(0.14, 0.28 * float(settings.violin_width)),
                rng=rng,
            )
            ax.scatter(
                x_jitter,
                values,
                s=float(settings.violin_point_size),
                color=str(settings.violin_point_color),
                alpha=float(settings.violin_point_alpha),
                linewidths=0.0,
                zorder=2,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_condition_display_label(condition, settings) for condition in condition_order],
        rotation=28,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel(str(settings.score_label), fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    if settings.show_suptitle:
        fig.suptitle("Fixation Peakiness Condition Comparison", fontsize=12)
    fig.subplots_adjust(
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=float(settings.top_margin),
        bottom=float(settings.bottom_margin),
    )
    save_figure(fig, out_path, ext=out_path.suffix.lstrip("."), dpi=dpi, facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)


def _plot_violin_by_region(
    settings: FixationPeakinessConditionComparisonPlotSettings,
    *,
    long_df: pd.DataFrame,
    region_order: Sequence[str],
    condition_order: Sequence[str],
    out_path: Path,
    dpi: Optional[int],
) -> None:
    figsize, _ = _resolve_figsize_and_dpi(settings)
    fig, axes = plt.subplots(
        1,
        len(region_order),
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.ravel()
    rng = np.random.default_rng(0)

    finite_values = pd.to_numeric(long_df["peakiness_score"], errors="coerce").to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size > 0:
        y_min = min(0.0, float(np.min(finite_values)))
        y_max = float(np.max(finite_values))
        y_span = max(y_max - y_min, y_max, 1.0)
        shared_ylim = (y_min - 0.04 * y_span, y_max + 0.12 * y_span)
    else:
        shared_ylim = (0.0, 1.0)

    positions = np.arange(len(condition_order), dtype=float)
    for idx, token in enumerate(region_order):
        ax = axes_flat[idx]
        region_df = long_df.loc[long_df["region_norm"].astype(str) == str(token)].copy()

        nonempty_data: list[np.ndarray] = []
        nonempty_positions: list[float] = []
        nonempty_conditions: list[str] = []
        for pos, condition in zip(positions, condition_order):
            values = pd.to_numeric(
                region_df.loc[region_df["condition"].astype(str) == str(condition), "peakiness_score"],
                errors="coerce",
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            nonempty_data.append(values)
            nonempty_positions.append(float(pos))
            nonempty_conditions.append(str(condition))

        if nonempty_data:
            violin_parts = ax.violinplot(
                nonempty_data,
                positions=nonempty_positions,
                widths=float(settings.violin_width),
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, condition in zip(violin_parts["bodies"], nonempty_conditions):
                color = str(settings.condition_colors.get(condition, "#7a7a7a"))
                body.set_facecolor(color)
                body.set_edgecolor(str(settings.violin_edgecolor))
                body.set_alpha(float(settings.violin_alpha))
                body.set_linewidth(0.8)

        for pos, condition in zip(positions, condition_order):
            values = pd.to_numeric(
                region_df.loc[region_df["condition"].astype(str) == str(condition), "peakiness_score"],
                errors="coerce",
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            ax.vlines(pos, q25, q75, color="#202020", linewidth=1.0, zorder=3)
            ax.scatter([pos], [median], s=18.0, color="#202020", zorder=4)
            if settings.show_violin_points:
                x_jitter = _jitter_values(
                    int(values.size),
                    center=float(pos),
                    width=min(0.14, 0.28 * float(settings.violin_width)),
                    rng=rng,
                )
                ax.scatter(
                    x_jitter,
                    values,
                    s=float(settings.violin_point_size),
                    color=str(settings.violin_point_color),
                    alpha=float(settings.violin_point_alpha),
                    linewidths=0.0,
                    zorder=2,
                )

        ax.set_title(_region_display_label(str(token), settings), fontsize=10, pad=6)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [_condition_display_label(condition, settings) for condition in condition_order],
            rotation=28,
            ha="right",
            fontsize=8,
        )
        ax.set_ylim(*shared_ylim)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        if idx == 0:
            ax.set_ylabel(str(settings.score_label), fontsize=9)
        else:
            ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_visible(False)

    if settings.show_suptitle:
        fig.suptitle("Fixation Peakiness by Condition and Region", fontsize=12)
    fig.subplots_adjust(
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=float(settings.top_margin),
        bottom=float(settings.bottom_margin),
        wspace=float(settings.panel_wspace),
    )
    save_figure(fig, out_path, ext=out_path.suffix.lstrip("."), dpi=dpi, facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)


def _plot_density_by_region(
    settings: FixationPeakinessConditionComparisonPlotSettings,
    *,
    long_df: pd.DataFrame,
    region_order: Sequence[str],
    condition_order: Sequence[str],
    out_path: Path,
    dpi: Optional[int],
) -> None:
    figsize, _ = _resolve_figsize_and_dpi(settings)
    fig, axes = plt.subplots(
        1,
        len(region_order),
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()

    finite_scores = pd.to_numeric(long_df["peakiness_score"], errors="coerce").to_numpy(dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size > 0:
        x_min = min(0.0, float(np.min(finite_scores)))
        x_max = float(np.max(finite_scores))
    else:
        x_min = 0.0
        x_max = 1.0
    x_span = max(x_max - x_min, x_max, 1.0)
    x_pad = 0.06 * x_span
    shared_xlim = (x_min - 0.25 * x_pad, x_max + x_pad)
    x_grid = np.linspace(shared_xlim[0], shared_xlim[1], num=max(64, int(settings.density_grid_n)), dtype=float)

    density_lookup: dict[tuple[str, str], np.ndarray] = {}
    max_density = 0.0
    for token in region_order:
        for condition in condition_order:
            values = pd.to_numeric(
                long_df.loc[
                    (long_df["region_norm"].astype(str) == str(token))
                    & (long_df["condition"].astype(str) == str(condition)),
                    "peakiness_score",
                ],
                errors="coerce",
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            density = _estimate_density_curve(values, x_grid=x_grid)
            density_lookup[(str(token), str(condition))] = density
            if density.size > 0:
                max_density = max(max_density, float(np.max(density)))
    shared_ymax = max(max_density, 1.0e-6) * 1.08

    legend_handles = []
    legend_labels = []
    for idx, token in enumerate(region_order):
        ax = axes_flat[idx]
        for condition in condition_order:
            density = density_lookup[(str(token), str(condition))]
            if density.size == 0 or not np.isfinite(density).any() or float(np.max(density)) <= 0.0:
                continue
            color = str(settings.condition_colors.get(str(condition), "#7a7a7a"))
            label = _condition_display_label(str(condition), settings)
            ax.fill_between(
                x_grid,
                0.0,
                density,
                color=color,
                alpha=float(settings.density_alpha),
                linewidth=0.0,
                zorder=1,
            )
            line, = ax.plot(
                x_grid,
                density,
                color=color,
                linewidth=float(settings.density_linewidth),
                label=label,
                zorder=2,
            )
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_title(_region_display_label(str(token), settings), fontsize=10, pad=6)
        ax.set_xlim(*shared_xlim)
        ax.set_ylim(0.0, shared_ymax)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        if idx == 0:
            ax.set_ylabel(str(settings.density_label), fontsize=9)
        else:
            ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_visible(False)

    fig.supxlabel(str(settings.score_label), fontsize=10, y=max(0.02, 0.45 * float(settings.bottom_margin)))
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_handles), 3),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.995),
        )
    if settings.show_suptitle:
        fig.suptitle("Fixation Peakiness Density by Condition and Region", fontsize=12)
    fig.subplots_adjust(
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=min(float(settings.top_margin), 0.84),
        bottom=float(settings.bottom_margin),
        wspace=float(settings.panel_wspace),
    )
    save_figure(fig, out_path, ext=out_path.suffix.lstrip("."), dpi=dpi, facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)


def plot_fixation_peakiness_condition_comparison(
    settings: FixationPeakinessConditionComparisonPlotSettings,
) -> Optional[dict[str, object]]:
    """Render a pooled across-region fixation peakiness comparison figure."""

    wide_df = _load_unit_peakiness_df(settings)
    if wide_df.empty:
        print("[plot] no fixation peakiness unit rows found")
        return None

    condition_order = _resolve_conditions(wide_df, settings)
    if not condition_order:
        print("[plot] no fixation peakiness condition columns available to render")
        return None

    long_df = _build_long_df(wide_df, settings, condition_order=condition_order)
    long_df = long_df.loc[np.isfinite(pd.to_numeric(long_df["peakiness_score"], errors="coerce"))].copy()
    if long_df.empty:
        print("[plot] no finite fixation peakiness rows remain after filtering")
        return None

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = _save_plot_path(settings, output_root=out_root, filename=str(settings.output_filename))
    _, dpi = _resolve_figsize_and_dpi(settings)
    _plot_pooled_violin(
        settings,
        long_df=long_df,
        condition_order=condition_order,
        out_path=out_path,
        dpi=dpi,
    )
    return {
        "output_path": out_path,
        "conditions": list(condition_order),
        "condition_summary": _summarize_conditions(long_df, condition_order=condition_order),
        "n_units_total": int(wide_df["unit_uuid"].astype(str).nunique()),
    }


def plot_fixation_peakiness_condition_comparison_by_region(
    settings: FixationPeakinessConditionComparisonPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> Optional[dict[str, object]]:
    """Render region-wise condition-comparison plots and paired statistics."""

    wide_df = _load_unit_peakiness_df(settings)
    if wide_df.empty:
        print("[plot] no fixation peakiness unit rows found")
        return None

    condition_order = _resolve_conditions(wide_df, settings)
    region_order = _resolve_regions(wide_df, settings, regions=regions)
    if not condition_order or not region_order:
        print("[plot] no fixation peakiness condition-comparison panels available to render")
        return None

    wide_df = wide_df.loc[wide_df["region_norm"].astype(str).isin(set(region_order))].copy()
    long_df = _build_long_df(wide_df, settings, condition_order=condition_order)
    long_df = long_df.loc[
        long_df["region_norm"].astype(str).isin(set(region_order))
        & np.isfinite(pd.to_numeric(long_df["peakiness_score"], errors="coerce"))
    ].copy()
    if long_df.empty:
        print("[plot] no finite fixation peakiness rows remain after filtering")
        return None

    stats_df = _build_within_region_stats(
        wide_df,
        settings,
        region_order=region_order,
        condition_order=condition_order,
    )
    mean_summary = _build_region_mean_summary(
        wide_df,
        settings,
        region_order=region_order,
        condition_order=condition_order,
    )

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    violin_out_path = _save_plot_path(
        settings,
        output_root=out_root,
        filename=str(settings.violin_by_region_output_filename),
    )
    density_out_path = _save_plot_path(
        settings,
        output_root=out_root,
        filename=str(settings.density_by_region_output_filename),
    )
    stats_out_path = out_root / ensure_filename(str(settings.stats_output_filename), ".csv")

    _, dpi = _resolve_figsize_and_dpi(settings)
    _plot_violin_by_region(
        settings,
        long_df=long_df,
        region_order=region_order,
        condition_order=condition_order,
        out_path=violin_out_path,
        dpi=dpi,
    )
    _plot_density_by_region(
        settings,
        long_df=long_df,
        region_order=region_order,
        condition_order=condition_order,
        out_path=density_out_path,
        dpi=dpi,
    )
    stats_df.to_csv(stats_out_path, index=False)

    return {
        "violin_output_path": violin_out_path,
        "density_output_path": density_out_path,
        "stats_output_path": stats_out_path,
        "regions": list(region_order),
        "conditions": list(condition_order),
        "stats_df": stats_df,
        "mean_summary": mean_summary,
        "condition_summary": _summarize_conditions(long_df, condition_order=condition_order),
    }
