"""Plot region-wise fixation preference-index heatmaps by condition pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_preference_index import (
    DEFAULT_INDEX_NAME_BY_PAIR,
)
from dal_monte_2022_analysis.ephys.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.runtime.io.plot_output import (
    normalize_extension,
    save_figure,
)
from dal_monte_2022_analysis.utils.filenames import ensure_filename
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


DEFAULT_REGION_ORDER = ("BLA", "ACCg", "dmPFC", "OFC")
DEFAULT_PAIR_ORDER = tuple(
    f"{cond_a}__vs__{cond_b}"
    for cond_a, cond_b in DEFAULT_INDEX_NAME_BY_PAIR.keys()
)
_NORM_MODE_UNIT_MAX_SUM = "unit_max_sum"
_NORM_MODE_PER_BIN_SUM = "per_bin_sum"
_INDEX_COLUMN_BY_NORM_MODE = {
    _NORM_MODE_UNIT_MAX_SUM: "preference_index_unit_max_sum",
    _NORM_MODE_PER_BIN_SUM: "preference_index_per_bin_sum",
}
_UNIT_FILTER_PAIR = "pair_selective"
_UNIT_FILTER_ANY = "any_selective"
_UNIT_FILTER_ALL = "all"
_W_R_CMAP = LinearSegmentedColormap.from_list(
    "index_abs_wr",
    ["#ffffff", "#b2182b"],
)


@dataclass
class FixationPreferenceIndexHeatmapPlotSettings:
    """Configuration for per-pair region heatmap plotting of preference indices."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    input_subdir: str = "ephys/psth/fixation_psth_preference_index"
    timeseries_filename: str = "preference_index_timeseries.csv"
    output_subdir: str = "ephys/psth/fixation_psth_preference_index/plots"
    output_filename: str = "preference_index_heatmaps"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    include_only_pair_selective_units: bool = True
    unit_filter_mode: Optional[str] = None
    sort_reference_pair_label: Optional[str] = None
    combine_pairs_into_single_figure: bool = True
    normalization_mode: str = _NORM_MODE_UNIT_MAX_SUM
    region_order: Optional[Sequence[str]] = DEFAULT_REGION_ORDER
    default_pair_order: Optional[Sequence[str]] = DEFAULT_PAIR_ORDER
    figure_width_in: float = 8.5
    figure_height_in: float = 4.4
    left_margin: float = 0.04
    right_margin: float = 0.992
    top_margin: float = 0.86
    bottom_margin: float = 0.26
    panel_wspace: float = 0.10
    panel_hspace: float = 0.24
    show_suptitle: bool = False
    colorbar_orientation: str = "horizontal"
    colorbar_label: str = "|Preference Index|"
    colorbar_fraction: float = 0.025
    colorbar_pad: float = 0.08
    colorbar_shrink: float = 0.72
    colorbar_aspect: float = 48.0


def _coerce_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return False
        return float(value) != 0.0
    token = str(value).strip().lower()
    return token in {"1", "true", "t", "yes", "y"}


def _normalize_region(region: object) -> str:
    return str(region).strip().lower()


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = str(item).strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def _resolve_figsize_and_dpi(
    settings: FixationPreferenceIndexHeatmapPlotSettings,
) -> tuple[list[float], Optional[int]]:
    cfg_dpi = None
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        _, cfg_dpi = resolve_figsize(plot_cfg)
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    return [float(settings.figure_width_in), float(settings.figure_height_in)], dpi


def _normalize_normalization_mode(mode: object) -> str:
    token = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        _NORM_MODE_UNIT_MAX_SUM: _NORM_MODE_UNIT_MAX_SUM,
        "max_sum": _NORM_MODE_UNIT_MAX_SUM,
        "unit_max": _NORM_MODE_UNIT_MAX_SUM,
        _NORM_MODE_PER_BIN_SUM: _NORM_MODE_PER_BIN_SUM,
        "per_bin": _NORM_MODE_PER_BIN_SUM,
        "binwise": _NORM_MODE_PER_BIN_SUM,
    }
    resolved = aliases.get(token)
    if resolved is None:
        raise ValueError(
            "Unsupported normalization_mode for fixation preference-index heatmap plotting. "
            "Expected one of: unit_max_sum, per_bin_sum."
        )
    return resolved


def _normalize_unit_filter_mode(mode: object) -> str:
    token = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        _UNIT_FILTER_PAIR: _UNIT_FILTER_PAIR,
        "pair": _UNIT_FILTER_PAIR,
        "pairwise": _UNIT_FILTER_PAIR,
        "ab": _UNIT_FILTER_PAIR,
        _UNIT_FILTER_ANY: _UNIT_FILTER_ANY,
        "any": _UNIT_FILTER_ANY,
        "unit": _UNIT_FILTER_ANY,
        "significant": _UNIT_FILTER_ANY,
        "any_significant": _UNIT_FILTER_ANY,
        _UNIT_FILTER_ALL: _UNIT_FILTER_ALL,
        "all_units": _UNIT_FILTER_ALL,
        "none": _UNIT_FILTER_ALL,
    }
    resolved = aliases.get(token)
    if resolved is None:
        raise ValueError(
            "Unsupported unit_filter_mode for fixation preference-index heatmap plotting. "
            "Expected one of: pair_selective, any_selective, all."
        )
    return resolved


def _resolve_unit_filter_mode(settings: FixationPreferenceIndexHeatmapPlotSettings) -> str:
    if settings.unit_filter_mode is not None and str(settings.unit_filter_mode).strip():
        return _normalize_unit_filter_mode(settings.unit_filter_mode)
    return _UNIT_FILTER_PAIR if bool(settings.include_only_pair_selective_units) else _UNIT_FILTER_ALL


def _resolve_value_column(
    df: pd.DataFrame,
    normalization_mode: str,
) -> str:
    preferred_column = _INDEX_COLUMN_BY_NORM_MODE[normalization_mode]
    if preferred_column in df.columns:
        return preferred_column

    if "preference_index" not in df.columns:
        raise ValueError(
            "Preference-index timeseries CSV is missing both the requested value column "
            f"'{preferred_column}' and legacy column 'preference_index'."
        )

    if "normalization_mode" in df.columns:
        mode_tokens: set[str] = set()
        for value in df["normalization_mode"].dropna().tolist():
            try:
                mode_tokens.add(_normalize_normalization_mode(value))
            except ValueError:
                continue
        if len(mode_tokens) == 1:
            only_mode = next(iter(mode_tokens))
            if only_mode != normalization_mode:
                raise ValueError(
                    "Requested plot normalization_mode "
                    f"'{normalization_mode}', but legacy timeseries contains only "
                    f"'{only_mode}' in 'preference_index'. Rerun analysis to store both modes."
                )

    print(
        "[plot] warning: requested normalized index column "
        f"'{preferred_column}' not found; falling back to legacy 'preference_index'."
    )
    return "preference_index"


def _load_timeseries_df(settings: FixationPreferenceIndexHeatmapPlotSettings) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_root = build_analysis_output_dir(cfg, settings.input_subdir)
    in_path = in_root / ensure_filename(settings.timeseries_filename, ".csv")
    if not in_path.exists():
        raise FileNotFoundError(f"Preference-index timeseries CSV not found: {in_path}")
    df = pd.read_csv(in_path)
    required = {
        "pair_label",
        "region",
        "unit_key",
        "bin_index",
        "bin_center_s",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Preference-index timeseries CSV missing required columns: "
            + ", ".join(missing)
        )
    if not any(
        column in df.columns
        for column in ("preference_index", "preference_index_unit_max_sum", "preference_index_per_bin_sum")
    ):
        raise ValueError(
            "Preference-index timeseries CSV must include at least one of: "
            "preference_index, preference_index_unit_max_sum, preference_index_per_bin_sum."
        )
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    df["pair_label"] = df["pair_label"].astype(str).map(lambda token: token.strip())
    df["region"] = df["region"].astype(str).map(lambda token: token.strip())
    df["region_norm"] = df["region"].map(_normalize_region)
    df["unit_key"] = df["unit_key"].astype(str).map(lambda token: token.strip())
    df["bin_index"] = pd.to_numeric(df["bin_index"], errors="coerce").astype("Int64")
    df["bin_center_s"] = pd.to_numeric(df["bin_center_s"], errors="coerce")
    for column in ("preference_index", "preference_index_unit_max_sum", "preference_index_per_bin_sum"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("is_selective_pair", "is_selective_unit", "is_selective_any_pair"):
        if column in df.columns:
            df[column] = df[column].map(_coerce_bool)
        else:
            df[column] = False
    if "index_name" in df.columns:
        df["index_name"] = df["index_name"].fillna("").astype(str)
    else:
        df["index_name"] = ""
    return df


def _resolve_pair_order(
    df: pd.DataFrame,
    *,
    requested_pairs: Optional[Sequence[str]],
    default_pair_order: Optional[Sequence[str]],
) -> list[str]:
    present_pairs = [str(token) for token in df["pair_label"].dropna().astype(str).unique().tolist()]
    if requested_pairs:
        requested = _dedupe_preserve_order([str(token) for token in requested_pairs])
        present_set = set(present_pairs)
        return [pair for pair in requested if pair in present_set]

    order: list[str] = []
    present_set = set(present_pairs)
    if default_pair_order is not None:
        for pair in _dedupe_preserve_order([str(token) for token in default_pair_order]):
            if pair in present_set:
                order.append(pair)
    for pair in sorted(present_pairs):
        if pair not in order:
            order.append(pair)
    return order


def _resolve_region_order(
    df: pd.DataFrame,
    *,
    requested_regions: Optional[Sequence[str]],
    default_region_order: Optional[Sequence[str]],
) -> list[str]:
    if requested_regions:
        return _dedupe_preserve_order([str(region) for region in requested_regions])
    if default_region_order:
        return _dedupe_preserve_order([str(region) for region in default_region_order])
    return sorted(df["region"].dropna().astype(str).unique().tolist())


def _safe_token(text: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text).strip())
    token = token.strip("_")
    return token or "unknown"


def _sorted_unit_tick_positions_labels(
    n_units: int,
    *,
    max_ticks: int = 4,
) -> tuple[np.ndarray, list[str]]:
    if n_units <= 0:
        return np.asarray([], dtype=float), []
    if n_units <= max(1, int(max_ticks)):
        labels = np.arange(1, n_units + 1, dtype=int)
    else:
        labels = np.rint(np.linspace(1.0, float(n_units), num=max(2, int(max_ticks)))).astype(int)
        labels = np.unique(np.clip(labels, 1, n_units))
    y_pos = n_units - labels.astype(float) + 0.5
    return y_pos, [str(int(v)) for v in labels.tolist()]


def _selective_any_unit_mask(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=bool)

    if "is_selective_unit" in df.columns and bool(df["is_selective_unit"].map(_coerce_bool).any()):
        selective_units = set(
            df.loc[df["is_selective_unit"].map(_coerce_bool), "unit_key"].astype(str).tolist()
        )
        return df["unit_key"].astype(str).isin(selective_units).to_numpy(dtype=bool)

    if "is_selective_any_pair" in df.columns and bool(df["is_selective_any_pair"].map(_coerce_bool).any()):
        selective_units = set(
            df.loc[df["is_selective_any_pair"].map(_coerce_bool), "unit_key"].astype(str).tolist()
        )
        return df["unit_key"].astype(str).isin(selective_units).to_numpy(dtype=bool)

    if "is_selective_pair" in df.columns:
        selective_units = set(
            df.loc[df["is_selective_pair"].map(_coerce_bool), "unit_key"].astype(str).tolist()
        )
        return df["unit_key"].astype(str).isin(selective_units).to_numpy(dtype=bool)

    return np.ones(len(df), dtype=bool)


def _order_units_by_abs_peak(
    piv: pd.DataFrame,
    *,
    bin_centers: np.ndarray,
) -> list[str]:
    sort_rows: list[tuple[float, float, str]] = []
    for unit_key, row in piv.iterrows():
        values = row.to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            peak_bin = np.inf
            peak_value = -np.inf
        else:
            abs_values = np.abs(values)
            idx = int(np.nanargmax(abs_values))
            peak_bin = float(bin_centers[idx])
            peak_value = float(abs_values[idx])
        sort_rows.append((peak_bin, peak_value, str(unit_key)))
    # Reverse sort with origin='lower' so earliest-peak units appear at the top.
    sort_rows.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return [unit_key for _peak_bin, _peak_val, unit_key in sort_rows]


def _region_matrix(
    region_df: pd.DataFrame,
    *,
    value_column: str,
    forced_unit_order: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if region_df.empty:
        return np.empty((0, 0), dtype=float), np.asarray([], dtype=float), []

    meta = (
        region_df.loc[:, ["bin_index", "bin_center_s"]]
        .dropna(subset=["bin_index", "bin_center_s"])
        .copy()
    )
    if meta.empty:
        return np.empty((0, 0), dtype=float), np.asarray([], dtype=float), []
    meta["bin_index"] = meta["bin_index"].astype(int)
    meta = meta.drop_duplicates(subset=["bin_index"]).sort_values("bin_index")
    bins = meta["bin_index"].astype(int).to_numpy()
    bin_centers = meta["bin_center_s"].astype(float).to_numpy()
    if bins.size == 0:
        return np.empty((0, 0), dtype=float), np.asarray([], dtype=float), []

    piv = region_df.pivot_table(
        index="unit_key",
        columns="bin_index",
        values=value_column,
        aggfunc="mean",
    )
    piv = piv.reindex(columns=bins)
    if piv.empty:
        return np.empty((0, bins.size), dtype=float), bin_centers, []

    if forced_unit_order is not None:
        ordered_units = _dedupe_preserve_order([str(unit) for unit in forced_unit_order])
        if ordered_units:
            present = set(piv.index.astype(str).tolist())
            ordered_units = [unit for unit in ordered_units if unit in present]
        if ordered_units:
            mat = piv.reindex(index=ordered_units).to_numpy(dtype=float)
            return mat, bin_centers, ordered_units

    ordered_units = _order_units_by_abs_peak(
        piv,
        bin_centers=bin_centers,
    )
    mat = piv.loc[ordered_units].to_numpy(dtype=float)
    return mat, bin_centers, ordered_units


def _index_name_for_pair(pair_df: pd.DataFrame, pair_label: str) -> str:
    if "index_name" in pair_df.columns:
        values = pair_df["index_name"].fillna("").astype(str).map(lambda token: token.strip())
        for token in values:
            if token:
                return token
    return pair_label


def _plot_one_pair(
    pair_df: pd.DataFrame,
    *,
    pair_label: str,
    index_name: str,
    region_order: Sequence[str],
    settings: FixationPreferenceIndexHeatmapPlotSettings,
    out_root: Path,
    ext: str,
    figsize: list[float],
    dpi: Optional[int],
    unit_order_by_region: Optional[dict[str, list[str]]] = None,
) -> dict:
    fig, axes = plt.subplots(1, len(region_order), figsize=figsize, dpi=dpi, squeeze=False)
    axes_row = axes[0]
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=float(settings.left_margin),
        right=float(settings.right_margin),
        top=float(settings.top_margin),
        bottom=float(settings.bottom_margin),
        wspace=float(settings.panel_wspace),
    )

    abs_max = float(np.nanmax(pair_df["plot_preference_index"].to_numpy(dtype=float)))
    if not np.isfinite(abs_max) or abs_max <= 0:
        abs_max = 1.0
    norm = Normalize(vmin=0.0, vmax=abs_max)

    im_ref = None
    n_units_by_region: dict[str, int] = {}
    for ax, region in zip(axes_row, region_order):
        region_df = pair_df.loc[pair_df["region_norm"] == _normalize_region(region)].copy()
        forced_unit_order = None
        if unit_order_by_region is not None:
            forced_unit_order = unit_order_by_region.get(_normalize_region(region))
        matrix, bin_centers, unit_order = _region_matrix(
            region_df,
            value_column="plot_preference_index",
            forced_unit_order=forced_unit_order,
        )
        n_units = int(len(unit_order))
        n_units_by_region[str(region)] = n_units

        if n_units <= 0 or matrix.size == 0 or bin_centers.size == 0:
            ax.set_facecolor("white")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.5,
                0.5,
                "No selective units",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.4,
                color="#666666",
            )
            ax.set_title(f"{region}\n(n=0)", fontsize=7.5, pad=2.0)
            continue

        if bin_centers.size > 1:
            half = 0.5 * float(np.mean(np.diff(bin_centers)))
        else:
            half = 0.005
        x0 = float(bin_centers[0]) - half
        x1 = float(bin_centers[-1]) + half
        im_ref = ax.imshow(
            matrix,
            cmap=_W_R_CMAP,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=[x0, x1, 0.0, float(matrix.shape[0])],
        )
        ax.axvline(0.0, color="#303030", linestyle="--", linewidth=0.55, alpha=0.95)
        ax.set_title(f"{region}\n(n={n_units})", fontsize=7.5, pad=2.0)
        ax.tick_params(axis="x", labelsize=6.0, length=2.0, pad=1.0)
        y_pos, y_labels = _sorted_unit_tick_positions_labels(n_units)
        ax.set_yticks(y_pos.tolist())
        ax.set_yticklabels(y_labels, fontsize=5.8)
        ax.tick_params(axis="y", labelsize=5.8, length=2.0, pad=1.0)
        ax.grid(False)

    for idx, ax in enumerate(axes_row):
        ax.set_xlabel("Time (s)", fontsize=6.8)
        if idx == 0:
            ax.set_ylabel("Sorted Unit Number", fontsize=6.8)
        else:
            ax.set_ylabel("Sorted Unit Number", fontsize=6.2)

    if im_ref is not None:
        orientation = str(settings.colorbar_orientation).strip().lower()
        if orientation not in {"horizontal", "vertical"}:
            orientation = "horizontal"
        cbar = fig.colorbar(
            im_ref,
            ax=axes_row.tolist(),
            orientation=orientation,
            fraction=float(settings.colorbar_fraction),
            pad=float(settings.colorbar_pad),
            shrink=float(settings.colorbar_shrink),
            aspect=max(1, int(round(float(settings.colorbar_aspect)))),
        )
        cbar.set_label(settings.colorbar_label, fontsize=6.8)
        if orientation == "horizontal":
            cbar.ax.tick_params(axis="x", labelsize=6.0, length=2.0, pad=1.0)
        else:
            cbar.ax.tick_params(axis="y", labelsize=6.0, length=2.0, pad=1.0)

    if settings.show_suptitle:
        fig.suptitle(
            f"{index_name} ({pair_label})",
            fontsize=7.6,
            y=min(0.995, float(settings.top_margin) + 0.07),
        )

    stem = Path(str(settings.output_filename).strip()).stem or "preference_index_heatmaps"
    pair_token = _safe_token(index_name if index_name else pair_label)
    out_path = out_root / f"{stem}__{pair_token}.{ext}"
    save_figure(
        fig,
        out_path,
        ext=ext,
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)
    return {
        "pair_label": pair_label,
        "index_name": index_name,
        "output_path": out_path,
        "n_units_total": int(pair_df["unit_key"].nunique()),
        "n_units_by_region": n_units_by_region,
    }


def plot_fixation_preference_index_heatmaps(
    settings: FixationPreferenceIndexHeatmapPlotSettings,
    *,
    pair_labels: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Plot one 1xN-region heatmap figure per fixation-pair preference index."""
    unit_filter_mode = _resolve_unit_filter_mode(settings)
    normalization_mode = _normalize_normalization_mode(settings.normalization_mode)
    settings.normalization_mode = normalization_mode
    df = _load_timeseries_df(settings)
    if df.empty:
        print("[plot] no preference-index rows found for heatmap plotting")
        return None
    if unit_filter_mode == _UNIT_FILTER_ANY:
        df = df.loc[_selective_any_unit_mask(df)].copy()
        if df.empty:
            print("[plot] no rows remain after any-selective unit filter")
            return None
    value_column = _resolve_value_column(df, normalization_mode)

    pair_order = _resolve_pair_order(
        df,
        requested_pairs=pair_labels,
        default_pair_order=settings.default_pair_order,
    )
    if not pair_order:
        print("[plot] no pair labels available after filters")
        return None

    region_order = _resolve_region_order(
        df,
        requested_regions=regions,
        default_region_order=settings.region_order,
    )
    if not region_order:
        print("[plot] no regions configured for heatmap plotting")
        return None

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    ext = normalize_extension(settings.output_extension, fallback="pdf")
    figsize, dpi = _resolve_figsize_and_dpi(settings)
    reference_unit_order_by_region = None
    sort_reference_pair = (
        str(settings.sort_reference_pair_label).strip()
        if settings.sort_reference_pair_label is not None
        else ""
    )
    if sort_reference_pair:
        ref_df = df.loc[df["pair_label"].astype(str) == sort_reference_pair].copy()
        if unit_filter_mode == _UNIT_FILTER_PAIR:
            ref_df = ref_df.loc[ref_df["is_selective_pair"].map(_coerce_bool)].copy()
        if ref_df.empty:
            print(
                "[plot] requested sort_reference_pair_label has no rows after filters; "
                f"pair={sort_reference_pair}"
            )
        else:
            ref_df["plot_preference_index"] = np.abs(
                pd.to_numeric(ref_df[value_column], errors="coerce").to_numpy(dtype=float)
            )
            reference_unit_order_by_region = {}
            for region in region_order:
                region_norm = _normalize_region(region)
                region_ref_df = ref_df.loc[ref_df["region_norm"] == region_norm].copy()
                _matrix, _centers, unit_order = _region_matrix(
                    region_ref_df,
                    value_column="plot_preference_index",
                )
                if unit_order:
                    reference_unit_order_by_region[region_norm] = unit_order
            if not reference_unit_order_by_region:
                reference_unit_order_by_region = None

    pair_plot_data: dict[str, pd.DataFrame] = {}
    pair_index_name: dict[str, str] = {}
    n_units_by_pair_region: dict[str, dict[str, int]] = {}
    for pair_label in pair_order:
        pair_df = df.loc[df["pair_label"].astype(str) == str(pair_label)].copy()
        if unit_filter_mode == _UNIT_FILTER_PAIR:
            pair_df = pair_df.loc[pair_df["is_selective_pair"].map(_coerce_bool)].copy()
        if pair_df.empty:
            continue
        pair_df["plot_preference_index"] = np.abs(
            pd.to_numeric(pair_df[value_column], errors="coerce").to_numpy(dtype=float)
        )
        pair_plot_data[str(pair_label)] = pair_df
        pair_index_name[str(pair_label)] = _index_name_for_pair(pair_df, pair_label)

    pair_order = [pair for pair in pair_order if pair in pair_plot_data]
    if not pair_order:
        print("[plot] no pair-specific heatmap figures were generated")
        return None

    outputs: list[dict] = []
    if settings.combine_pairs_into_single_figure:
        fig, axes = plt.subplots(
            len(pair_order),
            len(region_order),
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
        )
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(
            left=float(settings.left_margin),
            right=float(settings.right_margin),
            top=float(settings.top_margin),
            bottom=float(settings.bottom_margin),
            wspace=float(settings.panel_wspace),
            hspace=float(settings.panel_hspace),
        )

        all_values: list[np.ndarray] = []
        for pair_label in pair_order:
            vals = pair_plot_data[pair_label]["plot_preference_index"].to_numpy(dtype=float)
            all_values.append(vals)
        if all_values:
            cat_vals = np.concatenate(all_values, axis=0)
            finite_vals = cat_vals[np.isfinite(cat_vals)]
            abs_max = float(np.max(finite_vals)) if finite_vals.size > 0 else np.nan
        else:
            abs_max = np.nan
        if not np.isfinite(abs_max) or abs_max <= 0:
            abs_max = 1.0
        norm = Normalize(vmin=0.0, vmax=abs_max)

        im_ref = None
        for row_idx, pair_label in enumerate(pair_order):
            pair_df = pair_plot_data[pair_label]
            n_units_by_region: dict[str, int] = {}
            for col_idx, region in enumerate(region_order):
                ax = axes[row_idx, col_idx]
                region_norm = _normalize_region(region)
                region_df = pair_df.loc[pair_df["region_norm"] == region_norm].copy()
                forced_unit_order = None
                if reference_unit_order_by_region is not None:
                    forced_unit_order = reference_unit_order_by_region.get(region_norm)
                matrix, bin_centers, unit_order = _region_matrix(
                    region_df,
                    value_column="plot_preference_index",
                    forced_unit_order=forced_unit_order,
                )
                n_units = int(len(unit_order))
                n_units_by_region[str(region)] = n_units
                if row_idx == 0:
                    ax.set_title(str(region), fontsize=7.2, pad=2.0)

                if n_units <= 0 or matrix.size == 0 or bin_centers.size == 0:
                    ax.set_facecolor("white")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(
                        0.5,
                        0.5,
                        "No units",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        color="#666666",
                    )
                    continue

                if bin_centers.size > 1:
                    half = 0.5 * float(np.mean(np.diff(bin_centers)))
                else:
                    half = 0.005
                x0 = float(bin_centers[0]) - half
                x1 = float(bin_centers[-1]) + half
                im_ref = ax.imshow(
                    matrix,
                    cmap=_W_R_CMAP,
                    norm=norm,
                    aspect="auto",
                    interpolation="nearest",
                    origin="lower",
                    extent=[x0, x1, 0.0, float(matrix.shape[0])],
                )
                ax.axvline(0.0, color="#303030", linestyle="--", linewidth=0.5, alpha=0.95)
                y_pos, y_labels = _sorted_unit_tick_positions_labels(n_units)
                ax.set_yticks(y_pos.tolist())
                ax.set_yticklabels(y_labels, fontsize=5.4)
                ax.tick_params(axis="y", labelsize=5.4, length=1.8, pad=0.8)
                ax.tick_params(axis="x", labelsize=5.4, length=1.8, pad=0.8)
                if row_idx < len(pair_order) - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                ax.grid(False)

            n_units_by_pair_region[str(pair_label)] = n_units_by_region
            left_ax = axes[row_idx, 0]
            left_ax.set_ylabel("Sorted Unit Number", fontsize=6.2)
            left_ax.text(
                0.0,
                1.03,
                str(pair_index_name.get(str(pair_label), pair_label)),
                transform=left_ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.0,
                color="#1a1a1a",
            )
            for col_idx in range(1, len(region_order)):
                axes[row_idx, col_idx].set_ylabel("")

        for col_idx in range(len(region_order)):
            axes[-1, col_idx].set_xlabel("Time (s)", fontsize=6.2)

        if im_ref is not None:
            orientation = str(settings.colorbar_orientation).strip().lower()
            if orientation not in {"horizontal", "vertical"}:
                orientation = "horizontal"
            cbar = fig.colorbar(
                im_ref,
                ax=axes.ravel().tolist(),
                orientation=orientation,
                fraction=float(settings.colorbar_fraction),
                pad=float(settings.colorbar_pad),
                shrink=float(settings.colorbar_shrink),
                aspect=max(1, int(round(float(settings.colorbar_aspect)))),
            )
            cbar.set_label(settings.colorbar_label, fontsize=6.4)
            if orientation == "horizontal":
                cbar.ax.tick_params(axis="x", labelsize=5.6, length=1.8, pad=0.8)
            else:
                cbar.ax.tick_params(axis="y", labelsize=5.6, length=1.8, pad=0.8)

        if settings.show_suptitle:
            fig.suptitle(
                "Fixation Preference Index Heatmaps",
                fontsize=7.4,
                y=min(0.995, float(settings.top_margin) + 0.06),
            )

        stem = Path(str(settings.output_filename).strip()).stem or "preference_index_heatmaps"
        out_path = out_root / f"{stem}.{ext}"
        save_figure(
            fig,
            out_path,
            ext=ext,
            dpi=dpi,
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )
        plt.close(fig)
        outputs.append(
            {
                "output_path": out_path,
                "pair_labels": [str(pair) for pair in pair_order],
                "index_names": [str(pair_index_name.get(str(pair), pair)) for pair in pair_order],
                "n_units_by_pair_region": n_units_by_pair_region,
                "combined_pairs": True,
            }
        )
    else:
        for pair_label in pair_order:
            pair_df = pair_plot_data[str(pair_label)]
            out = _plot_one_pair(
                pair_df,
                pair_label=str(pair_label),
                index_name=str(pair_index_name[str(pair_label)]),
                region_order=region_order,
                settings=settings,
                out_root=out_root,
                ext=ext,
                figsize=figsize,
                dpi=dpi,
                unit_order_by_region=reference_unit_order_by_region,
            )
            outputs.append(out)

    if not outputs:
        print("[plot] no pair-specific heatmap figures were generated")
        return None

    return {
        "outputs": outputs,
        "pair_order": [str(pair) for pair in pair_order],
        "region_order": list(region_order),
        "combine_pairs_into_single_figure": bool(settings.combine_pairs_into_single_figure),
        "unit_filter_mode": unit_filter_mode,
        "sort_reference_pair_label": sort_reference_pair or None,
        "normalization_mode": normalization_mode,
        "value_column": value_column,
    }
