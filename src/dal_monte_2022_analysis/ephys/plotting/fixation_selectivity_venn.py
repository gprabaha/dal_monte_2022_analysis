"""Plot region-level Venn diagrams for fixation selectivity pairs."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    from matplotlib_venn import venn3, venn3_circles
except Exception as exc:  # pragma: no cover - runtime dependency guard
    venn3 = None
    venn3_circles = None
    _MATPLOTLIB_VENN_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_VENN_IMPORT_ERROR = None

from dal_monte_2022_analysis.behav.plotting.common import (
    apply_plotting_config,
    resolve_figsize,
)
from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.utils.parallel import get_n_processes
from dal_monte_2022_analysis.utils.paths import build_analysis_output_dir


PAIR_FACE_INT_NONINT = "face_interactive__vs__face_non_interactive"
PAIR_FACE_INT_OBJ = "face_interactive__vs__object"
PAIR_FACE_NONINT_OBJ = "face_non_interactive__vs__object"


@dataclass
class FixationSelectivityVennPlotSettings:
    """Configuration for region-level fixation selectivity Venn plots."""

    cfg_path: str
    plotting_cfg_path: str = "configs/plotting.yaml"
    selectivity_input_subdir: str = "ephys/psth/fixation_psth_selectivity"
    pair_summary_filename: str = "pair_selectivity.csv"
    output_subdir: str = "ephys/psth/fixation_psth_selectivity_venn"
    output_extension: str = "pdf"
    output_dpi: Optional[int] = 220
    use_parallel: bool = True
    max_procs: int = 16
    test_single: bool = False
    min_units_per_region: int = 1


def _as_bool(val) -> bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val) != 0
    if isinstance(val, (float, np.floating)):
        return float(val) != 0.0
    if val is None:
        return False
    token = str(val).strip().lower()
    return token in {"1", "true", "t", "yes", "y"}


def _safe_region(region: Optional[str]) -> str:
    if region is None:
        return "unknown"
    text = str(region).strip()
    if not text:
        return "unknown"
    text = text.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    return text or "unknown"


def _pct(count: int, total: int) -> float:
    return 0.0 if total <= 0 else (100.0 * float(count) / float(total))


def _pair_key_from_row(row: pd.Series) -> Optional[str]:
    if {"condition_a", "condition_b"}.issubset(row.index):
        a = str(row.get("condition_a", "")).strip()
        b = str(row.get("condition_b", "")).strip()
        pair = "__vs__".join(sorted([a, b]))
        if "face_interactive" in pair and "face_non_interactive" in pair:
            return PAIR_FACE_INT_NONINT
        if "face_interactive" in pair and "object" in pair:
            return PAIR_FACE_INT_OBJ
        if "face_non_interactive" in pair and "object" in pair:
            return PAIR_FACE_NONINT_OBJ

    label = str(row.get("pair_label", "")).strip()
    if label in {PAIR_FACE_INT_NONINT, PAIR_FACE_INT_OBJ, PAIR_FACE_NONINT_OBJ}:
        return label
    return None


def _load_pair_summary_df(
    settings: FixationSelectivityVennPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    cfg = load_config(settings.cfg_path)
    in_path = (
        build_analysis_output_dir(cfg, settings.selectivity_input_subdir)
        / (settings.pair_summary_filename if str(settings.pair_summary_filename).endswith(".csv") else f"{settings.pair_summary_filename}.csv")
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Pair selectivity CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        return df
    if "region" not in df.columns:
        df["region"] = "unknown"
    else:
        df["region"] = df["region"].astype(str).replace({"": "unknown"}).fillna("unknown")
    if "unit_key" not in df.columns:
        if {"date", "unit_uuid"}.issubset(df.columns):
            df["unit_key"] = df["date"].astype(str) + "|" + df["unit_uuid"].astype(str)
        else:
            raise ValueError("pair_selectivity.csv must include 'unit_key' or ('date' and 'unit_uuid').")
    if "is_selective_pair" not in df.columns:
        raise ValueError("pair_selectivity.csv missing required column 'is_selective_pair'.")

    df["is_selective_pair"] = df["is_selective_pair"].map(_as_bool)
    if regions is not None:
        allowed = {str(r) for r in regions}
        df = df.loc[df["region"].astype(str).isin(allowed)].copy()
    return df


def _compute_region_summary(region: str, df_region: pd.DataFrame) -> Optional[dict]:
    if df_region.empty:
        return None

    total_units = int(df_region["unit_key"].astype(str).nunique())
    if total_units <= 0:
        return None

    selective_rows = df_region.loc[df_region["is_selective_pair"]].copy()
    selective_rows["pair_key"] = selective_rows.apply(_pair_key_from_row, axis=1)
    selective_rows = selective_rows.loc[selective_rows["pair_key"].notna()].copy()

    set_a = set(selective_rows.loc[selective_rows["pair_key"] == PAIR_FACE_INT_NONINT, "unit_key"].astype(str))
    set_b = set(selective_rows.loc[selective_rows["pair_key"] == PAIR_FACE_INT_OBJ, "unit_key"].astype(str))
    set_c = set(selective_rows.loc[selective_rows["pair_key"] == PAIR_FACE_NONINT_OBJ, "unit_key"].astype(str))

    a_only = set_a - set_b - set_c
    b_only = set_b - set_a - set_c
    c_only = set_c - set_a - set_b
    ab_only = (set_a & set_b) - set_c
    ac_only = (set_a & set_c) - set_b
    bc_only = (set_b & set_c) - set_a
    abc = set_a & set_b & set_c

    seg_counts = {
        "a_only": int(len(a_only)),
        "b_only": int(len(b_only)),
        "c_only": int(len(c_only)),
        "ab_only": int(len(ab_only)),
        "ac_only": int(len(ac_only)),
        "bc_only": int(len(bc_only)),
        "abc": int(len(abc)),
    }
    set_counts = {
        "face_int_vs_nonint": int(len(set_a)),
        "face_int_vs_obj": int(len(set_b)),
        "face_nonint_vs_obj": int(len(set_c)),
    }
    any_selective = int(len(set_a | set_b | set_c))

    return {
        "region": region,
        "total_units": total_units,
        "any_selective": any_selective,
        "set_counts": set_counts,
        "segment_counts": seg_counts,
    }


def _resolve_figsize_and_dpi(settings: FixationSelectivityVennPlotSettings) -> tuple[list[float], Optional[int]]:
    if settings.plotting_cfg_path and Path(settings.plotting_cfg_path).exists():
        plot_cfg = load_config(settings.plotting_cfg_path)
        apply_plotting_config(plot_cfg)
        figsize, cfg_dpi = resolve_figsize(plot_cfg)
    else:
        figsize, cfg_dpi = (None, None)
    if figsize is None:
        figsize = [11.5, 7.0]
    dpi = settings.output_dpi if settings.output_dpi is not None else cfg_dpi
    return [float(figsize[0]), float(figsize[1])], dpi


def _segment_label(count: int, total: int) -> str:
    return f"{count}\n{_pct(count, total):.1f}%"


def _subset_sizes_from_segments(seg: dict) -> dict[str, int]:
    return {
        "100": int(seg["a_only"]),
        "010": int(seg["b_only"]),
        "001": int(seg["c_only"]),
        "110": int(seg["ab_only"]),
        "101": int(seg["ac_only"]),
        "011": int(seg["bc_only"]),
        "111": int(seg["abc"]),
    }


def _format_set_label(title: str, count: int, total: int) -> str:
    return f"{title}\n{count} ({_pct(count, total):.1f}%)"


def _render_region_venn(summary: dict, settings: FixationSelectivityVennPlotSettings) -> Path:
    if _MATPLOTLIB_VENN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "matplotlib-venn is required for advanced area-scaled Venn plots. "
            f"Import error: {_MATPLOTLIB_VENN_IMPORT_ERROR}"
        )

    cfg = load_config(settings.cfg_path)
    out_root = build_analysis_output_dir(cfg, settings.output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    region = str(summary["region"])
    total_units = int(summary["total_units"])
    set_counts = summary["set_counts"]
    seg = summary["segment_counts"]
    any_selective = int(summary["any_selective"])

    figsize, dpi = _resolve_figsize_and_dpi(settings)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    subset_sizes = _subset_sizes_from_segments(seg)
    venn = venn3(
        subsets=subset_sizes,
        set_labels=(
            _format_set_label("Int Face vs Non-Int Face", set_counts["face_int_vs_nonint"], total_units),
            _format_set_label("Int Face vs Object", set_counts["face_int_vs_obj"], total_units),
            _format_set_label("Non-Int Face vs Object", set_counts["face_nonint_vs_obj"], total_units),
        ),
        ax=ax,
    )
    venn3_circles(subsets=subset_sizes, ax=ax, linewidth=1.8, color="#303030")

    patch_colors = {
        "100": "#4C78A8",
        "010": "#F58518",
        "001": "#54A24B",
        "110": "#7F6BB0",
        "101": "#2D9FB0",
        "011": "#B36D4C",
        "111": "#666666",
    }
    for subset_id, color in patch_colors.items():
        patch = venn.get_patch_by_id(subset_id)
        if patch is None:
            continue
        patch.set_color(color)
        patch.set_alpha(0.36)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.0)

    missing_labels: list[str] = []
    for subset_id, count in subset_sizes.items():
        label = venn.get_label_by_id(subset_id)
        text = _segment_label(int(count), total_units)
        if label is None:
            missing_labels.append(f"{subset_id}: {text.replace(chr(10), ' ')}")
            continue
        label.set_text(text)
        label.set_fontsize(10)
        label.set_fontweight("bold")
        label.set_color("#111111")
        label.set_bbox(
            {
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "linewidth": 0.0,
            }
        )

    for idx, set_label in enumerate(venn.set_labels or []):
        if set_label is None:
            continue
        set_label.set_fontsize(9)
        set_label.set_fontweight("semibold")
        if idx == 0:
            set_label.set_position((set_label.get_position()[0] - 0.03, set_label.get_position()[1] - 0.01))
        elif idx == 1:
            set_label.set_position((set_label.get_position()[0] + 0.03, set_label.get_position()[1] - 0.01))
        else:
            set_label.set_position((set_label.get_position()[0], set_label.get_position()[1] + 0.02))

    detail_lines = [
        f"Total units: {total_units}",
        f"Any selective: {any_selective} ({_pct(any_selective, total_units):.1f}%)",
        "",
        "Exclusive segments:",
        f"A only: {seg['a_only']} ({_pct(seg['a_only'], total_units):.1f}%)",
        f"B only: {seg['b_only']} ({_pct(seg['b_only'], total_units):.1f}%)",
        f"C only: {seg['c_only']} ({_pct(seg['c_only'], total_units):.1f}%)",
        f"AB only: {seg['ab_only']} ({_pct(seg['ab_only'], total_units):.1f}%)",
        f"AC only: {seg['ac_only']} ({_pct(seg['ac_only'], total_units):.1f}%)",
        f"BC only: {seg['bc_only']} ({_pct(seg['bc_only'], total_units):.1f}%)",
        f"ABC: {seg['abc']} ({_pct(seg['abc'], total_units):.1f}%)",
    ]
    if missing_labels:
        detail_lines.extend(["", "Missing label placements:", *missing_labels])
    ax.text(
        1.03,
        0.98,
        "\n".join(detail_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.set_title(f"Region {region}: Fixation Selectivity Venn (Area-Scaled)")
    ax.axis("off")
    fig.subplots_adjust(left=0.05, right=0.76, top=0.9, bottom=0.06)

    ext = settings.output_extension if str(settings.output_extension).startswith(".") else f".{settings.output_extension}"
    out_path = out_root / f"region={_safe_region(region)}{ext}"
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.savefig(
        out_path,
        format=ext.lstrip("."),
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)
    return out_path


def _render_region_worker(args):
    summary, settings = args
    out_path = _render_region_venn(summary, settings)
    return {**summary, "output_path": out_path}


def build_fixation_selectivity_venn_summaries(
    settings: FixationSelectivityVennPlotSettings,
    *,
    regions: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Build one fixation selectivity Venn diagram per region."""
    pair_df = _load_pair_summary_df(settings, regions=regions)
    if pair_df.empty:
        print("[plot] no pair selectivity rows found for region Venn plotting")
        return []

    summaries: list[dict] = []
    for region, df_region in pair_df.groupby("region", sort=True, dropna=False):
        summary = _compute_region_summary(str(region), df_region.copy())
        if summary is None:
            continue
        if int(summary["total_units"]) < int(settings.min_units_per_region):
            continue
        summaries.append(summary)

    if settings.test_single and summaries:
        summaries = [summaries[0]]
    if not summaries:
        print("[plot] no regions passed filters for Venn plotting")
        return []

    outputs: list[dict] = []
    if settings.use_parallel and len(summaries) > 1:
        n_proc = get_n_processes(max_procs=settings.max_procs)
        with Pool(processes=n_proc) as pool:
            for result in tqdm(
                pool.imap_unordered(_render_region_worker, [(s, settings) for s in summaries]),
                total=len(summaries),
                desc=f"Plotting selectivity Venn ({n_proc} workers)",
                unit="region",
            ):
                outputs.append(result)
    else:
        for summary in tqdm(summaries, desc="Plotting selectivity Venn", unit="region"):
            outputs.append(_render_region_worker((summary, settings)))

    outputs.sort(key=lambda row: str(row["region"]))
    return outputs
