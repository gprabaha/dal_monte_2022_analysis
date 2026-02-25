"""Shared plotting utilities and style helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib as mpl
import numpy as np


def apply_plotting_config(cfg: dict) -> dict:
    """Apply rcParams from a plotting config dict."""
    rc_params = cfg.get("rc_params", {}) if cfg else {}
    if rc_params:
        mpl.rcParams.update(rc_params)
    return cfg


def resolve_figsize(cfg: dict) -> tuple[Optional[list[float]], Optional[int]]:
    """Return figure size and dpi from plotting config."""
    figure_cfg = cfg.get("figure", {}) if cfg else {}
    figsize = figure_cfg.get("figsize")
    dpi = figure_cfg.get("dpi")
    return figsize, dpi


def plot_points_individual(ax, x: Iterable[float], y: Iterable[float], **kwargs) -> None:
    """Plot points as individual Line2D objects for Illustrator editing."""
    for xi, yi in zip(x, y):
        ax.plot([xi], [yi], marker="o", linestyle="none", **kwargs)


def format_p_value(p: float) -> str:
    """Format a p-value for compact plot annotations."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"
