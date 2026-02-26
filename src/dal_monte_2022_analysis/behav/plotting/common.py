"""Shared plotting utilities and style helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib as mpl
import numpy as np

_SAVEFIG_PATCH_INSTALLED = False
_DISABLE_PDF_CLIPPING_MASKS = True


def _unclip_artist(artist) -> None:
    """Disable clipping for one matplotlib artist when supported."""
    if artist is None:
        return
    try:
        artist.set_clip_on(False)
    except Exception:
        pass
    try:
        artist.set_clip_path(None)
    except Exception:
        pass
    try:
        artist.set_clip_box(None)
    except Exception:
        pass


def _strip_figure_clipping(fig) -> None:
    """Remove clip paths from all artists in a figure before vector export."""
    for artist in fig.findobj():
        _unclip_artist(artist)


def _is_pdf_export(args: tuple, kwargs: dict) -> bool:
    """Infer whether savefig target format is PDF."""
    fmt = kwargs.get("format")
    if fmt is not None:
        return str(fmt).lower() == "pdf"
    if args:
        target = args[0]
        suffix = ""
        try:
            suffix = Path(target).suffix.lower()
        except TypeError:
            suffix = ""
        return suffix == ".pdf"
    return False


def _ensure_savefig_patch() -> None:
    """Patch Figure.savefig once so PDF exports can disable clipping masks globally."""
    global _SAVEFIG_PATCH_INSTALLED
    if _SAVEFIG_PATCH_INSTALLED:
        return

    figure_cls = mpl.figure.Figure
    original_savefig = figure_cls.savefig

    def _patched_savefig(self, *args, **kwargs):
        if _DISABLE_PDF_CLIPPING_MASKS and _is_pdf_export(args, kwargs):
            _strip_figure_clipping(self)
        return original_savefig(self, *args, **kwargs)

    figure_cls.savefig = _patched_savefig
    _SAVEFIG_PATCH_INSTALLED = True


def apply_plotting_config(cfg: dict) -> dict:
    """Apply rcParams from a plotting config dict."""
    global _DISABLE_PDF_CLIPPING_MASKS
    _ensure_savefig_patch()

    rc_params = cfg.get("rc_params", {}) if cfg else {}
    if rc_params:
        mpl.rcParams.update(rc_params)
    export_cfg = cfg.get("export", {}) if cfg else {}
    _DISABLE_PDF_CLIPPING_MASKS = bool(
        export_cfg.get("disable_pdf_clipping_masks", True)
    )
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
