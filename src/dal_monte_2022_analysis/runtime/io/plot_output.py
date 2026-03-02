"""Plot-output IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def normalize_extension(ext: str, *, fallback: str) -> str:
    """Normalize an extension token to lowercase without a leading dot."""
    text = str(ext).strip().lower().lstrip(".")
    return text if text else str(fallback).strip().lower().lstrip(".")


def save_figure(
    fig,
    out_path: str | Path,
    *,
    ext: Optional[str] = None,
    dpi: Optional[int] = None,
    facecolor: str = "white",
    edgecolor: str = "white",
    transparent: bool = False,
    pdf_compression: Optional[int] = None,
) -> Path:
    """Persist a matplotlib figure, creating parent directories as needed."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = normalize_extension(ext if ext is not None else path.suffix, fallback="png")

    save_kwargs = {
        "format": fmt,
        "facecolor": facecolor,
        "edgecolor": edgecolor,
        "transparent": bool(transparent),
    }
    if dpi is not None:
        save_kwargs["dpi"] = dpi

    if fmt == "pdf" and pdf_compression is not None:
        import matplotlib as mpl

        with mpl.rc_context({"pdf.compression": int(pdf_compression)}):
            fig.savefig(path, **save_kwargs)
    else:
        fig.savefig(path, **save_kwargs)
    return path
