"""Shared generic filename resolver helpers."""

from __future__ import annotations

from typing import Optional

def ensure_filename(name: str, suffix: str) -> str:
    """Return a non-empty filename with a required suffix."""
    text = str(name).strip()
    if not text:
        raise ValueError("Output filename cannot be empty.")
    return text if text.endswith(suffix) else f"{text}{suffix}"


def resolve_filename_override(override: Optional[str], default: str) -> str:
    """Return override when provided; otherwise default."""
    if override is None:
        return str(default)
    token = str(override).strip()
    return token if token else str(default)
