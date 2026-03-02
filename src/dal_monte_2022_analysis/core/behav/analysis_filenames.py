"""Behavior-analysis filename resolvers."""

from __future__ import annotations

from typing import Optional

from dal_monte_2022_analysis.utils.filenames import resolve_filename_override
from dal_monte_2022_analysis.utils.paths import build_fix_cross_correlation_output_filename


def resolve_fix_cross_correlation_filename(
    *,
    fixation_label: str,
    output_kind: str,
    time_scope: Optional[str],
    override: Optional[str] = None,
) -> str:
    """Resolve fix-cross-correlation filename with optional override."""
    default = build_fix_cross_correlation_output_filename(
        fixation_label,
        output_kind,
        time_scope=time_scope,
    )
    return resolve_filename_override(override, default)
