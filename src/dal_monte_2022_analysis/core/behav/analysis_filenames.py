"""Behavior-analysis filename resolvers."""

from __future__ import annotations

from typing import Optional

from dal_monte_2022_analysis.utils.filenames import resolve_filename_override


_CROSS_CORRELATION_SCOPE_ALIASES = {
    "whole": "whole",
    "whole_session": "whole",
    "all": "whole",
    "session": "whole",
    "interactive": "interactive",
    "interactive_only": "interactive",
    "non_interactive": "non_interactive",
    "non_interactive_only": "non_interactive",
    "noninteractive": "non_interactive",
}


def normalize_fix_cross_correlation_time_scope(scope: Optional[str]) -> str:
    """Normalize cross-correlation time-scope labels to canonical values."""
    raw = "whole" if scope is None else str(scope).strip().lower()
    token = raw.replace("-", "_").replace(" ", "_")
    normalized = _CROSS_CORRELATION_SCOPE_ALIASES.get(token)
    if normalized is None:
        allowed = ", ".join(sorted(set(_CROSS_CORRELATION_SCOPE_ALIASES.values())))
        raise ValueError(f"Unsupported cross-correlation time scope '{scope}'. Allowed: {allowed}.")
    return normalized


def build_fix_cross_correlation_output_filename(
    fixation_label: str,
    output_kind: str,
    *,
    time_scope: Optional[str] = "whole",
) -> str:
    """Build canonical cross-correlation output filenames."""
    scope = normalize_fix_cross_correlation_time_scope(time_scope)
    label = str(fixation_label).strip().lower().replace(" ", "_")
    kind = str(output_kind).strip().lower()

    if kind == "within":
        stem = f"within_session_{label}_fix_cross_correlation"
    elif kind == "cross":
        stem = f"cross_session_{label}_fix_cross_correlation"
    elif kind == "shuffle":
        stem = f"within_session_{label}_fix_cross_correlation_shuffle"
    elif kind == "lags":
        stem = f"{label}_crosscorrelation_lags"
    else:
        raise ValueError(
            "Unsupported cross-correlation output kind "
            f"'{output_kind}'. Expected one of: within, cross, shuffle, lags."
        )

    return f"{stem}__phase={scope}.pkl"


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
