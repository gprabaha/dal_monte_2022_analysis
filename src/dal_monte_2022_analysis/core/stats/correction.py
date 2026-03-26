"""P-value correction helpers shared across analyses and plotting."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


ALLOWED_PVALUE_CORRECTIONS: frozenset[str] = frozenset(
    {"none", "bonferroni", "holm", "fdr_bh"}
)
_PVALUE_CORRECTION_ALIASES = {
    "fdr": "fdr_bh",
    "bh": "fdr_bh",
    "benjamini_hochberg": "fdr_bh",
    "benjamini-hochberg": "fdr_bh",
}


def normalize_pvalue_correction(
    method: object,
    *,
    allowed: Optional[Sequence[str]] = None,
) -> str:
    """Normalize p-value correction names to canonical tokens."""
    token = str(method).strip().lower()
    resolved = _PVALUE_CORRECTION_ALIASES.get(token, token)
    allowed_tokens = (
        ALLOWED_PVALUE_CORRECTIONS
        if allowed is None
        else frozenset(str(item).strip().lower() for item in allowed)
    )
    if resolved not in allowed_tokens:
        raise ValueError(
            f"Unsupported p-value correction '{method}'. "
            f"Expected one of: {sorted(allowed_tokens)}"
        )
    return resolved


def adjust_pvalues(
    p_values: Sequence[float] | np.ndarray,
    method: object,
) -> np.ndarray:
    """Adjust p-values using a supported multiple-comparison method."""
    resolved = normalize_pvalue_correction(method)
    vec = np.asarray(p_values, dtype=float).reshape(-1)
    out = np.full(vec.shape, np.nan, dtype=float)
    finite = np.isfinite(vec)
    if not np.any(finite):
        return out.reshape(np.asarray(p_values).shape)

    vals = vec[finite]
    m = int(vals.size)
    if resolved == "none":
        out[finite] = vals
        return out.reshape(np.asarray(p_values).shape)
    if resolved == "bonferroni":
        out[finite] = np.minimum(vals * float(m), 1.0)
        return out.reshape(np.asarray(p_values).shape)

    order = np.argsort(vals)
    ranked = vals[order]
    if resolved == "holm":
        holm_ranked = (m - np.arange(m, dtype=float)) * ranked
        holm_ranked = np.maximum.accumulate(holm_ranked)
        holm_ranked = np.clip(holm_ranked, 0.0, 1.0)
        adjusted = np.empty(m, dtype=float)
        adjusted[order] = holm_ranked
        out[finite] = adjusted
        return out.reshape(np.asarray(p_values).shape)

    bh_ranked = ranked * (float(m) / np.arange(1.0, float(m) + 1.0))
    bh_ranked = np.minimum.accumulate(bh_ranked[::-1])[::-1]
    bh_ranked = np.clip(bh_ranked, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = bh_ranked
    out[finite] = adjusted
    return out.reshape(np.asarray(p_values).shape)


def reject_nulls(
    p_values: Sequence[float] | np.ndarray,
    *,
    alpha: float,
    method: object,
) -> np.ndarray:
    """Return a boolean rejection mask after multiple-comparison correction."""
    adjusted = np.asarray(adjust_pvalues(p_values, method), dtype=float)
    return np.isfinite(adjusted) & (adjusted < float(alpha))


def apply_adjusted_pvalues(
    df: pd.DataFrame,
    *,
    p_col: str,
    out_col: str,
    method: object,
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Append a corrected p-value column, optionally grouped before correction."""
    out = df.copy()
    out[out_col] = np.nan
    if out.empty or p_col not in out.columns:
        return out
    resolved = normalize_pvalue_correction(method)
    if group_cols is None or len(group_cols) == 0:
        out[out_col] = adjust_pvalues(out[p_col].to_numpy(dtype=float), resolved)
        return out
    for _, idx in out.groupby(list(group_cols), dropna=False).groups.items():
        out.loc[idx, out_col] = adjust_pvalues(
            out.loc[idx, p_col].to_numpy(dtype=float),
            resolved,
        )
    return out
