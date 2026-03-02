"""Statistical helpers shared across analyses and plotting."""

from .hypothesis import (
    paired_ttest_per_lag,
    two_sample_pvalues,
)

__all__ = [
    "paired_ttest_per_lag",
    "two_sample_pvalues",
]
