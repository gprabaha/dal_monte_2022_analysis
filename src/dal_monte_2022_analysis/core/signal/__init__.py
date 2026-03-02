"""Signal-processing domain logic."""

from .cross_correlation import (
    assert_lag_axis_match,
    fft_cross_correlation,
    normalize_cross_correlation_energy,
    normalize_cross_correlation_sqrt_bin_count,
    summarize_cross_correlation,
)

__all__ = [
    "assert_lag_axis_match",
    "fft_cross_correlation",
    "normalize_cross_correlation_energy",
    "normalize_cross_correlation_sqrt_bin_count",
    "summarize_cross_correlation",
]

