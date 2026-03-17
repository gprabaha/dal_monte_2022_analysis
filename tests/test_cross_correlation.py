"""Regression tests for cross-correlation helpers."""

from __future__ import annotations

import unittest

import numpy as np

from dal_monte_2022_analysis.core.signal.cross_correlation import fft_cross_correlation


class TestFftCrossCorrelation(unittest.TestCase):
    """Checks SciPy-backed lag ordering against simple known cases."""

    def test_fft_cross_correlation_matches_expected_lag_order(self) -> None:
        x = np.asarray([1.0, 0.0], dtype=float)
        y = np.asarray([0.0, 1.0], dtype=float)

        lags, corr = fft_cross_correlation(x, y)

        self.assertTrue(np.array_equal(lags, np.asarray([-1, 0, 1], dtype=np.int64)))
        self.assertTrue(np.allclose(corr, np.asarray([1.0, 0.0, 0.0], dtype=float)))

    def test_fft_cross_correlation_respects_max_lag(self) -> None:
        x = np.asarray([1.0, 2.0, 0.0], dtype=float)
        y = np.asarray([0.0, 1.0, -1.0], dtype=float)

        lags, corr = fft_cross_correlation(x, y, max_lag=1)

        self.assertTrue(np.array_equal(lags, np.asarray([-1, 0, 1], dtype=np.int64)))
        self.assertEqual(corr.shape, (3,))


if __name__ == "__main__":
    unittest.main()
