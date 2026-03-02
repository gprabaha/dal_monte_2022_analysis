"""Regression tests for fixation detection core logic."""

from __future__ import annotations

import unittest

import numpy as np

from dal_monte_2022_analysis.core.behav.fixation_detection import (
    detect_fixations_and_saccades,
)


class TestFixationDetection(unittest.TestCase):
    """Regression checks for public fixation-detection behavior."""

    def test_short_input_returns_empty_arrays(self) -> None:
        positions = np.zeros((100, 2), dtype=float)
        fix, sacc = detect_fixations_and_saccades(positions)
        self.assertEqual(fix.shape, (0, 2))
        self.assertEqual(sacc.shape, (0, 2))

    def test_long_input_returns_well_formed_intervals(self) -> None:
        rng = np.random.default_rng(13)
        steps = rng.normal(loc=0.0, scale=0.4, size=(900, 2))
        positions = np.cumsum(steps, axis=0)

        fix, sacc = detect_fixations_and_saccades(positions)

        self.assertEqual(fix.ndim, 2)
        self.assertEqual(sacc.ndim, 2)
        self.assertEqual(fix.shape[1], 2)
        self.assertEqual(sacc.shape[1], 2)
        self.assertTrue(np.issubdtype(fix.dtype, np.integer))
        self.assertTrue(np.issubdtype(sacc.dtype, np.integer))
        if fix.size:
            self.assertTrue(np.all(fix[:, 0] <= fix[:, 1]))
        if sacc.size:
            self.assertTrue(np.all(sacc[:, 0] <= sacc[:, 1]))


if __name__ == "__main__":
    unittest.main()

