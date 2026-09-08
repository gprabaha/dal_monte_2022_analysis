"""Tests for the pair-correlation chapter figures."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.plotting.fixation_pair_correlation_overview import (
    _decade,
    _decade_suffix,
    _rescale_to_decade,
)


class TestDecadeScaling(unittest.TestCase):
    """Bar panels cannot share a y-axis, so the axis label carries the decade."""

    def test_decade_puts_the_largest_value_in_one_to_ten(self):
        for value, expected in ((0.075, -2), (0.0049, -3), (0.00073, -4), (7.2, 0)):
            exponent = _decade([value])
            self.assertEqual(exponent, expected)
            self.assertTrue(1.0 <= abs(value) / 10.0 ** exponent < 10.0)

    def test_empty_and_all_zero_input_does_not_rescale(self):
        self.assertEqual(_decade([]), 0)
        self.assertEqual(_decade([0.0, 0.0]), 0)
        self.assertEqual(_decade([np.nan, np.inf]), 0)

    def test_suffix_is_a_readable_power_and_empty_at_zero(self):
        self.assertEqual(_decade_suffix(-3), ", ×10⁻³")
        self.assertEqual(_decade_suffix(-4), ", ×10⁻⁴")
        self.assertEqual(_decade_suffix(0), "")

    def test_rescaling_preserves_the_ratio_between_bars(self):
        """Scaling must not change what the figure says, only how it reads."""
        rows = pd.DataFrame({"mean": [0.004, 0.001, 0.002], "sem": [0.0002, 0.0001, 0.0001]})
        scaled, exponent = _rescale_to_decade(rows)
        self.assertEqual(exponent, -3)
        np.testing.assert_allclose(
            scaled["mean"] / scaled["mean"].iloc[0], rows["mean"] / rows["mean"].iloc[0]
        )
        np.testing.assert_allclose(scaled["mean"].to_numpy(), [4.0, 1.0, 2.0])
        # The error bars ride the same scale, or they would misstate the spread.
        np.testing.assert_allclose(scaled["sem"].to_numpy(), [0.2, 0.1, 0.1])

    def test_within_and_cross_region_land_on_different_decades(self):
        """The whole point: a ten-fold difference must show up in the label."""
        within = pd.DataFrame({"mean": [0.0049, 0.0011], "sem": [0.0002, 0.0001]})
        cross = pd.DataFrame({"mean": [0.00073, 0.00015], "sem": [0.00007, 0.00004]})
        self.assertNotEqual(_rescale_to_decade(within)[1], _rescale_to_decade(cross)[1])


if __name__ == "__main__":
    unittest.main()
