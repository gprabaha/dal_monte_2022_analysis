"""Tests for the spatial-decay analysis.

The load in this module goes through pair-coordination outputs on disk, so the
tests here exercise the pure pieces: the separation parsing, the decay fit, and
the two hypotheses the analysis is built to distinguish.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_pair_spatial_decay import (
    SEPARATION_CENTRES,
    SEPARATION_LABELS,
    exponential_decay,
    fit_decay,
    fit_decay_by_group,
    parse_channel_number,
    test_decay_flatness as run_decay_flatness_test,
)


def _profile(amplitude: float, length_constant: float, offset: float, noise: float = 0.0,
             seed: int = 0, region: str = "bla") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    separation = np.asarray(SEPARATION_CENTRES, dtype=float)
    values = exponential_decay(separation, amplitude, length_constant, offset)
    values = values + rng.normal(0.0, noise, values.shape)
    return pd.DataFrame(
        {
            "region": region,
            "separation_bin": list(SEPARATION_LABELS),
            "separation": separation,
            "mean": values,
            "sem": np.full(separation.shape, max(noise, 1e-6)),
            "n_pairs": 1000,
        }
    )


class TestSeparationParsing(unittest.TestCase):
    def test_channel_numbers_are_extracted(self):
        parsed = parse_channel_number(pd.Series(["SPK33", "SPK7", "SPK105"]))
        self.assertEqual(list(parsed), [33.0, 7.0, 105.0])

    def test_unparseable_labels_become_nan(self):
        parsed = parse_channel_number(pd.Series(["SPK", "", None]))
        self.assertTrue(parsed.isna().all())


class TestDecayFit(unittest.TestCase):
    def test_recovers_planted_parameters(self):
        fit = fit_decay(_profile(0.20, 3.0, 0.004), n_bootstrap=50)
        self.assertAlmostEqual(fit["amplitude"], 0.20, places=3)
        self.assertAlmostEqual(fit["length_constant"], 3.0, places=2)
        self.assertAlmostEqual(fit["offset"], 0.004, places=4)
        self.assertGreater(fit["r_squared"], 0.999)

    def test_survives_noise(self):
        fit = fit_decay(_profile(0.20, 3.0, 0.004, noise=0.004, seed=1), n_bootstrap=50)
        self.assertLess(abs(fit["length_constant"] - 3.0), 1.5)
        self.assertGreater(fit["r_squared"], 0.9)

    def test_too_few_bins_returns_nan_not_a_number(self):
        """An unidentifiable fit must not return something that looks measured."""
        fit = fit_decay(_profile(0.2, 3.0, 0.0).head(3), n_bootstrap=10)
        self.assertTrue(np.isnan(fit["length_constant"]))
        self.assertTrue(np.isnan(fit["amplitude"]))

    def test_bootstrap_interval_brackets_the_estimate(self):
        fit = fit_decay(_profile(0.20, 3.0, 0.004, noise=0.002, seed=2), n_bootstrap=100)
        self.assertLessEqual(fit["length_constant_low"], fit["length_constant"])
        self.assertGreaterEqual(fit["length_constant_high"], fit["length_constant"])

    def test_fit_by_group_returns_one_row_per_region(self):
        table = pd.concat(
            [_profile(0.2, 3.0, 0.004, region="bla"), _profile(0.05, 2.0, 0.001, region="ofc")]
        )
        fits = fit_decay_by_group(table, n_bootstrap=25)
        self.assertEqual(set(fits["region"]), {"bla", "ofc"})
        self.assertEqual(len(fits), 2)


class TestHypothesisDiscrimination(unittest.TestCase):
    """The analysis exists to tell these two apart; the test says it can."""

    def test_a_flat_profile_is_not_called_a_decay(self):
        """What a shared reference would produce."""
        flat = _profile(0.0, 5.0, 0.02, noise=0.0005, seed=3)
        result = run_decay_flatness_test(flat)
        self.assertGreater(float(result["p_value"].iloc[0]), 0.05)

    def test_a_decaying_profile_is_detected(self):
        """What local circuitry would produce."""
        decaying = _profile(0.20, 3.0, 0.004, noise=0.0005, seed=4)
        result = run_decay_flatness_test(decaying)
        self.assertLess(float(result["spearman_rho"].iloc[0]), -0.8)
        self.assertLess(float(result["p_value"].iloc[0]), 0.05)
        self.assertGreater(float(result["fold_decay"].iloc[0]), 5.0)


if __name__ == "__main__":
    unittest.main()
