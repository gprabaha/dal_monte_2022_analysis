"""Tests for the signal-correlation analysis.

The correlation primitive and the trial-count control are the parts worth
pinning: the first because every number depends on it, the second because it is
what decides whether a condition difference is shared tuning or estimation
noise.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_signal_correlation import (
    CONDITION_ORDER,
    SignalCorrelationSettings,
    estimate_timeline_reliability,
    normalized_cross_correlation,
    stratify_by_trial_ratio,
)


class TestNormalizedCrossCorrelation(unittest.TestCase):
    def test_identical_traces_correlate_at_one_at_zero_lag(self):
        rng = np.random.default_rng(0)
        trace = rng.normal(size=100)
        out = normalized_cross_correlation(trace, trace, max_lag=10)
        self.assertAlmostEqual(float(out[10]), 1.0, places=9)

    def test_scale_and_offset_invariant(self):
        """It is a correlation coefficient, not a dot product."""
        rng = np.random.default_rng(1)
        trace = rng.normal(size=100)
        plain = normalized_cross_correlation(trace, trace, max_lag=5)
        scaled = normalized_cross_correlation(trace, 7.0 * trace + 3.0, max_lag=5)
        np.testing.assert_allclose(plain, scaled, atol=1e-9)

    def test_recovers_a_planted_lag_with_the_documented_sign(self):
        """Positive lag means the first trace follows the second."""
        rng = np.random.default_rng(2)
        base = np.convolve(rng.normal(size=140), np.ones(7) / 7.0, mode="same")
        shift = 6
        lags = np.arange(-20, 21)

        # base[shift:] is base advanced, so it *leads*: negative lag.
        leading = normalized_cross_correlation(base[shift:], base[:-shift], max_lag=20)
        self.assertEqual(int(lags[int(np.nanargmax(leading))]), -shift)

        # The reverse assignment puts the first trace behind: positive lag.
        following = normalized_cross_correlation(base[:-shift], base[shift:], max_lag=20)
        self.assertEqual(int(lags[int(np.nanargmax(following))]), shift)

    def test_anticorrelated_traces_give_minus_one(self):
        rng = np.random.default_rng(3)
        trace = rng.normal(size=80)
        out = normalized_cross_correlation(trace, -trace, max_lag=4)
        self.assertAlmostEqual(float(out[4]), -1.0, places=9)

    def test_stays_bounded(self):
        rng = np.random.default_rng(4)
        out = normalized_cross_correlation(
            rng.normal(size=100), rng.normal(size=100), max_lag=30
        )
        finite = out[np.isfinite(out)]
        self.assertTrue(np.all(np.abs(finite) <= 1.0 + 1e-9))


class TestReliabilityDiagnostic(unittest.TestCase):
    def test_noiseless_timeline_is_fully_reliable(self):
        timeline = np.sin(np.linspace(0, 4 * np.pi, 100))
        self.assertAlmostEqual(
            estimate_timeline_reliability(timeline, np.zeros(100)), 1.0, places=9
        )

    def test_reports_negative_rather_than_clipping(self):
        """Smoothed means with unsmoothed SEMs give a negative estimate.

        That is the finding the analysis has to work around, so it must be
        visible rather than silently floored at zero.
        """
        timeline = np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.1
        value = estimate_timeline_reliability(timeline, np.full(100, 1.0))
        self.assertLess(value, 0.0)

    def test_missing_sem_is_undefined(self):
        self.assertTrue(np.isnan(estimate_timeline_reliability(np.ones(10), None)))


def _pairs_with_ratio_effect(n: int = 800, seed: int = 0, tuning_effect: float = 0.0) -> pd.DataFrame:
    """Pairs where the condition difference is driven by trial-count ratio.

    ``tuning_effect`` adds a genuine condition difference that does not depend on
    the ratio, so the two situations can be told apart.
    """
    rng = np.random.default_rng(seed)
    ratio = rng.uniform(1.0, 12.0, n)
    rows = {
        "pair_key": [f"p{i}" for i in range(n)],
        "scope": "within_region",
        "region_pair": "bla",
        "object_n_trials_1": 100.0,
        "face_interactive_n_trials_1": 100.0 * ratio,
    }
    frame = pd.DataFrame(rows)
    # An artifact: interactive face gains with the ratio and nothing else.
    frame["face_interactive_zero_lag_excess"] = (
        0.02 + 0.01 * (ratio - 1.0) + tuning_effect + rng.normal(0, 0.01, n)
    )
    frame["face_non_interactive_zero_lag_excess"] = 0.02 + rng.normal(0, 0.01, n)
    frame["object_zero_lag_excess"] = 0.02 + rng.normal(0, 0.01, n)
    return frame


class TestTrialCountStratification(unittest.TestCase):
    """The control that decides the analysis."""

    def test_an_artifact_grows_with_the_ratio(self):
        settings = SignalCorrelationSettings(min_pairs_per_group=50)
        strata = stratify_by_trial_ratio(_pairs_with_ratio_effect(), settings)
        self.assertGreaterEqual(len(strata), 3)
        differences = strata["face_interactive_minus_object"].to_numpy(dtype=float)
        ratios = strata["median_trial_ratio"].to_numpy(dtype=float)
        self.assertTrue(np.all(np.diff(ratios) > 0))
        # Grows monotonically, and is near zero in the lowest stratum.
        self.assertGreater(differences[-1], differences[0])
        self.assertLess(abs(differences[0]), 0.05)

    def test_a_genuine_effect_survives_in_every_stratum(self):
        settings = SignalCorrelationSettings(min_pairs_per_group=50)
        planted = _pairs_with_ratio_effect(seed=1, tuning_effect=0.5)
        planted["face_interactive_zero_lag_excess"] -= 0.01 * (
            planted["face_interactive_n_trials_1"] / planted["object_n_trials_1"] - 1.0
        )
        strata = stratify_by_trial_ratio(planted, settings)
        differences = strata["face_interactive_minus_object"].to_numpy(dtype=float)
        self.assertTrue(np.all(differences > 0.4))

    def test_returns_the_conditions_it_was_given(self):
        settings = SignalCorrelationSettings(min_pairs_per_group=50)
        strata = stratify_by_trial_ratio(_pairs_with_ratio_effect(), settings)
        for condition in CONDITION_ORDER:
            self.assertIn(condition, strata.columns)


if __name__ == "__main__":
    unittest.main()
