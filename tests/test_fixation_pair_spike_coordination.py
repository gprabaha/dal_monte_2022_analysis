"""Tests for neural pair spike-coordination analysis.

The correctness-critical parts are the null construction (the fast frequency
domain path must equal brute force) and the count matching (a null built from
more terms than the observed statistic would inflate every z-score).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_pair_spike_coordination import (
    CONDITION_ORDER,
    IDENTITY_TOLERANCE,
    PairSpikeCoordinationSettings,
    _cross_spectra_all_pairs,
    _random_derangement,
    _to_traces,
    assign_condition,
    build_zero_lag_diagnostics,
    compare_conditions,
    compute_condition_coordination,
    summarize_coordination,
    test_against_null as run_test_against_null,
    verify_null_identities,
    verify_null_sensitivity,
)


def _settings(**kwargs) -> PairSpikeCoordinationSettings:
    base = {
        "cfg_path": "",
        "n_trial_shuffle_draws": 25,
        "n_circular_shift_draws": 25,
        # The synthetic windows here are 64 bins, not the 1000 of real data, so
        # the production 50 ms floor would leave no admissible shift.
        "min_circular_shift_ms": 4.0,
    }
    base.update(kwargs)
    return PairSpikeCoordinationSettings(**base)


class TestNullConstruction(unittest.TestCase):
    def test_identities_match_brute_force(self):
        frame = verify_null_identities()
        self.assertTrue(frame["passes"].all(), frame.to_string(index=False))
        self.assertLess(float(frame["max_abs_error"].max()), IDENTITY_TOLERANCE)

    def test_all_pairs_matmul_equals_per_pair_loop(self):
        rng = np.random.default_rng(0)
        n_units, n_fixations, n_bins = 4, 6, 64
        trains = (rng.random((n_units, n_fixations, n_bins)) < 0.1).astype(float)
        spectra = np.fft.rfft(trains, n=n_bins, axis=-1)
        batched = _to_traces(_cross_spectra_all_pairs(spectra, spectra), n_bins)
        for first in range(n_units):
            for second in range(n_units):
                expected = np.fft.fftshift(
                    np.fft.irfft(
                        (spectra[first] * spectra[second].conj()).mean(axis=0), n=n_bins
                    )
                )
                np.testing.assert_allclose(batched[first, second], expected, atol=1e-12)

    def test_derangement_has_no_fixed_point_and_is_a_permutation(self):
        rng = np.random.default_rng(1)
        for size in (2, 5, 40):
            for _ in range(25):
                order = _random_derangement(size, rng)
                self.assertEqual(sorted(order.tolist()), list(range(size)))
                self.assertFalse(np.any(order == np.arange(size)))

    def test_null_uses_same_number_of_terms_as_observed(self):
        """A count-matched null is the whole point: same F, not F*(F-1)."""
        rng = np.random.default_rng(2)
        n_fixations = 12
        trains = (rng.random((2, n_fixations, 64)) < 0.1).astype(float)
        result = compute_condition_coordination(
            trains, settings=_settings(), bin_size_ms=1.0, rng=rng
        )
        self.assertEqual(int(result["n_fixations"]), n_fixations)

    def test_lag_sign_convention(self):
        """Positive lag means the first unit fires after the second."""
        n_bins, delay = 64, 5
        trains = np.zeros((2, 2, n_bins))
        for fixation in range(2):
            trains[0, fixation, 10] = 1.0
            trains[1, fixation, 10 + delay] = 1.0
        result = compute_condition_coordination(
            trains,
            settings=_settings(n_trial_shuffle_draws=2, n_circular_shift_draws=2),
            bin_size_ms=1.0,
            rng=np.random.default_rng(0),
        )
        lags = result["lags_ms"]
        self.assertEqual(float(lags[np.argmax(result["observed"][1, 0])]), float(delay))
        self.assertEqual(float(lags[np.argmax(result["observed"][0, 1])]), float(-delay))


class TestNullSensitivity(unittest.TestCase):
    """Each null must respond only to the structure it is meant to detect."""

    @classmethod
    def setUpClass(cls):
        cls.frame = verify_null_sensitivity(n_fixations=150, n_draws=60, seed=3).set_index(
            "scenario"
        )

    def test_independent_pair_sits_near_zero_for_both_nulls(self):
        for null_name in ("trial_shuffle", "circular_shift"):
            value = float(self.frame.loc["independent", f"{null_name}_mean_z_pm10ms"])
            self.assertLess(abs(value), 1.5, f"{null_name} is not centred on an uncoupled pair")

    def test_shared_rate_moves_trial_shuffle_but_not_circular_shift(self):
        shuffle = float(self.frame.loc["shared_rate", "trial_shuffle_mean_z_pm10ms"])
        shift = float(self.frame.loc["shared_rate", "circular_shift_mean_z_pm10ms"])
        self.assertGreater(shuffle, 1.0)
        self.assertGreater(shuffle, shift)

    def test_synchrony_is_detected_by_both_nulls_at_the_injected_lag(self):
        for null_name in ("trial_shuffle", "circular_shift"):
            self.assertGreater(
                float(self.frame.loc["synchronous", f"{null_name}_mean_z_pm10ms"]), 1.5
            )
            self.assertEqual(
                float(self.frame.loc["synchronous", f"{null_name}_peak_lag_ms"]), -4.0
            )

    def test_common_input_appears_at_exactly_zero_lag(self):
        for null_name in ("trial_shuffle", "circular_shift"):
            self.assertEqual(
                float(self.frame.loc["common_zero_lag", f"{null_name}_peak_lag_ms"]), 0.0
            )


class TestDegenerateNulls(unittest.TestCase):
    def test_silent_units_do_not_produce_enormous_z_scores(self):
        """Round-off over round-off must not manufacture a z of 1e17."""
        trains = np.zeros((2, 10, 64))
        trains[0, 0, 5] = 1.0
        trains[1, 3, 40] = 1.0
        result = compute_condition_coordination(
            trains, settings=_settings(), bin_size_ms=1.0, rng=np.random.default_rng(0)
        )
        for null_name in ("trial_shuffle", "circular_shift"):
            excess = result["observed"][0, 1] - result[f"{null_name}_mean"][0, 1]
            sd = result[f"{null_name}_sd"][0, 1]
            finite = sd > 0
            if finite.any():
                z = excess[finite] / sd[finite]
                self.assertLess(float(np.max(np.abs(z))), 1e3)

    def test_integer_input_traces_land_on_the_count_quantum(self):
        rng = np.random.default_rng(4)
        n_fixations = 8
        trains = (rng.random((2, n_fixations, 64)) < 0.1).astype(float)
        result = compute_condition_coordination(
            trains, settings=_settings(), bin_size_ms=1.0, rng=rng
        )
        scaled = result["observed"][0, 1] * n_fixations
        np.testing.assert_allclose(scaled, np.round(scaled), atol=1e-9)


class TestConditionAssignment(unittest.TestCase):
    def test_faces_split_by_interactive_state_objects_pooled(self):
        self.assertEqual(assign_condition("face", "interactive"), "face_interactive")
        self.assertEqual(assign_condition("face", "non_interactive"), "face_non_interactive")
        self.assertEqual(assign_condition("object", "interactive"), "object")
        self.assertEqual(assign_condition("object", "non_interactive"), "object")

    def test_unhandled_categories_are_dropped(self):
        self.assertIsNone(assign_condition("out_of_roi", "interactive"))
        self.assertIsNone(assign_condition("face", None))


def _synthetic_pairs(n_pairs: int = 60, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_pairs):
        for condition in CONDITION_ORDER:
            offset = 0.6 if condition == "face_interactive" else 0.0
            rows.append(
                {
                    "pair_key": f"pair{index}",
                    "date": f"010{index % 3}2018",
                    "session": str(index % 4),
                    "condition": condition,
                    "same_region": index % 2 == 0,
                    "region_pair": "bla" if index % 2 == 0 else "bla-accg",
                    "n_fixations": 30,
                    "both_selective": index % 3 == 0,
                    "trial_shuffle_mean_effect_pm10ms": rng.normal(offset, 1.0),
                    "trial_shuffle_mean_z_pm10ms": rng.normal(offset, 1.0),
                    "circular_shift_zero_lag_z": rng.normal(0.0, 1.0),
                    "circular_shift_mean_z_pm25ms": rng.normal(0.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


class TestSummariesAndTests(unittest.TestCase):
    def setUp(self):
        self.pairs = _synthetic_pairs()

    def test_summarize_adds_scope_and_bootstrap_interval(self):
        summary = summarize_coordination(self.pairs)
        self.assertIn("scope", summary.columns)
        self.assertTrue(set(summary["scope"]) <= {"within_region", "cross_region"})
        finite = summary.dropna(subset=["ci_low", "ci_high"])
        self.assertTrue((finite["ci_low"] <= finite["ci_high"]).all())

    def test_condition_comparison_is_paired_and_corrected(self):
        result = compare_conditions(self.pairs)
        self.assertIn("p_value_corrected", result.columns)
        self.assertIn("effect_size_rank_biserial", result.columns)
        # Every pair contributes all three conditions, so pairing loses nothing.
        self.assertTrue((result["n_pairs"] > 0).all())
        self.assertTrue(
            (result["p_value_corrected"].dropna() >= result["p_value"].dropna()).all()
        )

    def test_comparison_recovers_a_planted_condition_difference(self):
        result = compare_conditions(_synthetic_pairs(n_pairs=200, seed=6))
        planted = result.loc[
            (result["condition_a"] == "face_interactive")
            & (result["condition_b"] == "object")
        ]
        self.assertTrue(planted["significant"].all())
        self.assertTrue((planted["mean_difference"] > 0).all())

    def test_against_null_reports_one_row_per_group(self):
        result = run_test_against_null(self.pairs)
        self.assertEqual(len(result), 2 * len(CONDITION_ORDER))
        self.assertIn("p_value", result.columns)

    def test_zero_lag_diagnostics_flags_a_contaminated_day(self):
        pairs = _synthetic_pairs(n_pairs=120, seed=7)
        contaminated = pairs["date"] == "01002018"
        pairs.loc[contaminated, "circular_shift_zero_lag_z"] = 12.0
        diagnostics = build_zero_lag_diagnostics(pairs)
        self.assertIn("suspected_zero_lag_artifact", diagnostics.columns)
        worst = diagnostics.sort_values("frac_pairs_zero_lag_above", ascending=False)
        self.assertEqual(str(worst.iloc[0]["date"]), "01002018")


if __name__ == "__main__":
    unittest.main()
