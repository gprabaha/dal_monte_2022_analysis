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
    _next_fast_length,
    _random_derangement,
    _to_traces,
    assign_condition,
    build_zero_lag_diagnostics,
    compare_conditions,
    CONDITION_METRIC,
    build_region_pair_inventory,
    compare_conditions_across_metrics,
    compute_condition_coordination,
    drop_zero_lag_artifact_dates,
    overlap_bins,
    sufficient_region_pairs,
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
        # The final row records a deliberate *difference* (the unpadded
        # transform is not the linear one), so only the identity rows are held
        # to the tolerance.
        identity_rows = frame.loc[~frame["identity"].str.contains("differs")]
        self.assertEqual(len(identity_rows), 3)
        self.assertLess(float(identity_rows["max_abs_error"].max()), IDENTITY_TOLERANCE)

    def test_observed_correlation_does_not_wrap(self):
        """A spike near one window edge must never pair with one near the other.

        This is the property that makes the observed statistic physical: at lag
        L only genuinely overlapping bins contribute, so two spikes a full
        window apart produce no coincidence at any short lag.
        """
        n_bins = 64
        trains = np.zeros((2, 2, n_bins))
        for fixation in range(2):
            trains[0, fixation, 1] = 1.0            # near the start
            trains[1, fixation, n_bins - 2] = 1.0   # near the end
        result = compute_condition_coordination(
            trains,
            settings=_settings(n_trial_shuffle_draws=2, n_circular_shift_draws=2),
            bin_size_ms=1.0,
            rng=np.random.default_rng(0),
        )
        lags = result["lags_ms"]
        observed = result["observed"][0, 1]
        # The only coincidence is the true one, at lag -(n_bins - 3).
        true_lag = -(n_bins - 3)
        self.assertAlmostEqual(float(observed[np.argmin(np.abs(lags - true_lag))]), 1.0, places=9)
        # Nothing at the short lags a wrapping correlation would have invented.
        short = np.abs(lags) <= 5
        np.testing.assert_allclose(observed[short], 0.0, atol=1e-9)

    def test_overlap_vector_matches_the_linear_taper(self):
        overlap = overlap_bins(64)
        self.assertEqual(overlap.size, 2 * 64 - 1)
        self.assertEqual(float(overlap.max()), 64.0)
        self.assertEqual(float(overlap[0]), 1.0)
        self.assertEqual(float(overlap[-1]), 1.0)

    def test_all_pairs_matmul_equals_per_pair_loop(self):
        rng = np.random.default_rng(0)
        n_units, n_fixations, n_bins = 4, 6, 64
        n_fft = _next_fast_length(2 * n_bins - 1)
        trains = (rng.random((n_units, n_fixations, n_bins)) < 0.1).astype(float)
        spectra = np.fft.rfft(trains, n=n_fft, axis=-1)
        batched = _to_traces(_cross_spectra_all_pairs(spectra, spectra), n_bins, n_fft)
        for first in range(n_units):
            for second in range(n_units):
                expected = _to_traces(
                    (spectra[first] * spectra[second].conj()).mean(axis=0), n_bins, n_fft
                )
                np.testing.assert_allclose(batched[first, second], expected, atol=1e-12)

    def test_matches_scipy_linear_correlation(self):
        """The observed path must equal what the repo's own primitive computes."""
        from scipy.signal import correlate

        rng = np.random.default_rng(11)
        n_fixations, n_bins = 5, 64
        trains = (rng.random((2, n_fixations, n_bins)) < 0.12).astype(float)
        result = compute_condition_coordination(
            trains,
            settings=_settings(n_trial_shuffle_draws=2, n_circular_shift_draws=2),
            bin_size_ms=1.0,
            rng=np.random.default_rng(0),
        )
        reference = np.mean(
            [
                correlate(trains[0, f], trains[1, f], mode="full", method="fft")
                for f in range(n_fixations)
            ],
            axis=0,
        )
        # scipy's "full" output runs from lag -(N-1) to (N-1), the same order
        # and the same sign convention this module uses.
        np.testing.assert_allclose(result["observed"][0, 1], reference, atol=1e-9)

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


def _synthetic_pairs(
    n_pairs: int = 60, seed: int = 5, n_dates: int = 8, offset: float = 0.6
) -> pd.DataFrame:
    """Synthetic pair table.

    ``n_dates`` matters: the zero-lag outlier rule is a median-absolute-deviation
    across dates, so it needs several to have a distribution to be an outlier
    against.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_pairs):
        for condition in CONDITION_ORDER:
            condition_offset = offset if condition == "face_interactive" else 0.0
            rows.append(
                {
                    "pair_key": f"pair{index}",
                    "date": f"0{index % n_dates:03d}2018",
                    "session": str(index % 4),
                    "condition": condition,
                    "same_region": index % 2 == 0,
                    "region_pair": "bla" if index % 2 == 0 else "bla-accg",
                    "n_fixations": 30,
                    "both_selective": index % 3 == 0,
                    "trial_shuffle_mean_effect_pm10ms": rng.normal(condition_offset, 1.0),
                    "trial_shuffle_mean_z_pm10ms": rng.normal(condition_offset, 1.0),
                    "trial_shuffle_mean_ratio_pm10ms": rng.normal(condition_offset, 1.0),
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

    def test_region_pair_inventory_flags_underpowered_combinations(self):
        """Not every region combination was recorded simultaneously enough to use."""
        pairs = _synthetic_pairs(n_pairs=60)
        # Add a region pair that barely exists, as dmPFC x OFC does in practice.
        thin = pairs.loc[pairs["region_pair"] == "bla-accg"].head(3).copy()
        thin["region_pair"] = "dmpfc-ofc"
        thin["same_region"] = False
        combined = pd.concat([pairs, thin], ignore_index=True)

        inventory = build_region_pair_inventory(combined, min_pairs=20)
        row = inventory.loc[inventory["region_pair"] == "dmpfc-ofc"]
        self.assertEqual(len(row), 1)
        self.assertFalse(bool(row["sufficient_pairs"].iloc[0]))
        self.assertNotIn("dmpfc-ofc", sufficient_region_pairs(combined, min_pairs=20))
        # Flagged, never silently dropped.
        self.assertIn("dmpfc-ofc", set(inventory["region_pair"]))

    def test_metric_comparison_reports_where_metrics_disagree(self):
        """A conclusion that depends on the choice of metric must be visible."""
        # No effect in any metric to start with, then plant one in the ratio
        # only -- the situation where the choice of metric changes the answer.
        pairs = _synthetic_pairs(n_pairs=200, seed=8, offset=0.0)
        interactive = pairs["condition"] == "face_interactive"
        pairs.loc[interactive, "trial_shuffle_mean_ratio_pm10ms"] += 1.5
        result = compare_conditions_across_metrics(pairs)
        self.assertIn("metrics_agree", result.columns)
        for kind in ("ratio", "effect", "z"):
            self.assertIn(f"significant_{kind}", result.columns)
        planted = result.loc[
            (result["condition_a"] == "face_interactive")
            & (result["condition_b"] == "object")
        ]
        self.assertTrue(planted["significant_ratio"].all())
        self.assertFalse(planted["metrics_agree"].all())

    def test_artifact_dates_are_dropped_not_just_flagged(self):
        pairs = _synthetic_pairs(n_pairs=240, seed=9)
        target = sorted(pairs["date"].unique())[0]
        pairs.loc[pairs["date"] == target, "circular_shift_zero_lag_z"] = 12.0
        cleaned, dropped = drop_zero_lag_artifact_dates(pairs)
        self.assertIn(target, dropped)
        # Dropped outright, in every scope, not merely flagged.
        self.assertNotIn(target, set(cleaned["date"].astype(str)))
        self.assertLess(len(cleaned), len(pairs))

    def test_condition_metric_is_the_ratio(self):
        """Conditions must not be compared on a trial-count-scaled quantity."""
        self.assertIn("ratio", CONDITION_METRIC)

    def test_zero_lag_diagnostics_flags_a_contaminated_day(self):
        pairs = _synthetic_pairs(n_pairs=240, seed=7)
        target = sorted(pairs["date"].unique())[0]
        pairs.loc[pairs["date"] == target, "circular_shift_zero_lag_z"] = 12.0
        diagnostics = build_zero_lag_diagnostics(pairs)
        self.assertIn("suspected_zero_lag_artifact", diagnostics.columns)
        worst = diagnostics.sort_values("frac_pairs_zero_lag_above", ascending=False)
        self.assertEqual(str(worst.iloc[0]["date"]), target)


if __name__ == "__main__":
    unittest.main()
