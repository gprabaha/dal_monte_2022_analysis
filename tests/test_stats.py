"""Regression tests for shared statistical helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.core.stats import (
    adjust_pvalues,
    apply_adjusted_pvalues,
    normalize_pvalue_correction,
    one_sample_ttest_greater,
    one_sided_pvalue_from_ttest,
    permutation_label_statistic_test,
    permutation_mean_difference_test,
    reject_nulls,
    safe_one_sample_ttest_greater,
    safe_paired_ttest,
)


class TestPValueCorrectionHelpers(unittest.TestCase):
    """Checks for shared multiple-comparison helpers."""

    def test_normalize_pvalue_correction_aliases(self) -> None:
        self.assertEqual(normalize_pvalue_correction("bh"), "fdr_bh")
        self.assertEqual(normalize_pvalue_correction("benjamini-hochberg"), "fdr_bh")
        self.assertEqual(normalize_pvalue_correction("holm"), "holm")

    def test_adjust_pvalues_methods(self) -> None:
        vec = np.asarray([0.01, 0.03, 0.04], dtype=float)

        np.testing.assert_allclose(adjust_pvalues(vec, "none"), [0.01, 0.03, 0.04])
        np.testing.assert_allclose(adjust_pvalues(vec, "bonferroni"), [0.03, 0.09, 0.12])
        np.testing.assert_allclose(adjust_pvalues(vec, "holm"), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(adjust_pvalues(vec, "fdr_bh"), [0.03, 0.04, 0.04])

    def test_apply_adjusted_pvalues_respects_groups(self) -> None:
        df = pd.DataFrame(
            {
                "region": ["a", "a", "b", "b"],
                "p_value": [0.01, 0.03, 0.01, 0.03],
            }
        )
        out = apply_adjusted_pvalues(
            df,
            p_col="p_value",
            out_col="p_value_adjusted",
            method="bonferroni",
            group_cols=("region",),
        )
        np.testing.assert_allclose(
            out["p_value_adjusted"].to_numpy(dtype=float),
            [0.02, 0.06, 0.02, 0.06],
        )

    def test_reject_nulls_uses_adjusted_pvalues(self) -> None:
        mask = reject_nulls([0.01, 0.04, 0.2], alpha=0.05, method="holm")
        np.testing.assert_array_equal(mask, [True, False, False])


class TestHypothesisTestHelpers(unittest.TestCase):
    """Checks for shared test wrappers used across analyses."""

    def test_one_sided_pvalue_from_ttest(self) -> None:
        greater = one_sided_pvalue_from_ttest(
            np.asarray([2.0, -2.0], dtype=float),
            np.asarray([0.04, 0.04], dtype=float),
            alternative="greater",
        )
        less = one_sided_pvalue_from_ttest(
            np.asarray([2.0, -2.0], dtype=float),
            np.asarray([0.04, 0.04], dtype=float),
            alternative="less",
        )
        np.testing.assert_allclose(greater, [0.02, 0.98])
        np.testing.assert_allclose(less, [0.98, 0.02])

    def test_safe_paired_ttest_handles_constant_difference(self) -> None:
        stat, p_value, n_pairs = safe_paired_ttest([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
        self.assertEqual(n_pairs, 3)
        self.assertTrue(np.isinf(stat))
        self.assertEqual(p_value, 0.0)

    def test_safe_one_sample_ttest_greater(self) -> None:
        stat, p_value, n = safe_one_sample_ttest_greater([1.0, 1.0, 1.0], popmean=0.0)
        self.assertEqual(n, 3)
        self.assertTrue(np.isinf(stat))
        self.assertEqual(p_value, 0.0)

    def test_one_sample_ttest_greater_vectorized(self) -> None:
        values = np.asarray(
            [
                [0.6, 0.1],
                [0.8, 0.0],
                [0.7, -0.1],
                [0.9, 0.2],
            ],
            dtype=float,
        )
        stat, p_value = one_sample_ttest_greater(values, popmean=0.0, axis=0)
        self.assertEqual(stat.shape, (2,))
        self.assertEqual(p_value.shape, (2,))
        self.assertLess(p_value[0], 0.05)
        self.assertGreater(p_value[1], 0.05)


class TestPermutationHelpers(unittest.TestCase):
    """Checks for shared permutation-test drivers."""

    def test_permutation_mean_difference_test_without_permutations(self) -> None:
        observed, p_value = permutation_mean_difference_test(
            np.asarray([1.0, 2.0], dtype=float),
            np.asarray([3.0, 4.0], dtype=float),
            n_permutations=0,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(observed, -2.0)
        self.assertTrue(np.isnan(p_value))

    def test_permutation_label_statistic_test_without_permutations(self) -> None:
        data = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
        labels = np.asarray(["a", "a", "b", "b"], dtype=object)

        def statistic_fn(values: np.ndarray, group_labels: np.ndarray) -> float:
            return float(np.mean(values[group_labels == "a"]) - np.mean(values[group_labels == "b"]))

        observed, p_value = permutation_label_statistic_test(
            data,
            labels,
            n_permutations=0,
            rng=np.random.default_rng(0),
            statistic_fn=statistic_fn,
        )
        self.assertEqual(observed, -2.0)
        self.assertTrue(np.isnan(p_value))


if __name__ == "__main__":
    unittest.main()
