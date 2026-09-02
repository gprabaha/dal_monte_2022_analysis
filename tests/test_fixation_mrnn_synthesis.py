"""Tests for the mRNN cross-run synthesis.

The parts worth pinning are the ones a claim in the notebook rests on: the run
classifier and the directory-name audit (which decide what counts as comparable), the
subspace machinery behind the channel-identity result, and the paired condition contrast
behind the one invariant the analysis reports.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_synthesis import (
    ChannelSubspaces,
    channel_identity_test,
    classify_run_family,
    condition_contrast_by_source,
    flag_run_name_iteration_mismatch,
    random_subspace_alignment_null,
    subspace_alignment,
    summarize_run_families,
)


class TestRunClassification(unittest.TestCase):
    def test_known_prefixes_map_to_their_family(self) -> None:
        self.assertEqual(classify_run_family("bottleneck_dim_3"), "bottleneck sweep")
        self.assertEqual(classify_run_family("bottleneck3d_l1within_50u_250k_10init"), "bottleneck ensemble")
        self.assertEqual(classify_run_family("cca_42pc_tanh_h50_full"), "architecture comparison")

    def test_bottleneck_ensemble_wins_over_bottleneck_sweep(self) -> None:
        """Prefix order matters: ``bottleneck1d_`` must not fall through to the sweep."""
        self.assertEqual(classify_run_family("bottleneck1d_l1within_50u_100k_10init"), "bottleneck ensemble")

    def test_unknown_names_are_not_silently_assigned(self) -> None:
        self.assertEqual(classify_run_family("some_new_experiment"), "other")


class TestRunNameAudit(unittest.TestCase):
    def test_directory_name_that_overstates_its_iterations_is_flagged(self) -> None:
        inventory = pd.DataFrame(
            [
                {"run": "loss_necessity_50u_200k_w5", "final_iteration": 20000.0},
                {"run": "loss_necessity_50u_100k_w5", "final_iteration": 100000.0},
                {"run": "no_iteration_claim_in_name", "final_iteration": 12345.0},
            ]
        )
        flagged = flag_run_name_iteration_mismatch(inventory)
        self.assertEqual(list(flagged["run"]), ["loss_necessity_50u_200k_w5"])
        self.assertEqual(int(flagged["name_claims_iterations"].iloc[0]), 200000)
        self.assertEqual(int(flagged["actual_iterations"].iloc[0]), 20000)

    def test_missing_iteration_counts_are_skipped_rather_than_flagged(self) -> None:
        inventory = pd.DataFrame([{"run": "run_100k", "final_iteration": np.nan}])
        self.assertTrue(flag_run_name_iteration_mismatch(inventory).empty)


class TestFamilySummary(unittest.TestCase):
    def test_settings_spread_is_reported_not_collapsed(self) -> None:
        inventory = pd.DataFrame(
            [
                {"family": "sweep", "epochs": 100000, "hidden_units": 50, "activation": "tanh",
                 "lr": 3e-4, "l1_weight_scale": 0.01, "recurrent_bottleneck_dim": 3},
                {"family": "sweep", "epochs": 50000, "hidden_units": 50, "activation": "tanh",
                 "lr": 3e-4, "l1_weight_scale": 0.01, "recurrent_bottleneck_dim": 20},
            ]
        )
        summary = summarize_run_families(inventory)
        self.assertEqual(int(summary["n_runs"].iloc[0]), 2)
        # Both training lengths must survive: collapsing them is what hides the confound.
        self.assertIn("50000", summary["epochs"].iloc[0])
        self.assertIn("100000", summary["epochs"].iloc[0])


class TestSubspaceAlignment(unittest.TestCase):
    def test_identical_subspaces_align_perfectly(self) -> None:
        basis = np.linalg.qr(np.random.default_rng(0).normal(size=(20, 3)))[0]
        self.assertAlmostEqual(subspace_alignment(basis, basis), 1.0, places=10)

    def test_orthogonal_subspaces_do_not_align(self) -> None:
        basis = np.linalg.qr(np.random.default_rng(1).normal(size=(20, 6)))[0]
        self.assertAlmostEqual(subspace_alignment(basis[:, :3], basis[:, 3:]), 0.0, places=10)

    def test_alignment_is_invariant_to_rotation_within_a_subspace(self) -> None:
        """A subspace has no preferred basis, so the measure must not depend on one."""
        rng = np.random.default_rng(2)
        a = np.linalg.qr(rng.normal(size=(20, 3)))[0]
        b = np.linalg.qr(rng.normal(size=(20, 3)))[0]
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        self.assertAlmostEqual(subspace_alignment(a, b), subspace_alignment(a @ rotation, b), places=10)

    def test_random_subspaces_sit_well_above_zero(self) -> None:
        """The chance floor is the reason an absolute alignment cannot be read on its own."""
        draws = random_subspace_alignment_null(ambient_dim=42, rank=3, n_draws=200, seed=3)
        self.assertGreater(float(np.mean(draws)), 0.15)
        self.assertLess(float(np.mean(draws)), 0.30)


class TestChannelIdentityTest(unittest.TestCase):
    @staticmethod
    def _subspaces(seed: int, *, shared: bool) -> ChannelSubspaces:
        """Two pathways into one target region, either seed-locked or seed-random."""
        rng = np.random.default_rng(0 if shared else seed)
        write = {
            ("a", "c"): np.linalg.qr(rng.normal(size=(12, 2)))[0],
            ("b", "c"): np.linalg.qr(rng.normal(size=(12, 2)))[0],
        }
        read = {key: value for key, value in write.items()}
        return ChannelSubspaces(write=write, read=read, rank=2)

    def test_seed_locked_channels_are_detected_as_pathway_specific(self) -> None:
        subspaces = {seed: self._subspaces(seed, shared=True) for seed in range(4)}
        result = channel_identity_test(subspaces, kinds=("write",))
        self.assertAlmostEqual(float(result["mean_matched"].iloc[0]), 1.0, places=8)
        self.assertLess(float(result["p_one_sided"].iloc[0]), 0.05)

    def test_independent_channels_show_no_matched_advantage(self) -> None:
        """Effect size, not one draw's p-value: the pairings are not independent, so the
        rank-sum p-value over-rejects and a single replicate is not a calibration check."""
        differences = []
        for base in range(20):
            subspaces = {seed: self._subspaces(base * 10 + seed + 1, shared=False) for seed in range(5)}
            differences.append(float(channel_identity_test(subspaces, kinds=("write",))["difference"].iloc[0]))
        self.assertLess(abs(float(np.mean(differences))), 0.02)

    def test_rank_sum_p_value_is_anti_conservative_under_the_null(self) -> None:
        """Pinned deliberately: this is why the notebook reads ``difference`` alongside
        the p-value, and why a *positive* result here would need a resampling null."""
        rejections = 0
        for base in range(40):
            subspaces = {seed: self._subspaces(base * 10 + seed + 1, shared=False) for seed in range(5)}
            rejections += float(channel_identity_test(subspaces, kinds=("write",))["p_one_sided"].iloc[0]) < 0.05
        self.assertGreater(rejections / 40, 0.05)

    def test_only_subspaces_sharing_an_ambient_space_are_compared(self) -> None:
        """Write subspaces live in the target's PC space, so pathways into different
        targets are never compared to each other."""
        rng = np.random.default_rng(7)
        subspaces = {
            seed: ChannelSubspaces(
                write={
                    ("a", "c"): np.linalg.qr(rng.normal(size=(12, 2)))[0],
                    ("a", "d"): np.linalg.qr(rng.normal(size=(12, 2)))[0],
                },
                read={},
                rank=2,
            )
            for seed in range(3)
        }
        result = channel_identity_test(subspaces, kinds=("write",))
        # Two pathways, different targets: every mismatched pairing is excluded, so the
        # comparison is reported as undefined rather than as a null result.
        self.assertEqual(int(result["n_mismatched"].iloc[0]), 0)
        self.assertTrue(np.isnan(float(result["p_one_sided"].iloc[0])))


class TestConditionContrast(unittest.TestCase):
    @staticmethod
    def _contribution(offset: float) -> pd.DataFrame:
        """Two sources x three conditions x eight seeds, with a planted shift on ``bla``."""
        rng = np.random.default_rng(11)
        rows = []
        for seed in range(8):
            level = rng.normal(0.25, 0.08)  # large seed-to-seed spread, as in the real ensemble
            for condition in ("face_interactive", "face_non_interactive", "object"):
                shift = offset if condition == "face_interactive" else 0.0
                for source in ("bla", "ofc"):
                    rows.append(
                        {
                            "init_idx": seed,
                            "active_condition": condition,
                            "source_region": source,
                            "target_region": "ofc",
                            "relative_projection": level + (shift if source == "bla" else 0.0),
                        }
                    )
        return pd.DataFrame(rows)

    def test_a_within_seed_shift_is_detected_despite_large_between_seed_spread(self) -> None:
        contrasts = condition_contrast_by_source(self._contribution(-0.03))
        row = contrasts[
            (contrasts["source_region"] == "bla")
            & (contrasts["condition_a"] == "face_interactive")
            & (contrasts["condition_b"] == "face_non_interactive")
        ].iloc[0]
        self.assertAlmostEqual(float(row["difference"]), -0.03, places=8)
        self.assertLess(float(row["p_holm"]), 0.05)

    def test_an_unshifted_source_is_not_called_significant(self) -> None:
        contrasts = condition_contrast_by_source(self._contribution(-0.03))
        row = contrasts[
            (contrasts["source_region"] == "ofc")
            & (contrasts["condition_a"] == "face_interactive")
            & (contrasts["condition_b"] == "face_non_interactive")
        ].iloc[0]
        self.assertGreater(float(row["p_holm"]), 0.05)

    def test_holm_adjustment_never_lowers_a_p_value(self) -> None:
        contrasts = condition_contrast_by_source(self._contribution(-0.03))
        self.assertTrue(bool((contrasts["p_holm"] >= contrasts["p_raw"] - 1e-12).all()))


if __name__ == "__main__":
    unittest.main()
