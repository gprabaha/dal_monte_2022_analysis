"""Tests for the training-protocol sweep.

The sweep exists to measure how often a configuration *fails*, so the parts worth
pinning are the ones that could silently turn a failure into a success: the seed plan
being shared across configurations, the guardrails being disabled, and the ranking
putting reliability ahead of best-case fit.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_protocol import (
    PROTOCOL_ARCHITECTURE,
    ProtocolConfig,
    count_loss_transients,
    index_protocol_runs,
    protocol_job_commands,
    protocol_run_dir,
    protocol_seeds,
    protocol_sweep_grid,
    rank_protocols,
    score_protocol_runs,
    write_selected_protocol,
)


class TestSweepGrid(unittest.TestCase):
    def test_design_is_one_factorial_plus_two_control_arms(self) -> None:
        grid = protocol_sweep_grid(
            learning_rates=(1e-3, 3e-3), activations=("tanh", "softplus"), spectral_radii=(0.9, 1.1)
        )
        main = [c for c in grid if c.lr_schedule == "cosine" and c.gradient_clip_norm is None]
        schedule_control = [c for c in grid if c.lr_schedule == "constant"]
        clip_control = [c for c in grid if c.gradient_clip_norm is not None]
        self.assertEqual(len(main), 2 * 2 * 2)
        self.assertEqual(len(schedule_control), 2)
        self.assertEqual(len(clip_control), 2)
        self.assertEqual(len(grid), len(main) + len(schedule_control) + len(clip_control))

    def test_labels_are_unique_and_path_safe(self) -> None:
        """Labels become directory names, so they must not carry dots or separators."""
        grid = protocol_sweep_grid()
        self.assertEqual(len({config.label for config in grid}), len(grid))
        for config in grid:
            self.assertNotIn(".", config.label)
            self.assertNotIn("/", config.label)

    def test_labels_distinguish_every_swept_axis(self) -> None:
        """Two configurations differing only in activation must not collide on disk."""
        grid = protocol_sweep_grid(learning_rates=(1e-3,), activations=("tanh", "softplus"),
                                   spectral_radii=(0.9,))
        main = [c for c in grid if c.lr_schedule == "cosine" and c.gradient_clip_norm is None]
        self.assertEqual(len({c.label for c in main}), len(main))
        self.assertTrue(any("tanh" in c.label for c in main))
        self.assertTrue(any("softplus" in c.label for c in main))

    def test_clipping_control_uses_a_threshold_that_binds(self) -> None:
        """The historical value of 1.0 sat 40x above the measured gradients and was inert,
        so the control arm has to test a threshold in the range the gradients occupy."""
        clip_control = [c for c in protocol_sweep_grid() if c.gradient_clip_norm is not None]
        self.assertTrue(clip_control)
        for config in clip_control:
            self.assertLess(float(config.gradient_clip_norm), 0.5)

    def test_activation_and_spectral_radius_reach_the_run_settings(self) -> None:
        config = ProtocolConfig(label="probe", lr=1e-3, gradient_clip_norm=None,
                                activation="softplus", spectral_radius=0.9)
        overrides = config.overrides()
        self.assertEqual(overrides["activation"], "softplus")
        self.assertEqual(overrides["spectral_radius"], 0.9)

    def test_seed_plan_is_deterministic_and_shared(self) -> None:
        """Every configuration must see the same seeds, or the comparison is unpaired."""
        self.assertEqual(protocol_seeds(n_seeds=5), protocol_seeds(n_seeds=5))
        self.assertEqual(len(set(protocol_seeds(n_seeds=8))), 8)


class TestJobGeneration(unittest.TestCase):
    @staticmethod
    def _single_config() -> list[ProtocolConfig]:
        grid = protocol_sweep_grid(learning_rates=(1e-3,), activations=("tanh",), spectral_radii=(1.1,))
        return [c for c in grid if c.lr_schedule == "cosine" and c.gradient_clip_norm is None]

    def test_divergence_guardrails_are_disabled_in_the_written_config(self) -> None:
        """A retry or an early abort would convert a measured failure into a success."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            configs = self._single_config()
            commands, run_dirs = protocol_job_commands(
                configs, [7], root=root, repo_root=root,
                mrnn_cfg_path="configs/ephys_fixation_mrnn.yaml", epochs=10,
            )
            self.assertEqual(len(commands), 1)
            payload = yaml.safe_load((run_dirs[0] / "run_config.yaml").read_text())
            self.assertIsNone(payload["divergence_loss_threshold"])
            self.assertEqual(payload["epochs"], 10)
            self.assertEqual(payload["lr"], 1e-3)
            self.assertEqual(payload["lr_schedule"], "cosine")
            self.assertEqual(payload["activation"], "tanh")
            # The architecture stays frozen; only the swept axes move.
            self.assertEqual(payload["hidden_units"], PROTOCOL_ARCHITECTURE["hidden_units"])
            self.assertEqual(payload["recurrent_bottleneck_dim"], PROTOCOL_ARCHITECTURE["recurrent_bottleneck_dim"])

    def test_commands_target_the_explicit_run_dir_entry_point(self) -> None:
        """The general training CLI writes into the legacy scratch tree and retries on
        divergence; neither is acceptable here."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            commands, _ = protocol_job_commands(
                self._single_config(), [7], root=root, repo_root=root,
                mrnn_cfg_path="configs/ephys_fixation_mrnn.yaml", epochs=10,
            )
            self.assertIn("train_fixation_mrnn_into_run_dir.py", commands[0])
            self.assertNotIn("--scratch-id", commands[0])

    def test_completed_cells_are_not_resubmitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            configs = self._single_config()
            run_dir = protocol_run_dir(root, configs[0], 7)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoint_best.pth").write_bytes(b"")
            commands, run_dirs = protocol_job_commands(
                configs, [7], root=root, repo_root=root,
                mrnn_cfg_path="configs/ephys_fixation_mrnn.yaml", epochs=10,
            )
            self.assertEqual(commands, [])
            self.assertEqual(len(run_dirs), 1)


class TestTransientCounting(unittest.TestCase):
    def test_a_smooth_descent_has_no_transients(self) -> None:
        self.assertEqual(count_loss_transients(np.geomspace(1.0, 1e-4, 500)), 0)

    def test_upward_spikes_are_counted_and_descents_are_not(self) -> None:
        losses = np.geomspace(1.0, 1e-4, 300).copy()
        losses[100] = 10.0
        losses[200] = 10.0
        self.assertEqual(count_loss_transients(losses), 2)

    def test_non_finite_values_do_not_crash_the_count(self) -> None:
        losses = np.array([1.0, 0.5, np.nan, np.inf, 0.1])
        self.assertIsInstance(count_loss_transients(losses), int)


class TestRanking(unittest.TestCase):
    @staticmethod
    def _scores() -> pd.DataFrame:
        return pd.DataFrame([
            # Fits best on the seeds that work, but loses two of four.
            *[{"label": "brilliant_but_fragile", "lr": 3e-3, "gradient_clip_norm": None,
               "lr_schedule": "constant", "seed": s, "diverged": s >= 2,
               "best_loss": 1e-5 if s < 2 else np.nan, "final_over_best": 1.2,
               "n_transients": 9, "iterations": 100} for s in range(4)],
            # Slightly worse typical loss, but never fails and barely varies.
            *[{"label": "steady", "lr": 1e-3, "gradient_clip_norm": 1.0,
               "lr_schedule": "cosine", "seed": s, "diverged": False,
               "best_loss": 2e-5 + s * 1e-7, "final_over_best": 1.01,
               "n_transients": 0, "iterations": 100} for s in range(4)],
        ])

    def test_reliability_outranks_best_case_fit(self) -> None:
        ranked = rank_protocols(self._scores(), n_seeds=4)
        self.assertEqual(ranked.iloc[0]["label"], "steady")
        self.assertEqual(int(ranked.set_index("label").loc["brilliant_but_fragile", "n_failed"]), 2)
        self.assertEqual(float(ranked.set_index("label").loc["steady", "failure_rate"]), 0.0)

    def test_failed_seeds_are_excluded_from_the_loss_summary(self) -> None:
        """A diverged seed must not contribute a NaN or an outlier to the median."""
        ranked = rank_protocols(self._scores(), n_seeds=4).set_index("label")
        self.assertAlmostEqual(float(ranked.loc["brilliant_but_fragile", "median_best_loss"]), 1e-5, places=12)

    def test_selected_protocol_freezes_the_winner_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            configs = [
                ProtocolConfig(label="steady", lr=1e-3, gradient_clip_norm=1.0, lr_schedule="cosine"),
                ProtocolConfig(label="brilliant_but_fragile", lr=3e-3, gradient_clip_norm=None),
            ]
            ranked = rank_protocols(self._scores(), n_seeds=4)
            path = write_selected_protocol(ranked, configs, path=Path(tmp_dir) / "selected.yaml", epochs=50000)
            payload = yaml.safe_load(path.read_text())
            self.assertEqual(payload["selected_label"], "steady")
            self.assertEqual(payload["optimizer"]["lr"], 1e-3)
            self.assertEqual(payload["optimizer"]["gradient_clip_norm"], 1.0)
            self.assertEqual(payload["epochs"], 50000)


class TestInventory(unittest.TestCase):
    def test_cells_are_classified_as_complete_diverged_or_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            grid = protocol_sweep_grid(learning_rates=(1e-3,), activations=("tanh",), spectral_radii=(1.1,))
            configs = [c for c in grid if c.lr_schedule == "cosine" and c.gradient_clip_norm is None]
            seeds = [1, 2, 3]
            done = protocol_run_dir(root, configs[0], 1)
            done.mkdir(parents=True, exist_ok=True)
            (done / "checkpoint_best.pth").write_bytes(b"")
            failed = protocol_run_dir(root, configs[0], 2)
            failed.mkdir(parents=True, exist_ok=True)
            (failed / "training_failed.json").write_text("{}")

            inventory = index_protocol_runs(root, configs, seeds).set_index("seed")
            self.assertTrue(bool(inventory.loc[1, "complete"]))
            self.assertTrue(bool(inventory.loc[2, "diverged"]))
            self.assertTrue(bool(inventory.loc[3, "pending"]))

            scores = score_protocol_runs(index_protocol_runs(root, configs, seeds))
            self.assertEqual(len(scores), 3)
            self.assertTrue(scores["best_loss"].isna().all())


if __name__ == "__main__":
    unittest.main()


class TestIsolationDecomposition(unittest.TestCase):
    """Cutting a region off changes two things at once; a pooled score cannot say which."""

    @staticmethod
    def _fit() -> pd.DataFrame:
        from itertools import product

        rows = []
        for label, region_scores in {
            "full": {"a": 1.00, "b": 1.00, "c": 1.00},
            # 'a' cannot stand alone, but the others do not miss it.
            "isolate_a": {"a": 0.70, "b": 0.99, "c": 0.99},
            # 'b' is fine alone, but the others need it.
            "isolate_b": {"a": 0.80, "b": 0.99, "c": 0.80},
        }.items():
            for region, score in region_scores.items():
                for condition in ("face_interactive", "object"):
                    rows.append({"label": label, "region": region, "condition": condition,
                                 "r2_vs_ceiling": score, "seed": 1})
        return pd.DataFrame(rows)

    def test_the_two_costs_are_reported_separately(self) -> None:
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import decompose_isolation_fit

        result = decompose_isolation_fit(
            self._fit(),
            isolated_region_by_label={"isolate_a": "a", "isolate_b": "b"},
            baseline_label="full",
        ).set_index("label")

        # 'a' depends on the others: its own fit collapses, theirs does not.
        self.assertAlmostEqual(float(result.loc["isolate_a", "isolated_cost"]), 0.30, places=6)
        self.assertAlmostEqual(float(result.loc["isolate_a", "remaining_cost"]), 0.01, places=6)

        # 'b' is the opposite: it survives alone, the others do not survive without it.
        self.assertAlmostEqual(float(result.loc["isolate_b", "isolated_cost"]), 0.01, places=6)
        self.assertAlmostEqual(float(result.loc["isolate_b", "remaining_cost"]), 0.20, places=6)

    def test_a_missing_baseline_is_an_error_not_a_nan(self) -> None:
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import decompose_isolation_fit

        with self.assertRaises(ValueError):
            decompose_isolation_fit(
                self._fit(), isolated_region_by_label={"isolate_a": "a"}, baseline_label="absent"
            )


class TestLearningRateRetry(unittest.TestCase):
    """A constraint sweep has two possible failures and must not confuse them: the data
    cannot be reproduced under the constraint, or the inherited recipe cannot optimise it.
    Since the recipe is selected on the unconstrained model, the second is the likelier."""

    @staticmethod
    def _variants():
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import ModelVariant

        return [
            ModelVariant(label="dense", overrides={"recurrent_bottleneck_dim": None}, arm="baseline"),
            ModelVariant(label="rank03", overrides={"recurrent_bottleneck_dim": 3}, arm="rank"),
        ]

    @staticmethod
    def _convergence(rank_converged: int):
        return pd.DataFrame([
            {"label": "dense", "n_seeds": 3, "n_converged": 3, "median_best_loss": 1.5e-4},
            {"label": "rank03", "n_seeds": 3, "n_converged": rank_converged, "median_best_loss": 2.0e-4},
        ])

    def test_only_failures_are_retried(self) -> None:
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import retry_variants_for_nonconverged

        retries = retry_variants_for_nonconverged(
            self._variants(), self._convergence(rank_converged=1), inherited_lr=3e-4
        )
        self.assertTrue(all(r.label.startswith("rank03__lr") for r in retries))
        self.assertFalse(any("dense" in r.label for r in retries))

    def test_nothing_is_retried_when_everything_converged(self) -> None:
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import retry_variants_for_nonconverged

        self.assertEqual(
            retry_variants_for_nonconverged(
                self._variants(), self._convergence(rank_converged=3), inherited_lr=3e-4
            ),
            [],
        )

    def test_retries_only_go_downward_in_learning_rate(self) -> None:
        """Retrying upward would make an unstable variant less stable, not more."""
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import retry_variants_for_nonconverged

        retries = retry_variants_for_nonconverged(
            self._variants(), self._convergence(rank_converged=0), inherited_lr=1e-4
        )
        for retry in retries:
            self.assertLess(float(retry.overrides["lr"]), 1e-4)

    def test_a_variant_that_needed_a_lower_rate_is_reported_as_such(self) -> None:
        """Variants fitted at different step sizes stay comparable on fit, but the
        difference has to be visible rather than hidden in the label."""
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import resolve_best_converged

        convergence = pd.concat([
            self._convergence(rank_converged=1),
            pd.DataFrame([{"label": "rank03__lr1em04", "n_seeds": 3, "n_converged": 3,
                           "median_best_loss": 2.2e-4}]),
        ])
        resolved = resolve_best_converged(convergence, base_labels=["dense", "rank03"]).set_index("variant")
        self.assertFalse(bool(resolved.loc["dense", "needed_lower_lr"]))
        self.assertTrue(bool(resolved.loc["rank03", "needed_lower_lr"]))
        self.assertTrue(bool(resolved.loc["rank03", "converged"]))

    def test_a_variant_that_never_converges_is_not_silently_passed(self) -> None:
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import resolve_best_converged

        convergence = pd.concat([
            self._convergence(rank_converged=1),
            pd.DataFrame([{"label": "rank03__lr1em04", "n_seeds": 3, "n_converged": 2,
                           "median_best_loss": 2.2e-4}]),
        ])
        resolved = resolve_best_converged(convergence, base_labels=["rank03"]).set_index("variant")
        self.assertFalse(bool(resolved.loc["rank03", "converged"]))


class TestSubmissionClaims(unittest.TestCase):
    """A queued cell and a never-submitted cell are identical on disk.

    The checkpoint is written only when training ends, so nothing about a run directory
    distinguishes "waiting on the queue" from "never run". Submitting the difference
    between the variant list and what is on disk would therefore queue a duplicate of
    every unfinished cell, and the duplicate would train into the same directory as the
    original. These tests pin the record that makes the distinction, including the
    reconstruction of it for arrays submitted before the record existed.
    """

    def _variants(self):
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import ModelVariant

        return [ModelVariant(label="dense"), ModelVariant(label="rank03")]

    def test_claimed_cells_are_excluded_from_commands_and_counted_separately(self) -> None:
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import (
            index_variant_runs,
            variant_job_commands,
            variant_run_dir,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            variants = self._variants()
            seeds = [1, 2]
            claimed = variant_run_dir(root, variants[0], 1)

            commands, _ = variant_job_commands(
                variants, seeds, root=root, repo_root=root,
                protocol={"architecture": {}, "optimizer": {}, "epochs": 10},
                mrnn_cfg_path=Path("configs/ephys_fixation_mrnn.yaml"),
                exclude_run_dirs=[claimed],
            )
            self.assertEqual(len(commands), 3)
            self.assertNotIn(str(claimed), " ".join(commands))

            inventory = index_variant_runs(root, variants, seeds, in_flight=[claimed])
            queued = inventory[inventory["queued"]]
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued.iloc[0]["run_dir"], str(claimed))
            # A queued cell is not also pending: pending is what a submission would queue.
            self.assertEqual(int(inventory["pending"].sum()), 3)

    def test_a_record_only_claims_cells_while_its_array_is_live(self) -> None:
        from unittest.mock import patch

        from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_sweep as sweep

        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs_dir = Path(tmp_dir) / "_jobs"
            sweep.record_submission(jobs_dir, job_id="4242", run_dirs=[Path(tmp_dir) / "a"])

            with patch.object(sweep, "_array_job_states", return_value={"4242": {"RUNNING": 2}}):
                self.assertEqual(len(sweep.in_flight_run_dirs(jobs_dir)["run_dirs"]), 1)
            # Once the array leaves the queue its cells are free again, so a failed cell
            # can be resubmitted rather than being claimed forever by a dead job.
            with patch.object(sweep, "_array_job_states", return_value={}):
                self.assertEqual(len(sweep.in_flight_run_dirs(jobs_dir)["run_dirs"]), 0)

    def test_arrays_submitted_before_records_existed_are_reconstructed(self) -> None:
        from unittest.mock import patch

        from dal_monte_2022_analysis.ephys.analysis import fixation_mrnn_sweep as sweep

        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs_dir = Path(tmp_dir) / "_jobs"
            jobs_dir.mkdir(parents=True)
            run_dir = Path(tmp_dir) / "dense" / "seed=1"
            (jobs_dir / "bottleneck.txt").write_text(
                f"cd /repo && python train.py --run-dir {run_dir} --seed 1 --overwrite\n"
            )
            (jobs_dir / "job_id.txt").write_text("29250099\n")

            with patch.object(sweep, "_array_job_states",
                              return_value={"29250099": {"PENDING": 1}}):
                claimed = sweep.in_flight_run_dirs(jobs_dir)["run_dirs"]
            self.assertEqual(claimed, {str(run_dir.resolve())})
            self.assertTrue(sweep.submission_record_path(jobs_dir, "29250099").exists())


class TestAdequacy(unittest.TestCase):
    def test_adequacy_is_judged_on_the_worst_condition_not_the_mean(self) -> None:
        """A constraint that trades one condition for another has not been tolerated.

        Interactive-face fixations are the condition the objective finds hardest and the
        one the chapter is about, so a variant that keeps its average fit by giving up
        interactive-face structure must not pass.
        """
        from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import adequacy_table

        rows = []
        for label, offsets in (
            ("dense", {"face_interactive": 0.0, "object": 0.0}),
            ("traded", {"face_interactive": 0.10, "object": -0.10}),
        ):
            for seed in (1, 2, 3):
                for condition, offset in offsets.items():
                    rows.append({"label": label, "seed": seed, "condition": condition,
                                 "r2_vs_ceiling": 1.0 - offset})
        table = adequacy_table(pd.DataFrame(rows)).set_index("label")
        self.assertAlmostEqual(float(table.loc["traded", "mean_all"]), 1.0, places=6)
        self.assertFalse(bool(table.loc["traded", "adequate"]))
        self.assertTrue(bool(table.loc["dense", "adequate"]))
