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
