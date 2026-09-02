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
    def test_grid_is_the_full_product_with_unique_labels(self) -> None:
        grid = protocol_sweep_grid(learning_rates=(1e-4, 1e-3), gradient_clips=(None, 1.0),
                                   schedules=("constant", "cosine"))
        self.assertEqual(len(grid), 8)
        self.assertEqual(len({config.label for config in grid}), 8)

    def test_labels_survive_a_round_trip_through_a_path(self) -> None:
        """Labels become directory names, so they must not carry dots or minus signs."""
        for config in protocol_sweep_grid():
            self.assertNotIn(".", config.label)
            self.assertNotIn("/", config.label)

    def test_seed_plan_is_deterministic_and_shared(self) -> None:
        """Every configuration must see the same seeds, or the comparison is unpaired."""
        self.assertEqual(protocol_seeds(n_seeds=5), protocol_seeds(n_seeds=5))
        self.assertEqual(len(set(protocol_seeds(n_seeds=8))), 8)


class TestJobGeneration(unittest.TestCase):
    def test_divergence_guardrails_are_disabled_in_the_written_config(self) -> None:
        """A retry or an early abort would convert a measured failure into a success."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            configs = protocol_sweep_grid(learning_rates=(1e-3,), gradient_clips=(None,), schedules=("constant",))
            commands, run_dirs = protocol_job_commands(
                configs, [7], root=root, repo_root=root,
                mrnn_cfg_path="configs/ephys_fixation_mrnn.yaml", epochs=10,
            )
            self.assertEqual(len(commands), 1)
            payload = yaml.safe_load((run_dirs[0] / "run_config.yaml").read_text())
            self.assertIsNone(payload["divergence_loss_threshold"])
            self.assertEqual(payload["epochs"], 10)
            self.assertEqual(payload["lr"], 1e-3)
            # The architecture is frozen across the sweep; only the optimiser varies.
            self.assertEqual(payload["hidden_units"], PROTOCOL_ARCHITECTURE["hidden_units"])
            self.assertEqual(payload["recurrent_bottleneck_dim"], PROTOCOL_ARCHITECTURE["recurrent_bottleneck_dim"])

    def test_commands_target_the_explicit_run_dir_entry_point(self) -> None:
        """The general training CLI writes into the legacy scratch tree and retries on
        divergence; neither is acceptable here."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            configs = protocol_sweep_grid(learning_rates=(1e-3,), gradient_clips=(None,), schedules=("constant",))
            commands, _ = protocol_job_commands(
                configs, [7], root=root, repo_root=root,
                mrnn_cfg_path="configs/ephys_fixation_mrnn.yaml", epochs=10,
            )
            self.assertIn("train_fixation_mrnn_into_run_dir.py", commands[0])
            self.assertNotIn("--scratch-id", commands[0])

    def test_completed_cells_are_not_resubmitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            configs = protocol_sweep_grid(learning_rates=(1e-3,), gradient_clips=(None,), schedules=("constant",))
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
            configs = protocol_sweep_grid(learning_rates=(1e-3,), gradient_clips=(None,), schedules=("constant",))
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
