"""Regression tests for fixation PSTH trial spike-train storage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from dal_monte_2022_analysis.data.records.ephys import EphysUnitContext, UnitSpikeData
    from dal_monte_2022_analysis.ephys.features.fixation_psth import (
        FixationPSTHSettings,
        build_fixation_psth_trials_for_session,
    )
    from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path

    _HAS_FIX_PSTH_MODULES = True
except ModuleNotFoundError:
    _HAS_FIX_PSTH_MODULES = False


def _write_dataset_cfg(path: Path, processed_root: Path, analysis_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "dataset_name: test_dataset",
                f"raw_data_root: {processed_root}",
                f"processed_data_root: {processed_root}",
                f"analysis_output_root: {analysis_root}",
                "processed_data_layout:",
                '  pattern: "date={date}/session={session}/{modality}"',
            ]
        ),
        encoding="utf-8",
    )


@unittest.skipUnless(_HAS_FIX_PSTH_MODULES, "fixation PSTH modules are required for these tests")
class TestFixationPSTHTrialSpikeTrainStorage(unittest.TestCase):
    """Checks that fixation trial outputs keep a separate 1 ms spike-train vector."""

    def _build_single_fixation_trial_output(
        self,
        *,
        settings: FixationPSTHSettings,
        spike_ts: np.ndarray,
    ) -> dict:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            timeline_dir = processed_root / "date=20990101" / "session=s1" / "neural_timeline"
            timeline_dir.mkdir(parents=True, exist_ok=True)
            save_pickle_path(
                SimpleNamespace(
                    t=np.asarray([float(idx) / 1000.0 for idx in range(2001)], dtype=float),
                ),
                timeline_dir / "shared.pkl",
            )

            fixations_dir = processed_root / "date=20990101" / "session=s1" / "fixations"
            fixations_dir.mkdir(parents=True, exist_ok=True)
            fix_path = fixations_dir / "agent=m1.pkl"
            save_pickle_path(
                pd.DataFrame(
                    [
                        {
                            "start": 1000,
                            "stop": 1005,
                            "location": ["face"],
                            "monkey_name": "m1_monkey",
                        }
                    ]
                ),
                fix_path,
            )

            unit = UnitSpikeData(
                context=EphysUnitContext(
                    date="20990101",
                    session_name="20990101_s1",
                    unit_uuid="u1",
                    region="BLA",
                    spike_channel="1",
                    recorded_agent="m1",
                    area="BLA",
                ),
                spike_ts=np.asarray(spike_ts, dtype=float),
            )

            settings.cfg_path = str(cfg_path)
            row = {"date": "20990101", "session": "s1", "agent_paths": {"m1": fix_path}}
            data = build_fixation_psth_trials_for_session(settings, row, [unit])
            self.assertIsNotNone(data)
            assert data is not None
            return data

    def test_trial_builder_stores_spike_train_counts_and_metadata(self) -> None:
        settings = FixationPSTHSettings(
            cfg_path="",
            include_interactive_state=False,
            bin_size_ms=10.0,
            spike_train_bin_size_ms=1.0,
            window_pre_s=0.01,
            window_post_s=0.02,
            use_parallel=False,
        )
        data = self._build_single_fixation_trial_output(
            settings=settings,
            spike_ts=np.asarray([0.9905, 0.9998, 1.0002, 1.0098, 1.0102, 1.0197], dtype=float),
        )

        meta = data["meta"]
        trial_df = data["trials"]
        self.assertEqual(len(trial_df), 1)
        self.assertEqual(float(meta["bin_size_ms"]), 10.0)
        self.assertEqual(float(meta["bin_step_ms"]), 10.0)
        self.assertEqual(float(meta["spike_train_bin_size_ms"]), 1.0)
        self.assertEqual(len(np.asarray(meta["bin_centers_s_rel"], dtype=float)), 3)
        self.assertEqual(len(np.asarray(meta["spike_train_bin_centers_s_rel"], dtype=float)), 30)

        trial_row = trial_df.iloc[0]
        psth_counts = np.asarray(trial_row["psth_counts"], dtype=int).reshape(-1)
        spike_train_counts = np.asarray(trial_row["spike_train_counts"], dtype=int).reshape(-1)

        self.assertTrue(np.array_equal(psth_counts, np.asarray([2, 2, 2], dtype=int)))
        self.assertEqual(int(spike_train_counts.sum()), 6)
        self.assertEqual(np.flatnonzero(spike_train_counts).tolist(), [0, 9, 10, 19, 20, 29])

    def test_trial_builder_can_store_spike_train_only_outputs(self) -> None:
        settings = FixationPSTHSettings(
            cfg_path="",
            include_interactive_state=False,
            bin_size_ms=10.0,
            spike_train_bin_size_ms=1.0,
            store_psth_counts=False,
            store_spike_train_counts=True,
            window_pre_s=0.01,
            window_post_s=0.02,
            use_parallel=False,
        )
        data = self._build_single_fixation_trial_output(
            settings=settings,
            spike_ts=np.asarray([0.9998, 1.0002, 1.0098], dtype=float),
        )

        meta = data["meta"]
        trial_row = data["trials"].iloc[0]
        self.assertNotIn("bin_size_ms", meta)
        self.assertNotIn("bin_centers_s_rel", meta)
        self.assertEqual(float(meta["spike_train_bin_size_ms"]), 1.0)
        self.assertIn("spike_train_bin_centers_s_rel", meta)
        self.assertNotIn("psth_counts", trial_row.index)
        self.assertIn("spike_train_counts", trial_row.index)

    def test_trial_builder_supports_overlapping_psth_windows(self) -> None:
        settings = FixationPSTHSettings(
            cfg_path="",
            include_interactive_state=False,
            bin_size_ms=20.0,
            bin_step_ms=10.0,
            store_psth_counts=True,
            store_spike_train_counts=False,
            window_pre_s=0.02,
            window_post_s=0.02,
            use_parallel=False,
        )
        data = self._build_single_fixation_trial_output(
            settings=settings,
            spike_ts=np.asarray([0.9850, 0.9950, 1.0050, 1.0150], dtype=float),
        )

        meta = data["meta"]
        trial_row = data["trials"].iloc[0]
        psth_counts = np.asarray(trial_row["psth_counts"], dtype=int).reshape(-1)
        self.assertEqual(float(meta["bin_size_ms"]), 20.0)
        self.assertEqual(float(meta["bin_step_ms"]), 10.0)
        self.assertIsNone(meta["bin_edges_s_rel"])
        self.assertEqual(len(np.asarray(meta["bin_left_edges_s_rel"], dtype=float)), 3)
        self.assertEqual(len(np.asarray(meta["bin_right_edges_s_rel"], dtype=float)), 3)
        self.assertTrue(np.array_equal(psth_counts, np.asarray([2, 2, 2], dtype=int)))


if __name__ == "__main__":
    unittest.main()
