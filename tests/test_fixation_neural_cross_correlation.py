"""Regression tests for fixation neural cross-correlation session builds."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation import (
    WITHIN_ANALYSIS_KIND,
    FixationNeuralCrossCorrelationSettings,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_helpers import (
    build_fixation_neural_cross_correlations_for_session,
)


class TestFixationNeuralCrossCorrelation(unittest.TestCase):
    """Checks 1 ms-style signal selection and saved lag metadata."""

    def test_session_build_uses_windowed_spike_train_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trial_path = root / "fixations_spike_train_1ms.pkl"

            trials_df = pd.DataFrame(
                [
                    {
                        "date": "20990101",
                        "session": "s1",
                        "unit_uuid": "u1",
                        "region": "BLA",
                        "spike_channel": "ch1",
                        "fixation_agent": "m1",
                        "fixation_monkey_name": "m1",
                        "fixation_category": "face",
                        "interactive_state": "interactive",
                        "is_interactive": True,
                        "fixation_start_idx": 10,
                        "fixation_stop_idx": 12,
                        "fixation_start_time_s": 0.0,
                        "fixation_location": ("face",),
                        "psth_counts": np.asarray([9.0, 9.0, 9.0, 9.0], dtype=float),
                        "spike_train_counts": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
                    },
                    {
                        "date": "20990101",
                        "session": "s1",
                        "unit_uuid": "u2",
                        "region": "BLA",
                        "spike_channel": "ch2",
                        "fixation_agent": "m1",
                        "fixation_monkey_name": "m1",
                        "fixation_category": "face",
                        "interactive_state": "interactive",
                        "is_interactive": True,
                        "fixation_start_idx": 10,
                        "fixation_stop_idx": 12,
                        "fixation_start_time_s": 0.0,
                        "fixation_location": ("face",),
                        "psth_counts": np.asarray([9.0, 9.0, 9.0, 9.0], dtype=float),
                        "spike_train_counts": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
                    },
                ]
            )
            with trial_path.open("wb") as f:
                pickle.dump(
                    {
                        "meta": {
                            "date": "20990101",
                            "session": "s1",
                            "window_pre_s": 1.0,
                            "window_post_s": 1.0,
                            "spike_train_bin_centers_s_rel": np.asarray(
                                [-0.75, -0.25, 0.25, 0.75],
                                dtype=float,
                            ),
                            "spike_train_bin_edges_s_rel": np.asarray(
                                [-1.0, -0.5, 0.0, 0.5, 1.0],
                                dtype=float,
                            ),
                            "spike_train_bin_size_ms": 500.0,
                        },
                        "trials": trials_df,
                    },
                    f,
                )

            settings = FixationNeuralCrossCorrelationSettings(
                cfg_path=str(root / "dataset.yaml"),
                trial_input_modality="psth",
                trial_input_filename="fixations_spike_train_1ms.pkl",
                signal_input_column="spike_train_counts",
                signal_window_ms=(-500.0, 500.0),
                signal_transform="none",
                xcorr_normalization="none",
                max_lag=1,
                use_parallel=False,
            )
            out = build_fixation_neural_cross_correlations_for_session(
                settings,
                {"path": trial_path, "date": "20990101", "session": "s1"},
                analysis_kind=WITHIN_ANALYSIS_KIND,
                show_progress=False,
            )

            self.assertIsInstance(out, dict)
            assert out is not None
            meta = out["meta"]
            xcorr_df = out["cross_correlations"]
            pair_avg_df = out["pair_averages"]

            self.assertEqual(str(meta["source_filename"]), "fixations_spike_train_1ms.pkl")
            self.assertEqual(str(meta["signal_input_column"]), "spike_train_counts")
            self.assertEqual(list(meta["signal_window_ms"]), [-500.0, 500.0])
            self.assertEqual(int(meta["signal_n_bins"]), 2)
            self.assertAlmostEqual(float(meta["signal_bin_size_ms"]), 500.0, places=6)
            self.assertAlmostEqual(float(meta["bin_size_ms"]), 500.0, places=6)
            self.assertTrue(
                np.allclose(
                    np.asarray(meta["signal_bin_centers_s_rel"], dtype=float),
                    np.asarray([-0.25, 0.25], dtype=float),
                )
            )
            self.assertTrue(
                np.allclose(
                    np.asarray(meta["bin_centers_s_rel"], dtype=float),
                    np.asarray([-0.25, 0.25], dtype=float),
                )
            )
            self.assertTrue(
                np.allclose(
                    np.asarray(meta["signal_bin_edges_s_rel"], dtype=float),
                    np.asarray([-0.5, 0.0, 0.5], dtype=float),
                )
            )
            self.assertTrue(
                np.allclose(
                    np.asarray(meta["bin_edges_s_rel"], dtype=float),
                    np.asarray([-0.5, 0.0, 0.5], dtype=float),
                )
            )
            self.assertTrue(
                np.array_equal(
                    np.asarray(meta["lags"], dtype=np.int64),
                    np.asarray([-1, 0, 1], dtype=np.int64),
                )
            )

            self.assertEqual(len(xcorr_df), 1)
            self.assertNotIn("lags", xcorr_df.columns)
            row = xcorr_df.iloc[0]
            self.assertTrue(
                np.allclose(
                    np.asarray(row["cross_correlation"], dtype=float),
                    np.asarray([1.0, 0.0, 0.0], dtype=float),
                )
            )
            self.assertEqual(int(row["signal_bins_1"]), 2)
            self.assertEqual(int(row["signal_bins_2"]), 2)

            self.assertEqual(len(pair_avg_df), 1)
            pair_row = pair_avg_df.iloc[0]
            self.assertEqual(str(pair_row["condition"]), "face_interactive")
            self.assertEqual(int(pair_row["n_fixations"]), 1)


if __name__ == "__main__":
    unittest.main()
