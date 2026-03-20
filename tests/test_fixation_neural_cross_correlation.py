"""Regression tests for fixation neural cross-correlation session builds."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis import fixation_neural_cross_correlation_helpers as xcorr_helpers
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation import (
    CROSS_ANALYSIS_KIND,
    WITHIN_ANALYSIS_KIND,
    FixationNeuralCrossCorrelationSettings,
    run_within_region_fixation_neural_cross_correlation,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_helpers import (
    _build_fixation_neural_cross_correlation_session_result,
    build_fixation_neural_cross_correlations_for_session,
)


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



class TestFixationNeuralCrossCorrelation(unittest.TestCase):
    """Checks 1 ms-style signal selection and saved lag metadata."""

    def _write_trial_file(self, path: Path, trials_df: pd.DataFrame) -> None:
        with path.open("wb") as f:
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
            self._write_trial_file(trial_path, trials_df)

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


    def test_session_build_returns_zero_xcorr_when_any_signal_is_all_zero(self) -> None:
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
                        "spike_train_counts": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float),
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
                        "spike_train_counts": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
                    },
                ]
            )
            self._write_trial_file(trial_path, trials_df)

            settings = FixationNeuralCrossCorrelationSettings(
                cfg_path=str(root / "dataset.yaml"),
                trial_input_modality="psth",
                trial_input_filename="fixations_spike_train_1ms.pkl",
                signal_input_column="spike_train_counts",
                signal_window_ms=(-500.0, 500.0),
                signal_transform="none",
                xcorr_normalization="energy",
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
            row = out["cross_correlations"].iloc[0]
            self.assertTrue(
                np.allclose(
                    np.asarray(row["cross_correlation"], dtype=float),
                    np.zeros(3, dtype=float),
                )
            )

    def test_run_within_region_writes_raw_and_smoothed_signal_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            trial_dir = processed_root / "date=20990101" / "session=s1" / "psth"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_path = trial_dir / "fixations_spike_train_1ms.pkl"

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
                        "spike_train_counts": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
                        "smoothed_spike_train_counts": np.asarray([0.2, 0.6, 0.2, 0.0], dtype=float),
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
                        "spike_train_counts": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
                        "smoothed_spike_train_counts": np.asarray([0.0, 0.2, 0.6, 0.2], dtype=float),
                    },
                ]
            )
            self._write_trial_file(trial_path, trials_df)

            settings = FixationNeuralCrossCorrelationSettings(
                cfg_path=str(cfg_path),
                trial_input_modality="psth",
                trial_input_filename="fixations_spike_train_1ms.pkl",
                signal_input_column="spike_train_counts",
                signal_input_columns=("spike_train_counts", "smoothed_spike_train_counts"),
                signal_window_ms=(-500.0, 500.0),
                signal_transform="none",
                xcorr_normalization="none",
                max_lag=1,
                use_parallel=False,
            )
            with mock.patch.object(
                xcorr_helpers,
                "load_pickle_path",
                wraps=xcorr_helpers.load_pickle_path,
            ) as mock_load_pickle, mock.patch.object(
                xcorr_helpers,
                "_build_pair_tasks",
                wraps=xcorr_helpers._build_pair_tasks,
            ) as mock_build_pair_tasks:
                summary = run_within_region_fixation_neural_cross_correlation(
                    settings,
                    dates=["20990101"],
                    sessions=["s1"],
                    use_parallel=False,
                )

            signal_summaries = summary["signal_summaries"]
            self.assertEqual(summary["n_sessions_total"], 1)
            self.assertEqual(summary["n_session_signal_runs_total"], 2)
            self.assertEqual(mock_load_pickle.call_count, 1)
            self.assertEqual(mock_build_pair_tasks.call_count, 1)
            self.assertEqual(set(signal_summaries), {"spike_train_counts", "smoothed_spike_train_counts"})
            self.assertEqual(int(signal_summaries["spike_train_counts"]["n_sessions_written"]), 1)
            self.assertEqual(int(signal_summaries["smoothed_spike_train_counts"]["n_sessions_written"]), 1)

            raw_xcorr_path = (
                analysis_root
                / "ephys/psth/fixation_neural_cross_correlation/within_region"
                / "date=20990101"
                / "session=s1"
                / "fixations.pkl"
            )
            raw_pair_avg_path = (
                analysis_root
                / "ephys/psth/fixation_neural_cross_correlation/within_region"
                / "date=20990101"
                / "session=s1"
                / "pair_averages.pkl"
            )
            smooth_xcorr_path = (
                analysis_root
                / "ephys/psth/fixation_neural_cross_correlation/within_region"
                / "date=20990101"
                / "session=s1"
                / "fixations_smoothed.pkl"
            )
            smooth_pair_avg_path = (
                analysis_root
                / "ephys/psth/fixation_neural_cross_correlation/within_region"
                / "date=20990101"
                / "session=s1"
                / "pair_averages_smoothed.pkl"
            )

            self.assertTrue(raw_xcorr_path.exists())
            self.assertTrue(raw_pair_avg_path.exists())
            self.assertTrue(smooth_xcorr_path.exists())
            self.assertTrue(smooth_pair_avg_path.exists())

            with raw_xcorr_path.open("rb") as f:
                raw_obj = pickle.load(f)
            with smooth_xcorr_path.open("rb") as f:
                smooth_obj = pickle.load(f)

            self.assertEqual(str(raw_obj["meta"]["signal_input_column"]), "spike_train_counts")
            self.assertEqual(str(smooth_obj["meta"]["signal_input_column"]), "smoothed_spike_train_counts")

    def test_cross_region_session_result_reports_missing_partner_units(self) -> None:
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
                        "spike_train_counts": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
                    },
                ]
            )
            self._write_trial_file(trial_path, trials_df)

            settings = FixationNeuralCrossCorrelationSettings(
                cfg_path=str(root / "dataset.yaml"),
                trial_input_modality="psth",
                trial_input_filename="fixations_spike_train_1ms.pkl",
                signal_input_column="spike_train_counts",
                signal_window_ms=(-500.0, 500.0),
                signal_transform="none",
                xcorr_normalization="none",
                max_lag=None,
                use_parallel=False,
                anchor_region="BLA",
                partner_regions=("ACCg", "dmPFC", "OFC"),
            )
            result = _build_fixation_neural_cross_correlation_session_result(
                settings,
                {"path": trial_path, "date": "20990101", "session": "s1"},
                analysis_kind=CROSS_ANALYSIS_KIND,
                show_progress=False,
            )

            self.assertEqual(str(result["status"]), "skipped")
            self.assertIsNone(result["data"])
            row = result["session_report_row"]
            self.assertEqual(str(row["skip_reason"]), "no_partner_region_units")
            self.assertEqual(int(row["n_trial_rows"]), 2)
            self.assertEqual(int(row["n_fixation_groups"]), 1)
            self.assertEqual(int(row["n_unique_anchor_units"]), 2)
            self.assertEqual(int(row["n_unique_partner_units"]), 0)
            self.assertEqual(str(row["regions_present_in_fixation_groups"]), "bla")


if __name__ == "__main__":
    unittest.main()
