"""Regression tests for fixation neural xcorr pair meta-analysis outputs."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_pair_meta_analysis import (
    FixationNeuralCrossCorrelationPairMetaAnalysisSettings,
    run_cross_region_fixation_neural_cross_correlation_pair_meta_analysis,
    run_within_region_fixation_neural_cross_correlation_pair_meta_analysis,
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


def _write_xcorr_session_file(
    path: Path,
    *,
    analysis_kind: str,
    date: str,
    session: str,
    signal_input_column: str,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "meta": {
                    "analysis_kind": analysis_kind,
                    "date": date,
                    "session": session,
                    "signal_input_column": signal_input_column,
                    "lags": np.asarray([-1, 0, 1], dtype=np.int64),
                    "bin_size_ms": 1.0,
                },
                "cross_correlations": pd.DataFrame(rows),
            },
            f,
        )


class TestFixationNeuralCrossCorrelationPairMetaAnalysis(unittest.TestCase):
    def test_within_region_pair_meta_analysis_aggregates_raw_and_smoothed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            within_root = analysis_root / "ephys/psth/fixation_neural_cross_correlation/within_region"
            raw_rows_s1 = [
                {
                    "date": "20990101",
                    "session": "s1",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "is_interactive": True,
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "cross_correlation": np.asarray([0.2, 0.4, 0.6], dtype=float),
                },
                {
                    "date": "20990101",
                    "session": "s1",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "is_interactive": True,
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "cross_correlation": np.asarray([0.1, 0.3, 0.5], dtype=float),
                },
            ]
            raw_rows_s2 = [
                {
                    "date": "20990101",
                    "session": "s2",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "is_interactive": True,
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "cross_correlation": np.asarray([0.3, 0.5, 0.7], dtype=float),
                }
            ]
            smooth_rows_s1 = [
                {
                    "date": "20990101",
                    "session": "s1",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "is_interactive": True,
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "cross_correlation": np.asarray([0.4, 0.6, 0.8], dtype=float),
                }
            ]
            smooth_rows_s2 = [
                {
                    "date": "20990101",
                    "session": "s2",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "is_interactive": True,
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "cross_correlation": np.asarray([0.5, 0.7, 0.9], dtype=float),
                }
            ]

            _write_xcorr_session_file(
                within_root / "date=20990101" / "session=s1" / "fixations.pkl",
                analysis_kind="within_region",
                date="20990101",
                session="s1",
                signal_input_column="spike_train_counts",
                rows=raw_rows_s1,
            )
            _write_xcorr_session_file(
                within_root / "date=20990101" / "session=s2" / "fixations.pkl",
                analysis_kind="within_region",
                date="20990101",
                session="s2",
                signal_input_column="spike_train_counts",
                rows=raw_rows_s2,
            )
            _write_xcorr_session_file(
                within_root / "date=20990101" / "session=s1" / "fixations_smoothed.pkl",
                analysis_kind="within_region",
                date="20990101",
                session="s1",
                signal_input_column="smoothed_spike_train_counts",
                rows=smooth_rows_s1,
            )
            _write_xcorr_session_file(
                within_root / "date=20990101" / "session=s2" / "fixations_smoothed.pkl",
                analysis_kind="within_region",
                date="20990101",
                session="s2",
                signal_input_column="smoothed_spike_train_counts",
                rows=smooth_rows_s2,
            )

            settings = FixationNeuralCrossCorrelationPairMetaAnalysisSettings(
                cfg_path=str(cfg_path),
                signal_input_column="spike_train_counts",
                signal_input_columns=("spike_train_counts", "smoothed_spike_train_counts"),
                min_fixations=2,
            )
            summary = run_within_region_fixation_neural_cross_correlation_pair_meta_analysis(
                settings,
                dates=["20990101"],
                show_progress=False,
            )

            signal_summaries = summary["signal_summaries"]
            self.assertEqual(summary["n_date_signal_runs_total"], 2)
            self.assertEqual(int(signal_summaries["spike_train_counts"]["n_dates_written"]), 1)
            self.assertEqual(int(signal_summaries["smoothed_spike_train_counts"]["n_dates_written"]), 1)

            raw_out = (
                analysis_root
                / "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/within_region"
                / "date=20990101"
                / "pair_fixation_lag_mean_significance.pkl"
            )
            smooth_out = (
                analysis_root
                / "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/within_region"
                / "date=20990101"
                / "pair_fixation_lag_mean_significance_smoothed.pkl"
            )
            raw_csv = raw_out.with_suffix(".csv")
            smooth_csv = smooth_out.with_suffix(".csv")
            self.assertTrue(raw_out.exists())
            self.assertTrue(smooth_out.exists())
            self.assertTrue(raw_csv.exists())
            self.assertTrue(smooth_csv.exists())

            with raw_out.open("rb") as f:
                raw_obj = pickle.load(f)
            with smooth_out.open("rb") as f:
                smooth_obj = pickle.load(f)

            raw_df = raw_obj["pair_summaries"]
            smooth_df = smooth_obj["pair_summaries"]
            self.assertEqual(len(raw_df), 1)
            self.assertEqual(len(smooth_df), 1)

            raw_row = raw_df.iloc[0]
            smooth_row = smooth_df.iloc[0]
            self.assertEqual(str(raw_row["condition"]), "face_interactive")
            self.assertEqual(int(raw_row["n_fixations"]), 3)
            self.assertEqual(int(raw_row["n_sessions"]), 2)
            self.assertTrue(bool(raw_row["significant_above_zero"]))
            self.assertTrue(
                np.allclose(
                    np.asarray(raw_row["mean_cross_correlation"], dtype=float),
                    np.asarray([0.2, 0.4, 0.6], dtype=float),
                )
            )
            self.assertEqual(str(smooth_row["signal_input_column"]), "smoothed_spike_train_counts")
            self.assertEqual(int(smooth_row["n_fixations"]), 2)
            self.assertTrue(bool(smooth_row["significant_above_zero"]))

    def test_cross_region_pair_meta_analysis_uses_date_level_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            cross_root = analysis_root / "ephys/psth/fixation_neural_cross_correlation/cross_region"
            rows_s1 = [
                {
                    "date": "20990102",
                    "session": "s1",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "is_interactive": False,
                    "region_1": "BLA",
                    "region_2": "ACCg",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "v1",
                    "cross_correlation": np.asarray([-0.3, -0.2, -0.1], dtype=float),
                }
            ]
            rows_s2 = [
                {
                    "date": "20990102",
                    "session": "s2",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "is_interactive": False,
                    "region_1": "BLA",
                    "region_2": "ACCg",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "v1",
                    "cross_correlation": np.asarray([-0.2, -0.1, 0.0], dtype=float),
                }
            ]
            _write_xcorr_session_file(
                cross_root / "date=20990102" / "session=s1" / "fixations.pkl",
                analysis_kind="cross_region",
                date="20990102",
                session="s1",
                signal_input_column="spike_train_counts",
                rows=rows_s1,
            )
            _write_xcorr_session_file(
                cross_root / "date=20990102" / "session=s2" / "fixations.pkl",
                analysis_kind="cross_region",
                date="20990102",
                session="s2",
                signal_input_column="spike_train_counts",
                rows=rows_s2,
            )

            settings = FixationNeuralCrossCorrelationPairMetaAnalysisSettings(
                cfg_path=str(cfg_path),
                signal_input_column="spike_train_counts",
                signal_input_columns=("spike_train_counts",),
                min_fixations=2,
            )
            summary = run_cross_region_fixation_neural_cross_correlation_pair_meta_analysis(
                settings,
                dates=["20990102"],
                show_progress=False,
            )

            self.assertEqual(summary["n_date_signal_runs_total"], 1)
            out_path = (
                analysis_root
                / "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/cross_region"
                / "date=20990102"
                / "pair_fixation_lag_mean_significance.pkl"
            )
            self.assertTrue(out_path.exists())
            with out_path.open("rb") as f:
                obj = pickle.load(f)
            df = obj["pair_summaries"]
            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(str(row["group_label"]), "BLA__ACCg")
            self.assertEqual(str(row["condition"]), "object")
            self.assertEqual(int(row["n_fixations"]), 2)
            self.assertFalse(bool(row["significant_above_zero"]))


if __name__ == "__main__":
    unittest.main()
