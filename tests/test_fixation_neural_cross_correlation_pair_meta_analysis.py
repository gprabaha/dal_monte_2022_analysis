"""Regression tests for fixation neural xcorr sig-pair outputs."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_sig_xcorr_pairs import (
    FixationNeuralCrossCorrelationSigXcorrPairsSettings,
    build_fixation_neural_cross_correlation_sig_xcorr_pair_group_summary_table,
    run_cross_region_fixation_neural_cross_correlation_sig_xcorr_pairs,
    run_within_region_fixation_neural_cross_correlation_sig_xcorr_pairs,
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
    def test_within_region_sig_xcorr_pairs_aggregates_raw_and_smoothed(self) -> None:
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

            settings = FixationNeuralCrossCorrelationSigXcorrPairsSettings(
                cfg_path=str(cfg_path),
                signal_input_column="spike_train_counts",
                signal_input_columns=("spike_train_counts", "smoothed_spike_train_counts"),
                min_fixations=2,
            )
            summary = run_within_region_fixation_neural_cross_correlation_sig_xcorr_pairs(
                settings,
                dates=["20990101"],
                show_progress=False,
            )

            signal_summaries = summary["signal_summaries"]
            self.assertEqual(summary["n_date_signal_runs_total"], 2)
            self.assertEqual(int(signal_summaries["spike_train_counts"]["n_dates_written"]), 1)
            self.assertEqual(int(signal_summaries["smoothed_spike_train_counts"]["n_dates_written"]), 1)
            self.assertEqual(len(signal_summaries["spike_train_counts"]["group_summary_csv_output_paths"]), 1)
            self.assertEqual(len(signal_summaries["smoothed_spike_train_counts"]["group_summary_csv_output_paths"]), 1)

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
            raw_group_csv = raw_csv.with_name(f"{raw_csv.stem}_group_summary.csv")
            smooth_group_csv = smooth_csv.with_name(f"{smooth_csv.stem}_group_summary.csv")
            self.assertTrue(raw_out.exists())
            self.assertTrue(smooth_out.exists())
            self.assertTrue(raw_csv.exists())
            self.assertTrue(smooth_csv.exists())
            self.assertTrue(raw_group_csv.exists())
            self.assertTrue(smooth_group_csv.exists())

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
            self.assertTrue(bool(raw_row["significant_above_zero"]))
            self.assertAlmostEqual(float(raw_row["mean_cross_correlation_across_lags"]), 0.4)
            self.assertNotIn("mean_cross_correlation", raw_df.columns)
            self.assertNotIn("n_sessions", raw_df.columns)
            self.assertNotIn("sessions", raw_df.columns)
            self.assertNotIn("signal_input_column", raw_df.columns)
            self.assertEqual(str(raw_obj["meta"]["signal_input_column"]), "spike_train_counts")
            self.assertEqual(str(raw_obj["meta"]["signal_variant"]), "raw")

            self.assertEqual(int(smooth_row["n_fixations"]), 2)
            self.assertTrue(bool(smooth_row["significant_above_zero"]))
            self.assertAlmostEqual(float(smooth_row["mean_cross_correlation_across_lags"]), 0.65)
            self.assertEqual(str(smooth_obj["meta"]["signal_input_column"]), "smoothed_spike_train_counts")
            self.assertEqual(str(smooth_obj["meta"]["signal_variant"]), "smoothed")

            raw_group_df = raw_obj["group_summaries"]
            self.assertEqual(len(raw_group_df), 1)
            self.assertEqual(int(raw_group_df.iloc[0]["n_total_pairs"]), 1)
            self.assertEqual(int(raw_group_df.iloc[0]["n_sig_face_interactive_pairs"]), 1)
            self.assertEqual(int(raw_group_df.iloc[0]["n_sig_any_condition_pairs"]), 1)

            combined_group_df = build_fixation_neural_cross_correlation_sig_xcorr_pair_group_summary_table(
                [raw_out, smooth_out]
            )
            self.assertEqual(set(combined_group_df["signal_variant"].astype(str)), {"raw", "smoothed"})
            self.assertEqual(
                set(combined_group_df["signal_input_column"].astype(str)),
                {"spike_train_counts", "smoothed_spike_train_counts"},
            )
            self.assertTrue((combined_group_df["n_total_pairs"].astype(int) == 1).all())
            self.assertTrue((combined_group_df["n_sig_face_interactive_pairs"].astype(int) == 1).all())
            self.assertTrue((combined_group_df["n_sig_any_condition_pairs"].astype(int) == 1).all())

    def test_sig_xcorr_pairs_use_configured_roi_groups_for_condition_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            within_root = analysis_root / "ephys/psth/fixation_neural_cross_correlation/within_region"
            rows = [
                {
                    "date": "20990105",
                    "session": "s1",
                    "fixation_category": None,
                    "fixation_location": ("custom_face_roi",),
                    "interactive_state": "interactive",
                    "is_interactive": True,
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "cross_correlation": np.asarray([0.1, 0.2, 0.3], dtype=float),
                },
                {
                    "date": "20990105",
                    "session": "s1",
                    "fixation_category": None,
                    "fixation_location": ("custom_object_roi",),
                    "interactive_state": "non_interactive",
                    "is_interactive": False,
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "cross_correlation": np.asarray([0.2, 0.1, 0.0], dtype=float),
                },
            ]
            _write_xcorr_session_file(
                within_root / "date=20990105" / "session=s1" / "fixations.pkl",
                analysis_kind="within_region",
                date="20990105",
                session="s1",
                signal_input_column="spike_train_counts",
                rows=rows,
            )

            settings = FixationNeuralCrossCorrelationSigXcorrPairsSettings(
                cfg_path=str(cfg_path),
                signal_input_column="spike_train_counts",
                signal_input_columns=("spike_train_counts",),
                roi_groups={
                    "face": ("custom_face_roi",),
                    "object": ("custom_object_roi",),
                    "out_of_roi": ("custom_out_of_roi",),
                },
                min_fixations=1,
            )
            summary = run_within_region_fixation_neural_cross_correlation_sig_xcorr_pairs(
                settings,
                dates=["20990105"],
                show_progress=False,
            )

            self.assertEqual(summary["n_date_signal_runs_total"], 1)
            out_path = (
                analysis_root
                / "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/within_region"
                / "date=20990105"
                / "pair_fixation_lag_mean_significance.pkl"
            )
            self.assertTrue(out_path.exists())
            with out_path.open("rb") as f:
                obj = pickle.load(f)
            df = obj["pair_summaries"]
            self.assertEqual(set(df["condition"].astype(str)), {"face_interactive", "object"})
            self.assertEqual(int(df.loc[df["condition"].astype(str) == "face_interactive", "n_fixations"].iloc[0]), 1)
            self.assertEqual(int(df.loc[df["condition"].astype(str) == "object", "n_fixations"].iloc[0]), 1)

    def test_sig_xcorr_pairs_can_parallelize_across_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            within_root = analysis_root / "ephys/psth/fixation_neural_cross_correlation/within_region"
            for date in ("20990103", "20990104"):
                rows = [
                    {
                        "date": date,
                        "session": "s1",
                        "fixation_category": "face",
                        "interactive_state": "interactive",
                        "is_interactive": True,
                        "region_1": "BLA",
                        "region_2": "BLA",
                        "unit_uuid_1": "u1",
                        "unit_uuid_2": "u2",
                        "cross_correlation": np.asarray([0.1, 0.2, 0.3], dtype=float),
                    },
                    {
                        "date": date,
                        "session": "s1",
                        "fixation_category": "face",
                        "interactive_state": "interactive",
                        "is_interactive": True,
                        "region_1": "BLA",
                        "region_2": "BLA",
                        "unit_uuid_1": "u1",
                        "unit_uuid_2": "u2",
                        "cross_correlation": np.asarray([0.2, 0.3, 0.4], dtype=float),
                    },
                ]
                _write_xcorr_session_file(
                    within_root / f"date={date}" / "session=s1" / "fixations.pkl",
                    analysis_kind="within_region",
                    date=date,
                    session="s1",
                    signal_input_column="spike_train_counts",
                    rows=rows,
                )

            settings = FixationNeuralCrossCorrelationSigXcorrPairsSettings(
                cfg_path=str(cfg_path),
                signal_input_column="spike_train_counts",
                signal_input_columns=("spike_train_counts",),
                min_fixations=2,
                use_parallel=True,
                parallelize_across_dates=True,
            )

            pool_state: dict[str, object] = {}

            class FakePool:
                def __init__(self, processes: int):
                    pool_state["processes"] = int(processes)

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

                def imap_unordered(self, func, iterable, chunksize: int = 1):
                    tasks = list(iterable)
                    pool_state["tasks"] = tasks
                    pool_state["chunksize"] = int(chunksize)
                    for task in reversed(tasks):
                        yield func(task)

            with patch(
                "dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_sig_xcorr_pairs.Pool",
                FakePool,
            ), patch(
                "dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_sig_xcorr_pairs.get_n_processes",
                return_value=2,
            ):
                summary = run_within_region_fixation_neural_cross_correlation_sig_xcorr_pairs(
                    settings,
                    show_progress=False,
                )

            signal_summary = summary["signal_summaries"]["spike_train_counts"]
            self.assertEqual(int(pool_state["processes"]), 2)
            self.assertEqual(int(pool_state["chunksize"]), 1)
            self.assertEqual(len(pool_state["tasks"]), 2)
            self.assertTrue(bool(signal_summary["parallelized_across_dates"]))
            self.assertEqual(int(signal_summary["date_pool_n_procs"]), 2)
            self.assertEqual(int(signal_summary["n_dates_written"]), 2)
            self.assertEqual(int(summary["n_date_signal_runs_total"]), 2)
            self.assertEqual(
                signal_summary["output_paths"],
                [
                    str(
                        analysis_root
                        / "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/within_region"
                        / "date=20990103"
                        / "pair_fixation_lag_mean_significance.pkl"
                    ),
                    str(
                        analysis_root
                        / "ephys/psth/fixation_neural_cross_correlation/pair_meta_analysis/within_region"
                        / "date=20990104"
                        / "pair_fixation_lag_mean_significance.pkl"
                    ),
                ],
            )


    def test_cross_region_sig_xcorr_pairs_use_date_level_outputs(self) -> None:
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

            settings = FixationNeuralCrossCorrelationSigXcorrPairsSettings(
                cfg_path=str(cfg_path),
                signal_input_column="spike_train_counts",
                signal_input_columns=("spike_train_counts",),
                min_fixations=2,
            )
            summary = run_cross_region_fixation_neural_cross_correlation_sig_xcorr_pairs(
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
            self.assertNotIn("mean_cross_correlation", df.columns)
            self.assertNotIn("n_sessions", df.columns)
            self.assertNotIn("sessions", df.columns)

            group_df = obj["group_summaries"]
            self.assertEqual(len(group_df), 1)
            self.assertEqual(str(group_df.iloc[0]["group_label"]), "BLA__ACCg")
            self.assertEqual(int(group_df.iloc[0]["n_total_pairs"]), 1)
            self.assertEqual(int(group_df.iloc[0]["n_sig_object_pairs"]), 0)
            self.assertEqual(int(group_df.iloc[0]["n_sig_any_condition_pairs"]), 0)

            combined_group_df = build_fixation_neural_cross_correlation_sig_xcorr_pair_group_summary_table([out_path])
            self.assertEqual(len(combined_group_df), 1)
            self.assertEqual(str(combined_group_df.iloc[0]["signal_variant"]), "raw")
            self.assertEqual(str(combined_group_df.iloc[0]["signal_input_column"]), "spike_train_counts")


if __name__ == "__main__":
    unittest.main()
