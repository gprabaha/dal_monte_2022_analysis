"""Regression tests for fixation preference-index analysis outputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_preference_index import (
    FixationPSTHPreferenceIndexSettings,
    run_fixation_preference_index_analysis,
)
from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path


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


class TestFixationPreferenceIndex(unittest.TestCase):
    """Regression checks for pairwise per-bin preference-index outputs."""

    def test_preference_index_formula_and_significance_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_date = "20990101"
            dummy_session = "1"
            dummy_unit = "unit_001"
            dummy_monkey = "dummy_monkey_001"
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            psth_path = (
                processed_root
                / f"date={dummy_date}"
                / f"session={dummy_session}"
                / "psth"
                / "fixations.pkl"
            )
            psth_path.parent.mkdir(parents=True, exist_ok=True)

            bin_centers = np.asarray([-0.625, -0.375, -0.125, 0.125, 0.375, 0.625], dtype=float)
            trial_rows = [
                {
                    "date": dummy_date,
                    "session": dummy_session,
                    "unit_uuid": dummy_unit,
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "acc",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "psth_counts": np.asarray([9.0, 1.0, 1.0, 4.0, 4.0, 9.0], dtype=float),
                },
                {
                    "date": dummy_date,
                    "session": dummy_session,
                    "unit_uuid": dummy_unit,
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "acc",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "psth_counts": np.asarray([9.0, 1.0, 1.0, 5.0, 5.0, 9.0], dtype=float),
                },
                {
                    "date": dummy_date,
                    "session": dummy_session,
                    "unit_uuid": dummy_unit,
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "acc",
                    "fixation_category": "face",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([9.0, 2.0, 2.0, 2.0, 2.0, 9.0], dtype=float),
                },
                {
                    "date": dummy_date,
                    "session": dummy_session,
                    "unit_uuid": dummy_unit,
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "acc",
                    "fixation_category": "face",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([9.0, 2.0, 2.0, 2.0, 2.0, 9.0], dtype=float),
                },
                {
                    "date": dummy_date,
                    "session": dummy_session,
                    "unit_uuid": dummy_unit,
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "acc",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([9.0, 3.0, 3.0, 1.0, 1.0, 9.0], dtype=float),
                },
                {
                    "date": dummy_date,
                    "session": dummy_session,
                    "unit_uuid": dummy_unit,
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "acc",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([9.0, 3.0, 3.0, 1.0, 1.0, 9.0], dtype=float),
                },
            ]
            psth_obj = {
                "meta": {"bin_centers_s_rel": bin_centers},
                "trials": pd.DataFrame(trial_rows),
            }
            save_pickle_path(psth_obj, psth_path)

            sel_root = analysis_root / "ephys/psth/fixation_psth_selectivity"
            sel_root.mkdir(parents=True, exist_ok=True)
            pair_df = pd.DataFrame(
                [
                    {
                        "unit_key": f"{dummy_date}|{dummy_unit}",
                        "pair_label": "face_interactive__vs__face_non_interactive",
                        "is_selective_pair": True,
                    },
                    {
                        "unit_key": f"{dummy_date}|{dummy_unit}",
                        "pair_label": "face_interactive__vs__object",
                        "is_selective_pair": False,
                    },
                    {
                        "unit_key": f"{dummy_date}|{dummy_unit}",
                        "pair_label": "face_non_interactive__vs__object",
                        "is_selective_pair": True,
                    },
                ]
            )
            pair_df.to_csv(sel_root / "pair_selectivity.csv", index=False)
            unit_df = pd.DataFrame(
                [
                    {
                        "unit_key": f"{dummy_date}|{dummy_unit}",
                        "is_selective_unit": True,
                        "selective_pairs": (
                            "face_interactive__vs__face_non_interactive|"
                            "face_non_interactive__vs__object"
                        ),
                    }
                ]
            )
            unit_df.to_csv(sel_root / "unit_selectivity.csv", index=False)

            settings = FixationPSTHPreferenceIndexSettings(
                cfg_path=str(cfg_path),
                trial_input_modality="psth",
                trial_input_filename="fixations.pkl",
                selectivity_input_subdir="ephys/psth/fixation_psth_selectivity",
                output_subdir="ephys/psth/fixation_psth_preference_index",
                use_parallel=False,
            )
            result = run_fixation_preference_index_analysis(settings)
            out_df = result.get("timeseries")

            self.assertIsInstance(out_df, pd.DataFrame)
            self.assertFalse(out_df.empty)
            self.assertEqual(
                set(out_df["pair_label"].astype(str)),
                {
                    "face_interactive__vs__face_non_interactive",
                    "face_interactive__vs__object",
                    "face_non_interactive__vs__object",
                },
            )
            self.assertEqual(len(out_df), 3 * 4)
            self.assertTrue((out_df["bin_center_s"].astype(float) >= -0.5).all())
            self.assertTrue((out_df["bin_center_s"].astype(float) <= 0.5).all())

            row = out_df.loc[
                (out_df["unit_key"].astype(str) == f"{dummy_date}|{dummy_unit}")
                & (out_df["pair_label"].astype(str) == "face_interactive__vs__object")
                & np.isclose(out_df["bin_center_s"].astype(float), -0.375)
            ].iloc[0]
            self.assertAlmostEqual(float(row["preference_index"]), -8.0 / 22.0, places=6)
            self.assertEqual(str(row["index_name"]), "interactive_face_vs_object_index")
            self.assertEqual(str(row["normalization_mode"]), "unit_max_sum")
            self.assertFalse(bool(row["is_selective_pair"]))
            self.assertTrue(bool(row["is_selective_unit"]))
            self.assertTrue(bool(row["is_selective_any_pair"]))
            self.assertAlmostEqual(float(row["unit_pair_max_sum_fr_hz"]), 22.0, places=6)
            self.assertAlmostEqual(float(row["normalization_denominator_hz"]), 22.0, places=6)
            self.assertAlmostEqual(float(row["normalization_denominator_unit_max_sum_hz"]), 22.0, places=6)
            self.assertAlmostEqual(float(row["normalization_denominator_per_bin_sum_hz"]), 16.0, places=6)
            self.assertAlmostEqual(float(row["preference_index_unit_max_sum"]), -8.0 / 22.0, places=6)
            self.assertAlmostEqual(float(row["preference_index_per_bin_sum"]), -0.5, places=6)
            self.assertTrue(bool(row["index_valid_unit_max_sum"]))
            self.assertTrue(bool(row["index_valid_per_bin_sum"]))
            self.assertAlmostEqual(float(row["sum_fr_hz"]), 16.0, places=6)

            settings_per_bin = FixationPSTHPreferenceIndexSettings(
                cfg_path=str(cfg_path),
                trial_input_modality="psth",
                trial_input_filename="fixations.pkl",
                selectivity_input_subdir="ephys/psth/fixation_psth_selectivity",
                output_subdir="ephys/psth/fixation_psth_preference_index_per_bin",
                normalization_mode="per_bin_sum",
                use_parallel=False,
            )
            result_per_bin = run_fixation_preference_index_analysis(settings_per_bin)
            out_df_per_bin = result_per_bin.get("timeseries")
            self.assertIsInstance(out_df_per_bin, pd.DataFrame)
            self.assertFalse(out_df_per_bin.empty)
            row_per_bin = out_df_per_bin.loc[
                (out_df_per_bin["unit_key"].astype(str) == f"{dummy_date}|{dummy_unit}")
                & (out_df_per_bin["pair_label"].astype(str) == "face_interactive__vs__object")
                & np.isclose(out_df_per_bin["bin_center_s"].astype(float), -0.375)
            ].iloc[0]
            self.assertAlmostEqual(float(row_per_bin["preference_index"]), -0.5, places=6)
            self.assertEqual(str(row_per_bin["normalization_mode"]), "per_bin_sum")
            self.assertAlmostEqual(float(row_per_bin["normalization_denominator_hz"]), 16.0, places=6)
            self.assertAlmostEqual(float(row_per_bin["normalization_denominator_unit_max_sum_hz"]), 22.0, places=6)
            self.assertAlmostEqual(float(row_per_bin["normalization_denominator_per_bin_sum_hz"]), 16.0, places=6)
            self.assertAlmostEqual(float(row_per_bin["preference_index_unit_max_sum"]), -8.0 / 22.0, places=6)
            self.assertAlmostEqual(float(row_per_bin["preference_index_per_bin_sum"]), -0.5, places=6)
            self.assertTrue(bool(row_per_bin["index_valid_unit_max_sum"]))
            self.assertTrue(bool(row_per_bin["index_valid_per_bin_sum"]))

            out_csv = (
                analysis_root
                / "ephys/psth/fixation_psth_preference_index"
                / "preference_index_timeseries.csv"
            )
            self.assertTrue(out_csv.exists())

    def test_average_input_uses_explicit_bin_duration_for_hz_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_date = "20990102"
            dummy_unit = "unit_002"
            dummy_monkey = "dummy_monkey_002"
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            avg_path = (
                analysis_root
                / "ephys/psth/fixation_psth_index_averages"
                / f"date={dummy_date}"
                / "fixations.pkl"
            )
            avg_path.parent.mkdir(parents=True, exist_ok=True)

            bin_centers = np.asarray([-0.025, 0.0, 0.025], dtype=float)  # 25 ms stride
            avg_rows = [
                {
                    "date": dummy_date,
                    "unit_uuid": dummy_unit,
                    "region": "BLA",
                    "spike_channel": "ch2",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "bla",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "n_trials": 3,
                    "psth_mean": np.asarray([10.0, 10.0, 10.0], dtype=float),
                },
                {
                    "date": dummy_date,
                    "unit_uuid": dummy_unit,
                    "region": "BLA",
                    "spike_channel": "ch2",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "bla",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "n_trials": 1,
                    "psth_mean": np.asarray([20.0, 20.0, 20.0], dtype=float),
                },
                {
                    "date": dummy_date,
                    "unit_uuid": dummy_unit,
                    "region": "BLA",
                    "spike_channel": "ch2",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "bla",
                    "fixation_category": "face",
                    "interactive_state": "non_interactive",
                    "n_trials": 4,
                    "psth_mean": np.asarray([8.0, 8.0, 8.0], dtype=float),
                },
                {
                    "date": dummy_date,
                    "unit_uuid": dummy_unit,
                    "region": "BLA",
                    "spike_channel": "ch2",
                    "recorded_agent": "m1",
                    "recorded_monkey": dummy_monkey,
                    "area": "bla",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "n_trials": 4,
                    "psth_mean": np.asarray([6.0, 6.0, 6.0], dtype=float),
                },
            ]
            avg_obj = {
                "meta": {
                    "bin_centers_s_rel": bin_centers,
                    "target_bin_size_s": 0.05,  # 50 ms bin width
                    "target_bin_step_s": 0.025,
                },
                "averages": pd.DataFrame(avg_rows),
            }
            save_pickle_path(avg_obj, avg_path)

            sel_root = analysis_root / "ephys/psth/fixation_psth_selectivity"
            sel_root.mkdir(parents=True, exist_ok=True)
            pair_df = pd.DataFrame(
                [
                    {
                        "unit_key": f"{dummy_date}|{dummy_unit}",
                        "pair_label": "face_interactive__vs__face_non_interactive",
                        "is_selective_pair": True,
                    },
                    {
                        "unit_key": f"{dummy_date}|{dummy_unit}",
                        "pair_label": "face_interactive__vs__object",
                        "is_selective_pair": True,
                    },
                    {
                        "unit_key": f"{dummy_date}|{dummy_unit}",
                        "pair_label": "face_non_interactive__vs__object",
                        "is_selective_pair": False,
                    },
                ]
            )
            pair_df.to_csv(sel_root / "pair_selectivity.csv", index=False)
            unit_df = pd.DataFrame(
                [
                    {
                        "unit_key": f"{dummy_date}|{dummy_unit}",
                        "is_selective_unit": True,
                        "selective_pairs": (
                            "face_interactive__vs__face_non_interactive|"
                            "face_interactive__vs__object"
                        ),
                    }
                ]
            )
            unit_df.to_csv(sel_root / "unit_selectivity.csv", index=False)

            settings = FixationPSTHPreferenceIndexSettings(
                cfg_path=str(cfg_path),
                average_input_subdir="ephys/psth/fixation_psth_index_averages",
                average_input_filename="fixations.pkl",
                trial_input_modality="psth",
                trial_input_filename="fixations.pkl",
                selectivity_input_subdir="ephys/psth/fixation_psth_selectivity",
                output_subdir="ephys/psth/fixation_psth_preference_index_avg",
                use_parallel=False,
            )
            result = run_fixation_preference_index_analysis(settings, dates=[dummy_date])
            out_df = result.get("timeseries")

            self.assertIsInstance(out_df, pd.DataFrame)
            self.assertFalse(out_df.empty)
            self.assertEqual(int(len(out_df)), 3 * 3)
            self.assertEqual(str(result["meta"]["input_source"]), "average")
            self.assertAlmostEqual(float(result["meta"]["bin_duration_s"]), 0.05, places=9)
            self.assertAlmostEqual(float(result["meta"]["bin_stride_s"]), 0.025, places=9)

            row = out_df.loc[
                (out_df["unit_key"].astype(str) == f"{dummy_date}|{dummy_unit}")
                & (out_df["pair_label"].astype(str) == "face_interactive__vs__object")
                & np.isclose(out_df["bin_center_s"].astype(float), 0.0)
            ].iloc[0]
            self.assertEqual(int(row["n_trials_a"]), 4)
            self.assertEqual(int(row["n_trials_b"]), 4)
            self.assertAlmostEqual(float(row["mean_fr_a_hz"]), 250.0, places=6)
            self.assertAlmostEqual(float(row["mean_fr_b_hz"]), 120.0, places=6)
            self.assertAlmostEqual(float(row["difference_fr_hz"]), 130.0, places=6)
            self.assertAlmostEqual(float(row["sum_fr_hz"]), 370.0, places=6)
            self.assertAlmostEqual(float(row["unit_pair_max_sum_fr_hz"]), 370.0, places=6)
            self.assertAlmostEqual(float(row["normalization_denominator_hz"]), 370.0, places=6)


if __name__ == "__main__":
    unittest.main()
