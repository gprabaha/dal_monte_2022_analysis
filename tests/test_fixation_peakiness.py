"""Regression tests for fixation peakiness analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_peakiness import (
    FixationPeakinessSettings,
    run_fixation_peakiness_analysis,
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


def _write_average_fixture(path: Path) -> None:
    bin_centers = np.asarray([-0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35], dtype=float)

    split_rows = [
        {
            "date": "01012020",
            "unit_uuid": "u1",
            "region": "ACC",
            "spike_channel": "ch_u1",
            "recorded_agent": "m1",
            "area": "acc",
            "fixation_category": "face",
            "interactive_state": "interactive",
            "is_interactive": True,
            "n_trials": 12,
            "psth_mean": np.asarray([1.0, 1.0, 1.0, 2.0, 7.0, 2.0, 1.0, 1.0], dtype=float),
        },
        {
            "date": "01012020",
            "unit_uuid": "u1",
            "region": "ACC",
            "spike_channel": "ch_u1",
            "recorded_agent": "m1",
            "area": "acc",
            "fixation_category": "face",
            "interactive_state": "non_interactive",
            "is_interactive": False,
            "n_trials": 11,
            "psth_mean": np.asarray([1.0, 1.0, 1.2, 1.0, 1.4, 1.0, 1.1, 1.0], dtype=float),
        },
        {
            "date": "01012020",
            "unit_uuid": "u2",
            "region": "ACC",
            "spike_channel": "ch_u2",
            "recorded_agent": "m1",
            "area": "acc",
            "fixation_category": "face",
            "interactive_state": "interactive",
            "is_interactive": True,
            "n_trials": 10,
            "psth_mean": np.asarray([1.0, 1.0, 4.0, 1.0, 1.0, 3.8, 1.0, 1.0], dtype=float),
        },
        {
            "date": "01012020",
            "unit_uuid": "u2",
            "region": "ACC",
            "spike_channel": "ch_u2",
            "recorded_agent": "m1",
            "area": "acc",
            "fixation_category": "face",
            "interactive_state": "non_interactive",
            "is_interactive": False,
            "n_trials": 10,
            "psth_mean": np.asarray([1.0, 1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0], dtype=float),
        },
        {
            "date": "01012020",
            "unit_uuid": "u3",
            "region": "OFC",
            "spike_channel": "ch_u3",
            "recorded_agent": "m1",
            "area": "ofc",
            "fixation_category": "face",
            "interactive_state": "interactive",
            "is_interactive": True,
            "n_trials": 9,
            "psth_mean": np.asarray([0.5, 0.5, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5], dtype=float),
        },
        {
            "date": "01012020",
            "unit_uuid": "u3",
            "region": "OFC",
            "spike_channel": "ch_u3",
            "recorded_agent": "m1",
            "area": "ofc",
            "fixation_category": "face",
            "interactive_state": "non_interactive",
            "is_interactive": False,
            "n_trials": 9,
            "psth_mean": np.asarray([0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float),
        },
        {
            "date": "01012020",
            "unit_uuid": "u3",
            "region": "OFC",
            "spike_channel": "ch_u3",
            "recorded_agent": "m1",
            "area": "ofc",
            "fixation_category": "object",
            "interactive_state": "interactive",
            "is_interactive": True,
            "n_trials": 9,
            "psth_mean": np.asarray([10.0, 10.0, 12.0, 16.0, 18.0, 16.0, 12.0, 10.0], dtype=float),
        },
    ]

    unsplit_rows = [
        {
            "date": "01012020",
            "unit_uuid": "u1",
            "region": "ACC",
            "spike_channel": "ch_u1",
            "recorded_agent": "m1",
            "area": "acc",
            "fixation_category": "object",
            "interactive_state": np.nan,
            "is_interactive": np.nan,
            "n_trials": 8,
            "psth_mean": np.asarray([1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.0], dtype=float),
        },
        {
            "date": "01012020",
            "unit_uuid": "u2",
            "region": "ACC",
            "spike_channel": "ch_u2",
            "recorded_agent": "m1",
            "area": "acc",
            "fixation_category": "object",
            "interactive_state": np.nan,
            "is_interactive": np.nan,
            "n_trials": 8,
            "psth_mean": np.asarray([1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0], dtype=float),
        },
        {
            "date": "01012020",
            "unit_uuid": "u3",
            "region": "OFC",
            "spike_channel": "ch_u3",
            "recorded_agent": "m1",
            "area": "ofc",
            "fixation_category": "object",
            "interactive_state": np.nan,
            "is_interactive": np.nan,
            "n_trials": 14,
            "psth_mean": np.asarray([0.5, 0.5, 0.5, 0.6, 3.2, 0.6, 0.5, 0.5], dtype=float),
        },
    ]

    meta = {
        "convert_to_firing_rate_before_average": True,
        "psth_value_kind": "firing_rate_hz",
        "split_meta": {
            "bin_centers_s_rel": bin_centers,
            "output_bin_size_s": 0.1,
            "convert_to_firing_rate_before_average": True,
            "psth_value_kind": "firing_rate_hz",
        },
        "unsplit_meta": {
            "bin_centers_s_rel": bin_centers,
            "output_bin_size_s": 0.1,
            "convert_to_firing_rate_before_average": True,
            "psth_value_kind": "firing_rate_hz",
        },
    }
    save_pickle_path(
        {
            "meta": meta,
            "averages_split_by_interactive_state": pd.DataFrame(split_rows),
            "averages_unsplit_by_interactive_state": pd.DataFrame(unsplit_rows),
        },
        path,
    )


class TestFixationPeakiness(unittest.TestCase):
    """Regression checks for fixation peakiness scoring."""

    def test_peakiness_analysis_scores_units_regions_and_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            avg_path = (
                analysis_root
                / "ephys/psth/fixation_psth_averages"
                / "date=01012020"
                / "fixations_psth_10ms.pkl"
            )
            avg_path.parent.mkdir(parents=True, exist_ok=True)
            _write_average_fixture(avg_path)

            settings = FixationPeakinessSettings(
                cfg_path=str(cfg_path),
                average_input_subdir="ephys/psth/fixation_psth_averages",
                average_input_filename="fixations_psth_10ms.pkl",
                output_subdir="ephys/psth/fixation_peakiness",
                peak_distance_ms=20.0,
                mean_rate_floor_hz=0.5,
                competition_penalty_lambda=0.5,
                region_order=("ACC", "OFC"),
            )
            result = run_fixation_peakiness_analysis(
                settings,
                unit_uuids=("u1", "u3"),
            )

            condition_df = result["condition_peakiness"]
            unit_df = result["unit_peakiness"]
            region_df = result["region_summary"]
            queried_df = result["queried_units"]

            self.assertEqual(len(condition_df), 9)
            self.assertEqual(len(unit_df), 3)
            self.assertEqual(set(region_df["score_scope"].astype(str)), {
                "face_interactive",
                "face_non_interactive",
                "object",
                "any_condition_max",
            })

            u1 = unit_df.loc[unit_df["unit_uuid"].astype(str) == "u1"].iloc[0]
            u2 = unit_df.loc[unit_df["unit_uuid"].astype(str) == "u2"].iloc[0]
            u3 = unit_df.loc[unit_df["unit_uuid"].astype(str) == "u3"].iloc[0]

            self.assertEqual(str(u1["best_condition"]), "face_interactive")
            self.assertEqual(str(u3["best_condition"]), "object")
            self.assertGreater(float(u1["peakiness_score"]), float(u2["peakiness_score"]))
            self.assertGreater(float(u3["object_peakiness_score"]), 0.0)

            u2_face_int = condition_df.loc[
                (condition_df["unit_uuid"].astype(str) == "u2")
                & (condition_df["condition"].astype(str) == "face_interactive")
            ].iloc[0]
            self.assertGreater(float(u2_face_int["second_peak_prominence"]), 0.0)
            self.assertGreater(float(u2_face_int["competition_ratio"]), 0.5)
            self.assertLess(float(u2_face_int["dominance"]), 0.6)

            acc_any = region_df.loc[
                (region_df["region"].astype(str) == "ACC")
                & (region_df["score_scope"].astype(str) == "any_condition_max")
            ].iloc[0]
            self.assertEqual(int(acc_any["n_units"]), 2)
            self.assertEqual(int(acc_any["n_nonzero_score"]), 2)

            self.assertEqual(set(queried_df["unit_uuid"].astype(str)), {"u1", "u3"})

            out_root = analysis_root / "ephys/psth/fixation_peakiness"
            self.assertTrue((out_root / "unit_peakiness.csv").exists())
            self.assertTrue((out_root / "unit_condition_peakiness.csv").exists())
            self.assertTrue((out_root / "region_peakiness_summary.csv").exists())
            self.assertTrue((out_root / "results.pkl").exists())
