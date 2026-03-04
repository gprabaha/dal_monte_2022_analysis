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

            bin_centers = np.asarray([-0.375, -0.125, 0.125, 0.375], dtype=float)
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
                    "psth_counts": np.asarray([1.0, 1.0, 4.0, 4.0], dtype=float),
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
                    "psth_counts": np.asarray([1.0, 1.0, 5.0, 5.0], dtype=float),
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
                    "psth_counts": np.asarray([2.0, 2.0, 2.0, 2.0], dtype=float),
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
                    "psth_counts": np.asarray([2.0, 2.0, 2.0, 2.0], dtype=float),
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
                    "psth_counts": np.asarray([3.0, 3.0, 1.0, 1.0], dtype=float),
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
                    "psth_counts": np.asarray([3.0, 3.0, 1.0, 1.0], dtype=float),
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
            self.assertEqual(len(out_df), 3 * len(bin_centers))

            row = out_df.loc[
                (out_df["unit_key"].astype(str) == f"{dummy_date}|{dummy_unit}")
                & (out_df["pair_label"].astype(str) == "face_interactive__vs__object")
                & (out_df["bin_index"].astype(int) == 0)
            ].iloc[0]
            self.assertAlmostEqual(float(row["preference_index"]), -0.5, places=6)
            self.assertEqual(str(row["index_name"]), "interactive_face_vs_object_index")
            self.assertFalse(bool(row["is_selective_pair"]))
            self.assertTrue(bool(row["is_selective_unit"]))
            self.assertTrue(bool(row["is_selective_any_pair"]))

            out_csv = (
                analysis_root
                / "ephys/psth/fixation_psth_preference_index"
                / "preference_index_timeseries.csv"
            )
            self.assertTrue(out_csv.exists())


if __name__ == "__main__":
    unittest.main()
