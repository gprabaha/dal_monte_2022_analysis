"""Regression tests for fixation PSTH average SEM storage and plot trace source."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dal_monte_2022_analysis.ephys.features.fixation_psth import (
        FixationPSTHAverageSettings,
        build_fixation_psth_averages_for_date,
    )
    from dal_monte_2022_analysis.ephys.plotting.fixation_psth import (
        FixationPSTHUnitPlotSettings,
        _build_unit_condition_payloads,
    )

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
class TestFixationPSTHAverageSemAndPlotSource(unittest.TestCase):
    """Checks that average outputs include SEM and plotting uses precomputed traces."""

    def test_average_builder_stores_psth_sem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trial_path = root / "trials.pkl"

            trials_df = pd.DataFrame(
                [
                    {
                        "unit_uuid": "u1",
                        "fixation_category": "face",
                        "interactive_state": "interactive",
                        "psth_counts": np.asarray([1.0, 3.0], dtype=float),
                    },
                    {
                        "unit_uuid": "u1",
                        "fixation_category": "face",
                        "interactive_state": "interactive",
                        "psth_counts": np.asarray([5.0, 7.0], dtype=float),
                    },
                ]
            )
            obj = {
                "meta": {
                    "bin_centers_s_rel": np.asarray([-0.05, 0.05], dtype=float),
                    "bin_size_s": 0.1,
                },
                "trials": trials_df,
            }
            with trial_path.open("wb") as f:
                pickle.dump(obj, f)

            settings = FixationPSTHAverageSettings(
                cfg_path=str(root / "dataset.yaml"),
                smooth_before_average=False,
                split_by_interactive_state=True,
                categories=("face",),
            )
            result = build_fixation_psth_averages_for_date(settings, "20990101", [trial_path])

            self.assertIsNotNone(result)
            assert result is not None
            averages_df = result["averages"]
            self.assertIn("psth_mean", averages_df.columns)
            self.assertIn("psth_sem", averages_df.columns)
            self.assertEqual(len(averages_df), 1)

            row = averages_df.iloc[0]
            mean_vec = np.asarray(row["psth_mean"], dtype=float)
            sem_vec = np.asarray(row["psth_sem"], dtype=float)
            self.assertTrue(np.allclose(mean_vec, np.asarray([3.0, 5.0], dtype=float)))
            self.assertTrue(np.allclose(sem_vec, np.asarray([2.0, 2.0], dtype=float)))

    def test_plot_payloads_use_precomputed_average_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            avg_root = analysis_root / "ephys/psth/fixation_psth_averages/date=20990101"
            avg_root.mkdir(parents=True, exist_ok=True)
            split_avg_df = pd.DataFrame(
                [
                    {
                        "date": "20990101",
                        "unit_uuid": "u1",
                        "fixation_category": "face",
                        "interactive_state": "interactive",
                        "n_trials": 10,
                        "psth_mean": np.asarray([10.0, 20.0, 30.0], dtype=float),
                        "psth_sem": np.asarray([1.0, 2.0, 3.0], dtype=float),
                    },
                    {
                        "date": "20990101",
                        "unit_uuid": "u1",
                        "fixation_category": "face",
                        "interactive_state": "non_interactive",
                        "n_trials": 8,
                        "psth_mean": np.asarray([11.0, 21.0, 31.0], dtype=float),
                        "psth_sem": np.asarray([1.1, 2.1, 3.1], dtype=float),
                    },
                    {
                        "date": "20990101",
                        "unit_uuid": "u1",
                        "fixation_category": "object",
                        "interactive_state": "non_interactive",
                        "n_trials": 6,
                        "psth_mean": np.asarray([90.0, 91.0, 92.0], dtype=float),
                        "psth_sem": np.asarray([9.0, 9.1, 9.2], dtype=float),
                    },
                ]
            )
            unsplit_avg_df = pd.DataFrame(
                [
                    {
                        "date": "20990101",
                        "unit_uuid": "u1",
                        "fixation_category": "face",
                        "n_trials": 9,
                        "psth_mean": np.asarray([99.0, 99.0, 99.0], dtype=float),
                        "psth_sem": np.asarray([9.9, 9.9, 9.9], dtype=float),
                    },
                    {
                        "date": "20990101",
                        "unit_uuid": "u1",
                        "fixation_category": "object",
                        "n_trials": 6,
                        "psth_mean": np.asarray([12.0, 22.0, 32.0], dtype=float),
                        "psth_sem": np.asarray([1.2, 2.2, 3.2], dtype=float),
                    },
                ]
            )
            with (avg_root / "fixations_split.pkl").open("wb") as f:
                pickle.dump(
                    {
                        "meta": {
                            "date": "20990101",
                            "bin_centers_s_rel": np.asarray([-0.5, 0.0, 0.5], dtype=float),
                            "target_bin_size_s": 0.5,
                        },
                        "averages": split_avg_df,
                    },
                    f,
                )
            with (avg_root / "fixations_unsplit.pkl").open("wb") as f:
                pickle.dump(
                    {
                        "meta": {
                            "date": "20990101",
                            "bin_centers_s_rel": np.asarray([-0.5, 0.0, 0.5], dtype=float),
                            "target_bin_size_s": 0.5,
                        },
                        "averages": unsplit_avg_df,
                    },
                    f,
                )

            trial_df = pd.DataFrame(
                [
                    {
                        "unit_uuid": "u1",
                        "fixation_category": "face",
                        "interactive_state": "interactive",
                        "psth_counts": np.asarray([1.0, 1.0, 1.0], dtype=float),
                    },
                    {
                        "unit_uuid": "u1",
                        "fixation_category": "face",
                        "interactive_state": "non_interactive",
                        "psth_counts": np.asarray([2.0, 2.0, 2.0], dtype=float),
                    },
                    {
                        "unit_uuid": "u1",
                        "fixation_category": "object",
                        "interactive_state": "non_interactive",
                        "psth_counts": np.asarray([3.0, 3.0, 3.0], dtype=float),
                    },
                ]
            )

            settings = FixationPSTHUnitPlotSettings(
                cfg_path=str(cfg_path),
                smooth_before_average=False,
                use_precomputed_average_traces=True,
                average_trace_input_subdir="ephys/psth/fixation_psth_averages",
                average_trace_input_filename="fixations_split.pkl",
                average_trace_object_input_subdir="ephys/psth/fixation_psth_averages",
                average_trace_object_input_filename="fixations_unsplit.pkl",
                allow_trial_trace_fallback=True,
            )
            payloads = _build_unit_condition_payloads(
                trial_df,
                unit_key="20990101|u1",
                bin_centers=np.asarray([-0.5, 0.0, 0.5], dtype=float),
                bin_size_s=1.0,
                settings=settings,
            )

            by_key = {str(payload["key"]): payload for payload in payloads}
            self.assertEqual(set(by_key), {"face_interactive", "face_non_interactive", "object"})

            expected = {
                "face_interactive": (
                    np.asarray([20.0, 40.0, 60.0], dtype=float),
                    np.asarray([2.0, 4.0, 6.0], dtype=float),
                ),
                "face_non_interactive": (
                    np.asarray([22.0, 42.0, 62.0], dtype=float),
                    np.asarray([2.2, 4.2, 6.2], dtype=float),
                ),
                "object": (
                    np.asarray([24.0, 44.0, 64.0], dtype=float),
                    np.asarray([2.4, 4.4, 6.4], dtype=float),
                ),
            }
            for cond, (mean_expected, sem_expected) in expected.items():
                payload = by_key[cond]
                self.assertTrue(np.allclose(np.asarray(payload["mean_hz"], dtype=float), mean_expected))
                self.assertTrue(np.allclose(np.asarray(payload["sem_hz"], dtype=float), sem_expected))
                self.assertEqual(len(payload["spike_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
