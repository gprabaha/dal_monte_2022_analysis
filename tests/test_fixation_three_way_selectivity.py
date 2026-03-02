"""Regression tests for three-way fixation selectivity outputs and plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_selectivity import (
    FixationPSTHSelectivitySettings,
    run_fixation_selectivity_analysis,
)

try:
    from dal_monte_2022_analysis.ephys.plotting.fixation_three_way_selectivity_triangular import (
        FixationThreeWayTriangularPlotSettings,
        plot_fixation_three_way_selectivity_triangular,
    )

    _HAS_TRIANGLE_PLOTTING = True
except ModuleNotFoundError:
    _HAS_TRIANGLE_PLOTTING = False


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


class TestFixationSelectivityThreeWayOutputs(unittest.TestCase):
    """Regression checks for stored three-way per-unit window means."""

    def test_run_selectivity_writes_condition_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            psth_path = processed_root / "date=01012020" / "session=s1" / "psth" / "fixations.pkl"
            psth_path.parent.mkdir(parents=True, exist_ok=True)

            bin_centers = np.asarray([-0.375, -0.125, 0.125, 0.375], dtype=float)
            trial_rows = [
                {
                    "date": "01012020",
                    "session": "s1",
                    "unit_uuid": "u1",
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": "kiki",
                    "area": "acc",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "psth_counts": np.asarray([1.0, 1.0, 4.0, 4.0], dtype=float),
                },
                {
                    "date": "01012020",
                    "session": "s1",
                    "unit_uuid": "u1",
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": "kiki",
                    "area": "acc",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "psth_counts": np.asarray([1.0, 1.0, 5.0, 5.0], dtype=float),
                },
                {
                    "date": "01012020",
                    "session": "s1",
                    "unit_uuid": "u1",
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": "kiki",
                    "area": "acc",
                    "fixation_category": "face",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([2.0, 2.0, 2.0, 2.0], dtype=float),
                },
                {
                    "date": "01012020",
                    "session": "s1",
                    "unit_uuid": "u1",
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": "kiki",
                    "area": "acc",
                    "fixation_category": "face",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([2.0, 2.0, 2.0, 2.0], dtype=float),
                },
                {
                    "date": "01012020",
                    "session": "s1",
                    "unit_uuid": "u1",
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": "kiki",
                    "area": "acc",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([3.0, 3.0, 1.0, 1.0], dtype=float),
                },
                {
                    "date": "01012020",
                    "session": "s1",
                    "unit_uuid": "u1",
                    "region": "ACC",
                    "spike_channel": "ch1",
                    "recorded_agent": "m1",
                    "recorded_monkey": "kiki",
                    "area": "acc",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([3.0, 3.0, 1.0, 1.0], dtype=float),
                },
                {
                    "date": "01012020",
                    "session": "s1",
                    "unit_uuid": "u2",
                    "region": "OFC",
                    "spike_channel": "ch2",
                    "recorded_agent": "m1",
                    "recorded_monkey": "kiki",
                    "area": "ofc",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "psth_counts": np.asarray([2.0, 2.0, 2.0, 2.0], dtype=float),
                },
                {
                    "date": "01012020",
                    "session": "s1",
                    "unit_uuid": "u2",
                    "region": "OFC",
                    "spike_channel": "ch2",
                    "recorded_agent": "m1",
                    "recorded_monkey": "kiki",
                    "area": "ofc",
                    "fixation_category": "face",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=float),
                },
                {
                    "date": "01012020",
                    "session": "s1",
                    "unit_uuid": "u2",
                    "region": "OFC",
                    "spike_channel": "ch2",
                    "recorded_agent": "m1",
                    "recorded_monkey": "kiki",
                    "area": "ofc",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "psth_counts": np.asarray([3.0, 3.0, 3.0, 3.0], dtype=float),
                },
            ]
            trial_df = pd.DataFrame(trial_rows)
            psth_obj = {
                "meta": {"bin_centers_s_rel": bin_centers},
                "trials": trial_df,
            }
            from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path

            save_pickle_path(psth_obj, psth_path)

            settings = FixationPSTHSelectivitySettings(
                cfg_path=str(cfg_path),
                trial_input_modality="psth",
                trial_input_filename="fixations.pkl",
                use_parallel=False,
                min_trials_per_condition=2,
            )
            result = run_fixation_selectivity_analysis(settings)
            condition_df = result.get("condition_summary")

            self.assertIsInstance(condition_df, pd.DataFrame)
            self.assertFalse(condition_df.empty)
            self.assertEqual(set(condition_df["window_name"].astype(str)), {"pre_fix", "peri_fix", "post_fix"})
            self.assertEqual(
                len(condition_df),
                2 * 3,  # two units x three windows
            )

            pre_u1 = condition_df.loc[
                (condition_df["unit_uuid"].astype(str) == "u1")
                & (condition_df["window_name"].astype(str) == "pre_fix")
            ].iloc[0]
            self.assertAlmostEqual(float(pre_u1["mean_fr_face_interactive_hz"]), 4.0, places=6)
            self.assertAlmostEqual(float(pre_u1["mean_fr_face_non_interactive_hz"]), 8.0, places=6)
            self.assertAlmostEqual(float(pre_u1["mean_fr_object_hz"]), 12.0, places=6)
            self.assertAlmostEqual(float(pre_u1["relative_face_interactive"]), 1.0 / 6.0, places=6)
            self.assertAlmostEqual(float(pre_u1["relative_face_non_interactive"]), 2.0 / 6.0, places=6)
            self.assertAlmostEqual(float(pre_u1["relative_object"]), 3.0 / 6.0, places=6)

            out_csv = (
                analysis_root
                / "ephys/psth/fixation_psth_selectivity"
                / "condition_window_means.csv"
            )
            self.assertTrue(out_csv.exists())


@unittest.skipUnless(_HAS_TRIANGLE_PLOTTING, "matplotlib is required for triangular plotting tests")
class TestFixationThreeWayTriangularPlot(unittest.TestCase):
    """Regression checks for triangular population plotting."""

    def test_plot_triangular_population_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            summary_path = (
                analysis_root
                / "ephys/psth/fixation_psth_selectivity"
                / "condition_window_means.csv"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            rows = []
            for region in ("ACC", "OFC"):
                for window_name, start, stop in (
                    ("pre_fix", -500.0, 0.0),
                    ("post_fix", 0.0, 500.0),
                ):
                    rows.extend(
                        [
                            {
                                "unit_key": f"01012020|{region}|u1|{window_name}",
                                "unit_uuid": f"{region}_u1",
                                "region": region,
                                "window_name": window_name,
                                "window_start_ms": start,
                                "window_stop_ms": stop,
                                "mean_fr_face_interactive_hz": 4.0,
                                "mean_fr_face_non_interactive_hz": 2.0,
                                "mean_fr_object_hz": 1.0,
                            },
                            {
                                "unit_key": f"01012020|{region}|u2|{window_name}",
                                "unit_uuid": f"{region}_u2",
                                "region": region,
                                "window_name": window_name,
                                "window_start_ms": start,
                                "window_stop_ms": stop,
                                "mean_fr_face_interactive_hz": 1.0,
                                "mean_fr_face_non_interactive_hz": 2.0,
                                "mean_fr_object_hz": 4.0,
                            },
                        ]
                    )
            pd.DataFrame(rows).to_csv(summary_path, index=False)

            settings = FixationThreeWayTriangularPlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_selectivity",
                condition_summary_filename="condition_window_means.csv",
                output_subdir="ephys/psth/fixation_psth_selectivity_triangular",
                output_filename="triangular_test",
                output_extension="png",
                output_dpi=110,
                min_units_per_panel=1,
            )
            result = plot_fixation_three_way_selectivity_triangular(settings)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(len(result["regions"]), 2)
            self.assertEqual(len(result["windows"]), 2)
            self.assertEqual(len(result["panel_counts"]), 4)
            self.assertTrue(Path(result["output_path"]).exists())


if __name__ == "__main__":
    unittest.main()

