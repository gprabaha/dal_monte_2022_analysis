"""Regression tests for fixation PSTH variability analysis and plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_psth_variability import (
    FixationPSTHVariabilitySettings,
    run_fixation_psth_variability_analysis,
)
from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path

try:
    from dal_monte_2022_analysis.ephys.plotting.fixation_psth_variability import (
        FixationPSTHVariabilityPlotSettings,
        plot_fixation_psth_variability_violins,
        sns as _sns,
    )

    _HAS_VARIABILITY_PLOT = _sns is not None
except ModuleNotFoundError:
    _HAS_VARIABILITY_PLOT = False


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


class TestFixationPSTHVariabilityAnalysis(unittest.TestCase):
    """Checks per-unit variability summary construction and region stats."""

    def test_run_variability_analysis_from_average_psths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            avg_root = analysis_root / "ephys/psth/fixation_psth_averages" / "date=20990101"
            avg_root.mkdir(parents=True, exist_ok=True)
            split_path = avg_root / "fixations_psth_10ms_split_by_interactive_state.pkl"
            object_path = avg_root / "fixations_psth_10ms_unsplit_by_interactive_state.pkl"

            bin_centers = np.asarray([-0.01, 0.0, 0.01], dtype=float)
            split_rows = []
            object_rows = []
            bla_specs = [
                ("u1", 1.0, 2.0, 5.0),
                ("u2", 1.2, 2.2, 5.2),
                ("u3", 0.9, 1.9, 4.9),
                ("u4", 1.1, 2.1, 5.1),
            ]
            ofc_specs = [
                ("u10", 2.0, 2.2, 2.4),
                ("u11", 1.9, 2.1, 2.3),
            ]

            for region, specs in (("BLA", bla_specs), ("OFC", ofc_specs)):
                for unit_uuid, d_int, d_non, d_obj in specs:
                    split_rows.extend(
                        [
                            {
                                "date": "20990101",
                                "unit_uuid": unit_uuid,
                                "region": region,
                                "fixation_category": "face",
                                "interactive_state": "interactive",
                                "n_trials": 12,
                                "psth_mean": np.asarray([10.0 - d_int, 10.0, 10.0 + d_int], dtype=float),
                            },
                            {
                                "date": "20990101",
                                "unit_uuid": unit_uuid,
                                "region": region,
                                "fixation_category": "face",
                                "interactive_state": "non_interactive",
                                "n_trials": 12,
                                "psth_mean": np.asarray([12.0 - d_non, 12.0, 12.0 + d_non], dtype=float),
                            },
                        ]
                    )
                    object_rows.append(
                        {
                            "date": "20990101",
                            "unit_uuid": unit_uuid,
                            "region": region,
                            "fixation_category": "object",
                            "n_trials": 10,
                            "psth_mean": np.asarray([8.0 - d_obj, 8.0, 8.0 + d_obj], dtype=float),
                        }
                    )

            save_pickle_path(
                {"averages": pd.DataFrame(split_rows), "meta": {"bin_centers_s_rel": bin_centers}},
                split_path,
            )
            save_pickle_path(
                {"averages": pd.DataFrame(object_rows), "meta": {"bin_centers_s_rel": bin_centers}},
                object_path,
            )

            settings = FixationPSTHVariabilitySettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_averages",
                input_filename="fixations_psth_10ms_split_by_interactive_state.pkl",
                object_input_subdir="ephys/psth/fixation_psth_averages",
                object_input_filename="fixations_psth_10ms_unsplit_by_interactive_state.pkl",
                output_subdir="ephys/psth/fixation_psth_variability",
                alpha=0.05,
                pvalue_correction="fdr_bh",
                verbose_logging=False,
            )
            result = run_fixation_psth_variability_analysis(settings)

            unit_df = result["unit_summary"]
            stats_df = result["within_region_stats"]
            self.assertEqual(len(unit_df), 6)
            self.assertFalse(stats_df.empty)
            self.assertTrue(Path(result["unit_summary_path"]).exists())
            self.assertTrue(Path(result["within_region_stats_path"]).exists())
            self.assertIsNone(result["pickle_path"])

            expected_columns = {
                "date",
                "unit_uuid",
                "unit_key",
                "region",
                "face_interactive_variability",
                "face_non_interactive_variability",
                "object_variability",
            }
            self.assertTrue(expected_columns.issubset(set(unit_df.columns)))

            row = unit_df.loc[unit_df["unit_uuid"].astype(str) == "u1"].iloc[0]
            self.assertAlmostEqual(float(row["face_interactive_variability"]), 1.0, places=6)
            self.assertAlmostEqual(float(row["face_non_interactive_variability"]), 2.0, places=6)
            self.assertAlmostEqual(float(row["object_variability"]), 5.0, places=6)

            bla_pair = stats_df.loc[
                (stats_df["region"].astype(str) == "BLA")
                & (stats_df["condition_a"].astype(str) == "face_interactive")
                & (stats_df["condition_b"].astype(str) == "object")
            ].iloc[0]
            self.assertGreater(int(bla_pair["n_units_paired"]), 3)
            self.assertTrue(bool(bla_pair["significant_adjusted"]))

            saved_unit_df = pd.read_csv(result["unit_summary_path"])
            saved_stats_df = pd.read_csv(result["within_region_stats_path"])
            self.assertEqual(len(saved_unit_df), 6)
            self.assertFalse(saved_stats_df.empty)


    def test_run_variability_analysis_from_combined_average_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            avg_root = analysis_root / "ephys/psth/fixation_psth_averages" / "date=20990101"
            avg_root.mkdir(parents=True, exist_ok=True)
            combined_path = avg_root / "fixations_psth_10ms.pkl"

            bin_centers = np.asarray([-0.01, 0.0, 0.01], dtype=float)
            split_rows = [
                {
                    "date": "20990101",
                    "unit_uuid": "u1",
                    "region": "BLA",
                    "fixation_category": "face",
                    "interactive_state": "interactive",
                    "is_interactive": True,
                    "n_trials": 12,
                    "psth_mean": np.asarray([9.0, 10.0, 11.0], dtype=float),
                },
                {
                    "date": "20990101",
                    "unit_uuid": "u1",
                    "region": "BLA",
                    "fixation_category": "face",
                    "interactive_state": "non_interactive",
                    "is_interactive": False,
                    "n_trials": 12,
                    "psth_mean": np.asarray([10.0, 12.0, 14.0], dtype=float),
                },
                {
                    "date": "20990101",
                    "unit_uuid": "u1",
                    "region": "BLA",
                    "fixation_category": "object",
                    "interactive_state": "interactive",
                    "is_interactive": True,
                    "n_trials": 6,
                    "psth_mean": np.asarray([3.0, 8.0, 13.0], dtype=float),
                },
                {
                    "date": "20990101",
                    "unit_uuid": "u1",
                    "region": "BLA",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "is_interactive": False,
                    "n_trials": 4,
                    "psth_mean": np.asarray([2.0, 8.0, 14.0], dtype=float),
                },
            ]
            unsplit_rows = [
                {
                    "date": "20990101",
                    "unit_uuid": "u1",
                    "region": "BLA",
                    "fixation_category": "object",
                    "n_trials": 10,
                    "psth_mean": np.asarray([3.4, 8.0, 12.6], dtype=float),
                }
            ]

            save_pickle_path(
                {
                    "meta": {
                        "store_split_and_unsplit_together": True,
                        "split_meta": {"bin_centers_s_rel": bin_centers},
                        "unsplit_meta": {"bin_centers_s_rel": bin_centers},
                    },
                    "averages_split_by_interactive_state": pd.DataFrame(split_rows),
                    "averages_unsplit_by_interactive_state": pd.DataFrame(unsplit_rows),
                },
                combined_path,
            )

            settings = FixationPSTHVariabilitySettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_averages",
                input_filename="fixations_psth_10ms.pkl",
                object_input_subdir="ephys/psth/fixation_psth_averages",
                object_input_filename="fixations_psth_10ms.pkl",
                output_subdir="ephys/psth/fixation_psth_variability",
                min_paired_units_per_region=99,
                verbose_logging=False,
            )
            result = run_fixation_psth_variability_analysis(settings)

            unit_df = result["unit_summary"]
            self.assertEqual(len(unit_df), 1)
            row = unit_df.iloc[0]
            self.assertAlmostEqual(float(row["face_interactive_variability"]), 1.0, places=6)
            self.assertAlmostEqual(float(row["face_non_interactive_variability"]), 2.0, places=6)
            self.assertAlmostEqual(float(row["object_variability"]), 4.6, places=6)
            self.assertTrue(result["within_region_stats"].empty)

    def test_empty_outputs_write_header_only_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            settings = FixationPSTHVariabilitySettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_averages",
                input_filename="fixations_psth_10ms_split_by_interactive_state.pkl",
                object_input_subdir="ephys/psth/fixation_psth_averages",
                object_input_filename="fixations_psth_10ms_unsplit_by_interactive_state.pkl",
                output_subdir="ephys/psth/fixation_psth_variability",
                verbose_logging=False,
            )
            result = run_fixation_psth_variability_analysis(settings)

            self.assertTrue(result["unit_summary"].empty)
            self.assertTrue(result["within_region_stats"].empty)
            saved_unit_df = pd.read_csv(result["unit_summary_path"])
            saved_stats_df = pd.read_csv(result["within_region_stats_path"])
            self.assertTrue(saved_unit_df.empty)
            self.assertTrue(saved_stats_df.empty)
            self.assertIn("face_interactive_variability", saved_unit_df.columns)
            self.assertIn("p_value_adjusted", saved_stats_df.columns)


@unittest.skipUnless(
    _HAS_VARIABILITY_PLOT,
    "matplotlib and seaborn are required for fixation PSTH variability plotting tests",
)
class TestFixationPSTHVariabilityPlot(unittest.TestCase):
    """Checks violin rendering from saved summary/stat tables."""

    def test_plot_variability_violins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            in_root = analysis_root / "ephys/psth/fixation_psth_variability"
            in_root.mkdir(parents=True, exist_ok=True)

            summary_rows = []
            for region, base in zip(("BLA", "ACCg", "dmPFC", "OFC"), (1.0, 1.5, 2.0, 2.5)):
                for idx in range(4):
                    summary_rows.append(
                        {
                            "date": "20990101",
                            "unit_uuid": f"{region}_u{idx}",
                            "unit_key": f"20990101|{region}|u{idx}",
                            "region": region,
                            "face_interactive_variability": base + 0.05 * idx,
                            "face_non_interactive_variability": base + 0.45 + 0.05 * idx,
                            "object_variability": base + 0.90 + 0.05 * idx,
                        }
                    )
            stats_rows = []
            for region in ("BLA", "ACCg", "dmPFC", "OFC"):
                stats_rows.append(
                    {
                        "region": region,
                        "condition_a": "face_interactive",
                        "condition_b": "object",
                        "p_value_adjusted": 0.01,
                        "significant_adjusted": True,
                    }
                )

            pd.DataFrame(summary_rows).to_csv(in_root / "unit_condition_variability.csv", index=False)
            pd.DataFrame(stats_rows).to_csv(
                in_root / "within_region_condition_variability_stats.csv",
                index=False,
            )

            settings = FixationPSTHVariabilityPlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_variability",
                output_subdir="ephys/psth/fixation_psth_variability/plots",
                output_filename="variability_test",
                output_extension="png",
                output_dpi=110,
            )
            out = plot_fixation_psth_variability_violins(settings)
            self.assertIsNotNone(out)
            assert out is not None
            self.assertTrue(Path(out["output_path"]).exists())
            self.assertEqual(len(out["region_order"]), 4)
            self.assertEqual(len(out["condition_order"]), 3)


if __name__ == "__main__":
    unittest.main()
