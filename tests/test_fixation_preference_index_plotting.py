"""Regression tests for fixation preference-index heatmap plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dal_monte_2022_analysis.ephys.plotting.fixation_preference_index_heatmap import (
        FixationPreferenceIndexHeatmapPlotSettings,
        plot_fixation_preference_index_heatmaps,
    )

    _HAS_INDEX_PLOT = True
except ModuleNotFoundError:
    _HAS_INDEX_PLOT = False


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


@unittest.skipUnless(_HAS_INDEX_PLOT, "matplotlib is required for preference-index plotting tests")
class TestFixationPreferenceIndexPlotting(unittest.TestCase):
    """Regression checks for per-pair preference-index heatmap rendering."""

    def test_plot_preference_index_heatmaps_outputs_combined_pair_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_date = "20990101"
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            in_root = analysis_root / "ephys/fixation_preference_index"
            in_root.mkdir(parents=True, exist_ok=True)

            pairs = [
                ("face_interactive__vs__face_non_interactive", "interactive_face_preference_index"),
                ("face_interactive__vs__object", "interactive_face_vs_object_index"),
                ("face_non_interactive__vs__object", "non_interactive_face_vs_object_index"),
            ]
            bin_centers = np.asarray([-0.25, -0.05, 0.15, 0.35], dtype=float)
            rows = []
            for pair_idx, (pair_label, index_name) in enumerate(pairs):
                for region in ("BLA", "ACCg"):
                    for unit_idx in range(2):
                        unit_key = f"{dummy_date}|{region}|unit_{unit_idx + 1:03d}"
                        is_pair_selective = bool(unit_idx == (pair_idx % 2))
                        for bin_idx, center in enumerate(bin_centers):
                            rows.append(
                                {
                                    "pair_label": pair_label,
                                    "index_name": index_name,
                                    "region": region,
                                    "unit_key": unit_key,
                                    "bin_index": int(bin_idx),
                                    "bin_center_s": float(center),
                                    "preference_index": float((unit_idx + 1) * center),
                                    "preference_index_unit_max_sum": float((unit_idx + 1) * center),
                                    "preference_index_per_bin_sum": float((unit_idx + 1) * center * 0.5),
                                    "is_selective_pair": is_pair_selective,
                                }
                            )
                    for bin_idx, center in enumerate(bin_centers):
                        rows.append(
                            {
                                "pair_label": pair_label,
                                "index_name": index_name,
                                "region": region,
                                "unit_key": f"{dummy_date}|{region}|unit_nonselective",
                                "bin_index": int(bin_idx),
                                "bin_center_s": float(center),
                                "preference_index": float(center),
                                "preference_index_unit_max_sum": float(center),
                                "preference_index_per_bin_sum": float(center * 0.5),
                                "is_selective_pair": False,
                            }
                        )
            pd.DataFrame(rows).to_csv(in_root / "preference_index_timeseries.csv", index=False)

            settings = FixationPreferenceIndexHeatmapPlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/fixation_preference_index",
                timeseries_filename="preference_index_timeseries.csv",
                output_subdir="ephys/fixation_preference_index/plots",
                output_filename="index_heatmap_test",
                output_extension="png",
                output_dpi=110,
                include_only_pair_selective_units=True,
                normalization_mode="per_bin_sum",
                region_order=("BLA", "ACCg", "dmPFC", "OFC"),
                figure_width_in=8.5,
                figure_height_in=4.4,
            )
            out = plot_fixation_preference_index_heatmaps(settings)

            self.assertIsNotNone(out)
            assert out is not None
            self.assertEqual(str(out.get("normalization_mode")), "per_bin_sum")
            self.assertEqual(str(out.get("value_column")), "preference_index_per_bin_sum")
            self.assertTrue(bool(out.get("combine_pairs_into_single_figure")))
            outputs = out["outputs"]
            self.assertEqual(len(outputs), 1)
            row = outputs[0]
            self.assertTrue(bool(row.get("combined_pairs")))
            self.assertTrue(Path(row["output_path"]).exists())
            self.assertEqual(
                set(row["n_units_by_pair_region"].keys()),
                {
                    "face_interactive__vs__face_non_interactive",
                    "face_interactive__vs__object",
                    "face_non_interactive__vs__object",
                },
            )
            for pair_label in row["n_units_by_pair_region"]:
                self.assertEqual(int(row["n_units_by_pair_region"][pair_label]["BLA"]), 1)
                self.assertEqual(int(row["n_units_by_pair_region"][pair_label]["ACCg"]), 1)
                self.assertEqual(int(row["n_units_by_pair_region"][pair_label]["dmPFC"]), 0)
                self.assertEqual(int(row["n_units_by_pair_region"][pair_label]["OFC"]), 0)

    def test_plot_preference_index_heatmaps_any_selective_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_date = "20990101"
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            in_root = analysis_root / "ephys/fixation_preference_index"
            in_root.mkdir(parents=True, exist_ok=True)

            pairs = [
                ("face_interactive__vs__face_non_interactive", "interactive_face_preference_index"),
                ("face_interactive__vs__object", "interactive_face_vs_object_index"),
                ("face_non_interactive__vs__object", "non_interactive_face_vs_object_index"),
            ]
            bin_centers = np.asarray([-0.25, -0.05, 0.15, 0.35], dtype=float)
            rows = []
            for pair_idx, (pair_label, index_name) in enumerate(pairs):
                for region in ("BLA", "ACCg"):
                    for unit_idx in range(2):
                        unit_key = f"{dummy_date}|{region}|unit_{unit_idx + 1:03d}"
                        is_pair_selective = bool(unit_idx == (pair_idx % 2))
                        for bin_idx, center in enumerate(bin_centers):
                            rows.append(
                                {
                                    "pair_label": pair_label,
                                    "index_name": index_name,
                                    "region": region,
                                    "unit_key": unit_key,
                                    "bin_index": int(bin_idx),
                                    "bin_center_s": float(center),
                                    "preference_index": float((unit_idx + 1) * center),
                                    "preference_index_unit_max_sum": float((unit_idx + 1) * center),
                                    "preference_index_per_bin_sum": float((unit_idx + 1) * center * 0.5),
                                    "is_selective_pair": is_pair_selective,
                                }
                            )
                    for bin_idx, center in enumerate(bin_centers):
                        rows.append(
                            {
                                "pair_label": pair_label,
                                "index_name": index_name,
                                "region": region,
                                "unit_key": f"{dummy_date}|{region}|unit_nonselective",
                                "bin_index": int(bin_idx),
                                "bin_center_s": float(center),
                                "preference_index": float(center),
                                "preference_index_unit_max_sum": float(center),
                                "preference_index_per_bin_sum": float(center * 0.5),
                                "is_selective_pair": False,
                            }
                        )
            pd.DataFrame(rows).to_csv(in_root / "preference_index_timeseries.csv", index=False)

            settings = FixationPreferenceIndexHeatmapPlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/fixation_preference_index",
                timeseries_filename="preference_index_timeseries.csv",
                output_subdir="ephys/fixation_preference_index/plots",
                output_filename="index_heatmap_test_any_selective",
                output_extension="png",
                output_dpi=110,
                include_only_pair_selective_units=True,
                unit_filter_mode="any_selective",
                sort_reference_pair_label="face_interactive__vs__face_non_interactive",
                normalization_mode="unit_max_sum",
                region_order=("BLA", "ACCg", "dmPFC", "OFC"),
                figure_width_in=8.5,
                figure_height_in=4.4,
            )
            out = plot_fixation_preference_index_heatmaps(settings)

            self.assertIsNotNone(out)
            assert out is not None
            self.assertEqual(str(out.get("unit_filter_mode")), "any_selective")
            self.assertEqual(
                str(out.get("sort_reference_pair_label")),
                "face_interactive__vs__face_non_interactive",
            )
            outputs = out["outputs"]
            self.assertEqual(len(outputs), 1)
            row = outputs[0]
            self.assertTrue(bool(row.get("combined_pairs")))
            self.assertTrue(Path(row["output_path"]).exists())
            for pair_label in row["n_units_by_pair_region"]:
                self.assertEqual(int(row["n_units_by_pair_region"][pair_label]["BLA"]), 2)
                self.assertEqual(int(row["n_units_by_pair_region"][pair_label]["ACCg"]), 2)
                self.assertEqual(int(row["n_units_by_pair_region"][pair_label]["dmPFC"]), 0)
                self.assertEqual(int(row["n_units_by_pair_region"][pair_label]["OFC"]), 0)


if __name__ == "__main__":
    unittest.main()
