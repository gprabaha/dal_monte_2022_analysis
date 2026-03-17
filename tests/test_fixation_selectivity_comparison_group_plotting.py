"""Regression tests for fixation selectivity comparison-group plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from dal_monte_2022_analysis.ephys.plotting.fixation_selectivity_comparison_group import (
        FixationSelectivityComparisonGroupPlotSettings,
        plot_fixation_selectivity_comparison_group_summaries,
    )

    _HAS_COMPARISON_GROUP_PLOTTING = True
except ModuleNotFoundError:
    _HAS_COMPARISON_GROUP_PLOTTING = False


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


@unittest.skipUnless(
    _HAS_COMPARISON_GROUP_PLOTTING,
    "comparison-group plotting module is required for these tests",
)
class TestFixationSelectivityComparisonGroupPlotting(unittest.TestCase):
    """Checks comparison-group summary aggregation and file output."""

    def test_plot_comparison_group_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            selectivity_root = analysis_root / "ephys/psth/fixation_psth_selectivity"
            selectivity_root.mkdir(parents=True, exist_ok=True)
            pair_path = selectivity_root / "pair_selectivity__interactive_state_matched.csv"

            rows = [
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u1",
                    "unit_key": "20990101|u1",
                    "pair_label": "face_interactive__vs__face_non_interactive",
                    "is_selective_pair": True,
                    "significant_windows": "pre_fix",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u1",
                    "unit_key": "20990101|u1",
                    "pair_label": "face_interactive__vs__object_interactive",
                    "is_selective_pair": True,
                    "significant_windows": "peri_fix",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u1",
                    "unit_key": "20990101|u1",
                    "pair_label": "face_non_interactive__vs__object_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u1",
                    "unit_key": "20990101|u1",
                    "pair_label": "object_interactive__vs__object_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u2",
                    "unit_key": "20990101|u2",
                    "pair_label": "face_interactive__vs__face_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u2",
                    "unit_key": "20990101|u2",
                    "pair_label": "face_interactive__vs__object_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u2",
                    "unit_key": "20990101|u2",
                    "pair_label": "face_non_interactive__vs__object_non_interactive",
                    "is_selective_pair": True,
                    "significant_windows": "post_fix",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u2",
                    "unit_key": "20990101|u2",
                    "pair_label": "object_interactive__vs__object_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u3",
                    "unit_key": "20990101|u3",
                    "pair_label": "face_interactive__vs__face_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u3",
                    "unit_key": "20990101|u3",
                    "pair_label": "face_interactive__vs__object_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u3",
                    "unit_key": "20990101|u3",
                    "pair_label": "face_non_interactive__vs__object_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "BLA",
                    "date": "20990101",
                    "unit_uuid": "u3",
                    "unit_key": "20990101|u3",
                    "pair_label": "object_interactive__vs__object_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "OFC",
                    "date": "20990101",
                    "unit_uuid": "u10",
                    "unit_key": "20990101|u10",
                    "pair_label": "face_interactive__vs__face_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "OFC",
                    "date": "20990101",
                    "unit_uuid": "u10",
                    "unit_key": "20990101|u10",
                    "pair_label": "face_interactive__vs__object_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "OFC",
                    "date": "20990101",
                    "unit_uuid": "u10",
                    "unit_key": "20990101|u10",
                    "pair_label": "face_non_interactive__vs__object_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "OFC",
                    "date": "20990101",
                    "unit_uuid": "u10",
                    "unit_key": "20990101|u10",
                    "pair_label": "object_interactive__vs__object_non_interactive",
                    "is_selective_pair": True,
                    "significant_windows": "full_fix",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "OFC",
                    "date": "20990101",
                    "unit_uuid": "u11",
                    "unit_key": "20990101|u11",
                    "pair_label": "face_interactive__vs__face_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "OFC",
                    "date": "20990101",
                    "unit_uuid": "u11",
                    "unit_key": "20990101|u11",
                    "pair_label": "face_interactive__vs__object_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "OFC",
                    "date": "20990101",
                    "unit_uuid": "u11",
                    "unit_key": "20990101|u11",
                    "pair_label": "face_non_interactive__vs__object_non_interactive",
                    "is_selective_pair": False,
                    "significant_windows": "",
                },
                {
                    "comparison_label": "interactive_state_matched",
                    "region": "OFC",
                    "date": "20990101",
                    "unit_uuid": "u11",
                    "unit_key": "20990101|u11",
                    "pair_label": "object_interactive__vs__object_non_interactive",
                    "is_selective_pair": True,
                    "significant_windows": "pre_fix",
                },
            ]
            pd.DataFrame(rows).to_csv(pair_path, index=False)

            settings = FixationSelectivityComparisonGroupPlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_selectivity",
                output_subdir="ephys/psth/fixation_psth_selectivity_comparison_group_plots",
                selective_windows=("pre_fix", "peri_fix", "post_fix"),
            )
            result = plot_fixation_selectivity_comparison_group_summaries(settings)

            self.assertIsInstance(result, dict)
            assert result is not None
            self.assertTrue(Path(result["fraction_bar_output_path"]).exists())
            self.assertTrue(Path(result["overlap_matrix_output_path"]).exists())

            summaries = {
                str(row["region"]): row
                for row in result["region_summaries"]
            }
            self.assertEqual(set(summaries.keys()), {"BLA", "OFC"})

            bla = summaries["BLA"]
            self.assertEqual(int(bla["total_units"]), 3)
            self.assertEqual(int(bla["any_selective_units"]), 2)
            self.assertEqual(int(bla["pair_counts"]["face_interactive__vs__face_non_interactive"]), 1)
            self.assertEqual(int(bla["pair_counts"]["face_interactive__vs__object_interactive"]), 1)
            self.assertEqual(int(bla["pair_counts"]["face_non_interactive__vs__object_non_interactive"]), 1)
            self.assertEqual(int(bla["pair_counts"]["object_interactive__vs__object_non_interactive"]), 0)
            self.assertEqual(
                int(bla["pattern_counts"][(
                    "face_interactive__vs__face_non_interactive",
                    "face_interactive__vs__object_interactive",
                )]),
                1,
            )
            self.assertEqual(
                int(bla["pattern_counts"][("face_non_interactive__vs__object_non_interactive",)]),
                1,
            )

            ofc = summaries["OFC"]
            self.assertEqual(int(ofc["total_units"]), 2)
            self.assertEqual(int(ofc["any_selective_units"]), 1)
            self.assertEqual(int(ofc["pair_counts"]["object_interactive__vs__object_non_interactive"]), 1)


if __name__ == "__main__":
    unittest.main()
