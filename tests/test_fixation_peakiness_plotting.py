"""Regression tests for fixation peakiness plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from dal_monte_2022_analysis.ephys.plotting.fixation_peakiness import (
        FixationPeakinessPlotSettings,
        plot_fixation_peakiness_by_region,
    )

    _HAS_PEAKINESS_PLOTTING = True
except ModuleNotFoundError:
    _HAS_PEAKINESS_PLOTTING = False


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


@unittest.skipUnless(_HAS_PEAKINESS_PLOTTING, "matplotlib is required for peakiness plotting tests")
class TestFixationPeakinessPlot(unittest.TestCase):
    """Regression checks for peakiness distribution plotting."""

    def test_plot_peakiness_by_region_with_highlights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            in_root = analysis_root / "ephys/psth/fixation_peakiness"
            in_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "unit_uuid": "unit_uuid__145",
                        "region": "accg",
                        "peakiness_score": 1.20,
                        "best_condition": "object",
                    },
                    {
                        "unit_uuid": "unit_uuid__118",
                        "region": "accg",
                        "peakiness_score": 0.37,
                        "best_condition": "face_non_interactive",
                    },
                    {
                        "unit_uuid": "unit_uuid__1091",
                        "region": "ofc",
                        "peakiness_score": 0.26,
                        "best_condition": "object",
                    },
                    {
                        "unit_uuid": "unit_uuid__1011",
                        "region": "ofc",
                        "peakiness_score": 0.24,
                        "best_condition": "face_non_interactive",
                    },
                ]
            ).to_csv(in_root / "unit_peakiness.csv", index=False)

            settings = FixationPeakinessPlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_peakiness",
                output_subdir="ephys/psth/fixation_peakiness/plots",
                output_filename="peakiness_test",
                output_extension="png",
                region_order=("accg", "ofc"),
                region_labels={"accg": "ACCg", "ofc": "OFC"},
                highlight_units={
                    "ACCg": {
                        "phasic": {"unit_uuid": "145"},
                        "tonic": {"unit_uuid": "118"},
                    },
                    "OFC": {
                        "phasic": {"unit_uuid": "1091"},
                        "tonic": {"unit_uuid": "1011"},
                    },
                },
                highlight_style_order=("phasic", "tonic"),
            )
            result = plot_fixation_peakiness_by_region(settings)

            self.assertIsNotNone(result)
            assert result is not None
            output_path = Path(result["output_path"])
            self.assertTrue(output_path.exists())
            highlighted = result.get("highlighted_units", [])
            matched = [row for row in highlighted if bool(row.get("matched"))]
            self.assertEqual(len(matched), 4)
            by_style = {(row["region_label"], row["style"]): float(row["peakiness_score"]) for row in matched}
            self.assertGreater(by_style[("ACCg", "phasic")], by_style[("ACCg", "tonic")])
