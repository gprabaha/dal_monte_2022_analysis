"""Regression tests for three-way region-comparison analysis and plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_three_way_region_comparison import (
    FixationThreeWayRegionComparisonSettings,
    run_fixation_three_way_region_comparison,
)

try:
    from dal_monte_2022_analysis.ephys.plotting.fixation_three_way_region_comparison import (
        FixationThreeWayRegionComparisonPlotSettings,
        plot_fixation_three_way_region_comparison_heatmaps,
    )

    _HAS_REGION_PLOT = True
except ModuleNotFoundError:
    _HAS_REGION_PLOT = False


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


class TestFixationThreeWayRegionComparison(unittest.TestCase):
    """Regression checks for region-comparison summary construction."""

    def test_run_region_comparison_writes_pairwise_and_window_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            in_path = (
                analysis_root
                / "ephys/psth/fixation_psth_selectivity"
                / "condition_window_means.csv"
            )
            in_path.parent.mkdir(parents=True, exist_ok=True)

            rng = np.random.default_rng(7)
            rows = []
            templates = {
                "ACC": np.asarray([0.72, 0.20, 0.08], dtype=float),
                "OFC": np.asarray([0.22, 0.68, 0.10], dtype=float),
                "PFC": np.asarray([0.20, 0.22, 0.58], dtype=float),
            }
            for window_name, start, stop in (
                ("pre_fix", -500.0, 0.0),
                ("post_fix", 0.0, 500.0),
            ):
                for region, base in templates.items():
                    for u_idx in range(8):
                        noise = rng.normal(loc=0.0, scale=0.015, size=3)
                        vec = np.clip(base + noise, 1e-4, None)
                        vec = vec / vec.sum()
                        rows.append(
                            {
                                "unit_key": f"01012020|{region}|u{u_idx}",
                                "date": "01012020",
                                "unit_uuid": f"{region}_u{u_idx}",
                                "region": region,
                                "window_name": window_name,
                                "window_start_ms": start,
                                "window_stop_ms": stop,
                                "relative_face_interactive": float(vec[0]),
                                "relative_face_non_interactive": float(vec[1]),
                                "relative_object": float(vec[2]),
                                "all_conditions_observed": True,
                                "meets_min_trials": True,
                            }
                        )
            pd.DataFrame(rows).to_csv(in_path, index=False)

            settings = FixationThreeWayRegionComparisonSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_selectivity",
                condition_summary_filename="condition_window_means.csv",
                output_subdir="ephys/psth/fixation_psth_selectivity_region_comparison",
                min_units_per_region=5,
                n_permutations=80,
                random_seed=13,
                alpha=0.05,
            )
            result = run_fixation_three_way_region_comparison(settings)

            pair_df = result.get("pairwise_summary")
            win_df = result.get("window_summary")
            self.assertIsInstance(pair_df, pd.DataFrame)
            self.assertIsInstance(win_df, pd.DataFrame)
            self.assertFalse(pair_df.empty)
            self.assertFalse(win_df.empty)
            self.assertEqual(set(win_df["window_name"].astype(str)), {"pre_fix", "post_fix"})

            out_root = analysis_root / "ephys/psth/fixation_psth_selectivity_region_comparison"
            self.assertTrue((out_root / "pairwise_region_comparisons.csv").exists())
            self.assertTrue((out_root / "window_region_comparisons.csv").exists())
            self.assertTrue((out_root / "results.pkl").exists())


@unittest.skipUnless(_HAS_REGION_PLOT, "matplotlib is required for region-comparison plotting tests")
class TestFixationThreeWayRegionComparisonPlot(unittest.TestCase):
    """Regression checks for region-comparison heatmap rendering."""

    def test_plot_region_comparison_heatmaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            in_root = analysis_root / "ephys/psth/fixation_psth_selectivity_region_comparison"
            in_root.mkdir(parents=True, exist_ok=True)

            pair_df = pd.DataFrame(
                [
                    {
                        "window_name": "pre_fix",
                        "window_start_ms": -500.0,
                        "window_stop_ms": 0.0,
                        "region_a": "ACC",
                        "region_b": "OFC",
                        "centroid_distance_ilr": 1.2,
                        "p_value_adjusted": 0.01,
                        "significant": True,
                    },
                    {
                        "window_name": "post_fix",
                        "window_start_ms": 0.0,
                        "window_stop_ms": 500.0,
                        "region_a": "ACC",
                        "region_b": "OFC",
                        "centroid_distance_ilr": 0.4,
                        "p_value_adjusted": 0.2,
                        "significant": False,
                    },
                ]
            )
            window_df = pd.DataFrame(
                [
                    {
                        "window_name": "pre_fix",
                        "window_start_ms": -500.0,
                        "window_stop_ms": 0.0,
                        "global_p_value_adjusted": 0.02,
                    },
                    {
                        "window_name": "post_fix",
                        "window_start_ms": 0.0,
                        "window_stop_ms": 500.0,
                        "global_p_value_adjusted": 0.25,
                    },
                ]
            )
            pair_df.to_csv(in_root / "pairwise_region_comparisons.csv", index=False)
            window_df.to_csv(in_root / "window_region_comparisons.csv", index=False)

            settings = FixationThreeWayRegionComparisonPlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_selectivity_region_comparison",
                output_subdir="ephys/psth/fixation_psth_selectivity_region_comparison/plots",
                output_filename="region_comparison_test",
                output_extension="png",
                output_dpi=110,
                alpha=0.05,
            )
            out = plot_fixation_three_way_region_comparison_heatmaps(settings)
            self.assertIsNotNone(out)
            assert out is not None
            self.assertTrue(Path(out["output_path"]).exists())
            self.assertEqual(len(out["window_order"]), 2)
            self.assertEqual(len(out["region_order"]), 2)


if __name__ == "__main__":
    unittest.main()

