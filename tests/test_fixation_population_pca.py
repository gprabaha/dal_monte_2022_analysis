"""Regression tests for fixation population PCA analysis outputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_population_pca import (
    FixationPopulationPCASettings,
    run_fixation_population_pca_analysis,
)
from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path

try:
    import dal_monte_2022_analysis.ephys.plotting.fixation_population_pca as _pca_plot_module

    FixationPopulationPCAPlotSettings = _pca_plot_module.FixationPopulationPCAPlotSettings
    plot_fixation_population_pca_explained_variance_bars = (
        _pca_plot_module.plot_fixation_population_pca_explained_variance_bars
    )
    plot_fixation_population_pca_explained_variance_cumulative = (
        _pca_plot_module.plot_fixation_population_pca_explained_variance_cumulative
    )
    plot_fixation_population_pca_pairwise_geometry_violins = (
        _pca_plot_module.plot_fixation_population_pca_pairwise_geometry_violins
    )
    plot_fixation_population_pca_trajectories = (
        _pca_plot_module.plot_fixation_population_pca_trajectories
    )
    _HAS_PCA_PLOTTING = True
    _HAS_PCA_GEOMETRY_VIOLIN = getattr(_pca_plot_module, "sns", None) is not None
except ModuleNotFoundError:
    _HAS_PCA_PLOTTING = False
    _HAS_PCA_GEOMETRY_VIOLIN = False


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


def _condition_labels(condition: str) -> tuple[str, str]:
    if condition == "face_interactive":
        return "face", "interactive"
    if condition == "face_non_interactive":
        return "face", "non_interactive"
    if condition == "object":
        return "object", "non_interactive"
    raise ValueError(f"Unknown condition '{condition}'.")


def _build_pattern(bin_centers: np.ndarray, *, unit_scale: float, condition: str) -> np.ndarray:
    x = np.asarray(bin_centers, dtype=float)
    if condition == "face_interactive":
        vec = 6.0 + (1.5 * unit_scale) + np.sin(2.0 * np.pi * (x + 0.55))
    elif condition == "face_non_interactive":
        vec = 4.0 + (1.2 * unit_scale) + np.cos(2.0 * np.pi * (x + 0.35))
    elif condition == "object":
        vec = 3.0 + (0.8 * unit_scale) + 0.6 * np.sin(2.0 * np.pi * (x + 0.15))
    else:
        raise ValueError(f"Unsupported condition '{condition}'.")
    return np.clip(vec, 0.0, None)


def _run_population_pca_fixture(
    root: Path,
) -> tuple[Path, Path, np.ndarray, dict]:
    processed_root = root / "processed"
    analysis_root = root / "analysis"
    cfg_path = root / "dataset.yaml"
    _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

    date = "20990101"
    avg_root = analysis_root / "ephys/psth/fixation_psth_averages" / f"date={date}"
    avg_root.mkdir(parents=True, exist_ok=True)
    split_avg_path = avg_root / "fixations_split.pkl"
    unsplit_avg_path = avg_root / "fixations_unsplit.pkl"

    bin_centers = np.arange(-0.55, 0.56, 0.1, dtype=float)
    split_rows: list[dict] = []
    unsplit_rows: list[dict] = []
    for region, unit_ids in (("ACC", ("u_acc_1", "u_acc_2", "u_acc_3")), ("BLA", ("u_bla_1", "u_bla_2"))):
        for unit_idx, unit_id in enumerate(unit_ids):
            condition_order = ("face_interactive", "face_non_interactive", "object")
            if region == "BLA" and unit_id == "u_bla_2":
                condition_order = ("face_interactive", "face_non_interactive")
            for condition in condition_order:
                category, interactive_state = _condition_labels(condition)
                split_rows.append(
                    {
                        "date": date,
                        "unit_uuid": unit_id,
                        "region": region,
                        "spike_channel": f"ch_{unit_idx + 1}",
                        "recorded_agent": "m1",
                        "recorded_monkey": "test_monkey",
                        "area": region.lower(),
                        "fixation_category": category,
                        "interactive_state": interactive_state,
                        "n_trials": float(10 + unit_idx),
                        "psth_mean": _build_pattern(
                            bin_centers,
                            unit_scale=float(unit_idx + 1),
                            condition=condition,
                        ),
                    }
                )
                if condition == "object":
                    if region == "ACC" or unit_id == "u_bla_1":
                        unsplit_rows.append(
                            {
                                "date": date,
                                "unit_uuid": unit_id,
                                "region": region,
                                "spike_channel": f"ch_{unit_idx + 1}",
                                "recorded_agent": "m1",
                                "recorded_monkey": "test_monkey",
                                "area": region.lower(),
                                "fixation_category": "object",
                                "n_trials": float(10 + unit_idx),
                                "psth_mean": (
                                    _build_pattern(
                                        bin_centers,
                                        unit_scale=float(unit_idx + 1),
                                        condition=condition,
                                    )
                                    + 20.0
                                ),
                            }
                        )

    save_pickle_path(
        {"meta": {"bin_centers_s_rel": bin_centers}, "averages": pd.DataFrame(split_rows)},
        split_avg_path,
    )
    save_pickle_path(
        {"meta": {"bin_centers_s_rel": bin_centers}, "averages": pd.DataFrame(unsplit_rows)},
        unsplit_avg_path,
    )

    settings = FixationPopulationPCASettings(
        cfg_path=str(cfg_path),
        input_subdir="ephys/psth/fixation_psth_averages",
        input_filename="fixations_split.pkl",
        object_input_subdir="ephys/psth/fixation_psth_averages",
        object_input_filename="fixations_unsplit.pkl",
        prefer_trial_input=False,
        output_subdir="ephys/psth/fixation_population_pca",
        window_start_ms=-500.0,
        window_stop_ms=500.0,
        max_components=3,
        min_units_per_region=2,
        use_parallel=False,
        geometry_n_pcs=20,
    )
    result = run_fixation_population_pca_analysis(settings)
    return cfg_path, analysis_root, bin_centers, result


class TestFixationPopulationPCA(unittest.TestCase):
    """Regression checks for fixation population PCA build outputs."""

    def test_population_pca_outputs_include_concatenated_timecourses_and_cross_variance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path, analysis_root, bin_centers, result = _run_population_pca_fixture(Path(tmp_dir))
            date = "20990101"

            self.assertIn("ACC", result["regions"])
            self.assertNotIn("BLA", result["regions"])
            self.assertEqual(str(result["meta"].get("input_source")), "average:ephys/psth/fixation_psth_averages")

            acc_payload = result["regions"]["ACC"]
            int_matrix = np.asarray(
                acc_payload["condition_matrices_units_by_time"]["face_interactive"],
                dtype=float,
            )
            object_matrix = np.asarray(
                acc_payload["condition_matrices_units_by_time"]["object"],
                dtype=float,
            )
            unit_keys = [str(token) for token in np.asarray(acc_payload["unit_keys"], dtype=object).tolist()]
            u1_idx = unit_keys.index(f"{date}|u_acc_1")
            expected_u1_object = _build_pattern(
                bin_centers[np.logical_and(bin_centers >= -0.5, bin_centers <= 0.5)],
                unit_scale=1.0,
                condition="object",
            ) + 20.0

            self.assertEqual(int_matrix.shape, (3, 10))
            self.assertEqual(object_matrix.shape, (3, 10))
            self.assertEqual(int(acc_payload["concatenated_fit"]["n_samples"]), 30)
            self.assertEqual(int(acc_payload["concatenated_fit"]["n_components"]), 3)
            self.assertTrue(np.allclose(object_matrix[u1_idx], expected_u1_object))

            time_df = result["concatenated_timecourses"]
            self.assertIsInstance(time_df, pd.DataFrame)
            self.assertFalse(time_df.empty)
            acc_time_df = time_df.loc[time_df["region"].astype(str) == "ACC"].copy()
            self.assertEqual(len(acc_time_df), 3 * 10 * 3)

            explained_df = result["cross_condition_explained_variance"]
            self.assertIsInstance(explained_df, pd.DataFrame)
            self.assertFalse(explained_df.empty)
            acc_explained_df = explained_df.loc[explained_df["region"].astype(str) == "ACC"].copy()
            self.assertEqual(len(acc_explained_df), 3 * 3 * 3)
            self.assertEqual(set(acc_explained_df["n_components"].astype(int)), {1, 2, 3})

            geometry_time_df = result["pairwise_geometry_timecourses"]
            self.assertIsInstance(geometry_time_df, pd.DataFrame)
            self.assertFalse(geometry_time_df.empty)
            acc_geometry_df = geometry_time_df.loc[geometry_time_df["region"].astype(str) == "ACC"].copy()
            self.assertEqual(len(acc_geometry_df), 3 * 10 * 2)
            self.assertEqual(
                set(acc_geometry_df["metric_name"].astype(str)),
                {"euclidean_distance", "angle_degrees"},
            )
            self.assertEqual(set(acc_geometry_df["n_pcs_used"].astype(int)), {3})
            self.assertEqual(
                set(acc_geometry_df["condition_pair"].astype(str)),
                {
                    "face_interactive__vs__face_non_interactive",
                    "face_interactive__vs__object",
                    "face_non_interactive__vs__object",
                },
            )

            geometry_summary_df = result["pairwise_geometry_summary"]
            self.assertIsInstance(geometry_summary_df, pd.DataFrame)
            self.assertEqual(len(geometry_summary_df), 3 * 2)

            within_stats_df = result["pairwise_geometry_within_region_stats"]
            self.assertIsInstance(within_stats_df, pd.DataFrame)
            self.assertEqual(len(within_stats_df), 3 * 2)

            cross_stats_df = result["pairwise_geometry_cross_region_stats"]
            self.assertIsInstance(cross_stats_df, pd.DataFrame)
            self.assertTrue(cross_stats_df.empty)
            self.assertEqual(int(result["meta"]["geometry_n_pcs_requested"]), 20)
            self.assertEqual(int(result["meta"]["geometry_n_pcs_effective_max"]), 3)

            scores_map = acc_payload["concatenated_condition_scores_pc_by_time"]
            int_scores = np.asarray(scores_map["face_interactive"], dtype=float)
            nonint_scores = np.asarray(scores_map["face_non_interactive"], dtype=float)
            expected_distance = np.sqrt(np.sum((int_scores[:3, :] - nonint_scores[:3, :]) ** 2, axis=0))
            observed_distance = (
                acc_geometry_df.loc[
                    (acc_geometry_df["condition_pair"].astype(str) == "face_interactive__vs__face_non_interactive")
                    & (acc_geometry_df["metric_name"].astype(str) == "euclidean_distance")
                ]
                .sort_values("bin_index")["value"]
                .to_numpy(dtype=float)
            )
            self.assertTrue(np.allclose(observed_distance, expected_distance))

            observed_summary = geometry_summary_df.loc[
                (geometry_summary_df["region"].astype(str) == "ACC")
                & (
                    geometry_summary_df["condition_pair"].astype(str)
                    == "face_interactive__vs__face_non_interactive"
                )
                & (geometry_summary_df["metric_name"].astype(str) == "euclidean_distance")
            ].iloc[0]
            self.assertAlmostEqual(
                float(observed_summary["mean_value"]),
                float(np.mean(expected_distance)),
            )

            out_root = analysis_root / "ephys/psth/fixation_population_pca"
            self.assertTrue((out_root / "pca_fit_summary.csv").exists())
            self.assertTrue((out_root / "concatenated_pc_timecourses.csv").exists())
            self.assertTrue((out_root / "cross_condition_explained_variance.csv").exists())
            self.assertTrue((out_root / "pairwise_geometry_timecourses.csv").exists())
            self.assertTrue((out_root / "pairwise_geometry_summary.csv").exists())
            self.assertTrue((out_root / "pairwise_geometry_within_region_stats.csv").exists())
            self.assertTrue((out_root / "pairwise_geometry_cross_region_stats.csv").exists())
            self.assertTrue((out_root / "region_unit_inventory.csv").exists())
            self.assertTrue((out_root / "results.pkl").exists())

    def test_population_pca_requires_face_interactive_state_labels_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            date = "20990102"
            avg_path = analysis_root / "ephys/psth/fixation_psth_averages" / f"date={date}" / "fixations.pkl"
            avg_path.parent.mkdir(parents=True, exist_ok=True)

            bin_centers = np.asarray([-0.15, -0.05, 0.05, 0.15], dtype=float)
            rows = [
                {
                    "date": date,
                    "unit_uuid": "unit_1",
                    "region": "ACC",
                    "fixation_category": "face",
                    "n_trials": 4,
                    "psth_mean": np.asarray([2.0, 3.0, 4.0, 5.0], dtype=float),
                },
                {
                    "date": date,
                    "unit_uuid": "unit_1",
                    "region": "ACC",
                    "fixation_category": "object",
                    "interactive_state": "non_interactive",
                    "n_trials": 4,
                    "psth_mean": np.asarray([1.0, 1.5, 1.5, 1.0], dtype=float),
                },
            ]
            save_pickle_path(
                {"meta": {"bin_centers_s_rel": bin_centers}, "averages": pd.DataFrame(rows)},
                avg_path,
            )

            settings = FixationPopulationPCASettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_averages",
                input_filename="fixations.pkl",
                output_subdir="ephys/psth/fixation_population_pca",
                use_parallel=False,
            )

            with self.assertRaisesRegex(ValueError, "split_by_interactive_state=true"):
                run_fixation_population_pca_analysis(settings)


@unittest.skipUnless(_HAS_PCA_PLOTTING, "matplotlib is required for fixation population PCA plotting tests")
class TestFixationPopulationPCAPlotting(unittest.TestCase):
    """Regression checks for fixation population PCA plotting outputs."""

    def test_population_pca_plotting_outputs_trajectory_and_variance_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path, _, _, _ = _run_population_pca_fixture(Path(tmp_dir))
            settings = FixationPopulationPCAPlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_population_pca",
                input_filename="results.pkl",
                output_subdir="ephys/psth/fixation_population_pca/plots",
                output_extension="png",
                output_dpi=100,
                max_components_display=3,
            )

            trajectory_out = plot_fixation_population_pca_trajectories(
                settings,
                output_filename="population_pca_pc_trajectories_test",
            )
            self.assertIsNotNone(trajectory_out)
            assert trajectory_out is not None
            self.assertTrue(Path(str(trajectory_out["output_path"])).exists())

            explained_out = plot_fixation_population_pca_explained_variance_bars(
                settings,
                output_filename="population_pca_explained_variance_bars_test",
            )
            self.assertIsNotNone(explained_out)
            assert explained_out is not None
            self.assertTrue(Path(str(explained_out["output_path"])).exists())

            cumulative_out = plot_fixation_population_pca_explained_variance_cumulative(
                settings,
                output_filename="population_pca_explained_variance_cumulative_test",
            )
            self.assertIsNotNone(cumulative_out)
            assert cumulative_out is not None
            self.assertTrue(Path(str(cumulative_out["output_path"])).exists())


@unittest.skipUnless(
    _HAS_PCA_PLOTTING and _HAS_PCA_GEOMETRY_VIOLIN,
    "matplotlib and seaborn are required for fixation population PCA geometry violin plotting tests",
)
class TestFixationPopulationPCAGeometryViolinPlotting(unittest.TestCase):
    """Regression checks for pairwise PCA geometry violin rendering."""

    def test_population_pca_geometry_violin_outputs_four_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path, _, _, _ = _run_population_pca_fixture(Path(tmp_dir))
            settings = FixationPopulationPCAPlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_population_pca",
                input_filename="results.pkl",
                output_subdir="ephys/psth/fixation_population_pca/plots",
                output_extension="png",
                output_dpi=100,
                pairwise_violin_letter_width_in=7.0,
                pairwise_violin_letter_height_frac=0.26,
            )

            outputs = plot_fixation_population_pca_pairwise_geometry_violins(
                settings,
                output_filename="population_pca_pairwise_geometry_violin_test",
            )
            self.assertEqual(len(outputs), 4)
            self.assertEqual(
                {(str(row["metric_name"]), str(row["kind"])) for row in outputs},
                {
                    ("euclidean_distance", "within_region"),
                    ("euclidean_distance", "cross_region"),
                    ("angle_degrees", "within_region"),
                    ("angle_degrees", "cross_region"),
                },
            )
            for row in outputs:
                self.assertTrue(Path(str(row["output_path"])).exists())


if __name__ == "__main__":
    unittest.main()
