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


class TestFixationPopulationPCA(unittest.TestCase):
    """Regression checks for fixation population PCA build outputs."""

    def test_population_pca_outputs_include_concatenated_timecourses_and_cross_variance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
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
            )
            result = run_fixation_population_pca_analysis(settings)

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

            out_root = analysis_root / "ephys/psth/fixation_population_pca"
            self.assertTrue((out_root / "pca_fit_summary.csv").exists())
            self.assertTrue((out_root / "concatenated_pc_timecourses.csv").exists())
            self.assertTrue((out_root / "cross_condition_explained_variance.csv").exists())
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


if __name__ == "__main__":
    unittest.main()
