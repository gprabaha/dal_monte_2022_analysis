"""Regression tests for ROI-vs-period factorial window-collapsing behavior."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_roi_vs_period_factorial import (
    CELL_MEAN_UNIT_RANGE_NORM_AXIS_SOURCE,
    FixationROIVsPeriodFactorialSettings,
    _append_normalized_cell_mean_axis_rows,
    _attach_normalized_axis_values_to_window_summary,
    _build_axis_magnitude_input_table,
    _build_unit_axis_normalization_table,
    _build_unit_axis_collapsed_magnitude_table,
    _build_unit_axis_significance_table,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_roi_vs_period_factorial import (
    FixationROIVsPeriodFactorialPlotSettings,
    _extract_axis_magnitude_units,
    _extract_axis_signed_units,
)


def _analysis_settings() -> FixationROIVsPeriodFactorialSettings:
    return FixationROIVsPeriodFactorialSettings(
        cfg_path="unused.yaml",
        axis_comparison_mode="max_abs_across_windows",
        significance_windows=("pre_fix", "peri_fix", "post_fix"),
        windows_ms={
            "pre_fix": (-500.0, 0.0),
            "peri_fix": (-250.0, 250.0),
            "post_fix": (0.0, 500.0),
        },
    )


def _plot_settings() -> FixationROIVsPeriodFactorialPlotSettings:
    return FixationROIVsPeriodFactorialPlotSettings(
        cfg_path="unused.yaml",
        axis_comparison_mode="max_abs_across_windows",
        axis_magnitude_source="cell_means",
    )


class TestFixationROIVsPeriodFactorialCollapsedWindows(unittest.TestCase):
    """Checks max-window selection and plotting extraction for collapsed unit axes."""

    def test_collapsed_table_uses_max_abs_window_and_keeps_non_significant_axes(self) -> None:
        settings = _analysis_settings()
        unit_term_df = pd.DataFrame(
            [
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "axis_name": "face_object",
                    "window_name": "pre_fix",
                    "glm_testable": True,
                    "counts_toward_significance": True,
                    "significant_raw": True,
                    "significant_within_unit": True,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "axis_name": "face_object",
                    "window_name": "peri_fix",
                    "glm_testable": True,
                    "counts_toward_significance": True,
                    "significant_raw": False,
                    "significant_within_unit": False,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "axis_name": "face_object",
                    "window_name": "post_fix",
                    "glm_testable": True,
                    "counts_toward_significance": True,
                    "significant_raw": False,
                    "significant_within_unit": False,
                },
                {
                    "unit_key": "20200101|unit_b",
                    "date": "20200101",
                    "unit_uuid": "unit_b",
                    "region": "ACC",
                    "axis_name": "interactive_state",
                    "window_name": "pre_fix",
                    "glm_testable": True,
                    "counts_toward_significance": True,
                    "significant_raw": False,
                    "significant_within_unit": False,
                },
                {
                    "unit_key": "20200101|unit_b",
                    "date": "20200101",
                    "unit_uuid": "unit_b",
                    "region": "ACC",
                    "axis_name": "interactive_state",
                    "window_name": "peri_fix",
                    "glm_testable": True,
                    "counts_toward_significance": True,
                    "significant_raw": False,
                    "significant_within_unit": False,
                },
                {
                    "unit_key": "20200101|unit_b",
                    "date": "20200101",
                    "unit_uuid": "unit_b",
                    "region": "ACC",
                    "axis_name": "interactive_state",
                    "window_name": "post_fix",
                    "glm_testable": True,
                    "counts_toward_significance": True,
                    "significant_raw": False,
                    "significant_within_unit": False,
                },
            ]
        )
        unit_axis_df = pd.DataFrame(
            [
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "n_sessions": 2,
                    "window_name": "pre_fix",
                    "window_start_ms": -500.0,
                    "window_stop_ms": 0.0,
                    "axis_name": "face_object",
                    "axis_source": "glm_coef",
                    "value_signed": 0.5,
                    "value_abs": 0.5,
                    "p_value": 0.01,
                    "glm_testable": True,
                    "n_trials_total": 16,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "n_sessions": 2,
                    "window_name": "peri_fix",
                    "window_start_ms": -250.0,
                    "window_stop_ms": 250.0,
                    "axis_name": "face_object",
                    "axis_source": "glm_coef",
                    "value_signed": 0.9,
                    "value_abs": 0.9,
                    "p_value": 0.20,
                    "glm_testable": True,
                    "n_trials_total": 16,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "n_sessions": 2,
                    "window_name": "post_fix",
                    "window_start_ms": 0.0,
                    "window_stop_ms": 500.0,
                    "axis_name": "face_object",
                    "axis_source": "glm_coef",
                    "value_signed": 0.2,
                    "value_abs": 0.2,
                    "p_value": 0.30,
                    "glm_testable": True,
                    "n_trials_total": 16,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "n_sessions": 2,
                    "window_name": "pre_fix",
                    "window_start_ms": -500.0,
                    "window_stop_ms": 0.0,
                    "axis_name": "face_object",
                    "axis_source": "cell_means",
                    "value_signed": 1.0,
                    "value_abs": 1.0,
                    "p_value": np.nan,
                    "glm_testable": True,
                    "n_trials_total": 16,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "n_sessions": 2,
                    "window_name": "peri_fix",
                    "window_start_ms": -250.0,
                    "window_stop_ms": 250.0,
                    "axis_name": "face_object",
                    "axis_source": "cell_means",
                    "value_signed": 1.5,
                    "value_abs": 1.5,
                    "p_value": np.nan,
                    "glm_testable": True,
                    "n_trials_total": 16,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "n_sessions": 2,
                    "window_name": "post_fix",
                    "window_start_ms": 0.0,
                    "window_stop_ms": 500.0,
                    "axis_name": "face_object",
                    "axis_source": "cell_means",
                    "value_signed": -2.0,
                    "value_abs": 2.0,
                    "p_value": np.nan,
                    "glm_testable": True,
                    "n_trials_total": 16,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_b",
                    "date": "20200101",
                    "unit_uuid": "unit_b",
                    "region": "ACC",
                    "n_sessions": 1,
                    "window_name": "pre_fix",
                    "window_start_ms": -500.0,
                    "window_stop_ms": 0.0,
                    "axis_name": "interactive_state",
                    "axis_source": "glm_coef",
                    "value_signed": 0.1,
                    "value_abs": 0.1,
                    "p_value": 0.50,
                    "glm_testable": True,
                    "n_trials_total": 12,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_b",
                    "date": "20200101",
                    "unit_uuid": "unit_b",
                    "region": "ACC",
                    "n_sessions": 1,
                    "window_name": "peri_fix",
                    "window_start_ms": -250.0,
                    "window_stop_ms": 250.0,
                    "axis_name": "interactive_state",
                    "axis_source": "glm_coef",
                    "value_signed": -0.7,
                    "value_abs": 0.7,
                    "p_value": 0.60,
                    "glm_testable": True,
                    "n_trials_total": 12,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_b",
                    "date": "20200101",
                    "unit_uuid": "unit_b",
                    "region": "ACC",
                    "n_sessions": 1,
                    "window_name": "post_fix",
                    "window_start_ms": 0.0,
                    "window_stop_ms": 500.0,
                    "axis_name": "interactive_state",
                    "axis_source": "glm_coef",
                    "value_signed": 0.3,
                    "value_abs": 0.3,
                    "p_value": 0.70,
                    "glm_testable": True,
                    "n_trials_total": 12,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_b",
                    "date": "20200101",
                    "unit_uuid": "unit_b",
                    "region": "ACC",
                    "n_sessions": 1,
                    "window_name": "pre_fix",
                    "window_start_ms": -500.0,
                    "window_stop_ms": 0.0,
                    "axis_name": "interactive_state",
                    "axis_source": "cell_means",
                    "value_signed": 0.2,
                    "value_abs": 0.2,
                    "p_value": np.nan,
                    "glm_testable": True,
                    "n_trials_total": 12,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_b",
                    "date": "20200101",
                    "unit_uuid": "unit_b",
                    "region": "ACC",
                    "n_sessions": 1,
                    "window_name": "peri_fix",
                    "window_start_ms": -250.0,
                    "window_stop_ms": 250.0,
                    "axis_name": "interactive_state",
                    "axis_source": "cell_means",
                    "value_signed": -0.8,
                    "value_abs": 0.8,
                    "p_value": np.nan,
                    "glm_testable": True,
                    "n_trials_total": 12,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_b",
                    "date": "20200101",
                    "unit_uuid": "unit_b",
                    "region": "ACC",
                    "n_sessions": 1,
                    "window_name": "post_fix",
                    "window_start_ms": 0.0,
                    "window_stop_ms": 500.0,
                    "axis_name": "interactive_state",
                    "axis_source": "cell_means",
                    "value_signed": 0.4,
                    "value_abs": 0.4,
                    "p_value": np.nan,
                    "glm_testable": True,
                    "n_trials_total": 12,
                    "counts_toward_significance": True,
                },
            ]
        )

        sig_df = _build_unit_axis_significance_table(unit_term_df, settings=settings)
        collapsed_df = _build_unit_axis_collapsed_magnitude_table(
            unit_axis_df,
            sig_df,
            settings=settings,
        )
        mag_df = _build_axis_magnitude_input_table(
            unit_axis_df,
            settings=settings,
            unit_axis_collapsed_df=collapsed_df,
        )

        self.assertEqual(len(sig_df), 2)
        self.assertEqual(len(collapsed_df), 4)
        self.assertEqual(set(collapsed_df["window_name"].astype(str)), {"max_abs_across_windows"})

        glm_sig = collapsed_df.loc[
            (collapsed_df["unit_key"].astype(str) == "20200101|unit_a")
            & (collapsed_df["axis_name"].astype(str) == "face_object")
            & (collapsed_df["axis_source"].astype(str) == "glm_coef")
        ].iloc[0]
        self.assertTrue(bool(glm_sig["is_significant_axis"]))
        self.assertEqual(str(glm_sig["significant_windows"]), "pre_fix")
        self.assertEqual(str(glm_sig["selected_window_name"]), "peri_fix")
        self.assertAlmostEqual(float(glm_sig["value_signed"]), 0.9, places=6)
        self.assertFalse(bool(glm_sig["selected_window_is_significant"]))

        means_sig = collapsed_df.loc[
            (collapsed_df["unit_key"].astype(str) == "20200101|unit_a")
            & (collapsed_df["axis_name"].astype(str) == "face_object")
            & (collapsed_df["axis_source"].astype(str) == "cell_means")
        ].iloc[0]
        self.assertTrue(bool(means_sig["is_significant_axis"]))
        self.assertEqual(str(means_sig["selected_window_name"]), "post_fix")
        self.assertAlmostEqual(float(means_sig["value_signed"]), -2.0, places=6)
        self.assertAlmostEqual(float(means_sig["value_abs"]), 2.0, places=6)

        means_non_sig = collapsed_df.loc[
            (collapsed_df["unit_key"].astype(str) == "20200101|unit_b")
            & (collapsed_df["axis_name"].astype(str) == "interactive_state")
            & (collapsed_df["axis_source"].astype(str) == "cell_means")
        ].iloc[0]
        self.assertFalse(bool(means_non_sig["is_significant_axis"]))
        self.assertEqual(str(means_non_sig["selected_window_name"]), "peri_fix")
        self.assertAlmostEqual(float(means_non_sig["value_signed"]), -0.8, places=6)
        self.assertAlmostEqual(float(means_non_sig["value_abs"]), 0.8, places=6)

        self.assertEqual(len(mag_df), 2)
        self.assertEqual(set(mag_df["window_name"].astype(str)), {"max_abs_across_windows"})
        mag_lookup = {
            (str(row.unit_key), str(row.axis_name)): float(row.value_abs_norm)
            for row in mag_df.itertuples(index=False)
        }
        self.assertAlmostEqual(mag_lookup[("20200101|unit_a", "face_object")], 2.0, places=6)
        self.assertAlmostEqual(mag_lookup[("20200101|unit_b", "interactive_state")], 0.8, places=6)

    def test_plot_extractors_prefer_collapsed_rows_for_max_mode(self) -> None:
        payload = {
            "meta": {
                "axis_comparison_mode": "max_abs_across_windows",
                "significance_windows": ["pre_fix", "peri_fix", "post_fix"],
            },
            "unit_axis_values": pd.DataFrame(
                [
                    {
                        "unit_key": "20200101|unit_a",
                        "region": "acc",
                        "axis_name": "face_object",
                        "axis_source": "cell_means",
                        "window_name": "pre_fix",
                        "value_signed": 0.2,
                        "counts_toward_significance": True,
                    },
                    {
                        "unit_key": "20200101|unit_a",
                        "region": "acc",
                        "axis_name": "face_object",
                        "axis_source": "cell_means",
                        "window_name": "peri_fix",
                        "value_signed": 0.3,
                        "counts_toward_significance": True,
                    },
                ]
            ),
            "unit_axis_collapsed": pd.DataFrame(
                [
                    {
                        "unit_key": "20200101|unit_a",
                        "region": "acc",
                        "axis_name": "face_object",
                        "axis_source": "cell_means",
                        "axis_comparison_mode": "max_abs_across_windows",
                        "window_name": "max_abs_across_windows",
                        "selected_window_name": "post_fix",
                        "value_signed": -1.4,
                        "value_abs": 1.4,
                    }
                ]
            ),
        }
        settings = _plot_settings()

        mag_df = _extract_axis_magnitude_units(payload, settings)
        signed_df = _extract_axis_signed_units(payload, settings)

        self.assertEqual(len(mag_df), 1)
        self.assertEqual(len(signed_df), 1)
        self.assertEqual(str(mag_df.iloc[0]["window_name"]), "max_abs_across_windows")
        self.assertEqual(str(mag_df.iloc[0]["selected_window_name"]), "post_fix")
        self.assertAlmostEqual(float(mag_df.iloc[0]["value_abs_norm"]), 1.4, places=6)
        self.assertAlmostEqual(float(signed_df.iloc[0]["value_signed"]), -1.4, places=6)

    def test_unit_range_normalized_cell_mean_source_is_built_in_parallel(self) -> None:
        settings = _analysis_settings()
        unit_term_df = pd.DataFrame(
            [
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "axis_name": "face_object",
                    "window_name": "pre_fix",
                    "glm_testable": True,
                    "counts_toward_significance": True,
                    "significant_raw": True,
                    "significant_within_unit": True,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "axis_name": "face_object",
                    "window_name": "peri_fix",
                    "glm_testable": True,
                    "counts_toward_significance": True,
                    "significant_raw": False,
                    "significant_within_unit": False,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "axis_name": "face_object",
                    "window_name": "post_fix",
                    "glm_testable": True,
                    "counts_toward_significance": True,
                    "significant_raw": False,
                    "significant_within_unit": False,
                },
            ]
        )
        unit_axis_df = pd.DataFrame(
            [
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "window_name": "pre_fix",
                    "axis_name": "face_object",
                    "axis_source": "cell_means",
                    "axis_value_units": "hz_difference",
                    "value_signed": 1.0,
                    "value_abs": 1.0,
                    "p_value": np.nan,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "window_name": "peri_fix",
                    "axis_name": "face_object",
                    "axis_source": "cell_means",
                    "axis_value_units": "hz_difference",
                    "value_signed": 1.5,
                    "value_abs": 1.5,
                    "p_value": np.nan,
                    "counts_toward_significance": True,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "window_name": "post_fix",
                    "axis_name": "face_object",
                    "axis_source": "cell_means",
                    "axis_value_units": "hz_difference",
                    "value_signed": -2.0,
                    "value_abs": 2.0,
                    "p_value": np.nan,
                    "counts_toward_significance": True,
                },
            ]
        )
        unit_window_df = pd.DataFrame(
            [
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "window_name": "pre_fix",
                    "mean_fr_face_interactive_hz": 1.0,
                    "mean_fr_face_non_interactive_hz": 2.0,
                    "mean_fr_object_interactive_hz": 3.0,
                    "mean_fr_object_non_interactive_hz": 4.0,
                    "axis_face_object_from_means": 1.0,
                    "axis_interactive_state_from_means": 0.5,
                    "axis_cross_interaction_from_means": -0.5,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "window_name": "peri_fix",
                    "mean_fr_face_interactive_hz": 2.0,
                    "mean_fr_face_non_interactive_hz": 3.0,
                    "mean_fr_object_interactive_hz": 4.0,
                    "mean_fr_object_non_interactive_hz": 5.0,
                    "axis_face_object_from_means": 1.5,
                    "axis_interactive_state_from_means": 0.75,
                    "axis_cross_interaction_from_means": -0.75,
                },
                {
                    "unit_key": "20200101|unit_a",
                    "date": "20200101",
                    "unit_uuid": "unit_a",
                    "region": "ACC",
                    "window_name": "post_fix",
                    "mean_fr_face_interactive_hz": 1.5,
                    "mean_fr_face_non_interactive_hz": 2.5,
                    "mean_fr_object_interactive_hz": 3.5,
                    "mean_fr_object_non_interactive_hz": 5.0,
                    "axis_face_object_from_means": -2.0,
                    "axis_interactive_state_from_means": 1.0,
                    "axis_cross_interaction_from_means": -1.0,
                },
            ]
        )

        norm_df = _build_unit_axis_normalization_table(
            unit_window_df,
            significance_windows=settings.significance_windows,
        )
        self.assertEqual(len(norm_df), 1)
        self.assertAlmostEqual(float(norm_df.iloc[0]["unit_axis_normalization_scale_hz"]), 4.0, places=6)

        unit_window_aug = _attach_normalized_axis_values_to_window_summary(unit_window_df, norm_df)
        self.assertIn("axis_face_object_from_means_unit_range_norm", unit_window_aug.columns)
        post_row = unit_window_aug.loc[unit_window_aug["window_name"].astype(str) == "post_fix"].iloc[0]
        self.assertAlmostEqual(float(post_row["axis_face_object_from_means_unit_range_norm"]), -0.5, places=6)

        unit_axis_aug = _append_normalized_cell_mean_axis_rows(unit_axis_df, norm_df)
        self.assertEqual(
            set(unit_axis_aug["axis_source"].astype(str)),
            {"cell_means", str(CELL_MEAN_UNIT_RANGE_NORM_AXIS_SOURCE)},
        )

        sig_df = _build_unit_axis_significance_table(unit_term_df, settings=settings)
        collapsed_df = _build_unit_axis_collapsed_magnitude_table(
            unit_axis_aug,
            sig_df,
            settings=settings,
        )
        norm_row = collapsed_df.loc[
            (collapsed_df["unit_key"].astype(str) == "20200101|unit_a")
            & (collapsed_df["axis_name"].astype(str) == "face_object")
            & (collapsed_df["axis_source"].astype(str) == str(CELL_MEAN_UNIT_RANGE_NORM_AXIS_SOURCE))
        ].iloc[0]
        self.assertEqual(str(norm_row["selected_window_name"]), "post_fix")
        self.assertAlmostEqual(float(norm_row["value_signed"]), -0.5, places=6)
        self.assertAlmostEqual(float(norm_row["value_abs"]), 0.5, places=6)

        payload = {
            "meta": {
                "axis_comparison_mode": "max_abs_across_windows",
                "significance_windows": ["pre_fix", "peri_fix", "post_fix"],
            },
            "unit_axis_collapsed": collapsed_df,
        }
        plot_settings = FixationROIVsPeriodFactorialPlotSettings(
            cfg_path="unused.yaml",
            axis_comparison_mode="max_abs_across_windows",
            axis_magnitude_source=str(CELL_MEAN_UNIT_RANGE_NORM_AXIS_SOURCE),
        )
        mag_df = _extract_axis_magnitude_units(payload, plot_settings)
        signed_df = _extract_axis_signed_units(payload, plot_settings)
        self.assertEqual(len(mag_df), 1)
        self.assertEqual(len(signed_df), 1)
        self.assertAlmostEqual(float(mag_df.iloc[0]["value_abs_norm"]), 0.5, places=6)
        self.assertAlmostEqual(float(signed_df.iloc[0]["value_signed"]), -0.5, places=6)


if __name__ == "__main__":
    unittest.main()
