"""Regression tests for fixation detection core logic."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.core.behav.fixation_detection import (
    FixationDetectionConfig,
    coerce_fixation_detection_config,
    detect_fixations_and_saccades,
)
from dal_monte_2022_analysis.behav.features.gaze_event_detection import (
    annotate_fixation_locations,
    annotate_saccade_from_to,
    build_gaze_event_detection_settings,
)


class TestFixationDetection(unittest.TestCase):
    """Regression checks for public fixation-detection behavior."""

    def test_short_input_returns_empty_arrays(self) -> None:
        positions = np.zeros((100, 2), dtype=float)
        fix, sacc = detect_fixations_and_saccades(positions)
        self.assertEqual(fix.shape, (0, 2))
        self.assertEqual(sacc.shape, (0, 2))

    def test_config_defaults_to_one_khz_sampling_rate(self) -> None:
        cfg = coerce_fixation_detection_config()
        self.assertEqual(cfg.default_sampling_rate_hz, 1000.0)

    def test_config_mapping_is_coerced_and_validated(self) -> None:
        cfg = coerce_fixation_detection_config(
            {
                "default_sampling_rate_hz": 500.0,
                "minimum_input_duration_ms": 250.0,
                "global_kmeans_k_min": 3,
                "global_kmeans_k_max": 6,
            }
        )
        self.assertIsInstance(cfg, FixationDetectionConfig)
        self.assertEqual(cfg.default_sampling_rate_hz, 500.0)
        self.assertEqual(cfg.minimum_input_duration_ms, 250.0)
        self.assertEqual(cfg.global_kmeans_k_min, 3)
        self.assertEqual(cfg.global_kmeans_k_max, 6)

    def test_gaze_event_settings_embed_fixation_detection_config(self) -> None:
        settings = build_gaze_event_detection_settings(
            "configs/dataset.yaml",
            {
                "roi_assignment_expansion_fraction": 0.35,
                "fixation_detection": {
                    "default_sampling_rate_hz": 500.0,
                    "global_kmeans_k_min": 3,
                }
            },
        )
        self.assertEqual(settings.roi_assignment_expansion_fraction, 0.35)
        self.assertEqual(settings.fixation_detection.default_sampling_rate_hz, 500.0)
        self.assertEqual(settings.fixation_detection.global_kmeans_k_min, 3)

    def test_annotate_fixation_locations_expands_roi_assignment_window(self) -> None:
        pos_data = SimpleNamespace(
            x=np.array([10.8, 10.8], dtype=float),
            y=np.array([5.0, 5.0], dtype=float),
        )
        roi_data = SimpleNamespace(
            rois={"mouth": np.array([0.0, 0.0, 10.0, 10.0], dtype=float)}
        )
        fix_df = pd.DataFrame({"start": [0], "stop": [1]})
        row = {"date": "01012018", "session": "1"}

        with patch(
            "dal_monte_2022_analysis.behav.features.gaze_event_detection.load_config",
            return_value={},
        ), patch(
            "dal_monte_2022_analysis.behav.features.gaze_event_detection._load_positions",
            return_value=pos_data,
        ), patch(
            "dal_monte_2022_analysis.behav.features.gaze_event_detection.load_processed_pickle",
            return_value=roi_data,
        ):
            unexpanded = annotate_fixation_locations(
                "configs/dataset.yaml",
                row,
                "m1",
                fix_df,
                roi_expansion_fraction=0.0,
            )
            expanded = annotate_fixation_locations(
                "configs/dataset.yaml",
                row,
                "m1",
                fix_df,
                roi_expansion_fraction=0.2,
            )

        self.assertEqual(unexpanded.at[0, "location"], ["out_of_roi"])
        self.assertEqual(expanded.at[0, "location"], ["mouth"])

    def test_annotate_saccade_from_to_expands_roi_assignment_window(self) -> None:
        pos_data = SimpleNamespace(
            x=np.array([10.8, -0.8], dtype=float),
            y=np.array([5.0, 5.0], dtype=float),
        )
        roi_data = SimpleNamespace(
            rois={"mouth": np.array([0.0, 0.0, 10.0, 10.0], dtype=float)}
        )
        sacc_df = pd.DataFrame({"start": [0], "stop": [1]})
        row = {"date": "01012018", "session": "1"}

        with patch(
            "dal_monte_2022_analysis.behav.features.gaze_event_detection.load_config",
            return_value={},
        ), patch(
            "dal_monte_2022_analysis.behav.features.gaze_event_detection._load_positions",
            return_value=pos_data,
        ), patch(
            "dal_monte_2022_analysis.behav.features.gaze_event_detection.load_processed_pickle",
            return_value=roi_data,
        ):
            unexpanded = annotate_saccade_from_to(
                "configs/dataset.yaml",
                row,
                "m1",
                sacc_df,
                roi_expansion_fraction=0.0,
            )
            expanded = annotate_saccade_from_to(
                "configs/dataset.yaml",
                row,
                "m1",
                sacc_df,
                roi_expansion_fraction=0.2,
            )

        self.assertEqual(unexpanded.at[0, "from"], ["out_of_roi"])
        self.assertEqual(unexpanded.at[0, "to"], ["out_of_roi"])
        self.assertEqual(expanded.at[0, "from"], ["mouth"])
        self.assertEqual(expanded.at[0, "to"], ["mouth"])

    def test_long_input_returns_well_formed_intervals(self) -> None:
        rng = np.random.default_rng(13)
        steps = rng.normal(loc=0.0, scale=0.4, size=(900, 2))
        positions = np.cumsum(steps, axis=0)

        fix, sacc = detect_fixations_and_saccades(positions)

        self.assertEqual(fix.ndim, 2)
        self.assertEqual(sacc.ndim, 2)
        self.assertEqual(fix.shape[1], 2)
        self.assertEqual(sacc.shape[1], 2)
        self.assertTrue(np.issubdtype(fix.dtype, np.integer))
        self.assertTrue(np.issubdtype(sacc.dtype, np.integer))
        if fix.size:
            self.assertTrue(np.all(fix[:, 0] <= fix[:, 1]))
        if sacc.size:
            self.assertTrue(np.all(sacc[:, 0] <= sacc[:, 1]))

    def test_explicit_sampling_rate_keeps_detector_compatible(self) -> None:
        rng = np.random.default_rng(7)
        steps = rng.normal(loc=0.0, scale=0.3, size=(180, 2))
        positions = np.cumsum(steps, axis=0)

        fix, sacc = detect_fixations_and_saccades(positions, sampling_rate_hz=200.0)

        self.assertEqual(fix.ndim, 2)
        self.assertEqual(sacc.ndim, 2)
        self.assertEqual(fix.shape[1], 2)
        self.assertEqual(sacc.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
