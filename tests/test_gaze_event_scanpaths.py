"""Tests for scanpath-style gaze event plotting helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

from dal_monte_2022_analysis.runtime.io.gaze_event_qc import (
    AgentGazeEventArtifacts,
    find_paired_gaze_event_sessions,
    sample_random_paired_gaze_event_sessions,
)
from dal_monte_2022_analysis.behav.plotting.gaze_event_scanpaths import (
    compute_fixation_centers,
    compute_saccade_segments,
    plot_agent_gaze_event_scanpath,
    plot_gaze_event_example_sessions,
    resolve_scanpath_bounds,
)
from dal_monte_2022_analysis.data.records.behavioral import (
    BehaviorRunContext,
    PositionData,
    ROIRectsData,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class TestGazeEventScanpaths(unittest.TestCase):
    """Coverage for gaze-event scanpath sampling and rendering."""

    def setUp(self) -> None:
        context = BehaviorRunContext(
            date="20200101",
            session="1",
            agent="m1",
            monkey_name="Kuro",
        )
        self.position = PositionData(
            context=context,
            x=np.array([10.0, 12.0, np.nan, 18.0, 20.0, 22.0], dtype=float),
            y=np.array([50.0, 54.0, np.nan, 60.0, 63.0, 66.0], dtype=float),
        )
        self.fixations = pd.DataFrame(
            {
                "date": ["20200101", "20200101"],
                "session": ["1", "1"],
                "agent": ["m1", "m1"],
                "monkey_name": ["Kuro", "Kuro"],
                "start": [0, 3],
                "stop": [1, 5],
            }
        )
        self.saccades = pd.DataFrame(
            {
                "date": ["20200101", "20200101"],
                "session": ["1", "1"],
                "agent": ["m1", "m1"],
                "monkey_name": ["Kuro", "Kuro"],
                "start": [0, 2],
                "stop": [1, 5],
            }
        )
        self.rois = ROIRectsData(
            context=context,
            rois={
                "face": np.array([0.0, 40.0, 30.0, 80.0], dtype=float),
                "mouth": np.array([100.0, 200.0, 300.0, 400.0], dtype=float),
            },
        )

    def test_compute_fixation_centers_uses_mean_valid_positions(self) -> None:
        centers = compute_fixation_centers(self.position, self.fixations)
        np.testing.assert_allclose(
            centers,
            np.array([[11.0, 52.0], [20.0, 63.0]], dtype=float),
        )

    def test_compute_saccade_segments_uses_first_and_last_valid_positions(self) -> None:
        segments = compute_saccade_segments(self.position, self.saccades)
        np.testing.assert_allclose(
            segments,
            np.array(
                [
                    [[10.0, 50.0], [12.0, 54.0]],
                    [[18.0, 60.0], [22.0, 66.0]],
                ],
                dtype=float,
            ),
        )

    def test_find_paired_gaze_event_sessions_requires_all_agents_and_modalities(self) -> None:
        modality_rows = {
            modality: pd.DataFrame(
                [
                    {"date": "20200101", "session": "1", "agent": "m1"},
                    {"date": "20200101", "session": "1", "agent": "m2"},
                    {"date": "20200102", "session": "2", "agent": "m1"},
                ]
            )
            for modality in ("gaze_position", "fixations", "saccades", "roi_vertices")
        }

        with patch(
            "dal_monte_2022_analysis.runtime.io.gaze_event_qc.scan_processed_paths",
            side_effect=lambda cfg, modality, agents=None: modality_rows[modality].to_dict("records"),
        ):
            sessions = find_paired_gaze_event_sessions({"processed_data_root": "/tmp"})

        self.assertEqual(
            sessions.to_dict("records"),
            [{"date": "20200101", "session": "1"}],
        )

    def test_sample_random_paired_gaze_event_sessions_returns_requested_count(self) -> None:
        modality_rows = {
            modality: pd.DataFrame(
                [
                    {"date": "20200101", "session": "1", "agent": "m1"},
                    {"date": "20200101", "session": "1", "agent": "m2"},
                    {"date": "20200102", "session": "2", "agent": "m1"},
                    {"date": "20200102", "session": "2", "agent": "m2"},
                    {"date": "20200103", "session": "3", "agent": "m1"},
                    {"date": "20200103", "session": "3", "agent": "m2"},
                ]
            )
            for modality in ("gaze_position", "fixations", "saccades", "roi_vertices")
        }

        with patch(
            "dal_monte_2022_analysis.runtime.io.gaze_event_qc.scan_processed_paths",
            side_effect=lambda cfg, modality, agents=None: modality_rows[modality].to_dict("records"),
        ):
            sessions = sample_random_paired_gaze_event_sessions(
                {"processed_data_root": "/tmp"},
                n_sessions=2,
                random_state=7,
            )

        self.assertEqual(len(sessions), 2)
        self.assertEqual(list(sessions.columns), ["date", "session"])

    def test_plot_agent_gaze_event_scanpath_draws_rois_lines_and_points(self) -> None:
        payload = AgentGazeEventArtifacts(
            position=self.position,
            fixations=self.fixations,
            saccades=self.saccades,
            rois=self.rois,
        )
        fig, ax = plt.subplots(figsize=(4, 3))
        try:
            returned_ax = plot_agent_gaze_event_scanpath(
                payload,
                ax=ax,
                encode_event_order=True,
                label_rois=True,
            )
            self.assertIs(returned_ax, ax)
            self.assertTrue(ax.yaxis_inverted())
            self.assertEqual(len(ax.patches), len(self.rois.rois))
            self.assertTrue(any(isinstance(item, LineCollection) for item in ax.collections))
            self.assertGreaterEqual(len(ax.texts), 1)
        finally:
            plt.close(fig)

    def test_plot_agent_gaze_event_scanpath_expands_roi_boxes(self) -> None:
        payload = AgentGazeEventArtifacts(
            position=self.position,
            fixations=self.fixations,
            saccades=self.saccades,
            rois=self.rois,
        )
        fig, ax = plt.subplots(figsize=(4, 3))
        try:
            plot_agent_gaze_event_scanpath(
                payload,
                ax=ax,
                encode_event_order=False,
                roi_expansion_fraction=0.2,
            )
            mouth_patch = next(
                patch
                for patch in ax.patches
                if np.isclose(patch.get_x(), 80.0) and np.isclose(patch.get_y(), 180.0)
            )
            self.assertAlmostEqual(mouth_patch.get_width(), 240.0)
            self.assertAlmostEqual(mouth_patch.get_height(), 240.0)
        finally:
            plt.close(fig)

    def test_plot_agent_gaze_event_scanpath_clips_full_data_to_crop_window(self) -> None:
        outlier_position = PositionData(
            context=self.position.context,
            x=np.array([200.0, 220.0, 6000.0, 6200.0], dtype=float),
            y=np.array([300.0, 320.0, 300.0, 320.0], dtype=float),
        )
        outlier_fixations = pd.DataFrame(
            {
                "date": ["20200101", "20200101"],
                "session": ["1", "1"],
                "agent": ["m1", "m1"],
                "monkey_name": ["Kuro", "Kuro"],
                "start": [0, 2],
                "stop": [1, 3],
            }
        )
        outlier_saccades = pd.DataFrame(
            {
                "date": ["20200101", "20200101"],
                "session": ["1", "1"],
                "agent": ["m1", "m1"],
                "monkey_name": ["Kuro", "Kuro"],
                "start": [0, 2],
                "stop": [1, 3],
            }
        )
        payload = AgentGazeEventArtifacts(
            position=outlier_position,
            fixations=outlier_fixations,
            saccades=outlier_saccades,
            rois=self.rois,
        )
        fig, ax = plt.subplots(figsize=(4, 3))
        try:
            plot_agent_gaze_event_scanpath(payload, ax=ax, encode_event_order=False)
            line_collection = next(
                item for item in ax.collections if isinstance(item, LineCollection)
            )
            np.testing.assert_allclose(
                line_collection.get_segments(),
                np.array(
                    [
                        [[200.0, 300.0], [220.0, 320.0]],
                        [[6000.0, 300.0], [6200.0, 320.0]],
                    ],
                    dtype=float,
                ),
            )
            scatter = next(
                item for item in ax.collections if not isinstance(item, LineCollection)
            )
            np.testing.assert_allclose(
                scatter.get_offsets(),
                np.array([[210.0, 310.0], [6100.0, 310.0]], dtype=float),
            )
            self.assertEqual(ax.get_xlim(), (-1000.0, 1400.0))
            self.assertEqual(ax.get_ylim(), (1300.0, -700.0))
            self.assertTrue(line_collection.get_clip_on())
            self.assertTrue(scatter.get_clip_on())
        finally:
            plt.close(fig)

    def test_resolve_scanpath_bounds_uses_fixed_window_around_event_center(self) -> None:
        fixation_centers = compute_fixation_centers(self.position, self.fixations)
        saccade_segments = compute_saccade_segments(self.position, self.saccades)
        bounds = resolve_scanpath_bounds(
            fixation_centers,
            saccade_segments,
            self.rois,
        )
        expected_center = np.array([200.0, 300.0], dtype=float)
        actual_center = np.array(
            [
                (bounds[0] + bounds[1]) / 2.0,
                (bounds[2] + bounds[3]) / 2.0,
            ],
            dtype=float,
        )
        np.testing.assert_allclose(actual_center, expected_center)
        self.assertEqual(bounds[1] - bounds[0], 2400.0)
        self.assertEqual(bounds[3] - bounds[2], 2000.0)

    def test_plot_gaze_event_example_sessions_uses_detection_cfg_roi_expansion(self) -> None:
        payload = AgentGazeEventArtifacts(
            position=self.position,
            fixations=self.fixations,
            saccades=self.saccades,
            rois=self.rois,
        )
        session_payload = type(
            "SessionPayload",
            (),
            {
                "key": type(
                    "SessionKeyPayload",
                    (),
                    {"date": "20200101", "session": "1"},
                )(),
                "agents": {"m1": payload},
            },
        )()
        sessions = pd.DataFrame([{"date": "20200101", "session": "1"}])

        with patch(
            "dal_monte_2022_analysis.behav.plotting.gaze_event_scanpaths.load_gaze_event_session_artifacts",
            return_value=session_payload,
        ):
            fig, axes = plot_gaze_event_example_sessions(
                {"processed_data_root": "/tmp"},
                sessions,
                agents=("m1",),
                detection_cfg_or_path={"roi_assignment_expansion_fraction": 0.2},
            )
        try:
            mouth_patch = next(
                patch
                for patch in axes[0, 0].patches
                if np.isclose(patch.get_x(), 80.0) and np.isclose(patch.get_y(), 180.0)
            )
            self.assertAlmostEqual(mouth_patch.get_width(), 240.0)
            self.assertAlmostEqual(mouth_patch.get_height(), 240.0)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
