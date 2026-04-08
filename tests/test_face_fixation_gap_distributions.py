"""Regression tests for face-fixation gap distribution analysis and plotting."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd

from dal_monte_2022_analysis.behav.analysis.face_fixation_gap_distributions import (
    FaceFixationGapDistributionSettings,
    run_face_fixation_gap_distribution_analysis,
)
from dal_monte_2022_analysis.utils.io import save_pickle

try:
    import matplotlib

    matplotlib.use("Agg")

    from dal_monte_2022_analysis.behav.plotting.face_fixation_gap_distributions import (
        FaceFixationGapDistributionPlotSettings,
        plot_face_fixation_gap_distribution_figures,
    )

    _HAS_MPL = True
except ModuleNotFoundError:
    _HAS_MPL = False


def _write_yaml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")


class TestFaceFixationGapDistributions(unittest.TestCase):
    """Checks the saved gap distributions for the requested face-fixation views."""

    def test_analysis_writes_expected_m1_and_cross_monkey_gap_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            dataset_cfg_path = root / "configs" / "dataset.yaml"

            _write_yaml(
                dataset_cfg_path,
                f"""
                raw_data_root: {root / "raw"}
                processed_data_root: {processed_root}
                analysis_output_root: {analysis_root}
                agents: ["m1", "m2"]
                processed_data_layout:
                  pattern: "date={{date}}/session={{session}}/{{modality}}"
                """,
            )

            m1_fix_df = pd.DataFrame(
                [
                    {"start": 10, "stop": 20, "location": "face", "monkey_name": "Alpha"},
                    {"start": 40, "stop": 50, "location": "eyes_nf", "monkey_name": "Alpha"},
                    {"start": 230, "stop": 240, "location": "face", "monkey_name": "Alpha"},
                    {"start": 280, "stop": 290, "location": "mouth", "monkey_name": "Alpha"},
                    {"start": 450, "stop": 460, "location": "face", "monkey_name": "Alpha"},
                    {"start": 500, "stop": 520, "location": "face", "monkey_name": "Alpha"},
                    {"start": 610, "stop": 620, "location": "object", "monkey_name": "Alpha"},
                ]
            )
            m2_fix_df = pd.DataFrame(
                [
                    {"start": 25, "stop": 35, "location": "face", "monkey_name": "Bravo"},
                    {"start": 70, "stop": 80, "location": "face", "monkey_name": "Bravo"},
                    {"start": 470, "stop": 475, "location": "face", "monkey_name": "Bravo"},
                    {"start": 330, "stop": 340, "location": "object", "monkey_name": "Bravo"},
                ]
            )
            periods_df = pd.DataFrame(
                [
                    {"start": 0, "stop": 199, "state": "interactive"},
                    {"start": 200, "stop": 399, "state": "non_interactive"},
                    {"start": 400, "stop": 699, "state": "interactive"},
                ]
            )

            save_pickle(
                m1_fix_df,
                processed_root / "date=01012020/session=1/fixations/agent=m1.pkl",
            )
            save_pickle(
                m2_fix_df,
                processed_root / "date=01012020/session=1/fixations/agent=m2.pkl",
            )
            save_pickle(
                periods_df,
                processed_root / "date=01012020/session=1/interactive_periods/shared.pkl",
            )

            settings = FaceFixationGapDistributionSettings(
                cfg_path=str(dataset_cfg_path),
                max_pair_gap_ms=45.0,
            )
            m1_path, m1_m2_path, summary_path = run_face_fixation_gap_distribution_analysis(
                settings
            )

            self.assertTrue(m1_path.exists())
            self.assertTrue(m1_m2_path.exists())
            self.assertTrue(summary_path.exists())

            m1_out = pd.read_csv(m1_path)
            m1_m2_out = pd.read_csv(m1_m2_path)
            summary_out = pd.read_csv(summary_path)

            self.assertEqual(len(m1_out), 2)
            self.assertEqual(len(m1_m2_out), 10)

            interactive_start = sorted(
                m1_out.loc[
                    (m1_out["period_state"] == "interactive")
                    & (m1_out["gap_metric"] == "start_to_start"),
                    "gap_ms",
                ].tolist()
            )
            interactive_stop = sorted(
                m1_out.loc[
                    (m1_out["period_state"] == "interactive")
                    & (m1_out["gap_metric"] == "stop_to_start"),
                    "gap_ms",
                ].tolist()
            )
            noninteractive_start = m1_out.loc[
                (m1_out["period_state"] == "non_interactive")
                & (m1_out["gap_metric"] == "start_to_start"),
                "gap_ms",
            ].tolist()
            noninteractive_stop = m1_out.loc[
                (m1_out["period_state"] == "non_interactive")
                & (m1_out["gap_metric"] == "stop_to_start"),
                "gap_ms",
            ].tolist()

            self.assertEqual(interactive_start, [30.0])
            self.assertEqual(interactive_stop, [20.0])
            self.assertEqual(noninteractive_start, [])
            self.assertEqual(noninteractive_stop, [])

            cross_start = sorted(
                m1_m2_out.loc[m1_m2_out["gap_metric"] == "start_to_start", "gap_ms"].tolist()
            )
            cross_stop = sorted(
                m1_m2_out.loc[m1_m2_out["gap_metric"] == "stop_to_start", "gap_ms"].tolist()
            )
            transitions = m1_m2_out["transition"].value_counts().to_dict()

            self.assertEqual(cross_start, [15.0, 15.0, 20.0, 30.0, 30.0])
            self.assertEqual(cross_stop, [5.0, 5.0, 10.0, 20.0, 25.0])
            self.assertEqual(transitions, {"m1_to_m2": 6, "m2_to_m1": 4})

            summary_lookup = {
                (row["scope"], row["group_type"], row["group_label"]): row
                for _, row in summary_out.iterrows()
            }
            m1_overall = summary_lookup[("m1_face", "overall", "all")]
            m1_interactive = summary_lookup[("m1_face", "period_state", "interactive")]
            cross_overall = summary_lookup[("interactive_m1_m2_face", "overall", "all")]

            self.assertEqual(int(m1_overall["n_candidate_pairs"]), 3)
            self.assertEqual(int(m1_overall["n_kept_pairs"]), 1)
            self.assertEqual(int(m1_overall["n_discarded_pairs"]), 2)
            self.assertAlmostEqual(float(m1_overall["kept_fraction"]), 1.0 / 3.0)
            self.assertEqual(int(m1_interactive["n_candidate_pairs"]), 2)
            self.assertEqual(int(m1_interactive["n_kept_pairs"]), 1)
            self.assertEqual(int(cross_overall["n_candidate_pairs"]), 5)
            self.assertEqual(int(cross_overall["n_kept_pairs"]), 5)
            self.assertEqual(int(cross_overall["n_discarded_pairs"]), 0)


@unittest.skipUnless(_HAS_MPL, "matplotlib is required for plotting tests")
class TestFaceFixationGapDistributionPlotting(unittest.TestCase):
    """Checks that the two requested figures are written from saved CSV inputs."""

    def test_plotting_writes_both_gap_distribution_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            analysis_root = root / "analysis"
            dataset_cfg_path = root / "configs" / "dataset.yaml"
            plotting_cfg_path = root / "configs" / "plotting.yaml"
            analysis_dir = analysis_root / "face_fixation_gap_distributions"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            _write_yaml(
                dataset_cfg_path,
                f"""
                raw_data_root: {root / "raw"}
                processed_data_root: {root / "processed"}
                analysis_output_root: {analysis_root}
                agents: ["m1", "m2"]
                processed_data_layout:
                  pattern: "date={{date}}/session={{session}}/{{modality}}"
                """,
            )
            _write_yaml(
                plotting_cfg_path,
                """
                figure:
                  dpi: 120
                face_fixation_gap_distribution:
                  m1_figsize: [8.0, 3.4]
                  m1_m2_figsize: [8.0, 3.4]
                  histogram_bins: 20
                """,
            )

            pd.DataFrame(
                [
                    {
                        "date": "01012020",
                        "session": "1",
                        "agent": "m1",
                        "monkey_name": "Alpha",
                        "period_index": 0,
                        "period_state": "interactive",
                        "period_start": 0,
                        "period_stop": 199,
                        "prev_fixation_index": 0,
                        "prev_fixation_start": 10,
                        "prev_fixation_stop": 20,
                        "next_fixation_index": 1,
                        "next_fixation_start": 40,
                        "next_fixation_stop": 50,
                        "gap_metric": "start_to_start",
                        "gap_samples": 30,
                        "gap_ms": 30.0,
                        "sample_rate_hz": 1000.0,
                    },
                    {
                        "date": "01012020",
                        "session": "1",
                        "agent": "m1",
                        "monkey_name": "Alpha",
                        "period_index": 1,
                        "period_state": "non_interactive",
                        "period_start": 200,
                        "period_stop": 399,
                        "prev_fixation_index": 2,
                        "prev_fixation_start": 230,
                        "prev_fixation_stop": 240,
                        "next_fixation_index": 3,
                        "next_fixation_start": 280,
                        "next_fixation_stop": 290,
                        "gap_metric": "stop_to_start",
                        "gap_samples": 40,
                        "gap_ms": 40.0,
                        "sample_rate_hz": 1000.0,
                    },
                ]
            ).to_csv(
                analysis_dir / "within_session_m1_face_fixation_gap_distribution.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "date": "01012020",
                        "session": "1",
                        "period_index": 0,
                        "period_state": "interactive",
                        "period_start": 0,
                        "period_stop": 199,
                        "prev_agent": "m1",
                        "next_agent": "m2",
                        "transition": "m1_to_m2",
                        "prev_monkey_name": "Alpha",
                        "next_monkey_name": "Bravo",
                        "prev_fixation_index": 0,
                        "prev_fixation_start": 10,
                        "prev_fixation_stop": 20,
                        "next_fixation_index": 0,
                        "next_fixation_start": 25,
                        "next_fixation_stop": 35,
                        "gap_metric": "start_to_start",
                        "gap_samples": 15,
                        "gap_ms": 15.0,
                        "sample_rate_hz": 1000.0,
                    },
                    {
                        "date": "01012020",
                        "session": "1",
                        "period_index": 0,
                        "period_state": "interactive",
                        "period_start": 0,
                        "period_stop": 199,
                        "prev_agent": "m1",
                        "next_agent": "m2",
                        "transition": "m1_to_m2",
                        "prev_monkey_name": "Alpha",
                        "next_monkey_name": "Bravo",
                        "prev_fixation_index": 0,
                        "prev_fixation_start": 10,
                        "prev_fixation_stop": 20,
                        "next_fixation_index": 0,
                        "next_fixation_start": 25,
                        "next_fixation_stop": 35,
                        "gap_metric": "stop_to_start",
                        "gap_samples": 5,
                        "gap_ms": 5.0,
                        "sample_rate_hz": 1000.0,
                    },
                ]
            ).to_csv(
                analysis_dir / "within_session_interactive_m1_m2_face_fixation_gap_distribution.csv",
                index=False,
            )

            settings = FaceFixationGapDistributionPlotSettings(
                cfg_path=str(dataset_cfg_path),
                plotting_cfg_path=str(plotting_cfg_path),
            )
            m1_out, m1_m2_out = plot_face_fixation_gap_distribution_figures(settings)

            self.assertTrue(m1_out.exists())
            self.assertTrue(m1_m2_out.exists())
