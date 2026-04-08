"""Tests for combining date-level fixation PSTH averages into one sliced dataframe."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dal_monte_2022_analysis.ephys.analysis.fixation_psth_average_combiner import (
        FixationPSTHAverageCombinerSettings,
        combine_fixation_psth_average_dataframes,
    )
    from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path

    _HAS_FIX_PSTH_AVERAGE_COMBINER = True
except ModuleNotFoundError:
    _HAS_FIX_PSTH_AVERAGE_COMBINER = False


def _write_dataset_cfg(path: Path, analysis_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "dataset_name: test_dataset",
                f"processed_data_root: {analysis_root}",
                f"analysis_output_root: {analysis_root}",
                "processed_data_layout:",
                '  pattern: "date={date}/session={session}/{modality}"',
            ]
        ),
        encoding="utf-8",
    )


def _build_average_payload(date: str, n_bins: int = 1000) -> dict:
    centers = np.arange(n_bins, dtype=float) * 0.01 - 4.995
    split_df = pd.DataFrame(
        [
            {
                "date": date,
                "unit_uuid": f"{date}_split",
                "region": "bla",
                "area": "bla",
                "spike_channel": "SPK01",
                "recorded_agent": "m1",
                "recorded_monkey": None,
                "fixation_category": "face",
                "interactive_state": "interactive",
                "is_interactive": True,
                "n_trials": 4,
                "psth_mean": np.arange(n_bins, dtype=float),
                "psth_sem": np.arange(n_bins, dtype=float) + 1000.0,
                "fixation_location_labels": ("face", "eyes_nf"),
                "source_fixation_agents": ("m1",),
                "source_fixation_monkeys": ("Cronenberg", "Lynch"),
                "source_sessions": ("1", "2"),
                "source_interactive_states": ("interactive",),
            }
        ]
    )
    unsplit_df = pd.DataFrame(
        [
            {
                "date": date,
                "unit_uuid": f"{date}_unsplit",
                "region": "ofc",
                "area": "ofc",
                "spike_channel": "SPK02",
                "recorded_agent": "m1",
                "recorded_monkey": None,
                "fixation_category": "object",
                "interactive_state": None,
                "is_interactive": None,
                "n_trials": 3,
                "psth_mean": np.arange(n_bins, dtype=float) + 2000.0,
                "psth_sem": np.arange(n_bins, dtype=float) + 3000.0,
                "fixation_location_labels": ("left_nonsocial_object",),
                "source_fixation_agents": ("m1",),
                "source_fixation_monkeys": ("Lynch",),
                "source_sessions": ("1",),
                "source_interactive_states": ("interactive", "non_interactive"),
            }
        ]
    )
    return {
        "meta": {
            "date": date,
            "split_meta": {
                "bin_centers_s_rel": centers,
                "output_bin_size_ms": 10.0,
            },
            "unsplit_meta": {
                "bin_centers_s_rel": centers,
                "output_bin_size_ms": 10.0,
            },
        },
        "averages_split_by_interactive_state": split_df,
        "averages_unsplit_by_interactive_state": unsplit_df,
    }


@unittest.skipUnless(
    _HAS_FIX_PSTH_AVERAGE_COMBINER,
    "fixation PSTH average combiner module is required for this test",
)
class TestFixationPSTHAverageCombiner(unittest.TestCase):
    """Checks combined PSTH dataframe export and timeline slicing."""

    def test_combiner_slices_window_and_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            analysis_root = root / "analysis"
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, analysis_root=analysis_root)

            input_root = analysis_root / "ephys" / "psth" / "fixation_psth_averages"
            for date in ("01022019", "01032019"):
                save_pickle_path(
                    _build_average_payload(date),
                    input_root / f"date={date}" / "fixations_psth_10ms.pkl",
                )

            settings = FixationPSTHAverageCombinerSettings(
                cfg_path=str(cfg_path),
                window_start_s=-0.5,
                window_stop_s=0.5,
            )
            result = combine_fixation_psth_average_dataframes(settings)

            combined_df = result["dataframe"]
            timeline = np.asarray(result["timeline_s_rel"], dtype=float)
            self.assertEqual(len(combined_df), 4)
            self.assertEqual(set(combined_df["average_partition"]), {"split", "unsplit"})
            self.assertEqual(timeline.size, 100)
            self.assertNotIn("area", combined_df.columns)
            self.assertNotIn("fixation_location_labels", combined_df.columns)
            self.assertNotIn("source_fixation_agents", combined_df.columns)
            self.assertNotIn("source_sessions", combined_df.columns)
            self.assertNotIn("source_interactive_states", combined_df.columns)
            self.assertNotIn("source_average_path", combined_df.columns)
            self.assertTrue(np.allclose(timeline[:3], np.asarray([-0.495, -0.485, -0.475])))
            self.assertTrue(np.allclose(timeline[-3:], np.asarray([0.475, 0.485, 0.495])))

            split_row = combined_df.loc[
                (combined_df["date"] == "01022019")
                & (combined_df["average_partition"] == "split")
            ].iloc[0]
            unsplit_row = combined_df.loc[
                (combined_df["date"] == "01022019")
                & (combined_df["average_partition"] == "unsplit")
            ].iloc[0]
            self.assertTrue(
                np.allclose(
                    np.asarray(split_row["psth_mean"], dtype=float),
                    np.arange(450, 550, dtype=float),
                )
            )
            self.assertTrue(
                np.allclose(
                    np.asarray(split_row["psth_sem"], dtype=float),
                    np.arange(1450, 1550, dtype=float),
                )
            )
            self.assertTrue(
                np.allclose(
                    np.asarray(unsplit_row["psth_mean"], dtype=float),
                    np.arange(2450, 2550, dtype=float),
                )
            )
            self.assertTrue(Path(str(result["dataframe_path"])).exists())
            self.assertTrue(Path(str(result["timeline_path"])).exists())
