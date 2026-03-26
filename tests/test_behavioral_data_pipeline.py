"""Regression tests for behavioral source indexing and pruning orchestration."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.behav.preprocessing.clean_dataset import clean_dataset
from dal_monte_2022_analysis.data.loaders.behavioral import index_behavioral_source_data
from dal_monte_2022_analysis.data.records.behavioral import (
    BehaviorRunContext,
    NeuralTimelineData,
    PositionData,
    PupilSizeData,
)
from dal_monte_2022_analysis.data.transforms.annotate import load_pair_context_table
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path
from dal_monte_2022_analysis.utils.io import save_pickle


class TestBehavioralSourceIndexing(unittest.TestCase):
    """Checks for shared raw behavioral source indexing helpers."""

    @staticmethod
    def _write_yaml(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")

    def test_index_behavioral_source_data_uses_configured_pair_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_cfg_path = root / "configs" / "dataset.yaml"
            raw_root = root / "raw"
            position_root = raw_root / "eyetracking" / "aligned_raw_samples" / "position"
            metadata_root = raw_root / "metadata"
            position_root.mkdir(parents=True, exist_ok=True)
            metadata_root.mkdir(parents=True, exist_ok=True)

            (position_root / "01012020_position_2.mat").write_bytes(b"")
            (position_root / "01012020_position_1.mat").write_bytes(b"")
            (position_root / "01022020_position_1.mat").write_bytes(b"")
            (position_root / "ignore_me.mat").write_bytes(b"")

            pair_df = pd.DataFrame(
                {
                    "day_id": ["1012020", "01022020"],
                    "monkey_a": ["Alpha", "Beta"],
                    "monkey_b": ["Bravo", "Gamma"],
                }
            )
            pair_df.to_pickle(metadata_root / "pairs.pkl")

            self._write_yaml(
                dataset_cfg_path,
                """
                raw_data_root: ../raw
                processed_data_root: ../processed
                analysis_output_root: ../analysis
                agents: ["m1", "m2"]
                processed_data_layout:
                  pattern: "date={date}/session={session}/{modality}"
                pair_context:
                  path: metadata/pairs.pkl
                  date_column: day_id
                  agent_name_columns:
                    m1: monkey_a
                    m2: monkey_b
                modalities:
                  gaze_position:
                    folder: eyetracking/aligned_raw_samples/position
                    file_pattern: '(?P<date>\d{8})_position_(?P<session>\d+)\.mat'
                """,
            )

            pair_context = load_pair_context_table(cfg_path=str(dataset_cfg_path))
            self.assertEqual(list(pair_context["date"]), ["01012020", "01022020"])
            self.assertEqual(list(pair_context["m1_name"]), ["Alpha", "Beta"])
            self.assertEqual(list(pair_context["m2_name"]), ["Bravo", "Gamma"])

            index_df = index_behavioral_source_data("gaze_position", cfg_path=str(dataset_cfg_path))
            self.assertEqual(len(index_df), 3)
            self.assertEqual(
                list(index_df[["date", "session"]].itertuples(index=False, name=None)),
                [("01012020", "1"), ("01012020", "2"), ("01022020", "1")],
            )
            self.assertEqual(list(index_df["monkey_name_m1"]), ["Alpha", "Alpha", "Beta"])
            self.assertEqual(list(index_df["monkey_name_m2"]), ["Bravo", "Bravo", "Gamma"])


class TestBehavioralCleaningPipeline(unittest.TestCase):
    """Checks that pruning operates from processed local data, not raw sources."""

    @staticmethod
    def _write_yaml(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")

    def test_clean_dataset_reads_processed_store_without_raw_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_cfg_path = root / "configs" / "dataset.yaml"
            processed_root = root / "processed"

            self._write_yaml(
                dataset_cfg_path,
                """
                raw_data_root: ../raw
                processed_data_root: ../processed
                analysis_output_root: ../analysis
                agents: ["m1"]
                processed_data_layout:
                  pattern: "date={date}/session={session}/{modality}"
                modalities:
                  neural_timeline:
                    folder: eyetracking/aligned_raw_samples/time
                    file_pattern: '(?P<date>\d{8})_position_(?P<session>\d+)\.mat'
                """,
            )

            context = BehaviorRunContext(
                date="01012020",
                session="1",
                agent="m1",
                monkey_name="Alpha",
            )
            shared_context = BehaviorRunContext(
                date="01012020",
                session="1",
                agent=None,
                monkey_name=None,
            )

            save_pickle(
                NeuralTimelineData(
                    context=shared_context,
                    t=np.asarray([0.0, np.nan, 0.2, 0.3], dtype=float),
                ),
                processed_root / "date=01012020/session=1/neural_timeline/shared.pkl",
            )
            save_pickle(
                PositionData(
                    context=context,
                    x=np.asarray([1.0, np.nan, 3.0, 4.0], dtype=float),
                    y=np.asarray([2.0, np.nan, 4.0, 5.0], dtype=float),
                ),
                processed_root / "date=01012020/session=1/gaze_position/agent=m1.pkl",
            )
            save_pickle(
                PupilSizeData(
                    context=context,
                    d=np.asarray([1.0, np.nan, 3.0, 4.0], dtype=float),
                ),
                processed_root / "date=01012020/session=1/pupil_size/agent=m1.pkl",
            )

            clean_dataset(str(dataset_cfg_path), output_suffix="_cleaned")

            cleaned_timeline = load_pickle_path(
                processed_root / "date=01012020/session=1/neural_timeline_cleaned/shared.pkl"
            )
            cleaned_position = load_pickle_path(
                processed_root / "date=01012020/session=1/gaze_position_cleaned/agent=m1.pkl"
            )
            cleaned_pupil = load_pickle_path(
                processed_root / "date=01012020/session=1/pupil_size_cleaned/agent=m1.pkl"
            )

            self.assertEqual(len(cleaned_timeline.t), 3)
            self.assertEqual(len(cleaned_position.x), 3)
            self.assertEqual(len(cleaned_pupil.d), 3)
