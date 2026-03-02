"""Regression tests for config path normalization behavior."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from dal_monte_2022_analysis.config.load import (
    load_config,
    load_project_config,
)


class TestConfigPathNormalization(unittest.TestCase):
    """Ensure config path keys are resolved relative to config file locations."""

    @staticmethod
    def _write_yaml(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")

    def test_dataset_roots_resolve_relative_to_dataset_cfg_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_cfg_path = root / "configs" / "dataset.yaml"
            self._write_yaml(
                dataset_cfg_path,
                """
                raw_data_root: ./raw
                processed_data_root: ../processed
                analysis_output_root: ../analysis
                """,
            )

            cfg = load_config(dataset_cfg_path, config_type="dataset")

            self.assertEqual(cfg["raw_data_root"], (dataset_cfg_path.parent / "raw").resolve())
            self.assertEqual(cfg["processed_data_root"], (dataset_cfg_path.parent / "../processed").resolve())
            self.assertEqual(cfg["analysis_output_root"], (dataset_cfg_path.parent / "../analysis").resolve())

    def test_project_roots_and_referenced_cfgs_resolve_relative_to_project_cfg_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            project_cfg_path = root / "configs" / "project.yaml"
            self._write_yaml(
                project_cfg_path,
                """
                dataset_cfg_path: dataset.yaml
                ephys_data_cfg_path: ephys_data.yaml
                plotting_cfg_path: plotting.yaml
                raw_data_root: ../raw
                processed_data_root: ../processed
                analysis_output_root: ../analysis
                """,
            )

            cfg = load_project_config(project_cfg_path)

            self.assertEqual(cfg["dataset_cfg_path"], (project_cfg_path.parent / "dataset.yaml").resolve())
            self.assertEqual(cfg["ephys_data_cfg_path"], (project_cfg_path.parent / "ephys_data.yaml").resolve())
            self.assertEqual(cfg["plotting_cfg_path"], (project_cfg_path.parent / "plotting.yaml").resolve())
            self.assertEqual(cfg["raw_data_root"], (project_cfg_path.parent / "../raw").resolve())
            self.assertEqual(cfg["processed_data_root"], (project_cfg_path.parent / "../processed").resolve())
            self.assertEqual(cfg["analysis_output_root"], (project_cfg_path.parent / "../analysis").resolve())

    def test_absolute_dataset_paths_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_cfg_path = root / "configs" / "dataset.yaml"
            raw_root = (root / "raw_abs").resolve()
            processed_root = (root / "processed_abs").resolve()
            self._write_yaml(
                dataset_cfg_path,
                f"""
                raw_data_root: {raw_root}
                processed_data_root: {processed_root}
                """,
            )

            cfg = load_config(dataset_cfg_path, config_type="dataset")

            self.assertEqual(cfg["raw_data_root"], raw_root)
            self.assertEqual(cfg["processed_data_root"], processed_root)


if __name__ == "__main__":
    unittest.main()
