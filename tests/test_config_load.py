"""Regression tests for repo-root-relative config path normalization."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from dal_monte_2022_analysis.config.load import (
    get_repo_root,
    load_config,
    load_project_config,
    resolve_dataset_cfg_path,
    resolve_repo_path,
)


class TestConfigPathNormalization(unittest.TestCase):
    """Ensure config path keys are resolved relative to config file locations."""

    @staticmethod
    def _write_yaml(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")

    def test_dataset_roots_resolve_relative_to_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_cfg_path = root / "configs" / "dataset.yaml"
            self._write_yaml(
                dataset_cfg_path,
                """
                raw_data_root: raw
                processed_data_root: processed
                analysis_output_root: analysis
                """,
            )

            cfg = load_config(dataset_cfg_path, config_type="dataset")

            self.assertEqual(cfg["raw_data_root"], (root / "raw").resolve())
            self.assertEqual(cfg["processed_data_root"], (root / "processed").resolve())
            self.assertEqual(cfg["analysis_output_root"], (root / "analysis").resolve())

    def test_project_paths_resolve_relative_to_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            project_cfg_path = root / "configs" / "project.yaml"
            self._write_yaml(
                project_cfg_path,
                """
                dataset_cfg_path: configs/dataset.yaml
                ephys_data_cfg_path: configs/ephys_data.yaml
                plotting_cfg_path: configs/plotting.yaml
                raw_data_root: raw
                processed_data_root: processed
                analysis_output_root: analysis
                """,
            )

            cfg = load_project_config(project_cfg_path)

            self.assertEqual(cfg["dataset_cfg_path"], (root / "configs/dataset.yaml").resolve())
            self.assertEqual(cfg["ephys_data_cfg_path"], (root / "configs/ephys_data.yaml").resolve())
            self.assertEqual(cfg["plotting_cfg_path"], (root / "configs/plotting.yaml").resolve())
            self.assertEqual(cfg["raw_data_root"], (root / "raw").resolve())
            self.assertEqual(cfg["processed_data_root"], (root / "processed").resolve())
            self.assertEqual(cfg["analysis_output_root"], (root / "analysis").resolve())

    def test_resolve_dataset_cfg_path_uses_project_cfg_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            project_cfg_path = root / "configs" / "project.yaml"
            dataset_cfg_path = root / "configs" / "dataset.yaml"
            self._write_yaml(
                project_cfg_path,
                """
                dataset_cfg_path: configs/dataset.yaml
                ephys_data_cfg_path: configs/ephys_data.yaml
                """,
            )
            self._write_yaml(dataset_cfg_path, "processed_data_root: processed\nraw_data_root: raw\n")

            self.assertEqual(resolve_dataset_cfg_path(project_cfg_path), dataset_cfg_path.resolve())

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

    def test_resolve_repo_path_prefers_repo_root_candidate(self) -> None:
        resolved = resolve_repo_path("configs/project.yaml")
        self.assertEqual(resolved, (get_repo_root() / "configs/project.yaml").resolve())


if __name__ == "__main__":
    unittest.main()
