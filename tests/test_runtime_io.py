"""Regression tests for runtime IO adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ModuleNotFoundError:
    plt = None
    _HAS_MPL = False

from dal_monte_2022_analysis.core.behav.analysis_filenames import (
    build_fix_cross_correlation_output_filename,
    normalize_fix_cross_correlation_time_scope,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    build_processed_data_path,
    build_processed_out_dir,
    index_agent_paths,
    index_shared_paths,
    list_processed_modalities,
    scan_processed_paths,
)
from dal_monte_2022_analysis.runtime.io.processed_data import (
    build_processed_output_path,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import (
    build_analysis_output_dir,
    scan_analysis_date_paths,
    scan_analysis_paths,
)
if _HAS_MPL:
    from dal_monte_2022_analysis.runtime.io.plot_output import (
        normalize_extension,
        save_figure,
    )


class TestProcessedDataIo(unittest.TestCase):
    """Regression checks for processed-data runtime IO helpers."""

    def test_build_processed_paths_and_list_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cfg = {
                "processed_data_root": root,
                "processed_data_layout": {"pattern": "date={date}/session={session}/{modality}"},
            }
            row = {"date": "01012020", "session": "s1"}

            out_dir = build_processed_out_dir(cfg, row, "fixations")
            path = build_processed_data_path(cfg, row, "fixations", "m1")
            variant = build_processed_output_path(
                cfg,
                row,
                "fixations",
                "m1",
                output_suffix="_cleaned",
            )

            self.assertEqual(out_dir, root / "date=01012020/session=s1/fixations")
            self.assertEqual(path, out_dir / "agent=m1.pkl")
            self.assertEqual(variant, root / "date=01012020/session=s1/fixations_cleaned/agent=m1.pkl")

            (root / "date=01012020/session=s1/fixations/agent=m1.pkl").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (root / "date=01012020/session=s1/fixations/agent=m1.pkl").write_bytes(b"")
            (root / "date=01012020/session=s1/fixations_cleaned/agent=m1.pkl").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (root / "date=01012020/session=s1/fixations_cleaned/agent=m1.pkl").write_bytes(b"")
            self.assertEqual(list_processed_modalities(cfg), {"fixations", "fixations_cleaned"})

    def test_scan_and_index_processed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "date=01012020/session=s1/fixation_density/agent=m1.pkl").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (root / "date=01012020/session=s1/fixation_density/agent=m1.pkl").write_bytes(b"")
            (root / "date=01012020/session=s1/fixation_density/agent=m2.pkl").write_bytes(b"")
            (root / "date=01022020/session=s2/fixation_density/shared.pkl").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (root / "date=01022020/session=s2/fixation_density/shared.pkl").write_bytes(b"")

            cfg = {"processed_data_root": root}
            rows = scan_processed_paths(cfg, "fixation_density")
            self.assertEqual(len(rows), 3)

            m1_paths, m2_paths = index_agent_paths(cfg, "fixation_density")
            self.assertIn(("01012020", "s1"), m1_paths)
            self.assertIn(("01012020", "s1"), m2_paths)
            self.assertEqual(Path(m1_paths[("01012020", "s1")]).name, "agent=m1.pkl")
            self.assertEqual(Path(m2_paths[("01012020", "s1")]).name, "agent=m2.pkl")

            shared_paths = index_shared_paths(cfg, "fixation_density")
            self.assertIn(("01022020", "s2"), shared_paths)
            self.assertEqual(Path(shared_paths[("01022020", "s2")]).name, "shared.pkl")

    def test_index_functions_raise_on_missing_modality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"processed_data_root": Path(tmp_dir)}
            with self.assertRaises(RuntimeError):
                index_agent_paths(cfg, "missing_modality")
            with self.assertRaises(RuntimeError):
                index_shared_paths(cfg, "missing_modality")


class TestAnalysisIndexIo(unittest.TestCase):
    """Regression checks for analysis-output indexing helpers."""

    def test_build_analysis_output_dir_uses_analysis_root_when_present(self) -> None:
        cfg = {
            "analysis_output_root": Path("/tmp/analysis"),
            "processed_data_root": Path("/tmp/processed"),
        }
        self.assertEqual(
            build_analysis_output_dir(cfg, "behav/cross_correlation"),
            Path("/tmp/analysis/behav/cross_correlation"),
        )

    def test_scan_analysis_paths_date_session_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (
                root
                / "ephys/psth/fixation_neural_cross_correlation/within_region/date=01012020/session=s1/fixations.pkl"
            ).parent.mkdir(parents=True, exist_ok=True)
            (
                root
                / "ephys/psth/fixation_neural_cross_correlation/within_region/date=01012020/session=s1/fixations.pkl"
            ).write_bytes(b"")
            (
                root
                / "ephys/psth/fixation_neural_cross_correlation/within_region/date=01022020/session=s2/fixations.pkl"
            ).parent.mkdir(parents=True, exist_ok=True)
            (
                root
                / "ephys/psth/fixation_neural_cross_correlation/within_region/date=01022020/session=s2/fixations.pkl"
            ).write_bytes(b"")

            cfg = {"analysis_output_root": root}
            rows = scan_analysis_paths(
                cfg,
                "ephys/psth/fixation_neural_cross_correlation/within_region",
                filename="fixations.pkl",
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["date"], "01012020")
            self.assertEqual(rows[0]["session"], "s1")

    def test_scan_analysis_date_paths_date_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (
                root
                / "ephys/psth/fixation_psth_averages/date=01012020/fixations.pkl"
            ).parent.mkdir(parents=True, exist_ok=True)
            (root / "ephys/psth/fixation_psth_averages/date=01012020/fixations.pkl").write_bytes(b"")
            (
                root
                / "ephys/psth/fixation_psth_averages/date=01022020/fixations.pkl"
            ).parent.mkdir(parents=True, exist_ok=True)
            (root / "ephys/psth/fixation_psth_averages/date=01022020/fixations.pkl").write_bytes(b"")

            cfg = {"analysis_output_root": root}
            rows = scan_analysis_date_paths(
                cfg,
                "ephys/psth/fixation_psth_averages",
                filename="fixations.pkl",
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["date"], "01012020")
            self.assertEqual(rows[1]["date"], "01022020")


class TestBehaviorAnalysisFilenames(unittest.TestCase):
    """Regression checks for behavior-analysis filename helpers."""

    def test_cross_correlation_scope_normalization(self) -> None:
        self.assertEqual(normalize_fix_cross_correlation_time_scope("interactive_only"), "interactive")
        self.assertEqual(normalize_fix_cross_correlation_time_scope("noninteractive"), "non_interactive")
        self.assertEqual(normalize_fix_cross_correlation_time_scope(None), "whole")

    def test_cross_correlation_filename_builder(self) -> None:
        self.assertEqual(
            build_fix_cross_correlation_output_filename(
                "Face",
                "within",
                time_scope="interactive",
            ),
            "within_session_face_fix_cross_correlation__phase=interactive.pkl",
        )

@unittest.skipUnless(_HAS_MPL, "matplotlib is required for runtime plot-output tests")
class TestPlotOutputIo(unittest.TestCase):
    """Regression checks for shared figure-output helpers."""

    def test_normalize_extension(self) -> None:
        self.assertEqual(normalize_extension(".PDF", fallback="png"), "pdf")
        self.assertEqual(normalize_extension("", fallback="png"), "png")

    def test_save_figure_creates_parent_and_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "plots" / "example"
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot([0, 1], [0, 1], color="black")

            save_figure(fig, out_path, ext="png", dpi=120)
            plt.close(fig)
            self.assertTrue((Path(tmp_dir) / "plots" / "example.png").exists())


if __name__ == "__main__":
    unittest.main()
