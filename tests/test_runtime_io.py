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

from dal_monte_2022_analysis.runtime.io.processed_data import (
    index_agent_paths,
    index_shared_paths,
    scan_processed_paths,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import (
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

    def test_scan_analysis_paths_date_session_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (
                root
                / "ephys/psth/fixation_neural_crosscorr/within_region/date=01012020/session=s1/fixations.pkl"
            ).parent.mkdir(parents=True, exist_ok=True)
            (
                root
                / "ephys/psth/fixation_neural_crosscorr/within_region/date=01012020/session=s1/fixations.pkl"
            ).write_bytes(b"")
            (
                root
                / "ephys/psth/fixation_neural_crosscorr/within_region/date=01022020/session=s2/fixations.pkl"
            ).parent.mkdir(parents=True, exist_ok=True)
            (
                root
                / "ephys/psth/fixation_neural_crosscorr/within_region/date=01022020/session=s2/fixations.pkl"
            ).write_bytes(b"")

            cfg = {"analysis_output_root": root}
            rows = scan_analysis_paths(
                cfg,
                "ephys/psth/fixation_neural_crosscorr/within_region",
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
