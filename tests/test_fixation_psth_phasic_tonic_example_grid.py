"""Regression tests for phasic/tonic fixation PSTH example-grid plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.plotting.fixation_psth import (
    FixationPSTHUnitPlotSettings,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_psth_example_grid import (
    FixationPSTHExampleGridPlotSettings,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_psth_phasic_tonic_example_grid import (
    DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_REGIONS,
    DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_LABELS,
    DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_STYLES,
    normalize_example_response_style,
    parse_phasic_tonic_example_grid_unit_specs,
)
from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path

try:
    from dal_monte_2022_analysis.ephys.plotting.fixation_psth_phasic_tonic_example_grid import (
        plot_fixation_psth_phasic_tonic_example_grid,
    )

    _HAS_PLOT = True
except ModuleNotFoundError:
    _HAS_PLOT = False


def _write_dataset_cfg(path: Path, processed_root: Path, analysis_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "dataset_name: test_dataset",
                f"raw_data_root: {processed_root}",
                f"processed_data_root: {processed_root}",
                f"analysis_output_root: {analysis_root}",
                "processed_data_layout:",
                '  pattern: "date={date}/session={session}/{modality}"',
            ]
        ),
        encoding="utf-8",
    )


def _counts_for_style(style: str, condition: str, *, scale: float) -> np.ndarray:
    if style == "phasic":
        base = {
            "face_interactive": np.asarray([0.0, 2.0, 8.0, 2.0, 0.0], dtype=float),
            "face_non_interactive": np.asarray([0.0, 1.0, 4.0, 1.0, 0.0], dtype=float),
            "object": np.asarray([0.0, 0.5, 2.0, 0.5, 0.0], dtype=float),
        }
    else:
        base = {
            "face_interactive": np.asarray([2.0, 2.2, 2.4, 2.2, 2.0], dtype=float),
            "face_non_interactive": np.asarray([1.4, 1.6, 1.8, 1.6, 1.4], dtype=float),
            "object": np.asarray([0.8, 1.0, 1.2, 1.0, 0.8], dtype=float),
        }
    return base[condition] * float(scale)


def _write_trial_fixture(path: Path) -> None:
    bin_centers = np.asarray([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=float)
    rows = []
    specs = {
        "600": ("BLA", "phasic", 1.00),
        "591": ("BLA", "tonic", 1.10),
        "145": ("ACCg", "phasic", 0.85),
        "118": ("ACCg", "tonic", 0.95),
        "1796": ("dmPFC", "phasic", 1.20),
        "1516": ("dmPFC", "tonic", 1.05),
        "1091": ("OFC", "phasic", 1.15),
        "1011": ("OFC", "tonic", 0.90),
    }
    conditions = (
        ("face_interactive", "face", "interactive", True),
        ("face_non_interactive", "face", "non_interactive", False),
        ("object", "object", None, False),
    )
    for unit_uuid, (region, style, scale) in specs.items():
        for cond_key, fixation_category, interactive_state, is_interactive in conditions:
            counts = _counts_for_style(style, cond_key, scale=scale)
            for trial_idx in range(2):
                rows.append(
                    {
                        "date": "01012020",
                        "session": "s1",
                        "unit_uuid": unit_uuid,
                        "region": region,
                        "fixation_category": fixation_category,
                        "interactive_state": interactive_state,
                        "is_interactive": bool(is_interactive),
                        "psth_counts": counts + float(trial_idx),
                        "spike_train_counts": counts + float(trial_idx),
                    }
                )

    save_pickle_path(
        {
            "meta": {
                "bin_centers_s_rel": bin_centers,
            },
            "trials": pd.DataFrame(rows),
        },
        path,
    )


class TestFixationPSTHPhasicTonicExampleGridConfig(unittest.TestCase):
    """Regression checks for phasic/tonic example-grid config parsing."""

    def test_normalize_example_response_style(self) -> None:
        self.assertEqual(normalize_example_response_style("phasic"), "phasic")
        self.assertEqual(normalize_example_response_style("Classic Phasic"), "phasic")
        self.assertEqual(normalize_example_response_style("sustained"), "tonic")
        with self.assertRaises(ValueError):
            normalize_example_response_style("object")

    def test_parse_phasic_tonic_example_grid_unit_specs(self) -> None:
        cfg = {
            "phasic_tonic_example_grid_units": {
                "BLA": {"phasic": {"unit_uuid": "600"}, "tonic": {"unit_uuid": "591"}},
                "ACCg": {"phasic": {"unit_uuid": "145"}, "tonic": {"unit_uuid": "118"}},
                "dmPFC": {"phasic": {"unit_uuid": "1796"}, "tonic": {"unit_uuid": "1516"}},
                "OFC": {"phasic": {"unit_uuid": "1091"}, "tonic": {"unit_uuid": "1011"}},
            }
        }
        specs = parse_phasic_tonic_example_grid_unit_specs(cfg)
        self.assertEqual(len(specs), 8)
        self.assertEqual(
            {(spec.region, spec.preference, spec.unit_uuid) for spec in specs},
            {
                ("BLA", "phasic", "600"),
                ("BLA", "tonic", "591"),
                ("ACCg", "phasic", "145"),
                ("ACCg", "tonic", "118"),
                ("dmPFC", "phasic", "1796"),
                ("dmPFC", "tonic", "1516"),
                ("OFC", "phasic", "1091"),
                ("OFC", "tonic", "1011"),
            },
        )


@unittest.skipUnless(_HAS_PLOT, "matplotlib is required for phasic/tonic example-grid plotting tests")
class TestFixationPSTHPhasicTonicExampleGridPlot(unittest.TestCase):
    """Regression checks for phasic/tonic example-grid rendering."""

    def test_plot_phasic_tonic_example_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            trial_path = processed_root / "date=01012020" / "session=s1" / "psth" / "fixations_psth_10ms.pkl"
            trial_path.parent.mkdir(parents=True, exist_ok=True)
            _write_trial_fixture(trial_path)

            cfg = {
                "phasic_tonic_example_grid_units": {
                    "BLA": {"phasic": {"unit_uuid": "600"}, "tonic": {"unit_uuid": "591"}},
                    "ACCg": {"phasic": {"unit_uuid": "145"}, "tonic": {"unit_uuid": "118"}},
                    "dmPFC": {"phasic": {"unit_uuid": "1796"}, "tonic": {"unit_uuid": "1516"}},
                    "OFC": {"phasic": {"unit_uuid": "1091"}, "tonic": {"unit_uuid": "1011"}},
                }
            }
            unit_specs = parse_phasic_tonic_example_grid_unit_specs(cfg)

            unit_settings = FixationPSTHUnitPlotSettings(
                cfg_path=str(cfg_path),
                plotting_cfg_path="",
                trial_input_modality="psth",
                trial_input_filename="fixations_psth_10ms.pkl",
                raster_trial_input_modality=None,
                raster_trial_input_filename=None,
                use_precomputed_average_traces=False,
                allow_trial_trace_fallback=True,
                output_dpi=100,
                smooth_before_average=False,
                window_pre_s=0.5,
                window_post_s=0.5,
            )
            grid_settings = FixationPSTHExampleGridPlotSettings(
                unit_plot_settings=unit_settings,
                output_subdir="ephys/psth/phasic_tonic_example_grid_test",
                output_filename="phasic_tonic_grid_test",
                output_extension="png",
                output_dpi=100,
                figure_width_in=8.1,
                figure_height_in=3.2,
                column_regions=DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_REGIONS,
                row_preferences=DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_STYLES,
                row_labels=dict(DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_LABELS),
                display_window_s=(-1.0, 1.0),
                show_rate_window_rectangles=True,
            )

            out = plot_fixation_psth_phasic_tonic_example_grid(
                grid_settings,
                unit_specs=unit_specs,
            )

            self.assertIsNotNone(out)
            assert out is not None
            self.assertEqual(int(out["expected_cells"]), 8)
            self.assertEqual(int(out["resolved_cells"]), 8)
            self.assertEqual(list(out["row_preferences"]), ["phasic", "tonic"])
            self.assertEqual(list(out["column_regions"]), list(DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_REGIONS))
            self.assertEqual(tuple(out["display_window_s"]), (-1.0, 1.0))
            self.assertTrue(Path(out["output_path"]).exists())


if __name__ == "__main__":
    unittest.main()
