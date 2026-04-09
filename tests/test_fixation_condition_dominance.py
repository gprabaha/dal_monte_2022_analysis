"""Regression tests for fixation-condition dominance analysis and plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_condition_dominance import (
    FixationConditionDominanceSettings,
    run_fixation_condition_dominance_analysis,
)
from dal_monte_2022_analysis.runtime.io.processed_data import save_pickle_path

try:
    from dal_monte_2022_analysis.ephys.plotting.fixation_condition_dominance import (
        FixationConditionDominancePlotSettings,
        plot_fixation_condition_dominance_by_region,
    )

    _HAS_DOMINANCE_PLOTTING = True
except ModuleNotFoundError:
    _HAS_DOMINANCE_PLOTTING = False


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


def _trace(value: float) -> np.ndarray:
    return np.asarray([value, value, value, value], dtype=float)


def _write_average_fixture(path: Path) -> None:
    bin_centers = np.asarray([-0.75, -0.25, 0.25, 0.75], dtype=float)
    split_rows = []
    unsplit_rows = []
    specs = {
        "u1": ("ACC", 10.0, 2.0, 4.0),
        "u2": ("ACC", 1.0, 7.0, 3.0),
        "u3": ("OFC", 2.0, 3.0, 8.0),
        "u4": ("OFC", 5.0, 1.0, 5.0),
    }
    for unit_uuid, (region, face_int, face_nonint, obj) in specs.items():
        for state, value in (("interactive", face_int), ("non_interactive", face_nonint)):
            split_rows.append(
                {
                    "date": "01012020",
                    "unit_uuid": unit_uuid,
                    "region": region,
                    "spike_channel": f"ch_{unit_uuid}",
                    "recorded_agent": "m1",
                    "area": region.lower(),
                    "fixation_category": "face",
                    "interactive_state": state,
                    "is_interactive": state == "interactive",
                    "n_trials": 10,
                    "psth_mean": _trace(value),
                    "psth_sem": _trace(0.1),
                }
            )
        unsplit_rows.append(
            {
                "date": "01012020",
                "unit_uuid": unit_uuid,
                "region": region,
                "spike_channel": f"ch_{unit_uuid}",
                "recorded_agent": "m1",
                "area": region.lower(),
                "fixation_category": "object",
                "interactive_state": np.nan,
                "is_interactive": np.nan,
                "n_trials": 10,
                "psth_mean": _trace(obj),
                "psth_sem": _trace(0.1),
            }
        )

    meta = {
        "convert_to_firing_rate_before_average": True,
        "psth_value_kind": "firing_rate_hz",
        "split_meta": {
            "bin_centers_s_rel": bin_centers,
            "output_bin_size_s": 0.5,
            "convert_to_firing_rate_before_average": True,
            "psth_value_kind": "firing_rate_hz",
        },
        "unsplit_meta": {
            "bin_centers_s_rel": bin_centers,
            "output_bin_size_s": 0.5,
            "convert_to_firing_rate_before_average": True,
            "psth_value_kind": "firing_rate_hz",
        },
    }
    save_pickle_path(
        {
            "meta": meta,
            "averages_split_by_interactive_state": pd.DataFrame(split_rows),
            "averages_unsplit_by_interactive_state": pd.DataFrame(unsplit_rows),
        },
        path,
    )


class TestFixationConditionDominance(unittest.TestCase):
    """Regression checks for dominance summary construction."""

    def test_dominance_analysis_writes_all_raw_and_corrected_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            avg_path = (
                analysis_root
                / "ephys/psth/fixation_psth_averages"
                / "date=01012020"
                / "fixations_psth_10ms.pkl"
            )
            avg_path.parent.mkdir(parents=True, exist_ok=True)
            _write_average_fixture(avg_path)

            selectivity_root = analysis_root / "ephys/psth/fixation_psth_selectivity"
            selectivity_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "unit_key": "01012020|u1",
                        "is_selective_unit": True,
                        "is_selective_unit_raw": True,
                        "is_selective_unit_corrected": False,
                    },
                    {
                        "unit_key": "01012020|u2",
                        "is_selective_unit": False,
                        "is_selective_unit_raw": False,
                        "is_selective_unit_corrected": False,
                    },
                    {
                        "unit_key": "01012020|u3",
                        "is_selective_unit": True,
                        "is_selective_unit_raw": True,
                        "is_selective_unit_corrected": True,
                    },
                    {
                        "unit_key": "01012020|u4",
                        "is_selective_unit": True,
                        "is_selective_unit_raw": True,
                        "is_selective_unit_corrected": True,
                    },
                ]
            ).to_csv(selectivity_root / "unit_selectivity.csv", index=False)

            settings = FixationConditionDominanceSettings(
                cfg_path=str(cfg_path),
                average_input_subdir="ephys/psth/fixation_psth_averages",
                average_input_filename="fixations_psth_10ms.pkl",
                selectivity_input_subdir="ephys/psth/fixation_psth_selectivity",
                output_subdir="ephys/psth/fixation_condition_dominance",
                region_order=("ACC", "OFC"),
                tie_tolerance_hz=1e-12,
            )
            result = run_fixation_condition_dominance_analysis(settings)
            unit_df = result["unit_dominance"]
            summary_df = result["region_summary"]

            self.assertEqual(len(unit_df), 4)
            u4 = unit_df.loc[unit_df["unit_uuid"].astype(str) == "u4"].iloc[0]
            self.assertEqual(str(u4["dominance_status"]), "tie")

            def _count(subset: str, region: str, condition: str) -> int:
                row = summary_df.loc[
                    (summary_df["unit_subset"].astype(str) == subset)
                    & (summary_df["region"].astype(str) == region)
                    & (summary_df["dominant_condition"].astype(str) == condition)
                ].iloc[0]
                return int(row["n_units"])

            self.assertEqual(_count("all_units", "ACC", "face_interactive"), 1)
            self.assertEqual(_count("all_units", "ACC", "face_non_interactive"), 1)
            self.assertEqual(_count("all_units", "OFC", "object"), 1)
            self.assertEqual(_count("raw_selective_units", "ACC", "face_interactive"), 1)
            self.assertEqual(_count("corrected_selective_units", "ACC", "face_interactive"), 0)
            self.assertEqual(_count("corrected_selective_units", "OFC", "object"), 1)

            out_root = analysis_root / "ephys/psth/fixation_condition_dominance"
            self.assertTrue((out_root / "unit_condition_dominance.csv").exists())
            self.assertTrue((out_root / "region_condition_dominance_summary.csv").exists())
            self.assertTrue((out_root / "results.pkl").exists())


@unittest.skipUnless(_HAS_DOMINANCE_PLOTTING, "matplotlib is required for dominance plotting tests")
class TestFixationConditionDominancePlot(unittest.TestCase):
    """Regression checks for dominance bar plotting."""

    def test_plot_dominance_by_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            in_root = analysis_root / "ephys/psth/fixation_condition_dominance"
            in_root.mkdir(parents=True, exist_ok=True)
            rows = []
            for subset in ("all_units", "raw_selective_units", "corrected_selective_units"):
                for region in ("ACC", "OFC"):
                    for condition, count in (
                        ("face_interactive", 2),
                        ("face_non_interactive", 1),
                        ("object", 3),
                    ):
                        rows.append(
                            {
                                "unit_subset": subset,
                                "region": region,
                                "dominant_condition": condition,
                                "n_units": count,
                                "n_units_subset_total": 6,
                                "n_units_classified": 6,
                                "fraction_of_classified": count / 6.0,
                                "fraction_of_subset": count / 6.0,
                                "n_ties": 0,
                                "n_missing_condition": 0,
                            }
                        )
            pd.DataFrame(rows).to_csv(in_root / "region_condition_dominance_summary.csv", index=False)

            settings = FixationConditionDominancePlotSettings(
                cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_condition_dominance",
                output_subdir="ephys/psth/fixation_condition_dominance/plots",
                output_filename="dominance_test",
                output_extension="png",
                output_dpi=110,
                region_order=("ACC", "OFC"),
                figure_width_in=6.5,
                figure_height_in=4.2,
            )
            result = plot_fixation_condition_dominance_by_region(settings)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(Path(result["output_path"]).exists())
            self.assertEqual(len(result["regions"]), 2)
            self.assertEqual(len(result["unit_subsets"]), 3)


if __name__ == "__main__":
    unittest.main()
