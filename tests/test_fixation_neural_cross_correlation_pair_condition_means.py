"""Regression tests for fixation neural xcorr pair-condition mean outputs and stats."""

from __future__ import annotations

import pickle
import tempfile
import unittest

from pathlib import Path

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_helpers import (
    WITHIN_ANALYSIS_KIND,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_pair_condition_means import (
    FixationNeuralCrossCorrelationPairConditionMeanSettings,
    run_within_region_fixation_neural_cross_correlation_pair_condition_means,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_neural_cross_correlation_pair_condition_means import (
    FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    _fit_decay_models_for_side,
    build_fixation_neural_cross_correlation_pair_condition_mean_plot_payload,
    build_pair_condition_mean_fit_summary,
)



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



def _write_pair_average_session_file(
    path: Path,
    *,
    analysis_kind: str,
    date: str,
    session: str,
    signal_input_column: str,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "meta": {
                    "analysis_kind": analysis_kind,
                    "date": date,
                    "session": session,
                    "signal_input_column": signal_input_column,
                    "signal_variant": "smoothed" if signal_input_column != "spike_train_counts" else "raw",
                    "lags": np.asarray([-1, 0, 1], dtype=np.int64),
                    "bin_size_ms": 1.0,
                },
                "pair_averages": pd.DataFrame(rows),
            },
            f,
        )



def _write_pair_condition_mean_file(
    path: Path,
    *,
    analysis_kind: str,
    date: str,
    signal_input_column: str,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "meta": {
                    "analysis_kind": analysis_kind,
                    "date": date,
                    "signal_input_column": signal_input_column,
                    "signal_variant": "smoothed" if signal_input_column != "spike_train_counts" else "raw",
                    "condition_order": ["face_interactive", "face_non_interactive", "object"],
                    "lags": np.asarray([-1, 0, 1], dtype=np.int64),
                    "bin_size_ms": 1.0,
                },
                "pair_condition_means": pd.DataFrame(rows),
            },
            f,
        )


class TestFixationNeuralCrossCorrelationPairConditionMeans(unittest.TestCase):
    def test_within_region_pair_condition_means_aggregate_sessions_by_fixation_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            within_root = analysis_root / "ephys/psth/fixation_neural_cross_correlation/within_region"
            raw_rows_s1 = [
                {
                    "date": "20990110",
                    "session": "s1",
                    "group_label": "BLA",
                    "condition": "face_interactive",
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "n_fixations": 2,
                    "cross_correlation": np.asarray([1.0, 2.0, 3.0], dtype=float),
                },
                {
                    "date": "20990110",
                    "session": "s1",
                    "group_label": "BLA",
                    "condition": "object",
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "n_fixations": 1,
                    "cross_correlation": np.asarray([0.0, 1.0, 0.0], dtype=float),
                },
            ]
            raw_rows_s2 = [
                {
                    "date": "20990110",
                    "session": "s2",
                    "group_label": "BLA",
                    "condition": "face_interactive",
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "n_fixations": 1,
                    "cross_correlation": np.asarray([3.0, 4.0, 5.0], dtype=float),
                },
            ]
            smooth_rows_s1 = [
                {
                    "date": "20990110",
                    "session": "s1",
                    "group_label": "BLA",
                    "condition": "face_interactive",
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "n_fixations": 2,
                    "cross_correlation": np.asarray([2.0, 3.0, 4.0], dtype=float),
                },
            ]
            smooth_rows_s2 = [
                {
                    "date": "20990110",
                    "session": "s2",
                    "group_label": "BLA",
                    "condition": "face_interactive",
                    "region_1": "BLA",
                    "region_2": "BLA",
                    "unit_uuid_1": "u1",
                    "unit_uuid_2": "u2",
                    "n_fixations": 2,
                    "cross_correlation": np.asarray([4.0, 5.0, 6.0], dtype=float),
                },
            ]

            _write_pair_average_session_file(
                within_root / "date=20990110" / "session=s1" / "pair_averages.pkl",
                analysis_kind=WITHIN_ANALYSIS_KIND,
                date="20990110",
                session="s1",
                signal_input_column="spike_train_counts",
                rows=raw_rows_s1,
            )
            _write_pair_average_session_file(
                within_root / "date=20990110" / "session=s2" / "pair_averages.pkl",
                analysis_kind=WITHIN_ANALYSIS_KIND,
                date="20990110",
                session="s2",
                signal_input_column="spike_train_counts",
                rows=raw_rows_s2,
            )
            _write_pair_average_session_file(
                within_root / "date=20990110" / "session=s1" / "pair_averages_smoothed.pkl",
                analysis_kind=WITHIN_ANALYSIS_KIND,
                date="20990110",
                session="s1",
                signal_input_column="smoothed_spike_train_counts",
                rows=smooth_rows_s1,
            )
            _write_pair_average_session_file(
                within_root / "date=20990110" / "session=s2" / "pair_averages_smoothed.pkl",
                analysis_kind=WITHIN_ANALYSIS_KIND,
                date="20990110",
                session="s2",
                signal_input_column="smoothed_spike_train_counts",
                rows=smooth_rows_s2,
            )

            settings = FixationNeuralCrossCorrelationPairConditionMeanSettings(
                cfg_path=str(cfg_path),
                signal_input_column="spike_train_counts",
                signal_input_columns=("spike_train_counts", "smoothed_spike_train_counts"),
                use_parallel=False,
                parallelize_across_dates=False,
            )
            summary = run_within_region_fixation_neural_cross_correlation_pair_condition_means(
                settings,
                dates=["20990110"],
                show_progress=False,
            )

            self.assertEqual(int(summary["n_date_signal_runs_total"]), 2)
            raw_out = (
                analysis_root
                / "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/within_region"
                / "date=20990110"
                / "pair_condition_means.pkl"
            )
            smooth_out = raw_out.with_name("pair_condition_means_smoothed.pkl")
            self.assertTrue(raw_out.exists())
            self.assertTrue(smooth_out.exists())

            with raw_out.open("rb") as f:
                raw_obj = pickle.load(f)
            with smooth_out.open("rb") as f:
                smooth_obj = pickle.load(f)

            raw_df = raw_obj["pair_condition_means"]
            smooth_df = smooth_obj["pair_condition_means"]
            self.assertEqual(len(raw_df), 2)
            self.assertEqual(len(smooth_df), 1)
            self.assertNotIn("session", raw_df.columns)

            raw_face = raw_df.loc[raw_df["condition"].astype(str) == "face_interactive"].iloc[0]
            self.assertEqual(int(raw_face["n_fixations"]), 3)
            self.assertTrue(
                np.allclose(
                    np.asarray(raw_face["cross_correlation"], dtype=float),
                    np.asarray([5.0 / 3.0, 8.0 / 3.0, 11.0 / 3.0], dtype=float),
                )
            )
            self.assertAlmostEqual(float(raw_face["mean_cross_correlation_across_lags"]), 8.0 / 3.0)
            self.assertEqual(str(raw_obj["meta"]["signal_variant"]), "raw")
            self.assertEqual(str(smooth_obj["meta"]["signal_variant"]), "smoothed")

            smooth_face = smooth_df.iloc[0]
            self.assertEqual(int(smooth_face["n_fixations"]), 4)
            self.assertTrue(
                np.allclose(
                    np.asarray(smooth_face["cross_correlation"], dtype=float),
                    np.asarray([3.0, 4.0, 5.0], dtype=float),
                )
            )

    def test_fit_decay_models_for_side_prefers_exponential_for_curved_trace(self) -> None:
        x_axis = np.arange(-20.0, 21.0, 1.0, dtype=float)
        values = 0.2 + (1.8 * np.exp(-0.12 * np.abs(x_axis)))
        settings = FixationNeuralCrossCorrelationPairConditionMeanPlotSettings(
            cfg_path="unused",
            plotting_cfg_path="",
        )

        negative_fit = _fit_decay_models_for_side(
            settings,
            x_axis=x_axis,
            values=values,
            side="negative",
        )
        positive_fit = _fit_decay_models_for_side(
            settings,
            x_axis=x_axis,
            values=values,
            side="positive",
        )

        self.assertIsNotNone(negative_fit["exponential"]["r_squared"])
        self.assertIsNotNone(positive_fit["exponential"]["r_squared"])
        self.assertGreaterEqual(
            float(negative_fit["exponential"]["r_squared"]),
            float(negative_fit["linear"]["r_squared"]),
        )
        self.assertGreaterEqual(
            float(positive_fit["exponential"]["r_squared"]),
            float(positive_fit["linear"]["r_squared"]),
        )
        self.assertIn(str(negative_fit["selection"]), {"exponential", "both"})
        self.assertIn(str(positive_fit["selection"]), {"exponential", "both"})

    def test_plot_payload_builds_separate_raw_and_smoothed_group_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed_root = root / "processed"
            analysis_root = root / "analysis"
            processed_root.mkdir(parents=True, exist_ok=True)
            analysis_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, processed_root=processed_root, analysis_root=analysis_root)

            pair_mean_root = analysis_root / "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/within_region" / "date=20990111"
            raw_rows = []
            smooth_rows = []
            for idx, fi, fni, obj in (
                (1, 4.0, 0.2, 1.1),
                (2, 5.0, 0.1, 1.0),
                (3, 6.0, 0.3, 1.2),
                (4, 7.0, 0.2, 1.3),
            ):
                for rows, signal_scale in ((raw_rows, 1.0), (smooth_rows, 0.5)):
                    rows.extend(
                        [
                            {
                                "date": "20990111",
                                "group_label": "BLA",
                                "condition": "face_interactive",
                                "region_1": "BLA",
                                "region_2": "BLA",
                                "unit_uuid_1": f"u{idx}",
                                "unit_uuid_2": f"v{idx}",
                                "n_fixations": 5,
                                "cross_correlation": np.asarray([fi, fi, fi], dtype=float) * signal_scale,
                            },
                            {
                                "date": "20990111",
                                "group_label": "BLA",
                                "condition": "face_non_interactive",
                                "region_1": "BLA",
                                "region_2": "BLA",
                                "unit_uuid_1": f"u{idx}",
                                "unit_uuid_2": f"v{idx}",
                                "n_fixations": 5,
                                "cross_correlation": np.asarray([fni, fni, fni], dtype=float) * signal_scale,
                            },
                            {
                                "date": "20990111",
                                "group_label": "BLA",
                                "condition": "object",
                                "region_1": "BLA",
                                "region_2": "BLA",
                                "unit_uuid_1": f"u{idx}",
                                "unit_uuid_2": f"v{idx}",
                                "n_fixations": 5,
                                "cross_correlation": np.asarray([obj, obj, obj], dtype=float) * signal_scale,
                            },
                        ]
                    )

            _write_pair_condition_mean_file(
                pair_mean_root / "pair_condition_means.pkl",
                analysis_kind=WITHIN_ANALYSIS_KIND,
                date="20990111",
                signal_input_column="spike_train_counts",
                rows=raw_rows,
            )
            _write_pair_condition_mean_file(
                pair_mean_root / "pair_condition_means_smoothed.pkl",
                analysis_kind=WITHIN_ANALYSIS_KIND,
                date="20990111",
                signal_input_column="smoothed_spike_train_counts",
                rows=smooth_rows,
            )

            settings = FixationNeuralCrossCorrelationPairConditionMeanPlotSettings(
                cfg_path=str(cfg_path),
                plotting_cfg_path="",
                signal_input_column="spike_train_counts",
                signal_input_columns=("spike_train_counts", "smoothed_spike_train_counts"),
                use_parallel=False,
                min_pairs_for_significance=2,
            )
            payload = build_fixation_neural_cross_correlation_pair_condition_mean_plot_payload(
                settings,
                dates=["20990111"],
                analysis_kinds=(WITHIN_ANALYSIS_KIND,),
            )

            self.assertEqual(set(payload["results"].keys()), {
                (WITHIN_ANALYSIS_KIND, "spike_train_counts"),
                (WITHIN_ANALYSIS_KIND, "smoothed_spike_train_counts"),
            })

            raw_result = payload["results"][(WITHIN_ANALYSIS_KIND, "spike_train_counts")]
            fit_df = build_pair_condition_mean_fit_summary(settings, result=raw_result)
            self.assertEqual(len(fit_df), 3)
            self.assertEqual(set(fit_df["condition"].astype(str)), {"face_interactive", "face_non_interactive", "object"})
            self.assertTrue((fit_df["n_pairs"].astype(int) == 4).all())
            smooth_result = payload["results"][(WITHIN_ANALYSIS_KIND, "smoothed_spike_train_counts")]
            self.assertEqual(str(raw_result["signal_variant"]), "raw")
            self.assertEqual(str(smooth_result["signal_variant"]), "smoothed")
            self.assertEqual(len(raw_result["group_plot_map"]["BLA"]["face_interactive"]), 4)
            self.assertEqual(len(smooth_result["group_plot_map"]["BLA"]["object"]), 4)

            raw_mean_lag = raw_result["mean_lag_comparisons"]
            raw_per_lag = raw_result["per_lag_comparisons"]
            self.assertEqual(len(raw_mean_lag), 3)
            self.assertTrue(raw_per_lag.empty)
            self.assertTrue((raw_mean_lag["n_pairs"].astype(int) == 4).all())
            self.assertTrue(raw_mean_lag["significant"].astype(bool).any())


if __name__ == "__main__":
    unittest.main()
