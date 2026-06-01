"""Tests for the minimal fixation Elman mRNN workflow."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dal_monte_2022_analysis.ephys.modeling import (
    FixationMRNNRunSettings,
    FixationMRNNModel,
    build_fixation_mrnn_targets_from_dataframe,
    build_model_spec,
    compute_region_flow_field,
    extract_region_currents,
    reconstruction_accuracy,
    replay_fixation_mrnn_run,
    train_fixation_mrnn_scratch,
    variance_comparison,
)


def _synthetic_combined_dataframe(
    *,
    regions=("ofc", "bla", "dmpfc", "accg"),
    n_units_per_region=2,
    n_time=4,
) -> pd.DataFrame:
    rows = []
    condition_rows = [
        ("split", "face", "interactive", 1.0),
        ("split", "face", "non_interactive", 2.0),
        ("unsplit", "object", None, 3.0),
    ]
    for region_idx, region in enumerate(regions):
        for unit_idx in range(n_units_per_region):
            for partition, category, interactive_state, offset in condition_rows:
                base = 10.0 * region_idx + unit_idx + offset
                rows.append(
                    {
                        "date": "20990101",
                        "unit_uuid": f"{region}_unit_{unit_idx}",
                        "region": region,
                        "spike_channel": unit_idx,
                        "recorded_agent": "m1",
                        "average_partition": partition,
                        "fixation_category": category,
                        "interactive_state": interactive_state,
                        "n_trials": 5 + unit_idx,
                        "psth_mean": np.asarray([base + t for t in range(n_time)], dtype=float),
                        "psth_sem": np.ones(n_time, dtype=float),
                    }
                )
    return pd.DataFrame(rows)


def _write_dataset_cfg(path: Path, analysis_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"raw_data_root: {analysis_root}",
                f"processed_data_root: {analysis_root}",
                f"analysis_output_root: {analysis_root}",
            ]
        ),
        encoding="utf-8",
    )


class TestFixationMRNNTargets(unittest.TestCase):
    def test_target_builder_uses_global_normalization_and_region_dims(self) -> None:
        df = _synthetic_combined_dataframe()
        targets = build_fixation_mrnn_targets_from_dataframe(
            df,
            timeline_s=np.asarray([-0.02, -0.01, 0.0, 0.01], dtype=float),
            normalize_targets=True,
            pca_variance_threshold=0.95,
        )
        self.assertEqual(targets.condition_order, ("face_interactive", "face_non_interactive", "object"))
        self.assertEqual(targets.input_tensor.shape, (3, 4, 23))
        self.assertEqual(targets.raw_by_region["ofc"].shape, (3, 4, 2))
        raw_values = np.concatenate(df["psth_mean"].to_numpy())
        expected_scale = np.percentile(raw_values, 95) - np.percentile(raw_values, 5) + 5.0
        self.assertAlmostEqual(float(targets.normalization_scale), float(expected_scale))
        self.assertEqual(targets.output_dims_for_mode("raw_fr")["bla"], 2)
        pc_dims = targets.output_dims_for_mode("region_pcs")
        self.assertEqual(len(set(pc_dims.values())), 1)
        self.assertGreaterEqual(pc_dims["ofc"], 1)


class TestFixationMRNNTorchSmoke(unittest.TestCase):
    def test_model_forward(self) -> None:
        regions = ("ofc", "bla", "dmpfc", "accg")
        spec = build_model_spec(
            region_order=regions,
            output_dims_by_region={region: 2 for region in regions},
            hidden_units=3,
            activation="softplus",
            rec_constrained=False,
            inp_constrained=False,
            spectral_radius=1.0,
            device="cpu",
        )
        model = FixationMRNNModel(spec)
        out = model(
            torch.zeros((3, 4, 3), dtype=torch.float32),
            torch.zeros((3, model.total_num_units), dtype=torch.float32),
            noise=False,
        )
        self.assertEqual(tuple(out["output_by_region"]), regions)
        self.assertEqual(out["output"].shape, (3, 4, 8))
        self.assertEqual(out["h_seq"].shape[-1], model.total_num_units)

    def test_one_iteration_training_replay_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            analysis_root = root / "analysis"
            avg_root = analysis_root / "ephys/psth/fixation_psth_averages"
            avg_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, analysis_root)
            _synthetic_combined_dataframe().to_pickle(avg_root / "combined.pkl")
            with (avg_root / "timeline.pkl").open("wb") as f:
                pickle.dump(np.asarray([-0.02, -0.01, 0.0, 0.01], dtype=float), f)

            settings = FixationMRNNRunSettings(
                dataset_cfg_path=str(cfg_path),
                dataframe_filename="combined.pkl",
                timeline_filename="timeline.pkl",
                target_mode="raw_fr",
                hidden_units=3,
                epochs=1,
                seed=777,
                device="cpu",
            spectral_radius=1.0,
            temporal_basis_count=0,
        )
            result = train_fixation_mrnn_scratch(settings, scratch_id="test_raw", overwrite=True)
            replay = replay_fixation_mrnn_run(result["run_dir"], device="cpu")
            current_df, current_vectors = extract_region_currents(replay)
            self.assertTrue((Path(result["run_dir"]) / "checkpoint_final.pth").exists())
            self.assertTrue((Path(result["run_dir"]) / "seed_plan.json").exists())
            self.assertFalse(current_df.empty)
            self.assertIn("signed_projection", current_df.columns)
            self.assertTrue((current_df["relative_contribution"].abs() <= 1.0 + 1e-6).all())
            self.assertTrue(current_vectors)
            self.assertFalse(reconstruction_accuracy(replay).empty)
            self.assertFalse(variance_comparison(replay).empty)
            flow = compute_region_flow_field(
                replay,
                region="ofc",
                condition="face_interactive",
                time_idx=1,
                grid_points=3,
            )
            self.assertEqual(flow["u"].shape, (3, 3))
            self.assertEqual(flow["v"].shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
