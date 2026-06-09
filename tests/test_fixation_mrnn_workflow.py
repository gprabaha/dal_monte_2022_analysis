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
    backproject_region_pcs,
    build_fixation_mrnn_targets_from_dataframe,
    build_model_spec,
    compute_region_flow_field,
    extract_fixation_latent_dynamics,
    extract_region_currents,
    extract_region_current_vectors,
    pc_reconstructed_firing_rate_accuracy,
    reconstruction_accuracy,
    replay_fixation_mrnn_run,
    replay_fixation_mrnn_run_with_ablations,
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
        backprojected = backproject_region_pcs(targets.pcs_by_region["ofc"], targets.pca_by_region["ofc"])
        self.assertEqual(backprojected.shape, targets.raw_by_region["ofc"].shape)


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
        readout = model.readout_weight_matrix()
        self.assertEqual(readout.shape, (8, model.total_num_units))
        hidden_slices = model.hidden_region_slices()
        output_slices = model.output_region_slices()
        for output_region in regions:
            for hidden_region in regions:
                block = readout[output_slices[output_region], hidden_slices[hidden_region]]
                if output_region == hidden_region:
                    self.assertTrue(torch.equal(block, model.output_heads[output_region].weight))
                else:
                    self.assertTrue(torch.count_nonzero(block).item() == 0)
        self.assertTrue(torch.equal(model.readout_bias_vector()[:2], model.output_heads["ofc"].bias))

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
                l2_weight_scale=1e-6,
                l2_rate_scale=1e-6,
                correlation_loss_scale=0.1,
                variance_loss_scale=0.1,
            )
            result = train_fixation_mrnn_scratch(settings, scratch_id="test_raw", overwrite=True)
            replay = replay_fixation_mrnn_run(result["run_dir"], device="cpu")
            current_df, current_vectors = extract_region_currents(replay)
            latent = extract_fixation_latent_dynamics(replay)
            recurrent_vectors = extract_region_current_vectors(replay)
            self.assertTrue((Path(result["run_dir"]) / "checkpoint_final.pth").exists())
            self.assertTrue((Path(result["run_dir"]) / "seed_plan.json").exists())
            self.assertIn("temporal_derivative_loss", result["history"].columns)
            self.assertIn("correlation_loss", result["history"].columns)
            self.assertIn("variance_loss", result["history"].columns)
            self.assertIn("l2_weight_loss", result["history"].columns)
            self.assertFalse(current_df.empty)
            self.assertIn("signed_projection", current_df.columns)
            self.assertTrue((current_df["relative_contribution"].abs() <= 1.0 + 1e-6).all())
            self.assertTrue(current_vectors)
            self.assertEqual(tuple(latent), replay["condition_order"])
            self.assertIn("hidden_state", latent["face_interactive"])
            self.assertIn("recurrent_drive", latent["face_interactive"])
            self.assertEqual(latent["face_interactive"]["hidden_state"].shape, replay["h_seq"][0].shape)
            self.assertEqual(latent["face_interactive"]["recurrent_drive"].shape, replay["h_seq"][0].shape)
            self.assertIn(("ofc", "bla"), recurrent_vectors)
            self.assertEqual(
                recurrent_vectors[("ofc", "bla")].shape,
                (
                    len(replay["condition_order"]),
                    len(replay["checkpoint"]["timeline_s"]),
                    settings.hidden_units,
                ),
            )
            self.assertFalse(reconstruction_accuracy(replay).empty)
            self.assertFalse(variance_comparison(replay).empty)
            flow = compute_region_flow_field(
                replay,
                region="ofc",
                condition="face_interactive",
                time_idx=1,
                grid_points=3,
            )
            self.assertEqual(flow["region"], "ofc")
            self.assertEqual(flow["u"].shape, (3, 3))
            self.assertEqual(flow["v"].shape, (3, 3))

    def test_pc_training_backprojected_fr_metrics_and_ablation_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            analysis_root = root / "analysis"
            avg_root = analysis_root / "ephys/psth/fixation_psth_averages"
            avg_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, analysis_root)
            _synthetic_combined_dataframe(n_units_per_region=3).to_pickle(avg_root / "combined.pkl")
            with (avg_root / "timeline.pkl").open("wb") as f:
                pickle.dump(np.asarray([-0.02, -0.01, 0.0, 0.01], dtype=float), f)

            settings = FixationMRNNRunSettings(
                dataset_cfg_path=str(cfg_path),
                dataframe_filename="combined.pkl",
                timeline_filename="timeline.pkl",
                target_mode="region_pcs",
                hidden_units=3,
                epochs=1,
                seed=778,
                device="cpu",
                spectral_radius=1.0,
                temporal_basis_count=0,
                fr_reconstruction_loss_scale=0.1,
                fr_temporal_derivative_loss_scale=0.1,
                fr_temporal_curvature_loss_scale=0.1,
            )
            result = train_fixation_mrnn_scratch(settings, scratch_id="test_pc", overwrite=True)
            replay = replay_fixation_mrnn_run(result["run_dir"], device="cpu")
            self.assertIn("fr_reconstruction_loss", result["history"].columns)
            self.assertIn("pc_reconstructed_raw_by_region", replay["checkpoint"])
            self.assertFalse(pc_reconstructed_firing_rate_accuracy(replay).empty)

            ablated = replay_fixation_mrnn_run_with_ablations(
                result["run_dir"],
                ablations=[("ofc", "bla")],
                device="cpu",
            )
            self.assertEqual(ablated["ablated_connections"], (("ofc", "bla"),))
            self.assertEqual(ablated["output"].shape, replay["output"].shape)


if __name__ == "__main__":
    unittest.main()
