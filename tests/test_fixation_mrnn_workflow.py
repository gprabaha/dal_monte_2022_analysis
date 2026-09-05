"""Tests for the minimal fixation Elman mRNN workflow."""

from __future__ import annotations

import json
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
    compute_pairwise_regional_pc_cca,
    compute_region_flow_field,
    extract_fixation_latent_dynamics,
    extract_region_currents,
    extract_region_current_vectors,
    pc_reconstructed_firing_rate_accuracy,
    reconstruction_accuracy,
    replay_fixation_mrnn_run,
    replay_fixation_mrnn_run_with_ablations,
    condition_loss_weights,
    normalize_loss_weighting,
    normalize_lr_schedule,
    pc_loss_weights,
    resolve_checkpoint_path,
    train_fixation_mrnn_scratch,
    train_one_initialization,
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

    def test_target_builder_can_force_shared_pc_count(self) -> None:
        df = _synthetic_combined_dataframe(n_units_per_region=2)
        targets = build_fixation_mrnn_targets_from_dataframe(
            df,
            timeline_s=np.asarray([-0.02, -0.01, 0.0, 0.01], dtype=float),
            normalize_targets=True,
            pca_n_components=4,
            temporal_basis_count=0,
        )
        self.assertEqual(targets.pcs_by_region["ofc"].shape, (3, 4, 4))
        self.assertEqual(targets.pca_by_region["ofc"].components.shape, (4, 2))
        self.assertEqual(targets.output_dims_for_mode("region_pcs")["bla"], 4)

    def test_pairwise_cca_reports_region_pairs(self) -> None:
        df = _synthetic_combined_dataframe(n_units_per_region=3)
        targets = build_fixation_mrnn_targets_from_dataframe(
            df,
            timeline_s=np.asarray([-0.02, -0.01, 0.0, 0.01], dtype=float),
            normalize_targets=True,
            pca_n_components=3,
            temporal_basis_count=0,
        )
        cca_df, payloads = compute_pairwise_regional_pc_cca(
            targets.pcs_by_region,
            region_order=("ofc", "bla", "dmpfc", "accg"),
            max_components=3,
        )
        self.assertEqual(len(payloads), 6)
        self.assertEqual(set(cca_df["cca_dimension"].astype(int)), {1, 2, 3})
        self.assertTrue((cca_df["canonical_correlation"] <= 1.0 + 1e-5).all())


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

    def test_recurrent_connectivity_masks(self) -> None:
        regions = ("ofc", "bla", "dmpfc", "accg")
        within = FixationMRNNModel(
            build_model_spec(
                region_order=regions,
                output_dims_by_region={region: 2 for region in regions},
                hidden_units=3,
                recurrent_connectivity="within_region",
                spectral_radius=1.0,
                device="cpu",
            )
        )
        cross = FixationMRNNModel(
            build_model_spec(
                region_order=regions,
                output_dims_by_region={region: 2 for region in regions},
                hidden_units=3,
                recurrent_connectivity="cross_region_with_self_diagonal",
                spectral_radius=1.0,
                device="cpu",
            )
        )
        for model, mode in [(within, "within"), (cross, "cross")]:
            slices = model.hidden_region_slices()
            mask = model.mrnn.W_rec_mask.detach().cpu()
            for target_region in regions:
                for source_region in regions:
                    block = mask[slices[target_region], slices[source_region]]
                    if mode == "within" and target_region == source_region:
                        self.assertTrue(torch.equal(block, torch.ones_like(block)))
                    elif mode == "within":
                        self.assertEqual(torch.count_nonzero(block).item(), 0)
                    elif target_region != source_region:
                        self.assertTrue(torch.equal(block, torch.ones_like(block)))
                    else:
                        self.assertTrue(torch.equal(block, torch.eye(block.shape[0])))

    def test_inter_region_connections_use_low_rank_factors(self) -> None:
        regions = ("ofc", "bla")
        model = FixationMRNNModel(
            build_model_spec(
                region_order=regions,
                output_dims_by_region={region: 2 for region in regions},
                hidden_units=3,
                recurrent_connectivity="cross_region_with_self_diagonal",
                recurrent_bottleneck_dim=4,
                spectral_radius=1.0,
                device="cpu",
            )
        )
        self.assertGreater(len(model.inter_region_recurrent_parameters()), 0)
        for left, right in model.inter_region_recurrent_parameters():
            self.assertEqual(left.shape[1], 4)
            self.assertEqual(right.shape[0], 4)
        w_rec = model.recurrent_weight_matrix()
        slices = model.hidden_region_slices()
        self.assertGreater(torch.count_nonzero(w_rec[slices["ofc"], slices["bla"]]).item(), 0)
        self.assertEqual(len(model.within_region_recurrent_parameters()), len(regions))

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
            # An ablation has to actually reach the forward pass. Writing into
            # ``mrnn.W_rec`` does not: the wrapper reassembles its recurrent matrix from
            # the block parameters on every call and copies the result back over W_rec, so
            # an ablation applied there is silently discarded and every pathway measures as
            # free. Check the block is zero in the matrix the forward pass actually uses,
            # and that the replayed dynamics moved.
            intact_weight = replay["model"].recurrent_weight_matrix().detach()
            ablated_weight = ablated["model"].recurrent_weight_matrix().detach()
            slices = ablated["model"].hidden_region_slices()
            block = ablated_weight[slices["bla"], slices["ofc"]]
            self.assertEqual(float(block.abs().max()), 0.0)
            self.assertGreater(float((intact_weight - ablated_weight).abs().sum()), 0.0)
            self.assertFalse(torch.allclose(ablated["h_seq"], replay["h_seq"]))

    def test_balanced_condition_weighting_equalises_the_objective(self) -> None:
        """An absolute MSE gives each condition influence in proportion to its energy,
        which is why the low-energy condition is fitted worst."""
        target = torch.randn(3, 100, 8)
        target[0] *= 0.5  # a quarter of the energy of the other two
        uniform = condition_loss_weights(target, mode="uniform", device="cpu")
        balanced = condition_loss_weights(target, mode="balanced", device="cpu")
        self.assertTrue(torch.allclose(uniform, torch.ones_like(uniform)))

        centred = target - target.mean(dim=1, keepdim=True)
        energy = torch.mean(centred**2, dim=(1, 2))
        weighted = energy * balanced.flatten()
        # Every condition ends up contributing the same weighted energy.
        self.assertLess(float(weighted.max() / weighted.min()) - 1.0, 1e-4)

    def test_pc_whitening_lifts_the_low_variance_components(self) -> None:
        scales = torch.tensor([4.0, 2.0, 1.0, 0.5, 0.25])
        target = torch.randn(3, 200, 5) * scales
        uniform = pc_loss_weights(target, mode="uniform", floor=0.05, device="cpu").flatten()
        whiten = pc_loss_weights(target, mode="whiten", floor=0.05, device="cpu").flatten()
        sqrt_whiten = pc_loss_weights(target, mode="sqrt_whiten", floor=0.05, device="cpu").flatten()
        self.assertTrue(torch.allclose(uniform, torch.ones_like(uniform)))
        # Weight rises monotonically as component variance falls.
        self.assertTrue(bool((torch.diff(whiten) > 0).all()))
        # sqrt_whiten sits between the two.
        self.assertTrue(bool((sqrt_whiten[-1] < whiten[-1]).item()))
        self.assertTrue(bool((sqrt_whiten[-1] > uniform[-1]).item()))

    def test_pc_weight_floor_bounds_the_amplification(self) -> None:
        """Without a floor a near-empty component is amplified without bound and the
        objective is dominated by numerical dust."""
        target = torch.randn(3, 200, 4)
        target[:, :, 3] *= 1e-8
        weights = pc_loss_weights(target, mode="whiten", floor=0.05, device="cpu").flatten()
        self.assertLess(float(weights[3] / weights[0]), 25.0)

    def test_pc_weighting_does_not_reach_the_firing_rate_terms(self) -> None:
        """Firing-rate targets live in unit space, whose feature axis has a different
        length and no correspondence to the components. Carrying PC weights into those
        terms is a shape error waiting to happen, and this is the regression for it."""
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
                target_mode="region_pcs",
                hidden_units=3,
                epochs=1,
                seed=11,
                device="cpu",
                spectral_radius=1.0,
                temporal_basis_count=0,
                # Both the PC weighting and the firing-rate terms active at once.
                pc_loss_weighting="whiten",
                condition_loss_weighting="balanced",
                fr_reconstruction_loss_scale=1.0,
                fr_temporal_derivative_loss_scale=1.0,
            )
            result = train_one_initialization(settings, run_dir=root / "run", seed=11, overwrite=True)
            history = result["history"]
            self.assertTrue(np.isfinite(history["loss"].to_numpy(dtype=float)).all())
            self.assertGreater(float(history["fr_reconstruction_loss"].iloc[0]), 0.0)

    def test_unknown_weighting_names_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_loss_weighting("inverse_cubed", allowed=("uniform", "balanced"))
        self.assertEqual(normalize_loss_weighting(None, allowed=("uniform",)), "uniform")

    def test_blocked_pairs_remove_both_the_block_and_its_parameters(self) -> None:
        """The named connectivity modes cover global structures; blocked pairs cover
        everything else -- isolating a region, removing one directed pathway -- without a
        new mode for each. A blocked pair must get no parameters at all, so the model is
        genuinely smaller rather than merely masked."""
        regions = ("a", "b", "c")

        def build(blocked):
            torch.manual_seed(0)
            spec = build_model_spec(
                region_order=regions,
                output_dims_by_region={r: 3 for r in regions},
                hidden_units=6,
                device="cpu",
                input_dim=2,
                activation="tanh",
                spectral_radius=1.0,
                rec_constrained=False,
                inp_constrained=False,
                recurrent_connectivity="full",
                recurrent_bottleneck_dim=None,
                recurrent_blocked_pairs=blocked,
                batch_first=True,
                inp_noise=0.0,
                act_noise=0.0,
            )
            return FixationMRNNModel(spec)

        intact = build(())
        lesioned = build((("a", "b"),))

        weight = lesioned.recurrent_weight_matrix().detach().numpy()
        # Row block b, column block a is the a -> b pathway.
        self.assertLess(float(np.abs(weight[6:12, 0:6]).sum()), 1e-9)
        # Its neighbour is untouched.
        self.assertGreater(float(np.abs(weight[12:18, 0:6]).sum()), 1e-9)

        def trainable(model):
            return sum(p.numel() for n, p in model.named_parameters() if n.startswith("_inter"))

        self.assertLess(trainable(lesioned), trainable(intact))

    def test_blocked_pairs_survive_the_training_and_replay_round_trip(self) -> None:
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
                hidden_units=4,
                epochs=2,
                seed=3,
                device="cpu",
                spectral_radius=1.0,
                temporal_basis_count=0,
                recurrent_bottleneck_dim=None,
                recurrent_blocked_pairs=(("ofc", "bla"),),
            )
            run_dir = root / "run"
            train_one_initialization(settings, run_dir=run_dir, seed=3, overwrite=True)
            replay = replay_fixation_mrnn_run(run_dir, device="cpu")
            model = replay["model"]
            slices = model.hidden_region_slices()
            weight = model.recurrent_weight_matrix().detach().numpy()
            block = weight[slices["bla"], slices["ofc"]]
            self.assertLess(float(np.abs(block).sum()), 1e-9)

    def test_the_copied_recurrent_matrix_receives_no_gradient(self) -> None:
        """``mrnn.W_rec`` is a copy the forward pass overwrites from the block parameters.

        It is registered as a parameter, so any count taken from ``model.parameters()``
        includes it and overstates the model's real size roughly six-fold — which is how
        the \"as many parameters as data points\" reading of these fits arose. Pinning
        that it never receives a gradient is what licenses excluding it.
        """
        torch.manual_seed(0)
        spec = build_model_spec(
            region_order=("a", "b"),
            output_dims_by_region={"a": 4, "b": 4},
            hidden_units=10,
            device="cpu",
            input_dim=3,
            activation="tanh",
            spectral_radius=1.1,
            rec_constrained=False,
            inp_constrained=False,
            recurrent_connectivity="full",
            recurrent_bottleneck_dim=3,
            batch_first=True,
            inp_noise=0.0,
            act_noise=0.0,
        )
        model = FixationMRNNModel(spec)
        outputs = model(torch.randn(3, 20, 3), torch.zeros(3, 20), noise=False)
        sum(v.pow(2).mean() for v in outputs["output_by_region"].values()).backward()

        gradients = {name: parameter.grad is not None for name, parameter in model.named_parameters()}
        self.assertFalse(gradients["mrnn.W_rec"])
        self.assertFalse(gradients["mrnn.W_rec_mask"])
        # The blocks that actually drive the dynamics do get gradients.
        self.assertTrue(gradients["_within_region_param_a"])
        self.assertTrue(gradients["_inter_left_a_b"])
        self.assertTrue(gradients["mrnn.W_inp"])
        self.assertTrue(gradients["output_heads.a.weight"])

    def test_no_bottleneck_gives_dense_full_rank_inter_region_blocks(self) -> None:
        """The unconstrained baseline every bottleneck result has to be measured against.

        Before this existed the inter-region blocks were *always* factorized, so there was
        no way to fit a model with no rank constraint at all.
        """
        torch.manual_seed(0)
        spec = build_model_spec(
            region_order=("a", "b"),
            output_dims_by_region={"a": 4, "b": 4},
            hidden_units=12,
            device="cpu",
            input_dim=2,
            activation="tanh",
            spectral_radius=1.0,
            rec_constrained=False,
            inp_constrained=False,
            recurrent_connectivity="full",
            recurrent_bottleneck_dim=None,
            batch_first=True,
            inp_noise=0.0,
            act_noise=0.0,
        )
        model = FixationMRNNModel(spec)
        self.assertFalse(model.has_inter_region_bottleneck)
        self.assertIsNone(model.inter_region_bottleneck_dim)
        block = model.recurrent_weight_matrix().detach()[0:12, 12:24]
        self.assertEqual(int(np.linalg.matrix_rank(block.numpy())), 12)

    def test_rank_constraint_is_visible_in_the_assembled_matrix(self) -> None:
        for requested in (2, 5):
            torch.manual_seed(0)
            spec = build_model_spec(
                region_order=("a", "b"),
                output_dims_by_region={"a": 4, "b": 4},
                hidden_units=12,
                device="cpu",
                input_dim=2,
                activation="tanh",
                spectral_radius=1.0,
                rec_constrained=False,
                inp_constrained=False,
                recurrent_connectivity="full",
                recurrent_bottleneck_dim=requested,
                batch_first=True,
                inp_noise=0.0,
                act_noise=0.0,
            )
            model = FixationMRNNModel(spec)
            self.assertTrue(model.has_inter_region_bottleneck)
            block = model.recurrent_weight_matrix().detach()[0:12, 12:24]
            self.assertEqual(int(np.linalg.matrix_rank(block.numpy())), requested)

    def test_full_rank_factorization_is_not_the_same_model_as_dense(self) -> None:
        """``rank = hidden_units`` reaches full rank but carries twice the parameters and
        optimizes through a product, so it cannot stand in for the dense baseline."""
        counts = {}
        for requested in (None, 12):
            torch.manual_seed(0)
            spec = build_model_spec(
                region_order=("a", "b"),
                output_dims_by_region={"a": 4, "b": 4},
                hidden_units=12,
                device="cpu",
                input_dim=2,
                activation="tanh",
                spectral_radius=1.0,
                rec_constrained=False,
                inp_constrained=False,
                recurrent_connectivity="full",
                recurrent_bottleneck_dim=requested,
                batch_first=True,
                inp_noise=0.0,
                act_noise=0.0,
            )
            model = FixationMRNNModel(spec)
            counts[requested] = sum(p.numel() for p in model.parameters())
        self.assertGreater(counts[12], counts[None])

    def test_initial_spectral_radius_is_actually_applied(self) -> None:
        """The setting was passed to the underlying mrnntorch object whose W_rec this
        wrapper overwrites, so it silently did nothing for every block-parameterized run.

        The spectral radius sets how long the recurrent modes persist, which is the main
        handle on how much temporal structure the autonomous dynamics can carry -- a
        swept axis that does not move is worse than no axis at all.
        """
        for requested in (0.5, 0.9, 1.3):
            torch.manual_seed(0)
            spec = build_model_spec(
                region_order=("a", "b"),
                output_dims_by_region={"a": 3, "b": 3},
                hidden_units=8,
                device="cpu",
                input_dim=2,
                activation="tanh",
                spectral_radius=requested,
                rec_constrained=False,
                inp_constrained=False,
                recurrent_connectivity="full",
                recurrent_bottleneck_dim=2,
                batch_first=True,
                inp_noise=0.0,
                act_noise=0.0,
            )
            model = FixationMRNNModel(spec)
            eigenvalues = torch.linalg.eigvals(model.recurrent_weight_matrix().detach())
            realised = float(torch.max(torch.abs(eigenvalues)))
            self.assertAlmostEqual(realised, requested, places=4)

    def test_two_spectral_radii_give_different_initial_weights(self) -> None:
        """The regression test for how this was found: identical trained weights across
        a swept axis."""
        specs = []
        for requested in (0.9, 1.1):
            torch.manual_seed(0)
            specs.append(
                build_model_spec(
                    region_order=("a", "b"),
                    output_dims_by_region={"a": 3, "b": 3},
                    hidden_units=6,
                    device="cpu",
                    input_dim=2,
                    activation="tanh",
                    spectral_radius=requested,
                    rec_constrained=False,
                    inp_constrained=False,
                    recurrent_connectivity="full",
                    recurrent_bottleneck_dim=2,
                    batch_first=True,
                    inp_noise=0.0,
                    act_noise=0.0,
                )
            )
        weights = []
        for spec in specs:
            torch.manual_seed(0)
            weights.append(FixationMRNNModel(spec).recurrent_weight_matrix().detach())
        self.assertFalse(torch.allclose(weights[0], weights[1]))

    def test_spectral_radius_of_none_leaves_the_initialization_alone(self) -> None:
        torch.manual_seed(0)
        spec = build_model_spec(
            region_order=("a", "b"),
            output_dims_by_region={"a": 3, "b": 3},
            hidden_units=6,
            device="cpu",
            input_dim=2,
            activation="tanh",
            spectral_radius=None,
            rec_constrained=False,
            inp_constrained=False,
            recurrent_connectivity="full",
            recurrent_bottleneck_dim=2,
            batch_first=True,
            inp_noise=0.0,
            act_noise=0.0,
        )
        model = FixationMRNNModel(spec)
        self.assertTrue(torch.isfinite(model.recurrent_weight_matrix()).all())

    def test_lr_schedule_names_are_normalized(self) -> None:
        self.assertEqual(normalize_lr_schedule(None), "constant")
        self.assertEqual(normalize_lr_schedule("none"), "constant")
        self.assertEqual(normalize_lr_schedule("Cosine"), "cosine")
        self.assertEqual(normalize_lr_schedule("StepLR"), "step")
        with self.assertRaises(ValueError):
            normalize_lr_schedule("triangular")

    def test_schedules_move_the_learning_rate_and_constant_does_not(self) -> None:
        """The logged learning rate is what makes a schedule auditable after the fact."""
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

            base = dict(
                dataset_cfg_path=str(cfg_path),
                dataframe_filename="combined.pkl",
                timeline_filename="timeline.pkl",
                target_mode="raw_fr",
                hidden_units=3,
                epochs=20,
                lr=1e-2,
                seed=5,
                device="cpu",
                spectral_radius=1.0,
                temporal_basis_count=0,
            )
            rates = {}
            for name, extra in [
                ("constant", {}),
                ("cosine", dict(lr_schedule="cosine", lr_min_factor=0.01)),
                ("step", dict(lr_schedule="step", lr_step_size=5, lr_step_gamma=0.5)),
                ("warmup", dict(lr_schedule="constant", lr_warmup_iterations=5)),
            ]:
                result = train_one_initialization(
                    FixationMRNNRunSettings(**base, **extra),
                    run_dir=root / name,
                    seed=5,
                    overwrite=True,
                )
                rates[name] = result["history"]["learning_rate"].to_numpy(dtype=float)

            self.assertTrue(np.allclose(rates["constant"], 1e-2))
            self.assertLess(rates["cosine"][-1], rates["cosine"][0])
            self.assertLess(rates["step"][-1], rates["step"][0])
            # Warmup starts below the target rate and climbs back to it.
            self.assertLess(rates["warmup"][0], 1e-2)
            self.assertAlmostEqual(float(rates["warmup"][-1]), 1e-2, places=9)

    def test_best_iterate_is_checkpointed_separately_from_the_final_one(self) -> None:
        """The saved model must be the best one the run found, not the last one it saw.

        The optimizer's trajectory on this problem spikes, so the final iterate samples a
        noisy curve at an arbitrary point. Both checkpoints are written: analyses read the
        best, and the gap between them stays available as a convergence diagnostic.
        """
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
                hidden_units=4,
                epochs=40,
                # Large enough to make the loss curve non-monotone, which is the regime
                # the best-iterate rule exists for.
                lr=5e-2,
                seed=4242,
                device="cpu",
                spectral_radius=1.0,
                temporal_basis_count=0,
            )
            run_dir = root / "run"
            result = train_one_initialization(settings, run_dir=run_dir, seed=4242, overwrite=True)

            self.assertTrue((run_dir / "checkpoint_final.pth").exists())
            self.assertTrue((run_dir / "checkpoint_best.pth").exists())

            history = result["history"]
            losses = history["loss"].to_numpy(dtype=float)
            # The recorded best must be the minimum of the loss the run actually logged.
            self.assertAlmostEqual(float(result["best_loss"]), float(np.nanmin(losses)), places=10)
            self.assertEqual(int(result["best_iteration"]), int(np.nanargmin(losses)) + 1)
            self.assertGreaterEqual(float(result["final_over_best"]), 1.0)

            best = torch.load(run_dir / "checkpoint_best.pth", map_location="cpu", weights_only=False)
            self.assertEqual(best["selection_rule"], "min_total_loss")
            self.assertEqual(int(best["selected_iteration"]), int(result["best_iteration"]))

            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertTrue(manifest["has_best_checkpoint"])
            self.assertAlmostEqual(manifest["best_loss"], float(result["best_loss"]), places=10)

    def test_replay_reads_the_best_checkpoint_by_default(self) -> None:
        """``prefer`` selects which iterate is replayed, and the two differ when the run
        did not end on its best step."""
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
                hidden_units=4,
                epochs=40,
                lr=5e-2,
                seed=99,
                device="cpu",
                spectral_radius=1.0,
                temporal_basis_count=0,
            )
            run_dir = root / "run"
            train_one_initialization(settings, run_dir=run_dir, seed=99, overwrite=True)

            self.assertEqual(resolve_checkpoint_path(run_dir).name, "checkpoint_best.pth")
            self.assertEqual(resolve_checkpoint_path(run_dir, prefer="final").name, "checkpoint_final.pth")

            best_replay = replay_fixation_mrnn_run(run_dir, device="cpu")
            final_replay = replay_fixation_mrnn_run(run_dir, device="cpu", prefer="final")
            self.assertEqual(int(best_replay["checkpoint"]["selected_iteration"]), int(
                json.loads((run_dir / "manifest.json").read_text())["best_iteration"]
            ))
            self.assertNotIn("selected_iteration", final_replay["checkpoint"])

    def test_legacy_runs_without_a_best_checkpoint_fall_back_to_the_final_one(self) -> None:
        """Every run already on disk predates this change and must keep loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            (run_dir / "checkpoint_final.pth").write_bytes(b"")
            self.assertEqual(resolve_checkpoint_path(run_dir).name, "checkpoint_final.pth")
            self.assertEqual(resolve_checkpoint_path(run_dir, prefer="final").name, "checkpoint_final.pth")

    def test_training_divergence_threshold_writes_failed_manifest(self) -> None:
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
                epochs=2,
                seed=779,
                device="cpu",
                spectral_radius=1.0,
                temporal_basis_count=0,
                divergence_loss_threshold=0.0,
                divergence_patience=1,
                divergence_min_iteration=1,
            )

            with self.assertRaisesRegex(RuntimeError, "Training diverged"):
                train_fixation_mrnn_scratch(settings, scratch_id="test_diverged", overwrite=True)

            run_dir = analysis_root / "ephys/modeling/fixation_mrnn/scratch/test_diverged"
            self.assertFalse((run_dir / "checkpoint_final.pth").exists())
            self.assertTrue((run_dir / "training_failed.json").exists())
            with (run_dir / "manifest.json").open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure_reason"], "loss_above_divergence_threshold")


if __name__ == "__main__":
    unittest.main()


class TestSparsityMasks(unittest.TestCase):
    """Sparsity and low rank are independent constraints, and the code must keep them so."""

    @staticmethod
    def _model(**overrides):
        base = dict(
            region_order=("a", "b"), output_dims_by_region={"a": 4, "b": 4}, hidden_units=16,
            device="cpu", input_dim=3, activation="tanh", spectral_radius=1.0,
            rec_constrained=False, inp_constrained=False, recurrent_connectivity="full",
            recurrent_bottleneck_dim=None, batch_first=True, inp_noise=0.0, act_noise=0.0,
        )
        base.update(overrides)
        torch.manual_seed(0)
        return FixationMRNNModel(build_model_spec(**base))

    def _cross_block(self, model):
        slices = model.hidden_region_slices()
        return model.recurrent_weight_matrix().detach().numpy()[slices["b"], slices["a"]]

    def test_a_sparse_block_stays_high_rank(self) -> None:
        """The whole point of testing both: 25% of entries is not the same constraint as
        rank 4, even at the same parameter cost."""
        block = self._cross_block(self._model(cross_region_density=0.25))
        self.assertLess(float(np.mean(np.abs(block) > 1e-12)), 0.4)
        self.assertGreater(int(np.linalg.matrix_rank(block)), 8)

    def test_a_low_rank_block_stays_dense(self) -> None:
        block = self._cross_block(self._model(recurrent_bottleneck_dim=4))
        self.assertGreater(float(np.mean(np.abs(block) > 1e-12)), 0.9)
        self.assertEqual(int(np.linalg.matrix_rank(block)), 4)

    def test_masked_entries_receive_no_gradient_and_stay_zero(self) -> None:
        """Masked entries must be structurally absent, not merely initialised small --
        otherwise the constraint is a penalty and the model is not actually smaller."""
        model = self._model(cross_region_density=0.3)
        outputs = model(torch.randn(2, 12, 3), torch.zeros(2, 32), noise=False)
        sum(v.pow(2).mean() for v in outputs["output_by_region"].values()).backward()

        mask = model._block_masks[("a", "b")].numpy()
        parameter = model._inter_region_dense_params[("a", "b")]
        gradient = parameter.grad.detach().numpy()
        # Where the mask is zero the parameter cannot influence the loss.
        self.assertEqual(mask.shape, gradient.shape)
        self.assertLess(float(np.abs(gradient[mask == 0]).max()), 1e-12)
        # And the assembled block carries the same number of zeros the mask specifies.
        block = self._cross_block(model)
        self.assertEqual(int(np.sum(np.abs(block) < 1e-12)), int(np.sum(mask == 0)))

    def test_masked_entries_are_not_counted_as_present(self) -> None:
        dense = self._model().effective_recurrent_connections
        sparse = self._model(cross_region_density=0.25).effective_recurrent_connections
        self.assertLess(sparse, dense)

    def test_within_and_cross_density_act_independently(self) -> None:
        model = self._model(within_region_density=0.2, cross_region_density=1.0)
        slices = model.hidden_region_slices()
        weight = model.recurrent_weight_matrix().detach().numpy()
        within = weight[slices["a"], slices["a"]]
        cross = weight[slices["b"], slices["a"]]
        self.assertLess(float(np.mean(np.abs(within) > 1e-12)), 0.35)
        self.assertGreater(float(np.mean(np.abs(cross) > 1e-12)), 0.9)
