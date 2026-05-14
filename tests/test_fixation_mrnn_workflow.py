"""Tests for the fixation mRNN target, shuffle, and smoke-training workflow."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd
import torch

from dal_monte_2022_analysis.ephys.modeling import (
    FixationMRNNRunSettings,
    FixationMRNNModel,
    build_fixation_mrnn_targets_from_dataframe,
    build_model_spec,
    compute_fixation_mrnn_currents,
    compute_fixation_mrnn_eigenvalues,
    derive_internal_feature_order_by_region,
    derive_internal_region_order,
    replay_fixation_mrnn_run,
    train_fixation_mrnn_scratch,
)


_HAS_MRNNTORCH = importlib.util.find_spec("mrnntorch") is not None


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
            unit_uuid = f"{region}_unit_{unit_idx}"
            for partition, category, interactive_state, offset in condition_rows:
                base = 10.0 * region_idx + unit_idx + offset
                rows.append(
                    {
                        "date": "20990101",
                        "unit_uuid": unit_uuid,
                        "region": region,
                        "spike_channel": unit_idx,
                        "recorded_agent": "m1",
                        "average_partition": partition,
                        "fixation_category": category,
                        "interactive_state": interactive_state,
                        "n_trials": 5 + unit_idx,
                        "psth_mean": np.asarray(
                            [base + t for t in range(n_time)],
                            dtype=float,
                        ),
                        "psth_sem": np.ones(n_time, dtype=float),
                    }
                )
    return pd.DataFrame(rows)


def _write_dataset_cfg(path: Path, analysis_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "dataset_name: test_dataset",
                f"raw_data_root: {analysis_root}",
                f"processed_data_root: {analysis_root}",
                f"analysis_output_root: {analysis_root}",
            ]
        ),
        encoding="utf-8",
    )


class TestFixationMRNNTargetsAndShuffling(unittest.TestCase):
    """Checks target construction and deterministic shuffle behavior."""

    def test_target_builder_outputs_raw_and_pc_targets(self) -> None:
        df = _synthetic_combined_dataframe()
        timeline = np.asarray([-0.02, -0.01, 0.0, 0.01], dtype=float)

        targets = build_fixation_mrnn_targets_from_dataframe(
            df,
            timeline_s_rel=timeline,
            normalize_targets=True,
            pca_variance_threshold=0.95,
        )

        self.assertEqual(targets.condition_names, ("face_interactive", "face_non_interactive", "object"))
        self.assertEqual(targets.input_tensor.shape, (3, 4, 3))
        self.assertEqual(targets.raw_targets_by_region["ofc"].shape, (3, 4, 2))
        self.assertEqual(targets.region_unit_counts["accg"], 2)
        self.assertEqual(targets.raw_feature_order_by_region["bla"], ("bla_unit_0", "bla_unit_1"))
        self.assertGreaterEqual(
            float(targets.pca_metadata_by_region["ofc"].cumulative_explained_variance_ratio[
                targets.pca_metadata_by_region["ofc"].n_components - 1
            ]),
            0.95,
        )
        self.assertEqual(
            targets.pc_targets_by_region["dmpfc"].shape[-1],
            len(targets.pc_feature_order_by_region["dmpfc"]),
        )

    def test_seed_derived_shuffles_are_deterministic_and_within_region(self) -> None:
        regions = ("ofc", "bla", "dmpfc", "accg")
        first = derive_internal_region_order(regions, seed=123)
        second = derive_internal_region_order(regions, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(regions))

        features = {
            "ofc": ("ofc_a", "ofc_b", "ofc_c"),
            "bla": ("bla_a", "bla_b"),
        }
        feature_first = derive_internal_feature_order_by_region(features, seed=456)
        feature_second = derive_internal_feature_order_by_region(features, seed=456)
        self.assertEqual(feature_first, feature_second)
        self.assertEqual(set(feature_first["ofc"]), set(features["ofc"]))
        self.assertEqual(set(feature_first["bla"]), set(features["bla"]))


@unittest.skipUnless(_HAS_MRNNTORCH, "installed lowercase mrnntorch is required")
class TestFixationMRNNTorchSmoke(unittest.TestCase):
    """Small smoke tests for installed-mrnntorch model/training paths."""

    def test_model_forward_preserves_canonical_output_keys(self) -> None:
        canonical = ("ofc", "bla", "dmpfc", "accg")
        internal = ("accg", "ofc", "bla", "dmpfc")
        spec = build_model_spec(
            canonical_region_order=canonical,
            internal_region_order=internal,
            output_dims_by_region={region: 2 for region in canonical},
            hidden_units=3,
            activation="softplus",
            rec_constrained=False,
            inp_constrained=False,
            spectral_radius=1.0,
            device="cpu",
        )
        model = FixationMRNNModel(spec)
        inp = torch.zeros((3, 4, 3), dtype=torch.float32)
        x0 = torch.zeros((3, model.total_num_units), dtype=torch.float32)

        out = model(inp, x0, noise=False)

        self.assertEqual(tuple(out["output_by_region"].keys()), canonical)
        self.assertEqual(out["output"].shape, (3, 4, 8))
        self.assertEqual(out["x_seq"].shape[-1], model.total_num_units)

    def test_two_epoch_scratch_training_replay_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            analysis_root = root / "analysis"
            avg_root = analysis_root / "ephys/psth/fixation_psth_averages"
            avg_root.mkdir(parents=True, exist_ok=True)
            cfg_path = root / "dataset.yaml"
            _write_dataset_cfg(cfg_path, analysis_root)

            dataframe_filename = "combined.pkl"
            timeline_filename = "timeline.pkl"
            _synthetic_combined_dataframe().to_pickle(avg_root / dataframe_filename)
            with (avg_root / timeline_filename).open("wb") as f:
                pickle.dump(np.asarray([-0.02, -0.01, 0.0, 0.01], dtype=float), f)

            settings = FixationMRNNRunSettings(
                dataset_cfg_path=str(cfg_path),
                input_subdir="ephys/psth/fixation_psth_averages",
                dataframe_filename=dataframe_filename,
                timeline_filename=timeline_filename,
                output_subdir="ephys/modeling/fixation_mrnn",
                target_mode="raw_fr",
                hidden_units=3,
                epochs=2,
                seed=777,
                device="cpu",
                spectral_radius=1.0,
            )
            result = train_fixation_mrnn_scratch(
                settings,
                scratch_id="test_raw",
                overwrite=True,
            )
            self.assertTrue((Path(result["run_dir"]) / "checkpoint_final.pth").exists())
            self.assertTrue((Path(result["run_dir"]) / "history.csv").exists())

            replay = replay_fixation_mrnn_run(result["run_dir"], device="cpu")
            self.assertEqual(
                tuple(replay["canonical_output_by_region"].keys()),
                ("ofc", "bla", "dmpfc", "accg"),
            )

            current_df, _ = compute_fixation_mrnn_currents(replay)
            eig_df = compute_fixation_mrnn_eigenvalues(replay)
            self.assertFalse(current_df.empty)
            self.assertFalse(eig_df.empty)


if __name__ == "__main__":
    unittest.main()
