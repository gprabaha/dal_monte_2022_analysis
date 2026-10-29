"""Train one cell of the partner-identity grid: one architecture, one seed.

Deliberately a sibling of ``train_fixation_mrnn_into_run_dir.py`` rather than a mode of it:
the generic script resolves its target through ``FixationMRNNRunSettings.dataframe_filename``
and the untouched ``build_mrnn_training_dataframe``, which hard-casts ``region`` to the four
canonical categories. This script instead loads the canonical combined PSTH directly and
builds the architecture's target itself (full population PCA; a self-pair duplicates its
region into two blocks, a cross-pair just names its two canonical regions) via
``fixation_mrnn_partner_identity.build_architecture_targets``, and hands it to
``train_one_initialization`` through its ``targets=`` override -- the one hook that lets a
caller substitute its own target without duplicating the training loop.

    conda run -n gaze_processing python scripts/ephys/modeling/train_fixation_mrnn_partner_identity_cell.py \
        --dataset-cfg configs/dataset.yaml \
        --scored-region bla --partner-region bla --arm self \
        --label self_bla --run-dir <destination> --seed <seed>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_partner_identity import (
    PartnerIdentityArchitecture,
    TARGET_BUILD_SETTINGS,
    build_architecture_targets,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_bridge import (
    MRNN_REGION_ORDER,
    load_combined_fixation_psth,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_targets import build_fixation_mrnn_targets_from_dataframe
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_training import (
    FixationMRNNRunSettings,
    TrainingDivergedError,
    train_one_initialization,
)

#: The base model every task in the final series shares (``final/base_model.yaml``), plus
#: the frozen protocol (``chapter/00_training_protocol/selected_protocol.yaml``). Kept as
#: literal defaults here, the same way task 01's ladder freezes them into its own cells,
#: so this grid is trained under the identical objective and recipe.
BASE_SETTINGS = dict(
    target_mode="region_pcs", pca_variance_threshold=0.95, pca_n_components=42,
    normalize_targets=True, normalization_stabilizer=5.0, temporal_basis_count=0,
    hidden_units=40, activation="tanh", spectral_radius=1.1,
    recurrent_connectivity="full", recurrent_bottleneck_dim=None, within_region_bottleneck_dim=None,
    temporal_derivative_loss_scale=1.0, temporal_curvature_loss_scale=0.5,
    l1_weight_scale=0.0, train_initial_state=True,
    lr=3e-4, lr_schedule="cosine", gradient_clip_norm=0.05, lr_warmup_iterations=0, lr_min_factor=0.01,
    loss_aggregation="minimax", minimax_temperature=0.01,
    condition_loss_weighting="uniform", pc_loss_weighting="uniform",
    divergence_loss_threshold=None,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one cell of the partner-identity grid.")
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--input-subdir", default=TARGET_BUILD_SETTINGS["input_subdir"],
                        help="Canonical combined-PSTH subdir (no per-seed relabelling any more).")
    parser.add_argument("--dataframe-filename", default=TARGET_BUILD_SETTINGS["dataframe_filename"])
    parser.add_argument("--timeline-filename", default=TARGET_BUILD_SETTINGS["timeline_filename"])
    parser.add_argument("--scored-region", required=True)
    parser.add_argument("--partner-region", required=True)
    parser.add_argument("--arm", required=True, choices=("self", "cross"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override BASE_SETTINGS' learning rate -- used for a convergence retry.")
    parser.add_argument("--within-region-bottleneck-dim", type=int, default=None,
                        help="Override BASE_SETTINGS' within_region_bottleneck_dim (None reproduces the dense grid).")
    parser.add_argument("--recurrent-bottleneck-dim", type=int, default=None,
                        help="Override BASE_SETTINGS' recurrent_bottleneck_dim (None reproduces the dense grid).")
    args = parser.parse_args()

    arch = PartnerIdentityArchitecture(label=args.label, arm=args.arm, scored_region=args.scored_region,
                                       partner_region=args.partner_region)
    region_order = arch.virtual_regions()

    result = load_combined_fixation_psth(
        args.dataset_cfg, input_subdir=args.input_subdir,
        dataframe_filename=args.dataframe_filename, timeline_filename=args.timeline_filename,
    )
    # One normalisation scale, pooled once over the plain canonical dataframe, shared by
    # every architecture -- otherwise a self-pair's duplicated region would pool a slightly
    # different scale than a cross-pair's untouched one (see ``build_architecture_targets``).
    canonical_build_kwargs = {k: v for k, v in TARGET_BUILD_SETTINGS.items()
                             if k not in ("input_subdir", "dataframe_filename", "timeline_filename")}
    canonical = build_fixation_mrnn_targets_from_dataframe(
        result.dataframe, timeline_s=result.timeline_s_rel, region_order=MRNN_REGION_ORDER, **canonical_build_kwargs,
    )
    targets = build_architecture_targets(
        arch, result.dataframe, timeline_s=result.timeline_s_rel,
        normalization_scale_override=canonical.normalization_scale,
    )

    settings_kwargs = {k: v for k, v in BASE_SETTINGS.items() if k not in
                       ("normalize_targets", "normalization_stabilizer", "pca_variance_threshold")}
    if args.within_region_bottleneck_dim is not None:
        settings_kwargs["within_region_bottleneck_dim"] = args.within_region_bottleneck_dim
    if args.recurrent_bottleneck_dim is not None:
        settings_kwargs["recurrent_bottleneck_dim"] = args.recurrent_bottleneck_dim
    if args.lr is not None:
        settings_kwargs["lr"] = float(args.lr)
    settings = FixationMRNNRunSettings(
        region_order=region_order, epochs=int(args.epochs), seed=int(args.seed),
        device=args.device or "auto", **settings_kwargs,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "architecture.json").write_text(
        __import__("json").dumps({"label": arch.label, "arm": arch.arm, "scored_region": arch.scored_region,
                                  "partner_region": arch.partner_region, "region_order": list(region_order),
                                  "input_subdir": args.input_subdir, "lr": settings_kwargs["lr"],
                                  "within_region_bottleneck_dim": settings_kwargs["within_region_bottleneck_dim"],
                                  "recurrent_bottleneck_dim": settings_kwargs["recurrent_bottleneck_dim"]}, indent=1)
    )

    try:
        out = train_one_initialization(settings, run_dir=run_dir, seed=int(args.seed),
                                       overwrite=bool(args.overwrite), targets=targets)
    except TrainingDivergedError as error:
        print(f"[partner_identity] diverged: {error}")
        raise SystemExit(0)

    print(f"[partner_identity] run dir:    {out['run_dir']}")
    print(f"[partner_identity] best iter:  {out['best_iteration']} (loss {out['best_loss']:.6g})")
    print(f"[partner_identity] final/best: {out['final_over_best']:.4f}")


if __name__ == "__main__":
    main()
