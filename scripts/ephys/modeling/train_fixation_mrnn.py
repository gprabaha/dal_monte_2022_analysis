"""Train one fixation mRNN scratch or experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.ephys.modeling import (
    load_fixation_mrnn_config,
    settings_from_config,
    train_fixation_mrnn_scratch,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_training import TrainingDivergedError


def _override_dict(args: argparse.Namespace) -> dict[str, object]:
    return {
        "target_mode": args.target_mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "initialization_mode": args.initialization_mode,
        "n_initializations": args.n_initializations,
        "overwrite_seed_plan": True if args.overwrite_seed_plan else None,
        "device": args.device,
        "hidden_units": args.hidden_units,
        "temporal_basis_count": args.temporal_basis_count,
        "temporal_derivative_loss_scale": args.temporal_derivative_loss_scale,
        "temporal_curvature_loss_scale": args.temporal_curvature_loss_scale,
        "correlation_loss_scale": args.correlation_loss_scale,
        "variance_loss_scale": args.variance_loss_scale,
        "fr_reconstruction_loss_scale": args.fr_reconstruction_loss_scale,
        "fr_temporal_derivative_loss_scale": args.fr_temporal_derivative_loss_scale,
        "fr_temporal_curvature_loss_scale": args.fr_temporal_curvature_loss_scale,
        "l1_weight_scale": args.l1_weight_scale,
        "l1_rate_scale": args.l1_rate_scale,
        "l2_weight_scale": args.l2_weight_scale,
        "l2_rate_scale": args.l2_rate_scale,
        "gradient_clip_norm": args.gradient_clip_norm,
        "divergence_loss_threshold": args.divergence_loss_threshold,
        "divergence_patience": args.divergence_patience,
        "divergence_min_iteration": args.divergence_min_iteration,
        "initial_state": args.initial_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one fixation mRNN run.",
    )
    parser.add_argument(
        "--mrnn-cfg",
        default=str(_REPO_ROOT / "configs" / "ephys_fixation_mrnn.yaml"),
    )
    parser.add_argument("--scratch-id", default=None)
    parser.add_argument("--target-mode", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--initialization-mode",
        choices=("single", "multiple"),
        default=None,
    )
    parser.add_argument("--n-initializations", type=int, default=None)
    parser.add_argument("--overwrite-seed-plan", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-units", type=int, default=None)
    parser.add_argument("--temporal-basis-count", type=int, default=None)
    parser.add_argument("--temporal-derivative-loss-scale", type=float, default=None)
    parser.add_argument("--temporal-curvature-loss-scale", type=float, default=None)
    parser.add_argument("--correlation-loss-scale", type=float, default=None)
    parser.add_argument("--variance-loss-scale", type=float, default=None)
    parser.add_argument("--fr-reconstruction-loss-scale", type=float, default=None)
    parser.add_argument("--fr-temporal-derivative-loss-scale", type=float, default=None)
    parser.add_argument("--fr-temporal-curvature-loss-scale", type=float, default=None)
    parser.add_argument("--l1-weight-scale", type=float, default=None)
    parser.add_argument("--l1-rate-scale", type=float, default=None)
    parser.add_argument("--l2-weight-scale", type=float, default=None)
    parser.add_argument("--l2-rate-scale", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--divergence-loss-threshold", type=float, default=None)
    parser.add_argument("--divergence-patience", type=int, default=None)
    parser.add_argument("--divergence-min-iteration", type=int, default=None)
    parser.add_argument("--max-divergence-retries", type=int, default=0)
    parser.add_argument(
        "--initial-state",
        choices=("zeros", "normal", "uniform"),
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_fixation_mrnn_config(args.mrnn_cfg)
    scratch_id = args.scratch_id or str(cfg.get("scratch_id", "latest"))
    base_seed = int(args.seed if args.seed is not None else cfg.get("seed", 123456))
    retry_rng = None
    result = None
    max_retries = max(0, int(args.max_divergence_retries))
    for attempt_idx in range(max_retries + 1):
        if attempt_idx == 0:
            attempt_seed = base_seed
        else:
            if retry_rng is None:
                import numpy as np

                retry_rng = np.random.default_rng(base_seed)
            attempt_seed = int(retry_rng.integers(1, np.iinfo(np.int32).max))
            print(
                f"[modeling] retrying diverged run with seed={attempt_seed} "
                f"(attempt {attempt_idx + 1}/{max_retries + 1})"
            )
        overrides = {**_override_dict(args), "seed": attempt_seed}
        settings = settings_from_config(cfg, overrides=overrides)
        try:
            result = train_fixation_mrnn_scratch(
                settings,
                scratch_id=scratch_id,
                overwrite=bool(args.overwrite),
            )
            break
        except TrainingDivergedError:
            if attempt_idx >= max_retries:
                raise
    if result is None:
        raise RuntimeError("Training did not produce a result.")

    print(f"[modeling] run dir: {result['run_dir']}")
    if "checkpoint_path" in result:
        print(f"[modeling] checkpoint: {result['checkpoint_path']}")
    if "index" in result:
        print(f"[modeling] trained initializations: {len(result['index'])}")


if __name__ == "__main__":
    main()
