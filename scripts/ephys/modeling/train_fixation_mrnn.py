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
    parser.add_argument(
        "--initial-state",
        choices=("zeros", "normal", "uniform"),
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_fixation_mrnn_config(args.mrnn_cfg)
    settings = settings_from_config(cfg, overrides=_override_dict(args))

    scratch_id = args.scratch_id or str(cfg.get("scratch_id", "latest"))
    result = train_fixation_mrnn_scratch(
        settings,
        scratch_id=scratch_id,
        overwrite=bool(args.overwrite),
    )

    print(f"[modeling] run dir: {result['run_dir']}")
    if "checkpoint_path" in result:
        print(f"[modeling] checkpoint: {result['checkpoint_path']}")
    if "index" in result:
        print(f"[modeling] trained initializations: {len(result['index'])}")


if __name__ == "__main__":
    main()
