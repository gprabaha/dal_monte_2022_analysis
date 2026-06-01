"""Prepare an indexed fixation mRNN experiment run plan."""

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
    prepare_fixation_mrnn_run_plan,
    settings_from_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic run_plan.csv for fixation mRNN training.",
    )
    parser.add_argument(
        "--mrnn-cfg",
        default=str(_REPO_ROOT / "configs" / "ephys_fixation_mrnn.yaml"),
    )
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--overwrite-seed-plan", action="store_true")
    parser.add_argument(
        "--target-mode",
        action="append",
        default=None,
        help="Target mode to include. May be repeated. Defaults to config target_mode.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_fixation_mrnn_config(args.mrnn_cfg)
    settings = settings_from_config(cfg)
    if args.overwrite_seed_plan:
        settings.overwrite_seed_plan = True
    experiment_id = args.experiment_id or str(cfg.get("experiment_id", "fixation_mrnn_experiment"))
    n_runs = int(args.n_runs if args.n_runs is not None else cfg.get("n_runs", 100))
    seed_start = int(
        args.seed_start if args.seed_start is not None else cfg.get("seed_start", settings.seed)
    )
    target_modes = args.target_mode or [str(cfg.get("target_mode", settings.target_mode))]

    run_plan_path = prepare_fixation_mrnn_run_plan(
        settings,
        experiment_id=experiment_id,
        n_runs=n_runs,
        seed_start=seed_start,
        target_modes=target_modes,
        shuffle_region_order=False,
        shuffle_feature_order=False,
        overwrite=bool(args.overwrite),
    )
    print(f"[modeling] wrote run plan: {run_plan_path}")


if __name__ == "__main__":
    main()
