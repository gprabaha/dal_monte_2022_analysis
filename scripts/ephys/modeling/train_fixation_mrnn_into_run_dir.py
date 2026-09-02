"""Train one fixation mRNN into an explicit run directory.

The existing ``train_fixation_mrnn.py`` resolves its output as
``<analysis_output_root>/<output_subdir>/scratch/<scratch-id>``, which is the right
layout for exploratory runs but gives a sweep no control over where its cells land.
The rebuilt chapter runs are organised by task and configuration instead, so this entry
point takes the destination directly.

It also does not retry on divergence. The training CLI's retry path resamples the seed,
which is correct for "just get me a fit" and wrong for any sweep that is counting how
often a configuration fails.

    conda run -n gaze_processing python scripts/ephys/modeling/train_fixation_mrnn_into_run_dir.py \
        --mrnn-cfg <run_config.yaml> --run-dir <destination> --seed <seed>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.ephys.modeling import (
    load_fixation_mrnn_config,
    settings_from_config,
    train_one_initialization,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_training import TrainingDivergedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one fixation mRNN into an explicit run directory.")
    parser.add_argument("--mrnn-cfg", required=True, help="Frozen run config YAML for this cell.")
    parser.add_argument("--run-dir", required=True, help="Destination directory for this run's artifacts.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_fixation_mrnn_config(args.mrnn_cfg)
    overrides: dict[str, object] = {}
    if args.device is not None:
        overrides["device"] = args.device
    settings = settings_from_config(cfg, overrides=overrides)
    seed = int(args.seed if args.seed is not None else settings.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = train_one_initialization(settings, run_dir=run_dir, seed=seed, overwrite=bool(args.overwrite))
    except TrainingDivergedError as error:
        # A divergence is a result, not a crash: the sweep needs the failure recorded in
        # place. train_one_initialization has already written training_failed.json.
        print(f"[modeling] diverged: {error}")
        raise SystemExit(0)

    print(f"[modeling] run dir:    {result['run_dir']}")
    print(f"[modeling] best iter:  {result['best_iteration']} (loss {result['best_loss']:.6g})")
    print(f"[modeling] final/best: {result['final_over_best']:.4f}")
    print(f"[modeling] checkpoints: {result['checkpoint_path'].name}, "
          f"{result['best_checkpoint_path'].name if result['best_checkpoint_path'] else 'none'}")


if __name__ == "__main__":
    main()
