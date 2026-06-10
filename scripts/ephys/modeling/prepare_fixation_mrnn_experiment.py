"""Prepare persistent seeds for a multiple-initialization fixation mRNN run."""

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
    load_or_create_seed_plan,
    resolve_fixation_mrnn_output_root,
    settings_from_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mrnn-cfg",
        default=str(_REPO_ROOT / "configs" / "ephys_fixation_mrnn.yaml"),
    )
    parser.add_argument("--scratch-id", default=None)
    parser.add_argument("--n-initializations", type=int, default=None)
    parser.add_argument("--overwrite-seed-plan", action="store_true")
    args = parser.parse_args()

    cfg = load_fixation_mrnn_config(args.mrnn_cfg)
    settings = settings_from_config(cfg)
    settings.initialization_mode = "multiple"
    if args.n_initializations is not None:
        settings.n_initializations = int(args.n_initializations)
    scratch_id = args.scratch_id or str(cfg.get("scratch_id", "latest"))
    output_dir = resolve_fixation_mrnn_output_root(settings) / "scratch" / str(scratch_id)
    seeds = load_or_create_seed_plan(
        output_dir,
        settings,
        n_seeds=int(settings.n_initializations),
        overwrite=bool(args.overwrite_seed_plan),
    )
    print(f"[modeling] seed plan: {output_dir / settings.seed_plan_filename}")
    print(f"[modeling] seeds: {len(seeds)}")


if __name__ == "__main__":
    main()
