"""Rebuild fixation mRNN experiment index.csv from run artifacts."""

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
    rebuild_fixation_mrnn_experiment_index,
    settings_from_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild index.csv for a fixation mRNN experiment.",
    )
    parser.add_argument(
        "--mrnn-cfg",
        default=str(_REPO_ROOT / "configs" / "ephys_fixation_mrnn.yaml"),
    )
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    settings = settings_from_config(load_fixation_mrnn_config(args.mrnn_cfg))
    index_path = rebuild_fixation_mrnn_experiment_index(
        settings,
        experiment_id=args.experiment_id,
    )
    print(f"[modeling] wrote index: {index_path}")


if __name__ == "__main__":
    main()
