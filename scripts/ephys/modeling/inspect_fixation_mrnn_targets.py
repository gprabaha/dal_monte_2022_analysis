"""Inspect minimal fixation mRNN targets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.ephys.modeling import (
    load_fixation_mrnn_config,
    make_targets,
    normalize_target_mode,
    settings_from_config,
    summarize_targets,
)


def _print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n[{title}]")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 40, "display.width", 220):
        print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mrnn-cfg",
        default=str(_REPO_ROOT / "configs" / "ephys_fixation_mrnn.yaml"),
    )
    parser.add_argument("--target-mode", default=None)
    args = parser.parse_args()

    cfg = load_fixation_mrnn_config(args.mrnn_cfg)
    settings = settings_from_config(cfg, overrides={"target_mode": args.target_mode})
    target_mode = normalize_target_mode(settings.target_mode)
    targets = make_targets(settings)

    print(f"[dataframe] {targets.dataframe_path}")
    print(f"[timeline] {targets.timeline_path}")
    print(f"[conditions] {targets.condition_order}")
    print(f"[regions] {targets.region_order}")
    print(f"[input_shape] {targets.input_tensor.shape}")
    print(f"[normalization_scale] {targets.normalization_scale}")
    _print_frame("targets", summarize_targets(targets, target_mode=target_mode))


if __name__ == "__main__":
    main()
