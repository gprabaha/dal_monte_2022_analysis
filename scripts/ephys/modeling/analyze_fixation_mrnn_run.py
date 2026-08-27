"""Run minimal analyses for a trained fixation mRNN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.ephys.modeling import (
    extract_region_currents,
    reconstruction_accuracy,
    replay_fixation_mrnn_run,
    variance_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--analysis",
        choices=("currents", "reconstruction", "variance"),
        required=True,
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    replay = replay_fixation_mrnn_run(run_dir, device=args.device, noise=False)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.analysis == "currents":
        frame, _ = extract_region_currents(replay)
    elif args.analysis == "reconstruction":
        frame = reconstruction_accuracy(replay)
    else:
        frame = variance_comparison(replay)
    out_path = out_dir / f"{args.analysis}.csv"
    frame.to_csv(out_path, index=False)
    print(f"[modeling] wrote analysis output: {out_path}")


if __name__ == "__main__":
    main()
