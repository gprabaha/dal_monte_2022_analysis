"""Run lightweight analyses for a trained fixation mRNN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pandas as pd

from dal_monte_2022_analysis.ephys.modeling import (
    compute_fixation_mrnn_currents,
    compute_fixation_mrnn_eigenvalues,
    compute_fixation_mrnn_flow_fields,
    replay_fixation_mrnn_run,
)
from dal_monte_2022_analysis.utils.io import save_pickle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lightweight checkpoint analyses for fixation mRNN.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--analysis",
        choices=("currents", "eigenvalues", "flow_fields"),
        required=True,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--region", default="ofc")
    parser.add_argument("--condition", default="face_interactive")
    parser.add_argument("--num-points", type=int, default=7)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    replay = replay_fixation_mrnn_run(run_dir, device=args.device, noise=False)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.analysis == "currents":
        current_df, _ = compute_fixation_mrnn_currents(replay)
        out_path = out_dir / "currents.csv"
        current_df.to_csv(out_path, index=False)
    elif args.analysis == "eigenvalues":
        eig_df = compute_fixation_mrnn_eigenvalues(replay)
        out_path = out_dir / "eigenvalues.csv"
        eig_df.to_csv(out_path, index=False)
    else:
        flow_result = compute_fixation_mrnn_flow_fields(
            replay,
            region=args.region,
            condition=args.condition,
            num_points=args.num_points,
        )
        out_path = out_dir / f"flow_fields_{args.region}_{args.condition}.pkl"
        save_pickle(flow_result, out_path)

    print(f"[modeling] wrote analysis output: {out_path}")


if __name__ == "__main__":
    main()
