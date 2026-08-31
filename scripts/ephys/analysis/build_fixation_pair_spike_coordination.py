"""Build per-session neural pair spike-coordination tables.

One session's output holds every simultaneously recorded pair, for each fixation
condition, with the observed circular cross-correlation and both nulls.

    python scripts/ephys/analysis/build_fixation_pair_spike_coordination.py \
        --date 01312018 --session 5
"""

import argparse

from dal_monte_2022_analysis.ephys.analysis.fixation_pair_spike_coordination import (
    build_pair_spike_coordination_settings_from_config,
    iter_session_trial_paths,
    run_pair_spike_coordination_build,
    verify_null_identities,
    verify_null_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixation-condition spike coordination for neural pairs.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--coordination-cfg",
        default="configs/ephys_fixation_pair_spike_coordination.yaml",
    )
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--max-procs", type=int, default=None)
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave sessions that already have an output file untouched.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run the null identity and sensitivity checks, then exit.",
    )
    args = parser.parse_args()

    if args.verify_only:
        print("Null construction identities:")
        print(verify_null_identities().to_string(index=False))
        print("\nNull sensitivity (synthetic positive controls):")
        print(verify_null_sensitivity().round(3).to_string(index=False))
        return

    settings = build_pair_spike_coordination_settings_from_config(
        dataset_cfg_path=args.dataset_cfg,
        coordination_cfg_path=args.coordination_cfg,
    )
    if args.date:
        settings.dates = list(args.date)
    if args.session:
        settings.sessions = list(args.session)
    if args.no_parallel:
        settings.use_parallel = False
    if args.max_procs is not None:
        settings.max_procs = int(args.max_procs)
    if args.test_single:
        settings.test_single = True

    rows = iter_session_trial_paths(settings)
    print(f"sessions selected: {len(rows)}")
    written = run_pair_spike_coordination_build(settings, overwrite=not args.skip_existing)
    print(f"sessions written : {len(written)}")
    if written:
        print(f"first output     : {written[0]}")


if __name__ == "__main__":
    main()
