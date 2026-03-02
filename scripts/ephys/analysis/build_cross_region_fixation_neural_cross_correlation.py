"""Build cross-region fixation-level neural PSTH cross-correlations."""

import argparse
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation import (
    CROSS_ANALYSIS_KIND,
    apply_fixation_neural_cross_correlation_cli_overrides,
    build_fixation_neural_cross_correlation_settings_from_config,
    coerce_nonempty_str_list,
    iter_fixation_neural_cross_correlation_output_paths,
    print_fixation_neural_cross_correlation_example,
    run_cross_region_fixation_neural_cross_correlation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cross-region fixation-level neural PSTH cross-correlations.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--ephys-fixation-neural-crosscorr-cfg",
        default="configs/ephys_fixation_neural_cross_correlation.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--anchor-region", default=None)
    parser.add_argument("--partner-region", action="append", default=None)
    parser.add_argument("--include-region", action="append", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--max-lag", type=int, default=None)
    parser.add_argument(
        "--signal-transform",
        choices=["none", "demean", "zscore"],
        default=None,
    )
    parser.add_argument(
        "--xcorr-normalization",
        choices=["none", "energy"],
        default=None,
    )
    parser.add_argument("--no-show-example", action="store_true")
    parser.add_argument("--example-max-lags", type=int, default=12)
    args = parser.parse_args()

    settings = build_fixation_neural_cross_correlation_settings_from_config(
        dataset_cfg_path=args.dataset_cfg,
        ephys_fixation_neural_crosscorr_cfg_path=args.ephys_fixation_neural_crosscorr_cfg,
    )
    settings = apply_fixation_neural_cross_correlation_cli_overrides(
        settings,
        anchor_region=args.anchor_region,
        partner_regions=coerce_nonempty_str_list(args.partner_region),
        include_regions=coerce_nonempty_str_list(args.include_region),
        no_parallel=args.no_parallel,
        test_single=args.test_single,
        max_lag=args.max_lag,
        signal_transform=args.signal_transform,
        xcorr_normalization=args.xcorr_normalization,
    )

    summary = run_cross_region_fixation_neural_cross_correlation(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
    )
    print(
        "[analysis] cross-region fixation neural xcorr: "
        f"wrote {summary.get('n_sessions_written', 0)}/{summary.get('n_sessions_total', 0)} session files"
    )

    if not args.no_show_example:
        paths = iter_fixation_neural_cross_correlation_output_paths(
            dataset_cfg_path=args.dataset_cfg,
            output_subdir=settings.cross_output_subdir,
            output_filename=settings.cross_output_filename,
            date=args.date,
            session=args.session,
        )
        if not paths:
            print("\n[example] No cross-region output files found to preview.")
            return
        print_fixation_neural_cross_correlation_example(
            paths[0],
            analysis_kind=CROSS_ANALYSIS_KIND,
            max_lags=args.example_max_lags,
        )


if __name__ == "__main__":
    main()
