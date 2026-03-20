"""Build within-region date-level pair meta-analysis for fixation neural xcorr outputs."""

import argparse

from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_pair_meta_analysis import (
    build_fixation_neural_cross_correlation_pair_meta_analysis_settings_from_config,
    iter_fixation_neural_cross_correlation_pair_meta_analysis_output_paths,
    print_fixation_neural_cross_correlation_pair_meta_analysis_example,
    resolve_fixation_neural_cross_correlation_pair_meta_analysis_signal_columns,
    run_within_region_fixation_neural_cross_correlation_pair_meta_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build within-region date-level pair meta-analysis for fixation neural xcorr outputs.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument(
        "--ephys-fixation-neural-cross-correlation-cfg",
        "--ephys-fixation-neural-crosscorr-cfg",
        dest="ephys_fixation_neural_cross_correlation_cfg",
        default="configs/ephys_fixation_neural_cross_correlation.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--no-show-example", action="store_true")
    parser.add_argument("--example-max-lags", type=int, default=12)
    args = parser.parse_args()

    settings = build_fixation_neural_cross_correlation_pair_meta_analysis_settings_from_config(
        dataset_cfg_path=args.dataset_cfg,
        ephys_fixation_neural_cross_correlation_cfg_path=args.ephys_fixation_neural_cross_correlation_cfg,
    )
    signal_columns = resolve_fixation_neural_cross_correlation_pair_meta_analysis_signal_columns(settings)
    summary = run_within_region_fixation_neural_cross_correlation_pair_meta_analysis(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
    )

    signal_summaries = summary.get("signal_summaries", {})
    for signal_column in signal_columns:
        signal_summary = signal_summaries.get(signal_column, {})
        print(
            "[analysis] within-region neural xcorr pair meta-analysis ["
            f"{signal_column}]: wrote "
            f"{signal_summary.get('n_dates_written', 0)}/{signal_summary.get('n_dates_total', 0)} date files "
            f"from {signal_summary.get('n_session_files_total', 0)} session xcorr files"
        )
        if signal_summary.get("output_paths"):
            print(
                "[analysis] within-region pair meta output ["
                f"{signal_column}]: {signal_summary['output_paths'][0]}"
            )
        if signal_summary.get("csv_output_paths"):
            print(
                "[analysis] within-region pair meta csv ["
                f"{signal_column}]: {signal_summary['csv_output_paths'][0]}"
            )

    if not args.no_show_example:
        for signal_column in signal_columns:
            paths = iter_fixation_neural_cross_correlation_pair_meta_analysis_output_paths(
                dataset_cfg_path=args.dataset_cfg,
                output_subdir=settings.within_output_subdir,
                output_filename=settings.within_output_filename,
                signal_input_column=signal_column,
                date=args.date,
            )
            if not paths:
                print(f"\n[example] No within-region pair meta-analysis output files found for {signal_column}.")
                continue
            print(f"\n[example] within-region pair meta-analysis signal={signal_column}")
            print_fixation_neural_cross_correlation_pair_meta_analysis_example(
                paths[0],
                max_lags=args.example_max_lags,
            )


if __name__ == "__main__":
    main()
