"""Build cross-region date-level pair-condition mean neural xcorr outputs."""

import argparse

from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_pair_condition_means import (
    build_fixation_neural_cross_correlation_pair_condition_mean_settings_from_config,
    run_cross_region_fixation_neural_cross_correlation_pair_condition_means,
)



def _print_analysis_summary(signal_column: str, signal_summary: dict[str, object]) -> None:
    signal_variant = str(signal_summary.get("signal_variant", signal_column))
    print(
        "[analysis] cross-region pair-condition means "
        f"[{signal_variant} / {signal_column}]: wrote "
        f"{signal_summary.get('n_dates_written', 0)}/{signal_summary.get('n_dates_total', 0)} date files "
        f"from {signal_summary.get('n_session_files_total', 0)} session pair-average files; "
        f"{signal_summary.get('n_pair_condition_mean_rows_total', 0)} pair-condition rows"
    )
    output_paths = signal_summary.get("output_paths") or []
    if output_paths:
        print(f"[analysis] first output: {output_paths[0]}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cross-region date-level pair-condition mean neural xcorr outputs.",
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
    args = parser.parse_args()

    settings = build_fixation_neural_cross_correlation_pair_condition_mean_settings_from_config(
        dataset_cfg_path=args.dataset_cfg,
        ephys_fixation_neural_cross_correlation_cfg_path=args.ephys_fixation_neural_cross_correlation_cfg,
    )
    summary = run_cross_region_fixation_neural_cross_correlation_pair_condition_means(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
    )
    signal_summaries = summary.get("signal_summaries", {})
    for idx, signal_column in enumerate(summary.get("signal_input_columns", [])):
        _print_analysis_summary(str(signal_column), signal_summaries.get(str(signal_column), {}))
        if idx != len(summary.get("signal_input_columns", [])) - 1:
            print()


if __name__ == "__main__":
    main()
