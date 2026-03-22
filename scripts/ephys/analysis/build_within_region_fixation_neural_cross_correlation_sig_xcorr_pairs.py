"""Build within-region significant xcorr pair summaries for fixation neural xcorr outputs."""

import argparse

import pandas as pd

from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_sig_xcorr_pairs import (
    build_fixation_neural_cross_correlation_sig_xcorr_pair_group_summary_table,
    build_fixation_neural_cross_correlation_sig_xcorr_pairs_settings_from_config,
    resolve_fixation_neural_cross_correlation_sig_xcorr_pairs_signal_columns,
    run_within_region_fixation_neural_cross_correlation_sig_xcorr_pairs,
)


def _print_analysis_summary(signal_column: str, signal_summary: dict[str, object]) -> None:
    signal_variant = str(signal_summary.get("signal_variant", signal_column))
    print(
        "[analysis] within-region sig xcorr pairs "
        f"[{signal_variant} / {signal_column}]: wrote "
        f"{signal_summary.get('n_dates_written', 0)}/{signal_summary.get('n_dates_total', 0)} date files "
        f"from {signal_summary.get('n_session_files_total', 0)} session xcorr files; "
        f"{signal_summary.get('n_summary_rows_total', 0)} pair rows; "
        f"{signal_summary.get('n_group_summary_rows_total', 0)} group rows"
    )


def _print_group_summary_table(signal_column: str, signal_summary: dict[str, object]) -> None:
    signal_variant = str(signal_summary.get("signal_variant", signal_column))
    output_paths = signal_summary.get("output_paths") or []
    table = build_fixation_neural_cross_correlation_sig_xcorr_pair_group_summary_table(output_paths)
    if table.empty:
        print("[table] within-region sig xcorr pairs: no group summary rows")
        return
    display_df = table.drop(columns=["signal_variant", "signal_input_column"], errors="ignore")
    print(f"[table] within-region sig xcorr pairs [{signal_variant} / {signal_column}]")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 2000):
        print(display_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build within-region significant xcorr pair summaries for fixation neural xcorr outputs.",
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
    parser.add_argument("--no-show-table", "--no-show-example", dest="no_show_table", action="store_true")
    args = parser.parse_args()

    settings = build_fixation_neural_cross_correlation_sig_xcorr_pairs_settings_from_config(
        dataset_cfg_path=args.dataset_cfg,
        ephys_fixation_neural_cross_correlation_cfg_path=args.ephys_fixation_neural_cross_correlation_cfg,
    )
    signal_columns = resolve_fixation_neural_cross_correlation_sig_xcorr_pairs_signal_columns(settings)
    summary = run_within_region_fixation_neural_cross_correlation_sig_xcorr_pairs(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
    )

    signal_summaries = summary.get("signal_summaries", {})
    for idx, signal_column in enumerate(signal_columns):
        signal_summary = signal_summaries.get(signal_column, {})
        _print_analysis_summary(signal_column, signal_summary)
        if not args.no_show_table:
            _print_group_summary_table(signal_column, signal_summary)
        if idx != len(signal_columns) - 1:
            print()


if __name__ == "__main__":
    main()
