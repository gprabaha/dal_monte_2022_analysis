"""Plot within-region pair-condition mean neural xcorr summaries."""

from plot_fixation_neural_cross_correlation_pair_condition_means import run_plot_cli


def main() -> None:
    run_plot_cli(default_analysis_kind="within")


if __name__ == "__main__":
    main()
