"""Plot within-region fixation-level neural cross-correlation summaries."""

from plot_fixation_neural_cross_correlation import run_plot_cli


def main() -> None:
    run_plot_cli(default_analysis_kind="within")


if __name__ == "__main__":
    main()
