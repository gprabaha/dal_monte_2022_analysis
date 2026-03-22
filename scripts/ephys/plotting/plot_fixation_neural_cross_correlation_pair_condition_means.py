"""Plot pair-condition mean neural xcorr summaries across groups and fixation conditions."""

import argparse

import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_helpers import (
    CROSS_ANALYSIS_KIND,
    WITHIN_ANALYSIS_KIND,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_neural_cross_correlation_pair_condition_means import (
    FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    plot_fixation_neural_cross_correlation_pair_condition_mean_summaries,
)


def _as_float2(values):
    if values is None:
        return None
    if len(values) != 2:
        return None
    return [float(values[0]), float(values[1])]


def _print_mean_lag_table(subset_label: str, result_key: tuple[str, str], result: dict[str, object]) -> None:
    analysis_kind, signal_column = result_key
    signal_variant = str(result.get("signal_variant", signal_column))
    mean_lag_df = result.get("mean_lag_comparisons")
    mean_lag_df = mean_lag_df if isinstance(mean_lag_df, pd.DataFrame) else pd.DataFrame()
    if mean_lag_df.empty:
        print(f"[table] {subset_label} | {analysis_kind} pair-condition means [{signal_variant} / {signal_column}]: no mean-lag comparison rows")
        return
    display_df = mean_lag_df.loc[:, [
        col for col in (
            "group_label",
            "condition_a",
            "condition_b",
            "n_pairs",
            "mean_condition_a",
            "mean_condition_b",
            "mean_difference",
            "p_value",
            "significant",
        ) if col in mean_lag_df.columns
    ]].copy()
    print(f"[table] {subset_label} | {analysis_kind} pair-condition means [{signal_variant} / {signal_column}]")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 2000):
        print(display_df.to_string(index=False))


def _print_outputs(label: str, paths: list[str]) -> None:
    if not paths:
        print(f"[plot] {label}: none")
        return
    print(f"[plot] {label} ({len(paths)}):")
    for path in paths:
        print(f"  {path}")


def run_plot_cli(*, default_analysis_kind: str = "both") -> None:
    parser = argparse.ArgumentParser(
        description="Plot pair-condition mean neural xcorr summaries across groups and fixation conditions.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument(
        "--ephys-fixation-neural-cross-correlation-cfg",
        "--ephys-fixation-neural-crosscorr-cfg",
        dest="ephys_fixation_neural_cross_correlation_cfg",
        default="configs/ephys_fixation_neural_cross_correlation.yaml",
    )
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--output-extension", default=None)
    parser.add_argument("--output-dpi", type=int, default=None)
    parser.add_argument("--figsize", nargs=2, type=float, default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--no-per-day", action="store_true")
    parser.add_argument(
        "--analysis-kind",
        choices=["both", "within", "cross"],
        default=str(default_analysis_kind),
    )
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_neural_cross_correlation_cfg)
    settings = FixationNeuralCrossCorrelationPairConditionMeanPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        within_input_subdir=cfg.get(
            "pair_condition_mean_within_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/within_region",
        ),
        cross_input_subdir=cfg.get(
            "pair_condition_mean_cross_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/pair_condition_means/cross_region",
        ),
        within_input_filename=cfg.get("pair_condition_mean_within_output_filename", "pair_condition_means.pkl"),
        cross_input_filename=cfg.get("pair_condition_mean_cross_output_filename", "pair_condition_means.pkl"),
        signal_input_column=cfg.get("signal_input_column", "spike_train_counts"),
        signal_input_columns=cfg.get("signal_input_columns"),
        output_subdir=cfg.get(
            "pair_condition_mean_plot_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/pair_condition_mean_plots",
        ),
        output_extension=cfg.get("pair_condition_mean_plot_output_extension", "pdf"),
        output_dpi=cfg.get("pair_condition_mean_plot_output_dpi", 220),
        figsize=cfg.get("pair_condition_mean_plot_figsize"),
        condition_order=cfg.get(
            "plot_condition_order",
            ("face_interactive", "face_non_interactive", "object"),
        ),
        condition_labels=cfg.get(
            "plot_condition_labels",
            {
                "face_interactive": "Face (interactive)",
                "face_non_interactive": "Face (non-interactive)",
                "object": "Object",
            },
        ),
        condition_short_labels=cfg.get(
            "pair_condition_mean_plot_condition_short_labels",
            {
                "face_interactive": "FI",
                "face_non_interactive": "FNI",
                "object": "OBJ",
            },
        ),
        condition_colors=cfg.get(
            "plot_condition_colors",
            {
                "face_interactive": "#d62728",
                "face_non_interactive": "#1f77b4",
                "object": "#2ca02c",
            },
        ),
        significance_alpha=cfg.get(
            "pair_condition_mean_plot_significance_alpha",
            cfg.get("plot_significance_alpha", 0.05),
        ),
        mean_lag_significance_correction=cfg.get(
            "pair_condition_mean_plot_mean_lag_significance_correction",
            cfg.get("plot_significance_correction", "bonferroni"),
        ),
        per_lag_significance_correction=cfg.get(
            "pair_condition_mean_plot_per_lag_significance_correction",
            cfg.get("plot_significance_correction", "bonferroni"),
        ),
        min_pairs_for_significance=cfg.get("pair_condition_mean_plot_min_pairs_for_significance", 3),
        mean_trace_linewidth=cfg.get(
            "pair_condition_mean_plot_mean_trace_linewidth",
            cfg.get("plot_mean_trace_linewidth", 2.2),
        ),
        sem_alpha=cfg.get("pair_condition_mean_plot_sem_alpha", 0.12),
        between_condition_marker_size=cfg.get(
            "pair_condition_mean_plot_between_condition_marker_size",
            cfg.get("plot_between_condition_marker_size", 5.0),
        ),
        between_condition_marker_alpha=cfg.get(
            "pair_condition_mean_plot_between_condition_marker_alpha",
            cfg.get("plot_between_condition_marker_alpha", 0.95),
        ),
        mean_lag_annotation_fontsize=cfg.get("pair_condition_mean_plot_annotation_fontsize", 8.0),
        plot_lag_half_window_ms=cfg.get("pair_condition_mean_plot_lag_half_window_ms", 100.0),
        lag_tick_step_ms=cfg.get("pair_condition_mean_plot_lag_tick_step_ms", 50.0),
        use_parallel=cfg.get(
            "pair_condition_mean_plot_use_parallel",
            cfg.get("plot_use_parallel", True),
        ),
        max_procs=cfg.get("pair_condition_mean_plot_max_procs", cfg.get("plot_max_procs")),
        lag_test_min_lags_for_parallel=cfg.get("pair_condition_mean_plot_lag_test_min_lags_for_parallel", 256),
        lag_test_chunk_size=cfg.get("pair_condition_mean_plot_lag_test_chunk_size", 256),
    )

    if args.output_subdir is not None:
        settings.output_subdir = str(args.output_subdir)
    if args.output_extension is not None:
        settings.output_extension = str(args.output_extension)
    if args.output_dpi is not None:
        settings.output_dpi = int(args.output_dpi)
    if args.figsize is not None:
        settings.figsize = _as_float2(args.figsize)
    if args.no_parallel:
        settings.use_parallel = False

    if args.analysis_kind == "within":
        analysis_kinds = (WITHIN_ANALYSIS_KIND,)
    elif args.analysis_kind == "cross":
        analysis_kinds = (CROSS_ANALYSIS_KIND,)
    else:
        analysis_kinds = (WITHIN_ANALYSIS_KIND, CROSS_ANALYSIS_KIND)

    result = plot_fixation_neural_cross_correlation_pair_condition_mean_summaries(
        settings,
        dates=args.date,
        analysis_kinds=analysis_kinds,
        include_per_day_when_dates_unspecified=(args.date is None and not args.no_per_day),
    )

    print(
        "[plot] pair-condition means: "
        f"wrote {len(result.get('figure_outputs', []))} figure(s), "
        f"{len(result.get('mean_lag_stat_outputs', []))} mean-lag table(s)"
    )
    _print_outputs("figures", list(result.get("figure_outputs", [])))
    _print_outputs("mean-lag stats", list(result.get("mean_lag_stat_outputs", [])))

    primary_subset_label = str(result.get("subset_label") or "all_dates")
    for idx, key in enumerate(sorted(result.get("results", {}).keys())):
        _print_mean_lag_table(primary_subset_label, key, result["results"][key])
        if idx != len(result.get("results", {})) - 1:
            print()


def main() -> None:
    run_plot_cli(default_analysis_kind="both")


if __name__ == "__main__":
    main()
