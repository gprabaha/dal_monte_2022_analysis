"""Plot fixation-level neural cross-correlation summaries."""

import argparse

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_neural_cross_correlation import (
    CROSS_ANALYSIS_KIND,
    FixationNeuralCrossCorrelationPlotSettings,
    WITHIN_ANALYSIS_KIND,
    plot_fixation_neural_cross_correlation_summaries,
)


def _as_float2(values):
    if values is None:
        return None
    if len(values) != 2:
        return None
    return [float(values[0]), float(values[1])]


def run_plot_cli(*, default_analysis_kind: str = "both") -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot date-level and all-date neural cross-correlation summaries for "
            "within-region and cross-region neuron pairs."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument(
        "--ephys-fixation-neural-crosscorr-cfg",
        default="configs/ephys_fixation_neural_cross_correlation.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--date-extension", default=None)
    parser.add_argument("--region-extension", default=None)
    parser.add_argument("--date-dpi", type=int, default=None)
    parser.add_argument("--region-dpi", type=int, default=None)
    parser.add_argument("--date-figsize", nargs=2, type=float, default=None)
    parser.add_argument("--region-figsize", nargs=2, type=float, default=None)
    parser.add_argument("--max-pair-traces", type=int, default=None)
    parser.add_argument("--max-points-per-pdf-trace", type=int, default=None)
    parser.add_argument("--subplot-ncols", type=int, default=None)
    parser.add_argument("--normalize-traces", action="store_true")
    parser.add_argument("--normalization-method", choices=["none", "max_abs", "zscore"], default=None)
    parser.add_argument("--max-procs", type=int, default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--no-parallel-date-plots", action="store_true")
    parser.add_argument("--no-parallel-global-plots", action="store_true")
    parser.add_argument(
        "--analysis-kind",
        choices=["both", "within", "cross"],
        default=str(default_analysis_kind),
        help="Choose which xcorr output family to plot.",
    )
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_neural_crosscorr_cfg)
    settings = FixationNeuralCrossCorrelationPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        within_input_subdir=cfg.get(
            "within_output_subdir",
            "ephys/psth/fixation_neural_crosscorr/within_region",
        ),
        cross_input_subdir=cfg.get(
            "cross_output_subdir",
            "ephys/psth/fixation_neural_crosscorr/cross_region",
        ),
        within_input_filename=cfg.get("within_output_filename", "fixations.pkl"),
        cross_input_filename=cfg.get("cross_output_filename", "fixations.pkl"),
        within_pair_average_input_filename=cfg.get(
            "within_pair_average_output_filename",
            "pair_averages.pkl",
        ),
        cross_pair_average_input_filename=cfg.get(
            "cross_pair_average_output_filename",
            "pair_averages.pkl",
        ),
        output_subdir=cfg.get(
            "plot_output_subdir",
            "ephys/psth/fixation_neural_crosscorr/plots",
        ),
        date_output_extension=cfg.get("plot_date_output_extension", "png"),
        region_output_extension=cfg.get("plot_region_output_extension", "pdf"),
        date_output_dpi=cfg.get("plot_date_output_dpi", 220),
        region_output_dpi=cfg.get("plot_region_output_dpi", 220),
        date_figsize=cfg.get("plot_date_figsize"),
        region_figsize=cfg.get("plot_region_figsize"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
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
        condition_colors=cfg.get(
            "plot_condition_colors",
            {
                "face_interactive": "#d62728",
                "face_non_interactive": "#1f77b4",
                "object": "#2ca02c",
            },
        ),
        pair_trace_alpha=cfg.get("plot_pair_trace_alpha", 0.12),
        pair_trace_linewidth=cfg.get("plot_pair_trace_linewidth", 0.75),
        mean_trace_linewidth=cfg.get("plot_mean_trace_linewidth", 2.2),
        max_pair_traces_per_plot=cfg.get("plot_max_pair_traces_per_plot"),
        max_points_per_pdf_trace=cfg.get("plot_max_points_per_pdf_trace"),
        normalize_traces=cfg.get("plot_normalize_traces", False),
        normalization_method=cfg.get("plot_normalization_method", "max_abs"),
        subplot_ncols=cfg.get("plot_subplot_ncols", 3),
        use_parallel=cfg.get("plot_use_parallel", True),
        max_procs=cfg.get("plot_max_procs", 16),
        parallelize_date_plots=cfg.get("plot_parallelize_date_plots", True),
        parallelize_global_plots=cfg.get("plot_parallelize_global_plots", True),
        random_seed=cfg.get("plot_random_seed", 13),
        test_single=cfg.get("test_single", False),
    )

    if args.output_subdir is not None:
        settings.output_subdir = str(args.output_subdir)
    if args.date_extension is not None:
        settings.date_output_extension = str(args.date_extension)
    if args.region_extension is not None:
        settings.region_output_extension = str(args.region_extension)
    if args.date_dpi is not None:
        settings.date_output_dpi = int(args.date_dpi)
    if args.region_dpi is not None:
        settings.region_output_dpi = int(args.region_dpi)
    if args.date_figsize is not None:
        settings.date_figsize = _as_float2(args.date_figsize)
    if args.region_figsize is not None:
        settings.region_figsize = _as_float2(args.region_figsize)
    if args.max_pair_traces is not None:
        settings.max_pair_traces_per_plot = int(args.max_pair_traces)
    if args.max_points_per_pdf_trace is not None:
        settings.max_points_per_pdf_trace = int(args.max_points_per_pdf_trace)
    if args.subplot_ncols is not None:
        settings.subplot_ncols = max(1, int(args.subplot_ncols))
    if args.normalize_traces:
        settings.normalize_traces = True
    if args.normalization_method is not None:
        settings.normalization_method = str(args.normalization_method)
    if args.max_procs is not None:
        settings.max_procs = int(args.max_procs)
    if args.no_parallel:
        settings.use_parallel = False
    if args.no_parallel_date_plots:
        settings.parallelize_date_plots = False
    if args.no_parallel_global_plots:
        settings.parallelize_global_plots = False
    if args.test_single:
        settings.test_single = True

    if args.analysis_kind == "within":
        analysis_kinds = (WITHIN_ANALYSIS_KIND,)
    elif args.analysis_kind == "cross":
        analysis_kinds = (CROSS_ANALYSIS_KIND,)
    else:
        analysis_kinds = (WITHIN_ANALYSIS_KIND, CROSS_ANALYSIS_KIND)

    result = plot_fixation_neural_cross_correlation_summaries(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
        analysis_kinds=analysis_kinds,
    )

    date_outputs = result.get("date_outputs", [])
    global_outputs = result.get("global_outputs", [])
    within_counts = result.get("within_counts", {}) or {}
    cross_counts = result.get("cross_counts", {}) or {}
    print(
        "[plot] "
        f"analysis_kind={args.analysis_kind}: "
        f"wrote {len(date_outputs)} date-level plot(s) and {len(global_outputs)} global plot(s)"
    )
    if within_counts:
        print(
            "[plot] within aggregation: "
            f"files={within_counts.get('files', 0)}, "
            f"pair_avg_files={within_counts.get('files_using_pair_averages', 0)}, "
            f"rows={within_counts.get('rows', 0)}, "
            f"used={within_counts.get('used_rows', 0)}, "
            f"skipped={within_counts.get('skipped_rows', 0)}"
        )
    if cross_counts:
        print(
            "[plot] cross aggregation: "
            f"files={cross_counts.get('files', 0)}, "
            f"pair_avg_files={cross_counts.get('files_using_pair_averages', 0)}, "
            f"rows={cross_counts.get('rows', 0)}, "
            f"used={cross_counts.get('used_rows', 0)}, "
            f"skipped={cross_counts.get('skipped_rows', 0)}"
        )
    if date_outputs:
        print(f"[plot] first date-level output: {date_outputs[0]}")
    if global_outputs:
        print(f"[plot] first global output: {global_outputs[0]}")


def main() -> None:
    run_plot_cli(default_analysis_kind="both")


if __name__ == "__main__":
    main()
