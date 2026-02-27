"""Plot per-session interactive-period detection timelines and density traces."""

import argparse

from dal_monte_2022_analysis.behav.plotting.interactive_period_detection import (
    InteractivePeriodDetectionPlotSettings,
    plot_interactive_period_detection,
)


def main():
    """Parse CLI args and generate one interactive-period-detection plot per session."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-session fixation timelines and joint-density interactive-period "
            "detection traces."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--interactive-periods-cfg", default="configs/interactive_periods.yaml")
    parser.add_argument("--analysis-subdir", default="interactive_periods")
    parser.add_argument("--output-subdir", default="period_detection")
    parser.add_argument("--output-extension", default="png")
    parser.add_argument(
        "--figure-width-in",
        type=float,
        default=4.25,
        help="Figure width in inches (default is about half of letter width).",
    )
    parser.add_argument(
        "--figure-height-in",
        type=float,
        default=5.25,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--example-session",
        action="append",
        default=[],
        help=(
            "Session key to export to example subfolder as both PNG and PDF. "
            "Repeatable. Format: date|session"
        ),
    )
    parser.add_argument("--example-sessions-subfolder", default="example_sessions")
    parser.add_argument(
        "--example-pdf-rasterized",
        action="store_true",
        help="Rasterize traces/blocks in example PDFs (disable for Illustrator editing).",
    )
    parser.add_argument(
        "--max-density-points",
        type=int,
        default=3000,
        help=(
            "Maximum points per density trace after strided downsampling "
            "(controls vector anchor-point count)."
        ),
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Enable x/y grid lines (off by default).",
    )
    parser.add_argument(
        "--no-rasterize-density-traces",
        action="store_true",
        help="Keep density traces fully vectorized in PDF output (larger files).",
    )
    parser.add_argument(
        "--no-rasterize-interactive-blocks",
        action="store_true",
        help="Keep interactive-period background blocks fully vectorized.",
    )
    parser.add_argument(
        "--pdf-compression",
        type=int,
        default=6,
        help="PDF compression level (0-9). Higher gives smaller files.",
    )
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    settings = InteractivePeriodDetectionPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        interactive_periods_cfg_path=args.interactive_periods_cfg,
        analysis_subdir=args.analysis_subdir,
        output_subdir=args.output_subdir,
        output_extension=args.output_extension,
        example_sessions_subfolder=str(args.example_sessions_subfolder),
        example_session_keys=tuple(str(key) for key in args.example_session),
        example_pdf_force_vector=not bool(args.example_pdf_rasterized),
        figure_width_in=float(args.figure_width_in),
        figure_height_in=float(args.figure_height_in),
        session_parallel=not bool(args.no_parallel),
        test_single=bool(args.test_single),
        max_density_points=int(args.max_density_points),
        show_grid=bool(args.show_grid),
        rasterize_density_traces=not bool(args.no_rasterize_density_traces),
        rasterize_interactive_blocks=not bool(args.no_rasterize_interactive_blocks),
        pdf_compression=int(args.pdf_compression),
    )

    out_paths = plot_interactive_period_detection(settings)
    if settings.test_single:
        if out_paths:
            print(f"[plot] wrote test interactive-period-detection figure: {out_paths[0]}")
        else:
            print("[plot] no test interactive-period-detection figure was produced")
        return

    print(f"[plot] wrote {len(out_paths)} interactive-period-detection figure(s)")
    if out_paths:
        print(f"[plot] first output: {out_paths[0]}")


if __name__ == "__main__":
    main()
