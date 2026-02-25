"""Plot random-session raw vs smoothed pupil timecourses for QC."""

import argparse

from dal_monte_2022_analysis.behav.plotting.pupil_smoothing import (
    SmoothedPupilQCPlotSettings,
    plot_smoothed_pupil_timecourse_qc,
)


def main() -> None:
    """Parse CLI args and run smoothed-pupil QC plotting."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw and smoothed pupil traces for random sessions "
            "as a rows=sessions, cols=agents QC figure."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--analysis-subdir", default="pupil_smoothing")
    parser.add_argument(
        "--output-filename",
        default="smoothed_pupil_timecourse_qc.pdf",
    )
    parser.add_argument("--n-sessions", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--max-points-per-trace", type=int, default=30000)
    args = parser.parse_args()

    settings = SmoothedPupilQCPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        analysis_subdir=args.analysis_subdir,
        output_filename=args.output_filename,
        n_sessions=int(args.n_sessions),
        random_seed=int(args.random_seed),
        max_points_per_trace=int(args.max_points_per_trace),
    )

    out_path = plot_smoothed_pupil_timecourse_qc(settings)
    print(f"[plot] wrote smoothed pupil QC timecourse figure: {out_path}")


if __name__ == "__main__":
    main()
