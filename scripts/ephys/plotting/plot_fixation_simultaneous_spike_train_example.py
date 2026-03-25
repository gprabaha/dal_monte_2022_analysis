"""Plot one fixation's 1 ms spike trains for two simultaneously recorded regions."""

import argparse

import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_simultaneous_spike_train_example import (
    FixationSimultaneousSpikeTrainExamplePlotSettings,
    plot_fixation_simultaneous_spike_train_example,
)


def _as_float2(values):
    if values is None:
        return None
    if len(values) != 2:
        return None
    return [float(values[0]), float(values[1])]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 1 ms fixation-aligned spike trains for two simultaneously recorded regions.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument(
        "--ephys-fixation-neural-cross-correlation-cfg",
        "--ephys-fixation-neural-crosscorr-cfg",
        dest="ephys_fixation_neural_cross_correlation_cfg",
        default="configs/ephys_fixation_neural_cross_correlation.yaml",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--region-a", default=None)
    parser.add_argument("--region-b", default=None)
    parser.add_argument("--fixation-category", default=None)
    parser.add_argument("--interactive-state", default=None)
    parser.add_argument("--fixation-start-idx", type=int, default=None)
    parser.add_argument("--fixation-stop-idx", type=int, default=None)
    parser.add_argument("--fixation-rank", type=int, default=None)
    parser.add_argument("--min-units-per-region", type=int, default=None)
    parser.add_argument("--max-units-per-region", type=int, default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--output-extension", default=None)
    parser.add_argument("--output-dpi", type=int, default=None)
    parser.add_argument("--figsize", nargs=2, type=float, default=None)
    parser.add_argument("--time-window-ms", nargs=2, type=float, default=None)
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_neural_cross_correlation_cfg)
    settings = FixationSimultaneousSpikeTrainExamplePlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_modality=cfg.get("simultaneous_spike_train_example_input_modality", "psth"),
        input_filename=cfg.get("simultaneous_spike_train_example_input_filename", "fixations_spike_train_1ms.pkl"),
        output_subdir=cfg.get(
            "simultaneous_spike_train_example_output_subdir",
            "ephys/psth/fixation_neural_cross_correlation/simultaneous_spike_train_examples",
        ),
        output_extension=cfg.get("simultaneous_spike_train_example_output_extension", "pdf"),
        output_dpi=cfg.get("simultaneous_spike_train_example_output_dpi", 300),
        figsize=cfg.get("simultaneous_spike_train_example_figsize"),
        time_window_ms=cfg.get("simultaneous_spike_train_example_time_window_ms", [-500.0, 500.0]),
        date=str(cfg.get("simultaneous_spike_train_example_date", "01312018")),
        session=str(cfg.get("simultaneous_spike_train_example_session", "10")),
        region_a=cfg.get("simultaneous_spike_train_example_region_a", "bla"),
        region_b=cfg.get("simultaneous_spike_train_example_region_b", "accg"),
        fixation_category=cfg.get("simultaneous_spike_train_example_fixation_category", "face"),
        interactive_state=cfg.get("simultaneous_spike_train_example_interactive_state", "non_interactive"),
        fixation_start_idx=cfg.get("simultaneous_spike_train_example_fixation_start_idx"),
        fixation_stop_idx=cfg.get("simultaneous_spike_train_example_fixation_stop_idx"),
        fixation_rank=cfg.get("simultaneous_spike_train_example_fixation_rank", 1),
        min_units_per_region=cfg.get("simultaneous_spike_train_example_min_units_per_region", 10),
        max_units_per_region=cfg.get("simultaneous_spike_train_example_max_units_per_region"),
        line_width=cfg.get("simultaneous_spike_train_example_linewidth", 0.45),
        line_length=cfg.get("simultaneous_spike_train_example_linelength", 0.82),
        title_fontsize=cfg.get("simultaneous_spike_train_example_title_fontsize", 6.6),
        label_fontsize=cfg.get("simultaneous_spike_train_example_label_fontsize", 6.0),
        tick_fontsize=cfg.get("simultaneous_spike_train_example_tick_fontsize", 5.4),
        panel_wspace=cfg.get("simultaneous_spike_train_example_panel_wspace", 0.28),
    )

    if args.date is not None:
        settings.date = str(args.date)
    if args.session is not None:
        settings.session = str(args.session)
    if args.region_a is not None:
        settings.region_a = str(args.region_a)
    if args.region_b is not None:
        settings.region_b = str(args.region_b)
    if args.fixation_category is not None:
        settings.fixation_category = str(args.fixation_category)
    if args.interactive_state is not None:
        settings.interactive_state = str(args.interactive_state)
    if args.fixation_start_idx is not None:
        settings.fixation_start_idx = int(args.fixation_start_idx)
    if args.fixation_stop_idx is not None:
        settings.fixation_stop_idx = int(args.fixation_stop_idx)
    if args.fixation_rank is not None:
        settings.fixation_rank = max(1, int(args.fixation_rank))
    if args.min_units_per_region is not None:
        settings.min_units_per_region = max(1, int(args.min_units_per_region))
    if args.max_units_per_region is not None:
        settings.max_units_per_region = None if int(args.max_units_per_region) <= 0 else int(args.max_units_per_region)
    if args.output_subdir is not None:
        settings.output_subdir = str(args.output_subdir)
    if args.output_extension is not None:
        settings.output_extension = str(args.output_extension)
    if args.output_dpi is not None:
        settings.output_dpi = int(args.output_dpi)
    if args.figsize is not None:
        settings.figsize = _as_float2(args.figsize)
    if args.time_window_ms is not None:
        settings.time_window_ms = _as_float2(args.time_window_ms)

    selection_override_requested = any(
        value is not None
        for value in (
            args.date,
            args.session,
            args.region_a,
            args.region_b,
            args.fixation_category,
            args.interactive_state,
            args.fixation_rank,
            args.min_units_per_region,
        )
    )
    if selection_override_requested and args.fixation_start_idx is None and args.fixation_stop_idx is None:
        settings.fixation_start_idx = None
        settings.fixation_stop_idx = None

    result = plot_fixation_simultaneous_spike_train_example(settings)
    selected = result["selected_fixation"]
    print(f"[plot] output: {result['output_path']}")
    print(
        "[plot] selected fixation: "
        f"date={settings.date}, session={settings.session}, "
        f"regions={result['region_a']} vs {result['region_b']}, "
        f"category={selected.get('fixation_category')}, "
        f"interactive_state={selected.get('interactive_state')}, "
        f"start={selected.get('fixation_start_idx')}, stop={selected.get('fixation_stop_idx')}, "
        f"plotted_units={result['region_a_plotted_units']} + {result['region_b_plotted_units']}"
    )

    if args.list_candidates:
        candidates = result.get("candidate_fixations")
        candidates = candidates if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
        if candidates.empty:
            print("[plot] candidate fixation table: empty")
        else:
            display_cols = [
                col
                for col in (
                    "fixation_rank",
                    "fixation_category",
                    "interactive_state",
                    "fixation_start_idx",
                    "fixation_stop_idx",
                    "region_a_units",
                    "region_b_units",
                    "total_spikes",
                )
                if col in candidates.columns
            ]
            display_df = candidates.loc[:, display_cols].head(max(1, int(args.candidate_limit)))
            print("[plot] top fixation candidates:")
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 2000):
                print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
