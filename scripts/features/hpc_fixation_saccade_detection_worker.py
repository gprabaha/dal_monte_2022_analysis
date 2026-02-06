"""HPC worker script for fixation/saccade detection on a single task."""

import argparse

from dal_monte_2022_analysis.config.load import load_gaze_event_config
from dal_monte_2022_analysis.features.gaze_event_detection import (
    GazeEventDetectionSettings,
    process_and_save_gaze_events_for_row,
)


def main():
    """Parse CLI args and run gaze event detection for one session/agent."""
    parser = argparse.ArgumentParser(
        description="Fixation/saccade detection worker.",
    )
    parser.add_argument("--dataset-cfg", required=True)
    parser.add_argument("--gaze-event-cfg", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--agent", required=True)
    args = parser.parse_args()

    detection_cfg = load_gaze_event_config(args.gaze_event_cfg)
    settings = GazeEventDetectionSettings(
        cfg_path=args.dataset_cfg,
        input_modality=detection_cfg.get("input_modality", "gaze_position"),
        output_fixations_modality=detection_cfg.get("output_fixations_modality", "fixations"),
        output_saccades_modality=detection_cfg.get("output_saccades_modality", "saccades"),
        use_parallel=detection_cfg.get("use_parallel", True),
        test_single=False,
        agents=detection_cfg.get("agents"),
    )

    row = {"date": args.date, "session": args.session}
    process_and_save_gaze_events_for_row(settings, row, args.agent)


if __name__ == "__main__":
    main()
