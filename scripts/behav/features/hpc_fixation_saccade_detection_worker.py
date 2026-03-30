"""HPC worker script for fixation/saccade detection on a single task."""

import argparse

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.behav.features.gaze_event_detection import (
    build_gaze_event_detection_settings,
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

    detection_cfg = load_config(args.gaze_event_cfg)
    settings = build_gaze_event_detection_settings(args.dataset_cfg, detection_cfg)

    row = {"date": args.date, "session": args.session}
    process_and_save_gaze_events_for_row(settings, row, args.agent)


if __name__ == "__main__":
    main()
