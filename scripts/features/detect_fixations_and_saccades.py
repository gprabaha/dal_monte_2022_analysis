"""Run gaze event detection locally or via HPC job arrays."""

import argparse
import logging
from pathlib import Path

from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_gaze_event_config,
    load_hpc_config,
)
from dal_monte_2022_analysis.features.gaze_event_detection import (
    GazeEventDetectionSettings,
    detect_and_save_gaze_events_for_row,
    run_gaze_event_detection,
)
from dal_monte_2022_analysis.io.index_dataset import index_processed_dataset
from dal_monte_2022_analysis.utils.hpc_utils import (
    generate_gaze_event_job_file,
    submit_dsq_array_job,
    track_job_completion,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def _pick_random_task(settings: GazeEventDetectionSettings):
    cfg = load_dataset_config(settings.cfg_path)
    index_df = index_processed_dataset(cfg, settings.input_modality)
    index_df = index_df[index_df["agent"].notna()]
    if index_df.empty:
        return None
    # pick a random row
    row = index_df.sample(n=1).iloc[0].to_dict()
    print(f"Picked task: {row}")
    return row, row["agent"]


def main():
    parser = argparse.ArgumentParser(description="Detect fixations and saccades.")
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--gaze-event-cfg", default="configs/gaze_event_detection.yaml")
    parser.add_argument("--hpc-cfg", default="configs/hpc_gaze_event_detection.yaml")
    parser.add_argument("--run-hpc", action="store_true")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--agent", default=None)
    # When session/date/agent are provided, this script runs a single task.

    args = parser.parse_args()

    detection_cfg = load_gaze_event_config(args.gaze_event_cfg)
    hpc_cfg = load_hpc_config(args.hpc_cfg)

    settings = GazeEventDetectionSettings(
        cfg_path=args.dataset_cfg,
        input_modality=detection_cfg.get("input_modality", "gaze_position"),
        output_fixations_modality=detection_cfg.get("output_fixations_modality", "fixations"),
        output_saccades_modality=detection_cfg.get("output_saccades_modality", "saccades"),
        use_parallel=detection_cfg.get("use_parallel", False),
        test_single=detection_cfg.get("test_single", False),
        agents=detection_cfg.get("agents"),
    )

    if args.date and args.session and args.agent:
        row = {"date": args.date, "session": args.session}
        detect_and_save_gaze_events_for_row(settings, row, args.agent)
        return

    if args.run_hpc:
        cfg = load_dataset_config(settings.cfg_path)
        index_df = index_processed_dataset(cfg, settings.input_modality)
        tasks = []
        for _, row in index_df.iterrows():
            if row.get("agent") is None:
                continue
            tasks.append((row["date"], row["session"], row["agent"]))
        if settings.test_single and tasks:
            tasks = [tasks[0]]
        worker_script = Path(__file__).resolve()
        generate_gaze_event_job_file(
            tasks=tasks,
            job_file_path=hpc_cfg["job_file_path"],
            worker_script=worker_script,
            env_name=hpc_cfg["env_name"],
            dataset_cfg_path=args.dataset_cfg,
            gaze_event_cfg_path=args.gaze_event_cfg,
        )
        job_id = submit_dsq_array_job(
            job_file_path=hpc_cfg["job_file_path"],
            sbatch_script_path=hpc_cfg["sbatch_script_path"],
            log_dir=hpc_cfg["log_dir"],
            job_name=hpc_cfg["job_name"],
            partition=hpc_cfg["partition"],
            cpus_per_task=hpc_cfg["cpus_per_task"],
            mem_per_cpu=hpc_cfg["mem_per_cpu"],
            time_limit=hpc_cfg["time_limit"],
        )
        track_job_completion(job_id)
        return

    if settings.test_single:
        task = _pick_random_task(settings)
        if task is None:
            logger.warning("No data found for single run.")
            return
        row, agent = task
        detect_and_save_gaze_events_for_row(settings, row, agent)
        return

    run_gaze_event_detection(
        settings,
        use_parallel=settings.use_parallel,
        test_single=settings.test_single,
    )


if __name__ == "__main__":
    main()
