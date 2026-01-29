"""Run fixation and saccade detection locally or via HPC job arrays."""

import argparse
import logging
from pathlib import Path

from dal_monte_2022_analysis.features.fixation_detection import (
    FixationDetectionSettings,
    detect_and_save_for_row,
    run_detection,
)
from dal_monte_2022_analysis.io.index_dataset import index_dataset
from dal_monte_2022_analysis.config.load import load_dataset_config
from dal_monte_2022_analysis.utils.hpc_utils import (
    generate_fixation_job_file,
    submit_dsq_array_job,
    track_job_completion,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _pick_first_task(settings: FixationDetectionSettings, agent: str):
    cfg = load_dataset_config(settings.cfg_path)
    index_df = index_dataset(cfg, "gaze_position")
    if index_df.empty:
        return None
    row = index_df.iloc[0].to_dict()
    return row, agent


def main():
    parser = argparse.ArgumentParser(description="Detect fixations and saccades.")
    parser.add_argument("--cfg", default="configs/dataset.yaml")
    parser.add_argument("--input-modality", default="gaze_position")
    parser.add_argument("--output-fixations-modality", default="fixations")
    parser.add_argument("--output-saccades-modality", default="saccades")
    parser.add_argument("--mode", choices=["single", "batch", "generate-jobs", "submit-jobs"], default="single")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")

    # HPC options
    parser.add_argument("--env-name", default="base")
    parser.add_argument("--job-name", default="fixation_detection")
    parser.add_argument("--job-file", default="local_data/fixation_jobs.txt")
    parser.add_argument("--sbatch-script", default="local_data/fixation_jobs.sbatch")
    parser.add_argument("--log-dir", default="local_data/fixation_logs")
    parser.add_argument("--partition", default="general")
    parser.add_argument("--cpus-per-task", type=int, default=2)
    parser.add_argument("--mem-per-cpu", default="4G")
    parser.add_argument("--time-limit", default="02:00:00")

    args = parser.parse_args()

    settings = FixationDetectionSettings(
        cfg_path=args.cfg,
        input_modality=args.input_modality,
        output_fixations_modality=args.output_fixations_modality,
        output_saccades_modality=args.output_saccades_modality,
    )

    if args.mode == "single":
        cfg = load_dataset_config(settings.cfg_path)
        if args.date and args.session and args.agent:
            row = {"date": args.date, "session": args.session}
            detect_and_save_for_row(settings, row, args.agent)
            return
        if args.agent is None:
            args.agent = cfg["agents"][0]
        task = _pick_first_task(settings, args.agent)
        if task is None:
            logger.warning("No data found for single run.")
            return
        row, agent = task
        detect_and_save_for_row(settings, row, agent)
        return

    if args.mode == "batch":
        run_detection(
            settings,
            agents=None,
            use_parallel=args.use_parallel,
            test_single=args.test_single,
        )
        return

    worker_script = Path(__file__).resolve()
    if args.mode == "generate-jobs":
        cfg = load_dataset_config(settings.cfg_path)
        index_df = index_dataset(cfg, "gaze_position")
        tasks = []
        for _, row in index_df.iterrows():
            for agent in cfg["agents"]:
                tasks.append((row["date"], row["session"], agent))
        if args.test_single and tasks:
            tasks = [tasks[0]]
        generate_fixation_job_file(
            tasks=tasks,
            job_file_path=Path(args.job_file),
            worker_script=worker_script,
            env_name=args.env_name,
            cfg_path=args.cfg,
            input_modality=args.input_modality,
            output_fixations_modality=args.output_fixations_modality,
            output_saccades_modality=args.output_saccades_modality,
        )
        return

    if args.mode == "submit-jobs":
        job_id = submit_dsq_array_job(
            job_file_path=Path(args.job_file),
            sbatch_script_path=Path(args.sbatch_script),
            log_dir=Path(args.log_dir),
            job_name=args.job_name,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            time_limit=args.time_limit,
        )
        track_job_completion(job_id)
        return


if __name__ == "__main__":
    main()
