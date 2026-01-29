"""HPC helpers for job array submission."""

import subprocess
import time
import logging
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)


def write_job_file(job_file_path: Path, commands: Iterable[str]) -> None:
    """Write one command per line to a job file."""
    job_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(job_file_path, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")
    logger.info("Wrote job file to %s", job_file_path)


def generate_fixation_job_file(
    *,
    tasks: Iterable[tuple],
    job_file_path: Path,
    worker_script: Path,
    env_name: str,
    cfg_path: str,
    input_modality: str,
    output_fixations_modality: str,
    output_saccades_modality: str,
) -> None:
    """Generate a job file for fixation detection array jobs."""
    commands: List[str] = []
    for date, session, agent in tasks:
        cmd = (
            "module load miniconda; "
            f"conda activate {env_name}; "
            "python "
            f"{worker_script} "
            "--mode single "
            f"--cfg {cfg_path} "
            f"--date {date} "
            f"--session {session} "
            f"--agent {agent} "
            f"--input-modality {input_modality} "
            f"--output-fixations-modality {output_fixations_modality} "
            f"--output-saccades-modality {output_saccades_modality}"
        )
        commands.append(cmd)

    write_job_file(job_file_path, commands)


def submit_dsq_array_job(
    *,
    job_file_path: Path,
    sbatch_script_path: Path,
    log_dir: Path,
    job_name: str,
    partition: str,
    cpus_per_task: int,
    mem_per_cpu: str,
    time_limit: str,
) -> str:
    """Submit a dSQ array job and return the job ID."""
    log_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        f"module load dSQ; dsq --job-file {job_file_path} --batch-file {sbatch_script_path} "
        f"-o {log_dir} --status-dir {log_dir} --partition {partition} "
        f"--cpus-per-task {cpus_per_task} --mem-per-cpu {mem_per_cpu} "
        f"-t {time_limit} --mail-type FAIL",
        shell=True,
        check=True,
        executable="/bin/bash",
    )

    if not sbatch_script_path.exists():
        raise RuntimeError(f"dSQ job script was not created: {sbatch_script_path}")

    result = subprocess.run(
        f"sbatch --job-name=dsq_{job_name} "
        f"--output={log_dir}/{job_name}_%a.out "
        f"--error={log_dir}/{job_name}_%a.err "
        f"{sbatch_script_path}",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        executable="/bin/bash",
    )

    job_id = result.stdout.strip().split()[-1]
    logger.info("Submitted job array with ID: %s", job_id)
    return job_id


def track_job_completion(job_id: str, poll_secs: int = 30, log_every_secs: int = 60) -> None:
    """Track job array status using squeue."""
    logger.info("Tracking job array with ID: %s", job_id)
    start = time.time()
    last_log = start

    while True:
        result = subprocess.run(
            f"squeue --job {job_id} -h -o %T",
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash",
        )

        if result.returncode != 0:
            logger.error("Failed to check job status: %s", result.stderr.strip())
            break

        statuses = result.stdout.strip().split()
        if not statuses or all(s not in {"PENDING", "RUNNING", "CONFIGURING"} for s in statuses):
            logger.info("Job array %s has completed.", job_id)
            break

        if time.time() - last_log >= log_every_secs:
            logger.info("Still running... job array %s", job_id)
            last_log = time.time()

        time.sleep(poll_secs)
