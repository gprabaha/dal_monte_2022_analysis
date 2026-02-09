"""HPC helpers for job array submission."""

import subprocess
import time
from pathlib import Path
from typing import Iterable, List


def write_job_file(job_file_path: Path, commands: Iterable[str]) -> None:
    """Write one command per line to a job file.

    Args:
        job_file_path: Destination path for the job file.
        commands: Iterable of shell commands to write.
    """
    job_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(job_file_path, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")
    print(f"Wrote job file to {job_file_path}")


def generate_gaze_event_job_file(
    *,
    tasks: Iterable[tuple],
    job_file_path: Path,
    worker_script: Path,
    env_name: str,
    dataset_cfg_path: str,
    gaze_event_cfg_path: str,
) -> None:
    """Generate a job file for gaze-event array jobs.

    Args:
        tasks: Iterable of (date, session, agent) tuples.
        job_file_path: Output path for the job file.
        worker_script: Worker script that processes one task.
        env_name: Conda environment name to activate in each job.
        dataset_cfg_path: Path to dataset config passed to the worker.
        gaze_event_cfg_path: Path to gaze-event config passed to the worker.
    """
    commands: List[str] = []
    for date, session, agent in tasks:
        cmd = (
            "module load miniconda; "
            "conda init; "
            "conda deactivate; "
            f"conda activate {env_name}; "
            "python "
            f"{worker_script} "
            f"--dataset-cfg {dataset_cfg_path} "
            f"--gaze-event-cfg {gaze_event_cfg_path} "
            f"--date {date} "
            f"--session {session} "
            f"--agent {agent}"
        )
        commands.append(cmd)

    write_job_file(job_file_path, commands)


def generate_fixation_job_file(
    *,
    tasks,
    job_file_path,
    worker_script,
    env_name,
    cfg_path,
    input_modality,
    output_fixations_modality,
    output_saccades_modality,
) -> None:
    """Deprecated: use generate_gaze_event_job_file.

    Note:
        Parameters are kept for backward compatibility; only dataset cfg path
        is forwarded to the newer gaze-event job generator.
    """
    print("WARNING: generate_fixation_job_file is deprecated; use generate_gaze_event_job_file.")
    generate_gaze_event_job_file(
        tasks=tasks,
        job_file_path=job_file_path,
        worker_script=worker_script,
        env_name=env_name,
        dataset_cfg_path=cfg_path,
        gaze_event_cfg_path=cfg_path,
    )


def generate_fix_crosscorr_shuffle_job_file(
    *,
    tasks: Iterable[tuple[str, str]],
    job_file_path: Path,
    worker_script: Path,
    env_name: str,
    dataset_cfg_path: str,
    fix_crosscorr_cfg_path: str,
) -> None:
    """Generate a job file for within-session shuffled cross-correlation pairs."""
    commands: List[str] = []
    for date, session in tasks:
        cmd = (
            "module load miniconda; "
            f"conda activate {env_name}; "
            "python "
            f"{worker_script} "
            f"--dataset-cfg {dataset_cfg_path} "
            f"--fix-crosscorr-cfg {fix_crosscorr_cfg_path} "
            f"--date {date} "
            f"--session {session}"
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
    """Submit a dSQ array job and return the job ID.

    Args:
        job_file_path: Path to the job file with one command per line.
        sbatch_script_path: Path where dSQ writes the sbatch script.
        log_dir: Directory for stdout/stderr and status logs.
        job_name: Name prefix used for the submitted job.
        partition: SLURM partition to submit to.
        cpus_per_task: CPU count requested per task.
        mem_per_cpu: Memory per CPU (e.g., "4G").
        time_limit: SLURM time limit (e.g., "02:00:00").

    Returns:
        The SLURM job ID string.
    """
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
    print(f"Submitted job array with ID: {job_id}")
    return job_id


def track_job_completion(job_id: str, poll_secs: int = 30, log_every_secs: int = 60) -> None:
    """Track job array status using squeue until completion.

    Args:
        job_id: SLURM job ID to track.
        poll_secs: Poll interval for status checks.
        log_every_secs: Interval for emitting "still running" logs.
    """
    print(f"Tracking job array with ID: {job_id}")
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
            print(f"ERROR: Failed to check job status: {result.stderr.strip()}")
            break

        statuses = result.stdout.strip().split()
        if not statuses or all(s not in {"PENDING", "RUNNING", "CONFIGURING"} for s in statuses):
            print(f"Job array {job_id} has completed.")
            break

        if time.time() - last_log >= log_every_secs:
            print(f"Still running... job array {job_id}")
            last_log = time.time()

        time.sleep(poll_secs)
