"""HPC runtime helpers for job array submission."""

import shlex
import subprocess
import time
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

CONDA_INIT_SCRIPT = Path(
    "/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/etc/profile.d/conda.sh"
)


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


def _build_conda_python_command(
    *,
    env_name: str,
    python_argv: list[str],
    working_dir: Path | None = None,
) -> str:
    """Build one shell command that activates conda env and runs python argv."""
    if python_argv and python_argv[0] == "python" and "-u" not in python_argv[1:2]:
        python_argv = ["python", "-u", *python_argv[1:]]
    segments = [
        f"source {shlex.quote(str(CONDA_INIT_SCRIPT))}",
        f"conda activate {shlex.quote(str(env_name))}",
    ]
    if working_dir is not None:
        segments.append(f"cd {shlex.quote(str(working_dir))}")
    segments.append(shlex.join([str(token) for token in python_argv]))
    return " && ".join(segments)


def _format_repo_relative_path(path: str | Path, repo_root: Path | None) -> str:
    """Render a path relative to the repo root when possible."""
    path_obj = Path(path)
    if repo_root is None:
        return str(path_obj)

    try:
        return str(path_obj.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path_obj)


def generate_gaze_event_job_file(
    *,
    tasks: Iterable[tuple],
    job_file_path: Path,
    worker_script: Path,
    env_name: str,
    dataset_cfg_path: str,
    gaze_event_cfg_path: str,
    repo_root: Path | None = None,
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
        cmd = _build_conda_python_command(
            env_name=env_name,
            working_dir=repo_root,
            python_argv=[
                "python",
                _format_repo_relative_path(worker_script, repo_root),
                "--dataset-cfg",
                _format_repo_relative_path(dataset_cfg_path, repo_root),
                "--gaze-event-cfg",
                _format_repo_relative_path(gaze_event_cfg_path, repo_root),
                "--date",
                str(date),
                "--session",
                str(session),
                "--agent",
                str(agent),
            ],
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
    warnings.warn(
        (
            "generate_fixation_job_file is deprecated; "
            "use generate_gaze_event_job_file instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    generate_gaze_event_job_file(
        tasks=tasks,
        job_file_path=job_file_path,
        worker_script=worker_script,
        env_name=env_name,
        dataset_cfg_path=cfg_path,
        gaze_event_cfg_path=cfg_path,
    )


def generate_fix_cross_correlation_shuffle_job_file(
    *,
    tasks: Iterable[tuple[str, str]],
    job_file_path: Path,
    worker_script: Path,
    env_name: str,
    dataset_cfg_path: str,
    fix_cross_correlation_cfg_path: Optional[str] = None,
    fix_crosscorr_cfg_path: Optional[str] = None,
    time_scope: Optional[str] = None,
    repo_root: Path | None = None,
) -> None:
    """Generate a job file for within-session shuffled cross-correlation pairs."""
    if (
        fix_cross_correlation_cfg_path is None
        and fix_crosscorr_cfg_path is not None
    ):
        warnings.warn(
            (
                "fix_crosscorr_cfg_path is deprecated; "
                "use fix_cross_correlation_cfg_path instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
    cfg_path = fix_cross_correlation_cfg_path or fix_crosscorr_cfg_path
    if cfg_path is None:
        raise ValueError("Expected one of fix_cross_correlation_cfg_path or fix_crosscorr_cfg_path.")
    commands: List[str] = []
    for date, session in tasks:
        python_argv = [
            "python",
            _format_repo_relative_path(worker_script, repo_root),
            "--dataset-cfg",
            _format_repo_relative_path(dataset_cfg_path, repo_root),
            "--fix-cross-correlation-cfg",
            _format_repo_relative_path(cfg_path, repo_root),
            "--date",
            str(date),
            "--session",
            str(session),
        ]
        if time_scope:
            python_argv.extend(["--time-scope", str(time_scope)])
        cmd = _build_conda_python_command(
            env_name=env_name,
            working_dir=repo_root,
            python_argv=python_argv,
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
    gres: str | None = None,
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
        gres: Optional generic resource request, e.g. "gpu:1".

    Returns:
        The SLURM job ID string.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    dsq_cmd = [
        "dsq",
        "--job-file",
        str(job_file_path),
        "--batch-file",
        str(sbatch_script_path),
        "-o",
        str(log_dir),
        "--status-dir",
        str(log_dir),
        "--partition",
        str(partition),
        "--cpus-per-task",
        str(cpus_per_task),
        "--mem-per-cpu",
        str(mem_per_cpu),
        "-t",
        str(time_limit),
        "--mail-type",
        "FAIL",
    ]
    if gres:
        dsq_cmd.extend(["--gres", str(gres)])
    subprocess.run(
        ["bash", "-lc", f"module load dSQ && {shlex.join(dsq_cmd)}"],
        check=True,
    )

    if not sbatch_script_path.exists():
        raise RuntimeError(f"dSQ job script was not created: {sbatch_script_path}")

    result = subprocess.run(
        [
            "sbatch",
            f"--job-name=dsq_{job_name}",
            f"--output={log_dir}/{job_name}_%a.out",
            f"--error={log_dir}/{job_name}_%a.err",
            str(sbatch_script_path),
        ],
        check=True,
        capture_output=True,
        text=True,
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
            ["squeue", "--job", str(job_id), "-h", "-o", "%T"],
            capture_output=True,
            text=True,
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
