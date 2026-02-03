"""Parallelism helpers for batch processing."""

import os
import multiprocessing as mp


def get_n_processes(max_procs: int = 16) -> int:
    """Return a safe worker count based on SLURM or local CPU availability.

    Args:
        max_procs: Upper bound on workers to launch.

    Returns:
        The number of processes to use (at least 1, at most max_procs).
    """
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            n = int(slurm_cpus)
        except ValueError:
            n = 1
    else:
        n = mp.cpu_count()

    return max(1, min(n, max_procs))
