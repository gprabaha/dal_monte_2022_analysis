"""Parallelism helpers for runtime orchestration."""

import multiprocessing as mp
import os
from typing import Optional


def get_n_processes(max_procs: Optional[int] = 16) -> int:
    """Return a safe worker count based on SLURM or local CPU availability.

    Args:
        max_procs: Optional upper bound on workers to launch. ``None`` keeps
            the full detected worker count.

    Returns:
        The number of processes to use (at least 1, capped by ``max_procs``
        when provided).
    """
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            n = int(slurm_cpus)
        except ValueError:
            n = 1
    else:
        n = mp.cpu_count()

    n = max(1, int(n))
    if max_procs is None:
        return n
    return max(1, min(n, int(max_procs)))
