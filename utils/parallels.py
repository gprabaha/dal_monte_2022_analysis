import os
import multiprocessing as mp


def get_n_processes(max_procs: int = 8) -> int:
    """
    Determine number of worker processes.
    Uses SLURM_CPUS_PER_TASK if available, otherwise local CPU count.
    Caps at max_procs.
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
