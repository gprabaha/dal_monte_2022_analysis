"""Shared task-runner utilities for pipeline stages."""

from __future__ import annotations

from multiprocessing import Pool
from typing import Callable, Iterable, TypeVar

from tqdm import tqdm

from .parallel import get_n_processes


T = TypeVar("T")
R = TypeVar("R")


def run_tasks(
    worker_fn: Callable[[T], R],
    tasks: Iterable[T],
    *,
    desc: str,
    unit: str = "task",
    use_parallel: bool = True,
    max_procs: int = 8,
) -> list[R]:
    """Run tasks in serial or with multiprocessing while showing progress."""
    task_list = list(tasks)
    if not task_list:
        return []

    if not use_parallel or len(task_list) == 1:
        return [
            worker_fn(task)
            for task in tqdm(task_list, desc=f"{desc} (serial)", unit=unit)
        ]

    n_proc = get_n_processes(max_procs=max_procs)
    with Pool(processes=n_proc) as pool:
        return list(
            tqdm(
                pool.imap_unordered(worker_fn, task_list),
                total=len(task_list),
                desc=f"{desc} ({n_proc} workers)",
                unit=unit,
            )
        )
