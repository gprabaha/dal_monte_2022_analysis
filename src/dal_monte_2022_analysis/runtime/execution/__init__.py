"""Execution/runtime concurrency helpers."""

from .parallel import get_n_processes
from .task_runner import run_tasks

__all__ = ["get_n_processes", "run_tasks"]
