"""Regression tests for HPC runtime helpers."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dal_monte_2022_analysis.runtime.hpc.jobs import (
    generate_fix_cross_correlation_shuffle_job_file,
    generate_gaze_event_job_file,
    submit_dsq_array_job,
    track_job_completion,
    write_job_file,
)


class TestHpcJobs(unittest.TestCase):
    """Behavioral tests for job-file and submission helpers."""

    def test_write_job_file_writes_one_command_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            job_file = Path(tmp_dir) / "jobs" / "joblist.txt"
            write_job_file(job_file, ["echo one", "echo two"])
            contents = job_file.read_text().splitlines()
            self.assertEqual(contents, ["echo one", "echo two"])

    def test_generate_gaze_event_job_file_contains_expected_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            job_file = Path(tmp_dir) / "gaze_jobs.txt"
            generate_gaze_event_job_file(
                tasks=[("01012020", "session_a", "m1")],
                job_file_path=job_file,
                worker_script=Path("/tmp/worker.py"),
                env_name="analysis-env",
                dataset_cfg_path="/tmp/dataset.yaml",
                gaze_event_cfg_path="/tmp/gaze.yaml",
            )
            line = job_file.read_text().strip()
            self.assertIn("--dataset-cfg /tmp/dataset.yaml", line)
            self.assertIn("--gaze-event-cfg /tmp/gaze.yaml", line)
            self.assertIn("--date 01012020", line)
            self.assertIn("--session session_a", line)
            self.assertIn("--agent m1", line)

    def test_generate_fix_crosscorr_job_file_includes_time_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            job_file = Path(tmp_dir) / "shuffle_jobs.txt"
            generate_fix_cross_correlation_shuffle_job_file(
                tasks=[("01012020", "session_a")],
                job_file_path=job_file,
                worker_script=Path("/tmp/worker.py"),
                env_name="analysis-env",
                dataset_cfg_path="/tmp/dataset.yaml",
                fix_cross_correlation_cfg_path="/tmp/fix.yaml",
                time_scope="interactive",
            )
            line = job_file.read_text().strip()
            self.assertIn("--time-scope interactive", line)

    def test_submit_dsq_array_job_returns_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            job_file = tmp_path / "jobs.txt"
            sbatch_script = tmp_path / "submit.sbatch"
            log_dir = tmp_path / "logs"
            job_file.write_text("echo hi\n")
            sbatch_script.write_text("#!/bin/bash\n")

            first = subprocess.CompletedProcess(args="dsq", returncode=0, stdout="", stderr="")
            second = subprocess.CompletedProcess(
                args="sbatch",
                returncode=0,
                stdout="Submitted batch job 4242\n",
                stderr="",
            )
            with patch(
                "dal_monte_2022_analysis.runtime.hpc.jobs.subprocess.run",
                side_effect=[first, second],
            ) as run_mock:
                job_id = submit_dsq_array_job(
                    job_file_path=job_file,
                    sbatch_script_path=sbatch_script,
                    log_dir=log_dir,
                    job_name="test_job",
                    partition="short",
                    cpus_per_task=1,
                    mem_per_cpu="1G",
                    time_limit="00:05:00",
                )

            self.assertEqual(job_id, "4242")
            self.assertEqual(run_mock.call_count, 2)

    def test_track_job_completion_polls_until_done(self) -> None:
        first = subprocess.CompletedProcess(args="squeue", returncode=0, stdout="PENDING\n", stderr="")
        second = subprocess.CompletedProcess(args="squeue", returncode=0, stdout="", stderr="")
        with patch(
            "dal_monte_2022_analysis.runtime.hpc.jobs.subprocess.run",
            side_effect=[first, second],
        ) as run_mock:
            with patch("dal_monte_2022_analysis.runtime.hpc.jobs.time.sleep") as sleep_mock:
                track_job_completion("4242", poll_secs=0, log_every_secs=0)

        self.assertEqual(run_mock.call_count, 2)
        sleep_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()

