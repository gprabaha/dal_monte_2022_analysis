# hpc

Runtime artifacts for HPC submissions.

This directory is used by dSQ/SLURM submission workflows configured in:
- `configs/hpc_gaze_event_detection.yaml`
- `configs/hpc_fix_cross_correlation_shuffle.yaml`
- `configs/hpc_out_of_roi_fix_cross_correlation_shuffle.yaml`

Typical contents:
- generated job command files (`*.txt`)
- generated sbatch scripts (`*.sh`)
- per-task stdout/stderr logs
- dSQ status files

Notes:
- Files here are usually generated, not hand-edited.
- Safe to clean between runs if you no longer need logs.
