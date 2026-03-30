# hpc

Runtime artifacts for HPC submissions.

This directory is used by dSQ/SLURM submission workflows configured in:
- `configs/hpc_gaze_event_detection.yaml`
- `configs/hpc_face_fix_cross_correlation_shuffle.yaml`
- `configs/hpc_out_of_roi_fix_cross_correlation_shuffle.yaml`

Typical contents:
- generated job command files (`*.txt`)
- generated sbatch scripts (`*.sh`)
- hand-authored launchers grouped by domain (`behav/*.sbatch`, etc.)
- per-task stdout/stderr logs
- dSQ status files

Notes:
- Files here are usually generated, not hand-edited.
- Safe to clean between runs if you no longer need logs.
- Domain-specific log folders can be tracked when needed, for example
  `hpc/behav/logs/` for direct behavioral `sbatch` entrypoints.
