#!/bin/bash
#SBATCH --output /gpfs/milgram/pi/chang/pg496/repositories/dal_monte_2022_analysis/configs/hpc/gaze_event_logs
#SBATCH --array 0
#SBATCH --job-name dsq-gaze_event_jobs
#SBATCH --partition psych_day --cpus-per-task 8 --mem-per-cpu 2G -t 02:00:00 --mail-type FAIL

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/dal_monte_2022_analysis/configs/hpc/gaze_event_jobs.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/dal_monte_2022_analysis/configs/hpc/gaze_event_logs

