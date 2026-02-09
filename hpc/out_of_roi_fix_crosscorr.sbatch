#!/bin/bash
#SBATCH -J out_of_roi_fix_crosscorr
#SBATCH -p psych_day
#SBATCH -c 32
#SBATCH --mem-per-cpu=12G
#SBATCH -t 04:00:00
#SBATCH -o hpc/out_of_roi_fix_crosscorr.out
#SBATCH -e hpc/out_of_roi_fix_crosscorr.err

set -euo pipefail

bash -lc 'module load miniconda; conda deactivate; conda activate gaze_processing; python scripts/analysis/build_out_of_roi_fix_cross_correlation.py'
