#!/bin/bash
#SBATCH -J out_of_roi_fix_crosscorr_shuffle
#SBATCH -p psych_day
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -t 04:00:00
#SBATCH -o hpc/out_of_roi_fix_crosscorr_shuffle_submit.out
#SBATCH -e hpc/out_of_roi_fix_crosscorr_shuffle_submit.err

set -euo pipefail

bash -lc 'module load miniconda; conda init; conda deactivate; conda activate gaze_processing; python scripts/analysis/build_out_of_roi_fix_cross_correlation.py --mode shuffle_submit_hpc --hpc-cfg configs/hpc_out_of_roi_fix_cross_correlation_shuffle.yaml'
