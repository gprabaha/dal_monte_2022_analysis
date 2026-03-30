#!/usr/bin/env bash
#SBATCH --job-name=face_fix_crosscorr_all_scopes
#SBATCH --partition=psych_day
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/behav/logs/%x_%j.out
#SBATCH --error=hpc/behav/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

if [[ ! -f "scripts/behav/analysis/build_face_fix_cross_correlation.py" ]]; then
  echo "Expected repo root in SLURM_SUBMIT_DIR or current working directory, got: ${REPO_ROOT}" >&2
  exit 1
fi

module load miniconda
source /gpfs/milgram/apps/avx2/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate gaze_processing

DATASET_CFG="${DATASET_CFG:-configs/dataset.yaml}"
FACE_FIX_CROSS_CORRELATION_CFG="${FACE_FIX_CROSS_CORRELATION_CFG:-configs/face_fix_cross_correlation.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

SCOPES=(whole interactive non_interactive)
MODES=(observed shuffle_submit_hpc)

#SCOPES=(non_interactive)
MODES=(observed)

echo "Running face fixation cross-correlation for all scopes/modes"
echo "repo=${REPO_ROOT}"
echo "dataset_cfg=${DATASET_CFG}"
echo "face_fix_cross_correlation_cfg=${FACE_FIX_CROSS_CORRELATION_CFG}"
echo "python=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"

for scope in "${SCOPES[@]}"; do
  for mode in "${MODES[@]}"; do
    echo ""
    echo ">>> scope=${scope} mode=${mode}"
    "${PYTHON_BIN}" scripts/behav/analysis/build_face_fix_cross_correlation.py \
      --dataset-cfg "${DATASET_CFG}" \
      --face-fix-cross-correlation-cfg "${FACE_FIX_CROSS_CORRELATION_CFG}" \
      --mode "${mode}" \
      --time-scope "${scope}"
  done
done

echo ""
echo "Completed face fixation cross-correlation for whole/interactive/non_interactive."
