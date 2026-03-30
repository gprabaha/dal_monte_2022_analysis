#!/usr/bin/env bash
#SBATCH --job-name=fixation_density_builder
#SBATCH --partition=psych_day
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

if [[ ! -f "scripts/behav/features/build_fixation_density.py" ]]; then
  echo "Expected repo root in SLURM_SUBMIT_DIR or current working directory, got: ${REPO_ROOT}" >&2
  exit 1
fi

LOG_DIR="${REPO_ROOT}/hpc/behav/logs"
mkdir -p "${LOG_DIR}"
JOB_LOG_BASENAME="${SLURM_JOB_NAME:-fixation_density_builder}_${SLURM_JOB_ID:-manual}"
STDOUT_LOG="${LOG_DIR}/${JOB_LOG_BASENAME}.out"
STDERR_LOG="${LOG_DIR}/${JOB_LOG_BASENAME}.err"
exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)

module load miniconda
source /gpfs/milgram/apps/avx2/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate gaze_processing

DATASET_CFG="${DATASET_CFG:-configs/dataset.yaml}"
FIXATION_DENSITY_CFG="${FIXATION_DENSITY_CFG:-configs/fixation_density.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

echo "Running fixation density build"
echo "repo=${REPO_ROOT}"
echo "dataset_cfg=${DATASET_CFG}"
echo "fixation_density_cfg=${FIXATION_DENSITY_CFG}"
echo "python=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "stdout_log=${STDOUT_LOG}"
echo "stderr_log=${STDERR_LOG}"

"${PYTHON_BIN}" scripts/behav/features/build_fixation_density.py \
  --dataset-cfg "${DATASET_CFG}" \
  --fixation-density-cfg "${FIXATION_DENSITY_CFG}" \
  "$@"
