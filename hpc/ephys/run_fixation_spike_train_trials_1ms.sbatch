#!/usr/bin/env bash
#SBATCH --job-name=fixation_spike_train_trials_1ms
#SBATCH --partition=psych_day
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

if [[ ! -f "scripts/ephys/features/build_fixation_spike_train_trials_1ms.py" ]]; then
  echo "Expected repo root in SLURM_SUBMIT_DIR or current working directory, got: ${REPO_ROOT}" >&2
  exit 1
fi

LOG_DIR="${REPO_ROOT}/hpc/ephys/logs"
mkdir -p "${LOG_DIR}"
JOB_LOG_BASENAME="${SLURM_JOB_NAME:-fixation_spike_train_trials_1ms}_${SLURM_JOB_ID:-manual}"
STDOUT_LOG="${LOG_DIR}/${JOB_LOG_BASENAME}.out"
STDERR_LOG="${LOG_DIR}/${JOB_LOG_BASENAME}.err"
exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)

module load miniconda
source /gpfs/milgram/apps/avx2/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate gaze_processing

DATASET_CFG="${DATASET_CFG:-configs/dataset.yaml}"
EPHYS_CFG="${EPHYS_CFG:-configs/ephys_data.yaml}"
EPHYS_FIXATION_PSTH_CFG="${EPHYS_FIXATION_PSTH_CFG:-configs/ephys_fixation_psth.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

echo "Running 1 ms fixation spike-train trial build"
echo "repo=${REPO_ROOT}"
echo "dataset_cfg=${DATASET_CFG}"
echo "ephys_cfg=${EPHYS_CFG}"
echo "ephys_fixation_psth_cfg=${EPHYS_FIXATION_PSTH_CFG}"
echo "python=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "stdout_log=${STDOUT_LOG}"
echo "stderr_log=${STDERR_LOG}"

"${PYTHON_BIN}" scripts/ephys/features/build_fixation_spike_train_trials_1ms.py \
  --dataset-cfg "${DATASET_CFG}" \
  --ephys-cfg "${EPHYS_CFG}" \
  --ephys-fixation-psth-cfg "${EPHYS_FIXATION_PSTH_CFG}" \
  "$@"
