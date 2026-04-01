#!/usr/bin/env bash
#SBATCH --job-name=plot_fixation_psth_units
#SBATCH --partition=psych_day
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

if [[ ! -f "scripts/ephys/plotting/plot_fixation_psth_units.py" ]]; then
  echo "Expected repo root in SLURM_SUBMIT_DIR or current working directory, got: ${REPO_ROOT}" >&2
  exit 1
fi

LOG_DIR="${REPO_ROOT}/hpc/ephys/logs"
mkdir -p "${LOG_DIR}"
JOB_LOG_BASENAME="${SLURM_JOB_NAME:-plot_fixation_psth_units}"
STDOUT_LOG="${LOG_DIR}/${JOB_LOG_BASENAME}.out"
STDERR_LOG="${LOG_DIR}/${JOB_LOG_BASENAME}.err"
exec > >(tee "${STDOUT_LOG}") 2> >(tee "${STDERR_LOG}" >&2)

module load miniconda
source /gpfs/milgram/apps/avx2/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate gaze_processing

DATASET_CFG="${DATASET_CFG:-configs/dataset.yaml}"
PLOTTING_CFG="${PLOTTING_CFG:-configs/plotting.yaml}"
EPHYS_FIXATION_PSTH_CFG="${EPHYS_FIXATION_PSTH_CFG:-configs/ephys_fixation_psth.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

echo "Running fixation PSTH unit plotting"
echo "repo=${REPO_ROOT}"
echo "dataset_cfg=${DATASET_CFG}"
echo "plotting_cfg=${PLOTTING_CFG}"
echo "ephys_fixation_psth_cfg=${EPHYS_FIXATION_PSTH_CFG}"
echo "python=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "stdout_log=${STDOUT_LOG}"
echo "stderr_log=${STDERR_LOG}"

"${PYTHON_BIN}" scripts/ephys/plotting/plot_fixation_psth_units.py \
  --dataset-cfg "${DATASET_CFG}" \
  --plotting-cfg "${PLOTTING_CFG}" \
  --ephys-fixation-psth-cfg "${EPHYS_FIXATION_PSTH_CFG}" \
  "$@"
