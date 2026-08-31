#!/usr/bin/env bash
# Array job over recording dates. Each task builds every session for one date,
# parallelising across sessions inside the task.
#
#   sbatch --array=0-41 hpc/ephys/run_fixation_pair_spike_coordination.sbatch
#
# Size the array to the number of dates: the build prints the date list with
#   python scripts/ephys/analysis/build_fixation_pair_spike_coordination.py --help
# and the dates are the `date=*` folders under the processed data root.
#SBATCH --job-name=fixation_pair_spike_coordination
#SBATCH --partition=psych_day
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

BUILD_SCRIPT="scripts/ephys/analysis/build_fixation_pair_spike_coordination.py"
if [[ ! -f "${BUILD_SCRIPT}" ]]; then
  echo "Expected repo root in SLURM_SUBMIT_DIR or current working directory, got: ${REPO_ROOT}" >&2
  exit 1
fi

LOG_DIR="${REPO_ROOT}/hpc/ephys/logs"
mkdir -p "${LOG_DIR}"
JOB_LOG_BASENAME="${SLURM_JOB_NAME:-fixation_pair_spike_coordination}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-0}"
STDOUT_LOG="${LOG_DIR}/${JOB_LOG_BASENAME}.out"
STDERR_LOG="${LOG_DIR}/${JOB_LOG_BASENAME}.err"
exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)

module load miniconda
source /gpfs/milgram/apps/avx2/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate gaze_processing

DATASET_CFG="${DATASET_CFG:-configs/dataset.yaml}"
COORDINATION_CFG="${COORDINATION_CFG:-configs/ephys_fixation_pair_spike_coordination.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

# Each FFT is already multi-threaded by BLAS; the build parallelises across
# sessions, so pinning the inner threads to 1 avoids oversubscribing the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROCESSED_ROOT="$("${PYTHON_BIN}" -c "
from dal_monte_2022_analysis.config.load import load_config
print(load_config('${DATASET_CFG}')['processed_data_root'])
")"

mapfile -t DATES < <(find "${PROCESSED_ROOT}" -maxdepth 1 -type d -name 'date=*' -printf '%f\n' \
  | sed 's/^date=//' | sort)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#DATES[@]} )); then
  echo "Array task ${TASK_ID} exceeds the ${#DATES[@]} available dates; nothing to do."
  exit 0
fi
DATE="${DATES[${TASK_ID}]}"

echo "Running fixation pair spike-coordination build"
echo "repo=${REPO_ROOT}"
echo "date=${DATE}  (task ${TASK_ID} of ${#DATES[@]} dates)"
echo "dataset_cfg=${DATASET_CFG}"
echo "coordination_cfg=${COORDINATION_CFG}"
echo "cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "stdout_log=${STDOUT_LOG}"

"${PYTHON_BIN}" "${BUILD_SCRIPT}" \
  --dataset-cfg "${DATASET_CFG}" \
  --coordination-cfg "${COORDINATION_CFG}" \
  --date "${DATE}" \
  --max-procs "${SLURM_CPUS_PER_TASK:-8}" \
  "$@"
