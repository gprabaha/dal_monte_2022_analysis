#!/usr/bin/env bash
#SBATCH --job-name=fixation_detection_driver
#SBATCH --partition=psych_day
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=05:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate gaze_processing

DATASET_CFG="${DATASET_CFG:-configs/dataset.yaml}"
GAZE_EVENT_CFG="${GAZE_EVENT_CFG:-configs/gaze_event_detection.yaml}"
HPC_CFG="${HPC_CFG:-configs/hpc_gaze_event_detection.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

echo "Running fixation/saccade detection"
echo "repo=${REPO_ROOT}"
echo "dataset_cfg=${DATASET_CFG}"
echo "gaze_event_cfg=${GAZE_EVENT_CFG}"
echo "hpc_cfg=${HPC_CFG}"
echo "python=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"

"${PYTHON_BIN}" scripts/behav/features/detect_fixations_and_saccades.py \
  --dataset-cfg "${DATASET_CFG}" \
  --gaze-event-cfg "${GAZE_EVENT_CFG}" \
  --hpc-cfg "${HPC_CFG}" \
  "$@"
