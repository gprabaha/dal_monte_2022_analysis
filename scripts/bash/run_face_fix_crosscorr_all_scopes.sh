#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET_CFG="${DATASET_CFG:-configs/dataset.yaml}"
FACE_FIX_CROSSCORR_CFG="${FACE_FIX_CROSSCORR_CFG:-configs/face_fix_cross_correlation.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PYTHONPATH:-src}"

SCOPES=(whole interactive non_interactive)
#MODES=(observed shuffle_submit_hpc)

#SCOPES=(non_interactive)
MODES=(observed)

echo "Running face fixation cross-correlation for all scopes/modes"
echo "repo=${REPO_ROOT}"
echo "dataset_cfg=${DATASET_CFG}"
echo "face_fix_crosscorr_cfg=${FACE_FIX_CROSSCORR_CFG}"
echo "python=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"

for scope in "${SCOPES[@]}"; do
  for mode in "${MODES[@]}"; do
    echo ""
    echo ">>> scope=${scope} mode=${mode}"
    "${PYTHON_BIN}" scripts/analysis/build_face_fix_cross_correlation.py \
      --dataset-cfg "${DATASET_CFG}" \
      --face-fix-cross-correlation-cfg "${FACE_FIX_CROSSCORR_CFG}" \
      --mode "${mode}" \
      --time-scope "${scope}"
  done
done

echo ""
echo "Completed face fixation cross-correlation for whole/interactive/non_interactive."
