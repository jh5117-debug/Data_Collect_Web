#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash run_latest_nested_all.sh <dataset_dir> <run_source_dir> <stage1_manifest> <qwen_manifest>" >&2
  exit 2
fi

DATASET_DIR="$1"
RUN_SOURCE_DIR="$2"
STAGE1_MANIFEST="$3"
QWEN_MANIFEST="$4"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PATH="/home/hj/miniconda/envs/vigil-two-stage/bin:${PATH}"
export PYTHONPATH="${ROOT}/finetune/src:${ROOT}/finetune/experiments/latest_data/src:${ROOT}/finetune/experiments/participant_cv/src:${ROOT}:${PYTHONPATH:-}"

for fold in 0 1 2 3 4; do
  python "${ROOT}/finetune/experiments/latest_data/scripts/run_latest_nested_fold.py" \
    --outer-fold "${fold}" \
    --dataset-dir "${DATASET_DIR}" \
    --run-source-dir "${RUN_SOURCE_DIR}" \
    --stage1-manifest "${STAGE1_MANIFEST}" \
    --qwen-manifest "${QWEN_MANIFEST}"
done
