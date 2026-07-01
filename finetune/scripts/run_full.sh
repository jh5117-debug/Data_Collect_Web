#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash finetune/scripts/run_full.sh /absolute/path/to/vigil_dataset_export.zip" >&2
  exit 2
fi

DATA_ZIP="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT}/finetune/configs/full.yaml"
export PYTHONPATH="${ROOT}/finetune/src:${PYTHONPATH:-}"

python "${ROOT}/finetune/scripts/inspect_export.py" "${DATA_ZIP}" --report-dir "${ROOT}/finetune/reports"
DATASET_DIR="$(python "${ROOT}/finetune/scripts/prepare_dataset.py" "${DATA_ZIP}" --config "${CONFIG}" --output-root "${ROOT}/finetune/data/processed" | tail -n 1)"
FINGERPRINT="$(basename "${DATASET_DIR}")"
RUN_DIR="${ROOT}/finetune/runs/$(date -u +%Y%m%d_%H%M%S)_${FINGERPRINT}_full"
mkdir -p "${RUN_DIR}/dataset"
cp "${CONFIG}" "${RUN_DIR}/config_resolved.yaml"
cp "${DATASET_DIR}/dataset_report.json" "${RUN_DIR}/dataset/dataset_report.json"
cp "${DATASET_DIR}/dataset_report.md" "${RUN_DIR}/dataset/dataset_report.md"

python "${ROOT}/finetune/scripts/run_qwen_text_baseline.py" --dataset-dir "${DATASET_DIR}" --run-dir "${RUN_DIR}"
STAGE1_DIR="$(python "${ROOT}/finetune/scripts/extract_openwakeword_features.py" --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --run-dir "${RUN_DIR}" | tail -n 1)"
python "${ROOT}/finetune/scripts/train_stage1.py" --features-manifest "${STAGE1_DIR}/features_manifest.jsonl" --config "${CONFIG}" --run-dir "${RUN_DIR}"
python "${ROOT}/finetune/scripts/extract_qwen_encoder_features.py" --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --run-dir "${RUN_DIR}"
python "${ROOT}/finetune/scripts/train_stage2.py" --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --run-dir "${RUN_DIR}" --variant bce
python "${ROOT}/finetune/scripts/train_stage2.py" --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --run-dir "${RUN_DIR}" --variant bce_supcon
python "${ROOT}/finetune/scripts/evaluate_cascade.py" --dataset-dir "${DATASET_DIR}" --run-dir "${RUN_DIR}"

echo "${RUN_DIR}"
