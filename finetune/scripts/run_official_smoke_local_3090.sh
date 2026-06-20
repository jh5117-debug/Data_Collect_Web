#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/hj/Data_Collect_Web"
DEFAULT_ZIP="${ROOT}/finetune/data/vigil_dataset_export_20260620_020617.zip"
FINGERPRINT="235c8ad292faeeac"
CONFIG="${ROOT}/finetune/configs/official_smoke_3090.yaml"

usage() {
  cat <<'EOF'
Usage:
  bash finetune/scripts/run_official_smoke_local_3090.sh <physical_gpu_index> [dataset_zip]

Example:
  bash finetune/scripts/run_official_smoke_local_3090.sh 3

This script must be run from a normal HAL SSH shell, not from the Codex
sandbox. It intentionally fails instead of falling back to CPU, FFT features,
dummy encoders, Slurm, or more than one GPU.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

GPU_INDEX="$1"
DATA_ZIP="${2:-$DEFAULT_ZIP}"

if ! [[ "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: physical_gpu_index must be an integer, got: ${GPU_INDEX}" >&2
  exit 2
fi

cd "${ROOT}"
mkdir -p finetune/logs
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${ROOT}/finetune/logs/official_smoke_3090_${TIMESTAMP}_gpu${GPU_INDEX}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "official_smoke_start_utc=${TIMESTAMP}"
echo "root=${ROOT}"
echo "gpu_index=${GPU_INDEX}"
echo "data_zip=${DATA_ZIP}"
echo "log_path=${LOG_PATH}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: strict config not found: ${CONFIG}" >&2
  exit 2
fi

if [[ ! -f "${DATA_ZIP}" ]]; then
  echo "ERROR: dataset ZIP not found: ${DATA_ZIP}" >&2
  exit 2
fi

NVIDIA_SMI="$(command -v nvidia-smi || true)"
if [[ -z "${NVIDIA_SMI}" && -x /usr/bin/nvidia-smi ]]; then
  NVIDIA_SMI="/usr/bin/nvidia-smi"
fi
if [[ -z "${NVIDIA_SMI}" ]]; then
  echo "ERROR: nvidia-smi is not available in PATH or /usr/bin." >&2
  exit 2
fi

"${NVIDIA_SMI}" -L
GPU_NAME="$("${NVIDIA_SMI}" --query-gpu=name --format=csv,noheader,nounits -i "${GPU_INDEX}" | head -n 1 | xargs || true)"
if [[ -z "${GPU_NAME}" ]]; then
  echo "ERROR: selected GPU ${GPU_INDEX} is not visible to nvidia-smi." >&2
  exit 2
fi
if [[ "${GPU_NAME}" != *"RTX 3090"* ]]; then
  echo "ERROR: selected GPU ${GPU_INDEX} is not an RTX 3090: ${GPU_NAME}" >&2
  exit 2
fi
echo "selected_gpu_name=${GPU_NAME}"

"${NVIDIA_SMI}" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -i "${GPU_INDEX}"
"${NVIDIA_SMI}" pmon -c 1 || true

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${ROOT}/finetune/src:${PYTHONPATH:-}"

python - <<'PY'
import torch
from vigil_two_stage.strict_runtime import validate_one_visible_cuda_device

print("torch_version=", torch.__version__)
print("torch_cuda_version=", torch.version.cuda)
name = validate_one_visible_cuda_device(torch)
print("visible_cuda_device=", name)
print("visible_cuda_memory_gb=", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

python -m vigil_two_stage.strict_runtime \
  --config "${CONFIG}" \
  --require-openwakeword \
  --require-qwen-asr

DATASET_DIR="${ROOT}/finetune/data/processed/${FINGERPRINT}"
if [[ ! -f "${DATASET_DIR}/manifest_all.jsonl" ]]; then
  echo "processed dataset is missing; preparing from ZIP"
  python "${ROOT}/finetune/scripts/prepare_dataset.py" \
    --zip "${DATA_ZIP}" \
    --config "${CONFIG}" \
    --out-root "${ROOT}/finetune/data/processed"
fi

if [[ ! -f "${DATASET_DIR}/manifest_all.jsonl" ]]; then
  echo "ERROR: processed dataset manifest still missing: ${DATASET_DIR}/manifest_all.jsonl" >&2
  exit 2
fi

RUN_DIR="${ROOT}/finetune/runs/${TIMESTAMP}_${FINGERPRINT}_official_oww_3090_smoke"
mkdir -p "${RUN_DIR}/dataset"
cp "${CONFIG}" "${RUN_DIR}/config_resolved.yaml"
cp "${DATASET_DIR}/dataset_report.json" "${RUN_DIR}/dataset/dataset_report.json"
cp "${DATASET_DIR}/dataset_report.md" "${RUN_DIR}/dataset/dataset_report.md" 2>/dev/null || true

echo "run_dir=${RUN_DIR}"

STAGE1_DIR="$(python "${ROOT}/finetune/scripts/extract_openwakeword_features.py" \
  --dataset-dir "${DATASET_DIR}" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" | tail -n 1)"

python "${ROOT}/finetune/scripts/train_stage1.py" \
  --features-manifest "${STAGE1_DIR}/features_manifest.jsonl" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}"

python "${ROOT}/finetune/scripts/run_qwen_text_baseline.py" \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}" \
  --model-name "Qwen/Qwen3-ASR-1.7B"

python "${ROOT}/finetune/scripts/extract_qwen_encoder_features.py" \
  --dataset-dir "${DATASET_DIR}" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}"

python "${ROOT}/finetune/scripts/train_stage2.py" \
  --dataset-dir "${DATASET_DIR}" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --variant bce

python "${ROOT}/finetune/scripts/train_stage2.py" \
  --dataset-dir "${DATASET_DIR}" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --variant bce_supcon

python "${ROOT}/finetune/scripts/evaluate_cascade.py" \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}"

echo "official_smoke_completed_utc=$(date -u +%Y%m%d_%H%M%S)"
echo "run_dir=${RUN_DIR}"
echo "log_path=${LOG_PATH}"
