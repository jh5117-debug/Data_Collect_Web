#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash finetune/demo/run_demo.sh <physical_gpu_index> <run_dir>" >&2
  exit 2
fi

GPU_INDEX="$1"
RUN_DIR="$2"
PROJECT_ROOT="/home/hj/Data_Collect_Web"
ENV_BIN="/home/hj/miniconda/envs/vigil-two-stage/bin"
LOG_DIR="$PROJECT_ROOT/finetune/demo/logs"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/vigil_demo_${TIMESTAMP}_gpu${GPU_INDEX}.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 3
fi
if [[ ! -x "$ENV_BIN/python" ]]; then
  echo "Missing Python environment: $ENV_BIN/python" >&2
  exit 4
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 5
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_INDEX" | head -1)"
if [[ "$GPU_NAME" != *"RTX 3090"* ]]; then
  echo "Selected GPU $GPU_INDEX is not an RTX 3090: $GPU_NAME" >&2
  exit 6
fi

export PATH="$ENV_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJECT_ROOT/finetune/demo:$PROJECT_ROOT/finetune/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PROJECT_ROOT/finetune/cache/huggingface"
export HF_HUB_CACHE="$PROJECT_ROOT/finetune/cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/finetune/cache/huggingface/transformers"
export TORCH_HOME="$PROJECT_ROOT/finetune/cache/torch"

"$ENV_BIN/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing CPU fallback")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
print("visible CUDA device:", name)
if "RTX 3090" not in name:
    raise SystemExit(f"visible CUDA device is not an RTX 3090: {name}")
PY

for path in \
  "$RUN_DIR/stage1/checkpoint_best.pt" \
  "$RUN_DIR/stage1/threshold.json" \
  "$RUN_DIR/stage2_bce/checkpoint_best.pt" \
  "$RUN_DIR/stage2_bce/threshold.json" \
  "$RUN_DIR/stage2_bce_supcon/checkpoint_best.pt" \
  "$RUN_DIR/stage2_bce_supcon/threshold.json" \
  "$RUN_DIR/model_selection.json"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required demo artifact: $path" >&2
    exit 7
  fi
done

"$ENV_BIN/python" finetune/demo/app.py \
  --run-dir "$RUN_DIR" \
  --host 127.0.0.1 \
  --port 7860 2>&1 | tee "$LOG_PATH"

