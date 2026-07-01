#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${1:-}"
RUN_DIR="${2:-/home/hj/Data_Collect_Web/finetune/model_bundles/vigil_latest_optimized_20260626_085405}"
PORT="${PORT:-7861}"
HOST="127.0.0.1"

if [[ -z "${GPU_INDEX}" ]]; then
  echo "usage: bash finetune/demo_live_assistant/scripts/run_demo.sh <GPU_INDEX> [RUN_DIR]" >&2
  exit 2
fi

cd /home/hj/Data_Collect_Web
export PATH="/home/hj/miniconda/envs/vigil-two-stage/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="finetune/src:finetune/demo:finetune/demo_live_assistant:."

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; refusing CPU fallback")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
if "RTX 3090" not in name:
    raise SystemExit(f"visible GPU is not RTX 3090: {name}")
print(f"visible_gpu={name}")
PY

echo "VIGIL local HAL browser assistant demo"
echo "URL: http://${HOST}:${PORT}"
echo "SSH tunnel: ssh -L ${PORT}:127.0.0.1:${PORT} hal"

exec python finetune/demo_live_assistant/app.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --run-dir "${RUN_DIR}"
