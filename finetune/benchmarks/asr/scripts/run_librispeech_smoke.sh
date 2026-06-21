#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hj/Data_Collect_Web"
BENCH_ROOT="$PROJECT_ROOT/finetune/benchmarks/asr"
ENV_BIN="/home/hj/miniconda/envs/vigil-two-stage/bin"
PY="$ENV_BIN/python"
GPU_INDEX="${1:-}"
MODEL_NAME="${2:-Qwen/Qwen3-ASR-1.7B}"
RUN_NAME="${3:-qwen3_asr_1_7b_smoke}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="$BENCH_ROOT/logs"
SAFE_RUN_NAME="$(echo "$RUN_NAME" | tr '/: ' '___' | tr -cd 'A-Za-z0-9_.-')"
LOG_PATH="$LOG_DIR/librispeech_smoke_${TIMESTAMP}_gpu${GPU_INDEX:-unset}.log"
RUN_DIR="$BENCH_ROOT/runs/${TIMESTAMP}_${SAFE_RUN_NAME}_smoke"
SANITY_DIR="$BENCH_ROOT/runs/${TIMESTAMP}_${SAFE_RUN_NAME}_sanity"
mkdir -p "$LOG_DIR"

usage() {
  echo "Usage: bash finetune/benchmarks/asr/scripts/run_librispeech_smoke.sh <physical_gpu_index> [model_name] [run_name]" >&2
}

main() {
  if [[ -z "$GPU_INDEX" ]]; then
    usage
    exit 2
  fi
  cd "$PROJECT_ROOT"
  local cache_root="$PROJECT_ROOT/finetune/cache"
  mkdir -p "$LOG_DIR" "$cache_root/huggingface/hub" "$cache_root/huggingface/transformers" "$cache_root/torch"

  if [[ ! -x "$PY" ]]; then
    echo "Missing benchmark Python environment: $PY" >&2
    exit 3
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required in the normal HAL SSH shell" >&2
    exit 4
  fi
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_INDEX" | head -1)"
  if [[ "$gpu_name" != *"RTX 3090"* ]]; then
    echo "Selected GPU $GPU_INDEX is not an RTX 3090: $gpu_name" >&2
    exit 5
  fi

  export PATH="$ENV_BIN:$PATH"
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  export PYTHONPATH="$BENCH_ROOT/src:$PROJECT_ROOT/finetune/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HOME="$cache_root/huggingface"
  export HF_HUB_CACHE="$cache_root/huggingface/hub"
  export TRANSFORMERS_CACHE="$cache_root/huggingface/transformers"
  export TORCH_HOME="$cache_root/torch"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  bash "$BENCH_ROOT/scripts/bootstrap_asr_benchmark.sh"

  "$PY" - <<'PY'
import importlib.util
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing CPU fallback")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
print("visible CUDA device:", name)
if "RTX 3090" not in name:
    raise SystemExit(f"visible CUDA device is not an RTX 3090: {name}")
if importlib.util.find_spec("qwen_asr") is None:
    raise SystemExit("qwen_asr is required")
PY

  bash "$BENCH_ROOT/scripts/download_librispeech_eval.sh"
  PYTHONPATH="$BENCH_ROOT/src:$PROJECT_ROOT/finetune/src:$PROJECT_ROOT" "$PY" -m pytest -q "$BENCH_ROOT/tests"
  "$PY" -m compileall -q "$BENCH_ROOT/src" "$BENCH_ROOT/scripts"
  bash -n "$BENCH_ROOT/scripts/"*.sh

  local baseline_flag=()
  if [[ "$MODEL_NAME" == "Qwen/Qwen3-ASR-1.7B" ]]; then
    baseline_flag=(--require-baseline-model)
  fi

  "$PY" "$BENCH_ROOT/scripts/run_qwen_librispeech.py" \
    --manifest "$BENCH_ROOT/manifests/smoke_all.jsonl" \
    --output-dir "$SANITY_DIR" \
    --run-name "${RUN_NAME}_sanity" \
    --model "$MODEL_NAME" \
    --limit 1 \
    --resume \
    "${baseline_flag[@]}"

  "$PY" "$BENCH_ROOT/scripts/run_qwen_librispeech.py" \
    --manifest "$BENCH_ROOT/manifests/smoke_all.jsonl" \
    --output-dir "$RUN_DIR" \
    --run-name "$RUN_NAME" \
    --model "$MODEL_NAME" \
    --resume \
    "${baseline_flag[@]}"

  "$PY" "$BENCH_ROOT/scripts/run_qwen_librispeech.py" \
    --manifest "$BENCH_ROOT/manifests/smoke_all.jsonl" \
    --output-dir "$RUN_DIR" \
    --run-name "$RUN_NAME" \
    --model "$MODEL_NAME" \
    --resume \
    "${baseline_flag[@]}"

  echo "SMOKE_RUN_DIR=$RUN_DIR"
  echo "LOG_PATH=$LOG_PATH"
}

main "$@" 2>&1 | tee "$LOG_PATH"
exit "${PIPESTATUS[0]}"
