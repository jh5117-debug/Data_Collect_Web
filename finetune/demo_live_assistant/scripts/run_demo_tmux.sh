#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${1:-}"
RUN_DIR="${2:-/home/hj/Data_Collect_Web/finetune/model_bundles/vigil_latest_optimized_20260626_085405}"
if [[ -z "${GPU_INDEX}" ]]; then
  echo "usage: bash finetune/demo_live_assistant/scripts/run_demo_tmux.sh <GPU_INDEX> [RUN_DIR]" >&2
  exit 2
fi

cd /home/hj/Data_Collect_Web
mkdir -p finetune/demo_live_assistant/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="finetune/demo_live_assistant/logs/demo_${STAMP}_gpu${GPU_INDEX}.log"
SESSION="vigil_browser_assistant_demo"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new -d -s "${SESSION}" \
  "cd /home/hj/Data_Collect_Web && bash finetune/demo_live_assistant/scripts/run_demo.sh '${GPU_INDEX}' '${RUN_DIR}' > '${LOG}' 2>&1"

echo "tmux session: ${SESSION}"
echo "log: ${LOG}"
echo "URL: http://127.0.0.1:7861"
echo "SSH tunnel: ssh -L 7861:127.0.0.1:7861 hal"
