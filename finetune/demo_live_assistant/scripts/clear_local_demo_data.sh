#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hj/Data_Collect_Web/finetune/demo_live_assistant/local_data"
case "${ROOT}" in
  */finetune/demo_live_assistant/local_data) ;;
  *)
    echo "Refusing to clear unsafe path: ${ROOT}" >&2
    exit 3
    ;;
esac

rm -rf "${ROOT}"
mkdir -p "${ROOT}"
echo "cleared ${ROOT}"
