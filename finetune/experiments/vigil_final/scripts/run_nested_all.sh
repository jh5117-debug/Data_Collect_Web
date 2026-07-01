#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hj/Data_Collect_Web"
PY="${PYTHON:-/home/hj/miniconda/envs/vigil-two-stage/bin/python}"
export PATH="/home/hj/miniconda/envs/vigil-two-stage/bin:${PATH}"
export PYTHONPATH="finetune/src:finetune/experiments/vigil_final/src:."

cd "${ROOT}"
for spec in "0:1" "1:2" "2:3" "3:4" "4:6"; do
  fold="${spec%%:*}"
  gpu="${spec##*:}"
  tmux new -d -s "vigil_nested_outer_${fold}" \
    "cd ${ROOT} && PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:\$PATH PYTHONPATH=finetune/src:finetune/experiments/vigil_final/src:. CUDA_VISIBLE_DEVICES=${gpu} CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PY} finetune/experiments/vigil_final/scripts/run_nested_outer_fold.py --outer-fold ${fold} --gpu ${gpu}"
done
