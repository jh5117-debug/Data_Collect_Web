#!/usr/bin/env bash
set -euo pipefail
cd /home/hj/Data_Collect_Web
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
PYTHONPATH=finetune/src:finetune/experiments/participant_cv/src:. \
python finetune/experiments/participant_cv/scripts/run_few_shot_target.py
