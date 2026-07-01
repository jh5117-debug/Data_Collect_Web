#!/usr/bin/env bash
#SBATCH --job-name=vigil-qwen-features
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=finetune/runs/slurm_qwen_features_%j.log

set -euo pipefail
cd /home/hj/Data_Collect_Web
echo "Run finetune/scripts/extract_qwen_encoder_features.py after preparing a dataset directory."
