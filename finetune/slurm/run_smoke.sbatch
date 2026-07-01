#!/usr/bin/env bash
#SBATCH --job-name=vigil-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=finetune/runs/slurm_smoke_%j.log

set -euo pipefail
cd /home/hj/Data_Collect_Web
bash finetune/scripts/run_smoke.sh /home/hj/Data_Collect_Web/finetune/data/vigil_dataset_export_20260620_020617.zip
