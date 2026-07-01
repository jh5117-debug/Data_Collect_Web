# VIGIL Latest-Data Experiments

This package records the reproducible workflow for the June 26, 2026 latest VIGIL export.

The workflow keeps Qwen ASR transcript text separate from keyword-spotting labels:

- `qwen_asr/*.jsonl` contains only the audio path and the official ASR training text.
- `keyword_spotting/*.jsonl` contains labels, prompt groups, phrase IDs, participant aliases, and split metadata.

Generated ZIPs, audio, private aliases, manifests, logs, runs, checkpoints, transcript caches, predictions, and feature caches are ignored by Git.

## Main Commands

```bash
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
PYTHONPATH=finetune/src:finetune/experiments/latest_data/src:. \
python finetune/experiments/latest_data/scripts/download_latest_export.py

PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
PYTHONPATH=finetune/src:finetune/experiments/latest_data/src:. \
python finetune/experiments/latest_data/scripts/prepare_latest_dataset.py \
  --zip-path finetune/data/vigil_dataset_export_latest_<UTC_TIMESTAMP>.zip
```
