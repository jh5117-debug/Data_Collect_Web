# VIGIL False-Trigger Regression Audit

This package turns Shaw's real ROS bag false-trigger examples into a reproducible, privacy-safe regression audit.

The private ROS bag archive lives under:

```text
finetune/experiments/false_trigger_regression/private/rosbag-trigger-word.zip
```

The private folder, extracted DB3 files, generated WAVs, local manifests, runs, logs, and predictions are ignored by Git.

## Goals

- Inspect ROS bag metadata and SQLite topic/message tables without ROS 2 decoding.
- Extract audio and transcript hints from Shaw's known VIGIL rosbag CDR layout without requiring ROS 2 message imports.
- Score extracted WAVs with the current VIGIL detector when audio extraction succeeds.
- Diagnose whether false triggers look like integration/window/cache bugs or model hard-negative bias.
- Keep `go`, `joe`, `joke`, and `yo` as held-out regression cases unless the team deliberately collects new training data.

## Commands

Inspect the zip and generate sanitized reports:

```bash
PYTHONPATH=finetune/experiments/false_trigger_regression/src:. \
python finetune/experiments/false_trigger_regression/scripts/inspect_rosbag_zip.py
```

Attempt ROS 2 audio extraction:

```bash
PYTHONPATH=finetune/experiments/false_trigger_regression/src:. \
python finetune/experiments/false_trigger_regression/scripts/extract_rosbag_audio.py
```

Run score audit if `runs/<timestamp>/rosbag_cases.jsonl` exists. This loads the current VIGIL runtime and needs one visible RTX 3090:

```bash
CUDA_VISIBLE_DEVICES=<GPU_INDEX> \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=finetune/src:finetune/experiments/false_trigger_regression/src:. \
python finetune/experiments/false_trigger_regression/scripts/run_false_trigger_score_audit.py \
  --manifest finetune/experiments/false_trigger_regression/runs/<timestamp>/rosbag_cases.jsonl
```

Current audited result on Shaw's bag:

- `Go.`: rejected
- `Joe.`: final false accept
- `Joke.`: rejected
- `VIGIL.`: accepted

Feature and embedding hashes differ across all four windows, so the current evidence does not support a stale-cache or identical-window bug.

Summarize all available reports:

```bash
PYTHONPATH=finetune/experiments/false_trigger_regression/src:. \
python finetune/experiments/false_trigger_regression/scripts/summarize_false_trigger_audit.py
```
