# Codex Handoff: VIGIL Shared Qwen Runtime

## Branch And Commit

- Branch: `research/vigil-shared-qwen-runtime-20260627`
- Latest commit: `6b48a06 Add shared Qwen runtime diagnostic and adapter`

## Qwen Runtime

- qwen-asr version: `0.0.6`
- Model: `Qwen/Qwen3-ASR-1.7B`
- Source file: `/home/hj/miniconda/envs/vigil-two-stage/lib/python3.12/site-packages/qwen_asr/inference/qwen3_asr.py`
- Backend: `transformers`

## Stage 2 Checkpoint And Config

- Bundle: `finetune/model_bundles/vigil_latest_optimized_20260626_085405`
- Variant: `stage2_bce_supcon`
- Stage 2 threshold: `0.9877771735191345`
- Checkpoint exists: `True`

## Shared-Qwen Status

- Status: `blocked_by_runtime_interface`
- Blocker: The current public qwen_asr wrapper does not expose decoder-compatible audio hidden states and does not accept externally supplied audio hidden states for decoding.
- Success proof available: `False`

## Parity And Non-Regression

- Transcript parity: `blocked`
- Stage 2 score parity: `blocked`
- LibriSpeech check: `not_run_because_shared_path_blocked`
- VIGIL trigger metric parity: `not_run_because_shared_path_blocked`

## Cost

- Extra Qwen encoder median cost remains `13.663365971297026` ms per Stage 1 candidate.

## GPU Assignment

- Real diagnostic used one idle local RTX 3090 through `CUDA_VISIBLE_DEVICES`.

## Tmux Sessions

- None required for this branch.

## Artifact Paths

- Reports: `finetune/experiments/shared_qwen_runtime/reports`
- Final report: `finetune/experiments/shared_qwen_runtime/reports/FINAL_SHARED_QWEN_RUNTIME_REPORT.md`
- Deep inspection JSON: `finetune/experiments/shared_qwen_runtime/reports/qwen_runtime_deep_inspection.json`
- Call counter JSON: `finetune/experiments/shared_qwen_runtime/reports/call_counter_diagnostic.json`

## Exact Next Command

```bash
cd /home/hj/Data_Collect_Web && PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH PYTHONPATH=finetune/src:finetune/experiments/shared_qwen_runtime/src:. pytest -q finetune/experiments/shared_qwen_runtime/tests
```

## Push Status

- Commit `6b48a06` pushed to `origin/research/vigil-shared-qwen-runtime-20260627`.
