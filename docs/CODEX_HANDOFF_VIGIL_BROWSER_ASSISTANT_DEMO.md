# Codex Handoff: VIGIL Browser Assistant Demo

## Branch And Commit

- Branch: `research/vigil-browser-assistant-demo-20260627`
- Latest commit: `1dcbba1 Add local VIGIL browser assistant demo`

## Demo URL

- Local: `http://127.0.0.1:7861`
- SSH tunnel: `ssh -L 7861:127.0.0.1:7861 hal`

## Model Loading Status

- Real mode loaded successfully.
- `/health` returned mode `real`.
- Loaded flags: openWakeWord `true`, Stage 1 head `true`, Qwen ASR `true`, Stage 2 head `true`.
- Default run dir: `/home/hj/Data_Collect_Web/finetune/model_bundles/vigil_latest_optimized_20260626_085405`

## Checkpoint Paths

- Stage 1: `finetune/model_bundles/vigil_latest_optimized_20260626_085405/stage1/checkpoint_best.pt`
- Stage 2: `finetune/model_bundles/vigil_latest_optimized_20260626_085405/stage2_bce_supcon/checkpoint_best.pt`

## GPU Assignment

- Physical GPU 6 selected after verification.
- `CUDA_VISIBLE_DEVICES=6`.
- Demo process uses one visible `NVIDIA GeForce RTX 3090`.

## Screens Implemented

- Local profile.
- Onboarding recordings.
- Calibration result.
- Assistant listening.

## Onboarding Status

- Local profile and clip upload implemented.
- Calibration requires at least 3 accepted positive VIGIL clips.
- Bounded positive-bias demo calibration implemented.

## Assistant Listening Status

- Start/stop/reset implemented.
- HTTP chunk upload implemented.
- Cooldown implemented.
- LLM / VQA response intentionally not implemented.

## Transcript Status

- Real mode uses frozen Qwen through existing runtime.
- Mock/partial mode clearly reports non-real status in `/health`.

## VIGIL Trigger Status

- Real mode uses the current two-stage runtime when load succeeds.
- Stage 2 may use an extra Qwen encoder forward for candidates.

## Tests

- `PYTHONPATH=finetune/src:finetune/demo_live_assistant:. pytest -q finetune/demo_live_assistant/tests`: `11 passed`.
- `PYTHONPATH=finetune/src:. pytest -q finetune/tests`: `56 passed`.
- `PYTHONPATH=finetune/benchmarks/asr/src:finetune/src:. pytest -q finetune/benchmarks/asr/tests`: `17 passed`.
- `python -m compileall -q finetune/src finetune/demo_live_assistant`: passed.
- `bash -n finetune/demo_live_assistant/scripts/*.sh`: passed.

## Validation

- Tmux session: `vigil_browser_assistant_demo`.
- Log: `finetune/demo_live_assistant/logs/demo_20260628_034213_gpu6.log`.
- `GET /health`: passed, real mode.
- `GET /`: browser app HTML served.
- Profile creation: passed.
- Onboarding upload: 3 accepted positive support clips uploaded.
- Calibration: `ok`, support count `3`, method `bounded_positive_bias_demo`, bias `1.0`.
- Positive VIGIL chunk: trigger detected `true`, assistant state `ASSISTANT_STATE`, Stage 1 score `0.9995500445365906`, Stage 2 score `0.9879599809646606`.
- Negative chunk: trigger detected `false`, assistant state `LISTENING`, Stage 1 score `0.005684383679181337`, Stage 2 score `null`.
- Browser microphone capture was not human-tested.

## Exact Next Command

```bash
ssh -L 7861:127.0.0.1:7861 hal
```

## Push Status

- Commit `1dcbba1` pushed to `origin/research/vigil-browser-assistant-demo-20260627`.
