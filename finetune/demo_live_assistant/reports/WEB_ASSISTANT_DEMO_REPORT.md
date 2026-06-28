# Web Assistant Demo Report

## Status

- Demo package: `finetune/demo_live_assistant`
- URL: `http://127.0.0.1:7861`
- SSH tunnel: `ssh -L 7861:127.0.0.1:7861 hal`
- Scope: local HAL research demo only.

## Screens Implemented

- Local profile screen.
- Onboarding recording screen.
- Calibration result screen.
- Assistant listening screen.

## Backend Routes

- `GET /`
- `GET /health`
- `POST /api/profile`
- `POST /api/onboarding/clip`
- `GET /api/onboarding/clip/{clip_id}/audio`
- `DELETE /api/onboarding/clip/{clip_id}`
- `POST /api/onboarding/calibrate`
- `POST /api/assistant/start`
- `POST /api/assistant/chunk`
- `POST /api/assistant/stop`
- `POST /api/assistant/reset`
- `GET /api/assistant/session/{session_id}`

## Model Loading Status

- Real mode loaded successfully on one visible RTX 3090.
- `/health` mode: `real`
- Loaded flags: openWakeWord `true`, Stage 1 head `true`, Qwen ASR `true`, Stage 2 head `true`.
- Tmux session: `vigil_browser_assistant_demo`
- Log: `finetune/demo_live_assistant/logs/demo_20260628_034213_gpu6.log`

## Onboarding And Calibration

The demo stores accepted clips under ignored local data. Calibration requires at least three accepted positive VIGIL clips. It uses bounded positive-bias demo calibration and does not update Qwen, openWakeWord, or Stage 2 weights.

## Assistant Listening

The assistant chunk route updates a rolling transcript, Stage 1 score, Stage 2 score, calibrated Stage 2 score, thresholds, cooldown state, and assistant state. The UI enters Assistant / VQA state on accepted VIGIL trigger.

## Transcript Branch

Real mode uses the loaded frozen Qwen runtime. Mock/partial mode reports its mode through `/health` and uses deterministic local placeholder transcript behavior for route validation.

## Trigger Branch

Real mode reuses the current two-stage runtime. Stage 2 may use one extra Qwen encoder forward for candidates. Shared-Qwen hidden-state reuse is not claimed.

## Validation Examples

- `/health`: passed.
- `GET /`: served the browser app HTML.
- Profile creation: passed.
- Onboarding upload: 3 accepted positive support clips uploaded.
- Calibration: `ok`, support count `3`, method `bounded_positive_bias_demo`, bias `1.0`.
- Assistant session start: passed.
- Positive VIGIL chunk: expected label `1`, trigger detected `true`, assistant state `ASSISTANT_STATE`, Stage 1 score `0.9995500445365906`, Stage 2 score `0.9879599809646606`, latency `1723.2751678675413` ms.
- Negative chunk: expected label `0`, trigger detected `false`, assistant state `LISTENING`, Stage 1 score `0.005684383679181337`, Stage 2 score `null`, latency `127.6231212541461` ms.
- Browser microphone capture was not human-tested.

## Run Command

```bash
bash finetune/demo_live_assistant/scripts/run_demo_tmux.sh 6
```

## Limitations

- Browser microphone capture is not human-tested yet.
- No LLM / VQA response is implemented.
- No production website, Supabase, Render, Vercel, cloud ASR, or cloud LLM is used.
- Shared-Qwen hidden-state reuse remains unsolved.
