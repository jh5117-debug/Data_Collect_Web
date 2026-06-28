# Web Assistant Demo Report

## Status

- Demo package: `finetune/demo_live_assistant`
- URL: `http://127.0.0.1:7861`
- SSH tunnel: `ssh -L 7861:127.0.0.1:7861 hal`
- Scope: local HAL research demo only.

## Screens Implemented

- Local profile screen.
- Onboarding recording screen aligned with the familiar data-collection recorder rhythm but simplified:
  common VIGIL prompt chips, selected transcript, record, stop, playback, accept, delete, accepted rows, and positive count.
- Calibration result screen.
- Assistant listening screen with low-latency chunked transcript, red VIGIL/Virgil highlighting, accepted-trigger activation line, scores, thresholds, and debug.

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
- Log: `finetune/demo_live_assistant/logs/demo_20260628_230627_gpu6.log`

## Onboarding And Calibration

The demo stores accepted clips under ignored local data. Calibration requires at least three accepted positive VIGIL clips. It now extracts Stage 2 embeddings for those support clips, builds a normalized 128D few-shot prototype, and saves it in the local profile calibration JSON. Qwen, openWakeWord, and Stage 2 weights are not updated. Assistant start is rejected until `calibration_active` is true.

## Assistant Listening

The assistant chunk route updates a rolling transcript, Stage 1 score, Stage 2 score, calibrated Stage 2 score, thresholds, cooldown state, and assistant state. The browser records independently encoded 1.2s segments so each upload is ffmpeg-decodable and feels closer to updating every few spoken words. Bad chunks no longer surface as HTTP 500; errors are returned in debug while listening continues. The UI enters Assistant / VQA state on accepted VIGIL trigger.

## Transcript Branch

Real mode uses one loaded frozen Qwen3-ASR-1.7B weight instance. The assistant chunk route runs Qwen `transcribe(..., language=None)` on every microphone segment and extracts the structured result at `$[0].text`, so rolling transcript is not dependent on VIGIL trigger acceptance. Mock/partial mode reports its mode through `/health` and uses deterministic local placeholder transcript behavior for route validation.

## Trigger Branch

Real mode reuses the current two-stage runtime. Stage 2 may use one extra Qwen feature path for candidate windows. This is not a second Qwen weight copy; it is extra compute through the same frozen Qwen runtime. Shared-Qwen hidden-state reuse is not claimed.

## Validation Examples

- `/health`: passed.
- `GET /`: served the browser app HTML.
- Profile creation: passed.
- Onboarding upload: 3 accepted positive support clips uploaded.
- Calibration real smoke with temporary synthetic WAV support: `ok`, support count `3`, method `few_shot_qwen_stage2_prototype`, prototype dim `128`, support pairwise mean similarity `0.9938476085662842`, calibration latency `1608.5919998586178` ms, Qwen weights updated `false`, Stage 2 weights updated `false`.
- Uncalibrated assistant session start: HTTP `400`, calibration requirement returned.
- Calibrated assistant session start: passed, session created.
- Positive VIGIL chunk: expected label `1`, trigger detected `true`, assistant state `ASSISTANT_STATE`, Stage 1 score `0.9995500445365906`, Stage 2 score `0.9879599809646606`, latency `1723.2751678675413` ms.
- Negative chunk: expected label `0`, trigger detected `false`, assistant state `LISTENING`, Stage 1 score `0.005684383679181337`, Stage 2 score `null`, latency `127.6231212541461` ms.
- Public LibriSpeech API smoke after transcript fix: trigger detected `false`, assistant state `LISTENING`, transcript preview `Introducing such a person to us.`, Qwen weight instances `1`, transcript extraction path `$[0].text`, transcript error `None`, Stage 2 Qwen feature path used `false`.
- Bad WebM chunk robustness smoke: HTTP status `200`, trigger detected `false`, no route-level HTTP 500.
- Browser microphone capture was opened by the user; the user reported VIGIL triggering worked. Rolling transcript was then fixed and validated through the local chunk API.

## SSH Tunnel And Inference Location

The demo binds to HAL localhost only. An SSH tunnel maps a laptop-local port to HAL's private `127.0.0.1:7861`. Browser rendering, microphone capture, and audio segment encoding happen on the laptop. openWakeWord, Stage 1, Qwen ASR, and Stage 2 inference happen inside the HAL Python process on the selected RTX 3090.

## Run Command

```bash
bash finetune/demo_live_assistant/scripts/run_demo_tmux.sh 6
```

## Limitations

- Browser microphone capture is not human-tested yet.
- No LLM / VQA response is implemented.
- No production website, Supabase, Render, Vercel, cloud ASR, or cloud LLM is used.
- Shared-Qwen hidden-state reuse remains unsolved.
