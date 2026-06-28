# Codex Handoff: VIGIL Browser Assistant Demo

## Branch And Commit

- Branch: `research/vigil-browser-assistant-demo-20260627`
- Latest commit: this handoff update is included in `Align browser demo recording and transcript flow`; run `git log -1 --oneline` for the exact hash.

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
- Onboarding UI is now a lighter familiar version of the data-collection recorder:
  common VIGIL prompt chips, selected transcript display, record, stop, playback, accept, delete, accepted rows, and positive count.
- Calibration requires at least 3 accepted positive VIGIL clips.
- Bounded positive-bias demo calibration implemented.

## Assistant Listening Status

- Start/stop/reset implemented.
- HTTP chunk upload implemented with independently encoded 1.2s MediaRecorder segments.
- Bad/unparseable audio chunks no longer produce HTTP 500; debug records transcript or trigger path errors.
- Cooldown implemented.
- LLM / VQA response intentionally not implemented.
- Browser UI highlights `VIGIL`/`Virgil` in red in the rolling transcript and adds a `VIGIL Assistant activated` line when trigger is accepted.

## Transcript Status

- Real mode uses one frozen Qwen3-ASR-1.7B weight instance through the existing runtime.
- Assistant chunks now run Qwen `transcribe(..., language=None)` independently on every microphone segment and extract `$[0].text`.
- The current segment duration is 1.2s to approximate updating every few spoken words. It is still chunked ASR, not true token streaming.
- Rolling transcript no longer depends on VIGIL trigger acceptance.
- Mock/partial mode clearly reports non-real status in `/health`.

## VIGIL Trigger Status

- Real mode uses the current two-stage runtime when load succeeds.
- Stage 2 may use an extra Qwen feature path for candidates. This is extra compute, not a second Qwen weight copy.

## Tests

- `PYTHONPATH=finetune/src:finetune/demo_live_assistant:. pytest -q finetune/demo_live_assistant/tests`: `14 passed`.
- `PYTHONPATH=finetune/src:. pytest -q finetune/tests`: `56 passed`.
- `PYTHONPATH=finetune/benchmarks/asr/src:finetune/src:. pytest -q finetune/benchmarks/asr/tests`: `17 passed`.
- `python -m compileall -q finetune/src finetune/demo_live_assistant`: passed.
- `bash -n finetune/demo_live_assistant/scripts/*.sh`: passed.

## Validation

- Tmux session: `vigil_browser_assistant_demo`.
- Log: `finetune/demo_live_assistant/logs/demo_20260628_045447_gpu6.log`.
- `GET /health`: passed, real mode.
- `GET /`: browser app HTML served.
- Profile creation: passed.
- Onboarding upload: 3 accepted positive support clips uploaded.
- Calibration: `ok`, support count `3`, method `bounded_positive_bias_demo`, bias `1.0`.
- Positive VIGIL chunk: trigger detected `true`, assistant state `ASSISTANT_STATE`, Stage 1 score `0.9995500445365906`, Stage 2 score `0.9879599809646606`.
- Negative chunk: trigger detected `false`, assistant state `LISTENING`, Stage 1 score `0.005684383679181337`, Stage 2 score `null`.
- Public LibriSpeech API smoke after transcript fix:
  trigger detected `false`, assistant state `LISTENING`, transcript preview `Introducing such a person to us.`, Qwen weight instances `1`, transcript extraction path `$[0].text`, transcript error `None`, Stage 2 Qwen feature path used `false`.
- Bad WebM chunk robustness smoke: HTTP status `200`, trigger detected `false`, no route-level HTTP 500.
- Browser microphone capture was opened by the user; the user reported VIGIL triggering worked. Rolling transcript fix was validated through the local chunk API.

## SSH Tunnel And Compute Location

- HAL binds the demo only to `127.0.0.1:7861`, so it is private to HAL unless forwarded.
- The laptop's `127.0.0.1` is the laptop itself, not HAL.
- `ssh -L 7862:127.0.0.1:7861 hj@130.149.110.182` maps laptop port 7862 to HAL's private port 7861 over SSH.
- Browser rendering, microphone capture, and WebM segment encoding run on the laptop.
- openWakeWord, Stage 1, Qwen ASR, and Stage 2 inference run in the HAL Python process on the selected RTX 3090.

## Exact Next Command

```bash
ssh -L 7862:127.0.0.1:7861 hj@130.149.110.182
```

## Push Status

- Previous commit `ebe298e` pushed to `origin/research/vigil-browser-assistant-demo-20260627`; latest local robustness update pending commit/push.
