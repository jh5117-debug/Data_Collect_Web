# VIGIL Trigger Integration

This is the main handoff file for Shaw, Andy, David, and anyone integrating the VIGIL trigger into a larger assistant or robot stack.

## What This Repository Contains

- A production-style VIGIL Recorder web app for collecting browser microphone examples.
- Offline two-stage wake-word training and evaluation code under `finetune/`.
- LibriSpeech ASR preservation benchmarks for frozen Qwen3-ASR.
- Local HAL browser demo code for onboarding, calibration, rolling transcript, and trigger-state visualization.
- Target-doctor few-shot ablation reports showing where personalization helps.

## Branches Merged Into Main

- `feature/vigil-two-stage-smoke-20260620`
- `research/vigil-eval-live-demo-20260624`
- `research/vigil-latest-data-optimization-20260626`
- `research/vigil-stage1-fewshot-sharedqwen-20260626`
- `research/vigil-shared-qwen-runtime-20260627`
- `research/vigil-browser-assistant-demo-20260627`
- `research/vigil-target-doctor-fewshot-ablation-20260630`

## Current Architecture

```text
16 kHz microphone audio
  -> continuous frozen Qwen3-ASR branch
       -> transcript / report text
  -> parallel VIGIL trigger branch
       -> Stage 1 openWakeWord candidate detector
       -> Stage 2 frozen-Qwen-feature verifier
       -> trigger event
```

The ASR branch and the trigger branch use the same frozen Qwen model family conceptually. The current production-safe Stage 2 path still performs an extra Qwen encoder forward for the verifier. Hidden-state sharing from the public `qwen_asr` runtime is not verified.

## Minimal Python API Shape

Input:

- 16 kHz mono waveform as `float32`, or
- a WAV file that can be converted to 16 kHz mono.

Output:

```python
{
    "trigger_detected": bool,
    "stage1_score": float,
    "stage2_score": float | None,
    "trigger_timestamp": float | None,
    "rolling_transcript": str | None,
}
```

Suggested CLI:

```bash
python -m vigil_trigger.run --wav sample.wav --model-run finetune/runs/20260624_075127_0fad4c7828149099_full
```

## Suggested ROS 2 Integration

1. Create a VIGIL trigger node.
2. Subscribe to microphone/audio chunks.
3. Buffer audio into the Stage 1 streaming window.
4. Run Stage 2 only on Stage 1 candidates.
5. Publish a trigger event with scores and timestamp.
6. Optionally publish rolling transcript and the command segment after activation.

Suggested topics:

- `/audio/chunk`
- `/vigil/trigger`
- `/vigil/transcript`
- `/vigil/debug_scores`

## Current Limitations

- Qwen hidden-state sharing is not solved through the public runtime.
- Stage 2 may need an extra Qwen encoder forward.
- The browser assistant is a local HAL demo, not a deployed product.
- The demo does not implement downstream LLM or VQA responses.
- Qwen ASR weights and openWakeWord feature extractor remain frozen.

## How To Run

Smoke training/evaluation:

```bash
cd /home/hj/Data_Collect_Web
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH bash finetune/scripts/run_official_smoke_local_3090.sh
```

Browser demo on HAL:

```bash
cd /home/hj/Data_Collect_Web
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
bash finetune/demo_live_assistant/scripts/run_demo.sh 6 \
  /home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full
```

Tunnel from a laptop:

```bash
ssh -N -L 7862:127.0.0.1:7861 hj@130.149.110.182
```

Few-shot ablation report:

```text
finetune/experiments/fewshot_ablation/reports/FEWSHOT_ABLATION_FINAL_REPORT.md
```

## Reports To Read

- `docs/VIGIL_CURRENT_STATUS.md`
- `docs/VIGIL_MODEL_ARCHITECTURE.md`
- `docs/VIGIL_EXPERIMENT_RESULTS.md`
- `docs/VIGIL_BROWSER_DEMO.md`
- `docs/VIGIL_DATA_COLLECTION_PROTOCOL.md`
- `docs/VIGIL_MAIN_MERGE_SUMMARY.md`

## What Not To Commit Or Share

Do not commit or share raw audio, WAV/WebM/M4A/FLAC files, ZIP exports, Supabase dumps, SQLite databases, local demo data, model weights, checkpoints, feature caches, Qwen weights, Hugging Face caches, logs, predictions, `.env` files, credentials, or participant private names/emails.
