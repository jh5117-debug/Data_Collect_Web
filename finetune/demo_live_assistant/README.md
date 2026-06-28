# VIGIL Local HAL Browser Assistant Demo

This is a local research demo for the VIGIL voice trigger module inside an ASR-based clinical workflow. It is not a production website and not a commercial assistant product.

## Workflow

Microphone audio is shown as two parallel branches:

- Continuous frozen Qwen ASR branch for rolling transcript.
- Parallel VIGIL trigger branch:
  - Stage 1: openWakeWord candidate detector.
  - Stage 2: frozen-Qwen-feature verifier.
  - On trigger: enter Assistant / VQA state.

Downstream LLM / VQA response generation is intentionally not implemented.

## Screens

1. Local profile: user enters a local name and gets a local profile ID.
2. Onboarding recordings: a light version of the data-collection recorder flow with common VIGIL prompts, Record, Stop, playback, Accept, Delete, and accepted rows.
3. Calibration result: bounded positive-bias demo calibration using only accepted positive clips.
4. Assistant listening: low-latency chunked rolling transcript, VIGIL/Virgil highlighting, Stage 1 score, Stage 2 score, thresholds, calibration status, trigger state, and cooldown.

## Model Notes

- Qwen ASR is frozen.
- The browser assistant loads one Qwen3-ASR-1.7B weight instance.
- openWakeWord is frozen.
- Stage 2 uses frozen Qwen audio features and a small verifier head.
- Rolling transcript uses Qwen `transcribe` on each independently encoded microphone segment and extracts `$[0].text`.
- Browser microphone segments are currently 1.2s, which is a practical approximation of updating every few spoken words. This is not true token-streaming ASR.
- Current prototype may run an extra Qwen feature path for Stage 2 candidates.
- Shared-Qwen hidden-state reuse is not solved in this demo.

If the real model cannot load, the app starts in partial/mock mode and `/health` reports that clearly.

## Run

```bash
cd /home/hj/Data_Collect_Web
bash finetune/demo_live_assistant/scripts/run_demo.sh 6
```

URL:

```text
http://127.0.0.1:7861
```

SSH tunnel:

```bash
ssh -L 7861:127.0.0.1:7861 hal
```

If the laptop already uses local port 7861:

```bash
ssh -L 7862:127.0.0.1:7861 hj@130.149.110.182
```

Then open `http://127.0.0.1:7862` on the laptop. The browser and microphone run on the laptop; model inference runs on HAL.

Tmux:

```bash
bash finetune/demo_live_assistant/scripts/run_demo_tmux.sh 6
```

Current tmux session:

```text
vigil_browser_assistant_demo
```

Current log:

```text
finetune/demo_live_assistant/logs/demo_20260628_045447_gpu6.log
```

## Clear Local Data

```bash
bash finetune/demo_live_assistant/scripts/clear_local_demo_data.sh
```

Only `finetune/demo_live_assistant/local_data/` is removed.

## Limitations

- Browser microphone capture is not validated unless a human opens the page and records audio.
- No Supabase, Vercel, Render, external ASR API, OpenAI API, cloud LLM, or production database is used.
- No LLM / VQA answer is implemented.
- Shared-Qwen hidden-state reuse remains blocked by the public qwen_asr wrapper.
